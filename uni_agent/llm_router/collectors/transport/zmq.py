"""ZMQTransport — ZMQ replay + sub dual socket transport.

Connects to per-endpoint ZMQ addresses (sub + replay), subscribes to
live events, and delivers raw payloads to the handler callback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import zmq
import zmq.asyncio

from uni_agent.llm_router.collectors.transport.base import Transport
from uni_agent.llm_router.logging import get_router_logger

logger = get_router_logger("zmq-transport")


@dataclass
class _EndpointSocketSet:
    """Per-endpoint ZMQ socket bundle (internal — not exported)."""

    node_id: str
    context: zmq.asyncio.Context
    sub_socket: zmq.asyncio.Socket
    replay_socket: zmq.asyncio.Socket
    closed: bool = False


class ZMQTransport(Transport):
    """ZMQ transport — replay + sub dual socket per endpoint.

    Each endpoint gets its own ZMQ context, socket pair, and
    background coroutine — all endpoints subscribe concurrently.

    Args:
        endpoints: ``{node_id: [sub_ip:port, replay_ip:port, publisher, topic]}``
        base_retry_delay: Initial retry delay in seconds.
        max_retry_delay: Maximum retry delay cap in seconds.
        max_retry_attempts: Maximum number of retries per endpoint.
        retry_backoff_factor: Exponential backoff multiplier.
    """

    def __init__(
        self,
        endpoints: dict[str, list[str]],
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        max_retry_attempts: int = 5,
        retry_backoff_factor: float = 2.0,
    ) -> None:
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._max_retry_attempts = max_retry_attempts
        self._retry_backoff_factor = retry_backoff_factor

        # endpoints: {node_id: [sub_endpoint, replay_endpoint, publisher, topic]}
        self._sub_endpoints: dict[str, str] = {}
        self._replay_endpoints: dict[str, str] = {}
        self._topics: dict[str, str] = {}
        for node_id, addrs in endpoints.items():
            self._parse_one_endpoint(node_id, addrs)

        self._stopped = False
        self._endpoint_sockets: dict[str, _EndpointSocketSet] = {}
        self._retry_counts: dict[str, int] = {}
        self._sub_tasks: dict[str, asyncio.Task] = {}
        # Held for dynamic add/remove — subscribe records the running loop
        # (so add_endpoint can schedule the per-endpoint task on it) and the
        # handler (so a late-added endpoint feeds the same decode path).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handler: Callable[[bytes | str, str], None] | None = None

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Spawn per-endpoint subscription tasks, deliver payloads to handler."""
        self._loop = asyncio.get_running_loop()
        self._handler = handler
        sub_tasks = []
        for node_id in self._sub_endpoints:
            sub_addr = self._sub_endpoints[node_id]
            replay_addr = self._replay_endpoints[node_id]
            t = asyncio.create_task(self._subscribe_for_endpoint(node_id, sub_addr, replay_addr, handler))
            self._sub_tasks[node_id] = t
            sub_tasks.append(t)
        try:
            await asyncio.gather(*sub_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in self._sub_tasks.values():
                t.cancel()
        finally:
            self._close_all_zmq_sockets()

    # ── Dynamic endpoint management ─────────────────────────────────────

    def _parse_one_endpoint(self, node_id: str, addrs: list[str]) -> tuple[str, str, str]:
        """Validate + parse one endpoint into (sub_addr, replay_addr, topic).

        Shared by ``__init__`` (bulk) and ``add_endpoint`` (single) so the
        4-element / publisher=='zmq' validation lives in one place. Mutates
        ``_sub_endpoints`` / ``_replay_endpoints`` / ``_topics``.
        """
        if len(addrs) < 4:
            raise ValueError(
                f"endpoint '{node_id}' needs 4 elements [sub, replay, publisher, topic], got {len(addrs)}"
            )
        if addrs[2] != "zmq":
            raise ValueError(f"endpoint '{node_id}' publisher must be 'zmq', got '{addrs[2]}'")
        self._sub_endpoints[node_id] = f"tcp://{addrs[0]}"
        self._replay_endpoints[node_id] = f"tcp://{addrs[1]}"
        self._topics[node_id] = addrs[3]
        return self._sub_endpoints[node_id], self._replay_endpoints[node_id], self._topics[node_id]

    def add_endpoint(self, node_id: str, endpoint: Any) -> None:
        """Register a new endpoint and start its subscription task.

        ``endpoint`` is the 4-element list ``[sub, replay, "zmq", topic]``
        (same form ``__init__`` accepts). Parsed via ``_parse_one_endpoint``
        then the per-endpoint subscription coroutine is created as an
        ``asyncio.Task`` on the running subscribe loop (this method runs on
        the balancer's thread, not the loop thread). Fire-and-forget — the
        task connects with backoff on its own; the caller doesn't wait.
        """
        sub_addr, replay_addr, _ = self._parse_one_endpoint(node_id, list(endpoint))
        if self._loop is None or self._handler is None:
            raise RuntimeError("ZMQTransport.add_endpoint called before subscribe started")
        # If an old task lingers for this id (re-add after remove), drop it.
        old = self._sub_tasks.pop(node_id, None)
        if old is not None and not old.done():
            old.cancel()
        # Create the real asyncio.Task on the loop thread and stash it in
        # _sub_tasks, so remove_endpoint/stop cancel the actual coroutine
        # (not just the run_coroutine_threadsafe Future wrapper).
        def _spawn() -> None:
            self._sub_tasks[node_id] = self._loop.create_task(  # type: ignore[union-attr]
                self._subscribe_for_endpoint(node_id, sub_addr, replay_addr, self._handler)
            )
        self._loop.call_soon_threadsafe(_spawn)  # type: ignore[union-attr]

    def remove_endpoint(self, node_id: str) -> None:
        """Stop an endpoint's subscription and close its ZMQ sockets.

        Synchronous: cancels the per-endpoint task and closes its sockets so
        a gone replica's endpoint stops immediately. The task's own finally
        block re-runs ``_close_zmq_sockets_for`` — idempotent via the
        ``closed`` guard, no double-term.
        """
        sub_addr = self._sub_endpoints.pop(node_id, None)
        self._replay_endpoints.pop(node_id, None)
        self._topics.pop(node_id, None)
        task = self._sub_tasks.pop(node_id, None)
        if task is not None and not task.done():
            task.cancel()
        if sub_addr is not None:
            self._close_zmq_sockets_for(node_id)
        self._retry_counts.pop(node_id, None)

    def stop(self) -> None:
        """Signal stop, cancel tasks, close ZMQ sockets. No loop dependency.

        This method only include:
          1. sets the stop flag so subscribe loops exit,
          2. cancels tasks (synchronous, no loop ref needed),
          3. closes sockets/contexts (idempotent via ``closed`` guard).
        """
        self._stopped = True
        for task in self._sub_tasks.values():
            task.cancel()
        self._sub_tasks.clear()
        self._close_all_zmq_sockets()

    # ── ZMQ connection management ───────────────────────────────────────

    async def _connect_zmq_for(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
    ) -> bool:
        """Create ZMQ context and replay + sub dual socket for a single endpoint."""
        try:
            ctx = zmq.asyncio.Context()

            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.connect(sub_addr)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topics[node_id])

            replay_socket = ctx.socket(zmq.REQ)
            replay_socket.connect(replay_addr)

            self._endpoint_sockets[node_id] = _EndpointSocketSet(
                node_id=node_id,
                context=ctx,
                sub_socket=sub_socket,
                replay_socket=replay_socket,
            )
            self._retry_counts[node_id] = 0
            return True

        except zmq.ZMQError as exc:
            logger.warning(f"ZMQ connection error for node {node_id}: {exc}")
            self._close_zmq_sockets_for(node_id)
            return False

    def _close_zmq_sockets_for(self, node_id: str) -> None:
        """Safely close ZMQ sockets and context for a single endpoint."""
        sockets = self._endpoint_sockets.pop(node_id, None)
        if sockets is None or sockets.closed:
            return
        sockets.closed = True
        sockets.sub_socket.close(linger=0)
        sockets.replay_socket.close(linger=0)
        sockets.context.term()

    def _close_all_zmq_sockets(self) -> None:
        """Close all per-endpoint ZMQ sockets and contexts."""
        for node_id in list(self._endpoint_sockets.keys()):
            self._close_zmq_sockets_for(node_id)

    async def _reconnect_with_backoff_for(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
    ) -> bool:
        """Exponential backoff reconnect for a single endpoint."""
        retry_count = self._retry_counts.get(node_id, 0)
        while retry_count < self._max_retry_attempts:
            delay = min(
                self._base_retry_delay * (self._retry_backoff_factor**retry_count),
                self._max_retry_delay,
            )
            await asyncio.sleep(delay)
            retry_count += 1
            self._retry_counts[node_id] = retry_count

            if await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                return True

        return False

    # ── Per-endpoint subscription ────────────────────────────────────────

    async def _subscribe_for_endpoint(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
        handler: Callable[[bytes | str, str], None],
    ) -> None:
        """Per-endpoint subscription: connect → replay → subscribe loop."""
        try:
            if not await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                    return

            await self._replay_historical_data_for(node_id, handler)

            sockets = self._endpoint_sockets.get(node_id)
            if sockets is None:
                return

            while not self._stopped:
                try:
                    parts = await sockets.sub_socket.recv_multipart()
                    payload = parts[-1]
                    handler(payload, node_id)
                except zmq.ZMQError:
                    self._close_zmq_sockets_for(node_id)
                    if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                        break
                    await self._replay_historical_data_for(node_id, handler)

        except asyncio.CancelledError:
            pass
        finally:
            self._close_zmq_sockets_for(node_id)

    # ── Replay ──────────────────────────────────────────────────────────

    async def _replay_historical_data_for(
        self,
        node_id: str,
        handler: Callable[[bytes | str, str], None],
    ) -> None:
        """Request replay of historical data for a single endpoint.

        Sends "replay" request and receives msgpack-encoded response.
        Degrades to subscription-only on failure.
        """
        sockets = self._endpoint_sockets.get(node_id)
        if sockets is None or sockets.replay_socket is None:
            return

        try:
            await sockets.replay_socket.send(b"replay")

            try:
                replay_data = await asyncio.wait_for(
                    sockets.replay_socket.recv(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                return  # timeout → degrade to subscription-only

            # replay_data is msgpack-encoded, pass directly to handler
            if replay_data:
                handler(replay_data, node_id)

        except zmq.ZMQError as exc:
            logger.warning(f"ZMQ replay error for node {node_id}: {exc}")
