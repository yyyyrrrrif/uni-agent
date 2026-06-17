"""Collector — unified collector interface combining Transport + Decoder."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future

from uni_agent.llm_router.collectors.decoder.base import Decoder
from uni_agent.llm_router.collectors.transport.base import Transport
from uni_agent.llm_router.collectors.updates import KVCacheUpdate, MetricsUpdate
from uni_agent.llm_router.config.collector import CollectorConfig
from uni_agent.llm_router.store.data_store import DataStore

logger = logging.getLogger(__name__)


class Collector:
    """Unified collector — composes Transport + Decoder.

    Args:
        transport: Transport instance (ZMQ, HTTP, etc.)
        decoder: Decoder instance (vLLM KV, vLLM Metrics, etc.)
    """

    def __init__(self, transport: Transport, decoder: Decoder) -> None:
        self._transport = transport
        self._decoder = decoder
        self._data_store = DataStore()
        self._future: Future | None = None
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the collector — launch event-loop thread and subscribe.
        """

        def handler(raw_data: bytes | str, node_id: str) -> None:
            """Handler: decode and dispatch to the right store write path."""
            result = self._decoder.decode(raw_data, node_id)
            if result is None:
                logger.warning("the return of decoder.decode is None.")
                return
            if isinstance(result, KVCacheUpdate):
                self._write_kv_update(result)
            elif isinstance(result, MetricsUpdate):
                self._write_metrics_update(result)

        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
        )
        self._loop_thread.start()

        self._future = asyncio.run_coroutine_threadsafe(
            self._transport.subscribe(handler),
            self._loop,
        )

    def _write_kv_update(self, update: KVCacheUpdate) -> None:
        """Write KVCacheUpdate via DataStore."""
        if update.block_size is not None:
            self._data_store.set_block_size(update.block_size)
        if update.clear_all:
            self._data_store.clear_kv_node(update.node_id)
        if update.remove_blocks:
            self._data_store.remove_kv_blocks(update.node_id, update.remove_blocks)
        if update.add_blocks:
            self._data_store.add_kv_blocks(update.node_id, update.add_blocks)

    def _write_metrics_update(self, update: MetricsUpdate) -> None:
        """Write MetricsUpdate via DataStore."""
        self._data_store.refresh_metrics({update.node_id: update.metrics})

    def stop(self) -> None:
        """
        Stop the collector — cancel tasks, drain cleanup, stop event-loop thread.
        """
        # Transport closes protocol-level resources (sockets/clients);
        # we own task cancellation and finally-block draining below.
        self._transport.stop()

        if self._loop.is_running():
            # Cancel all tasks and wait for their finally blocks inside the loop
            # so that aclose() runs while the loop is still alive.
            async def _cancel_and_drain() -> None:
                current = asyncio.current_task()
                tasks = [t for t in asyncio.all_tasks() if not t.done() and t is not current]
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            drain = asyncio.run_coroutine_threadsafe(_cancel_and_drain(), self._loop)
            try:
                drain.result(timeout=15)
            except Exception as exc:
                logger.debug("Error draining tasks on stop: %s", exc)

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            self._loop_thread = None

        self._future = None


# ── Factory function ───────────────────────────────────────────────────


def get_collector(
    name: str,
    collectors_config: CollectorConfig,
    server_addresses: dict[str, str] | None = None,
    kv_event_endpoints: dict[str, list[str]] | None = None,
) -> Collector:
    """Create a Collector by name — one place does both composition and config binding.

    Args:
        name: Collector type — ``"vllm_metrics"`` or ``"vllm_zmq"``.
        collectors_config: ``CollectorConfig`` carrying connection-type knobs.
        server_addresses: ``{node_id: ip:port}`` for HTTP transport
            (used by ``"vllm_metrics"``).
        kv_event_endpoints: ``{node_id: [sub_addr, replay_addr]}`` for ZMQ
            transport (used by ``"vllm_zmq"``).

    Returns:
        Configured ``Collector`` instance.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "vllm_metrics":
        from uni_agent.llm_router.collectors.decoder.vllm.metrics import VLLMMetricsDecoder
        from uni_agent.llm_router.collectors.transport.http import HTTPTransport

        hp = collectors_config.http_polling
        transport = HTTPTransport(
            endpoints=server_addresses or {},
            interval=hp["polling_interval"],
            http_timeout=hp["http_timeout"],
        )
        return Collector(transport, VLLMMetricsDecoder())

    if name == "vllm_zmq":
        from uni_agent.llm_router.collectors.decoder.vllm.kv import VLLMKVDecoder
        from uni_agent.llm_router.collectors.transport.zmq import ZMQTransport

        lc = collectors_config.long_connection
        transport = ZMQTransport(
            endpoints=kv_event_endpoints or {},
            base_retry_delay=lc["base_retry_delay"],
            max_retry_delay=lc["max_retry_delay"],
            max_retry_attempts=lc["max_retry_attempts"],
            retry_backoff_factor=lc["retry_backoff_factor"],
        )
        return Collector(transport, VLLMKVDecoder())

    raise ValueError(f"Unknown collector: '{name}'. Available: ['vllm_metrics', 'vllm_zmq']")
