"""HTTPTransport — Prometheus HTTP polling transport.

Polls ``http://{address}/metrics`` for each endpoint at a fixed interval
and delivers response text to the handler callback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import httpx

from uni_agent.llm_router.collectors.transport.base import Transport

logger = logging.getLogger(__name__)


class HTTPTransport(Transport):
    """HTTP polling transport — fetches Prometheus metrics from endpoints.

    Each endpoint is polled at ``interval`` via ``httpx.AsyncClient``.
    Response text is delivered to the handler callback for decoding.

    Args:
        endpoints: ``{node_id: ip:port}`` — each address polls
            ``http://{address}/metrics``.
        interval: Polling interval in seconds.
        http_timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoints: dict[str, str],
        interval: float = 5.0,
        http_timeout: float = 10.0,
    ) -> None:
        self._endpoints: dict[str, str] = {nid: f"http://{addr}/metrics" for nid, addr in endpoints.items()}
        self._interval = interval
        self._http_timeout = http_timeout
        self._client: httpx.AsyncClient | None = None

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Start the HTTP polling loop — delivers response text to handler."""
        self._client = httpx.AsyncClient(timeout=self._http_timeout)
        try:
            while True:
                # Snapshot keys so a concurrent add_endpoint/remove_endpoint
                # mutating self._endpoints from another thread doesn't raise
                # "dictionary changed size during iteration" mid-poll. A
                # newly-added endpoint joins on the next loop tick (≤ one
                # ``interval`` of latency); a removed one stops now.
                coros = {nid: self._client.get(url) for nid, url in list(self._endpoints.items())}
                responses = await asyncio.gather(*coros.values(), return_exceptions=True)
                for nid, resp in zip(coros.keys(), responses, strict=False):
                    if isinstance(resp, Exception):
                        continue  # failed node — handler falls back to defaults
                    if resp.status_code != 200:
                        logger.warning("Failed to fetch metrics from %s: HTTP %s", nid, resp.status_code)
                        continue
                    try:
                        handler(resp.text, nid)
                    except Exception as exc:
                        logger.debug("Handler error for node %s: %s", nid, exc)
                await asyncio.sleep(self._interval)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            client, self._client = self._client, None
            if client is not None:
                try:
                    await client.aclose()
                except Exception as exc:
                    # May fail if called outside an async context (e.g. GC finalizer)
                    logger.debug("HTTPTransport: aclose failed during cleanup: %s", exc)

    # ── Dynamic endpoint management ─────────────────────────────────────

    def add_endpoint(self, node_id: str, endpoint: Any) -> None:
        """Register a new endpoint — it joins the next poll tick.

        ``endpoint`` is the raw ``ip:port`` address (same form ``__init__``
        accepts); it is parsed into ``http://{ip:port}/metrics`` here.
        Synchronous dict mutation is safe against the poll loop's
        ``list(self._endpoints.items())`` snapshot. The new endpoint is first
        polled on the next loop iteration (≤ one ``interval`` latency).
        """
        self._endpoints[node_id] = f"http://{endpoint}/metrics"

    def remove_endpoint(self, node_id: str) -> None:
        """Stop polling an endpoint — drop it from the next snapshot.

        Any in-flight request this tick is left to finish (its response is
        discarded by the loop since the id is no longer in the snapshot).
        No per-endpoint protocol resource to close — the shared
        ``httpx.AsyncClient`` is closed once in ``subscribe``'s finally.
        """
        self._endpoints.pop(node_id, None)

    def stop(self) -> None:
        """No protocol-level resources to close here.

        Per the Transport contract, task cancellation is owned by the
        ``Collector``. The httpx client's ``aclose()`` runs in
        ``subscribe``'s finally block, drained by ``Collector``'s
        ``_cancel_and_drain`` — nothing for this method to do.
        """
