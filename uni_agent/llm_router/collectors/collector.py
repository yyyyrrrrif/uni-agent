"""Collector — unified collector interface combining Transport + Decoder."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from concurrent.futures import Future
from typing import Any

from uni_agent.llm_router.collectors.decoder.base import Decoder
from uni_agent.llm_router.collectors.transport.base import Transport
from uni_agent.llm_router.collectors.updates import KVCacheUpdate, MetricsUpdate
from uni_agent.llm_router.config.collector import CollectorConfig
from uni_agent.llm_router.metric_spec import MetricKey
from uni_agent.llm_router.store.data_store import DataStore

logger = logging.getLogger(__name__)

# Log polled Prometheus metrics every N metrics-writes (≈every 10 s at the
# default 1 s polling interval × a few replicas). Lets us compare what the
# collector feeds the router against vllm's own engine-stats log.
_METRICS_LOG_EVERY_POLLS = 30

# Cumulative metrics tracked for windowed deltas in the evidence log. Single
# source of truth — ``_delta`` consumers below read these by key, and the
# per-replica prev-snapshot iterates the same tuple.
_CUMULATIVE_KEYS: tuple[str, ...] = (
    MetricKey.TTFT_SECONDS_SUM,
    MetricKey.TTFT_COUNT,
    MetricKey.QUEUE_TIME_SECONDS_SUM,
    MetricKey.QUEUE_TIME_COUNT,
    MetricKey.TPOT_SECONDS_SUM,
    MetricKey.TPOT_COUNT,
    MetricKey.PROMPT_TOKENS,
    MetricKey.PROMPT_TOKENS_CACHED,
    MetricKey.GENERATION_TOKENS,
    MetricKey.EXTERNAL_PREFIX_CACHE_HITS,
)


def _avg(delta_sum: float, delta_cnt: float) -> float:
    """Windowed average = delta_sum / delta_cnt, or NaN if no samples."""
    return delta_sum / delta_cnt if delta_cnt > 0 else float("nan")


def _ms(value: float) -> str:
    """Format a seconds value as millis for the evidence log ('-' if NaN)."""
    return f"{value * 1000:.1f}" if value == value else "-"


# Log a kv-events tally (events + blocks by type) every N applied updates.
# Lets us see BlockStored/BlockRemoved flow — esp. whether mc-off groups (no
# mooncake → no kv-events emission) get any events at all.
_KV_EVENT_LOG_EVERY = 500


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
        # Periodic evidence-log bookkeeping (metrics decoder only). The
        # decoder itself is stateless — it returns MetricsUpdate and we merge
        # it here, so the merged-store snapshot the log reads is current.
        self._metrics_poll_count = 0
        # Previous cumulative snapshot per node — for windowed delta
        # computation. {node_id: {canonical_key: value}}
        self._metrics_prev: dict[str, dict[str, float]] = {}
        # kv-event tallies for periodic summary logging (kv decoder only).
        self._kv_event_counts: dict[str, int] = defaultdict(int)
        self._kv_block_counts: dict[str, int] = defaultdict(int)
        self._kv_last_logged_total = 0

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the collector — launch event-loop thread and subscribe."""

        def run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

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
            target=run_loop,
            daemon=True,
        )
        self._loop_thread.start()

        self._future = asyncio.run_coroutine_threadsafe(
            self._transport.subscribe(handler),
            self._loop,
        )

    # ── Dynamic endpoint management ─────────────────────────────────────

    def add_endpoint(self, node_id: str, endpoint: Any) -> None:
        """Forward to the transport — start collecting a new endpoint."""
        self._transport.add_endpoint(node_id, endpoint)

    def remove_endpoint(self, node_id: str) -> None:
        """Forward to the transport — stop collecting an endpoint."""
        self._transport.remove_endpoint(node_id)

    def _write_kv_update(self, update: KVCacheUpdate) -> None:
        """Write KVCacheUpdate via DataStore, then emit a periodic kv-events tally."""
        if update.block_size is not None:
            self._data_store.set_block_size(update.block_size)
        if update.clear_all:
            self._data_store.clear_kv_node(update.node_id)
        if update.remove_blocks:
            self._data_store.remove_kv_blocks(update.node_id, update.remove_blocks)
        if update.add_blocks:
            self._data_store.add_kv_blocks(update.node_id, update.add_blocks)

        # Tally for periodic summary — observe BlockStored/BlockRemoved flow.
        n_added = len(update.add_blocks) if update.add_blocks else 0
        n_removed = len(update.remove_blocks) if update.remove_blocks else 0
        if update.clear_all:
            self._kv_event_counts["clear"] += 1
        if n_added:
            self._kv_event_counts["stored"] += 1
            self._kv_block_counts["stored"] += n_added
        if n_removed:
            self._kv_event_counts["removed"] += 1
            self._kv_block_counts["removed"] += n_removed
        total = sum(self._kv_event_counts.values())
        if total - self._kv_last_logged_total >= _KV_EVENT_LOG_EVERY:
            self._kv_last_logged_total = total
            logger.info(
                f"kv-events tally: events={dict(self._kv_event_counts)} "
                f"blocks={dict(self._kv_block_counts)} (total_events={total}) | "
                f"retained_blocks/replica={self._data_store.per_replica_block_counts()}"
            )

    def _write_metrics_update(self, update: MetricsUpdate) -> None:
        """Write MetricsUpdate via DataStore, then emit a periodic evidence log."""
        self._data_store.refresh_metrics({update.node_id: update.metrics})

        # Periodic visibility into what the collector fed the router — compare
        # against vllm's own "GPU KV cache usage" engine-stats log line.
        self._metrics_poll_count += 1
        if self._metrics_poll_count % _METRICS_LOG_EVERY_POLLS == 0:
            self._log_evidence_window(update.node_id)

    def _log_evidence_window(self, node_id: str) -> None:
        """Emit a windowed evidence summary for one replica.

        Computes deltas vs the previous snapshot for cumulative counters/
        histograms so each line is a rate/average over ~``_METRICS_LOG_EVERY_POLLS``
        polls (≈30 s at the default 1 s interval). This is the raw feed for the
        B−A / D−C evidence chain (TTFT↓, prompt_tokens↓, cached↑ for kvcare).

        Read from the merged store snapshot (refresh already happened above)
        rather than the per-poll ``update.metrics`` — a transiently-missing
        scrape line would otherwise zero a cumulative counter and corrupt the
        window delta.
        """
        snap = self._data_store.get_metrics(node_id)
        prev = self._metrics_prev.get(node_id, {})

        def _delta(key: str) -> float:
            cur = float(snap.get(key, 0) or 0)
            return cur - float(prev.get(key, cur) or 0)

        kv = snap.get(MetricKey.KV_CACHE_USAGE_PERC)
        run = snap.get(MetricKey.NUM_REQUESTS_RUNNING)
        wait = snap.get(MetricKey.NUM_REQUESTS_WAITING)

        # Windowed TTFT/queue/TPOT averages (delta_sum / delta_count).
        ttft_avg = _avg(_delta(MetricKey.TTFT_SECONDS_SUM), _delta(MetricKey.TTFT_COUNT))
        queue_avg = _avg(_delta(MetricKey.QUEUE_TIME_SECONDS_SUM), _delta(MetricKey.QUEUE_TIME_COUNT))
        # prefill_time = TTFT - queue_wait. TTFT includes queue; subtracting it
        # isolates the real prefill compute cost that prefix-sharing reduces.
        prefill_t = (ttft_avg - queue_avg) if (ttft_avg == ttft_avg and queue_avg == queue_avg) else float("nan")
        tpot_avg = _avg(_delta(MetricKey.TPOT_SECONDS_SUM), _delta(MetricKey.TPOT_COUNT))

        # Token deltas over the window (prefill computed vs cached, decode, external).
        d_prefill = _delta(MetricKey.PROMPT_TOKENS)
        d_cached = _delta(MetricKey.PROMPT_TOKENS_CACHED)
        d_decode = _delta(MetricKey.GENERATION_TOKENS)
        d_external = _delta(MetricKey.EXTERNAL_PREFIX_CACHE_HITS)
        cache_hit_pct = 100.0 * d_cached / (d_cached + d_prefill) if (d_cached + d_prefill) > 0 else float("nan")

        kv_str = f"{kv:.3f}" if isinstance(kv, float) else kv
        hit_str = f"{cache_hit_pct:.1f}" if cache_hit_pct == cache_hit_pct else "-"
        logger.info(
            f"vllm-evidence replica={node_id} kv={kv_str} run={run} wait={wait} | "
            f"TTFT={_ms(ttft_avg)}ms queue={_ms(queue_avg)}ms prefillT={_ms(prefill_t)}ms TPOT={_ms(tpot_avg)}ms | "
            f"prefill={int(d_prefill)} cached={int(d_cached)} (hit={hit_str}%) "
            f"decode={int(d_decode)} external={int(d_external)} [poll #{self._metrics_poll_count}]"
        )

        # Snapshot current cumulative values for next window's delta.
        self._metrics_prev[node_id] = {k: float(snap.get(k, 0) or 0) for k in _CUMULATIVE_KEYS}

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
