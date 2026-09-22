# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collector — unified collector interface combining Transport + Parser."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from typing import Any

from ..config.collector import CollectorConfig
from ..debug import get_debug_var, is_debug_enabled
from ..insight import WriteEvent, WriteKind, emitter
from ..store.data_store import DataStore
from ..types import EmitKey, MetricKey
from ..utils.knob import coerce_knob_value
from ..utils.prefix_cache import get_prefix_hashes, resolve_prefix_hashes
from .parse import KVCacheUpdate, MetricsUpdate, Parser, StickyUpdate
from .transport.base import Transport

logger = logging.getLogger(__name__)


DEFAULT_COLLECTOR_KNOBS: dict[str, float | int] = {
    "http_timeout": 10.0,
    "base_retry_delay": 1.0,
    "max_retry_delay": 30.0,
    "max_retry_attempts": 5,
    "retry_backoff_factor": 2.0,
}


_METRICS_LOG_INTERVAL_S = 30
_DISPATCH_LOG_INTERVAL_S = 5.0
_KV_EVENT_LOG_INTERVAL_S = 500


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
    MetricKey.ESTIMATED_FLOPS_PER_GPU,
)

# Per-request bookkeeping key: in-flight tokens recorded at dispatch (the blocks
# this dispatch had to allocate, see ``_book_dispatch_blocks``). verl #7115
# releases carry no token list, so the release-side ``INFLIGHT_TOKENS`` delta
# is folded from this row — same acquire-record / release-consume shape as the
# per-request "turn" counter below.
_INFLIGHT_TOKENS_KEY = "inflight_tokens"

# Per-request bookkeeping key: the replica this request's prefix blocks are
# pinned on (``KVCacheStore.pin_inflight_blocks``). Release unpins through it and
# deletes it, so its presence at acquire time means a release went missing — the
# only way to notice, since the held-block set has no engine-side event.
_INFLIGHT_BLOCK_PIN_KEY = "inflight_blocks_pin"


def _avg(delta_sum: float, delta_cnt: float) -> float:
    """Windowed average = delta_sum / delta_cnt, or NaN if no samples."""
    return delta_sum / delta_cnt if delta_cnt > 0 else float("nan")


def _ms(value: float) -> str:
    """Format a seconds value as millis for the evidence log ('-' if NaN)."""
    return f"{value * 1000:.1f}" if value == value else "-"


class Collector:
    """Unified collector — composes Transport + Parser.

    Args:
        transport: Transport instance (ZMQ, HTTP, etc.)
        parser: Parser instance (vLLM KV, vLLM Metrics, etc.)
    """

    def __init__(self, transport: Transport, parser: Parser) -> None:
        self._transport = transport
        self._parser = parser
        self._data_store = DataStore()
        self._future: Future | None = None
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None
        # Periodic evidence-log state. The parser is stateless (returns
        # MetricsUpdate; merged here), so the log reads a current snapshot.
        self._metrics_poll_count = 0
        # Previous cumulative snapshot per node — for windowed delta
        # computation. {node_id: {canonical_key: value}}
        self._metrics_prev: dict[str, dict[str, float]] = {}
        # kv-event tallies for periodic summary logging (kv parser only).
        self._kv_event_counts: dict[str, int] = defaultdict(int)
        self._kv_block_counts: dict[str, int] = defaultdict(int)
        self._kv_last_logged_total = 0
        # Last-emit time for the dispatched/completed/inflight_turn_sum snapshot (throttled).
        self._dispatch_last_log: float = 0.0
        # Cumulative dispatch-time prompt tokens vs the booked part of them —
        # the realized allocation saving the in-flight token gauge nets out.
        # Logged with the dispatch snapshot (``router-inflight-tokens``).
        self._inflight_token_sums: dict[str, int] = {"dispatched": 0, "uncached": 0}
        # Dispatches whose block list was unresolvable (no prompt_ids / block size
        # unknown): nothing was pinned, so their capacity is invisible in the
        # held-block account. Counted rather than silently dropped.
        self._unaccounted_dispatches: int = 0

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the collector — launch event-loop thread and subscribe."""

        def run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        def handler(raw_data: bytes | str, node_id: str) -> None:
            """Handler: parse and dispatch to the right store write path."""
            result = self._parser.parse(raw_data, node_id)
            if isinstance(result, KVCacheUpdate):
                self._write_kv_update(result)
            elif isinstance(result, MetricsUpdate):
                self._write_metrics_update(result)
            elif isinstance(result, StickyUpdate):
                self._write_sticky_update(result)
            else:
                # None is normal for statistic parsers that skip an event
                # (e.g. StickyParser on_release); demote to debug to avoid per-turn noise.
                logger.debug(f"parser.parse returned no update: {result}")

        if getattr(self._transport, "is_async", True):
            self._loop_thread = threading.Thread(
                target=run_loop,
                daemon=True,
            )
            self._loop_thread.start()

            self._future = asyncio.run_coroutine_threadsafe(
                self._transport.subscribe(handler),
                self._loop,
            )
        else:
            # CallbackTransport: subscribe only registers callbacks — run it on a
            # throwaway loop; stop() just unregisters.
            tmp_loop = asyncio.new_event_loop()
            try:
                tmp_loop.run_until_complete(self._transport.subscribe(handler))
            finally:
                tmp_loop.close()

    def _write_kv_update(self, update: KVCacheUpdate) -> None:
        """Write KVCacheUpdate via DataStore, then emit a periodic kv-events tally."""
        if update.block_size is not None:
            self._data_store.set_block_size(update.block_size)
        if update.clear_all:
            self._data_store.clear_kv_node(update.node_id)
        for layer, hashes in update.remove_blocks.items():
            if hashes:
                self._data_store.remove_kv_blocks(update.node_id, hashes, layer=layer)
        for layer, hashes in update.add_blocks.items():
            if hashes:
                self._data_store.add_kv_blocks(update.node_id, hashes, layer=layer)

        # Tally for periodic summary — observe BlockStored/BlockRemoved flow.
        n_added = sum(len(v) for v in update.add_blocks.values())
        n_removed = sum(len(v) for v in update.remove_blocks.values())
        if update.clear_all:
            self._kv_event_counts["clear"] += 1
        if n_added:
            self._kv_event_counts["stored"] += 1
            self._kv_block_counts["stored"] += n_added
        if n_removed:
            self._kv_event_counts["removed"] += 1
            self._kv_block_counts["removed"] += n_removed
            emitter.on_write(
                WriteEvent(
                    kind=WriteKind.KV_REMOVED,
                    node=update.node_id,
                    deltas={EmitKey.KV_EVICTIONS: n_removed},
                )
            )
        total = sum(self._kv_event_counts.values())
        if total - self._kv_last_logged_total >= _KV_EVENT_LOG_INTERVAL_S:
            self._kv_last_logged_total = total
            # Parser-side translation failures: non-zero means the event-side
            # retained count is drifting above the engine's truth.
            hash_map_stats = getattr(self._parser, "stats", None)
            logger.debug(
                f"kv-events tally: events={dict(self._kv_event_counts)} "
                f"blocks={dict(self._kv_block_counts)} (total_events={total}) | "
                f"retained_blocks/replica={self._data_store.per_replica_block_counts()} | "
                f"hash-map={dict(hash_map_stats or {})}"
            )

    def _write_metrics_update(self, update: MetricsUpdate) -> None:
        """Write MetricsUpdate via DataStore, then forward to the insight emitter.

        Delta updates (acquire/release) route to ``incr_metrics``; the in-flight
        turn sum (``INFLIGHT_TURN_SUM``) and the release-side token delta
        (``INFLIGHT_TOKENS``) are folded into the same locked write from
        per-request rows recorded at dispatch — acquire records the request's
        current turn and its booked token count, release subtracts both
        (release changes neither, so it subtracts what acquire recorded).
        verl #7115 releases carry no token list, which is why the count lives in
        a per-request row rather than the release event.
        On acquire the parser's raw ``prompt_len`` token delta is rewritten to
        the blocks this dispatch actually has to allocate (see
        :meth:`_book_dispatch_blocks`) and the request's prefix blocks are pinned
        on the chosen replica, which refreshes the absolute ``INFLIGHT_BLOCKS``
        gauge — the *held* occupancy the capacity strategy turns into free
        capacity. Release unpins the same blocks through the acquire-time memo
        (:meth:`_release_inflight_blocks`).
        Both acquire and release refresh the throttled ``router-dispatch``
        snapshot. Absolute (non-delta) updates are polled gauges, handled
        below. When rl-insight emit is on, each write also builds a
        :class:`WriteEvent` from the store's returned post-write values and hands
        it to the emitter (the 14 B-class signals).
        """
        if update.is_delta:
            # Batch the parser's signed deltas in one locked PerReplica write.
            # Turn lives in PerRequestStore (separate lock); look it up first so
            # it joins this batch (no second PerReplica lock cycle). Turn fires
            # only on dispatch/release.
            deltas = dict(update.metrics)
            is_acquire = MetricKey.DISPATCHED_COUNT in deltas
            is_release = MetricKey.COMPLETED_COUNT in deltas
            if is_acquire:
                if update.request_id is None:
                    logger.debug(
                        "dispatch (DISPATCHED_COUNT) update missing request_id — "
                        "skipping turn, block pinning and token rewrite (raw prompt_len booked)"
                    )
                else:
                    deltas[MetricKey.INFLIGHT_TURN_SUM] = self._data_store.incr_per_request(update.request_id, "turn")
                    raw_tokens = deltas.get(MetricKey.INFLIGHT_TOKENS, 0)
                    # A pin marker still on the request means its previous dispatch
                    # never released (a dropped COMPLETED_COUNT): unpin that set and
                    # net out its still-booked tokens, otherwise both leak forever.
                    stale_replica = self._data_store.get_per_request(update.request_id, _INFLIGHT_BLOCK_PIN_KEY, None)
                    stale_booked = 0
                    if stale_replica is not None:
                        logger.warning(
                            f"re-acquire without release request={update.request_id} on {stale_replica} "
                            f"— unpinning the previous block set"
                        )
                        self._data_store.unpin_inflight_blocks(
                            stale_replica, get_prefix_hashes(update.request_id, self._data_store) or []
                        )
                        stale_booked = self._data_store.get_per_request(update.request_id, _INFLIGHT_TOKENS_KEY, 0)
                    booked, hash_strs = self._book_dispatch_blocks(
                        update.node_id, update.request_id, update.prompt_ids, raw_tokens
                    )
                    booked_delta = booked - stale_booked
                    deltas[MetricKey.INFLIGHT_TOKENS] = booked_delta
                    self._inflight_token_sums["dispatched"] += raw_tokens
                    self._inflight_token_sums["uncached"] += booked_delta
                    if not hash_strs:
                        self._unaccounted_dispatches += 1
                    self._data_store.set_per_request(update.request_id, _INFLIGHT_TOKENS_KEY, booked)
                    if hash_strs:
                        self._data_store.set_per_request(update.request_id, _INFLIGHT_BLOCK_PIN_KEY, update.node_id)
                    self._refresh_inflight_blocks(update.node_id)
            elif is_release:
                if update.request_id is None:
                    logger.debug(
                        "release (COMPLETED_COUNT) update missing request_id — skipping turn/token/block subtraction"
                    )
                else:
                    deltas[MetricKey.INFLIGHT_TURN_SUM] = -self._data_store.get_per_request(
                        update.request_id, "turn", 0
                    )
                    deltas[MetricKey.INFLIGHT_TOKENS] = -self._data_store.get_per_request(
                        update.request_id, _INFLIGHT_TOKENS_KEY, 0
                    )
                    self._release_inflight_blocks(update.node_id, update.request_id)
            new_values = self._data_store.incr_metrics(update.node_id, deltas)
            if is_acquire or is_release:
                emitter.on_write(
                    WriteEvent(
                        kind=WriteKind.ACQUIRE if is_acquire else WriteKind.RELEASE,
                        node=update.node_id,
                        deltas=deltas,
                        new_values=new_values,
                        turn_sum=new_values.get(MetricKey.INFLIGHT_TURN_SUM),
                        inflight_count=new_values.get(MetricKey.INFLIGHT_COUNT),
                    )
                )
            self._maybe_log_dispatch_stats()
            return
        snapshots = self._data_store.refresh_metrics({update.node_id: update.metrics})
        emitter.on_write(
            WriteEvent(
                kind=WriteKind.POLL,
                node=update.node_id,
                new_values=snapshots.get(update.node_id, {}),
                load=self._data_store.kv_cache_load(update.node_id),
            )
        )

        # Periodic visibility into what the collector fed the router — compare
        # against vllm's own "GPU KV cache usage" engine-stats log line.
        self._metrics_poll_count += 1
        if self._metrics_poll_count % _METRICS_LOG_INTERVAL_S == 0:
            # Emit evidence for ALL known replicas, not just the one that happened
            # to be polled at this poll-count tick. Metrics polling is serial
            # (one replica per poll), so emitting only ``update.node_id`` here
            # sampled ~1/N of replicas per window → some replicas never got an
            # evidence line (e.g. 4/8 seen). Each replica keeps its own
            # ``_metrics_prev`` baseline, so windowed deltas stay correct.
            for nid in self._data_store.get_metric_node_ids():
                self._log_evidence_window(nid)

    def _book_dispatch_blocks(
        self,
        node_id: str,
        request_id: str,
        prompt_ids: tuple[int, ...],
        raw_tokens: int | float,
    ) -> tuple[int, list[str]]:
        """Pin this dispatch's prefix blocks; return ``(booked_tokens, hash_strs)``.

        **Ledger 1** (``INFLIGHT_TOKENS``) books what this dispatch has to
        *allocate*: the blocks that are neither resident in the chosen replica's
        prefix cache nor already pinned by another in-flight request, times the
        block size. That is the block-granular refinement of the earlier
        ``prompt_len × (1 − gpu_hit)``, which double-counted a concurrently
        dispatched shared prefix — "not in the cache yet" is not "free", so the
        second rollout of the same prompt booked its whole prefill twice over.
        **Ledger 2** (``INFLIGHT_BLOCKS``) is the absolute held set, pinned here
        and re-read right after by :meth:`_refresh_inflight_blocks`.

        Falls back to booking the raw ``prompt_len`` when there is nothing to
        pin: no ``prompt_ids`` forwarded, a block size that is not learned yet
        (the hash chain is unresolvable), or an empty chain. Erring on the raw
        length keeps the gauge conservative (never books less than the true
        footprint) and matches the pre-change behavior; those dispatches are
        tallied in ``_unaccounted_dispatches`` so the blind spot is visible.
        """
        block_size = self._data_store.get_block_size()
        if not prompt_ids or not block_size:
            return int(raw_tokens), []
        hash_strs = resolve_prefix_hashes(list(prompt_ids), request_id, self._data_store)
        if not hash_strs:
            return int(raw_tokens), []
        new_blocks = self._data_store.pin_inflight_blocks(node_id, hash_strs)
        return new_blocks * int(block_size), hash_strs

    def _release_inflight_blocks(self, node_id: str, request_id: str) -> None:
        """Unpin a finishing request's blocks and refresh its replica's held gauge.

        The block list comes from the acquire-time hashing memo (the release
        event carries no prompt under verl #7115) and the replica from the
        acquire-time pin marker. A missing memo means the held set cannot be
        unwound for this request: warn (the gauge only errs high) rather than
        subtract a guessed amount. Unpinning is reference-counted, so a block
        shared with another in-flight request stays held.
        """
        replica_id = self._data_store.get_per_request(request_id, _INFLIGHT_BLOCK_PIN_KEY, None)
        if replica_id is None:
            # Nothing was pinned at acquire (no prompt forwarded, or the block size
            # was still unknown) — the held set never knew about this request.
            logger.debug(f"release with no pinned blocks request={request_id} on {node_id}")
            return
        hash_strs = get_prefix_hashes(request_id, self._data_store)
        if hash_strs is None:
            # The pin table cannot be unwound for this request: the block list it
            # pinned is gone (per-request row evicted). Leaking is the safe
            # direction — the held gauge only errs high, never pretends blocks
            # are free while an in-flight request still holds them.
            logger.warning(
                f"release without block list request={request_id} on {replica_id} — in-flight block set may leak"
            )
            hash_strs = []
        self._data_store.unpin_inflight_blocks(replica_id, hash_strs)
        self._data_store.del_per_request(request_id, _INFLIGHT_BLOCK_PIN_KEY)
        self._refresh_inflight_blocks(replica_id)

    def _refresh_inflight_blocks(self, replica_id: str) -> None:
        """Write a replica's held-block count as an absolute gauge.

        A signed delta cannot express release semantics: a finishing request may
        free zero blocks (they are still held by another in-flight request), and
        a dispatch may free several (it hit blocks that were resident but
        unpinned). Reading the pin table's cardinality after every acquire and
        release keeps the gauge a true state rather than a running sum that can
        drift with one missed event.
        """
        self._data_store.refresh_metrics(
            {replica_id: {MetricKey.INFLIGHT_BLOCKS: self._data_store.inflight_block_count(replica_id)}}
        )

    def _write_sticky_update(self, update: StickyUpdate) -> None:
        """Apply a StickyUpdate to the per-request store (sticky key) via DataStore."""
        if update.action == "put":
            self._data_store.put_sticky_binding(update.request_id, update.replica_id)
        elif update.action == "invalidate":
            self._data_store.invalidate_sticky_binding(update.request_id)
        elif update.action == "invalidate_replica":
            for rid in update.replica_ids:
                self._data_store.invalidate_sticky_replica(rid)
        else:
            logger.warning(f"unknown StickyUpdate action: {update.action}")

    def _maybe_log_dispatch_stats(self) -> None:
        """Emit per-replica dispatched/completed/inflight_turn_sum/prompt_len_sum counters at most every interval.

        Reads each dispatched replica's cumulative counters from PerReplicaStore and
        logs them (the ``router-dispatch`` line); the plot derives trailing-5-min
        dispatched / completed / avg-turn / RPM / avg-prompt-len from their per-replica
        deltas. Time-throttled so the cadence is load-independent (idle stretches emit nothing).
        """
        now = time.monotonic()
        if now - self._dispatch_last_log < _DISPATCH_LOG_INTERVAL_S:
            return
        self._dispatch_last_log = now
        for rep in self._data_store.get_metric_node_ids():
            snap = self._data_store.get_metrics(rep)
            dispatched = snap.get(MetricKey.DISPATCHED_COUNT, 0)
            if not dispatched:  # skip replicas that never received a dispatch
                continue
            completed = snap.get(MetricKey.COMPLETED_COUNT, 0)
            inflight_turn_sum = snap.get(MetricKey.INFLIGHT_TURN_SUM, 0)
            prompt_len_sum = snap.get(MetricKey.PROMPT_LEN_SUM, 0)
            logger.info(
                f"router-dispatch replica={rep} dispatched={dispatched} completed={completed} "
                f"inflight_turn_sum={inflight_turn_sum} prompt_len_sum={prompt_len_sum}"
            )
        # What the INFLIGHT_TOKENS gauge books vs the raw dispatched prompt
        # tokens: the gap is the prefix-cache hit the gauge nets out. Compare
        # against vllm's own cached/prefill counters in the evidence log.
        # ``unaccounted_dispatches`` counts dispatches with no resolvable block
        # list (nothing pinned, so the held-block gauge cannot see them).
        dispatched_tokens = self._inflight_token_sums["dispatched"]
        if dispatched_tokens or self._unaccounted_dispatches:
            uncached_tokens = self._inflight_token_sums["uncached"]
            hit = f"{1.0 - uncached_tokens / dispatched_tokens:.3f}" if dispatched_tokens else "-"
            logger.info(
                f"router-inflight-tokens dispatched_prompt_tokens={dispatched_tokens} "
                f"uncached_tokens={uncached_tokens} (dispatch-time gpu_hit={hit}) "
                f"unaccounted_dispatches={self._unaccounted_dispatches}"
            )

    def _log_evidence_window(self, node_id: str) -> None:
        """Emit a windowed evidence summary for one replica.

        Deltas vs the previous snapshot, over ~``_METRICS_LOG_INTERVAL_S``
        polls (≈30 s). Reads the merged store snapshot (not the per-poll
        update) so a transiently-missing scrape line doesn't zero a
        cumulative counter.
        """
        snap = self._data_store.get_metrics(node_id)
        prev = self._metrics_prev.get(node_id, {})

        def _delta(key: str) -> float:
            cur = float(snap.get(key, 0) or 0)
            return cur - float(prev.get(key, cur) or 0)

        # kv = retained blocks (cached-freeable + running-with-hash); usage =
        # vLLM's running-only fraction. Emit both: cache-fill (kv) vs running
        # pressure (usage).
        kv = self._data_store.kv_cache_load(node_id)
        usage_raw = snap.get(MetricKey.KV_CACHE_USAGE_PERC)
        run = snap.get(MetricKey.NUM_REQUESTS_RUNNING)
        wait = snap.get(MetricKey.NUM_REQUESTS_WAITING)

        # Windowed TTFT/queue/TPOT averages (delta_sum / delta_count).
        ttft_avg = _avg(_delta(MetricKey.TTFT_SECONDS_SUM), _delta(MetricKey.TTFT_COUNT))
        queue_avg = _avg(_delta(MetricKey.QUEUE_TIME_SECONDS_SUM), _delta(MetricKey.QUEUE_TIME_COUNT))
        # prefill_time = TTFT - queue (TTFT includes the queue wait).
        prefill_t = (ttft_avg - queue_avg) if (ttft_avg == ttft_avg and queue_avg == queue_avg) else float("nan")
        tpot_avg = _avg(_delta(MetricKey.TPOT_SECONDS_SUM), _delta(MetricKey.TPOT_COUNT))

        # Token deltas over the window (prefill computed vs cached, decode, external).
        d_prefill = _delta(MetricKey.PROMPT_TOKENS)
        d_cached = _delta(MetricKey.PROMPT_TOKENS_CACHED)
        d_decode = _delta(MetricKey.GENERATION_TOKENS)
        d_external = _delta(MetricKey.EXTERNAL_PREFIX_CACHE_HITS)
        d_flops = _delta(MetricKey.ESTIMATED_FLOPS_PER_GPU)
        cache_hit_pct = 100.0 * d_cached / (d_cached + d_prefill) if (d_cached + d_prefill) > 0 else float("nan")

        kv_str = f"{kv:.3f}" if isinstance(kv, float) else kv
        usage_str = f"{float(usage_raw):.3f}" if usage_raw is not None else "-"
        hit_str = f"{cache_hit_pct:.1f}" if cache_hit_pct == cache_hit_pct else "-"
        logger.debug(
            f"vllm-evidence replica={node_id} kv={kv_str} usage={usage_str} run={run} wait={wait} | "
            f"TTFT={_ms(ttft_avg)}ms queue={_ms(queue_avg)}ms prefillT={_ms(prefill_t)}ms TPOT={_ms(tpot_avg)}ms | "
            f"prefill={int(d_prefill)} cached={int(d_cached)} (hit={hit_str}%) "
            f"decode={int(d_decode)} external={int(d_external)} flops={int(d_flops)} [poll #{self._metrics_poll_count}]"
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
                logger.debug(f"Error draining tasks on stop: {exc}")

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            self._loop_thread = None

        self._future = None


# ── Factory function ───────────────────────────────────────────────────


def _debug_knob(name: str) -> Any:
    """Return knob ``name``: the coerced ``UNI_AGENT_ROUTER_<NAME>`` override, or the default."""
    default = DEFAULT_COLLECTOR_KNOBS[name]
    if not is_debug_enabled():
        return default
    raw = get_debug_var(name)
    if raw is None:
        return default
    return coerce_knob_value(name, raw, default)


def get_collector(
    name: str,
    collectors_config: CollectorConfig,
    server_addresses: dict[str, str] | None = None,
    kv_event_endpoints: dict[str, list[str]] | None = None,
    balancer_handler=None,
) -> Collector:
    """Create a Collector by name — one place does both composition and config binding.

    Connection-type knobs fall back to ``DEFAULT_COLLECTOR_KNOBS``; per-collector
    ``UNI_AGENT_ROUTER_*`` debug overrides (when the debug master switch is on)
    are resolved into local values — never written onto ``collectors_config`` —
    and each collector only reads the knobs its transport uses.

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
        from .parse.vllm.metrics import VLLMMetricsParser
        from .transport.http import HTTPTransport

        transport = HTTPTransport(
            endpoints=server_addresses or {},
            interval=collectors_config.http_interval,
            http_timeout=_debug_knob("http_timeout"),
        )
        return Collector(transport, VLLMMetricsParser())

    if name == "vllm_zmq":
        from .parse.vllm.kv import VLLMKVParser
        from .transport.zmq import ZMQTransport

        transport = ZMQTransport(
            endpoints=kv_event_endpoints or {},
            base_retry_delay=_debug_knob("base_retry_delay"),
            max_retry_delay=_debug_knob("max_retry_delay"),
            max_retry_attempts=_debug_knob("max_retry_attempts"),
            retry_backoff_factor=_debug_knob("retry_backoff_factor"),
        )
        return Collector(transport, VLLMKVParser())

    if name == "sticky_stat":
        from .parse.basic.sticky import StickyParser
        from .transport.callback import CallbackTransport

        return Collector(CallbackTransport(balancer_handler), StickyParser())

    if name == "inflight_stat":
        from .parse.basic.inflight import InflightParser
        from .transport.callback import CallbackTransport

        return Collector(CallbackTransport(balancer_handler), InflightParser())

    raise ValueError(
        f"Unknown collector: '{name}'. Available: ['vllm_metrics', 'vllm_zmq', 'sticky_stat', 'inflight_stat']"
    )
