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

"""Emit-side bridge: pushes kvc-aware router signals to rl-insight.

A single module-level singleton (``emitter``, used like ``logger``) fans router
state out to rl-insight (→ Prometheus → Grafana). It is the only place that
knows *how* to emit; the contract for *what* is emitted lives in
``types/emit_spec.py`` (``EMIT_SPECS``), and the rl-insight API mapping is a thin
fan-out by metric type.

Two mount points feed the same singleton:

  * **B-class — store writes** (14 signals): the collector builds a
    :class:`WriteEvent` after a store write and calls :meth:`Emitter.on_write`,
    which dispatches by ``kind`` to the 14 B-class primitives.
  * **A-class — strategy ``score()``** (6 signals): the strategy calls
    :meth:`Emitter.on_score` with the components it just computed, and
    :meth:`Emitter.on_route` with the ``score()`` self-timing.

Env-gated and lazily initialized after ``verl/utils/tracking.py``'s
``RLInsightLogger`` template: every public method short-circuits on
``enabled()`` so that with rl-insight off (the default) the router pays ~zero
cost. Observability must never crash the host it observes, so unknown score
components and missing event fields are skipped silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ..types.emit_spec import EMIT_SPECS, EmitKey

__all__ = ["Emitter", "WriteEvent", "WriteKind", "emitter"]


class WriteKind:
    """Discriminator for :class:`WriteEvent` — which store write path fired."""

    POLL = "poll"  # vLLM /metrics absolute snapshot (refresh_metrics)
    ACQUIRE = "acquire"  # request dispatched (incr_metrics on acquire)
    RELEASE = "release"  # request completed (incr_metrics on release)
    KV_REMOVED = "kv_removed"  # KV blocks removed (remove_kv_blocks)


# B-class gauge primitives read from a poll event's ``new_values`` (the 3 vLLM
# levels + the 4 vLLM cumulative counters). kv_cache_load is router-derived and
# rides on WriteEvent.load instead.
_POLL_GAUGES: tuple[str, ...] = (
    EmitKey.KV_CACHE_USAGE_PERC,
    EmitKey.NUM_REQUESTS_RUNNING,
    EmitKey.NUM_REQUESTS_WAITING,
    EmitKey.PROMPT_TOKENS,
    EmitKey.PROMPT_TOKENS_CACHED,
    EmitKey.EXTERNAL_PREFIX_CACHE_HITS,
    EmitKey.ESTIMATED_FLOPS_PER_GPU,
)


@dataclass
class WriteEvent:
    """What the collector hands to :meth:`Emitter.on_write` after a store write.

    ``deltas`` / ``new_values`` are keyed by canonical metric name (the same
    strings as the ``EmitKey`` values for emitted metrics). Each ``kind`` fills
    only the fields it has; the others stay at their defaults and the emitter
    skips them.

    Attributes:
        kind: One of :class:`WriteKind`.
        node: Replica id (becomes the ``replica`` label).
        deltas: Counter increments for this write (canonical name → delta).
        new_values: Gauge absolutes after this write (canonical name → value).
        load: Router-derived ``kv_cache_load`` (poll only).
        turn_sum: Absolute "in-flight turn sum" (acquire/release) — the
            numerator of ``inflight_avg_turn``.
        inflight_count: Absolute in-flight request count (acquire/release) —
            the denominator. Used only for the division; not emitted itself
            (PromQL derives it from dispatched − completed).
    """

    kind: str
    node: str
    deltas: dict[str, float] = field(default_factory=dict)
    new_values: dict[str, float] = field(default_factory=dict)
    load: float | None = None
    turn_sum: int | None = None
    inflight_count: int | None = None


class Emitter:
    """Singleton bridge from router state to rl-insight.

    Env-gated (``VERL_RL_INSIGHT_ENABLE=1``) and lazily wired to ``rl_insight``.
    Reuses the trainer's ``RLInsightLogger`` enable switch so one flag turns on
    trainer + router observability together.
    """

    ENABLE_ENV = "VERL_RL_INSIGHT_ENABLE"

    def __init__(self) -> None:
        self._rl_insight: Any = None
        self._init_done: bool = False

    # ── wiring ────────────────────────────────────────────────────────
    def enabled(self) -> bool:
        """Return whether router rl-insight emit is on in this process."""
        return os.getenv(self.ENABLE_ENV) == "1"

    def _get_rl_insight(self) -> Any:
        if self._rl_insight is None:
            import rl_insight

            self._rl_insight = rl_insight
        return self._rl_insight

    def _ensure_init(self) -> None:
        """Initialize rl_insight once (idempotent; re-init is a no-op)."""
        if self._init_done:
            return
        # No project/experiment_name: the router is a metric source, not a
        # trainer run. rl_insight.init() reads RL_INSIGHT_SERVER_URL itself and
        # is a no-op if the trainer already initialized it in this process.
        self._get_rl_insight().init()
        self._init_done = True

    def _reset(self) -> None:
        """Clear cached wiring (test isolation / reconfiguration)."""
        self._rl_insight = None
        self._init_done = False

    # ── A-class: strategy score() ─────────────────────────────────────
    def on_score(self, replica: str, components: dict[str, float]) -> None:
        """Emit A-class score components (load/s_cache/avail/need/remaining).

        The active strategy passes only the components it computed (the two
        strategies are mutually exclusive at runtime). Unknown keys — anything
        not in ``EMIT_SPECS`` — are skipped so a future component does not crash
        the score path before it is added to the contract.

        Args:
            replica: Replica id (the ``replica`` label).
            components: Mapping of canonical score-key → computed value.
        """
        if not self.enabled():
            return
        self._ensure_init()
        for key, value in components.items():
            if key in EMIT_SPECS:
                self._emit(key, value, replica)

    def on_route(self, latency_s: float) -> None:
        """Emit one ``score()`` policy-scoring latency sample (global, no replica).

        Args:
            latency_s: ``score()`` body duration in seconds (self-timed by the
                strategy's ``try``/``finally``).
        """
        if not self.enabled():
            return
        self._ensure_init()
        self._emit(EmitKey.ROUTE_LATENCY_SECONDS, latency_s)

    # ── B-class: store writes ─────────────────────────────────────────
    def on_write(self, event: WriteEvent) -> None:
        """Emit the B-class primitives for one store write, dispatched by kind.

        poll→8, acquire→4, release→3, kv_removed→1 emit calls (some primitives,
        e.g. inflight_tokens, fire from more than one kind).
        """
        if not self.enabled():
            return
        self._ensure_init()
        replica = event.node
        kind = event.kind
        if kind == WriteKind.POLL:
            for key in _POLL_GAUGES:
                self._emit_from(event.new_values, key, replica)
            if event.load is not None:
                self._emit(EmitKey.KV_CACHE_LOAD, event.load, replica)
        elif kind == WriteKind.ACQUIRE:
            self._emit_from(event.deltas, EmitKey.DISPATCHED_COUNT, replica)
            self._emit_from(event.deltas, EmitKey.PROMPT_LEN_SUM, replica)
            self._emit_from(event.new_values, EmitKey.INFLIGHT_TOKENS, replica)
            self._emit_avg_turn(replica, event)
        elif kind == WriteKind.RELEASE:
            self._emit_from(event.deltas, EmitKey.COMPLETED_COUNT, replica)
            self._emit_from(event.new_values, EmitKey.INFLIGHT_TOKENS, replica)
            self._emit_avg_turn(replica, event)
        elif kind == WriteKind.KV_REMOVED:
            self._emit_from(event.deltas, EmitKey.KV_EVICTIONS, replica)

    def _emit_from(self, src: dict[str, float], key: str, replica: str) -> None:
        """Emit ``key`` from ``src`` (counter deltas or gauge absolutes) if present."""
        value = src.get(key)
        if value is not None:
            self._emit(key, value, replica)

    def _emit_avg_turn(self, replica: str, event: WriteEvent) -> None:
        """Emit inflight_avg_turn = turn_sum / inflight_count (emitter-side division).

        The store/collector supplies the two absolutes; the emitter
        performs the division (presentation computes the ratio, state holds
        state). When the replica is idle (count 0) the average is defined as 0.
        """
        if event.turn_sum is None or event.inflight_count is None:
            return
        avg = event.turn_sum / event.inflight_count if event.inflight_count else 0.0
        self._emit(EmitKey.INFLIGHT_AVG_TURN, avg, replica)

    # ── fan-out ───────────────────────────────────────────────────────
    def _emit(self, key: str, value: float, replica: str | None = None) -> None:
        """Map one canonical key to the right rl_insight call via ``EMIT_SPECS``."""
        spec = EMIT_SPECS[key]
        api = self._get_rl_insight()
        labels: dict[str, Any] = {"replica": replica} if spec["labels"] else {}
        metric_type = spec["type"]
        if metric_type == "counter":
            api.metric_count(key, value, spec["help"], **labels)
        elif metric_type == "gauge":
            api.metric_gauge(key, value, spec["help"], **labels)
        else:  # histogram
            api.metric_histogram(key, value, spec["help"], buckets=spec.get("buckets"), **labels)


# Module-level singleton — import and use as ``emitter.on_score(...)``.
emitter = Emitter()
