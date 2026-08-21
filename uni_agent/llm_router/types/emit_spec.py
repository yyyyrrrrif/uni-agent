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

"""Emit-side metric specifications — emit canonical names, types, labels, buckets.

This module is an **emit-side data definition layer** — it defines the canonical
names, Prometheus types, label sets, histogram buckets, and help strings for the
signals the router pushes to rl-insight (→ Prometheus → Grafana).

``metric_spec`` is the inbound layer (keys vLLM is scraped on); ``emit_spec`` is
the outbound layer (keys the router emits on). Many names overlap, but the two
layers are kept independent so each is a single source of truth for its
direction.

The rl_insight API mapping (``metric_count`` / ``metric_gauge`` /
``metric_histogram``) lives in the emitter, not here. Adding a metric = adding
one entry in ``EMIT_SPECS`` below; the emitter then emits it unchanged.
"""

from __future__ import annotations

from typing import Any

# ── Emit canonical key constants ─────────────────────────────────────
# The emitter references keys via EmitKey constants — never raw strings.


class EmitKey:
    """Canonical emit-side metric key names.

    Split by value lifecycle anchor (not file location):
      * B-class — settle into the store (polled or written on events); emitted
        from the store ``on_write`` hook.
      * A-class — computed inside ``score()`` and existing nowhere else; emitted
        from the strategy via ``on_score`` / ``on_route``.
    """

    # ── B-class · store on_write (14) ──
    # vLLM cumulative counters, forwarded as-is (gauge set of the cumulative).
    KV_CACHE_USAGE_PERC: str = "kv_cache_usage_perc"
    NUM_REQUESTS_RUNNING: str = "num_requests_running"
    NUM_REQUESTS_WAITING: str = "num_requests_waiting"
    KV_CACHE_LOAD: str = "kv_cache_load"
    INFLIGHT_TOKENS: str = "inflight_tokens"
    PROMPT_TOKENS: str = "prompt_tokens"
    PROMPT_TOKENS_CACHED: str = "prompt_tokens_cached"
    EXTERNAL_PREFIX_CACHE_HITS: str = "external_prefix_cache_hits"
    ESTIMATED_FLOPS_PER_GPU: str = "estimated_flops_per_gpu"
    DISPATCHED_COUNT: str = "dispatched_count"
    COMPLETED_COUNT: str = "completed_count"
    PROMPT_LEN_SUM: str = "prompt_len_sum"
    INFLIGHT_AVG_TURN: str = "inflight_avg_turn"
    KV_EVICTIONS: str = "kv_evictions"
    # ── A-class · strategy on_score / on_route (6) ──
    LOAD: str = "load"
    S_CACHE: str = "s_cache"
    AVAIL: str = "avail"
    NEED: str = "need"
    REMAINING: str = "remaining"
    ROUTE_LATENCY_SECONDS: str = "route_latency_seconds"


# ── Histogram bucket constants (shared) ──────────────────────────────

# Ratio scores in [0, 1] (prefix-load-aware ``load``, weighted cache hit rate
# ``s_cache``). Ten even buckets: i/10 for i in 1..10.
_RATIO_BUCKETS: tuple[float, ...] = tuple(i / 10 for i in range(1, 11))

# score() policy-scoring latency, in seconds. Four decades (0.1-1 / 1-10 /
# 10-100 / 100-1000 ms), each split into 5 even buckets = 20 buckets. Values are
# the ms boundaries divided by 1000.
_ROUTE_LATENCY_BUCKETS_S: tuple[float, ...] = (
    0.00028,
    0.00046,
    0.00064,
    0.00082,
    0.001,
    0.0028,
    0.0046,
    0.0064,
    0.0082,
    0.01,
    0.028,
    0.046,
    0.064,
    0.082,
    0.1,
    0.28,
    0.46,
    0.64,
    0.82,
    1.0,
)


# ── Emit definitions (single source of truth) ────────────────────────
# key   = canonical emit name (matches the EmitKey constant values)
# value = property dict:
#   type    : "counter" | "gauge" | "histogram" — emitter picks the rl_insight API
#   labels  : list of label names; ["replica"] for per-replica, [] for global
#   buckets : histogram bucket upper bounds, or None for prometheus defaults.
#             Omitted for counter / gauge.
#   help    : Prometheus HELP string
EMIT_SPECS: dict[str, dict[str, Any]] = {
    # ── B-class gauge · vLLM cumulative counters forwarded as-is ──
    # vLLM already maintains these as monotonic counters; the emitter sets the
    # cumulative value directly (no prev, no delta). rate() on a gauge works.
    EmitKey.PROMPT_TOKENS: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Cumulative prefill tokens computed locally (cache miss), forwarded from vLLM",
    },
    EmitKey.PROMPT_TOKENS_CACHED: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Cumulative prompt tokens served from prefix cache, forwarded from vLLM",
    },
    EmitKey.EXTERNAL_PREFIX_CACHE_HITS: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Cumulative external (cross-replica) prefix-cache hits, forwarded from vLLM",
    },
    EmitKey.ESTIMATED_FLOPS_PER_GPU: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Cumulative estimated FLOPs per GPU, forwarded from vLLM (MFU = rate / peak)",
    },
    # ── B-class gauge · instantaneous levels (poll / event write) ──
    EmitKey.KV_CACHE_USAGE_PERC: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "vLLM KV cache usage percentage",
    },
    EmitKey.NUM_REQUESTS_RUNNING: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Requests currently running on the replica",
    },
    EmitKey.NUM_REQUESTS_WAITING: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Requests waiting to be processed on the replica",
    },
    EmitKey.KV_CACHE_LOAD: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Router-derived KV occupancy (retained blocks / num_gpu_blocks)",
    },
    EmitKey.INFLIGHT_TOKENS: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Prompt tokens across in-flight requests on the replica (acquire +plen / release -plen)",
    },
    # ── B-class gauge · derived ──
    EmitKey.INFLIGHT_AVG_TURN: {
        "type": "gauge",
        "labels": ["replica"],
        "help": "Average turn of in-flight requests on the replica (instantaneous level)",
    },
    # ── B-class counter · routing events (delta available; prometheus_client accumulates) ──
    EmitKey.DISPATCHED_COUNT: {
        "type": "counter",
        "labels": ["replica"],
        "help": "Cumulative dispatched requests (acquire +1) — pair with completed_count for dispatch rates",
    },
    EmitKey.COMPLETED_COUNT: {
        "type": "counter",
        "labels": ["replica"],
        "help": "Cumulative completed requests (release +1) — realized-throughput share",
    },
    EmitKey.PROMPT_LEN_SUM: {
        "type": "counter",
        "labels": ["replica"],
        "help": "Cumulative sum of dispatched prompt lengths (avg prompt len = rate / rate(dispatched_count))",
    },
    EmitKey.KV_EVICTIONS: {
        "type": "counter",
        "labels": ["replica"],
        "help": (
            "Cumulative KV block removals (vLLM removed events; not pure evictions "
            "— includes request-completion releases)"
        ),
    },
    # ── A-class histogram · strategy components (per-dispatch × per-replica observe) ──
    EmitKey.LOAD: {
        "type": "histogram",
        "labels": ["replica"],
        "buckets": _RATIO_BUCKETS,
        "help": "Prefix-load-aware load score (replica-state component, [0, 1])",
    },
    EmitKey.S_CACHE: {
        "type": "histogram",
        "labels": ["replica"],
        "buckets": _RATIO_BUCKETS,
        "help": "Weighted three-layer prefix cache hit rate (request-replica component, [0, 1])",
    },
    EmitKey.AVAIL: {
        "type": "histogram",
        "labels": ["replica"],
        "buckets": None,
        "help": "Capacity-token available headroom = cap * (1 - kv_usage); buckets TBD after sampling",
    },
    EmitKey.NEED: {
        "type": "histogram",
        "labels": ["replica"],
        "buckets": None,
        "help": "Capacity-token need = plen * (1 - gpu_hit); buckets TBD after sampling",
    },
    EmitKey.REMAINING: {
        "type": "histogram",
        "labels": ["replica"],
        "buckets": None,
        "help": "Capacity-token remaining = avail - need (may be negative); buckets TBD after sampling",
    },
    # ── A-class histogram · route latency (per-call, global — no replica label) ──
    EmitKey.ROUTE_LATENCY_SECONDS: {
        "type": "histogram",
        "labels": [],
        "buckets": _ROUTE_LATENCY_BUCKETS_S,
        "help": "score() policy-scoring latency, seconds (self-timed in score() try/finally)",
    },
}
