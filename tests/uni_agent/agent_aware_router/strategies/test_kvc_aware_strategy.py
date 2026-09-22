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

"""Unit tests for the LLM router strategy module (strategies/ package).

Unified combined score (one pass, no fast/slow branching):
    S = α·S_cache + (1-α)·S_load
    S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit   (weights sum to 1)
    S_load  = 1 - load                                         (bigger = less loaded)
    load    = a·kv + b·min(1, running/max_num_seqs) + c·min(1, waiting/max_num_seqs)
              + d·min(1, inflight/max_num_seqs)
              (a+b+c+d=1; default 0.4/0.2/0.1/0.3; bigger = more loaded)

Overload (used only by the sticky short-circuit): ``load > load_threshold``
(default 0.9). Combined scoring never consults overload.
Default cache weights: {gpu:0.7, cpu:0.2, ssd:0.1}.
"""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.strategies import route
from uni_agent.agent_aware_router.strategies.base import ReplicaInfo
from uni_agent.agent_aware_router.strategies.kvc_aware import (
    DEFAULT_LOAD_WEIGHTS,
    STICKY_TOP_SCORE,
    KVCacheAwareStrategy,
    StrategyError,
)
from uni_agent.agent_aware_router.strategies.routing import RoutingStrategy
from uni_agent.agent_aware_router.types import Layer, MetricKey, SlowCut

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _strat(**kwargs) -> KVCacheAwareStrategy:
    """Build a KVCacheAwareStrategy with required boilerplate fields filled in.

    Calls ``set_capacity(64)`` so the load formula's running/waiting terms
    are deterministic — mimics what the Balancer does after construction.
    """
    defaults = dict(
        alpha=0.7,
        load_threshold=0.9,
        layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
        # Fixed test baseline — intentionally decoupled from DEFAULT_LOAD_WEIGHTS
        # so behavior tests stay stable when the production default changes.
        load_weights=(0.4, 0.2, 0.1, 0.3),
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64, 1024)
    return strat


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=rid) for rid in ids]


PROMPT_IDS = [1, 2, 3]

# Long prompt for the tests where ``need = plen × (1 − gpu_hit)`` must be large
# enough to outrank a capacity difference (1024 tokens = 64 blocks at 16).
LONG_PROMPT_IDS = list(range(1024))


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeRouteDataProvider:
    """In-memory replica metrics for unit tests.

    Each replica entry is a plain dict with the following optional keys:
      kv_cache_usage_perc  – KV cache usage ratio (default 1.0)
      num_requests_running – requests in flight (default 0)
      num_requests_waiting – requests in the queue (default 0)
      inflight_count       – in-flight acquire/release counter (default 0)
      inflight_tokens      – booked (newly allocated) in-flight prompt tokens (default 0)
      inflight_blocks      – held in-flight KV blocks, dedup + ref counted (default 0)
      gpu_hit_pct          – GPU prefix cache hit percent 0-100 (default 0)
      tiers                – dict mapping tier name to hit rate (default {})
    """

    def __init__(self, data: dict[str, dict], sticky: dict[str, str] | None = None):
        self._data = data
        self._sticky = sticky or {}
        self._per_request: dict[str, dict] = {}

    def get_sticky_binding(self, request_id: str) -> str | None:
        return self._sticky.get(request_id)

    def put_sticky_binding(self, request_id: str, replica_id: str) -> None:
        self._sticky[request_id] = replica_id

    def get_metric(self, replica_id: str, key: str) -> float | int:
        entry = self._data.get(replica_id, {})
        if key == MetricKey.KV_CACHE_USAGE_PERC:
            return entry.get("kv_cache_usage_perc", 1.0)
        if key == MetricKey.NUM_REQUESTS_RUNNING:
            return entry.get("num_requests_running", 0)
        if key == MetricKey.NUM_REQUESTS_WAITING:
            return entry.get("num_requests_waiting", 0)
        if key == MetricKey.INFLIGHT_COUNT:
            return entry.get("inflight_count", 0)
        return entry.get(key, 0.0)

    def get_metrics(self, replica_id: str) -> dict:
        entry = self._data.get(replica_id, {})
        return {
            MetricKey.KV_CACHE_USAGE_PERC: entry.get("kv_cache_usage_perc", 1.0),
            MetricKey.NUM_REQUESTS_RUNNING: entry.get("num_requests_running", 0),
            MetricKey.NUM_REQUESTS_WAITING: entry.get("num_requests_waiting", 0),
            MetricKey.INFLIGHT_COUNT: entry.get("inflight_count", 0),
        }

    def get_layer_prefix_hit_rate(self, replica_id: str, hash_strs: list[str], layer: str = Layer.GPU) -> float:
        entry = self._data.get(replica_id, {})
        if layer == Layer.GPU:
            return entry.get("gpu_hit_pct", 0) / 100.0
        return entry.get("tiers", {}).get(layer, 0.0)

    def kv_cache_load(self, replica_id: str) -> float | None:
        # Unit tests key the load signal on kv_cache_usage_perc (no kv-events /
        # retained blocks simulated); mirror it so the load formula sees it.
        return self._data.get(replica_id, {}).get("kv_cache_usage_perc", 1.0)

    def get_metric_node_ids(self) -> list[str]:
        return list(self._data.keys())

    def get_block_size(self) -> int | None:
        # Block size is learned from KV events; tests default to vLLM's 16.
        return 16

    def get_per_request(self, request_id: str, key: str, default=None):
        return self._per_request.get(request_id, {}).get(key, default)

    def set_per_request(self, request_id: str, key: str, value) -> None:
        self._per_request.setdefault(request_id, {})[key] = value


class ConstantStrategy:
    """Returns a fixed per-replica score list (for route() composition tests)."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        return list(self._scores)


class BadLengthStrategy:
    """Returns a wrong-length list to exercise the contract check in route()."""

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        return [1.0]


class RaisingStrategy:
    """Raises inside score() to exercise route()'s exception wrapping."""

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        raise KeyError("boom")


# --------------------------------------------------------------------------- #
# Unified combined score (one pass: α·S_cache + (1-α)·S_load)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level1
class TestKVCAwareCombinedScore:
    def test_three_layer_cache_weighted_sum(self):
        """
        Feature: S = α·S_cache + (1-α)·S_load; S_cache is a three-layer weighted sum
        Description: two light-load replicas (running=0); rep_a has gpu+cpu+ssd hits
        Expectation: scores = [0.766, 0.322]; rep_a ranks first
          rep_a: load=0.4·0.2=0.08 → s_load=0.92; s_cache=0.70; score=0.7·0.70+0.3·0.92=0.766
          rep_b: load=0.4·0.4=0.16 → s_load=0.84; s_cache=0.10; score=0.7·0.10+0.3·0.84=0.322
        """
        strat = _strat(slow_cut=SlowCut.PREFIX_LOAD_AWARE)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.2,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.6, "ssd": 0.2},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.4,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.3, "ssd": 0.4},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))
        assert scores == pytest.approx([0.766, 0.322])
        # full formula applied: score = α·s_cache + (1-α)·s_load (cache term participates, NOT zeroed)
        assert scores[0] == pytest.approx(0.7 * 0.70 + 0.3 * 0.92)  # rep_a: s_cache=0.70, s_load=0.92
        assert scores[1] == pytest.approx(0.7 * 0.10 + 0.3 * 0.84)  # rep_b: s_cache=0.10, s_load=0.84
        # higher cache hit + lower load ranks first
        assert scores[0] > scores[1]


# --------------------------------------------------------------------------- #
# StrategyRegistry
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level1
class TestKVCAwareLoad:
    def test_missing_metrics_defaults_to_high_load(self):
        """
        Feature: unknown replica defaults to kv=1.0 → load=0.4 (not 1.0); no cache
        Description: score a replica whose id is absent from the provider
        Expectation: load=0.4·1.0=0.4 → s_load=0.6 → score=0.3·0.6=0.18
        """
        strat = _strat(slow_cut=SlowCut.PREFIX_LOAD_AWARE)
        provider = FakeRouteDataProvider({})
        scores = strat.score(PROMPT_IDS, provider, _replicas("ghost"))
        assert scores == pytest.approx([0.18])


# --------------------------------------------------------------------------- #
# _resolve_kv_usage: kv_cache_load drives the load formula
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level1
class TestResolveKVUsage:
    def test_kv_cache_load_drives_load_formula(self):
        """
        Feature: _resolve_kv_usage uses kv_cache_load (not kv_cache_usage_perc)
        Description: data kv_cache_usage_perc=0.9 but kv_cache_load=0.1
        Expectation: load uses kv_cache_load (0.1): load=0.4·0.1=0.04, s_load=0.96,
                     s_cache=0 → score=0.3·0.96=0.288 (not 0.192 from kv=0.9)
        """

        class _LoadProvider(FakeRouteDataProvider):
            def __init__(self, data, load):
                super().__init__(data)
                self._load = load

            def kv_cache_load(self, replica_id):
                return self._load.get(replica_id)

        strat = _strat(slow_cut=SlowCut.PREFIX_LOAD_AWARE)
        provider = _LoadProvider(
            {"rep": {"kv_cache_usage_perc": 0.9, "num_requests_running": 0, "num_requests_waiting": 0}},
            {"rep": 0.1},
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep"))
        assert scores == pytest.approx([0.288])


# --------------------------------------------------------------------------- #
# _cache_score: three-layer weighted hit (gpu + cpu + ssd)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestKVCAwareCacheScore:
    def test_custom_weights_respected(self):
        """
        Feature: _cache_score honors custom layer_weights
        Description: weights {gpu:0.5,cpu:0.3,ssd:0.2}; all hits = 1.0 (gpu_hit_pct=100)
        Expectation: 0.5 + 0.3 + 0.2 = 1.0
        """
        strat = _strat(layer_weights={"gpu": 0.5, "cpu": 0.3, "ssd": 0.2})
        provider = FakeRouteDataProvider({"rep": {"gpu_hit_pct": 100, "tiers": {"cpu": 1.0, "ssd": 1.0}}})
        s_cache, _ = strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS)
        assert s_cache == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Tier weights in the cache term
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level1
class TestKVCAwareTierWeights:
    def test_cpu_weight_higher_than_ssd(self):
        """
        Feature: cpu tier weight (0.2) > ssd tier weight (0.1) in the cache term
        Description: two light-load replicas; one has cpu hit, other ssd hit
        Expectation: cpu-hit replica scores higher
          cpu_hit: load=0.2→s_load=0.8; s_cache=0.2·0.6=0.12; score=0.7·0.12+0.3·0.8=0.324
          ssd_hit: load=0.2→s_load=0.8; s_cache=0.1·0.8=0.08; score=0.7·0.08+0.3·0.8=0.296
        """
        strat = _strat(slow_cut=SlowCut.PREFIX_LOAD_AWARE)
        provider = FakeRouteDataProvider(
            {
                "cpu_hit": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.6, "ssd": 0.0},
                },
                "ssd_hit": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.8},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("cpu_hit", "ssd_hit"))
        assert scores == pytest.approx([0.324, 0.296])
        # formula breakdown: score = α·s_cache + (1-α)·s_load; both share load=0.2→s_load=0.8
        assert scores[0] == pytest.approx(0.7 * (0.2 * 0.6) + 0.3 * 0.8)  # cpu: w_cpu·cpu_hit
        assert scores[1] == pytest.approx(0.7 * (0.1 * 0.8) + 0.3 * 0.8)  # ssd: w_ssd·ssd_hit
        assert scores[0] > scores[1]


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestKVCAwareConstruction:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha": 1.5},
            {"alpha": -0.1},
            {"load_threshold": 0},
            {"load_threshold": 1.0},
            {"layer_weights": {"gpu": 0.7, "cpu": 0.2, "ssd": -0.1}},
            {"layer_weights": {"nvme": 1.0}},
            {"layer_weights": {"gpu": 1.0, "cpu": 0.2, "ssd": 0.1}},
            {"layer_weights": {"gpu": 0.7, "cpu": 0.3}},
            {"load_weights": (0.5, 0.3)},
            {"load_weights": (0.5, 0.5, 0.5)},
            {"load_weights": (-0.1, 0.6, 0.5, 0.0)},
        ],
    )
    def test_invalid_construction_raises(self, kwargs):
        """
        Feature: invalid constructor arguments raise StrategyError
        Description: construct KVCacheAwareStrategy with each invalid kwarg
        Expectation: raises StrategyError for each case
        """
        with pytest.raises(StrategyError):
            _strat(**kwargs)

    def test_valid_three_key_weights_accepted(self):
        strat = _strat(layer_weights={"gpu": 0.5, "cpu": 0.3, "ssd": 0.2})
        assert strat.layer_weights == {"gpu": 0.5, "cpu": 0.3, "ssd": 0.2}


# --------------------------------------------------------------------------- #
# set_capacity
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestSetCapacity:
    @staticmethod
    def _new_strat() -> KVCacheAwareStrategy:
        """A fresh strategy with set_capacity not yet called."""
        return KVCacheAwareStrategy(
            alpha=0.7,
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
        )

    def test_set_capacity_updates_max_num_seqs(self):
        """
        Feature: set_capacity records max_num_seqs
        Description: call set_capacity(16, 1024) on a fresh strategy
        Expectation: _max_num_seqs == 16
        """
        strat = self._new_strat()
        strat.set_capacity(16, 1024)
        assert strat._max_num_seqs == 16

    @pytest.mark.parametrize("max_num_seqs,block_size", [(0, 0), (-1, -1)])
    def test_set_capacity_rejects_non_positive(self, max_num_seqs, block_size):
        """
        Feature: set_capacity rejects zero or negative capacity
        Description: call set_capacity with (0, 0) and (-1, -1)
        Expectation: raises StrategyError
        """
        strat = self._new_strat()
        with pytest.raises(StrategyError):
            strat.set_capacity(max_num_seqs, block_size)

    def test_compute_load_raises_before_set_capacity(self):
        """
        Feature: _compute_load requires set_capacity to be called first
        Description: call _compute_load on a fresh strategy (no set_capacity)
        Expectation: raises StrategyError matching "set_capacity"
        """
        strat = self._new_strat()
        with pytest.raises(StrategyError, match="set_capacity"):
            strat._compute_load(0.5, 0, 0)


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestStrategyContract:
    def test_score_length_and_stateless_repeatable(self):
        """
        Feature: score() returns a replica-length list and is stateless across calls
        Description: call score() twice on the same two-replica inputs
        Expectation: len(scores) == len(replicas); the two calls produce identical results
        """
        strat = _strat()
        assert isinstance(strat, RoutingStrategy)

        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    # cap=1600 → avail = 1600 - 0 blocks (rep_a) vs 1600 - 97×16
                    # (rep_b): a strict argmax, not an exact tie (which the
                    # soft-pick band would randomize across repeated calls).
                    "inflight_tokens": 0,
                    "inflight_blocks": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 2,
                    "num_requests_waiting": 0,
                    "inflight_tokens": 100,
                    "inflight_blocks": 97,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.5, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas)
        assert len(scores) == len(replicas)
        assert strat.score(PROMPT_IDS, provider, replicas) == pytest.approx(scores)


# --------------------------------------------------------------------------- #
# route() composition
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestFromConfig:
    def test_from_config_maps_fields_and_matches_direct_construction(self):
        """
        Feature: from_config maps fields and behaves like direct construction
        Description: build from a non-default cfg; compare fields and score() against a directly-built strategy
        Expectation: fields match the config; _max_num_seqs is None until set_capacity; score matches direct
        """
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        # The config only persists load_threshold; the balancer attaches the
        # remaining knobs with defaults — mirror that here (with one override).
        cfg.alpha = 0.6
        cfg.layer_weights = {"gpu": 0.6, "cpu": 0.3, "ssd": 0.1}
        strat_from_cfg = KVCacheAwareStrategy.from_config(cfg)

        # ── field mapping ──
        assert strat_from_cfg.alpha == pytest.approx(0.6)
        assert strat_from_cfg.load_threshold == pytest.approx(0.85)
        assert strat_from_cfg.layer_weights == {"gpu": 0.6, "cpu": 0.3, "ssd": 0.1}
        assert strat_from_cfg._max_num_seqs is None  # not set until set_capacity()
        assert strat_from_cfg.load_weights == DEFAULT_LOAD_WEIGHTS  # from_config lands on the default

        # ── behavioral equivalence: from_config vs direct construction ──
        strat_from_cfg.set_capacity(64, 1024)
        strat_direct = _strat(
            alpha=0.6,
            load_threshold=0.85,
            layer_weights={"gpu": 0.6, "cpu": 0.3, "ssd": 0.1},
            load_weights=DEFAULT_LOAD_WEIGHTS,
        )
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    "inflight_tokens": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.92,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    # cap=1600 → avail = 1600 - 97×16 = 48 on rep_b, i.e. below
                    # the 160-token gate, while rep_a's empty book leaves it
                    # eligible → both strategies pick rep_a deterministically
                    # (the soft-pick band must not decide here).
                    "inflight_tokens": 100,
                    "inflight_blocks": 97,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        assert strat_from_cfg.score(PROMPT_IDS, provider, replicas) == pytest.approx(
            strat_direct.score(PROMPT_IDS, provider, replicas)
        )


# --------------------------------------------------------------------------- #
# Debug env overrides (UNI_AGENT_ROUTER_*) read in from_config
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestFromConfigDebugEnv:
    """``from_config`` applies ``UNI_AGENT_ROUTER_*`` overrides when debug is on.

    The strategy reads the process env directly (no rollout.custom plumbing):
    each tunable knob is read via ``get_debug_var``, coerced to its default
    type, and merged into the local constructor kwargs — ``cfg`` itself is
    never mutated. Off by default.
    """

    @staticmethod
    def _env_with(debug: str | None, **knobs: str) -> dict[str, str]:
        env: dict[str, str] = {}
        if debug is not None:
            env["UNI_AGENT_ROUTER_DEBUG"] = debug
        for k, v in knobs.items():
            env[f"UNI_AGENT_ROUTER_{k.upper()}"] = v
        return env

    def test_debug_off_ignores_env(self, monkeypatch):
        """Gate off → overrides never reach the strategy (default behavior)."""
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with(None, SLOW_CUT="least-inflight", OVERLOAD_MODE="None"),
        )
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.slow_cut == SlowCut.CAPACITY_TOKEN_AWARE  # default, not least-inflight

    def test_pure_sticky_baseline(self, monkeypatch):
        """slow_cut=least-inflight + overload_mode=None → sticky never falls back."""
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig
        from uni_agent.agent_aware_router.types import OverloadMode

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", SLOW_CUT="least-inflight", OVERLOAD_MODE="None"),
        )
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.slow_cut == SlowCut.LEAST_INFLIGHT
        assert strat.overload_mode == OverloadMode.NONE
        assert strat.do_shortcut is True

    def test_coercion_types(self, monkeypatch):
        """bool / enum / float / dict-json all coerce to their knob types."""
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with(
                "1",
                ALPHA="0.5",
                DO_SHORTCUT="false",
                MEMORY_OVERLOAD_FILTER="no",
                LAYER_WEIGHTS='{"gpu": 0.6, "cpu": 0.3, "ssd": 0.1}',
            ),
        )
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.alpha == pytest.approx(0.5)
        assert strat.do_shortcut is False
        assert strat.memory_overload_filter is False
        assert strat.layer_weights == {Layer.GPU: 0.6, Layer.CPU: 0.3, Layer.SSD: 0.1}

    def test_tie_tolerance_env_override_and_error_layering(self, monkeypatch):
        """TIE_TOLERANCE: coercion first (ConfigError), then range (StrategyError)."""
        from uni_agent.agent_aware_router.config.base import ConfigError
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", TIE_TOLERANCE="0.2"),
        )
        assert KVCacheAwareStrategy.from_config(cfg).tie_tolerance == pytest.approx(0.2)

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", TIE_TOLERANCE="high"),
        )
        with pytest.raises(ConfigError, match="tie_tolerance"):
            KVCacheAwareStrategy.from_config(cfg)

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", TIE_TOLERANCE="1.5"),
        )
        with pytest.raises(StrategyError, match="tie_tolerance"):
            KVCacheAwareStrategy.from_config(cfg)

    @pytest.mark.parametrize(
        ("knob", "value", "match"),
        [
            ("SLOW_CUT", "nope", "slow_cut"),
            ("OVERLOAD_MODE", "sometimes", "overload_mode"),
            ("DO_SHORTCUT", "maybe", "do_shortcut"),
            ("ALPHA", "high", "alpha"),
            ("TIE_TOLERANCE", "high", "tie_tolerance"),
            ("LAYER_WEIGHTS", "not json", "layer_weights"),
            ("LAYER_WEIGHTS", "[1, 2]", "layer_weights"),
            ("LAYER_WEIGHTS", '{"npu": 0.5}', "layer_weights"),
        ],
    )
    def test_bad_values_raise_configerror(self, monkeypatch, knob, value, match):
        """Malformed env values fail fast (ConfigError), not silent wrong strategy."""
        from uni_agent.agent_aware_router.config.base import ConfigError
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", **{knob: value}),
        )
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        with pytest.raises(ConfigError, match=match):
            KVCacheAwareStrategy.from_config(cfg)

    def test_config_object_not_polluted(self, monkeypatch):
        """Debug overrides land in constructor kwargs, never on the config object.

        Reusing the same cfg after disabling debug must produce a default
        strategy — no residual override leaks across constructions.
        """
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        env_on = self._env_with("1", ALPHA="0.5")
        monkeypatch.setattr("uni_agent.agent_aware_router.debug.os.environ", env_on)
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        assert KVCacheAwareStrategy.from_config(cfg).alpha == pytest.approx(0.5)
        assert not hasattr(cfg, "alpha")  # cfg untouched

        monkeypatch.setattr("uni_agent.agent_aware_router.debug.os.environ", {})
        assert KVCacheAwareStrategy.from_config(cfg).alpha == pytest.approx(0.7)  # default, not 0.5


# --------------------------------------------------------------------------- #
# Sticky-session short-circuit (is_overloaded uses load > load_threshold)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestStickyShortCircuit:
    """Sticky replica wins when bound + present + not overloaded; else fall through.

    Overload now means ``load > load_threshold`` (default 0.9) — i.e. the bound
    replica is genuinely saturated (kv≈1, running≈max_num_seqs, big backlog). With
    the default four-term load weights (0.4/0.2/0.1/0.3) the kv+running+waiting
    terms cap at 0.7, so the inflight term (weight 0.3) must be >0 to push load
    past 0.9; the "saturated" cases below therefore feed inflight=max_num_seqs.
    """

    def _provider(self, sticky=None, **per_replica):
        """Build a FakeRouteDataProvider from {rep_id: metrics_dict} + optional sticky."""
        return FakeRouteDataProvider(per_replica, sticky=sticky)

    # ── score() sticky short-circuit ───────────────────────────────────────
    def test_sticky_hit_not_overloaded_short_circuits(self):
        """Feature: bound + present + not overloaded → sticky replica gets top score.
        Description: sticky binds r1→rep_b; rep_b light (load=0.12); rep_a has better
        combined score but must NOT win.
        Expectation: scores = [0.0, STICKY_TOP_SCORE]; route() picks rep_b
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            sticky={"r1": "rep_b"},
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas, "r1")
        assert scores == [0.0, STICKY_TOP_SCORE]
        ranking = route(strat, PROMPT_IDS, provider, replicas, "r1")
        assert ranking[0] == "rep_b"

    def test_sticky_hit_overloaded_falls_back_to_combined(self):
        """Feature: bound but saturated (load>0.9) → no short-circuit, combined scoring.
        Description: sticky binds r1→rep_b; rep_b saturated (kv=1,r=64,w=1000,inflight=64 → load=1.0);
        rep_a light with gpu hit.
        Expectation: rep_a wins (combined), not the saturated sticky rep_b
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            sticky={"r1": "rep_b"},
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={
                "kv_cache_usage_perc": 1.0,
                "num_requests_running": 64,
                "num_requests_waiting": 1000,
                "inflight_count": 64,
                "gpu_hit_pct": 0,
            },
        )
        replicas = _replicas("rep_a", "rep_b")
        ranking = route(strat, PROMPT_IDS, provider, replicas, "r1")
        assert ranking[0] == "rep_a"


# --------------------------------------------------------------------------- #
# _compute_load (load formula) — each term exercised with an explicit weight
# vector, decoupled from DEFAULT_LOAD_WEIGHTS so config changes don't erode
# the formula-coverage assertions.
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level1
class TestLoadFormula:
    @pytest.mark.parametrize(
        "load_weights,kv,running,waiting,inflight,expected",
        [
            # baseline: all-zero inputs → load=0
            ((0.4, 0.2, 0.1, 0.3), 0.0, 0, 0, 0, 0.0),
            # kv term (weight a)
            ((1.0, 0.0, 0.0, 0.0), 0.5, 0, 0, 0, 0.5),
            # running term (weight b), clamped to 1.0 when running > mns
            ((0.0, 1.0, 0.0, 0.0), 0.8, 128, 0, 0, 1.0),
            # waiting term (weight c)
            ((0.0, 0.0, 1.0, 0.0), 0.0, 0, 10, 0, 10 / 64),
            # inflight term (weight d)
            ((0.0, 0.0, 0.0, 1.0), 0.0, 0, 0, 32, 0.5),
        ],
    )
    def test_compute_load_terms(self, load_weights, kv, running, waiting, inflight, expected):
        """
        Feature: load = a·kv + b·min(1,running/mns) + c·min(1,waiting/mns) + d·min(1,inflight/mns)
        Description: isolate each weighted term with a one-hot load_weights vector
        Expectation: each term contributes its weighted value; running clamps to 1.0
        """
        s = _strat(load_weights=load_weights)
        assert s._compute_load(kv, running, waiting, inflight) == pytest.approx(expected)


class TestDefaultWeights:
    def test_default_weights_tuple(self):
        assert DEFAULT_LOAD_WEIGHTS == (0.5, 0.0, 0.0, 0.5)
        assert sum(DEFAULT_LOAD_WEIGHTS) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Fallback modes: memory_overload_filter (sticky overload gate) + slow_cut (fallback scoring)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestFallbackModes:
    """The two formerly-coupled ``USE_VERL_STICKY`` behaviors are now independent
    config knobs: ``memory_overload_filter`` gates the sticky overload check, and
    ``slow_cut`` selects the fallback scoring (``least-inflight`` mirrors verl
    GlobalRequestLoadBalancer)."""

    def test_overloaded_sticky_defers_to_the_capacity_book(self):
        """Bound replica over the sticky gate → the capacity book decides, not the binding.

        The sticky short-circuit declines when ``is_overloaded`` fires (kv_perc >
        load_threshold here). The rebind then comes from the capacity path, which
        reads the router's own book: rep_a holds 97 blocks (avail = 1600-1552 = 48
        < thresh=160 → filtered) while rep_b is empty → rep_b wins. The engine-side
        saturation (kv_perc=0.95) and the router-side saturation agree here, which
        is the steady-state shape: kv blocks are occupied by the requests the
        router dispatched, and it still books them until release.
        """
        strat = _strat(load_threshold=0.9, memory_overload_filter=False)
        provider = FakeRouteDataProvider(
            {
                # cap must be > 0 (num_gpu_blocks) so the post-fallback ranking has a
                # strict winner: with cap=0 both replicas tie and soft-pick randomizes.
                "rep_a": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.95,
                    "num_requests_running": 64,
                    "num_requests_waiting": 1000,
                    "inflight_tokens": 1550,
                    "inflight_blocks": 97,
                },
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 1.0},
            },
            sticky={"r1": "rep_a"},
        )
        ranking = route(strat, PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_b"  # saturated bound replica is not re-picked

    def test_miss_routes_to_least_inflight(self):
        """slow_cut=least-inflight: pick the replica with the fewest in-flight requests."""
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT)
        provider = FakeRouteDataProvider(
            {"rep_a": {"inflight_count": 5}, "rep_b": {"inflight_count": 2}},
        )
        ranking = route(strat, PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_b"

    def test_inflight_tie_keeps_pool_order(self):
        """slow_cut=least-inflight tie-break: equal inflight → first replica in pool order."""
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT)
        provider = FakeRouteDataProvider(
            {"rep_a": {"inflight_count": 3}, "rep_b": {"inflight_count": 3}},
        )
        ranking = route(strat, PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_a"


# --------------------------------------------------------------------------- #
# Capacity-gated token routing (slow_cut=capacity-token-aware)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.cpu
@pytest.mark.level0
class TestCapacityTokenAware:
    """``slow_cut=capacity-token-aware``: free capacity is the router's own book.

    ``avail = cap - INFLIGHT_BLOCKS × block_size`` (the blocks the in-flight
    requests pin on the replica, deduplicated and ref-counted by the collector's
    acquire/release hooks), the gate excludes replicas whose held blocks nearly
    fill the pool, and the largest post-prefill ``remaining = avail - need``
    wins. Neither ``kv_perc`` nor ``INFLIGHT_TOKENS`` decides — both are logged
    as cross-checks.

    cap = num_gpu_blocks × block_size. Tests use num_gpu_blocks=100 and the fake
    provider's block_size=16 → cap=1600; the gate threshold is
    ``cap × (1 - load_threshold)`` — at the default ``load_threshold=0.9`` that
    is 160 free tokens. ``PROMPT_IDS`` is 3 tokens long, so ``need`` is 0-3.
    """

    def _cap_strat(self, **kwargs) -> KVCacheAwareStrategy:
        kwargs.setdefault("slow_cut", SlowCut.CAPACITY_TOKEN_AWARE)
        return _strat(**kwargs)

    def test_gate_filters_booked_replicas_and_picks_max_remaining(self):
        """
        Feature: eligible = avail >= cap·(1-load_threshold); winner = argmax(remaining)
        Description: 3 unbound replicas — rep_a holds 99 blocks
          (avail=16 < thresh=160) → filtered; rep_b (avail=800) and rep_c (avail=480)
          are eligible → argmax(remaining) picks rep_b
        Expectation: scores = [0.0, STICKY_TOP_SCORE, 0.0]; route() picks rep_b
          rep_a: avail=1600-99×16=16 < 160 → filtered (despite gpu_hit=100)
          rep_b: avail=1600-50×16=800 → remaining=797 → winner
          rep_c: avail=1600-70×16=480 → remaining=477 → eligible but smaller
        """
        strat = self._cap_strat()
        replicas = _replicas("rep_a", "rep_b", "rep_c")
        provider = FakeRouteDataProvider(
            {
                # rep_a: perfect cache but its held blocks nearly fill the pool → filtered.
                "rep_a": {
                    "num_gpu_blocks": 100,
                    "inflight_tokens": 1584,
                    "inflight_blocks": 99,
                    "gpu_hit_pct": 100,
                },
                # rep_b: no cache but plenty of room → avail=800.
                "rep_b": {"num_gpu_blocks": 100, "inflight_tokens": 800, "inflight_blocks": 50, "gpu_hit_pct": 0},
                # rep_c: eligible but less remaining than rep_b → argmax is non-trivial.
                "rep_c": {"num_gpu_blocks": 100, "inflight_tokens": 1120, "inflight_blocks": 70, "gpu_hit_pct": 0},
            },
        )
        # No sticky binding anywhere: the capacity path is the ONLY path (the old
        # cold-start branch is gone), so this is also the unbound-request case.
        scores = strat.score(PROMPT_IDS, provider, replicas, request_id="r1")
        assert scores == [0.0, STICKY_TOP_SCORE, 0.0]
        # anti-kidnapping via route(): the cache-rich rep_a is dropped despite best cache
        assert route(strat, PROMPT_IDS, provider, replicas, "r1")[0] == "rep_b"

    def test_avail_tracks_inflight_blocks_not_tokens_or_kv_perc(self):
        """
        Feature: free capacity = cap − INFLIGHT_BLOCKS × block_size; kv_perc and
          INFLIGHT_TOKENS are observation-only
        Description: rep_a reports 99% kv usage but holds no blocks; rep_b reports
          0% kv usage and has booked 1500 tokens, but its 94 held blocks are the
          real occupancy (all of its in-flight prompts were prefix-cache hits, so
          the token ledger — which books only *new allocations* — reads 0)
        Expectation: rep_a eligible (avail=1600) and wins; rep_b filtered
          (avail=1600−1504=96 < 160) despite its empty token ledger
        """
        strat = self._cap_strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.99, "inflight_tokens": 0},
                "rep_b": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.0,
                    "inflight_tokens": 1500,
                    "inflight_blocks": 94,
                },
            },
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), request_id="r1")
        assert scores == [STICKY_TOP_SCORE, 0.0]

    def test_gate_threshold_via_load_threshold_flips_the_winner(self):
        """
        Feature: the gate (not the order) is what load_threshold tunes
        Description: long prompt (1024 tokens) so need is big: rep_a avail=144 with a
          full cache hit (need=0 → remaining=144), rep_b avail=800 but no cache
          (need=1024 → remaining=−224)
        Expectation:
          load_threshold=0.9 (thresh=160): rep_a filtered (144 < 160) → rep_b wins
          load_threshold=0.99 (thresh=16): rep_a eligible → 144 > −224 → rep_a wins
        """
        data = {
            "rep_a": {"num_gpu_blocks": 100, "inflight_tokens": 1450, "inflight_blocks": 91, "gpu_hit_pct": 100},
            "rep_b": {"num_gpu_blocks": 100, "inflight_tokens": 800, "inflight_blocks": 50, "gpu_hit_pct": 0},
        }
        replicas = _replicas("rep_a", "rep_b")
        strict = self._cap_strat(load_threshold=0.9)
        assert strict.score(LONG_PROMPT_IDS, FakeRouteDataProvider(dict(data)), replicas, request_id="r1") == [
            0.0,
            STICKY_TOP_SCORE,
        ]
        loose = self._cap_strat(load_threshold=0.99)
        assert loose.score(LONG_PROMPT_IDS, FakeRouteDataProvider(dict(data)), replicas, request_id="r1") == [
            STICKY_TOP_SCORE,
            0.0,
        ]

    def test_all_overloaded_falls_back_to_the_full_pool(self):
        """
        Feature: empty gate → pool = all replicas, same remaining order (no error)
        Description: every replica is below thresh=160 → fallback
        Expectation: rep_a (avail=16 → remaining 13) beats rep_b (avail=0 → −3)
        """
        strat = self._cap_strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "inflight_tokens": 1584, "inflight_blocks": 99},
                "rep_b": {"num_gpu_blocks": 100, "inflight_tokens": 1592, "inflight_blocks": 100},
            },
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), request_id="r1")
        assert scores == [STICKY_TOP_SCORE, 0.0]

    def test_cap_unknown_falls_back_to_inflight_request_count(self):
        """
        Feature: cap=0 (first KV event not in) → the token account has no yardstick
        Description: no num_gpu_blocks metric at all; rank by INFLIGHT_COUNT instead
          (2 vs 0 → rep_b), and exact ties spread through the soft-pick band
        Expectation: rep_b wins on count; an all-idle pair spreads across both
        """
        strat = self._cap_strat(do_shortcut=False)
        loaded = FakeRouteDataProvider(
            {
                "rep_a": {"inflight_count": 2},
                "rep_b": {"inflight_count": 0},
            }
        )
        assert strat.score(PROMPT_IDS, loaded, _replicas("rep_a", "rep_b"), request_id="r1") == [
            0.0,
            STICKY_TOP_SCORE,
        ]

        idle = FakeRouteDataProvider(
            {
                "rep_a": {"inflight_count": 0},
                "rep_b": {"inflight_count": 0},
            }
        )
        winners = {
            strat.score(PROMPT_IDS, idle, _replicas("rep_a", "rep_b"), request_id="r1").index(STICKY_TOP_SCORE)
            for _ in range(200)
        }
        assert winners == {0, 1}  # exact tie → soft-pick spreads (no pool[0] collapse)

    def test_soft_pick_randomizes_within_tolerance_band(self):
        """
        Feature: tie_tolerance soft-pick — near-equal remaining values share the win
        Description: rep_a book-full → filtered; rep_b remaining=797, rep_c remaining=781
          (gap 16 ≤ 0.05·797 ≈ 39.9 → same tolerance band)
        Expectation: repeated calls return both rep_b and rep_c; neither is pinned
        """
        strat = self._cap_strat(do_shortcut=False, tie_tolerance=0.05)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "inflight_blocks": 99},
                "rep_b": {"num_gpu_blocks": 100, "inflight_blocks": 50},
                "rep_c": {"num_gpu_blocks": 100, "inflight_blocks": 51},
            }
        )
        winners = {
            strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"), request_id="r1").index(
                STICKY_TOP_SCORE
            )
            for _ in range(200)
        }
        assert winners == {1, 2}

    def test_soft_pick_strict_outside_tolerance_band(self):
        """
        Feature: soft-pick only randomizes inside the band — clear wins stay deterministic
        Description: rep_b remaining=1597 (empty book), rep_c remaining=781; gap 816 >
          0.05·1597 ≈ 79.9
        Expectation: every call picks rep_b (strict argmax)
        """
        strat = self._cap_strat(do_shortcut=False, tie_tolerance=0.05)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "inflight_blocks": 99},
                "rep_b": {"num_gpu_blocks": 100, "inflight_blocks": 0},
                "rep_c": {"num_gpu_blocks": 100, "inflight_blocks": 51},
            }
        )
        winners = {
            strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"), request_id="r1").index(
                STICKY_TOP_SCORE
            )
            for _ in range(50)
        }
        assert winners == {1}

    def test_tie_tolerance_zero_is_strict_argmax(self):
        """
        Feature: tie_tolerance=0 reproduces the pre-soft-pick strict argmax
        Description: same near-equal pair as the band test (797 vs 781), tol=0
        Expectation: every call picks rep_b — the tie band is exactly zero width
        """
        strat = self._cap_strat(do_shortcut=False, tie_tolerance=0.0)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "inflight_blocks": 99},
                "rep_b": {"num_gpu_blocks": 100, "inflight_blocks": 50},
                "rep_c": {"num_gpu_blocks": 100, "inflight_blocks": 51},
            }
        )
        winners = {
            strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"), request_id="r1").index(
                STICKY_TOP_SCORE
            )
            for _ in range(50)
        }
        assert winners == {1}

    def test_soft_pick_negative_best_band_is_symmetric(self):
        """
        Feature: gap form on a negative best (no-eligible pool) stays in-band
        Description: values [-16, -24] → best=-16, gap=8. tol=0.5 → band 0.5·16=8 (tie);
          tol=0.05 → band 0.8 (strict). The multiplicative form would give an empty set here
        Expectation: tol=0.5 spreads; tol=0.05 pins index 0
        """
        loose = self._cap_strat(tie_tolerance=0.5)
        assert {loose._soft_pick([-16.0, -24.0], maximize=True) for _ in range(200)} == {0, 1}
        strict = self._cap_strat(tie_tolerance=0.05)
        assert {strict._soft_pick([-16.0, -24.0], maximize=True) for _ in range(50)} == {0}

    def test_soft_pick_zero_best_only_exact_ties(self):
        """
        Feature: best=0 has no relative band (gap=0) — only exact zeros tie
        Description: values [0, 0, -5] with the default tol=0.05
        Expectation: candidates are the two zeros; -5 never wins
        """
        strat = self._cap_strat()
        assert {strat._soft_pick([0.0, 0.0, -5.0], maximize=True) for _ in range(200)} == {0, 1}

    def test_tie_tolerance_validation_default_and_repr(self):
        """
        Feature: tie_tolerance is an internal knob with [0, 1] validation
        Description: out-of-range direct construction; default and from_config values;
          __repr__ carries the field
        Expectation: StrategyError for -0.1 / 1.5; default 0.05 everywhere; repr mentions it
        """
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        for bad in (-0.1, 1.5):
            with pytest.raises(StrategyError, match="tie_tolerance"):
                self._cap_strat(tie_tolerance=bad)
        strat = self._cap_strat()
        assert strat.tie_tolerance == pytest.approx(0.05)
        assert "tie_tolerance=0.05" in repr(strat)
        from_cfg = KVCacheAwareStrategy.from_config(KVCAwareStrategyConfig(load_threshold=0.85))
        assert from_cfg.tie_tolerance == pytest.approx(0.05)  # knob default, not a cfg field
