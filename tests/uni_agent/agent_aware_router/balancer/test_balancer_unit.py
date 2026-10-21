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

from __future__ import annotations

import time
from threading import Barrier
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf

import uni_agent.agent_aware_router.balancer as balancer_module
from uni_agent.agent_aware_router.balancer import CapacityDeferred, KVCAwareBalancer
from uni_agent.agent_aware_router.config.base import ConfigError
from uni_agent.agent_aware_router.store import DataStore
from uni_agent.agent_aware_router.strategies.kvc_aware import KVCacheAwareStrategy
from uni_agent.agent_aware_router.strategies.routing import route
from uni_agent.agent_aware_router.types import MetricKey

from ._helpers import _FakeCollectorManager, _make_balancer, _router_config

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def remote(self):
        self.calls += 1
        return self._fn()


class _RolloutServerHandle:
    def __init__(self, rollout_config, address=("127.0.0.1", 8000)):
        def get_rollout_config():
            if isinstance(rollout_config, Exception):
                raise rollout_config
            return rollout_config

        self.get_rollout_config = _RemoteMethod(get_rollout_config)
        self.get_server_address = _RemoteMethod(lambda: address)


def _kv_source(**overrides):
    config = {
        "publisher": "zmq",
        "endpoint": "tcp://10.0.0.1:41233",
        "replay_endpoint": "tcp://10.0.0.1:41234",
        "topic": "kv-events",
    }
    config.update(overrides)
    return {"0": config}


def _rollout_config(**overrides):
    config = {
        "custom": {
            "agent_framework": {
                "router": {"load_threshold": 0.75, "http_interval": 3.0},
            },
        },
        "max_num_seqs": 128,
        "max_num_batched_tokens": 4096,
    }
    config.update(overrides)
    return OmegaConf.create(config)


class TestKVEventSourceDiscovery:
    @pytest.mark.parametrize(
        ("replay_endpoint", "expected_replay"),
        [("tcp://10.0.0.1:41234", "10.0.0.1:41234"), (None, "")],
    )
    def test_single_dp_source_uses_collector_shape(self, replay_endpoint, expected_replay):
        assert KVCAwareBalancer._parse_kv_event_sources(_kv_source(replay_endpoint=replay_endpoint)) == [
            "10.0.0.1:41233",
            expected_replay,
            "zmq",
            "kv-events",
        ]

    def test_empty_response_disables_zmq_discovery(self):
        assert KVCAwareBalancer._parse_kv_event_sources({}) is None

    @pytest.mark.parametrize(
        ("sources", "error"),
        [
            (None, "invalid KV-event sources"),
            ({"0": None}, "invalid KV-event config"),
            (_kv_source(publisher="redis"), "unsupported KV-event publisher"),
            (_kv_source(topic=None), "invalid KV-event topic"),
            (_kv_source(endpoint="ipc:///tmp/kv-events"), "unsupported KV-event endpoint"),
            ({"0": {}, "1": {}}, "does not support multiple DP ranks"),
        ],
    )
    def test_invalid_sources_fail_fast(self, sources, error):
        with pytest.raises(RuntimeError, match=error):
            KVCAwareBalancer._parse_kv_event_sources(sources)

    def test_fetch_uses_vllm_http_api(self, monkeypatch):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=_kv_source())

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        def client_factory(**kwargs):
            assert kwargs == {"timeout": 5.0, "trust_env": False}
            return real_client(transport=transport)

        monkeypatch.setattr(balancer_module.httpx, "Client", client_factory)

        assert KVCAwareBalancer._fetch_kv_event_endpoints("10.0.0.1:8000") == [
            "10.0.0.1:41233",
            "10.0.0.1:41234",
            "zmq",
            "kv-events",
        ]
        assert requests[0].url == httpx.URL("http://10.0.0.1:8000/kv_event_sources")

    def test_provider_discovers_sources_in_parallel_and_tolerates_unavailable_servers(self, monkeypatch):
        monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
        concurrent_fetches = Barrier(3, timeout=5)

        def fetch(_cls, address):
            concurrent_fetches.wait()
            if address.endswith(":8001"):
                raise httpx.ConnectError("offline")
            if address.endswith(":8002"):
                return None
            return ["10.0.0.1:41233", "10.0.0.1:41234", "zmq", "kv-events"]

        monkeypatch.setattr(KVCAwareBalancer, "_fetch_kv_event_endpoints", classmethod(fetch))
        servers = {
            "s0": _RolloutServerHandle(_rollout_config(), ("10.0.0.1", 8000)),
            "s1": _RolloutServerHandle(_rollout_config(), ("10.0.0.1", 8001)),
            "s2": _RolloutServerHandle(_rollout_config(), ("10.0.0.1", 8002)),
        }

        balancer = KVCAwareBalancer(servers, _router_config(), provider_factory=_FakeCollectorManager)

        assert balancer._provider.server_addresses == {
            "s0": "10.0.0.1:8000",
            "s1": "10.0.0.1:8001",
            "s2": "10.0.0.1:8002",
        }
        assert balancer._provider.kv_event_endpoints == {"s0": ["10.0.0.1:41233", "10.0.0.1:41234", "zmq", "kv-events"]}


class TestConstruction:
    def test_wires_strategy_and_collectors(self):
        balancer = _make_balancer({"s0": "h0"})

        assert balancer._provider.started
        assert balancer._provider.collection_names == [
            "inflight_stat",
            "sticky_stat",
            "vllm_metrics",
            "vllm_zmq",
        ]
        assert isinstance(balancer._strategy, KVCacheAwareStrategy)

    def test_rejects_empty_servers_and_missing_strategy(self):
        with pytest.raises(ValueError, match="servers must be non-empty"):
            KVCAwareBalancer({}, _router_config())
        with pytest.raises(ConfigError):
            KVCAwareBalancer({"s0": "h0"}, OmegaConf.create({}))

    def test_fetches_rollout_config_once(self, monkeypatch):
        monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
        monkeypatch.setattr(KVCAwareBalancer, "_fetch_kv_event_endpoints", classmethod(lambda _cls, _addr: None))
        server = _RolloutServerHandle(_rollout_config())

        balancer = KVCAwareBalancer({"s0": server}, _router_config(), provider_factory=_FakeCollectorManager)

        assert server.get_rollout_config.calls == 1
        assert balancer._config.strategy.load_threshold == 0.75
        assert balancer._config.collector.http_interval == 3.0
        assert balancer._strategy._max_num_seqs == 128
        assert balancer._strategy._max_num_batched_tokens == 4096

    def test_uses_next_server_when_rollout_config_rpc_fails(self, monkeypatch):
        monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
        monkeypatch.setattr(KVCAwareBalancer, "_fetch_kv_event_endpoints", classmethod(lambda _cls, _addr: None))
        failed = _RolloutServerHandle(RuntimeError("actor unavailable"), ("10.0.0.1", 8000))
        healthy = _RolloutServerHandle(_rollout_config(max_num_seqs=32), ("10.0.0.1", 8001))

        balancer = KVCAwareBalancer(
            {"failed": failed, "healthy": healthy},
            _router_config(),
            provider_factory=_FakeCollectorManager,
        )

        assert failed.get_rollout_config.calls == 1
        assert healthy.get_rollout_config.calls == 1
        assert balancer._strategy._max_num_seqs == 32

    def test_invalid_capacity_values_use_defaults(self, monkeypatch):
        monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
        monkeypatch.setattr(KVCAwareBalancer, "_fetch_kv_event_endpoints", classmethod(lambda _cls, _addr: None))
        server = _RolloutServerHandle(_rollout_config(max_num_seqs=0, max_num_batched_tokens="invalid"))

        balancer = KVCAwareBalancer({"s0": server}, _router_config(), provider_factory=_FakeCollectorManager)

        assert balancer._strategy._max_num_seqs == 256
        assert balancer._strategy._max_num_batched_tokens == 2048

    @pytest.mark.parametrize(
        ("rollout_config", "expected_override", "expected_capacity"),
        [
            (
                {
                    "custom": {"agent_framework": {"router": {"load_threshold": 0.5}}},
                    "max_num_seqs": 32,
                },
                {"load_threshold": 0.5},
                32,
            ),
            (
                SimpleNamespace(
                    custom=SimpleNamespace(
                        agent_framework=SimpleNamespace(router={"load_threshold": 0.5}),
                    ),
                    max_num_seqs=32,
                ),
                {"load_threshold": 0.5},
                32,
            ),
            ({}, None, 256),
        ],
    )
    def test_rollout_config_helpers_accept_mappings_and_objects(
        self, rollout_config, expected_override, expected_capacity
    ):
        assert KVCAwareBalancer._router_override(rollout_config) == expected_override
        assert KVCAwareBalancer._rollout_config_int(rollout_config, "max_num_seqs", 256) == expected_capacity


class TestBalancerProtocol:
    def test_status_describes_runtime_state(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})

        assert balancer.get_all_servers() == ["s0", "s1"]
        assert balancer.get_status() == {
            "servers": ["s0", "s1"],
            "provider": "_FakeCollectorManager",
            "strategies": [{"type": "KVCacheAwareStrategy"}],
            "route_calls": 0,
            "sticky_size": 0,
            "total_inflight": 0,
            "probe_sessions_tracked": 0,
            "probe_sessions_total": 0,
            "probe_turns_total": 0,
            "admission_admit_total": 0,
            "admission_queue_length": 0,
            "admission_queue_admit_total": 0,
            "admission_aging_count": 0,
            "admission_ghost_pruned": 0,
        }

    def test_acquire_delegates_and_maps_selected_handle(self, monkeypatch):
        seen = {}

        def fake_route(strategy, prompt_ids, store, replicas, request_id):
            seen.update(
                strategy=strategy,
                prompt_ids=prompt_ids,
                store=store,
                replica_ids=[replica.replica_id for replica in replicas],
                request_id=request_id,
            )
            return ["s1", "s0"]

        monkeypatch.setattr(balancer_module, "route", fake_route)
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})

        assert balancer.acquire_server("request-1", [7, 8, 9]) == ("s1", "h1")
        assert seen == {
            "strategy": balancer._strategy,
            "prompt_ids": [7, 8, 9],
            "store": balancer._store,
            "replica_ids": ["s0", "s1"],
            "request_id": "request-1",
        }

    def test_empty_ranking_raises(self, monkeypatch):
        monkeypatch.setattr(balancer_module, "route", lambda *_args: [])
        balancer = _make_balancer({"s0": "h0"})

        with pytest.raises(RuntimeError, match="pool-empty"):
            balancer.acquire_server("request-1", [1])

    def test_callback_failure_does_not_break_other_callbacks(self):
        balancer = _make_balancer({"s0": "h0"})
        received = []

        def fail(*_args):
            raise RuntimeError("broken callback")

        balancer.register_call_back("custom", fail)
        balancer.register_call_back("custom", lambda value: received.append(value))

        balancer._fire("custom", "payload")

        assert received == ["payload"]

    def test_periodic_route_stats_are_cumulative(self, monkeypatch):
        balancer = _make_balancer({"s0": "h0"})
        balancer._ROUTE_LOG_EVERY = 2
        stats = []
        monkeypatch.setattr(
            balancer_module.logger,
            "info",
            lambda message, *args: stats.append(message % args),
        )

        for index in range(4):
            balancer.acquire_server(f"request-{index}", [index])

        # The turn-phase probe emits one INFO line per acquire; keep only the
        # periodic route-stats lines for this assertion.
        stats = [line for line in stats if line.startswith("route-stats:")]
        assert len(stats) == 2
        assert stats[0].startswith("route-stats: calls=2 ")
        assert stats[1].startswith("route-stats: calls=4 ")
        assert balancer._route_calls == 4
        assert balancer._route_time_total_ms >= balancer._route_time_max_ms >= 0

    def test_add_and_remove_servers_update_pool_and_counters(self):
        balancer = _make_balancer({"s0": "h0"})

        balancer.add_servers({"s0": "new-h0", "s1": "h1"})
        assert balancer._servers == {"s0": "new-h0", "s1": "h1"}
        assert balancer._inflight == {"s0": 0, "s1": 0}

        balancer.remove_servers(["s0", "missing"])
        balancer.remove_servers([])
        assert balancer.get_all_servers() == ["s1"]

    def test_statistic_callbacks_unregister_on_provider_stop(self):
        balancer = _make_balancer({"s0": "h0"})
        assert any(balancer._callbacks.values())

        balancer._provider.stop()
        balancer._provider.stop()

        assert not any(balancer._callbacks.values())


class TestTurnPhaseProbe:
    """Passive REASONING/ACTING timeline reconstruction from acquire/release."""

    def test_probe_reconstructs_reasoning_acting_timeline(self, monkeypatch):
        balancer = _make_balancer({"s0": "h0"})
        lines = []
        monkeypatch.setattr(
            balancer_module.logger,
            "info",
            lambda message, *args: lines.append(message % args),
        )

        # Turn 1: cold start; release closes REASONING; turn 2 measures the gap.
        balancer.acquire_server("sess-a", [1])
        balancer.release_server("s0", request_id="sess-a")
        balancer.acquire_server("sess-a", [1])

        acquires = [line for line in lines if "event=acquire" in line]
        releases = [line for line in lines if "event=release" in line]
        assert acquires[0] == "session-phase: rid=sess-a event=acquire turn=1 acting_gap=cold_start"
        assert releases[0].startswith("session-phase: rid=sess-a event=release reasoning_dur=")
        assert acquires[1].startswith("session-phase: rid=sess-a event=acquire turn=2 acting_gap=")
        assert "cold_start" not in acquires[1]

        status = balancer.get_status()
        assert status["probe_sessions_total"] == 1
        assert status["probe_turns_total"] == 2
        assert status["probe_sessions_tracked"] == 1

    def test_probe_ignores_release_without_request_id(self, monkeypatch):
        balancer = _make_balancer({"s0": "h0"})
        lines = []
        monkeypatch.setattr(
            balancer_module.logger,
            "info",
            lambda message, *args: lines.append(message % args),
        )

        balancer.acquire_server("sess-a", [1])
        balancer.release_server("s0", request_id=None)

        assert not [line for line in lines if "event=release" in line]
        # The anchor stays pending; the next release still closes turn 1.
        balancer.release_server("s0", request_id="sess-a")
        releases = [line for line in lines if "event=release" in line]
        assert len(releases) == 1
        assert releases[0].startswith("session-phase: rid=sess-a event=release reasoning_dur=")


# ── Turn-phase admission (design doc §5) ────────────────────────────────────

_BLOCK_SIZE = 16


def _make_admission_balancer(mode="enforce", servers=None, admission=None):
    servers = servers or {"s0": "h0", "s1": "h1"}
    adm = {"mode": mode}
    adm.update(admission or {})
    cfg = OmegaConf.create(
        {
            "strategy": {"_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig"},
            "admission": adm,
        }
    )
    return KVCAwareBalancer(servers, cfg, provider_factory=_FakeCollectorManager)


def _set_capacity(balancer, num_gpu_blocks=100, replicas=("s0", "s1")):
    """Token capacity = num_gpu_blocks × 16 (default 100 blocks = 1600 tokens)."""
    store = balancer._store
    store.set_block_size(_BLOCK_SIZE)
    store.refresh_metrics({sid: {MetricKey.NUM_GPU_BLOCKS: num_gpu_blocks} for sid in replicas})


def _release_on(balancer, rid, replica, ctx, gap, now=None):
    """Plant a session in the ACTING state (released `gap` seconds ago).

    Returns the anchor timestamp so tests can share one exact ``now`` with
    ledger evaluations (no perf_counter drift between setup and assert).
    """
    if now is None:
        now = time.perf_counter()
    balancer._probe_release_ts[rid] = now - gap
    balancer._probe_ctx_tokens[rid] = ctx
    balancer._probe_last_replica[rid] = replica
    return now


def _ledger_of(balancer, ledgers, replica):
    return next(led for led in ledgers if led.replica_id == replica)


class TestAdmissionLedger:
    """Two-level commitment ledger: pin (hard) + acting_hold (soft) + reserves."""

    def test_avail_admit_formula_components(self):
        balancer = _make_admission_balancer()
        _set_capacity(balancer, num_gpu_blocks=6250)  # 100k tokens
        now = _release_on(balancer, "act-1", "s0", ctx=20000, gap=0)
        balancer._store.pin_inflight_blocks("s0", [f"p{i}" for i in range(100)])  # 1600 tokens
        balancer._inflight["s0"] = 2  # decode_reserve = 2 × 512

        _cap, _bs, ledgers = balancer._admission_ledger(balancer._replica_infos(), now)

        assert _cap == 100000
        s0 = _ledger_of(balancer, ledgers, "s0")
        s1 = _ledger_of(balancer, ledgers, "s1")
        # avail = cap − pin_tokens − acting_hold − decode_reserve − buffer_base
        assert s0.avail_admit == pytest.approx(100000 - 1600 - 20000 - 1024 - 512)
        assert s1.avail_admit == pytest.approx(100000 - 512)
        assert s0.pin_blocks == 100
        assert s0.decode_reserve == 1024

    def test_hold_decays_at_half_life(self):
        balancer = _make_admission_balancer()
        now = _release_on(balancer, "w", "s0", ctx=8000, gap=120)  # gap == τ → half survived

        assert balancer._acting_hold("s0", now) == pytest.approx(4000)
        assert balancer._acting_hold("s0", now + 120) == pytest.approx(2000)
        assert balancer._acting_hold("s1", now) == 0  # claim lives on its last replica only

    def test_hold_disabled_via_q9_switch(self):
        balancer = _make_admission_balancer(admission={"enable_acting_hold": False})
        now = time.perf_counter()
        _release_on(balancer, "w", "s0", ctx=8000, gap=0)

        assert balancer._acting_hold("s0", now) == 0
        assert balancer._hold_of("w", now) == 0

    def test_pin_and_hold_do_not_double_count(self):
        balancer = _make_admission_balancer()
        _set_capacity(balancer)
        now = time.perf_counter()
        # In-flight request: pinned, no release anchor → pin side only.
        balancer._store.pin_inflight_blocks("s0", ["e1", "e2"])
        # Released session: anchor present, nothing pinned → hold side only.
        _release_on(balancer, "r1", "s0", ctx=1000, gap=5)

        _cap, _bs, ledgers = balancer._admission_ledger(
            [r for r in balancer._replica_infos() if r.replica_id == "s0"], now
        )
        s0 = ledgers[0]
        assert s0.pin_blocks == 2
        assert s0.acting_hold == pytest.approx(1000 * 2.0 ** (-5 / 120))

    def test_queued_session_keeps_claim_while_waiting(self):
        """A deferred turn-2 request keeps its hold claim (anchor restored) and
        the retry re-prices it from the ORIGINAL release (自融资闭环)."""
        balancer = _make_admission_balancer()
        _set_capacity(balancer)  # 1600 tokens per replica
        rid = "queued-1"

        # Turn 1 fits (gate bypassed — nothing on the ledger yet).
        server_id, _handle = balancer.acquire_server(rid, list(range(20)))
        balancer.release_server(server_id, request_id=rid)
        assert balancer._probe_turns[rid] == 1

        # Fill both replicas so turn 2 cannot fit (88 manual blocks = 1408 tokens).
        for sid in ("s0", "s1"):
            balancer._store.pin_inflight_blocks(sid, [f"m{i}" for i in range(88)])
            balancer._inflight[sid] = 1
        with pytest.raises(CapacityDeferred, match="admission-defer"):
            balancer.acquire_server(rid, list(range(20)))

        # The claim survived queueing: anchor restored, gap keeps growing.
        assert balancer._find_in_queue(rid) is not None
        assert rid in balancer._probe_release_ts
        assert balancer._hold_of(rid, time.perf_counter()) == pytest.approx(20, rel=0.05)
        assert balancer.get_status()["admission_queue_length"] == 1

        # Free capacity on s0 → the retry dequeues onto the sticky replica;
        # inc ≈ 0 because the request's own hold covers its chain (self-financing).
        balancer._store.unpin_inflight_blocks("s0", [f"m{i}" for i in range(80)])
        server_id, _handle = balancer.acquire_server(rid, list(range(20)))
        assert server_id == "s0"
        assert balancer._find_in_queue(rid) is None
        assert balancer._admitted_ready == {}
        # The retry is the SAME logical turn — not re-counted.
        assert balancer._probe_turns[rid] == 2


class TestAdmissionIncrement:
    """inc = predict_pin_delta − hold_s (§5.4): dedup, fallback, self-financing."""

    def test_store_predict_pin_delta_dedups_shared_prefix(self):
        store = DataStore()
        store.set_block_size(_BLOCK_SIZE)
        chain = [f"h{i}" for i in range(10)]
        store.pin_inflight_blocks("s0", chain[:8])

        assert store.predict_pin_delta("s0", chain) == 2  # 8 covered, 2 new
        assert store.predict_pin_delta("s1", chain) == 10  # nothing pinned there
        assert store.predict_pin_delta("s0", []) == 0
        # Ref-count: one unpin of a twice-pinned hash leaves it held.
        store.pin_inflight_blocks("s0", chain[:2])
        store.unpin_inflight_blocks("s0", chain[:8])
        assert store.predict_pin_delta("s0", chain) == 8

    def test_self_financing_return_is_free_at_gap_zero(self):
        """gap→0: the session's own hold covers its whole chain → inc ≈ 0."""
        balancer = _make_admission_balancer()
        _set_capacity(balancer)
        now = time.perf_counter()
        _release_on(balancer, "back", "s0", ctx=320, gap=0)  # 20 blocks
        chain = [f"c{i}" for i in range(20)]
        _cap, block_size, ledgers = balancer._admission_ledger(balancer._replica_infos(), now)

        view_s0 = balancer._increment_for(
            ledgers[0], chain, 320, own_hold=320, request_id="back", block_size=block_size
        )
        view_s1 = balancer._increment_for(
            ledgers[1], chain, 320, own_hold=320, request_id="back", block_size=block_size
        )

        assert view_s0.inc == pytest.approx(0)  # 320 − 320: free return onto its own cache
        assert view_s1.inc == pytest.approx(320)  # elsewhere: full price, no claim there

    def test_fallback_counts_full_ctx_without_chain(self):
        """No chain available (block size unknown) → whole prompt counted new."""
        balancer = _make_admission_balancer()
        ledger = balancer_module._ReplicaLedger(
            "s0", pin_blocks=0, acting_hold=0.0, decode_reserve=0, avail_admit=10000.0
        )

        view = balancer._increment_for(ledger, [], ctx_tokens=3200, own_hold=0.0, request_id="x", block_size=16)

        assert view.predicted == 200  # ceil(3200 / 16)
        assert view.inc == pytest.approx(3200)

    def test_counterexample_old_criterion_would_oversell(self):
        """§5.4 numbers: cap=100k; pin 30k; holds A 40k + C 5k + B 17.8k →
        avail ≈ 6.1k. Returning B (hit 93% in the old criterion: need 2.1k)
        must DEFER under the ledger-consistent inc ≈ 12.2k."""
        balancer = _make_admission_balancer()
        _set_capacity(balancer, num_gpu_blocks=6250)  # 100k tokens
        now = time.perf_counter()
        balancer._store.pin_inflight_blocks("s0", [f"e{i}" for i in range(1875)])  # E1: 30k tokens
        balancer._inflight["s0"] = 1
        _release_on(balancer, "A", "s0", ctx=40000, gap=0, now=now)
        _release_on(balancer, "C", "s0", ctx=5000, gap=0, now=now)
        _release_on(balancer, "B", "s0", ctx=30000, gap=90, now=now)

        _cap, block_size, ledgers = balancer._admission_ledger(
            [r for r in balancer._replica_infos() if r.replica_id == "s0"], now
        )
        s0 = ledgers[0]
        hold_b = 30000 * 2.0 ** (-90 / 120)  # ≈ 17.8k
        # 100k − 30k(pin) − 45k(A+C) − hold_b − 512(resv) − 512(buffer)
        assert s0.avail_admit == pytest.approx(100000 - 30000 - 45000 - hold_b - 1024, rel=1e-6)

        # B's chain is fully unpinned (released, cache evicted from the pin set):
        # the old need=ctx×(1−hit)=2.1k would admit; inc must not.
        chain_b = [f"b{i}" for i in range(1875)]
        view = balancer._increment_for(s0, chain_b, 30000, own_hold=hold_b, request_id="B", block_size=block_size)

        assert view.inc == pytest.approx(30000 - hold_b, rel=1e-6)  # ≈ 12.2k
        assert not view.feasible()  # inc 12.2k > avail 6.1k → defer, invariant holds


class TestAdmissionGate:
    def test_gate_narrows_feasible_set_and_predict_checks(self, monkeypatch):
        captured = {}
        lines = []

        def fake_route(strategy, prompt_ids, store, replicas, request_id):
            captured["replicas"] = [r.replica_id for r in replicas]
            return [r.replica_id for r in replicas]

        monkeypatch.setattr(balancer_module, "route", fake_route)
        monkeypatch.setattr(balancer_module.logger, "info", lambda msg, *a: lines.append(msg % a))
        balancer = _make_admission_balancer()
        _set_capacity(balancer)
        # s0 full (88 manual blocks + decode reserve), s1 free.
        balancer._store.pin_inflight_blocks("s0", [f"m{i}" for i in range(88)])
        balancer._inflight["s0"] = 1

        server_id, _handle = balancer.acquire_server("gate-1", list(range(20)))

        assert captured["replicas"] == ["s1"]  # sticky yields to the hard gate (D1)
        assert server_id == "s1"
        assert balancer.get_status()["admission_admit_total"] == 1
        # 20 tokens → one full block, unpinned on s1: the collector's real pin
        # must land exactly the predicted increment (§5.8 predict-check).
        assert "predict-check: rid=gate-1 predicted=1 actual=1 delta=0" in lines

    def test_gate_off_by_default_routes_everywhere(self, monkeypatch):
        captured = {}

        def fake_route(strategy, prompt_ids, store, replicas, request_id):
            captured["replicas"] = [r.replica_id for r in replicas]
            return [r.replica_id for r in replicas]

        monkeypatch.setattr(balancer_module, "route", fake_route)
        balancer = _make_balancer()  # no admission config → off
        _set_capacity(balancer)
        for sid in ("s0", "s1"):
            balancer._store.pin_inflight_blocks(sid, [f"m{i}" for i in range(99)])

        balancer.acquire_server("off-1", list(range(20)))

        assert captured["replicas"] == ["s0", "s1"]  # current behavior untouched

    def test_shadow_mode_logs_but_never_intercepts(self, monkeypatch):
        lines = []
        monkeypatch.setattr(balancer_module.logger, "info", lambda msg, *a: lines.append(msg % a))
        monkeypatch.setattr(
            balancer_module,
            "route",
            lambda _s, _p, _st, replicas, _r: [r.replica_id for r in replicas],
        )
        balancer = _make_admission_balancer(mode="shadow")
        _set_capacity(balancer)
        for sid in ("s0", "s1"):
            balancer._store.pin_inflight_blocks(sid, [f"m{i}" for i in range(88)])
            balancer._inflight[sid] = 1

        server_id, _handle = balancer.acquire_server("shadow-1", list(range(20)))

        assert server_id in ("s0", "s1")  # routed despite infeasible ledger
        assert balancer.get_status()["admission_queue_length"] == 0  # never queues
        assert any("decision=shadow-defer" in line for line in lines)
        assert not any("decision=enqueue" in line for line in lines)

    def test_cap_unknown_bypasses_gate(self, monkeypatch):
        captured = {}

        def fake_route(strategy, prompt_ids, store, replicas, request_id):
            captured["replicas"] = [r.replica_id for r in replicas]
            return [r.replica_id for r in replicas]

        monkeypatch.setattr(balancer_module, "route", fake_route)
        balancer = _make_admission_balancer()  # enforce, but no KV events yet

        balancer.acquire_server("cold-1", list(range(5000)))  # huge prompt still routes

        assert captured["replicas"] == ["s0", "s1"]  # §6.3: cap==0 → gate bypassed


class TestAdmissionQueue:
    def _filled_balancer(self):
        balancer = _make_admission_balancer()
        _set_capacity(balancer)
        for sid in ("s0", "s1"):
            balancer._store.pin_inflight_blocks(sid, [f"m{i}" for i in range(88)])
            balancer._inflight[sid] = 1
        return balancer

    def test_no_jump_new_arrival_queues_behind_backlog(self):
        balancer = self._filled_balancer()
        with pytest.raises(CapacityDeferred):
            balancer.acquire_server("first", list(range(20)))  # enqueued

        # Free s1: a fresh small arrival WOULD fit, but strict FIFO parks it
        # too (no-jump, §6.4) — even a bigger one is parked the same way.
        balancer._store.unpin_inflight_blocks("s1", [f"m{i}" for i in range(80)])
        with pytest.raises(CapacityDeferred, match="admission-defer"):
            balancer.acquire_server("second", list(range(1000)))
        assert balancer.get_status()["admission_queue_length"] == 2

        # The queue head's retry dequeues first (FIFO); the follower's
        # increment (1000 tokens → 62 blocks) no longer fits behind it.
        server_id, _handle = balancer.acquire_server("first", list(range(20)))
        assert server_id == "s1"
        assert [e.request_id for e in balancer._defer_queue] == ["second"]

    def test_retry_rescan_throttled(self, monkeypatch):
        """Deep-queue guard (§5.6.3): 1 s retry storms must not rescan the
        whole queue every time — only the first retry inside the interval."""
        balancer = self._filled_balancer()
        with pytest.raises(CapacityDeferred):
            balancer.acquire_server("q1", list(range(20)))  # enqueued

        scans = []
        real_scan = balancer._scan_defer_queue

        def counting_scan(replicas, now):
            scans.append(now)
            return real_scan(replicas, now)

        monkeypatch.setattr(balancer, "_scan_defer_queue", counting_scan)

        # Two retries a few µs apart: only the first triggers a rescan.
        with pytest.raises(CapacityDeferred):
            balancer.acquire_server("q1", list(range(20)))
        assert balancer._last_scan_ts > 0
        with pytest.raises(CapacityDeferred):
            balancer.acquire_server("q1", list(range(20)))
        assert len(scans) == 1
        assert [e.request_id for e in balancer._defer_queue] == ["q1"]  # both defers were honest

        # Past the interval the retry rescans again (hold-decay transitions
        # wait out at most _RESCAN_MIN_INTERVAL).
        balancer._last_scan_ts -= balancer_module.KVCAwareBalancer._RESCAN_MIN_INTERVAL * 2
        with pytest.raises(CapacityDeferred):
            balancer.acquire_server("q1", list(range(20)))
        assert len(scans) == 2

    def test_priority_groups_warm_new_cold_fifo(self):
        balancer = _make_admission_balancer()
        now = time.perf_counter()

        def add(rid, turn, enqueued_age, ctx=100, gap=None):
            balancer._defer_queue.append(
                balancer_module._DeferredEntry(
                    request_id=rid,
                    enqueued_at=now - enqueued_age,
                    turn=turn,
                    ctx_tokens=ctx,
                    block_hashes=[f"{rid}-h0"],
                    sticky="s0",
                )
            )
            if gap is not None:
                balancer._probe_release_ts[rid] = now - gap
                balancer._probe_ctx_tokens[rid] = ctx
                balancer._probe_last_replica[rid] = "s0"

        add("cold", turn=3, enqueued_age=1, gap=200)  # G3: waiting eroded the claim
        add("new", turn=1, enqueued_age=2)  # G2
        add("warm", turn=3, enqueued_age=3, gap=10)  # G1: fresh cache
        add("aged", turn=3, enqueued_age=61, gap=200)  # G3 by age, G1 by aging

        view = balancer._priority_view(now)

        assert [e.request_id for e in view] == ["aged", "warm", "new", "cold"]
        assert balancer._admission_aging_count == 1  # counted once, stable across calls
        balancer._priority_view(now)
        assert balancer._admission_aging_count == 1

    def test_skip_lets_smaller_entry_pass(self):
        balancer = _make_admission_balancer()
        _set_capacity(balancer)
        # s0 half full (avail = 1600 − 800 − 512 = 288), s1 full.
        balancer._store.pin_inflight_blocks("s0", [f"m{i}" for i in range(50)])
        balancer._store.pin_inflight_blocks("s1", [f"m{i}" for i in range(88)])
        balancer._inflight["s1"] = 1
        now = time.perf_counter()
        for rid, n_blocks in (("big", 60), ("small", 1)):  # same group, FIFO order
            balancer._defer_queue.append(
                balancer_module._DeferredEntry(
                    request_id=rid,
                    enqueued_at=now,
                    turn=1,
                    ctx_tokens=n_blocks * _BLOCK_SIZE,
                    block_hashes=[f"{rid}-h{i}" for i in range(n_blocks)],
                    sticky="s0",
                )
            )

        balancer._scan_defer_queue(balancer._replica_infos(), now)

        # big: 960 tokens > 288 (s0) and > −832 (s1) → skipped; small: 16 ≤ 288
        # → admitted onto its sticky replica.
        assert "small" in balancer._admitted_ready
        assert [e.request_id for e in balancer._defer_queue] == ["big"]
        assert balancer.get_status()["admission_queue_admit_total"] == 1

    def test_ghost_ttl_prunes_dead_entries(self):
        balancer = _make_admission_balancer()
        now = time.perf_counter()
        _release_on(balancer, "ghost", "s0", ctx=100, gap=0)
        balancer._defer_queue.append(
            balancer_module._DeferredEntry(
                request_id="ghost",
                enqueued_at=now - 601,
                turn=2,
                ctx_tokens=100,
                block_hashes=["g-h0"],
                sticky="s0",
            )
        )

        balancer._admission_gc(now + 1)

        assert balancer._defer_queue == []
        assert balancer.get_status()["admission_ghost_pruned"] == 1
        assert "ghost" not in balancer._probe_release_ts  # hold claim dropped with the ghost


class TestCapacityDeferredContract:
    @staticmethod
    def _verl_branch(message: str, only_hybrid: bool) -> str:
        """Mirror of FullyAsyncLLMServerClient._acquire_server's retry loop."""
        if "No available servers in load balancer" in message and only_hybrid:
            return "retry-legacy"
        if "admission-defer" in message:
            return "retry-admission"
        return "raise"

    def test_exception_is_runtime_error_with_dual_markers(self):
        exc = CapacityDeferred("admission-defer rid=r pos=1 | No available servers in load balancer")
        assert isinstance(exc, RuntimeError)
        assert "admission-defer" in str(exc)
        assert "No available servers in load balancer" in str(exc)

    def test_verl_branch_behavior(self):
        defer = str(CapacityDeferred("admission-defer rid=r pos=1 | No available servers in load balancer"))
        # Upgraded loop: own marker catches defer even in mixed deployments.
        assert self._verl_branch(defer, only_hybrid=False) == "retry-admission"
        # Fork diff lost: the legacy branch still retries (兜底), never dies.
        assert self._verl_branch(defer, only_hybrid=True) == "retry-legacy"
        # Pool-empty alignment revives the legacy branch (§5.6.4 死分支复活).
        assert self._verl_branch("pool-empty | No available servers in load balancer", True) == "retry-legacy"
        # Real bugs stay loud.
        assert self._verl_branch("strategy crashed: KeyError", False) == "raise"

    def test_pool_empty_messages_carry_verl_string(self):
        balancer = _make_balancer({"s0": "h0"})
        with pytest.raises(RuntimeError) as excinfo:
            route(type(balancer._strategy), None, balancer._store, [], "r")

        assert "pool-empty" in str(excinfo.value)
        assert "No available servers in load balancer" in str(excinfo.value)
