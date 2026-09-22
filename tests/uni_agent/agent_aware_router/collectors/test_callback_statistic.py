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

"""Tests for the Balancer-callback statistic path.

Covers the Phase-1 components that turn Balancer callbacks into store writes:
``StatisticEvent`` (pack contract), ``StickyParser`` / ``InflightParser``
(parsers), and ``CallbackTransport`` (the pure-forwarder transport that
registers on the Balancer). Together these mirror what the network collectors
do, but driven by the Balancer's own request-path hooks.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import BLOCK_SIZE

from uni_agent.agent_aware_router.collectors.parse import MetricsUpdate, StickyUpdate
from uni_agent.agent_aware_router.collectors.parse.basic.inflight import InflightParser
from uni_agent.agent_aware_router.collectors.parse.basic.sticky import StickyParser
from uni_agent.agent_aware_router.collectors.transport.callback import (
    CallbackTransport,
    StatisticEvent,
)
from uni_agent.agent_aware_router.strategies import route
from uni_agent.agent_aware_router.types import MetricKey

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


class TestStatisticEvent:
    def test_inputs_params(self):
        ev = StatisticEvent("on_acquire", request_id="r1", replica_id="s0", server_ids=("s0", "s1"))
        assert ev.event == "on_acquire"
        assert ev.request_id == "r1"
        assert ev.replica_id == "s0"
        assert ev.server_ids == ("s0", "s1")
        assert ev.prompt_len == 0  # default — no prompt forwarded

        with pytest.raises(AttributeError):
            ev.event = "x"  # type: ignore[misc]


class TestStickyParser:
    def test_on_acquire_emits_put(self):
        upd = StickyParser().parse(StatisticEvent("on_acquire", request_id="r1", replica_id="s0"), "")
        assert isinstance(upd, StickyUpdate)
        assert upd.action == "put"
        assert upd.request_id == "r1"
        assert upd.replica_id == "s0"

    def test_on_servers_removed_emits_invalidate_replica(self):
        upd = StickyParser().parse(StatisticEvent("on_servers_removed", server_ids=["s0", "s1"]), "")
        assert isinstance(upd, StickyUpdate)
        assert upd.action == "invalidate_replica"
        assert upd.replica_ids == ("s0", "s1")

    def test_return_is_none(self):
        d = StickyParser()
        assert d.parse(b"bytes", "") is None
        assert d.parse("str", "") is None
        assert d.parse(StatisticEvent("on_acquire"), "") is None
        assert d.parse(StatisticEvent("on_release", replica_id="s0"), "") is None


class TestInflightParser:
    def test_on_acquire_emits_inflight_plus_dispatched_delta(self):
        upd = InflightParser().parse(StatisticEvent("on_acquire", request_id="r1", replica_id="s0", prompt_len=42), "")
        assert isinstance(upd, MetricsUpdate)
        assert upd.node_id == "s0"
        assert upd.metrics == {
            MetricKey.INFLIGHT_COUNT: 1,
            MetricKey.INFLIGHT_TOKENS: 42,  # no prompt forwarded → 0 token delta
            MetricKey.DISPATCHED_COUNT: 1,
            MetricKey.PROMPT_LEN_SUM: 42,  # no prompt forwarded → 0 length delta
        }
        assert upd.is_delta is True
        assert upd.request_id == "r1"  # carried so the collector attributes the dispatch's turn
        assert upd.prompt_ids == ()  # no token list on the event → nothing to look up

    def test_on_acquire_forwards_prompt_ids(self):
        """The token list rides along so the collector can net out the cache hit."""
        upd = InflightParser().parse(
            StatisticEvent(
                "on_acquire",
                request_id="r1",
                replica_id="s0",
                prompt_len=3,
                prompt_ids=(1, 2, 3),
            ),
            "",
        )
        assert isinstance(upd, MetricsUpdate)
        assert upd.prompt_ids == (1, 2, 3)
        assert upd.metrics[MetricKey.INFLIGHT_TOKENS] == 3  # raw until the collector rewrites it

    def test_on_release_emits_inflight_minus_completed_delta(self):
        upd = InflightParser().parse(StatisticEvent("on_release", replica_id="s0", request_id="r1"), "")
        assert isinstance(upd, MetricsUpdate)
        assert upd.metrics == {
            MetricKey.INFLIGHT_COUNT: -1,
            MetricKey.COMPLETED_COUNT: 1,
            # no INFLIGHT_TOKENS here — verl #7115 releases carry no length; the
            # collector folds the negative token delta from acquire-time bookkeeping
        }
        assert upd.is_delta is True
        assert upd.request_id == "r1"  # carried so the collector can attribute the release

    def test_returns_none(self):
        assert InflightParser().parse(b"bytes", "") is None
        assert InflightParser().parse(StatisticEvent("on_servers_removed", server_ids=["s0"]), "") is None
        assert InflightParser().parse(StatisticEvent("on_acquire"), "") is None


class _FakeBalancer:
    """Minimal Balancer stand-in exposing register/un_register_call_back."""

    def __init__(self):
        self.callbacks: dict[str, list] = {}

    def register_call_back(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def un_register_call_back(self, event, fn):
        lst = self.callbacks.get(event, [])
        if fn in lst:
            lst.remove(fn)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestCallbackTransport:
    def test_is_async_false(self):
        assert CallbackTransport(_FakeBalancer()).is_async is False

    def test_subscribe_registers_three_hooks_and_forwards(self):
        balancer = _FakeBalancer()
        transport = CallbackTransport(balancer)
        received: list = []
        _run(transport.subscribe(lambda raw, nid: received.append(raw)))

        assert set(balancer.callbacks) == {"on_acquire", "on_release", "on_servers_removed"}
        # one callback per hook
        assert all(len(lst) == 1 for lst in balancer.callbacks.values())

        balancer.callbacks["on_acquire"][0]("r1", "s0")
        balancer.callbacks["on_release"][0]("s0")
        balancer.callbacks["on_servers_removed"][0](["s1", "s2"])

        assert received == [
            StatisticEvent("on_acquire", request_id="r1", replica_id="s0"),
            StatisticEvent("on_release", replica_id="s0"),
            StatisticEvent("on_servers_removed", server_ids=("s1", "s2")),
        ]

    def test_on_acquire_callback_forwards_prompt_ids(self):
        balancer = _FakeBalancer()
        transport = CallbackTransport(balancer)
        received: list = []
        _run(transport.subscribe(lambda raw, nid: received.append(raw)))

        balancer.callbacks["on_acquire"][0]("r1", "s0", [7, 8, 9])
        assert received == [
            StatisticEvent("on_acquire", request_id="r1", replica_id="s0", prompt_len=3, prompt_ids=(7, 8, 9)),
        ]

    def test_on_release_callback_forwards_request_id(self):
        balancer = _FakeBalancer()
        transport = CallbackTransport(balancer)
        received: list = []
        _run(transport.subscribe(lambda raw, nid: received.append(raw)))

        # (server_id, request_id) — the 2nd arg threads the routing request id
        # so the collector can attribute the release (turn / token subtraction).
        balancer.callbacks["on_release"][0]("s0", "r1")
        assert received == [
            StatisticEvent("on_release", replica_id="s0", request_id="r1"),
        ]

    def test_stop_unregisters_all(self):
        balancer = _FakeBalancer()
        transport = CallbackTransport(balancer)
        _run(transport.subscribe(lambda raw, nid: None))
        transport.stop()
        assert all(not lst for lst in balancer.callbacks.values())

    def test_stop_is_idempotent(self):
        transport = CallbackTransport(_FakeBalancer())
        _ = transport.subscribe  # method exists; not yet subscribed
        transport.stop()  # must not raise even with empty registry


class TestCollectorCallbackIntegration:
    """End-to-end: Collector(CallbackTransport, parser) → handler → DataStore.

    Exercises the is_async=False start path (tmp loop runs the loop-free
    subscribe), the handler's StickyUpdate/MetricsUpdate dispatch, and the
    store writes — the whole Phase-1 statistic chain.
    """

    @pytest.fixture(autouse=True)
    def _reset_singletons(self):
        from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
        from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
        from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore

        PerReplicaStore._instance = None
        PerRequestStore._instance = None
        KVCacheStore._instance = None  # carries per-request block pins now
        yield
        PerReplicaStore._instance = None
        PerRequestStore._instance = None
        KVCacheStore._instance = None

    def test_sticky_collector_writes_binding_on_acquire(self):
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), StickyParser())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0")
            assert DataStore().get_sticky_binding("r1") == "s0"
        finally:
            collector.stop()

    def test_inflight_collector_applies_acquire_release_delta_and_prompt_len(self):
        """Feature: InflightParser on_acquire/on_release drive inflight/token/dispatched/completed
          deltas and accumulate PROMPT_LEN_SUM from the prompt_ids length.
        Description: two acquires (r1 len-3, r2 len-10) to s0, one release of r1 (-3 tokens, folded
          from acquire-time bookkeeping — releases carry no length under verl #7115), and an
          acquire to s1 with no prompt — exercising both the delta metrics and prompt-len sum.
        Expectation:
          s0 INFLIGHT_COUNT=1, INFLIGHT_TOKENS=10 (3+10-3), DISPATCHED_COUNT=2, COMPLETED_COUNT=1
          s0 PROMPT_LEN_SUM=13 (3+10); s1 PROMPT_LEN_SUM=0 (None prompt → 0)
        """
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightParser())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", [1, 2, 3])  # +1 inflight, +3 tokens, len 3
            balancer.callbacks["on_acquire"][0]("r2", "s0", list(range(10)))  # +1 inflight, +10 tokens, len 10
            balancer.callbacks["on_release"][0]("s0", "r1")  # -1 inflight, -3 tokens (bookkeeping), +1 completed
            balancer.callbacks["on_acquire"][0]("r3", "s1", None)  # no prompt → PROMPT_LEN_SUM 0
            ds = DataStore()
            # acquire/release delta metrics on s0
            assert ds.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 1
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 10  # 3 + 10 - 3
            assert ds.get_metric("s0", MetricKey.DISPATCHED_COUNT) == 2
            assert ds.get_metric("s0", MetricKey.COMPLETED_COUNT) == 1
            # prompt-len sum: s0 accumulates 3+10; s1 gets 0 (None prompt not forwarded)
            assert ds.get_metric("s0", MetricKey.PROMPT_LEN_SUM) == 13
            assert ds.get_metric("s1", MetricKey.PROMPT_LEN_SUM) == 0
        finally:
            collector.stop()

    def test_acquire_books_uncached_tokens_and_release_subtracts_the_same(self):
        """Feature: INFLIGHT_TOKENS nets out the chosen replica's prefix-cache hit.
        Description: a 64-token prompt (4 blocks of 16) whose first two blocks
          the replica already caches → gpu_hit 0.5. Acquire must book 32 tokens
          (the uncached half), not 64; the release must subtract exactly that 32.
        Expectation:
          after acquire: INFLIGHT_TOKENS=32, PROMPT_LEN_SUM=64 (raw request size)
          after release: INFLIGHT_TOKENS=0, COMPLETED_COUNT=1
        """
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.store.data_store import DataStore
        from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes

        prompt = list(range(4 * BLOCK_SIZE))
        ds = DataStore()
        ds.set_block_size(BLOCK_SIZE)
        # Routing resolves the chain first (the strategy's per-request memo) —
        # the collector reuses it. Only the first half of it is stored on s0.
        chain = resolve_prefix_hashes(prompt, "r1", ds)
        assert len(chain) == 4
        ds.add_kv_blocks("s0", chain[:2])

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightParser())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 32  # 64 × (1 − 0.5)
            assert ds.get_metric("s0", MetricKey.PROMPT_LEN_SUM) == 64  # evidence stays raw
            assert ds.get_per_request("r1", "inflight_tokens", None) == 32

            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0
            assert ds.get_metric("s0", MetricKey.COMPLETED_COUNT) == 1
        finally:
            collector.stop()

    def test_fully_cached_prompt_books_zero_inflight_tokens(self):
        """A 100%-cached prompt adds no KV footprint → books 0 (release stays 0)."""
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.store.data_store import DataStore
        from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes

        prompt = list(range(2 * BLOCK_SIZE))
        ds = DataStore()
        ds.set_block_size(BLOCK_SIZE)
        chain = resolve_prefix_hashes(prompt, "r1", ds)
        ds.add_kv_blocks("s0", chain)

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightParser())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0
            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0  # symmetric, not negative
        finally:
            collector.stop()

    def test_missing_prompt_ids_or_block_size_falls_back_to_raw_len(self):
        """No token list / no learned block size → no hit evidence → book raw plen."""
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.store.data_store import DataStore

        ds = DataStore()  # block_size never learned → the chain is unresolvable
        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightParser())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", list(range(64)))  # no block size
            balancer.callbacks["on_acquire"][0]("r2", "s0", None)  # no prompt at all
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 64  # raw, not 0
        finally:
            collector.stop()

    def test_inflight_collector_tracks_turns_and_turn_sum(self):
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightParser())
        collector.start()
        try:
            # r1 dispatched three times (turns 1,2,3) to s0,s1,s0; r2 once (turn 1)
            # to s1. Each acquire also bumps INFLIGHT/DISPATCHED; the dispatch's
            # turn is added to the receiving replica's INFLIGHT_TURN_SUM (the
            # inflight collector carries request_id and does the PerRequestStore work).
            balancer.callbacks["on_acquire"][0]("r1", "s0")
            balancer.callbacks["on_acquire"][0]("r1", "s1")
            balancer.callbacks["on_acquire"][0]("r1", "s0")
            balancer.callbacks["on_acquire"][0]("r2", "s1")

            ds = DataStore()
            # per-request turn is global (Nth dispatch of that request_id overall)
            assert ds.get_per_request("r1", "turn", 0) == 3
            assert ds.get_per_request("r2", "turn", 0) == 1
            # ...but each dispatch's turn is added to the receiving replica's
            # INFLIGHT_TURN_SUM (per-replica, in PerReplicaStore; no releases here so
            # the in-flight sum equals the cumulative): s0 got r1's turn-1 + turn-3 = 4;
            # s1 got r1's turn-2 + r2's turn-1 = 3.
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TURN_SUM) == 4
            assert ds.get_metric("s1", MetricKey.INFLIGHT_TURN_SUM) == 3
        finally:
            collector.stop()

    def test_request_id_without_dispatch_does_not_record_turn(self):
        """Turn is gated on DISPATCHED_COUNT (a dispatch), not request_id presence.

        A hypothetical per-request delta that carries request_id but does NOT
        bump DISPATCHED_COUNT must not touch the turn table or INFLIGHT_TURN_SUM —
        guards against a future request_id-carrying update overloading the turn path.
        """
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.collectors.parse import MetricsUpdate
        from uni_agent.agent_aware_router.store.data_store import DataStore

        collector = Collector(CallbackTransport(_FakeBalancer()), InflightParser())
        # request_id present, but no DISPATCHED_COUNT in the delta → not a dispatch.
        collector._write_metrics_update(
            MetricsUpdate(
                node_id="s0",
                metrics={MetricKey.INFLIGHT_COUNT: 1},
                is_delta=True,
                request_id="r1",
            )
        )
        ds = DataStore()
        assert ds.get_per_request("r1", "turn", 0) == 0  # no dispatch → no turn recorded
        assert ds.get_metric("s0", MetricKey.INFLIGHT_TURN_SUM) == 0


class TestInflightBlockPin:
    """Held-block account: acquire pins, release unpins, ``INFLIGHT_BLOCKS`` is absolute.

    ``INFLIGHT_BLOCKS`` is what the capacity strategy turns into free capacity
    (``avail = cap − blocks × block_size``). It follows the *held* block set, not
    the booked-token sum: shared prefixes count once, and a release that frees
    nothing (another request still holds those blocks) must leave the gauge
    where it was.
    """

    @pytest.fixture(autouse=True)
    def _reset_singletons(self):
        from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
        from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
        from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore

        PerReplicaStore._instance = None
        PerRequestStore._instance = None
        KVCacheStore._instance = None
        yield
        PerReplicaStore._instance = None
        PerRequestStore._instance = None
        KVCacheStore._instance = None

    @staticmethod
    def _start(block_size: int = BLOCK_SIZE):
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.store.data_store import DataStore

        ds = DataStore()
        ds.set_block_size(block_size)
        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightParser())
        collector.start()
        return collector, balancer, ds

    def test_concurrent_same_prefix_counts_once_and_survives_one_release(self):
        """Two rollouts of one prompt: dispatched twice, held once, freed once.

        Feature: the held gauge de-duplicates by block hash and a release only
          drops what the releasing request uniquely held.
        Description: r1 and r2 dispatch the identical 4-block prompt to s0 (no
          BlockStored yet, so neither is a cache hit — the case where the v1
          token sum booked both prompts twice over).
        Expectation: ``INFLIGHT_BLOCKS`` = 4 after both acquires (not 8);
          ``INFLIGHT_TOKENS`` books 64 then 0 (r2 allocates nothing);
          releasing r1 leaves 4 (r2 still holds all of them); releasing r2 → 0.
        """
        prompt = list(range(4 * BLOCK_SIZE))
        collector, balancer, ds = self._start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 4
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 4 * BLOCK_SIZE

            balancer.callbacks["on_acquire"][0]("r2", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 4  # dedup
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 4 * BLOCK_SIZE  # +0

            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 4  # r2 still holds all
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0  # r1's booking only

            balancer.callbacks["on_release"][0]("s0", "r2")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 0
            assert ds.get_metric("s0", MetricKey.COMPLETED_COUNT) == 2
        finally:
            collector.stop()

    def test_subset_prefix_release_frees_only_the_unshared_blocks(self):
        """A prefix-sharing pair releases block by block, not all-or-nothing.

        Feature: ref-counted unpin.
        Description: r1 holds 4 blocks; r2 holds the first 2 of them.
        Expectation: releasing r1 leaves 2 (r2's shared blocks); releasing r2 → 0.
        """
        prompt = list(range(4 * BLOCK_SIZE))
        collector, balancer, ds = self._start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            balancer.callbacks["on_acquire"][0]("r2", "s0", prompt[: 2 * BLOCK_SIZE])
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 4

            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 2

            balancer.callbacks["on_release"][0]("s0", "r2")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 0
        finally:
            collector.stop()

    def test_cache_hit_blocks_are_held_even_though_they_book_no_tokens(self):
        """The v1 blind spot: a fully-cached prompt books 0 tokens but pins its blocks.

        Feature: ``INFLIGHT_BLOCKS`` counts held blocks regardless of hit/miss.
        Description: s0 already caches the whole 2-block prompt; the dispatch
          allocates nothing (``INFLIGHT_TOKENS`` += 0) but does pin 2 blocks.
        Expectation: blocks = 2, tokens = 0; release returns both to 0.
        """
        from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes

        prompt = list(range(2 * BLOCK_SIZE))
        collector, balancer, ds = self._start()
        try:
            ds.add_kv_blocks("s0", resolve_prefix_hashes(prompt, None, ds))

            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 2

            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 0
        finally:
            collector.stop()

    def test_no_prompt_ids_pins_nothing_and_is_tallied(self):
        """No prompt / unknown block size → nothing pinned, and the blind spot is counted.

        Feature: an unaccountable dispatch must not silently vanish from the
          held-block account.
        Description: r1 dispatches with ``prompt_ids=None`` on a replica whose
          block size is unknown (no KV event yet).
        Expectation: ``INFLIGHT_BLOCKS`` = 0, ``INFLIGHT_TOKENS`` falls back to the
          raw prompt length (conservative), ``_unaccounted_dispatches`` = 1; the
          release is a clean no-op.
        """
        prompt = list(range(4 * BLOCK_SIZE))
        collector, balancer, ds = self._start(block_size=BLOCK_SIZE)
        # Forget the learned block size: the chain is unresolvable from now on.
        ds._kv.block_size = None
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 0
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 4 * BLOCK_SIZE
            assert collector._unaccounted_dispatches == 1

            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 0
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0  # balanced
        finally:
            collector.stop()

    def test_release_without_block_list_warns_and_keeps_the_conservative_gauge(self, monkeypatch):
        """An evicted per-request row must warn, not crash or invent a subtraction.

        Feature: a release that cannot recover the pinned block list leaks
          (errs high) and says so.
        Description: acquire normally, then evict the per-request hash memo —
          simulating ``PerRequestStore`` LRU eviction before the release.
        Expectation: a WARNING names the request; the held gauge stays at 4 (the
          leak direction) and the release still completes.
        """
        from uni_agent.agent_aware_router.collectors import collector as collector_module

        prompt = list(range(4 * BLOCK_SIZE))
        collector, balancer, ds = self._start()
        # The router's logging namespace is mounted with propagate=False, so the
        # module logger is captured directly rather than through caplog.
        warnings: list[str] = []
        monkeypatch.setattr(collector_module.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 4
            ds.del_per_request("r1", "prefix_hashes")  # evicted before the release

            balancer.callbacks["on_release"][0]("s0", "r1")

            assert any("in-flight block set may leak" in w and "r1" in w for w in warnings), warnings
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 4  # conservative
            assert ds.get_metric("s0", MetricKey.COMPLETED_COUNT) == 1
        finally:
            collector.stop()

    def test_reacquire_without_release_does_not_double_count(self, monkeypatch):
        """A dropped release must not double the held set for the next turn.

        Feature: re-acquire detects the stale pin marker, unpins first, and nets
          out the still-booked tokens.
        Description: r1 acquires the same prompt twice, with no release between
          (a lost COMPLETED_COUNT), then releases once.
        Expectation: a WARNING is logged; ``INFLIGHT_BLOCKS`` = 4 (not 8) and
          ``INFLIGHT_TOKENS`` returns to 0 after the single release.
        """
        from uni_agent.agent_aware_router.collectors import collector as collector_module

        prompt = list(range(4 * BLOCK_SIZE))
        collector, balancer, ds = self._start()
        warnings: list[str] = []
        monkeypatch.setattr(collector_module.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)

            assert any("re-acquire without release" in w for w in warnings), warnings
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 4
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 4 * BLOCK_SIZE

            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 0
            assert ds.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0
        finally:
            collector.stop()

    def test_avail_holds_steady_across_a_shared_prefix_dispatch(self):
        """End-to-end: collector pins → ``INFLIGHT_BLOCKS`` → the strategy's ``avail``.

        Feature: the free capacity the capacity strategy ranks with is the held
          block count, so a shared-prefix rollout neither fills the replica twice
          nor frees it early.
        Description: s0's pool is 10 blocks (cap 160 tokens, gate 16 tokens) and
          it holds 9 (avail 16 — right at the gate); s1 is pinned full (avail 0 →
          filtered). r1 and r2 are dispatched the same 9-block prompt on s0.
        Expectation: the gauge stays 9 after r2's acquire (nothing new is
          allocated) and after r1's release (r2 still holds all 9); routing keeps
          picking s0, which it could not do if either step moved the gauge.
        """
        from uni_agent.agent_aware_router.strategies.base import ReplicaInfo
        from uni_agent.agent_aware_router.strategies.kvc_aware import KVCacheAwareStrategy
        from uni_agent.agent_aware_router.types import SlowCut

        prompt = list(range(9 * BLOCK_SIZE))
        collector, balancer, ds = self._start()
        ds.refresh_metrics(
            {
                "s0": {MetricKey.NUM_GPU_BLOCKS: 10},
                "s1": {MetricKey.NUM_GPU_BLOCKS: 10, MetricKey.INFLIGHT_BLOCKS: 10},
            }
        )
        strat = KVCacheAwareStrategy(
            alpha=0.7,
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            load_weights=(0.4, 0.2, 0.1, 0.3),
            slow_cut=SlowCut.CAPACITY_TOKEN_AWARE,
            do_shortcut=False,
            tie_tolerance=0.0,
        )
        strat.set_capacity(64, 1024)
        replicas = [ReplicaInfo(replica_id="s0"), ReplicaInfo(replica_id="s1")]
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 9
            assert route(strat, prompt, ds, replicas, "r1")[0] == "s0"

            balancer.callbacks["on_acquire"][0]("r2", "s0", prompt)
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 9  # dedup, not 18
            assert route(strat, prompt, ds, replicas, "r2")[0] == "s0"

            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 9  # r2 still holds
            assert route(strat, prompt, ds, replicas, "r1")[0] == "s0"

            balancer.callbacks["on_release"][0]("s0", "r2")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 0
            assert route(strat, prompt, ds, replicas, "r2")[0] == "s0"
        finally:
            collector.stop()

    def test_pins_are_isolated_per_replica(self):
        """The gauge is per replica: dispatching to s1 must not touch s0's pins."""
        prompt = list(range(2 * BLOCK_SIZE))
        collector, balancer, ds = self._start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", prompt)
            balancer.callbacks["on_acquire"][0]("r2", "s1", prompt)

            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 2
            assert ds.get_metric("s1", MetricKey.INFLIGHT_BLOCKS) == 2

            balancer.callbacks["on_release"][0]("s0", "r1")
            assert ds.get_metric("s0", MetricKey.INFLIGHT_BLOCKS) == 0
            assert ds.get_metric("s1", MetricKey.INFLIGHT_BLOCKS) == 2
        finally:
            collector.stop()


class _RecordingRLInsight:
    """Records metric_* calls (mirrors the rl_insight high-level API)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def init(self, *args, **kwargs) -> None:  # noqa: D401 - test double
        """No-op stand-in for rl_insight.init."""

    def metric_count(self, name, amount, documentation="", **labels):
        self.calls.append(("counter", name, amount, dict(labels)))

    def metric_gauge(self, name, value, documentation="", **labels):
        self.calls.append(("gauge", name, value, dict(labels)))

    def metric_histogram(self, name, value, documentation="", *, buckets=None, **labels):
        self.calls.append(("histogram", name, value, dict(labels)))


class TestCollectorEmitsToInsight:
    """B-class end-to-end: with rl-insight emit ON, collector writes forward to rl_insight.

    Drives the real ``Collector(CallbackTransport, InflightParser)`` through
    acquire/release and feeds poll/kv updates directly, asserting the recording
    rl_insight double receives exactly the B-class primitives the emitter maps.
    """

    @pytest.fixture(autouse=True)
    def _enable_emit(self, monkeypatch):
        from uni_agent.agent_aware_router.insight.emitter import emitter
        from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
        from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore

        PerReplicaStore._instance = None
        PerRequestStore._instance = None
        import sys

        from uni_agent import rl_insight as facade

        monkeypatch.setenv(facade.ENABLE_ENV, "1")
        monkeypatch.setattr(facade, "_enabled", True)  # pre-flip gate cache
        self._emitter = emitter
        self._rl = _RecordingRLInsight()
        monkeypatch.setitem(sys.modules, "rl_insight", self._rl)
        yield
        PerReplicaStore._instance = None
        PerRequestStore._instance = None

    def _by_name(self):
        return {c[1]: c for c in self._rl.calls}

    def test_acquire_forwards_dispatched_promptlen_tokens_avgturn(self):
        from uni_agent.agent_aware_router.collectors.collector import Collector

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightParser())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", [1, 2, 3])  # plen 3, turn 1
        finally:
            collector.stop()

        by_name = self._by_name()
        assert by_name[MetricKey.DISPATCHED_COUNT] == ("counter", MetricKey.DISPATCHED_COUNT, 1, {"replica": "s0"})
        assert by_name[MetricKey.PROMPT_LEN_SUM] == ("counter", MetricKey.PROMPT_LEN_SUM, 3, {"replica": "s0"})
        assert by_name[MetricKey.INFLIGHT_TOKENS] == ("gauge", MetricKey.INFLIGHT_TOKENS, 3, {"replica": "s0"})
        # inflight_avg_turn = in-flight turn sum (1) / in-flight count (1) = 1.0
        assert by_name["inflight_avg_turn"] == ("gauge", "inflight_avg_turn", 1.0, {"replica": "s0"})

    def test_release_forwards_completed_tokens_and_drives_avgturn_to_zero(self):
        from uni_agent.agent_aware_router.collectors.collector import Collector

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightParser())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0", [1, 2, 3])  # plen 3, turn 1
            balancer.callbacks["on_release"][0]("s0", "r1")  # release r1 (turn + tokens subtracted)
        finally:
            collector.stop()

        by_name = self._by_name()
        # release path: completed +1, tokens back to 0, avg turn 0 (idle: count 0)
        assert by_name[MetricKey.COMPLETED_COUNT] == ("counter", MetricKey.COMPLETED_COUNT, 1, {"replica": "s0"})
        assert by_name[MetricKey.INFLIGHT_TOKENS] == ("gauge", MetricKey.INFLIGHT_TOKENS, 0, {"replica": "s0"})
        assert by_name["inflight_avg_turn"] == ("gauge", "inflight_avg_turn", 0.0, {"replica": "s0"})

    def test_poll_forwards_levels_cumulatives_and_load(self):
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.collectors.parse import MetricsUpdate

        collector = Collector(CallbackTransport(_FakeBalancer()), InflightParser())
        collector._write_metrics_update(
            MetricsUpdate(
                node_id="s0",
                metrics={
                    MetricKey.KV_CACHE_USAGE_PERC: 0.5,
                    MetricKey.NUM_REQUESTS_RUNNING: 2,
                    MetricKey.NUM_REQUESTS_WAITING: 1,
                    MetricKey.PROMPT_TOKENS: 1000,
                    MetricKey.PROMPT_TOKENS_CACHED: 200,
                    MetricKey.EXTERNAL_PREFIX_CACHE_HITS: 50,
                    MetricKey.ESTIMATED_FLOPS_PER_GPU: 999,
                },
                is_delta=False,
            )
        )

        emitted = {c[1] for c in self._rl.calls}
        assert emitted == {
            MetricKey.KV_CACHE_USAGE_PERC,
            MetricKey.NUM_REQUESTS_RUNNING,
            MetricKey.NUM_REQUESTS_WAITING,
            MetricKey.PROMPT_TOKENS,
            MetricKey.PROMPT_TOKENS_CACHED,
            MetricKey.EXTERNAL_PREFIX_CACHE_HITS,
            MetricKey.ESTIMATED_FLOPS_PER_GPU,
            "kv_cache_load",
        }
        assert all(c[0] == "gauge" for c in self._rl.calls)  # all 8 are gauges
        # kv_cache_load is 0.0 here (no retained blocks in the fresh store)
        assert self._by_name()["kv_cache_load"] == ("gauge", "kv_cache_load", 0.0, {"replica": "s0"})

    def test_kv_removed_forwards_evictions(self):
        from uni_agent.agent_aware_router.collectors.collector import Collector
        from uni_agent.agent_aware_router.collectors.parse import KVCacheUpdate
        from uni_agent.agent_aware_router.types import Layer

        collector = Collector(CallbackTransport(_FakeBalancer()), InflightParser())
        collector._write_kv_update(KVCacheUpdate(node_id="s0", remove_blocks={Layer.GPU: ["h1", "h2", "h3"]}))

        assert self._rl.calls == [("counter", "kv_evictions", 3, {"replica": "s0"})]
