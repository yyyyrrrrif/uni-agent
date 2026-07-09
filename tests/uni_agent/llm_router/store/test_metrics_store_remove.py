"""Unit tests for MetricsStore.remove + DataStore.remove_servers (elastic removal).

These cover the store-side of ``KVCAwareBalancer.remove_servers`` without a
GPU/vLLM dependency. The collector/transport add/remove behaviour is covered
by ``collectors/test_dynamic_add_remove_servers.py`` (real vLLM, GPU).
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router.metric_spec import MetricKey
from uni_agent.llm_router.store.data_store import DataStore
from uni_agent.llm_router.store.metrics_store import MetricsStore

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestMetricsStoreRemove:
    def test_remove_drops_node(self):
        """remove(node_id) drops the node's metric dict."""
        store = MetricsStore.singleton()
        store.refresh({"n1": {MetricKey.KV_CACHE_USAGE_PERC: 42.0}})
        assert MetricKey.KV_CACHE_USAGE_PERC in store.get("n1")
        store.remove("n1")
        assert store.get("n1") == {}

    def test_remove_unknown_id_is_noop(self):
        """remove on a node that's absent is a no-op (no KeyError)."""
        store = MetricsStore.singleton()
        store.remove("does-not-exist")  # must not raise
        assert "does-not-exist" not in store.all_ids()

    def test_remove_does_not_touch_other_nodes(self):
        """remove(n1) leaves n2's data intact."""
        store = MetricsStore.singleton()
        store.refresh(
            {
                "n1": {MetricKey.NUM_REQUESTS_RUNNING: 5},
                "n2": {MetricKey.NUM_REQUESTS_RUNNING: 7},
            }
        )
        store.remove("n1")
        assert "n2" in store.all_ids()
        assert store.get("n2", MetricKey.NUM_REQUESTS_RUNNING) == 7


class TestDataStoreRemoveServers:
    def test_remove_servers_clears_metrics_and_kv(self):
        """remove_servers drops the replica's metrics + kv blocks."""
        store = DataStore()
        store.refresh_metrics({"n1": {MetricKey.KV_CACHE_USAGE_PERC: 0.5}})
        store.add_kv_blocks("n1", ["h1", "h2"])
        assert store.kv_node_has_blocks("n1")
        assert "n1" in store.get_metric_node_ids()

        store.remove_servers(["n1"])

        assert "n1" not in store.get_metric_node_ids()
        assert not store.kv_node_has_blocks("n1")

    def test_remove_servers_leaves_other_replicas(self):
        """Removing n1 leaves n2's metrics + kv blocks intact."""
        store = DataStore()
        store.refresh_metrics(
            {
                "n1": {MetricKey.KV_CACHE_USAGE_PERC: 0.5},
                "n2": {MetricKey.NUM_REQUESTS_RUNNING: 3},
            }
        )
        store.add_kv_blocks("n1", ["h1"])
        store.add_kv_blocks("n2", ["h2"])

        store.remove_servers(["n1"])

        assert "n2" in store.get_metric_node_ids()
        assert store.kv_node_has_blocks("n2")
        assert not store.kv_node_has_blocks("n1")

    def test_remove_servers_unknown_id_is_noop(self):
        """Removing an unknown id doesn't raise and leaves the store intact."""
        store = DataStore()
        store.refresh_metrics({"n1": {MetricKey.KV_CACHE_USAGE_PERC: 0.5}})
        store.remove_servers(["nope"])  # must not raise
        assert "n1" in store.get_metric_node_ids()

    def test_remove_servers_does_not_reset_block_size(self):
        """block_size is global — remove_servers must not clear it."""
        store = DataStore()
        store.set_block_size(16)
        store.remove_servers(["n1"])
        assert store.get_block_size() == 16
