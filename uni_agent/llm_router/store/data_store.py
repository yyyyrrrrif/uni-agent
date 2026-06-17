"""StoreProvider — unified data access layer for all stores."""

from __future__ import annotations

from typing import Any

from uni_agent.llm_router.store.kv_cache_store import KVCacheStore
from uni_agent.llm_router.store.metrics_store import MetricsStore


class DataStore:
    """Unified data access layer — single entry point for all store operations.

    Wraps the singleton stores and exposes a unified interface for all
    reads and writes.  Stateless — instantiate once and reuse.

    Usage:
        provider = StoreProvider()
        # Write metrics
        provider.refresh_metrics({'node1': {'kv_cache_usage_perc': 45.0}})
        # Read metrics
        value = provider.get_metric('node1', 'kv_cache_usage_perc')
        # Write KV cache blocks
        provider.add_kv_blocks('node1', ['hash1', 'hash2'])
    """

    def __init__(self) -> None:
        self._metrics = MetricsStore.singleton()
        self._kv = KVCacheStore.singleton()

    # ── MetricsStore operations ─────────────────────────────────────────

    def get_metric(self, node_id: str, key: str) -> Any:
        """Query a single metric by canonical key.

        Args:
            node_id: Target node.
            key: Canonical metric key (e.g., ``MetricKey.KV_CACHE_USAGE_PERC``).

        Returns:
            Metric value; falls back to ``METRIC_SPECS`` default if absent.

        Raises:
            KeyError: If key is not a valid canonical key.
        """
        return self._metrics.get(node_id, key)

    def get_metrics(self, node_id: str) -> dict[str, Any]:
        """Get a node's full metrics snapshot.

        Args:
            node_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node is absent.
        """
        return self._metrics.get(node_id)

    def get_metric_node_ids(self) -> list[str]:
        """Return all node IDs that have metrics in the store."""
        return self._metrics.all_ids()

    def refresh_metrics(self, new_data: dict[str, dict[str, Any]]) -> None:
        """Batch refresh metrics from collectors.

        For each node in ``new_data``: merge with existing data
        (new values overwrite same keys).  Nodes NOT in ``new_data``
        are left untouched.

        Args:
            new_data: Dict of {node_id: {canonical_key: value}}.
        """
        self._metrics.refresh(new_data)

    # ── KVCacheStore operations ─────────────────────────────────────────

    def get_block_size(self) -> int | None:
        """Get learned block size.

        Returns:
            Block size in tokens, or None if not yet learned.
        """
        return self._kv.block_size

    def set_block_size(self, size: int) -> None:
        """Set block size (learned from first BlockStored event).

        Args:
            size: Block size in tokens.
        """
        if self._kv.block_size is None:
            self._kv.block_size = size

    def add_kv_blocks(self, node_id: str, block_hashes: list[str]) -> None:
        """Add KV cache blocks to a node.

        Args:
            node_id: Target node.
            block_hashes: List of local prefix hashes to add.
        """
        self._kv.add_blocks(node_id, block_hashes)

    def remove_kv_blocks(self, node_id: str, block_hashes: list[str]) -> None:
        """Remove KV cache blocks from a node.

        Args:
            node_id: Target node.
            block_hashes: List of local prefix hashes to remove.
        """
        self._kv.remove_blocks(node_id, block_hashes)

    def clear_kv_node(self, node_id: str) -> None:
        """Clear all KV cache blocks for a node.

        Args:
            node_id: Target node.
        """
        self._kv.clear_replica(node_id)

    def get_kv_block_count(self) -> int:
        """Return the number of unique block hashes currently cached."""
        return len(self._kv.replicas_by_block)

    def kv_node_has_blocks(self, node_id: str) -> bool:
        """Return True if node_id appears in at least one cached block."""
        return any(node_id in replicas for replicas in self._kv.replicas_by_block.values())

    def has_kv_block(self, block_hash: str) -> bool:
        """Return True if block_hash is present in the cache index."""
        return block_hash in self._kv.replicas_by_block

    # ── KV cache prefix hit rate queries ────────────────────────────────

    def get_gpu_prefix_hit_rate(self, prompt_ids: list[int]) -> dict[str, int]:
        """Match prefix hashes against cached blocks, return per-node hit percent.

        Args:
            prompt_ids: Current request's prompt token IDs.

        Returns:
            Dict of node_id → prefix_match_percent (0–100).
            Empty dict if block_size is unknown or no full blocks.
        """
        return self._kv.get_gpu_prefix_hit_rate(prompt_ids)

    def get_tier_prefix_hit_rate(
        self,
        node_id: str,
        prompt_ids: list[int],
        tier: str,
    ) -> float:
        """Query tier-level prefix cache hit rate.

        Args:
            node_id: Target node.
            prompt_ids: Current request's prompt token IDs.
            tier: ``"cpu"`` or ``"ssd"``.

        Returns:
            Hit rate 0.0–1.0.
        """
        return self._kv.get_tier_prefix_hit_rate(node_id, prompt_ids, tier)
