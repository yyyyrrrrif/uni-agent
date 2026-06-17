"""Unified metrics store for reading/writing polling metric data."""

from __future__ import annotations

import threading
from typing import Any

from uni_agent.llm_router.metric_spec import METRIC_SPECS


class MetricsStore:
    """Unified metrics store: ``{node_id: {canonical_key: value}}``.

    - ``get(node_id, key)``  → single value; falls back to ``METRIC_SPECS[key]["default"]``;
                               raises ``KeyError`` if key is not a valid canonical key
    - ``get(node_id)``       → entire node dict (empty dict if unknown)
    - ``refresh(new_data)``  → batch update; only updates nodes present in ``new_data``,
                               existing nodes NOT in ``new_data`` are left untouched
    """

    _instance: MetricsStore | None = None

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def singleton(cls) -> MetricsStore:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, node_id: str, key: str | None = None) -> Any | dict[str, Any]:
        """Read metrics.

        ``get(node_id, key)``  → single value, falls back to spec default.
            Raises ``KeyError`` if ``key`` is not a valid canonical key
            (i.e. not present in ``METRIC_SPECS``).
        ``get(node_id)``       → entire node dict
        """
        if key is None:
            return dict(self._data.get(node_id, {}))

        if key not in METRIC_SPECS:
            raise KeyError(f"Unknown metric key '{key}'. Valid keys: {sorted(METRIC_SPECS.keys())}")
        node = self._data.get(node_id, {})
        if key in node:
            return node[key]
        return METRIC_SPECS[key]["default"]

    def refresh(self, new_data: dict[str, dict[str, Any]]) -> None:
        """Batch refresh from collectors.

        For each node in ``new_data``: merge with existing data
        (new values overwrite same keys).  Nodes NOT in ``new_data``
        are left untouched.
        """
        with self._lock:
            for node_id, metrics in new_data.items():
                existing = self._data.get(node_id, {})
                merged = dict(existing)
                merged.update(metrics)
                self._data[node_id] = merged

    def all_ids(self) -> list[str]:
        """Return all node IDs currently in the store."""
        with self._lock:
            return list(self._data.keys())
