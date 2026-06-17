"""Structured update commands — decoder output, applied by the Collector.

Each Update is a pure dataclass produced by a Decoder (``decode()``) and
consumed by ``Collector``, which dispatches by type to the right
``DataStore`` write path.  They live under ``collectors/`` (the layer that
both produces and consumes them), not ``store/`` — the store layer stays
free of any knowledge of concrete update types.
"""

from uni_agent.llm_router.collectors.updates.kv_update import KVCacheUpdate
from uni_agent.llm_router.collectors.updates.metrics_update import MetricsUpdate

__all__ = ["KVCacheUpdate", "MetricsUpdate"]
