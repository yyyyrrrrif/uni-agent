"""VLLMMetricsDecoder — vLLM Prometheus metrics decoder.

Parses Prometheus exposition-format text and returns structured metrics.
Store writes are handled by Collector via DataStore.
"""

from __future__ import annotations

import logging
from typing import Any

from uni_agent.llm_router.collectors.decoder.base import Decoder
from uni_agent.llm_router.collectors.updates import MetricsUpdate
from uni_agent.llm_router.metric_spec import METRIC_SPECS, MetricKey

logger = logging.getLogger(__name__)


class VLLMMetricsDecoder(Decoder):
    """vLLM Prometheus metrics decoder — parses HTTP response text.

    Returns structured metrics.

    vLLM Prometheus raw name → canonical key mapping:
        ``vllm:kv_cache_usage_perc``  → ``KV_CACHE_USAGE_PERC``
        ``vllm:num_requests_running`` → ``NUM_REQUESTS_RUNNING``
        ``vllm:num_requests_waiting`` → ``NUM_REQUESTS_WAITING``
    """

    _PROMETHEUS_MAP: dict[str, str] = {
        "vllm:kv_cache_usage_perc": MetricKey.KV_CACHE_USAGE_PERC,
        "vllm:num_requests_running": MetricKey.NUM_REQUESTS_RUNNING,
        "vllm:num_requests_waiting": MetricKey.NUM_REQUESTS_WAITING,
    }

    def decode(self, raw_data: bytes | str, node_id: str) -> MetricsUpdate | None:
        """Parse Prometheus text and return structured metrics.

        Args:
            raw_data: HTTP response text (Prometheus exposition format).
            node_id: Source endpoint identifier.

        Returns:
            MetricsUpdate with parsed metrics, or None if decode failed.
        """
        # HTTP delivers str; ignore bytes data
        if isinstance(raw_data, bytes):
            logger.debug("VLLMMetricsDecoder received bytes data, expected str — skipping")
            return None

        result: dict[str, Any] = {}
        for line in raw_data.splitlines():
            if line.startswith("#") or not line.strip():
                continue

            try:
                raw_name = line.split("{")[0] if "{" in line else line.split()[0]
                value = float(line.split()[-1])
            except (ValueError, IndexError):
                logger.warning(f"decode failed: {line}")
                continue

            canonical = self._PROMETHEUS_MAP.get(raw_name)
            if not canonical:
                continue
            value_type = METRIC_SPECS[canonical].get("value_type", float)
            result[canonical] = value_type(value)

        return MetricsUpdate(node_id=node_id, metrics=result)
