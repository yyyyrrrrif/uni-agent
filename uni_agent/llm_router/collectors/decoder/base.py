"""Decoder — abstract base for data decoding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Decoder(ABC):
    """Abstract base for data decoders.

    Subclasses implement ``decode()`` with their backend-specific
    parsing logic.
    """

    @abstractmethod
    def decode(self, raw_data: bytes | str, node_id: str) -> Any:
        """Decode raw data and return structured result.

        Args:
            raw_data: Raw payload — ``bytes`` (from ZMQ) or ``str``
                (from HTTP response text).
            node_id: Source endpoint/node identifier.

        Returns:
            Structured update object (e.g., KVCacheUpdate, MetricsUpdate).
            Returns None if decode fails.
        """
