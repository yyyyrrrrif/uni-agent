"""Transport — abstract base for data transport layers.

A Transport fetches raw data from a network source (ZMQ, HTTP, etc.)
and delivers it to a handler callback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Transport(ABC):
    """Abstract base for data transport layers.

    Subclasses implement ``subscribe()`` with their protocol-specific
    connection and data-fetch logic.  ``stop()`` cancels connections
    and blocks until cleanup is complete.
    """

    @abstractmethod
    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Start data acquisition and deliver each item to handler.

        Args:
            handler: Callback that receives (raw_data, node_id).
                raw_data is ``bytes`` (ZMQ) or ``str`` (HTTP response text).
                node_id identifies the source endpoint/node.
        """

    @abstractmethod
    def stop(self) -> None:
        """Signal stop and close protocol-level resources (sockets/clients).

        Implementations should only:
          1. set a stop flag so subscribe loops exit,
          2. close sockets / contexts / http clients (idempotently).
        """

    # ── Dynamic endpoint management ─────────────────────────────────────

    def add_endpoint(self, node_id: str, endpoint: Any) -> None:
        """Register a new endpoint and start collecting from it.

        Called by ``Collector.add_endpoint`` when a server is added to the
        pool. The endpoint type is transport-specific (``str`` address for
        HTTP, ``list[str]`` for ZMQ) and must match what ``__init__`` accepts.

        Default raises ``NotImplementedError``; subclasses override with
        their protocol-specific connect + task-spawn logic.
        """
        raise NotImplementedError

    def remove_endpoint(self, node_id: str) -> None:
        """Stop collecting from an endpoint and release its resources.

        Called by ``Collector.remove_endpoint`` when a server leaves the
        pool. Synchronously cancels the endpoint's task and closes its
        protocol resources so a subsequent ``stop()`` doesn't touch it.

        Default raises ``NotImplementedError``; subclasses override.
        """
        raise NotImplementedError
