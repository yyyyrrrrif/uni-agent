"""Transport — abstract base for data transport layers.

A Transport fetches raw data from a network source (ZMQ, HTTP, etc.)
and delivers it to a handler callback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


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
