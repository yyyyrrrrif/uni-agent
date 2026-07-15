"""StickySessionTable — request_id → replica_id LRU mapping for sticky routing.

Holds the sticky-session affinity table for the KVCAware router. The Balancer
owns one instance and threads it into ``route()`` → ``strategy.score()`` so the
strategy can short-circuit to a sticky replica when it is not overloaded; the
table is also written back (``put``) by the Balancer after each routing
decision so subsequent turns of the same ``request_id`` stay affinity-bound.

Design notes:
- **LRU eviction**: backed by ``cachetools.LRUCache`` (same dep verl's
  ``GlobalRequestLoadBalancer`` uses). Access (``get``/``__contains__``)
  refreshes recency, so a hot conversation is never evicted in favour of a
  cold one.
- **No locking**: the ``KVCAwareBalancer`` is a single Ray actor running
  ``acquire_server`` serially, so the table is accessed from one thread.
  Callers that share the table across threads must wrap it themselves.
- **Replica removal**: ``invalidate_replica`` bulk-clears every request_id
  bound to a removed replica, so stale stickiness never routes to a dead
  server. This is O(n) in table size; ``remove_servers`` is a rare, elastic
  event, so that cost is fine.

Reference: verl ``router.py`` ``DEFAULT_ROUTING_CACHE_SIZE = 10000``.
"""

from __future__ import annotations

from cachetools import LRUCache

from uni_agent.llm_router.logging import get_router_logger

logger = get_router_logger("sticky-session")

DEFAULT_STICKY_MAX_SIZE = 10000


class StickySessionTable:
    """LRU map of ``request_id → replica_id`` for sticky-session routing.

    Read by strategies during scoring (``get``); written by the Balancer after
    each routing decision (``put``); invalidated on server removal
    (``invalidate_replica``) or per-request expiry (``invalidate``).
    """

    def __init__(self, max_size: int = DEFAULT_STICKY_MAX_SIZE) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self._max_size = int(max_size)
        # request_id → (replica_id, turn, tokens). ``turn`` is the balancer's
        # per-request route count (kept for logging / comparison); ``tokens`` is
        # the request's accumulated prompt length (len(prompt_ids)), the direct
        # migration-cost signal used to pick the cheapest request to move off an
        # overloaded replica.
        self._table: LRUCache[str, tuple[str, int, int]] = LRUCache(maxsize=self._max_size)
        logger.info(f"StickySessionTable created: max_size={self._max_size}")

    @property
    def max_size(self) -> int:
        """Configured LRU capacity."""
        return self._max_size

    def get(self, request_id: str) -> str | None:
        """Return the bound replica_id, refreshing LRU recency on hit.

        ``None`` means no sticky binding (cold start, or evicted by LRU).
        Accessing the key on hit moves it to most-recently-used, mirroring
        verl ``GlobalRequestLoadBalancer``'s ``__contains__`` + ``__getitem__``
        pattern.
        """
        if request_id in self._table:
            return self._table[request_id][0]
        return None

    def get_turn(self, request_id: str) -> int | None:
        """Return the stored turn (route count) for a binding, or ``None``."""
        if request_id in self._table:
            return self._table[request_id][1]
        return None

    def get_tokens(self, request_id: str) -> int | None:
        """Return the stored accumulated token count for a binding, or ``None``."""
        if request_id in self._table:
            return self._table[request_id][2]
        return None

    def put(self, request_id: str, replica_id: str, turn: int = 1, tokens: int = 0) -> None:
        """Bind / refresh ``request_id → (replica_id, turn, tokens)``.

        Inserting an existing key refreshes recency and updates the bound
        replica (e.g. when overload-fallback or migration routed elsewhere), the
        turn, and the accumulated ``tokens``. ``turn``/``tokens`` default so
        legacy two/three-arg callers keep working.
        """
        self._table[request_id] = (replica_id, int(turn), int(tokens))

    def cheapest_on(self, replica_id: str) -> tuple[str, int, int] | None:
        """Return ``(request_id, turn, tokens)`` of the lowest-**tokens** binding there.

        Used by the balancer's migration gate to pick the cheapest request to
        move off an overloaded replica (fewest accumulated tokens = least prefix
        KV to re-prefill). ``turn`` is carried for logging/comparison only. Ties
        break on the first-seen request_id. Returns ``None`` if no binding points
        at ``replica_id``. O(n) scan — only called on the rare migration trigger.
        """
        best: tuple[str, int, int] | None = None
        for rid, (sid, turn, tokens) in self._table.items():
            if sid == replica_id and (best is None or tokens < best[2]):
                best = (rid, turn, tokens)
        return best

    def remove(self, request_id: str) -> None:
        """Drop a single request_id's binding (alias of :meth:`invalidate`)."""
        self._table.pop(request_id, None)

    def invalidate(self, request_id: str) -> None:
        """Drop a single request_id's binding (e.g. stale replica hit)."""
        self._table.pop(request_id, None)

    def invalidate_replica(self, replica_id: str) -> None:
        """Drop every binding pointing at a removed replica.

        Called from ``KVCAwareBalancer.remove_servers`` so a server going away
        doesn't leave sticky entries routing into the void. O(n) in table size
        — acceptable for the rare elastic-removal path.
        """
        stale = [rid for rid, (sid, _turn, _tokens) in self._table.items() if sid == replica_id]
        for rid in stale:
            self._table.pop(rid, None)
        if stale:
            logger.info(
                f"invalidate_replica: replica={replica_id} cleared {len(stale)} bindings",
            )

    def __len__(self) -> int:
        return len(self._table)

    def status(self) -> dict:
        """Return a debugging snapshot of the table state."""
        return {
            "max_size": self._max_size,
            "size": len(self._table),
        }
