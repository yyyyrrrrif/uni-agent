"""CollectorProvider — lifecycle manager for data collectors.

Strategies no longer query metrics through the provider — they read from the
unified ``DataStore`` (which wraps the singleton ``MetricsStore`` /
``KVCacheStore``). The provider now owns only collector construction and
lifecycle (start/stop) and dynamic endpoint add/remove; metric-query proxies
that used to live here have moved to ``DataStore`` (see
``DataStore.get_retained_occupancy``).
"""

from __future__ import annotations

from uni_agent.llm_router.collectors.collector import Collector, get_collector
from uni_agent.llm_router.config.collector import CollectorConfig

# Imported lazily-typed (avoid circular import at module load): the concrete
# transports are referenced only for isinstance dispatch in add_servers, not
# for construction.
from uni_agent.llm_router.collectors.transport.http import HTTPTransport
from uni_agent.llm_router.collectors.transport.zmq import ZMQTransport


class CollectorProvider:
    """Lifecycle manager for data collectors.

    Args:
        collectors_config: ``CollectorConfig`` — connection tuning parameters.
        collection_names: List of collection names to initialize (e.g.
            ``["vllm_metrics", "vllm_zmq"]``).
        server_addresses: ``{node_id: ip:port}`` for HTTP transport.
        kv_event_endpoints: ``{node_id: [sub_addr, replay_addr]}`` for ZMQ transport.
    """

    def __init__(
        self,
        collectors_config: CollectorConfig,
        collection_names: list[str],
        server_addresses: dict[str, str] | None = None,
        kv_event_endpoints: dict[str, list[str]] | None = None,
    ) -> None:
        self._collectors: list[Collector] = [
            get_collector(
                name,
                collectors_config,
                server_addresses=server_addresses,
                kv_event_endpoints=kv_event_endpoints,
            )
            for name in collection_names
        ]

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all collectors."""
        for collector in self._collectors:
            collector.start()

    def stop(self) -> None:
        """Stop all collectors."""
        for collector in self._collectors:
            collector.stop()

    # ── Dynamic endpoint management ─────────────────────────────────────

    def add_servers(
        self,
        server_addresses: dict[str, str],
        kv_event_endpoints: dict[str, list[str]],
    ) -> None:
        """Start collecting from newly-added servers.

        Each collector's transport is dispatched by type (no name→kind
        mapping table): HTTP transports get each server's ``ip:port`` from
        ``server_addresses``; ZMQ transports get the 4-element list from
        ``kv_event_endpoints``. A server absent from the dict its transport
        needs is skipped for that collector (e.g. an mc-off replica with no
        ZMQ endpoint — mirrors the init-time skip).
        """
        for collector in self._collectors:
            transport = collector._transport
            if isinstance(transport, HTTPTransport):
                for node_id, endpoint in server_addresses.items():
                    collector.add_endpoint(node_id, endpoint)
            elif isinstance(transport, ZMQTransport):
                for node_id, endpoint in kv_event_endpoints.items():
                    collector.add_endpoint(node_id, endpoint)

    def remove_servers(self, server_ids: list[str]) -> None:
        """Stop collecting from removed servers.

        Endpoint removal is keyed only by ``node_id`` (no type dispatch
        needed), so every collector is told to drop each id — HTTP pops its
        dict entry, ZMQ cancels its task + closes sockets. Collectors that
        never had the id no-op.
        """
        for collector in self._collectors:
            for sid in server_ids:
                collector.remove_endpoint(sid)
