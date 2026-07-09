"""Helpers for balancer unit tests.

Defines ``FakeDataStore`` (query stub), ``_FakeCollectorProvider`` (lifecycle
stub), and helper functions.  Patching is done by ``conftest.py`` via a
session-scoped autouse fixture (``_conditional_patch``) that only fires when
balancer ut tests are selected, so it never leaks to Ray workers in other test
directories.
"""

from __future__ import annotations

from omegaconf import OmegaConf


class FakeDataStore:
    """Stand-in for ``DataStore`` — answers the strategy query interface.

    In the new architecture routing reads metrics from ``DataStore`` (passed to
    ``route()`` as ``store``), not from the provider. Unit tests therefore
    inject a ``FakeDataStore`` as ``balancer._store`` (via ``_fake_init_provider``)
    so strategies read empty/default values without touching the real
    singleton-backed ``DataStore``. Per-replica metrics can be supplied at
    construction by tests that need non-empty data.
    """

    def __init__(self, metrics: dict | None = None):
        self._metrics = metrics or {}

    def get_metric(self, replica_id, key):
        return self._metrics.get(replica_id, {}).get(key, 0.0)

    def get_metrics(self, replica_id):
        return dict(self._metrics.get(replica_id, {}))

    def get_gpu_prefix_hit_rate(self, prompt_ids):
        return {}

    def get_tier_prefix_hit_rate(self, replica_id, prompt_ids, tier):
        return 0.0

    def get_retained_occupancy(self, replica_id):
        return None

    def remove_servers(self, server_ids):
        """Stand-in for ``DataStore.remove_servers`` — drop the replicas' metrics."""
        for sid in server_ids:
            self._metrics.pop(sid, None)


class _FakeCollectorProvider:
    """Stand-in for ``CollectorProvider`` — a pure lifecycle stub.

    The provider only constructs and starts/stops collectors; routing reads
    metrics from ``DataStore`` (here ``FakeDataStore``), not from the provider.
    So this fake just records that ``start()`` ran so ``test_b03`` /
    ``get_status`` can assert the lifecycle was driven. ``add_servers`` /
    ``remove_servers`` record their args so dynamic-endpoint tests can assert
    the provider was wired into ``add_servers``/``remove_servers``.
    """

    def __init__(self, collectors_config, collection_names, server_addresses=None, kv_event_endpoints=None):
        self.collectors_config = collectors_config
        self.collection_names = collection_names
        self.server_addresses = server_addresses
        self.kv_event_endpoints = kv_event_endpoints
        self.started = False
        self.stopped = False
        self.added_servers: list[tuple[dict, dict]] = []
        self.removed_servers: list[list[str]] = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def add_servers(self, server_addresses, kv_event_endpoints):
        self.added_servers.append((dict(server_addresses), dict(kv_event_endpoints)))

    def remove_servers(self, server_ids):
        self.removed_servers.append(list(server_ids))


def _router_config(weight: float = 1.0):
    """Build a minimal router_config (OmegaConf) the Balancer accepts."""
    return OmegaConf.create(
        {
            "router_class": "uni_agent.llm_router.balancer.KVCAwareBalancer",
            "strategies": [
                {
                    "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                    "weight": weight,
                    "collector_names": ["vllm_zmq"],
                },
            ],
        }
    )


def _fake_init_provider(self):
    """Replacement for KVCAwareBalancer._init_provider in unit tests.

    Injects a lifecycle-only ``_FakeCollectorProvider`` (so construction is
    observable) AND a ``FakeDataStore`` as ``self._store``, overriding the real
    ``DataStore()`` the Balancer constructed in ``__init__``. Strategies read
    from ``self._store`` via ``route()``, so they see the fake, not the real
    singleton-backed store — keeping unit tests hermetic from collector data.
    """
    collection_names = sorted({name for cfg in self._config.strategies for name in cfg.collector_names})
    self._provider = _FakeCollectorProvider(
        self._config.collector,
        collection_names,
    )
    self._provider.start()
    self._store = FakeDataStore()


def _fake_resolve_max_num_seqs(servers):
    """Return a fixed capacity — unit tests have string handles, not Ray actors."""
    return 16


def _make_balancer(servers=None):
    """Build a balancer over the given servers (default two)."""
    from uni_agent.llm_router.balancer import KVCAwareBalancer

    if servers is None:
        servers = {"s0": "h0", "s1": "h1"}
    return KVCAwareBalancer(servers, _router_config())
