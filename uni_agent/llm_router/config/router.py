"""Top-level KVCAwareConfig and parsing logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from uni_agent.llm_router.config.base import (
    ConfigError,
    StrategyConfig,
    _multiline_repr,
)
from uni_agent.llm_router.config.cache import CacheStoreConfig
from uni_agent.llm_router.config.collector import CollectorConfig

# ============================================================
# Top-level KVCAwareConfig
# ============================================================

_DEFAULT_COLLECTOR = CollectorConfig()
_DEFAULT_CACHE_STORE = CacheStoreConfig()

# Conditional-migration knobs forwarded verbatim from YAML into KVCAwareConfig.
_MIGRATE_KEYS = (
    "migrate_max_tokens",
    "migrate_max_turns",
    "imbalance_over_factor",
    "imbalance_min_idle",
    "imbalance_idle_load",
    "imbalance_gap",
)


@dataclass(repr=False)
class KVCAwareConfig:
    """Top-level config for KVCAwareBalancer, parsed from OmegaConf DictConfig.

    VeRL loads router YAML from ``router_config_path``, then passes the
    resulting DictConfig to the Balancer constructor.  The Balancer calls
    ``from_config(cfg)`` to obtain this fully-resolved dataclass instance.

    Attributes:
        strategies: Polymorphic strategy list (each with ``_target_``).
        collector: Collector module connection-type tuning config.
        cache_store: CacheStore configuration.
        sticky_max_size: Capacity of the request_id→replica sticky-session LRU
            table. Bound conversations stay affinity-bound to their replica
            until it is removed or evicted by LRU. Default 10000 (mirrors verl
            ``DEFAULT_ROUTING_CACHE_SIZE``).
        migrate_max_tokens: Conditional-migration gate (long-tail mitigation),
            default OFF (``0``). When > 0, on an overloaded + imbalanced pool the
            balancer migrates the *cheapest* request on the straggler replica —
            the one with the fewest accumulated tokens (``len(prompt_ids)``, i.e.
            least prefix KV to re-prefill), and only if that request's
            ``tokens <= migrate_max_tokens`` — to the lightest replica. Long
            (large-token) conversations are never moved: their 50-100K-token
            prefix costs more to rebuild than the rebalance saves (PR
            verl#6533's KV-aware finding).
        migrate_max_turns: Optional *secondary* gate on the same migration,
            default OFF (``0`` = not enforced). When > 0, the migrated request
            must additionally satisfy ``turn <= migrate_max_turns``. ``turn`` is
            always tracked and logged for comparison with the token signal even
            when this gate is off.
        imbalance_over_factor: (A) straggler if ``bound_load >= mean_load × this``.
        imbalance_min_idle: (B) need ``>=`` this many near-idle replicas (headroom).
        imbalance_idle_load: a replica with ``load <= this`` counts as near-idle.
        imbalance_gap: (C) require ``bound_load - min_load >= this``.
    """

    strategies: list[StrategyConfig]  # required, no default
    collector: CollectorConfig = field(default_factory=lambda: _DEFAULT_COLLECTOR)
    cache_store: CacheStoreConfig = field(default_factory=lambda: _DEFAULT_CACHE_STORE)
    sticky_max_size: int = 10000
    # ── Conditional migration / imbalance gate (default OFF) ──────────────
    migrate_max_tokens: int = 0  # primary cost gate + enable switch (0 = off)
    migrate_max_turns: int = 0  # optional secondary gate (0 = not enforced)
    imbalance_over_factor: float = 1.5
    imbalance_min_idle: int = 2
    imbalance_idle_load: float = 0.2
    imbalance_gap: float = 0.15

    @classmethod
    def from_config(cls, cfg: DictConfig | dict) -> KVCAwareConfig:
        """Two-step parsing of VeRL-transmitted config.

        Step 1: OmegaConf.merge for auto-recursive dataclass fields
                (collector, cache_store).

        Step 2: Manual traversal of strategies (list or dict from Hydra
                defaults composition) — instantiate each ``_target_`` entry.
        """
        if not isinstance(cfg, DictConfig | dict):
            raise ConfigError(f"cfg must be DictConfig or dict, got {type(cfg)}")

        cfg = OmegaConf.create(cfg)

        # ── Extract polymorphic sections before merge ──────────────
        # NOTE: VeRL passes a plain dict (from hydra.compose → to_container) that
        # also carries `router_class` (Balancer FQN for VeRL's importlib lookup).
        # `router_class` is VeRL-side metadata, NOT part of the config domain —
        # from_config only extracts strategies/collector/cache_store below, so
        # it is silently ignored. Top-level `_target_` is defensively popped too
        # (legacy/structured-input compatibility); the current YAML has none.
        strategies_raw = _extract_strategies(cfg)

        # ── Step 1: merge dataclass-typed fields (collector, cache_store) ──
        defaults = OmegaConf.create(
            {
                "collector": OmegaConf.structured(CollectorConfig),
                "cache_store": OmegaConf.structured(CacheStoreConfig),
            }
        )
        kwargs_for_merge = OmegaConf.create(cfg)
        # Remove polymorphic sections to avoid ReadonlyConfigError
        for key in ("strategies", "_target_"):
            if key in kwargs_for_merge:
                OmegaConf.set_struct(kwargs_for_merge, False)
                kwargs_for_merge.pop(key, None)
                OmegaConf.set_struct(kwargs_for_merge, True)

        # Validate non-dict types for collector/cache_store
        if (
            "collector" in kwargs_for_merge
            and kwargs_for_merge.collector is not None
            and not isinstance(kwargs_for_merge.collector, dict | DictConfig)
        ):
            raise ConfigError(f"collector must be a dict, got {type(kwargs_for_merge.collector).__name__}")
        if (
            "cache_store" in kwargs_for_merge
            and kwargs_for_merge.cache_store is not None
            and not isinstance(kwargs_for_merge.cache_store, dict | DictConfig)
        ):
            raise ConfigError(f"cache_store must be a dict, got {type(kwargs_for_merge.cache_store).__name__}")

        merged = OmegaConf.merge(defaults, kwargs_for_merge)
        config_obj = OmegaConf.to_object(merged)

        # Extract resolved dataclass fields
        if isinstance(config_obj, dict):
            collector_cfg = config_obj.get("collector") or CollectorConfig()
            cache_store_cfg = config_obj.get("cache_store") or CacheStoreConfig()
            sticky_max_size = config_obj.get("sticky_max_size", 10000)
            migrate_kwargs = {
                k: config_obj[k]
                for k in _MIGRATE_KEYS
                if k in config_obj and config_obj[k] is not None
            }
        else:
            collector_cfg = getattr(config_obj, "collector", None) or CollectorConfig()
            cache_store_cfg = getattr(config_obj, "cache_store", None) or CacheStoreConfig()
            sticky_max_size = getattr(config_obj, "sticky_max_size", 10000)
            migrate_kwargs = {
                k: getattr(config_obj, k)
                for k in _MIGRATE_KEYS
                if getattr(config_obj, k, None) is not None
            }

        # ── Step 2: parse strategies (polymorphic list) ────────────
        if strategies_raw is None:
            raise ConfigError("strategies is required — must be explicitly configured")
        strategies = _parse_polymorphic_list(strategies_raw, StrategyConfig, "strategies")

        # ── Validate and construct ─────────────────────────────────
        result = cls(
            strategies=strategies,
            collector=collector_cfg,
            cache_store=cache_store_cfg,
            sticky_max_size=int(sticky_max_size),
            **migrate_kwargs,
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Validate the full config. Raises ConfigError with all violations."""
        errors: list[str] = []

        if not self.strategies:
            errors.append("strategies must be non-empty")
        elif not isinstance(self.strategies, list):
            errors.append("strategies must be a list")
        else:
            total_weight = sum(s.weight for s in self.strategies)
            if not (0.9 <= total_weight <= 1.1):
                errors.append(f"sum of strategy weights must be ~1.0, got {total_weight}")

        if self.sticky_max_size <= 0:
            errors.append(f"sticky_max_size must be > 0, got {self.sticky_max_size}")

        # Migration knobs (only active when migrate_max_tokens > 0).
        if self.migrate_max_tokens < 0:
            errors.append(f"migrate_max_tokens must be >= 0, got {self.migrate_max_tokens}")
        if self.migrate_max_turns < 0:
            errors.append(f"migrate_max_turns must be >= 0, got {self.migrate_max_turns}")
        if self.imbalance_over_factor < 1.0:
            errors.append(f"imbalance_over_factor must be >= 1.0, got {self.imbalance_over_factor}")
        if self.imbalance_min_idle < 1:
            errors.append(f"imbalance_min_idle must be >= 1, got {self.imbalance_min_idle}")
        if not 0 <= self.imbalance_idle_load <= 1:
            errors.append(f"imbalance_idle_load must be in [0, 1], got {self.imbalance_idle_load}")
        if self.imbalance_gap < 0:
            errors.append(f"imbalance_gap must be >= 0, got {self.imbalance_gap}")

        if errors:
            raise ConfigError("; ".join(errors))

    def __repr__(self) -> str:
        """Multi-line indented repr (delegates to the shared _multiline_repr)."""
        return _multiline_repr(self)


# ============================================================
# Helper functions
# ============================================================


def _extract_strategies(cfg: DictConfig) -> list[Any] | None:
    """Extract strategies from cfg, handling both list and dict formats.

    Hydra defaults composition produces a dict (keyed by strategy name),
    but direct YAML can be a list. Both are supported.
    """
    if "strategies" not in cfg:
        return None
    val = cfg["strategies"]
    if val is None:
        return None
    if isinstance(val, list | ListConfig):
        return list(val)
    if isinstance(val, dict | DictConfig):
        # Hydra defaults composition → dict of {name: strategy_cfg}
        return list(val.values())
    # Not a list or dict — will be caught by validation
    return val


def _parse_polymorphic_list(
    items: list[Any],
    base_class: type,
    list_name: str,
) -> list[Any]:
    """Parse a polymorphic list where each item has ``_target_`` for hydra.instantiate.

    Validates that each instantiated item is a subclass of ``base_class``.
    """
    result: list[Any] = []
    if not items:
        return result

    for i, item in enumerate(items):
        if not isinstance(item, dict | DictConfig):
            raise ConfigError(f"{list_name}[{i}] must be a dict, got {type(item)}")

        item_conf = OmegaConf.create(item) if isinstance(item, dict) else item

        if "_target_" not in item_conf:
            raise ConfigError(f"{list_name}[{i}] must have '_target_' key, got keys: {list(item_conf.keys())}")

        try:
            parsed = instantiate(item_conf)
        except (InstantiationException, ImportError, AttributeError) as e:
            raise ConfigError(f"{list_name}[{i}] failed to instantiate _target_ '{item_conf._target_}': {e}") from e

        if not isinstance(parsed, base_class):
            raise ConfigError(
                f"{list_name}[{i}] _target_ must inherit {base_class.__name__}, got {type(parsed).__name__}"
            )

        result.append(parsed)

    return result
