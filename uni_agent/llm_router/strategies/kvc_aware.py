"""KVCache-aware runtime strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig
from uni_agent.llm_router.logging import get_router_logger
from uni_agent.llm_router.strategies.registry import StrategyRegistry
from uni_agent.llm_router.types import Layer, MetricKey

if TYPE_CHECKING:
    from uni_agent.llm_router.store import DataStore
    from uni_agent.llm_router.strategies.base import ReplicaInfo

logger = get_router_logger("kvc-aware-strategy")

STICKY_TOP_SCORE = 1e9

DEFAULT_LOAD_WEIGHTS: tuple[float, float, float] = (0.4, 0.3, 0.3)


def load_normalized(
    kv_usage: float,
    running: int | float,
    waiting: int | float,
    *,
    max_num_seqs: int,
    weights: tuple[float, float, float] = DEFAULT_LOAD_WEIGHTS,
) -> float:
    """load = a·kv_usage + b·running/max_num_seqs + c·waiting/max_num_seqs (∈ [0,1], bigger = more loaded)."""
    a, b, c = weights
    running_usage = min(1.0, float(running) / float(max_num_seqs))
    waiting_usage = min(1.0, float(waiting) / float(max_num_seqs))
    return a * float(kv_usage) + b * running_usage + c * waiting_usage


def replica_load_from_store(
    store: DataStore,
    replica_id: str,
    *,
    max_num_seqs: int,
    weights: tuple[float, float, float] = DEFAULT_LOAD_WEIGHTS,
) -> float:
    """Normalized load ∈ [0, 1] for one replica: read kv/running/waiting from the
    store and apply :func:`load_normalized`.

    The single source of truth for "how loaded is a replica", shared by the
    strategy's scoring/overload checks and the balancer's migration-gate
    imbalance judgment — so both see identical load numbers.
    """
    m = store.get_metrics(replica_id)
    kv_usage = store.kv_cache_load(replica_id)
    running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
    waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
    return load_normalized(kv_usage, running, waiting, max_num_seqs=max_num_seqs, weights=weights)


class StrategyError(Exception):
    """Strategy construction or scoring error."""


class KVCacheAwareStrategy:
    """Runtime strategy constructed from a ``KVCAwareStrategyConfig``."""

    def __init__(
        self,
        *,
        alpha: float,
        load_threshold: float,
        layer_weights: dict[Layer, float],
        collector_names: list[str],
        weight: float,
        load_weights: tuple[float, float, float] = DEFAULT_LOAD_WEIGHTS,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise StrategyError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 < load_threshold < 1:
            raise StrategyError(f"load_threshold must be in (0, 1), got {load_threshold}")
        _valid_layers = {Layer.GPU, Layer.CPU, Layer.SSD}
        if set(layer_weights.keys()) != _valid_layers:
            raise StrategyError(f"layer_weights keys must be {_valid_layers}, got {set(layer_weights.keys())}")
        for layer_key, layer_weight in layer_weights.items():
            if layer_weight < 0:
                raise StrategyError(f"layer_weights[{layer_key}] must be >= 0, got {layer_weight}")
        weights_sum = sum(layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise StrategyError(f"layer_weights values must sum to 1.0, got {weights_sum}")
        if len(load_weights) != 3 or any(w < 0 for w in load_weights):
            raise StrategyError(f"load_weights must be 3 non-negative values, got {load_weights}")
        if abs(sum(load_weights) - 1.0) > 1e-6:
            raise StrategyError(f"load_weights must sum to 1.0, got {sum(load_weights)}")

        self.alpha = float(alpha)
        self.load_threshold = float(load_threshold)
        self.layer_weights = dict(layer_weights)
        self.collector_names = collector_names
        self.weight = weight
        self.load_weights = tuple(load_weights)
        self._max_num_seqs: int | None = None
        logger.info(
            f"KVCacheAwareStrategy created: alpha={self.alpha:.2f}, "
            f"load_threshold={self.load_threshold:.2f}, load_weights={self.load_weights}"
        )

    def set_capacity(self, max_num_seqs: int) -> None:
        """Inject ``--max-num-seqs`` from the server handle's rollout config."""
        if not isinstance(max_num_seqs, int) or max_num_seqs <= 0:
            raise StrategyError(f"max_num_seqs must be a positive int, got {max_num_seqs}")
        self._max_num_seqs = max_num_seqs
        logger.info(f"KVCacheAwareStrategy capacity set: max_num_seqs={max_num_seqs}")

    @classmethod
    def from_config(cls, cfg: KVCAwareStrategyConfig) -> KVCacheAwareStrategy:
        """Construct from config. ``max_num_seqs`` is injected by the Balancer
        via ``set_capacity`` after fetching from the server handle."""
        return cls(
            alpha=cfg.alpha,
            load_threshold=cfg.load_threshold,
            layer_weights=cfg.layer_weights,
            collector_names=cfg.collector_names,
            weight=cfg.weight,
        )

    def _compute_load(self, kv_usage: float, running: int | float, waiting: int | float) -> float:
        if self._max_num_seqs is None:
            raise StrategyError("set_capacity() must be called before routing")
        return load_normalized(kv_usage, running, waiting, max_num_seqs=self._max_num_seqs, weights=self.load_weights)

    def _replica_load(self, store: DataStore, replica_id: str) -> float:
        """Normalized load ∈ [0, 1] for one replica, via the shared formula.

        Delegates to the module-level :func:`replica_load_from_store` (the single
        source of truth also used by the balancer's migration gate); the only
        strategy-specific part is the capacity guard.
        """
        if self._max_num_seqs is None:
            raise StrategyError("set_capacity() must be called before routing")
        return replica_load_from_store(
            store, replica_id, max_num_seqs=self._max_num_seqs, weights=self.load_weights
        )

    def is_overloaded(
        self,
        store: DataStore,
        replica: ReplicaInfo,
    ) -> bool:
        """Return True if ``replica`` is overloaded (``load > load_threshold``).

        Used only by the sticky short-circuit to decide whether to send a
        returning session back to its bound replica. Combined scoring never
        consults overload.
        """
        return self._replica_load(store, replica.replica_id) > self.load_threshold

    def _sticky_shortcut(
        self,
        store: DataStore,
        replicas: list[ReplicaInfo],
        request_id: str | None,
        sticky_table: Any,
        turn: int | None,
    ) -> list[float] | None:
        """Return a pre-built score list if a sticky replica should win, else None.

        Sticky replica wins when: ``request_id``/``sticky_table`` are provided,
        the bound replica is present in ``replicas``, and it is NOT overloaded.
        On win, returns a list with ``STICKY_TOP_SCORE`` at the bound replica's
        index and ``0.0`` elsewhere. On miss / overload / absence, returns
        ``None`` so the caller falls through to combined scoring.

        Note: this strategy is a stateless scorer. Conditional migration (moving
        the youngest request off an overloaded, imbalanced replica) is decided
        and applied by the ``KVCAwareBalancer`` before scoring runs — by the
        time this method sees an overloaded bound replica, any migration has
        already rebound the affected request, so here overload just falls back
        to combined scoring exactly as before. ``turn`` is accepted for
        interface parity but unused by the scorer.
        """
        if not request_id or sticky_table is None:
            return None
        sticky_id = sticky_table.get(request_id)
        if sticky_id is None:
            return None
        for idx, replica in enumerate(replicas):
            if replica.replica_id == sticky_id:
                if self._replica_load(store, replica.replica_id) > self.load_threshold:
                    logger.info(f"score(): STICKY replica={sticky_id} OVERLOADED → fallback to COMBINED scoring")
                    return None
                logger.info(f"score(): STICKY replica={sticky_id} HIT → short-circuit (top score)")
                scores = [0.0] * len(replicas)
                scores[idx] = STICKY_TOP_SCORE
                return scores
        logger.info(f"score(): sticky replica={sticky_id} not in pool, fallback to combined scoring")
        return None

    def score(
        self,
        prompt_ids: list[int] | None,
        store: DataStore,
        replicas: list[ReplicaInfo],
        request_id: str | None = None,
        sticky_table: Any = None,
        turn: int | None = None,
    ) -> list[float]:
        """Score each replica. Larger is better.

        S = α·S_cache + (1-α)·S_load
        """
        if not isinstance(replicas, list):
            raise StrategyError(f"replicas must be a list, got {type(replicas).__name__}")
        if not replicas:
            return []
        # Sticky short-circuit: bound, non-overloaded replica wins outright.
        shortcut = self._sticky_shortcut(store, replicas, request_id, sticky_table, turn)
        if shortcut is not None:
            return shortcut
        effective_prompt_ids = prompt_ids or []

        result = []
        for replica in replicas:
            m = store.get_metrics(replica.replica_id)
            kv_usage = store.kv_cache_load(replica.replica_id)
            running = m.get(MetricKey.NUM_REQUESTS_RUNNING, 0)
            waiting = m.get(MetricKey.NUM_REQUESTS_WAITING, 0)
            load = self._compute_load(kv_usage, running, waiting)
            s_load = 1.0 - load
            s_cache, gpu_hit = self._cache_score(store, replica, effective_prompt_ids)
            score = self.alpha * s_cache + (1 - self.alpha) * s_load
            result.append(score)
            logger.info(
                f"score(): replica={replica.replica_id} kv={kv_usage:.3f} running={running} waiting={waiting} "
                f"→ load={load:.4f} s_load={s_load:.4f} | gpu_hit={gpu_hit:.2f} s_cache={s_cache:.4f} "
                f"({self.alpha:.2f}·cache + {1 - self.alpha:.2f}·load) → score={score:.4f}"
            )
        scores_str = ", ".join(f"{r.replica_id}={result[i]:.4f}" for i, r in enumerate(replicas))
        logger.info(f"score(): COMBINED scores: {scores_str}")
        return result

    def _cache_score(
        self,
        store: DataStore,
        replica: ReplicaInfo,
        prompt_ids: list[int],
    ) -> tuple[float, float]:
        """Three-layer weighted prefix-cache hit score ∈ [0, 1].

            S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit

        Each ``*_hit`` comes from ``get_layer_prefix_hit_rate`` (0.0–1.0; cpu/ssd
        return 0.0 until the mooncake tier collector is wired). Returns
        ``(s_cache, gpu_hit)`` so the caller logs gpu_hit without re-querying.
        """
        gpu_hit = store.get_layer_prefix_hit_rate(replica.replica_id, prompt_ids, Layer.GPU) or 0.0
        cpu_hit = store.get_layer_prefix_hit_rate(replica.replica_id, prompt_ids, Layer.CPU) or 0.0
        ssd_hit = store.get_layer_prefix_hit_rate(replica.replica_id, prompt_ids, Layer.SSD) or 0.0
        w = self.layer_weights
        s_cache = w[Layer.GPU] * gpu_hit + w[Layer.CPU] * cpu_hit + w[Layer.SSD] * ssd_hit
        return s_cache, gpu_hit


StrategyRegistry.register(KVCAwareStrategyConfig, KVCacheAwareStrategy)
