# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Balancer orchestration for agent-aware routing."""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import ray
from omegaconf import DictConfig, OmegaConf

from .collectors import CollectorManager
from .config import KVCAwareConfig
from .store import DataStore
from .strategies import (
    ReplicaInfo,
    StrategyRegistry,
    route,
)
from .types import MetricKey
from .utils.prefix_cache import resolve_prefix_hashes

logger = logging.getLogger(__name__)


class CapacityDeferred(RuntimeError):
    """Admission queue rejection: no capacity right now, retry shortly.

    The message carries both our own marker (``admission-defer``) and verl's
    existing retry-matching string ("No available servers in load balancer").
    The own marker is what the upgraded verl retry loop matches; the legacy
    string is the fallback — a fork diff lost in an upstream merge degrades
    to the legacy branch's 1 s retry instead of the request dying.
    """


@dataclass
class _DeferredEntry:
    """One request parked in the admission defer queue.

    Carries the resolved block-hash chain (not prompt_ids — ~1.3 KB vs
    ~128 KB per 32 k-token entry); the increment is recomputed against the
    *current* pin set at each scan, so the chain is the minimal sufficient
    state.
    """

    request_id: str
    enqueued_at: float  # time.perf_counter()
    turn: int
    ctx_tokens: int
    block_hashes: list
    sticky: str | None  # last replica this session released on (probe ledger)
    aged: bool = False  # anti-starvation promotion already applied


@dataclass
class _ReplicaLedger:
    """Per-replica commitment-ledger snapshot for one admission decision."""

    replica_id: str
    pin_blocks: int
    acting_hold: float
    decode_reserve: int
    avail_admit: float  # cap − pin_tokens − acting_hold − decode_reserve − buffer_base


@dataclass
class _DecisionView:
    """Ledger snapshot extended with this request's increment on one replica."""

    ledger: _ReplicaLedger
    predicted: int  # predict_pin_delta (blocks already converted to tokens)
    hold_s: float  # the request's own claim on this replica (0 when cold/new)
    inc: float  # max(0, predicted − hold_s) — the ledger-consistent increment

    @property
    def replica_id(self) -> str:
        return self.ledger.replica_id

    @property
    def avail_admit(self) -> float:
        return self.ledger.avail_admit

    def feasible(self) -> bool:
        return self.avail_admit >= self.inc


class KVCAwareBalancer:
    """Route requests and manage collector-backed state."""

    _ROUTE_LOG_EVERY = 64
    _MAX_DISCOVERY_WORKERS = 8
    _DISCOVERY_TIMEOUT_SECONDS = 5.0

    # Turn-phase probe bounds: sessions finalize/abort without notifying the
    # balancer, so the ledger is pruned by TTL once it exceeds the cap.
    _PROBE_MAX_SESSIONS = 8192
    _PROBE_TTL_SECONDS = 3600.0

    # Pin watchdog bound: a REASONING phase lasting longer than this means its
    # dispatch's pin will never be released (§5.8).
    _MAX_REASONING_SECONDS = 1800.0

    # Deep-queue rescan guard (§5.6.3): every 1 s client retry of a queued
    # request would otherwise trigger a full O(queue × replicas × chain) scan;
    # at 100+ queued entries that saturates the actor. Retry-driven rescans are
    # throttled to this interval — budget changes themselves (dispatch/release)
    # always rescan immediately, so only hold-decay-only transitions wait out
    # the interval (≤250 ms, well under the 1 s retry period).
    _RESCAN_MIN_INTERVAL = 0.25

    def __init__(
        self,
        servers: dict[str, Any],
        config: dict[str, Any] | DictConfig | None = None,
        provider_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not servers:
            raise ValueError("servers must be non-empty")
        if provider_factory is None:
            provider_factory = CollectorManager
        self._provider_factory = provider_factory

        self._servers: dict[str, Any] = dict(servers)
        rollout_config = self._fetch_rollout_config()
        self._config = self._init_config(config, rollout_config)
        logger.info("KVCAwareBalancer, config=%s", self._config)

        primary_cfg = self._config.strategy
        self._strategy = StrategyRegistry.get(type(primary_cfg)).from_config(primary_cfg)
        self._strategy_summary = str(self._strategy)

        self._inflight: dict[str, int] = {sid: 0 for sid in self._servers}
        max_num_seqs = self._rollout_config_int(rollout_config, "max_num_seqs", 256)
        max_num_batched_tokens = self._rollout_config_int(rollout_config, "max_num_batched_tokens", 2048)
        if hasattr(self._strategy, "set_capacity"):
            self._strategy.set_capacity(max_num_seqs, max_num_batched_tokens)
        logger.info(
            "KVCAwareBalancer: max_num_seqs=%s, max_num_batched_tokens=%s",
            max_num_seqs,
            max_num_batched_tokens,
        )

        self._route_calls = 0
        self._route_time_total_ms = 0.0
        self._route_time_max_ms = 0.0

        # Turn-phase probe (passive, "channel 1"): the gateway passes
        # request_id=session_id into backend.generate, so request_id is stable
        # per episode. acquire() marks REASONING start and release() its end;
        # the next acquire of the same session then measures the acting/idle
        # gap. No agent-side instrumentation involved.
        self._probe_acquire_ts: dict[str, float] = {}  # request_id -> last acquire time
        self._probe_release_ts: dict[str, float] = {}  # request_id -> last release time
        self._probe_turns: dict[str, int] = {}  # request_id -> turn count
        self._probe_sessions = 0
        # Probe ledger extensions for admission (design doc §5.2): the prompt
        # size of the last acquire and where the last release happened — the
        # per-session inputs of the soft-claim (acting_hold) ledger.
        self._probe_ctx_tokens: dict[str, int] = {}  # request_id -> last acquire's len(prompt_ids)
        self._probe_last_replica: dict[str, str] = {}  # request_id -> last release's server_id

        # Turn-phase admission (design doc §5): the defer queue is the ONE
        # queue of the system (vLLM-internal waiting stays ≈ 0 by the gate's
        # invariant). Pull-based dequeue — scans run on ledger-change events
        # (release) and on client retries; the sync actor never blocks.
        self._admission = self._config.admission
        self._defer_queue: list[_DeferredEntry] = []
        # request_id -> (server_id, admitted_at, predicted_pin_delta): admitted
        # assignments waiting ≤ ready_ttl for the client's retry to pick up.
        # Their predicted pin deltas count as pending promises in the ledger
        # (the real pin only lands when the collector processes the pickup).
        self._admitted_ready: dict[str, tuple[str, float, int]] = {}
        self._admission_decisions = 0
        self._admission_admit_total = 0
        self._admission_queue_admit_total = 0
        self._admission_aging_count = 0
        self._admission_ghost_pruned = 0
        self._last_scan_ts = 0.0  # retry-driven rescan throttle (deep-queue guard)
        self._callbacks: dict[str, list[Callable[..., None]]] = {
            "on_acquire": [],
            "on_release": [],
            "on_servers_removed": [],
        }
        self._store = DataStore()
        self._init_provider()

    def _fetch_rollout_config(self) -> Any | None:
        """Return the first available rollout config."""
        for handle in self._servers.values():
            if not hasattr(handle, "get_rollout_config"):
                continue
            try:
                return ray.get(handle.get_rollout_config.remote())
            except Exception as e:  # noqa: BLE001
                logger.warning("get_rollout_config failed on %s: %s", type(handle).__name__, e)
        return None

    @staticmethod
    def _read_nested(obj: Any, *keys: str) -> Any:
        """Walk ``obj`` down ``keys``, accepting mappings or plain objects."""
        current = obj
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict | DictConfig):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    @classmethod
    def _router_override(cls, rollout_config: Any | None) -> dict[str, Any] | None:
        if rollout_config is None:
            return None
        router = cls._read_nested(rollout_config, "custom", "agent_framework", "router")
        if not router:
            return None
        if isinstance(router, DictConfig):
            return OmegaConf.to_container(router, resolve=True)
        return dict(router)

    @staticmethod
    def _rollout_config_int(rollout_config: Any | None, name: str, default: int) -> int:
        if rollout_config is None:
            return default
        if isinstance(rollout_config, dict | DictConfig):
            value = rollout_config.get(name, default)
        else:
            value = getattr(rollout_config, name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            logger.warning("server returned invalid %s=%r; using default=%s", name, value, default)
            return default
        return value

    def _init_config(self, config: dict[str, Any] | DictConfig | None, rollout_config: Any | None) -> KVCAwareConfig:
        _config = KVCAwareConfig.from_config(config)
        _config.apply_override(self._router_override(rollout_config))
        return _config

    def _init_provider(self) -> None:
        """Resolve per-server endpoints from vLLM HTTP servers, then start collectors.

        Non-actor handles (plain strings in unit tests) have no
        ``get_server_address``; discovery is skipped and collectors fall back
        to configured/default endpoints.
        """
        collection_names = sorted(self._strategy.collection_names)
        address_futures = []
        replica_ids = []
        for replica_id, handle in self._servers.items():
            if not hasattr(handle, "get_server_address"):
                logger.warning(
                    "server '%s' handle has no get_server_address remote (type=%s); "
                    "skipping dynamic endpoint discovery",
                    replica_id,
                    type(handle).__name__,
                )
                continue
            replica_ids.append(replica_id)
            address_futures.append(handle.get_server_address.remote())

        server_addresses: dict[str, str] = {}
        if replica_ids:
            addresses = ray.get(address_futures)
            server_addresses = {
                replica_id: f"{ip}:{port}" for replica_id, (ip, port) in zip(replica_ids, addresses, strict=True)
            }
        kv_event_endpoints = self._discover_kv_event_endpoints(server_addresses)
        self._provider = self._provider_factory(
            self._config.collector,
            collection_names,
            server_addresses=server_addresses,
            kv_event_endpoints=kv_event_endpoints,
            balancer_handler=self,
        )
        self._provider.start()

    def _discover_kv_event_endpoints(self, server_addresses: dict[str, str]) -> dict[str, list[str]]:
        if not server_addresses:
            return {}

        endpoints: dict[str, list[str]] = {}
        max_workers = min(len(server_addresses), self._MAX_DISCOVERY_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kv-event-discovery") as executor:
            futures = {
                executor.submit(self._fetch_kv_event_endpoints, address): replica_id
                for replica_id, address in server_addresses.items()
            }
            for future in as_completed(futures):
                replica_id = futures[future]
                try:
                    source = future.result()
                except httpx.HTTPError as e:
                    logger.warning("failed to discover KV-event endpoint from server '%s': %s", replica_id, e)
                    continue
                if source is not None:
                    endpoints[replica_id] = source
        return endpoints

    @staticmethod
    def _collector_address(endpoint: Any) -> str:
        """Convert a vLLM TCP URI into the address expected by ZMQTransport."""
        if endpoint is None:
            return ""
        if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
            raise RuntimeError(f"unsupported KV-event endpoint: {endpoint!r}")
        return endpoint.removeprefix("tcp://")

    @classmethod
    def _parse_kv_event_sources(cls, sources: Any) -> list[str] | None:
        """Convert vLLM's rank-keyed response into the current single-DP shape."""
        if not isinstance(sources, dict):
            raise RuntimeError(f"vLLM returned invalid KV-event sources: {sources!r}")
        if not sources:
            return None
        if len(sources) > 1:
            raise RuntimeError(
                "agent-aware-router does not support multiple DP ranks per rollout server yet: "
                f"reported ranks={sorted(sources)}"
            )

        rank, config = next(iter(sources.items()))
        if not isinstance(config, dict):
            raise RuntimeError(f"vLLM returned invalid KV-event config for DP rank {rank}: {config!r}")
        publisher = config.get("publisher")
        if publisher != "zmq":
            raise RuntimeError(f"unsupported KV-event publisher for DP rank {rank}: {publisher!r}")
        topic = config.get("topic")
        if not isinstance(topic, str):
            raise RuntimeError(f"invalid KV-event topic for DP rank {rank}: {topic!r}")
        return [
            cls._collector_address(config.get("endpoint")),
            cls._collector_address(config.get("replay_endpoint")),
            publisher,
            topic,
        ]

    @classmethod
    def _fetch_kv_event_endpoints(cls, server_address: str) -> list[str] | None:
        """Query the endpoint-discovery API provided by vLLM #55844."""
        with httpx.Client(timeout=cls._DISCOVERY_TIMEOUT_SECONDS, trust_env=False) as client:
            response = client.get(f"http://{server_address}/kv_event_sources")
            response.raise_for_status()
            return cls._parse_kv_event_sources(response.json())

    def register_call_back(self, event: str, fn: Callable[..., None]) -> None:
        """Register an event callback."""
        self._callbacks.setdefault(event, []).append(fn)

    def un_register_call_back(self, event: str, fn: Callable[..., None]) -> None:
        """Remove ``fn`` from ``event``'s callback list (idempotent)."""
        lst = self._callbacks.get(event, [])
        if fn in lst:
            lst.remove(fn)

    def _fire(self, event: str, *args: Any) -> None:
        """Invoke every registered callback for ``event``; errors are swallowed."""
        for fn in self._callbacks.get(event, []):
            try:
                fn(*args)
            except Exception as exc:  # noqa: BLE001
                logger.warning("callback %s failed: %s: %s", event, type(exc).__name__, exc)

    def get_all_servers(self) -> list[str]:
        """List all active server ids."""
        return list(self._servers.keys())

    def get_status(self) -> dict:
        """Construction + routing snapshot for debugging."""
        return {
            "servers": list(self._servers.keys()),
            "provider": type(self._provider).__name__,
            "strategies": [{"type": type(self._strategy).__name__}],  # legacy key; single strategy
            "route_calls": self._route_calls,
            "sticky_size": self._store.sticky_status()["size"],
            "total_inflight": self.get_total_inflight(),
            "probe_sessions_tracked": len(self._probe_turns),
            "probe_sessions_total": self._probe_sessions,
            "probe_turns_total": sum(self._probe_turns.values()),
            "admission_admit_total": self._admission_admit_total,
            "admission_queue_length": len(self._defer_queue),
            "admission_queue_admit_total": self._admission_queue_admit_total,
            "admission_aging_count": self._admission_aging_count,
            "admission_ghost_pruned": self._admission_ghost_pruned,
        }

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server and notify statistic collectors."""
        if request_id is not None:
            # Probe: REASONING ends here (generate() releases in finally);
            # duration = release - matching acquire.
            now = time.perf_counter()
            acquire_ts = self._probe_acquire_ts.pop(request_id, None)
            if acquire_ts is not None:
                logger.info(
                    "session-phase: rid=%s event=release reasoning_dur=%.3fs",
                    request_id,
                    now - acquire_ts,
                )
            self._probe_release_ts[request_id] = now
            self._probe_last_replica[request_id] = server_id
        if self._inflight.get(server_id, 0) > 0:
            self._inflight[server_id] -= 1
        self._fire("on_release", server_id, request_id)
        # Ledger-change event: release shifted pin→hold and freed decode_reserve.
        # A scan right after the collector's unpin gives queued entries the
        # first shot at the freed budget (pull-based dequeue, §5.6.3).
        if self._defer_queue and self._admission.mode == "enforce":
            self._scan_defer_queue(self._replica_infos(), time.perf_counter())

    def _probe_prune(self, now: float) -> None:
        """Drop probe entries idle longer than the TTL once the cap is exceeded."""
        if len(self._probe_turns) <= self._PROBE_MAX_SESSIONS:
            return
        cutoff = now - self._PROBE_TTL_SECONDS
        for rid, acquire_ts in list(self._probe_acquire_ts.items()):
            if acquire_ts >= cutoff:
                continue
            # A session that released recently but never re-acquired still has
            # a usable acting-gap anchor; keep it.
            release_ts = self._probe_release_ts.get(rid)
            if release_ts is not None and release_ts >= cutoff:
                continue
            self._probe_acquire_ts.pop(rid, None)
            self._probe_release_ts.pop(rid, None)
            self._probe_turns.pop(rid, None)
            self._probe_ctx_tokens.pop(rid, None)
            self._probe_last_replica.pop(rid, None)

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Return the highest-ranked server id and handle.

        With the admission gate on (design doc §5), the request first passes
        the two-level commitment ledger; when no replica can take the
        increment it is parked in the defer queue and ``CapacityDeferred`` is
        raised — the verl client retries every 1 s, either picking up a queued
        admission (``_admitted_ready``) or re-driving a dequeue scan.
        """
        now = time.perf_counter()

        # Probe bookkeeping. A retry of a queued/ready request is the SAME
        # logical turn (verl retries every 1 s) — do not re-count it.
        last_release = self._probe_release_ts.pop(request_id, None)
        acting_gap = None if last_release is None else now - last_release
        was_ready = self._admitted_ready.pop(request_id, None)
        was_queued = self._find_in_queue(request_id)
        if was_ready is None and was_queued is None:
            turn = self._probe_turns.get(request_id, 0) + 1
            if turn == 1:
                self._probe_sessions += 1
            self._probe_turns[request_id] = turn
        else:
            turn = self._probe_turns.get(request_id, 1)
        # The request's own soft claim (hold_s) is priced from the PREVIOUS
        # turn's prompt size at the pre-pop anchor — the self-financing input
        # of inc (§5.4). Read it before the ctx dict is overwritten.
        prev_ctx = self._probe_ctx_tokens.get(request_id, 0)
        own_hold = 0.0
        if last_release is not None and prev_ctx > 0:
            own_hold = prev_ctx * 2.0 ** (-(now - last_release) / self._admission.tau)
        self._probe_ctx_tokens[request_id] = len(prompt_ids or [])
        self._probe_prune(now)
        self._admission_gc(now)

        # Pickup: a previous scan already admitted this request — dispatch
        # straight to the pre-chosen replica without re-routing.
        if was_ready is not None:
            server_id, admitted_at, predicted = was_ready
            logger.info(
                "admission: rid=%s turn=%s decision=queue-pickup wait=%.1fs target=%s",
                request_id,
                turn,
                now - admitted_at,
                server_id,
            )
            self._dispatch_admitted(request_id, server_id, prompt_ids, turn, acting_gap, now, predicted)
            return server_id, self._servers[server_id]

        replicas = self._replica_infos()

        if self._admission.mode == "enforce":
            if was_queued is not None:
                # Retry of a queued request: rescan — budget may have appeared.
                # Throttled: at deep queue every 1 s retry would otherwise
                # rescan the whole queue and saturate the actor (§5.6.3).
                self._maybe_rescan(replicas, now)
                ready = self._admitted_ready.pop(request_id, None)
                if ready is not None:
                    server_id, admitted_at, predicted = ready
                    logger.info(
                        "admission: rid=%s turn=%s decision=queue-pickup wait=%.1fs target=%s",
                        request_id,
                        turn,
                        now - admitted_at,
                        server_id,
                    )
                    self._dispatch_admitted(request_id, server_id, prompt_ids, turn, acting_gap, now, predicted)
                    return server_id, self._servers[server_id]
                # Still queued: keep the claim on the ledger, tell the client
                # to retry (排队保留索取权，§5.2/§5.3).
                self._restore_release_anchor(request_id, last_release)
                pos = self._priority_view(now).index(was_queued) + 1
                logger.info(
                    "admission: rid=%s turn=%s decision=defer pos=%d wait=%.1fs",
                    request_id,
                    turn,
                    pos,
                    now - was_queued.enqueued_at,
                )
                raise CapacityDeferred(self._defer_message(request_id, pos))

            if self._defer_queue:
                # No-jump (§6.4): strict FIFO — a new arrival queues behind the
                # backlog even when it would fit.
                entry = self._enqueue_deferred(request_id, prompt_ids, turn, None, now)
                self._restore_release_anchor(request_id, last_release)
                pos = self._priority_view(now).index(entry) + 1
                logger.info("admission: rid=%s turn=%s decision=enqueue reason=no-jump pos=%d", request_id, turn, pos)
                raise CapacityDeferred(self._defer_message(request_id, pos))

        views: dict[str, _DecisionView] | None = None
        if self._admission.mode != "off":
            replicas, views = self._admission_gate(request_id, prompt_ids, replicas, turn, last_release, own_hold, now)

        self._route_calls += 1
        t0 = time.perf_counter()
        ranking = route(
            self._strategy,
            prompt_ids,
            self._store,
            replicas,
            request_id,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        self._route_time_total_ms += dt_ms
        self._route_time_max_ms = max(self._route_time_max_ms, dt_ms)
        if not ranking:
            # Transient unavailability (empty pool) — message-aligned with
            # verl's retry branch so the request retries in 1 s instead of
            # dying (§5.6.4). The anchor is restored: nothing dispatched.
            self._restore_release_anchor(request_id, last_release)
            raise RuntimeError("pool-empty | No available servers in load balancer")
        server_id = ranking[0]
        predicted = views[server_id].predicted if views is not None else None
        self._dispatch_admitted(request_id, server_id, prompt_ids, turn, acting_gap, now, predicted)
        logger.debug(
            "request=%s routed to server=%s (ranking=%s, pool=%s, route=%.2fms, strategy=[%s])",
            request_id,
            server_id,
            ranking,
            list(self._servers),
            dt_ms,
            self._strategy_summary,
        )
        if self._route_calls % self._ROUTE_LOG_EVERY == 0:
            logger.info(
                "route-stats: calls=%s total=%.3fs mean=%.2fms max=%.2fms strategy=[%s]",
                self._route_calls,
                self._route_time_total_ms / 1000,
                self._route_time_total_ms / self._route_calls,
                self._route_time_max_ms,
                self._strategy_summary,
            )
        return server_id, self._servers[server_id]

    # ── Turn-phase admission: commitment ledger (design doc §5.3/§5.4) ──

    def _replica_infos(self) -> list[ReplicaInfo]:
        return [ReplicaInfo(replica_id=sid) for sid in self._servers]

    def _token_capacity(self) -> int:
        """Per-replica KV token capacity = ``num_gpu_blocks × block_size`` (0 = unknown)."""
        for node_id in self._store.get_metric_node_ids():
            nblk = self._store.get_metric(node_id, MetricKey.NUM_GPU_BLOCKS)
            if nblk and nblk > 0:
                return int(nblk) * int(self._store.get_block_size() or 16)
        return 0

    def _hold_of(self, request_id: str, now: float) -> float:
        """One session's own soft claim (0 unless ACTING/QUEUED — §5.4 hold_s)."""
        if not self._admission.enable_acting_hold:
            return 0.0
        release_ts = self._probe_release_ts.get(request_id)
        if release_ts is None:
            return 0.0
        ctx = self._probe_ctx_tokens.get(request_id, 0)
        if ctx <= 0:
            return 0.0
        return ctx * 2.0 ** (-(now - release_ts) / self._admission.tau)

    def _acting_hold(self, replica_id: str, now: float) -> float:
        """Soft claims of all ACTING∪QUEUED sessions last released on ``replica_id``.

        Each claim is ``ctx(s) × 2^(−gap/τ)`` — one decay prices both
        P(return)↓ and P(cache alive)↓ (§5.3).
        """
        if not self._admission.enable_acting_hold:
            return 0.0
        tau = self._admission.tau
        total = 0.0
        for rid, release_ts in self._probe_release_ts.items():
            if self._probe_last_replica.get(rid) != replica_id:
                continue
            ctx = self._probe_ctx_tokens.get(rid, 0)
            if ctx > 0:
                total += ctx * 2.0 ** (-(now - release_ts) / tau)
        return total

    def _pending_pin_blocks(self) -> dict[str, int]:
        """Pin deltas of admitted-but-not-yet-dispatched assignments, by target.

        Between a scan's queue-admit and the client's ≤1 s pickup the real pin
        has not landed; counting these as promises keeps concurrent decisions
        from double-spending the same budget.
        """
        pending: dict[str, int] = {}
        for _sid, _ts, predicted in self._admitted_ready.values():
            pending[_sid] = pending.get(_sid, 0) + predicted
        return pending

    def _admission_ledger(self, replicas: list[ReplicaInfo], now: float) -> tuple[int, int, list[_ReplicaLedger]]:
        """Two-level commitment ledger per replica (§5.3).

        ``avail_admit = cap − pin_tokens − acting_hold − decode_reserve − buffer_base``;
        cap unknown (first KV event not in) ⇒ gate bypassed (avail = +inf, §6.3).
        """
        cap = self._token_capacity()
        block_size = self._store.get_block_size() or 16
        pending = self._pending_pin_blocks()
        ledgers = []
        for r in replicas:
            rid = r.replica_id
            pin_blocks = self._store.inflight_block_count(rid) + pending.get(rid, 0)
            hold = self._acting_hold(rid, now)
            reserve = self._admission.decode_buffer * self._inflight.get(rid, 0)
            if cap > 0:
                avail = cap - pin_blocks * block_size - hold - reserve - self._admission.buffer_base
            else:
                avail = math.inf
            ledgers.append(_ReplicaLedger(rid, pin_blocks, hold, reserve, avail))
        return cap, block_size, ledgers

    def _increment_for(
        self,
        ledger: _ReplicaLedger,
        block_hashes: list,
        ctx_tokens: int,
        own_hold: float,
        request_id: str,
        block_size: int,
    ) -> _DecisionView:
        """Ledger-consistent increment for one request on one replica (§5.4).

        ``inc = max(0, predict_pin_delta − hold_s)``; when the chain is
        unavailable (block size unknown / no KV events yet) the whole prompt
        is counted as new — the conservative fallback.
        """
        hold_s = own_hold if self._probe_last_replica.get(request_id) == ledger.replica_id else 0.0
        if block_hashes:
            predicted = self._store.predict_pin_delta(ledger.replica_id, block_hashes)
        elif ctx_tokens:
            predicted = math.ceil(ctx_tokens / block_size) if block_size else 0
        else:
            predicted = 0
        return _DecisionView(ledger, predicted, hold_s, max(0.0, predicted * block_size - hold_s))

    def _admission_gate(
        self,
        request_id: str,
        prompt_ids: list[int] | None,
        replicas: list[ReplicaInfo],
        turn: int,
        last_release: float | None,
        own_hold: float,
        now: float,
    ) -> tuple[list[ReplicaInfo], dict[str, _DecisionView]]:
        """Evaluate the gate on the empty-queue fast path (§5.1 ④/⑤).

        Returns the replicas to route on (narrowed to the feasible set when
        the gate binds — the strategy still picks within it, §5.5) plus the
        per-replica decision views for predict-check. Raises
        ``CapacityDeferred`` when nothing is feasible (enforce only).
        """
        prompt_ids = prompt_ids or []
        block_hashes = resolve_prefix_hashes(list(prompt_ids), request_id, self._store)
        cap, block_size, ledgers = self._admission_ledger(replicas, now)
        views = {
            led.replica_id: self._increment_for(led, block_hashes, len(prompt_ids), own_hold, request_id, block_size)
            for led in ledgers
        }
        self._admission_decisions += 1
        self._maybe_ledger_check(ledgers, cap, block_size)

        feasible = [v.replica_id for v in views.values() if v.feasible()]
        if self._admission.mode == "shadow":
            # P0 shadow (§8.2): full gate logic, log-only — never intercepts.
            if feasible:
                self._admission_admit_total += 1
                self._log_decision(request_id, turn, "admit", views)
            else:
                self._log_decision(request_id, turn, "shadow-defer", views)
            return replicas, views

        if not feasible:
            self._log_decision(request_id, turn, "enqueue", views)
            self._enqueue_deferred(request_id, prompt_ids, turn, block_hashes, now)
            self._restore_release_anchor(request_id, last_release)
            self._scan_defer_queue(replicas, now)  # uniform with §5.6.4 (cannot admit this one — just ruled out)
            raise CapacityDeferred(self._defer_message(request_id, len(self._defer_queue)))

        self._admission_admit_total += 1
        self._log_decision(request_id, turn, "admit", views)
        if len(feasible) < len(views):
            replicas = [r for r in replicas if r.replica_id in set(feasible)]
        return replicas, views

    def _dispatch_admitted(
        self,
        request_id: str,
        server_id: str,
        prompt_ids: list[int] | None,
        turn: int,
        acting_gap: float | None,
        now: float,
        predicted: int | None,
    ) -> None:
        """Shared dispatch bookkeeping for direct admits and queue pickups."""
        # Probe: REASONING starts at dispatch (a queued retry dispatched only
        # now — the release pairing anchors to this timestamp).
        self._probe_acquire_ts[request_id] = now
        self._probe_last_replica[request_id] = server_id
        if acting_gap is None:
            logger.info("session-phase: rid=%s event=acquire turn=%s acting_gap=cold_start", request_id, turn)
        else:
            logger.info("session-phase: rid=%s event=acquire turn=%s acting_gap=%.3fs", request_id, turn, acting_gap)
        self._inflight[server_id] = self._inflight.get(server_id, 0) + 1
        before = self._store.inflight_block_count(server_id)
        self._fire("on_acquire", request_id, server_id, prompt_ids)
        if predicted is not None:
            # predict-check (§5.8): predicted pin increment vs what the
            # collector actually pinned — chain-resolution drift detector.
            actual = self._store.inflight_block_count(server_id) - before
            logger.info(
                "predict-check: rid=%s predicted=%d actual=%d delta=%d",
                request_id,
                predicted,
                actual,
                actual - predicted,
            )

    def _log_decision(self, request_id: str, turn: int, decision: str, views: dict[str, _DecisionView]) -> None:
        """The ``admission:`` decision line (§5.8); defer-ish decisions carry
        the pin/hold/resv attribution split."""
        avail = ",".join(f"{v.replica_id}:{v.avail_admit:.0f}" for v in views.values())
        incs = ",".join(f"{v.replica_id}:{int(round(v.inc))}" for v in views.values())
        if decision == "admit":
            logger.info("admission: rid=%s turn=%s decision=admit inc=%s avail={%s}", request_id, turn, incs, avail)
            return
        pins = ",".join(f"{v.replica_id}:{v.ledger.pin_blocks}" for v in views.values())
        holds = ",".join(f"{v.replica_id}:{int(v.ledger.acting_hold)}" for v in views.values())
        resvs = ",".join(f"{v.replica_id}:{v.ledger.decode_reserve}" for v in views.values())
        logger.info(
            "admission: rid=%s turn=%s decision=%s inc=%s avail={%s} pin={%s} hold={%s} resv={%s}",
            request_id,
            turn,
            decision,
            incs,
            avail,
            pins,
            holds,
            resvs,
        )

    def _maybe_ledger_check(self, ledgers: list[_ReplicaLedger], cap: int, block_size: int) -> None:
        """Sampled ledger-vs-truth cross-validation (§5.8 ledger-check)."""
        if cap <= 0 or self._admission_decisions % self._admission.ledger_check_every:
            return
        for led in ledgers:
            gpu_perc = self._store.get_metric(led.replica_id, MetricKey.KV_CACHE_USAGE_PERC)
            num_waiting = self._store.get_metric(led.replica_id, MetricKey.NUM_REQUESTS_WAITING)
            occupied_pred = (led.pin_blocks * block_size + led.acting_hold) / cap
            logger.info(
                "ledger-check: replica=%s pin=%d hold=%d occupied_pred=%.3f gpu_perc=%s delta=%s num_waiting=%s",
                led.replica_id,
                led.pin_blocks,
                int(led.acting_hold),
                occupied_pred,
                "-" if gpu_perc is None else f"{gpu_perc:.3f}",
                "-" if gpu_perc is None else f"{occupied_pred - gpu_perc:+.3f}",
                "-" if num_waiting is None else num_waiting,
            )

    def _pin_watchdog(self, now: float) -> None:
        """Warn about REASONING phases far beyond a plausible duration.

        A dispatch that never releases holds its pin forever — once the pin
        enters the gate formula a leak permanently defers that replica
        (§5.8/§6.1). v1 only warns; auto-unbind is a later decision.
        """
        stuck: dict[str, tuple[int, float]] = {}
        for rid, ts in self._probe_acquire_ts.items():
            age = now - ts
            if age <= self._MAX_REASONING_SECONDS:
                continue
            replica = self._probe_last_replica.get(rid, "unknown")
            count, oldest = stuck.get(replica, (0, 0.0))
            stuck[replica] = (count + 1, max(oldest, age))
        for replica, (count, oldest) in stuck.items():
            logger.warning("pin-watchdog: replica=%s leaked=%d oldest_age=%.0fs", replica, count, oldest)

    def _admission_gc(self, now: float) -> None:
        """Lazy TTL cleanup: ghost queue entries + expired ready assignments."""
        if self._admission.mode != "off":
            # Watchdog gates P0's exit criterion "无泄漏告警" (§8.2) — a leak
            # distorts the shadow ledger exactly as it would the real one.
            self._pin_watchdog(now)
        if self._defer_queue:
            kept = []
            for entry in self._defer_queue:
                if now - entry.enqueued_at > self._admission.queue_ttl:
                    self._admission_ghost_pruned += 1
                    logger.info(
                        "admission: rid=%s decision=ghost-prune wait=%.1fs",
                        entry.request_id,
                        now - entry.enqueued_at,
                    )
                    # The episode is assumed dead — drop its hold anchor too.
                    self._probe_release_ts.pop(entry.request_id, None)
                else:
                    kept.append(entry)
            self._defer_queue = kept
        if self._admitted_ready:
            stale = [
                rid for rid, (_sid, ts, _p) in self._admitted_ready.items() if now - ts > self._admission.ready_ttl
            ]
            for rid in stale:
                del self._admitted_ready[rid]
                logger.info("admission: rid=%s decision=ready-expired", rid)

    def _find_in_queue(self, request_id: str) -> _DeferredEntry | None:
        return next((e for e in self._defer_queue if e.request_id == request_id), None)

    def _enqueue_deferred(
        self,
        request_id: str,
        prompt_ids: list[int] | None,
        turn: int,
        block_hashes: list | None,
        now: float,
    ) -> _DeferredEntry:
        """Park a request (§5.6.1): the resolved hash chain (~1.3 KB), not the
        prompt; the increment is recomputed against the current pin set at
        each scan."""
        if block_hashes is None:
            block_hashes = resolve_prefix_hashes(list(prompt_ids or []), request_id, self._store)
        entry = _DeferredEntry(
            request_id=request_id,
            enqueued_at=now,
            turn=turn,
            ctx_tokens=len(prompt_ids or []),
            block_hashes=block_hashes,
            sticky=self._probe_last_replica.get(request_id),
        )
        self._defer_queue.append(entry)
        return entry

    def _restore_release_anchor(self, request_id: str, last_release: float | None) -> None:
        """Keep a queued session's claim on the ledger (§5.2/§5.3).

        acquire popped the release anchor at entry; the request did not
        dispatch, so the anchor is written back — the session stays in the
        ACTING∪QUEUED hold set and its gap keeps growing from the original
        release."""
        if last_release is not None:
            self._probe_release_ts[request_id] = last_release

    def _priority_view(self, now: float) -> list[_DeferredEntry]:
        """Queue as a priority view (§5.6.2): G1 warm续轮 > G2 新会话 > G3
        cold 续轮, strict FIFO within a group; entries waiting longer than
        ``aging_seconds`` are promoted into G1 (anti-starvation)."""
        tau = self._admission.tau

        def group(entry: _DeferredEntry) -> int:
            if not entry.aged and now - entry.enqueued_at > self._admission.aging_seconds:
                entry.aged = True
                self._admission_aging_count += 1
            if entry.aged:
                return 0
            if entry.turn == 1:
                return 1  # G2: new session
            release_ts = self._probe_release_ts.get(entry.request_id)
            warm = release_ts is not None and (now - release_ts) <= tau
            return 0 if warm else 2  # G1 warm续轮 / G3 cold续轮

        return sorted(self._defer_queue, key=lambda e: (group(e), e.enqueued_at))

    def _maybe_rescan(self, replicas: list[ReplicaInfo], now: float) -> None:
        """Retry-driven rescan, throttled to ``_RESCAN_MIN_INTERVAL`` (deep-queue guard).

        Release/dispatch events rescan immediately (they are the drain
        mechanism); only the 1 s retry storms are rate-limited. A skipped
        rescan loses nothing structural — budget changes are release-driven
        and release always rescans — at most a hold-decay-only transition
        waits out the interval (≤250 ms, far under the 1 s retry period).
        """
        if now - self._last_scan_ts < self._RESCAN_MIN_INTERVAL:
            return
        self._scan_defer_queue(replicas, now)
        self._last_scan_ts = now

    def _scan_defer_queue(self, replicas: list[ReplicaInfo], now: float) -> None:
        """Pull-based dequeue (§5.6.3): try parked requests against the
        current ledger, priority order, skipping entries that don't fit —
        smaller/warmer ones may still pass (ThunderAgent-style skip).
        Admissions land in ``_admitted_ready``; the client's 1 s retry picks
        them up."""
        if not self._defer_queue or not replicas:
            return
        _cap, block_size, ledgers = self._admission_ledger(replicas, now)
        for entry in self._priority_view(now):
            own_hold = self._hold_of(entry.request_id, now)
            views = {
                led.replica_id: self._increment_for(
                    led,
                    entry.block_hashes,
                    entry.ctx_tokens,
                    own_hold,
                    entry.request_id,
                    block_size,
                )
                for led in ledgers
            }
            feasible = [v for v in views.values() if v.feasible()]
            if not feasible:
                continue  # skip: 放不下——更小/更暖的项仍可越过它出队
            sticky = next((v for v in feasible if v.replica_id == entry.sticky), None)
            target = sticky or max(feasible, key=lambda v: v.avail_admit - v.inc)
            self._defer_queue.remove(entry)
            self._admitted_ready[entry.request_id] = (target.replica_id, now, target.predicted)
            # Charge the in-scan ledger so later entries in this same pass
            # cannot double-spend the budget before the real pin lands.
            target.ledger.pin_blocks += target.predicted
            target.ledger.avail_admit -= target.inc
            self._admission_queue_admit_total += 1
            self._fire("on_deferred_admit", entry.request_id, target.replica_id)
            logger.info(
                "admission: rid=%s decision=queue-admit wait=%.1fs inc=%d target=%s",
                entry.request_id,
                now - entry.enqueued_at,
                int(round(target.inc)),
                target.replica_id,
            )

    def _defer_message(self, request_id: str, pos: int) -> str:
        """Dual-marker message (§5.6.4): the own marker drives the upgraded
        verl branch; the legacy string is the fork-diff-lost fallback (the
        legacy branch retries it in 1 s instead of letting the request die)."""
        return f"admission-defer rid={request_id} pos={pos} | No available servers in load balancer"

    def require_acquire_fields(self) -> list[str]:
        """Declare fields forwarded by verl during acquire."""
        return ["prompt_ids"]

    def require_release_fields(self) -> list[str]:
        """Declare fields forwarded by verl during release."""
        return ["request_id"]

    def clear_sticky_cache(self) -> dict:
        """Clear sticky bindings while preserving prefix checkpoints."""
        cleared = self._store.clear_sticky_bindings()
        loads = dict(self._inflight)
        logger.info("clear_sticky_cache: cleared %s binding(s); server_loads=%s", cleared, loads)
        return {"cleared_entries": cleared, "server_loads": loads}

    def get_total_inflight(self) -> int:
        """Total in-flight requests across the pool (verl #7115 drain polling)."""
        return sum(self._inflight.values())

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the pool (provider is keyed by init-time addresses, untouched here)."""
        for sid, handle in servers.items():
            self._servers[sid] = handle
            self._inflight.setdefault(sid, 0)

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers; fires ``on_servers_removed`` to invalidate sticky bindings."""
        for sid in server_ids:
            self._servers.pop(sid, None)
            self._inflight.pop(sid, None)
        if server_ids:
            self._fire("on_servers_removed", server_ids)
