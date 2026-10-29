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

"""SessionPhaseLedger — per-session phase probe (generation vs tool-call).

The balancer's acquire/release stream is enough to classify every session:

- ``reasoning``: acquire → release — a generation request is in flight; its
  occupancy is on the hard ``INFLIGHT_BLOCKS`` account.
- ``acting``: release → next acquire — the agent is executing its tool call;
  no request is in flight, but the session's KV cache still claims space on
  the replica it last ran on.

``acting_hold_by_replica`` prices those still-claimed caches as
``ctx_tokens × 2^(−gap/τ)`` — the soft-commitment term the capacity account
subtracts from free tokens (docs/design/turn-phase-admission.md §5.2/§5.3).
Written at event time, decayed at read time, so the hold is exact at the
instant a routing decision consults it. A session that never returns (episode
end, agent crash) needs no end-of-session signal: its hold decays to
negligible within ~10τ and its row is TTL-pruned long after. Singleton, like
the other stores; tests reset ``_instance``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Session phases (docs/design/turn-phase-admission.md §3).
PHASE_REASONING = "reasoning"
PHASE_ACTING = "acting"

# Ledger bounds — TTL-pruned lazily on writes, hard-capped by last activity.
DEFAULT_MAX_SESSIONS = 8192
DEFAULT_TTL_SECONDS = 3600.0
# Minimum spacing between TTL sweeps (a sweep walks the whole table).
_PRUNE_MIN_INTERVAL_S = 60.0


@dataclass
class _SessionRow:
    """One session's phase state (the probe's per-request ledger entry)."""

    phase: str
    replica_id: str
    ctx_tokens: int
    release_ts: float  # monotonic anchor of the acting phase (0.0 while reasoning)
    last_event_ts: float  # monotonic; TTL-prune anchor


class SessionPhaseLedger:
    """``request_id → phase`` ledger rebuilt purely from acquire/release events.

    Acquire marks a session reasoning on its replica; release flips it to
    acting and anchors the release time — the acting gap that decays the
    session's hold starts there. Re-acquire without an intervening release
    (a dropped release event) simply overwrites the row.
    """

    _instance: SessionPhaseLedger | None = None

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        if max_sessions <= 0:
            raise ValueError(f"max_sessions must be > 0, got {max_sessions}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        self._max_sessions = int(max_sessions)
        self._ttl_seconds = float(ttl_seconds)
        self._lock = threading.Lock()
        self._rows: dict[str, _SessionRow] = {}
        self._last_prune_ts = 0.0
        logger.info(f"SessionPhaseLedger created: max_sessions={self._max_sessions}, ttl_seconds={self._ttl_seconds}")

    @classmethod
    def singleton(cls) -> SessionPhaseLedger:
        """Return the shared singleton (tests construct a fresh instance / reset _instance)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def size(self) -> int:
        """Number of tracked sessions."""
        with self._lock:
            return len(self._rows)

    def mark_reasoning(self, request_id: str, replica_id: str, ctx_tokens: int) -> None:
        """Mark ``request_id`` reasoning: a generation request acquired on ``replica_id``.

        ``ctx_tokens`` (the acquire's prompt length) becomes the cache claim
        the session holds while acting after its release.
        """
        now = time.monotonic()
        with self._lock:
            self._maybe_prune(now)
            self._rows[request_id] = _SessionRow(
                phase=PHASE_REASONING,
                replica_id=replica_id,
                ctx_tokens=max(0, int(ctx_tokens)),
                release_ts=0.0,
                last_event_ts=now,
            )

    def mark_acting(self, request_id: str, replica_id: str) -> None:
        """Mark ``request_id`` acting: released on ``replica_id``, tool-call phase begins.

        A release with no recorded acquire (should not happen — acquire always
        precedes release) is recorded with ctx 0 so the anomaly is visible in
        the account rather than silently dropped.
        """
        now = time.monotonic()
        with self._lock:
            self._maybe_prune(now)
            row = self._rows.get(request_id)
            if row is None:
                logger.debug(f"release with no recorded acquire request={request_id} — recording with ctx=0")
                self._rows[request_id] = _SessionRow(
                    phase=PHASE_ACTING,
                    replica_id=replica_id,
                    ctx_tokens=0,
                    release_ts=now,
                    last_event_ts=now,
                )
                return
            row.phase = PHASE_ACTING
            row.replica_id = replica_id
            row.release_ts = now
            row.last_event_ts = now

    def acting_hold_by_replica(self, half_life_s: float, now: float | None = None) -> dict[str, float]:
        """Soft commitment per replica: ``Σ ctx × 2^(−gap/τ)`` over acting sessions.

        Sessions still reasoning contribute nothing — their occupancy is
        already on the hard ``INFLIGHT_BLOCKS`` account. ``half_life_s`` is τ;
        a non-positive τ disables decay (full-weight holds).
        """
        if now is None:
            now = time.monotonic()
        holds: dict[str, float] = {}
        with self._lock:
            for row in self._rows.values():
                if row.phase != PHASE_ACTING:
                    continue
                gap = max(0.0, now - row.release_ts)
                factor = 1.0 if half_life_s <= 0 else 2.0 ** (-gap / half_life_s)
                holds[row.replica_id] = holds.get(row.replica_id, 0.0) + row.ctx_tokens * factor
        return holds

    def phase_of(self, request_id: str) -> str | None:
        """Current phase of ``request_id`` (``None`` if untracked)."""
        with self._lock:
            row = self._rows.get(request_id)
            return None if row is None else row.phase

    def summary(self) -> dict[str, Any]:
        """Phase counts, acting-gap stats, and per-replica acting ctx sums.

        Raw material for the periodic ``session-ledger:`` log — dead sessions
        surface here as a growing acting population with old gaps (their
        decaying hold is the live ``acting_hold_by_replica`` reading).
        """
        now = time.monotonic()
        with self._lock:
            reasoning = 0
            gaps: list[float] = []
            acting_ctx: dict[str, int] = {}
            for row in self._rows.values():
                if row.phase == PHASE_REASONING:
                    reasoning += 1
                else:
                    gaps.append(now - row.release_ts)
                    acting_ctx[row.replica_id] = acting_ctx.get(row.replica_id, 0) + row.ctx_tokens
            return {
                "sessions": len(self._rows),
                "reasoning": reasoning,
                "acting": len(gaps),
                "gap_mean_s": sum(gaps) / len(gaps) if gaps else 0.0,
                "gap_max_s": max(gaps) if gaps else 0.0,
                "acting_ctx": acting_ctx,
            }

    def _maybe_prune(self, now: float) -> None:
        """Drop TTL-expired rows (caller holds the lock); hard-cap by last activity."""
        if now - self._last_prune_ts < _PRUNE_MIN_INTERVAL_S and len(self._rows) < self._max_sessions:
            return
        self._last_prune_ts = now
        expired = [rid for rid, row in self._rows.items() if now - row.last_event_ts > self._ttl_seconds]
        for rid in expired:
            del self._rows[rid]
        if len(self._rows) >= self._max_sessions:
            by_age = sorted(self._rows.items(), key=lambda item: item[1].last_event_ts)
            for rid, _ in by_age[: len(self._rows) - self._max_sessions + 1]:
                del self._rows[rid]

    def reset(self) -> None:
        """Clear all ledger state (test helper; not used on the hot path)."""
        with self._lock:
            self._rows.clear()
