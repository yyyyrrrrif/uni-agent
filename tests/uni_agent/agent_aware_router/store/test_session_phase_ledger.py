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

"""SessionPhaseLedger unit tests — phase transitions, hold decay, pruning."""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.store.session_phase_ledger import (
    _PRUNE_MIN_INTERVAL_S,
    PHASE_ACTING,
    PHASE_REASONING,
    SessionPhaseLedger,
)

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _fresh_singleton():
    SessionPhaseLedger._instance = None
    yield
    SessionPhaseLedger._instance = None


def _ledger() -> SessionPhaseLedger:
    return SessionPhaseLedger.singleton()


def test_phase_transitions_reasoning_then_acting():
    ledger = _ledger()
    assert ledger.phase_of("r1") is None
    ledger.mark_reasoning("r1", "s0", 1000)
    assert ledger.phase_of("r1") == PHASE_REASONING
    # Reasoning sessions are on the hard inflight account, not the hold account.
    assert ledger.acting_hold_by_replica(120.0) == {}
    ledger.mark_acting("r1", "s0")
    assert ledger.phase_of("r1") == PHASE_ACTING


def test_hold_halves_after_one_half_life():
    ledger = _ledger()
    ledger.mark_reasoning("r1", "s0", 1000)
    ledger.mark_acting("r1", "s0")
    release_ts = ledger._rows["r1"].release_ts
    assert ledger.acting_hold_by_replica(120.0, now=release_ts) == {"s0": 1000.0}
    assert ledger.acting_hold_by_replica(120.0, now=release_ts + 120.0) == {"s0": pytest.approx(500.0)}
    # 5 half-lives out: 1000 × 2^(−5) = 31.25 — effectively negligible, no end signal needed.
    assert ledger.acting_hold_by_replica(120.0, now=release_ts + 600.0) == {"s0": pytest.approx(31.25)}


def test_hold_partitions_by_release_replica_and_sums():
    ledger = _ledger()
    ledger.mark_reasoning("r1", "s0", 1000)
    ledger.mark_reasoning("r2", "s0", 500)
    ledger.mark_reasoning("r3", "s1", 2000)
    ledger.mark_acting("r1", "s0")
    ledger.mark_acting("r2", "s0")
    ledger.mark_acting("r3", "s1")
    holds = ledger.acting_hold_by_replica(120.0)
    assert holds["s0"] == pytest.approx(1500.0)
    assert holds["s1"] == pytest.approx(2000.0)


def test_reacquire_moves_session_back_to_reasoning():
    ledger = _ledger()
    ledger.mark_reasoning("r1", "s0", 1000)
    ledger.mark_acting("r1", "s0")
    # Turn 2 dispatched (possibly elsewhere): the hold leaves the account at once.
    ledger.mark_reasoning("r1", "s1", 2000)
    assert ledger.acting_hold_by_replica(120.0) == {}
    ledger.mark_acting("r1", "s1")
    assert ledger.acting_hold_by_replica(120.0) == {"s1": pytest.approx(2000.0)}


def test_release_without_acquire_records_zero_ctx():
    ledger = _ledger()
    ledger.mark_acting("ghost", "s0")
    assert ledger.acting_hold_by_replica(120.0) == {"s0": 0.0}


def test_ttl_prune_drops_expired_rows():
    ledger = _ledger()
    ledger.mark_reasoning("r1", "s0", 1000)
    ledger.mark_acting("r1", "s0")
    ledger.mark_reasoning("r2", "s0", 1)
    now = ledger._rows["r2"].last_event_ts
    ledger._rows["r1"].last_event_ts = now - 4000.0  # older than the 3600s TTL
    # The mark_* calls armed the 60s prune throttle at ~now; cross it to force the sweep.
    ledger._maybe_prune(now=now + _PRUNE_MIN_INTERVAL_S)
    assert ledger.phase_of("r1") is None
    assert ledger.phase_of("r2") == PHASE_REASONING


def test_hard_cap_evicts_least_recently_active():
    ledger = SessionPhaseLedger(max_sessions=2, ttl_seconds=3600.0)
    ledger.mark_reasoning("r1", "s0", 1)
    ledger.mark_reasoning("r2", "s0", 1)
    ledger.mark_reasoning("r3", "s0", 1)  # cap=2 hit → r1 (oldest) evicted
    assert ledger.phase_of("r1") is None
    assert ledger.phase_of("r2") == PHASE_REASONING
    assert ledger.phase_of("r3") == PHASE_REASONING


def test_summary_counts_phases_gaps_and_ctx():
    ledger = _ledger()
    ledger.mark_reasoning("r1", "s0", 1000)
    ledger.mark_reasoning("r2", "s0", 500)
    ledger.mark_acting("r2", "s0")
    s = ledger.summary()
    assert s["sessions"] == 2
    assert s["reasoning"] == 1
    assert s["acting"] == 1
    assert s["acting_ctx"] == {"s0": 500}
    assert 0 <= s["gap_max_s"] < 5.0
    assert 0 <= s["gap_mean_s"] <= s["gap_max_s"]
