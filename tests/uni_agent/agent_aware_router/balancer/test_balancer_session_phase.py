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

"""Session-phase probe through the real callback collectors (acquire/release → ledger)."""

from __future__ import annotations

import logging

import pytest

from uni_agent.agent_aware_router.store.session_phase_ledger import (
    PHASE_ACTING,
    PHASE_REASONING,
    SessionPhaseLedger,
)

from ._helpers import _make_balancer

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


def test_acquire_marks_reasoning_and_release_marks_acting():
    balancer = _make_balancer({"s0": "h0", "s1": "h1"})
    ledger = SessionPhaseLedger.singleton()

    server_id, _ = balancer.acquire_server("request-1", [1, 2, 3])
    assert ledger.phase_of("request-1") == PHASE_REASONING
    assert ledger.acting_hold_by_replica(120.0) == {}

    balancer.release_server(server_id, request_id="request-1")
    assert ledger.phase_of("request-1") == PHASE_ACTING
    # Released on server_id with ctx=3 and gap≈0 → the full claim sits there.
    assert ledger.acting_hold_by_replica(120.0) == {server_id: pytest.approx(3.0, rel=1e-3)}


def test_reacquire_returns_hold_to_zero():
    balancer = _make_balancer({"s0": "h0"})
    ledger = SessionPhaseLedger.singleton()
    server_id, _ = balancer.acquire_server("request-1", [1, 2, 3])
    balancer.release_server(server_id, request_id="request-1")
    assert ledger.acting_hold_by_replica(120.0)

    balancer.acquire_server("request-1", [1, 2, 3])  # turn 2: sticky rebinds → same replica
    assert ledger.phase_of("request-1") == PHASE_REASONING
    assert ledger.acting_hold_by_replica(120.0) == {}


def test_release_without_acquire_records_zero_ctx():
    balancer = _make_balancer({"s0": "h0"})
    ledger = SessionPhaseLedger.singleton()
    balancer.release_server("s0", request_id="ghost")
    assert ledger.acting_hold_by_replica(120.0) == {"s0": 0.0}


def test_session_phase_log_lines(caplog):
    balancer = _make_balancer({"s0": "h0"})
    with caplog.at_level(logging.INFO):
        server_id, _ = balancer.acquire_server("request-1", [1, 2, 3])
        balancer.release_server(server_id, request_id="request-1")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("session-phase:")]
    assert any("rid=request-1" in m and "phase=reasoning" in m and "ctx=3" in m for m in lines)
    assert any("rid=request-1" in m and "phase=acting" in m and "ctx=3" in m for m in lines)
