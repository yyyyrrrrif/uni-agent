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

"""Admission config: turn-phase admission gate knobs (design doc §5.7)."""

from __future__ import annotations

from dataclasses import dataclass

from .base import ConfigError, _multiline_repr

_MODES = ("off", "shadow", "enforce")


@dataclass(repr=False)
class AdmissionConfig:
    """Config for the turn-phase admission gate.

    Attributes:
        mode: ``off`` (no gate — current behavior), ``shadow`` (gate computed
            and logged as ``decision=shadow-defer`` but never intercepts), or
            ``enforce`` (hard gate; infeasible requests enter the defer queue).
        tau: Soft-claim half-life in seconds — prices ``acting_hold`` decay and
            splits the queue's warm/cold groups.
        decode_buffer: Decode token budget reserved per in-flight request.
        buffer_base: Unmodeled slack (estimation tails, event-pipe latency).
        aging_seconds: Queue wait after which an entry is promoted into the
            top priority group (anti-starvation).
        queue_ttl: Ghost cleanup bound — entries waiting longer are pruned
            (the client never retried; the episode is assumed dead).
        ready_ttl: How long a queue-admitted assignment waits in the ready
            mailbox for the client's retry to pick it up.
        enable_acting_hold: Soft-claim (acting_hold) toggle — the Q9 A/B lever.
        ledger_check_every: Sample period (in admission decisions) for the
            ``ledger-check`` cross-validation log line.
    """

    mode: str = "off"
    tau: float = 120.0
    decode_buffer: int = 512
    buffer_base: int = 512
    aging_seconds: float = 60.0
    queue_ttl: float = 600.0
    ready_ttl: float = 60.0
    enable_acting_hold: bool = True
    ledger_check_every: int = 64

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ConfigError(f"admission.mode must be one of {_MODES}, got {self.mode!r}")
        if self.tau <= 0:
            raise ConfigError(f"admission.tau must be > 0, got {self.tau}")
        if self.decode_buffer < 0:
            raise ConfigError(f"admission.decode_buffer must be >= 0, got {self.decode_buffer}")
        if self.buffer_base < 0:
            raise ConfigError(f"admission.buffer_base must be >= 0, got {self.buffer_base}")
        if self.aging_seconds <= 0:
            raise ConfigError(f"admission.aging_seconds must be > 0, got {self.aging_seconds}")
        if self.queue_ttl <= 0:
            raise ConfigError(f"admission.queue_ttl must be > 0, got {self.queue_ttl}")
        if self.ready_ttl <= 0:
            raise ConfigError(f"admission.ready_ttl must be > 0, got {self.ready_ttl}")
        if self.ledger_check_every <= 0:
            raise ConfigError(f"admission.ledger_check_every must be > 0, got {self.ledger_check_every}")

    __repr__ = _multiline_repr
