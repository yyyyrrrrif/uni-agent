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

"""CPU unit tests for the in-flight *held block* ref-count table (``pin``/``unpin``).

The table answers "how many blocks do the in-flight requests pin on this
replica" — the free-capacity account (``avail = cap − inflight_blocks × block_size``)
that replaced the Σ-new-allocation ``inflight_tokens`` sum.
"""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types import Layer

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

PREFIX = ["h1", "h2", "h3", "h4", "h5"]


def _store() -> KVCacheStore:
    """A fresh non-singleton store."""
    return KVCacheStore()


def test_pin_counts_new_blocks_then_ref_counts_shared_ones() -> None:
    """Two in-flight requests sharing a prefix hold the union once, not the sum.

    Feature: ``|held|`` de-duplicates by block hash and only drops a block when
      its last holder unpins.
    Description: R pins 5 blocks; D pins the first 3 of the same hashes.
    Expectation: pin returns 5 then 0; ``|held|`` = 5 throughout; releasing D
      frees nothing (R still holds all 5); releasing R empties the table.
    """
    store = _store()

    assert store.pin_inflight_blocks("s0", PREFIX) == 5  # all five are new
    assert store.inflight_block_count("s0") == 5

    assert store.pin_inflight_blocks("s0", PREFIX[:3]) == 0  # already held by R
    assert store.inflight_block_count("s0") == 5  # not 8: shared blocks count once

    store.unpin_inflight_blocks("s0", PREFIX[:3])
    assert store.inflight_block_count("s0") == 5  # D's blocks are still held by R

    store.unpin_inflight_blocks("s0", PREFIX)
    assert store.inflight_block_count("s0") == 0


def test_pin_excludes_blocks_already_resident_in_the_prefix_cache() -> None:
    """A block that is already cached is not a new allocation (but is still held).

    Feature: the return value is "blocks that need allocating", i.e. neither
      ``cached_on`` nor ``held``.
    Description: s0 caches h1/h2, then two requests pin [h1..h3].
    Expectation: the first pin reports 1 new block; the second reports 0 (h3 is
      now held by the first request even though no BlockStored event arrived);
      ``|held|`` = 3 both times.
    """
    store = _store()
    store.add_blocks("s0", ["h1", "h2"], layer=Layer.GPU)

    assert store.pin_inflight_blocks("s0", ["h1", "h2", "h3"]) == 1
    assert store.inflight_block_count("s0") == 3
    assert store.pin_inflight_blocks("s0", ["h1", "h2", "h3"]) == 0


def test_held_blocks_are_per_replica() -> None:
    """Pins are per replica: the same hash held on s0 is still new on s1."""
    store = _store()
    assert store.pin_inflight_blocks("s0", PREFIX) == 5

    assert store.pin_inflight_blocks("s1", PREFIX) == 5  # s1 holds nothing yet
    assert store.inflight_block_count("s0") == 5
    assert store.inflight_block_count("s1") == 5

    store.unpin_inflight_blocks("s0", PREFIX)
    assert store.inflight_block_count("s0") == 0
    assert store.inflight_block_count("s1") == 5  # releasing s0 must not touch s1


def test_unpin_unknown_blocks_is_a_noop_and_never_goes_negative() -> None:
    """Unknown hashes are ignored; a doubled unpin leaves an empty table."""
    store = _store()
    store.pin_inflight_blocks("s0", ["h1"])

    store.unpin_inflight_blocks("s0", ["never-pinned", "h1", "h1"])
    assert store.inflight_block_count("s0") == 0
    assert store.pin_inflight_blocks("s0", ["h1"]) == 1  # h1 was really released

    store.unpin_inflight_blocks("ghost-replica", ["h1"])  # replica with no pin table
    assert store.inflight_block_count("ghost-replica") == 0


def test_empty_hash_list_pins_nothing() -> None:
    """No resolvable prompt (no prompt_ids / unknown block size) pins nothing."""
    store = _store()
    assert store.pin_inflight_blocks("s0", []) == 0
    assert store.inflight_block_count("s0") == 0


def test_clear_replica_drops_its_pins() -> None:
    """Removing a replica must not leave phantom in-flight load behind."""
    store = _store()
    store.pin_inflight_blocks("s0", PREFIX)
    store.pin_inflight_blocks("s1", PREFIX)

    store.clear_replica("s0")

    assert store.inflight_block_count("s0") == 0
    assert store.inflight_block_count("s1") == 5


def test_pin_feeds_exactly_the_allocating_side_of_a_shared_dispatch() -> None:
    """Concurrent same-prefix dispatch is not double-counted (the v1 blind spot).

    Feature: with the prefix *not* yet resident, the second dispatch of an
      identical prompt allocates nothing new.
    Description: two 5-block requests on a cold replica.
    Expectation: Σ ``pin`` = 5 (v1's ``plen × (1 − hit)`` would have summed 10
      blocks: neither request hits the cache, so both booked their whole prompt).
    """
    store = _store()
    first = store.pin_inflight_blocks("s0", PREFIX)
    second = store.pin_inflight_blocks("s0", PREFIX)
    assert first + second == 5
    assert store.inflight_block_count("s0") == 5
