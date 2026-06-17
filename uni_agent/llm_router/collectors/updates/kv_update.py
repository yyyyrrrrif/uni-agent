"""KVCacheUpdate — structured update command for KVCacheStore."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCacheUpdate:
    """Structured update command for KVCacheStore.

    Attributes:
        node_id: Target endpoint identifier.
        add_blocks: Block hashes to add (empty if none).
        remove_blocks: Block hashes to remove (empty if none).
        clear_all: If True, clear all blocks for this node.
        block_size: Block size learned from this payload (None if not present).
    """

    node_id: str
    add_blocks: list[str]
    remove_blocks: list[str]
    clear_all: bool
    block_size: int | None
