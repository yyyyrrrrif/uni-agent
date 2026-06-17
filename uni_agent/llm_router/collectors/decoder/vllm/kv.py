"""VLLMKVDecoder — vLLM KV-cache event decoder.

Decodes msgpack payloads from ZMQ and returns structured update commands.
Store writes are handled by Collector via DataStore.
"""

from __future__ import annotations

import logging
from typing import Any

import msgpack

from uni_agent.llm_router.collectors.decoder.base import Decoder
from uni_agent.llm_router.collectors.decoder.vllm.kv_event import KVCacheEvent
from uni_agent.llm_router.collectors.updates import KVCacheUpdate
from uni_agent.llm_router.utils.hash import compute_hash

logger = logging.getLogger(__name__)


class VLLMKVDecoder(Decoder):
    """vLLM KV-cache decoder — msgpack payload → KVCacheUpdate.

    Parses msgpack payloads and returns structured update commands.

    Attributes:
        remote_to_local_block_hash: Mapping from vLLM remote block_hash
            to locally-computed prefix hash (str).  Used for chained
            hash computation.
        _block_size: Learned block size from first event.
    """

    def __init__(self) -> None:
        self.remote_to_local_block_hash: dict[str, str] = {}
        self._block_size: int | None = None

    def decode(self, raw_data: bytes | str, node_id: str) -> KVCacheUpdate | None:
        """Decode msgpack payload and return structured update command.

        Handles both single event (real-time) and multiple events (replay):
          - Single: [timestamp, [[tag, fields...], ...]]
          - Multiple: [[timestamp, [...]], [timestamp, [...]]]

        Args:
            raw_data: ZMQ payload bytes (msgpack-encoded).
            node_id: The endpoint that sent this payload.

        Returns:
            KVCacheUpdate with operations to apply, or None if decode failed.
        """
        # ZMQ delivers bytes; ignore string data (shouldn't happen for this decoder)
        if isinstance(raw_data, str):
            logger.debug("VLLMKVDecoder received string data, expected bytes — skipping")
            return None

        try:
            raw = msgpack.unpackb(raw_data, raw=False)

            # Determine if raw is single event or multiple events
            # Single: [timestamp, [...]] where timestamp is int
            # Multiple: [[timestamp, [...]], ...] where first element is list
            if not isinstance(raw, list) or len(raw) == 0:
                logger.warning("Unexpected msgpack format from node %s", node_id)
                return None
            event_payloads = raw if isinstance(raw[0], list) else [raw]

            # Aggregate all operations from all payloads
            add_blocks: list[str] = []
            remove_blocks: list[str] = []
            clear_all = False
            learned_block_size: int | None = None

            for payload in event_payloads:
                events = KVCacheEvent.from_raw(payload, default_node_id=node_id)

                for event in events:
                    if event.event_type == "stored":
                        result = self._process_stored(event)
                        if result is None:
                            continue
                        add_blocks.extend(result["add_blocks"])
                        if result["block_size"] is not None:
                            learned_block_size = result["block_size"]

                    elif event.event_type == "removed":
                        remove_blocks.extend(self._process_removed(event))

                    elif event.event_type == "clear":
                        clear_all = True

                    else:
                        raise ValueError(f"Unknow event.event_type {event.event_type}.")

            return KVCacheUpdate(
                node_id=node_id,
                add_blocks=add_blocks,
                remove_blocks=remove_blocks,
                clear_all=clear_all,
                block_size=learned_block_size,
            )

        except (msgpack.UnpackException, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to decode msgpack payload from node %s: %s",
                node_id,
                exc,
            )
            return None

    # ── Event processors ────────────────────────────────────────────────

    def _process_stored(self, event: KVCacheEvent) -> dict[str, Any] | None:
        """Process BlockStored: compute local hashes.

        Returns:
            Dict with "add_blocks" (list[str]) and "block_size" (int | None).
        """
        seed = 0

        if event.token_ids is None:
            logger.debug("Stored event has no token_ids — skipping")
            return None

        # Learn block_size from first event
        learned_block_size = None
        if self._block_size is None and event.block_size is not None:
            self._block_size = event.block_size
            learned_block_size = event.block_size

        local_parent_hash = seed
        if event.parent_block_hash is not None:
            local_parent_str = self.remote_to_local_block_hash.get(event.parent_block_hash)
            if local_parent_str is not None:
                local_parent_hash = int(local_parent_str)

        local_hashes: list[str] = []
        for i, block_bytes in enumerate(event.token_ids):
            if i >= len(event.block_hashes):
                break
            local_hash_int = compute_hash(
                local_parent_hash,
                block_bytes,
                seed=seed,
            )
            local_hash_str = str(local_hash_int)
            bh = event.block_hashes[i]
            self.remote_to_local_block_hash[bh] = local_hash_str
            local_hashes.append(local_hash_str)
            local_parent_hash = local_hash_int  # chain

        return {"add_blocks": local_hashes, "block_size": learned_block_size}

    def _process_removed(self, event: KVCacheEvent) -> list[str]:
        """Process BlockRemoved: convert remote hashes to local.

        Returns:
            List of local hashes to remove.
        """
        local_hashes = [
            self.remote_to_local_block_hash[bh] for bh in event.block_hashes if bh in self.remote_to_local_block_hash
        ]
        # Clean up mapping
        for bh in event.block_hashes:
            self.remote_to_local_block_hash.pop(bh, None)

        return local_hashes
