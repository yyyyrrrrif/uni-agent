"""Tests for CollectorProvider dynamic add_servers / remove_servers against real vLLM.

Test flow:
1. Launch the first vLLM server (session-shared, ``vllm_kv_service``).
2. Construct a ``CollectorProvider`` that knows ONLY the first server —
   mirroring how the balancer wires it at construction.
3. Launch a second vLLM server (``vllm_kv_service_2``) on distinct ports.
4. ``provider.add_servers(...)`` — feed server #2's endpoints in AFTER
   construction; verify the provider starts collecting its KV events.
5. ``provider.remove_servers(...)`` + ``store.remove_servers(...)`` — stop
   collecting from server #2 and clear its data; verify it's gone from the
   store while server #1's data is untouched.

This is the integration counterpart to ``balancer.test_b20~b22``: those
assert the *wiring* (provider/store got the call) with fakes; this asserts
the *behavior* (real collection starts/stops) against live vLLM.
"""

from __future__ import annotations

import time

import pytest
from conftest import (
    NODE_ID,
    NODE_ID_2,
    VLLM_MODEL,
    ZMQ_REPLAY_PORT,
    ZMQ_REPLAY_PORT_2,
    ZMQ_SUB_PORT,
    ZMQ_SUB_PORT_2,
    send_inference_request,
)

from uni_agent.llm_router.collectors.provider import CollectorProvider
from uni_agent.llm_router.config.collector import CollectorConfig
from uni_agent.llm_router.store.data_store import DataStore


def _kv_endpoint(node_id: str, sub_port: int, replay_port: int) -> dict[str, list[str]]:
    """Build the ZMQ 4-element endpoint dict for one node."""
    return {node_id: [f"127.0.0.1:{sub_port}", f"127.0.0.1:{replay_port}", "zmq", "kv-events"]}


def _wait_for_blocks(store: DataStore, node_id: str, timeout: float = 30.0) -> bool:
    """Poll until ``node_id`` has at least one cached block (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.kv_node_has_blocks(node_id):
            return True
        time.sleep(1.0)
    return False


@pytest.mark.st
@pytest.mark.gpu
class TestDynamicAddRemoveServersWithRealService:
    """add_servers starts real collection; remove_servers stops it + clears data."""

    def test_add_then_remove_server_2(self, vllm_kv_service, vllm_kv_service_2):
        """
        Feature: provider.add_servers starts collecting a server added after construction;
                 provider.remove_servers stops it; store.remove_servers clears its data
        Expectation:
            - After add_servers(server #2) + an inference request, server #2 has cached
              blocks (collection started for a server the provider didn't know at build).
            - server #1 (known at construction) also accumulates blocks.
            - After remove_servers(server #2) + store.remove_servers([server #2]),
              server #2 is gone from the store while server #1 is untouched.
        """
        store = DataStore()

        # Provider constructed with ONLY server #1 — server #2 is unknown.
        provider = CollectorProvider(
            CollectorConfig(),
            ["vllm_zmq"],
            kv_event_endpoints=_kv_endpoint(NODE_ID, ZMQ_SUB_PORT, ZMQ_REPLAY_PORT),
        )
        provider.start()
        try:
            time.sleep(3.0)  # let server #1's sub task connect

            # ── add server #2 after construction ─────────────────────────
            provider.add_servers(
                server_addresses={},
                kv_event_endpoints=_kv_endpoint(NODE_ID_2, ZMQ_SUB_PORT_2, ZMQ_REPLAY_PORT_2),
            )
            time.sleep(5.0)  # let the new sub task connect + replay

            # Trigger KV events on both servers.
            send_inference_request(vllm_kv_service, VLLM_MODEL, "hello from server one")
            send_inference_request(vllm_kv_service_2, VLLM_MODEL, "hello from server two")
            time.sleep(5.0)

            assert _wait_for_blocks(store, NODE_ID_2), (
                f"server #2 '{NODE_ID_2}' should have cached blocks after add_servers + inference"
            )
            assert _wait_for_blocks(store, NODE_ID), (
                f"server #1 '{NODE_ID}' should have cached blocks (known at construction)"
            )

            # ── remove server #2 ──────────────────────────────────────────
            provider.remove_servers([NODE_ID_2])
            store.remove_servers([NODE_ID_2])
            time.sleep(2.0)

            # server #2's kv data cleared.
            assert not store.kv_node_has_blocks(NODE_ID_2), (
                f"server #2 '{NODE_ID_2}' should have no cached blocks after remove_servers"
            )
            # server #1 untouched by the removal.
            assert store.kv_node_has_blocks(NODE_ID), (
                f"server #1 '{NODE_ID}' data must survive removal of server #2"
            )
        finally:
            provider.stop()
