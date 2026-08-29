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

"""The vllm_kv_events rollout backend registers in verl's replica registry."""

import pytest

pytestmark = [pytest.mark.st, pytest.mark.cpu]

vllm = pytest.importorskip("vllm")  # noqa: F841  (server import chain needs it)


def test_registry_mount():
    import uni_agent.llm_router.server  # noqa: F401  (import side-effect registers)
    from uni_agent.llm_router.server import KV_EVENTS_ROLLOUT_NAME
    from uni_agent.llm_router.server.replica import KvEventsReplica
    from verl.workers.rollout.replica import RolloutReplicaRegistry, get_rollout_replica_class

    assert get_rollout_replica_class(KV_EVENTS_ROLLOUT_NAME) is KvEventsReplica
    assert RolloutReplicaRegistry.get(KV_EVENTS_ROLLOUT_NAME) is KvEventsReplica


def test_http_server_surface():
    from uni_agent.llm_router.server.http_server import KvEventsHttpServer
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    # Subclass contract: the port-allocation hooks and the kv-events getter
    # exist (get_rollout_config ships in verl's vLLMHttpServer instead).
    assert issubclass(KvEventsHttpServer, vLLMHttpServer)
    for name in (
        "get_kv_events_endpoints",
        "_preprocess_engine_kwargs",
        "_assign_kv_events_ports",
        "_release_kv_events_ports",
        "run_server",
        "run_headless",
    ):
        assert hasattr(KvEventsHttpServer, name), f"missing {name}"
