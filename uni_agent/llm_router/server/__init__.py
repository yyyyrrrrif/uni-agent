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

"""kv-events-aware vLLM rollout server, mounted via verl's replica registry.

Importing this module registers the ``vllm_kv_events`` rollout backend in
verl's ``RolloutReplicaRegistry`` — the same registry-based extension point
verl-omni uses, so no verl-side code is needed. Select it with
``actor_rollout_ref.rollout.name: vllm_kv_events``; everything else
(engine kwargs, ``router_config_path``) is standard verl config.

The backend is a drop-in superset of plain ``vllm``: with no
``kv-events-config`` in engine kwargs it behaves identically, and when
``enable_kv_cache_events`` is set it additionally allocates per-replica port
blocks (see ``http_server``) and exposes ``get_kv_events_endpoints`` for the
router (``get_rollout_config`` ships in verl itself).
"""

from verl.workers.rollout.replica import RolloutReplicaRegistry

KV_EVENTS_ROLLOUT_NAME = "vllm_kv_events"


def _load_kv_events_replica():
    from uni_agent.llm_router.server.replica import KvEventsReplica

    return KvEventsReplica


RolloutReplicaRegistry.register(KV_EVENTS_ROLLOUT_NAME, _load_kv_events_replica)
