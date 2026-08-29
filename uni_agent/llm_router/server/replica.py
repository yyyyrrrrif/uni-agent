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

"""KvEventsReplica: launches KvEventsHttpServer actors instead of vLLMHttpServer."""

import ray

from uni_agent.llm_router.server.http_server import KvEventsHttpServer
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica


class KvEventsReplica(vLLMReplica):
    """vLLMReplica whose per-node actors are KvEventsHttpServer instances."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(KvEventsHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "uni_agent_"
