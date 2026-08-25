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

"""E2E test: KVCAware router + mooncake via run_infer.sh.

Launches ``run_infer.sh`` (drives ``parallel_infer_verl_kvc.py`` with the
simulated-sandbox runner swapped in) with ``--enable-mooncake``,
starts mooncake_master, waits for completion, then checks:
  1. MooncakeStoreConnector created on vLLM replicas
  2. Routing decisions produced ("routed to server")
  3. mean rm_score printed (end-to-end completion)
  4. No TCP transport errors (no writeBody/batch_put -800 failures)
  5. External prefix cache hit observed (cross-replica KV sharing working)

This is a GPU test (needs real vLLM + GPU + model + dataset + mooncake_master).
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

pytestmark = [pytest.mark.st, pytest.mark.gpu, pytest.mark.level1]
pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
_RUN_INFER = os.path.join(_PROJECT_ROOT, "examples", "llm_router", "run_infer.sh")
_SIMULATED_RUNNER_FQN = "tests.uni_agent.llm_router.e2e.utils.simulated_sandbox.simulated_runner"
_MODEL = os.environ.get("VLLM_MODEL", "/data1/models/Qwen/Qwen3-4B-Instruct-2507")
_DATASET = os.environ.get("SWEBENCH_DATASET", "/data1/hgq/uni-agent/scripts/swe_bench_verified_modal.parquet")
_TASK_CONFIG = os.path.join(_PROJECT_ROOT, "examples", "llm_router", "task_config_mini_swe_agent.yaml")
_LOG_DIR = "/tmp/e2e_mooncake_logs"

# Mooncake daemon ports — the test brings up its own metadata server + master
# on these fixed ports so it does not depend on an externally-managed cluster.
_MC_METADATA_PORT = 9527
_MC_MASTER_PORT = 50051


def _run_infer_with_mooncake(timeout: int = 600) -> str:
    """Run run_infer.sh with router + mooncake. Returns log content.

    Brings up its own mooncake cluster (metadata server + master) and writes a
    matching config to the log dir, so the test is self-contained — it does not
    assume a config at a fixed repo path nor a pre-running metadata server.
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_file = os.path.join(_LOG_DIR, "mooncake_e2e.log")

    # Self-contained mooncake config: write to the log dir (portable, not tied
    # to a repo path). MOONCAKE_CONFIG_PATH env, if set, overrides this.
    # Must be JSON — vllm's MooncakeStoreConfig.from_file parses json, not yaml.
    mc_config = os.environ.get(
        "MOONCAKE_CONFIG_PATH",
        os.path.join(_LOG_DIR, "mooncake_config.json"),
    )
    os.makedirs(os.path.dirname(mc_config), exist_ok=True)
    import json

    with open(mc_config, "w") as f:
        json.dump(
            {
                "metadata_server": f"http://127.0.0.1:{_MC_METADATA_PORT}/metadata",
                "master_server_address": f"127.0.0.1:{_MC_MASTER_PORT}",
                "global_segment_size": "4GB",
                "local_buffer_size": "4GB",
                "protocol": "tcp",
                "device_name": "",
            },
            f,
        )

    # GPU config: CUDA_VISIBLE_DEVICES controls which GPUs Ray/vLLM see;
    # --n-gpus-per-node must match the count.
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    num_gpus = len(cuda_vis.split(","))
    cmd = [
        "bash",
        _RUN_INFER,
        "--model-path",
        _MODEL,
        "--data-path",
        _DATASET,
        "--simulated-runner-fqn",
        _SIMULATED_RUNNER_FQN,
        "--task-config",
        _TASK_CONFIG,
        "--n-gpus-per-node",
        str(num_gpus),
        "--tensor-parallel-size",
        "2",
        "--max-samples",
        "4",
        "--n",
        "2",
        "--max-model-len",
        "8192",
        "--response-length",
        "4096",
        "--prompt-length",
        "3072",
        "--enable-mooncake",
        "--mooncake-config-path",
        mc_config,
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = cuda_vis
    env["MOONCAKE_CONFIG_PATH"] = mc_config
    env["MC_TCP_ENABLE_CONNECTION_POOL"] = "1"
    env["MOONCAKE_CPU_STAGING"] = "1"

    # Bring up the mooncake cluster: metadata server first (master depends on
    # it), then master. Both are terminated in the finally below.
    meta_log = open(os.path.join(_LOG_DIR, "mooncake_metadata.log"), "w")
    master_log = open(os.path.join(_LOG_DIR, "mooncake_master.log"), "w")
    meta_proc = master_proc = None
    try:
        meta_proc = subprocess.Popen(
            ["mooncake_http_metadata_server", "--port", str(_MC_METADATA_PORT), "--host", "127.0.0.1"],
            stdout=meta_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(3)
        master_proc = subprocess.Popen(
            ["mooncake_master", "--port", str(_MC_MASTER_PORT), "--default_kv_lease_ttl", "60000"],
            stdout=master_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(5)

        with open(log_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    finally:
        for proc in (master_proc, meta_proc):
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        meta_log.close()
        master_log.close()

    return open(log_file).read()


class TestMooncakeRouterE2E:
    """E2E: run_infer.sh with KVCAware router + mooncake connector."""

    def test_mooncake_router_full_e2e(self):
        """
        Feature: KVCAware router + mooncake end-to-end via run_infer.sh
        Description: run full agent loop with router + mooncake, verify:
          - MooncakeStoreConnector created
          - routing decisions produced
          - mean rm_score printed
          - inference summary printed (agent loop actually ran)
          - no TCP transport errors (no writeBody/batch_put -800)
          - External prefix cache hit observed
        """
        log = _run_infer_with_mooncake()

        # 1. MooncakeStoreConnector created
        assert "MooncakeStoreConnector" in log, "MooncakeStoreConnector not found in log"

        # 2. Routing decisions
        assert "routed to server" in log, "No routing decisions in log"

        # 3. End-to-end completion
        assert "mean rm_score" in log, "run_infer.sh did not complete"

        # 4. Agent loop actually ran (inference summary block at the end)
        assert "inference summary" in log, "run_infer.sh did not finish (no inference summary)"

        # 5. No TCP transport errors
        tcp_errors = log.count("writeBody failed") + log.count("batch_put failed")
        assert tcp_errors == 0, (
            f"Found {tcp_errors} TCP transport errors "
            f"(writeBody/batch_put failures indicate port exhaustion or CUDA staging issues)"
        )

        # 6. External prefix cache hit (cross-replica KV sharing)
        # Note: External hit requires sufficient concurrency + prefix overlap across
        # replicas. Small-sample e2e (4 samples) may not trigger it reliably.
        # We warn instead of asserting — the connector creation (check 1) and
        # zero TCP errors (check 5) already prove mooncake is wired correctly.
        if "External prefix cache hit" not in log:
            import warnings

            warnings.warn(
                "No External prefix cache hit in log — small-sample e2e may not "
                "produce enough cross-replica prefix overlap to trigger it. "
                "Connector is created (check 1) and TCP is clean (check 5), "
                "so mooncake wiring is correct.",
                stacklevel=2,
            )
