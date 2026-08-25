# llm_router Inference Quick Start

Last updated: 08/25/2026

## What is this

This example runs **mini-swe-agent** (blackbox) SWE-bench agentic inference on
verl's **KV-cache-aware router** — hosted in the `uni_agent.llm_router` package
and injected into verl by FQN (`rollout.router_config_path` + a `pkg://` router
YAML; no verl-side registration needed). Sessions run on the uni-agent
**framework + task_runner path** (`uni_agent.framework.task_runner.run_task`):
vLLM replicas sit behind the kvcaware router (KV-cache hit rate + load aware
dispatch), a gateway pool drives `mini_swe_agent` sessions in openyuanrong
remote sandboxes, and each task's reward is reported back. No trainer is started.

mini-swe-agent runs **inside** the sandbox from a prebuilt tool image (mounted
at `/opt/mini-swe-agent`) and reaches the policy gateway through a reverse tunnel
(`proxy_port` + runtime-injected `upstream`), so the sandbox cluster and the
training cluster do not need to reach each other — only the training side must
access the sandbox service (API + image pull).

Core components:
- `KVCAwareBalancer` — routing framework, manages component lifecycle and routing decisions
- `Collector` — collects vLLM KV events and Prometheus metrics for decisions
- `Strategy` — scoring strategy, combines KV cache hit rate and load
- `Store` — singleton storage for collected metrics and KV block states

The task runner / reward / dataset code is reused from the installed `uni-agent`
package (`uni_agent.tasks`, `uni_agent.sandbox`).

## Prerequisites

1. This repo (uni-agent repo) with the `verl` submodule initialized, plus
   `pip install -e .` so the `uni_agent` package (which now hosts the router)
   resolves.
2. An **OpenYuanrong** remote-sandbox endpoint
   (`OPENYUANRONG_SERVER_ADDRESS` / `OPENYUANRONG_TOKEN`) — the only sandbox
   provider with reverse-tunnel support, which the mini-swe-agent tool-image
   mount requires.
3. The **mini-swe-agent tool image**, built and pushed to a registry the sandbox
   service can pull. Build it from the sibling example:
   ```bash
   bash examples/mini_swe_agent/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
   ```
   The bundled `task_config_mini_swe_agent.yaml` mounts
   `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest` at
   `/opt/mini-swe-agent`; point the `mounts[].image_url` there if you push
   elsewhere.
4. A dataset parquet (SWE-bench verified). Point `--data-path` at any compatible
   parquet (generate with `python -m uni_agent.tasks.swe_bench.preprocess
   --local-save-dir ~/data/swe_agent`). The schema must carry
   `extra_info.tools_kwargs` with the `task` dict the task runner resolves
   (older parquets with `tools_kwargs: {env, reward}` are rejected — regenerate).

## Run

`run_infer.sh` is a thin wrapper: it exports the Ray-worker environment
(OpenYuanrong creds + observability env) and forwards all CLI flags to
`run_infer.py`. `--task-config` is required — it selects the
task / agent / model knobs per row. See the full flag list with defaults via
`--help`:

```bash
bash examples/llm_router/run_infer.sh --help

# Smoke test (1 sample, kv-events on)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_mini_swe_agent.yaml \
    --max-samples 1 --kv-events

# Full run (omit --max-samples for the whole dataset)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_mini_swe_agent.yaml \
    --tensor-parallel-size 2 --n-gpus-per-node 8 --kv-events

# With mooncake cross-replica KV sharing (mooncake master runs separately)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_mini_swe_agent.yaml \
    --enable-mooncake --kv-events

# Ascend (vllm-ascend) backend
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_mini_swe_agent.yaml \
    --device ascend --enable-mooncake
```

Key CLI flags (see `--help` for the complete list with defaults):

| Flag | Default | Description |
|------|---------|-------------|
| `--data-path` | `~/data/swe_agent/swe_bench_verified.parquet` | Dataset parquet |
| `--model-path` / `--model` | `~/models/Qwen3-Coder-30B-A3B-Instruct` | Model path |
| `--task-config` | (required) | YAML task config (`- name:` entries) driving each row |
| `--max-samples` / `--limit` | all | Samples to run (omit for the full dataset) |
| `--n` | `1` | Rollout sessions per instance |
| `--shuffle` / `--seed` | off / `42` | Shuffle before sampling, with a reproducible seed |
| `--prompt-length` / `--response-length` | `4096` / `8192` | Token lengths |
| `--max-model-len` | engine clamp | vLLM maximum context length |
| `--tensor-parallel-size` / `--n-gpus-per-node` / `--nnodes` | `4` / `8` / `1` | Parallelism |
| `--concurrency` | env `GLOBAL_CONCURRENCY` | Max in-flight gateway sessions |
| `--gateway-count` / `--tool-parser` | `4` / `qwen3_coder` | Gateway pool / tool-call parser |
| `--gpu-memory-utilization` | `0.9` | vLLM GPU memory utilization (0-1) |
| `--router-config-path` | `pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml` | Packaged router YAML |
| `--max-num-seqs` | `256` | Max concurrent sequences per engine |
| `--kv-events` | off | Enable vLLM kv-events (kvcaware load signal) |
| `--enable-mooncake` / `--mooncake-config-path` | off / `mooncake_config.json` | Cross-replica KV sharing |
| `--device` | `gpu` | `gpu` or `ascend` (selects mooncake connector class) |
| `--alpha` / `--load-threshold` / `--slow-cut` / `--overload-mode` / `--do-shortcut` | from `kvcaware.yaml` | kvcaware strategy[0] overrides |

## Environment variables

Only variables read via `os.environ` inside Ray-spawned workers (not CLI
flags) — `run_infer.sh` exports them; set them in the shell before invoking:

| Var | Default | Description |
|-----|---------|-------------|
| `OPENYUANRONG_SERVER_ADDRESS` / `OPENYUANRONG_TOKEN` | empty | OpenYuanrong remote sandbox auth |
| `OPENYUANRONG_TUNNEL_SSL_VERIFY` | `0` | Sandbox reverse-tunnel TLS verify (0 = disabled) |
| `SANDBOX_NAME_PREFIX` | `mini-swe-` | Prefix for created sandbox names |
| `VERL_LOGGING_LEVEL` | `INFO` | verl logging level |
| `RL_INSIGHT_SERVER_URL` | `http://127.0.0.1:18080` | rl-insight observability server |

## Observability

The run log carries the router's dispatch evidence — `routed to server` lines
from the balancer, `score(): COMBINED` strategy logs, and kv-events collector
metrics. At the end, `run_infer.py` prints an `inference summary`
block with the mean rm_score and a per-session breakdown, and writes a JSON
result file when `--result-path` is set.

## Experiment matrices

Two drivers sweep a sticky-vs-kvcaware matrix over `(concurrency × context)`,
retrying each run until the `inference summary` success sentinel lands in its
log. Both hardcode `--device ascend` (vllm-ascend) and `--kv-events`; the
kvcaware cells vary `--load-threshold` over `0.1..0.9`.

- `ascend-exps.sh` — single node (16 NPU, TP=4 → 4 replicas). Edit the `MODEL` /
  `DATASET` / `MAX_SAMPLES` vars at the top, then `bash examples/llm_router/ascend-exps.sh`.
- `multi-node-ascend-exps.sh` — 6 nodes (this host = Ray head + 5
  passwordless-SSH workers, each entered via `docker exec hgq-verl-ascend`;
  48 NPU / TP=4 → 12 replicas). It brings the Ray cluster up and tears it down
  per attempt. Edit the `WORKERS[]` host list and `MODEL` / `DATASET` vars first.

`run_infer.sh` is the single underlying entry point; both drivers just loop it.

## Notes

- The task runner requires an OpenYuanrong remote sandbox; without it sessions
  fail fast. The reverse tunnel (`proxy_port` + runtime `upstream`) is supported
  only on the openyuanrong provider — configuring `proxy_port` on any other
  provider is rejected loudly.
- Build the tool image (`examples/mini_swe_agent/build_tool.sh`) and push it to
  a registry the sandbox service can pull before the first run; the task config
  mounts it at `/opt/mini-swe-agent`.
- `agent.step_limit` caps the number of mini-swe-agent turns; `agent.run_timeout`
  caps the in-sandbox agent process (s). Keep `--max-model-len` comfortably
  above the per-episode token budget so vLLM does not reject long transcripts.
- Dataset schema must carry `extra_info.tools_kwargs` with the `task` dict the
  task runner resolves (uni-agent format). If your parquet predates it,
  regenerate with `python -m uni_agent.tasks.swe_bench.preprocess`.
