# AgentAware Router

## 1. Motivation

### 1.1 Workload characteristics of agentic RL rollout

The rollout phase of agentic RL differs substantially from traditional single-turn RL rollout in workload shape:

- **Multi-turn dialogue**: One sample maps to one agent session, often spanning hundreds of tool-calls per session, with a single prefill frequently reaching tens of thousands of tokens. Turn counts vary widely across sessions — some converge in a few turns, others take dozens or hundreds.
- **Strong prefix reuse**: Multiple requests within the same session share a continuously growing prefix history. If each turn can hit the KV-cache retained from the previous turn, large amounts of prefill recompute are skipped; otherwise, per-turn recompute causes prefill cost to accumulate linearly.
- **KV-cache eviction**: Sessions are long-lived, and their growing prefix KV is only valuable if it stays on the original replica. But replica KV capacity is limited — a single session's working set can consume more than half of it. The engine evicts older prefix blocks to make room for new requests, the next turn misses, and prefill must be recomputed — a vicious cycle of "fill → evict → miss → recompute."
- **GPU bubbles**: Because the average request context length is large and the distribution is wide, there are typically many GPU bubbles within a batch or across replicas, further lowering MFU and degrading end-to-end performance.

### 1.2 Design goals

- **Prefix-hit priority**: Subsequent turns of a session return to the original replica and hit the retained KV whenever possible, breaking the evict → miss → recompute cycle.
- **Migration triggered only on genuine overload**: Unbind and migrate only when a replica is truly overloaded — the prefill-recompute cost paid for shedding load must be justified by an explicit criterion.
- **Reduce GPU bubbles**: Through scheduling and multi-level KV-cache management, effectively alleviate the GPU-bubble problem.

---

## 2. Agent Aware Router in Detail

### 2.1 Overall architecture

`kvcaware` is a router with pluggable strategies, registered in `LoadBalancerRegistry` under the name `"kvcaware"` and wired in on the `LLMServerManager` side via `get_router_handle`. Core components:

```
KVCAwareBalancer (balancer.py)       ← pure shell: lifecycle + route() delegation
   ├── strategies/                   ← scoring algorithms (core)
   │    ├── kvc_aware.py             ← KVCacheAwareStrategy (main strategy)
   │    ├── routing.py               ← route() weighted ranking
   │    ├── base.py                  ← ReplicaInfo
   │    └── registry.py              ← config-type → runtime-strategy-class registry
   ├── store/                        ← singleton store
   │    ├── data_store.py            ← metrics + sticky binding + active_sessions
   │    ├── kv_cache_store.py        ← prefix-hash chain + three-layer hit rate
   │    ├── per_replica_store.py / per_request_store.py
   ├── collectors/                   ← collect vLLM signals
   │    ├── manager.py               ← collector lifecycle
   │    ├── decoder/vllm/metrics.py  ← /metrics polling → dict
   │    ├── decoder/vllm/kv.py, kv_event.py  ← kv-events zmq → incremental block state
   │    └── transport/{http,zmq,callback}.py
   ├── config/                       ← KVCAwareConfig + StrategyConfig
   ├── types/                        ← SlowCut / OverloadMode / Layer / MetricKey enums
   └── insight/emitter.py            ← push to rl-insight → Prometheus → Grafana
```

### 2.2 Routing decision flow: `score()`

On each `acquire_server(request_id, prompt_ids)`, the `Balancer` calls `route()`, which invokes each strategy's `score()` for a weighted ranking and takes `ranking[0]`. The decision tree of `KVCacheAwareStrategy.score()`:

```
1. STICKY short-circuit (do_shortcut=True)
   ├── request_id present and a sticky-bound replica exists
   │    └── not overloaded? → HIT, return [STICKY_TOP_SCORE, 0, ...] (preserve prefix locality)
   │        overloaded? → fallback
   └── else → enter slow-cut
2. SLOW-CUT (fallback scoring after the short-circuit misses)
   └── CAPACITY_TOKEN_AWARE  → capacity gate + token increment, pick max remaining
```

### 2.3 CAPACITY_TOKEN_AWARE scoring

CAPACITY_TOKEN_AWARE "picks the replica with the highest prefill hit rate among those that still have spare capacity." The computation is:

```
avail[i]     = cap × (1 - kv_cache_usage_perc[i])                   # pure available tokens
need[i]      = len(prompt_ids) × (1 - gpu_hit[i])                   # tokens to recompute at replica i
remaining[i] = avail[i] - need[i]                                   # remaining after assignment
eligible[i]  = avail[i] >= cap × (1 - capacity_reserve_threshold)   # pure capacity hard gate
pick         = argmax(eligible, remaining)                          # among eligible, pick max remaining
```

- Prefill hit rate participates via the `need` term: a replica with a higher hit rate needs fewer recomputed tokens, so `remaining` is larger and it wins among eligible replicas.
- Load balancing first filters via the pure-capacity hard gate `eligible` — replicas that are full are filtered out and no longer assigned. Then ranking participates via the `avail` term: a replica with higher `avail` has lower load and thus larger `remaining`.

---

## 3. Experiments

This example shows how to run SWE-bench agentic inference on the verl backend using the **KV-cache-aware router** (`kvcaware`) on openyuanrong.

The entry point is `examples/llm_router/run_infer.sh`, whose shell script runs `examples/llm_router/parallel_infer_verl_kvc.py`.

### 3.1 Execution path

The kvcaware router sits between verl's `LLMServerManager` and the multiple vLLM replicas, routing each turn of each session to an appropriate replica; the rest of the pipeline (agent framework, sandbox, trajectory return) is reused from uni-agent:

```text
verl LLMServerManager
    -> KVCAwareBalancer (router)  ── routes ──>  vLLM replicas ×N
    -> AgentFrameworkRolloutAdapter
    -> Uni-Agent Gateway sessions
    -> Task Runner (mini-swe-agent) + sandbox
    -> TransferQueue trajectories and rewards
```

The router intervenes at the very first layer: every turn's `acquire_server` is decided by it, and the KV-cache occupancy and prefix-hit state on the replica side are collected back into the router as decision signals for the next turn.

### 3.2 Prerequisites

1. This repository (including the `verl` submodule) with `pip install -e .` so the `uni_agent` package (which hosts the router) resolves.
2. An AKernel remote sandbox endpoint (`AKERNEL_SERVER_ADDRESS` / `AKERNEL_TOKEN`).
   The accompanying `task_config_openyuanrong.yaml` maps the provider-agnostic `swebench/**` images in the dataset to the openyuanrong SWR repository (`swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/**:v2`) via `sandbox.image_map`, so a plain SWE-bench parquet can be used directly.
3. Dataset parquet (SWE-bench verified). Point `--data-path` at any compatible parquet (generatable with uni-agent's `examples/data_preprocess/swe_bench_verified.py`).
4. Environment variables:
   These list only the variables read via `os.environ` inside Ray workers (not CLI flags) — `run_infer.sh` exports them, so set them in your shell before invoking:

| Variable | Default | Description |
|-----|---------|------|
| `AKERNEL_SERVER_ADDRESS` / `AKERNEL_TOKEN` | empty | AKernel remote sandbox credentials |
| `AKERNEL_TUNNEL_SSL_VERIFY` | `0` | AKernel tunnel TLS verification (0 = disabled) |
| `VERL_LOGGING_LEVEL` | `INFO` | verl log level |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward evaluation timeout inside the sandbox (seconds) |
| `RL_INSIGHT_SERVER_URL` | `http://127.0.0.1:18080` | rl-insight observability service |

### 3.3 Running

`run_infer.sh` is a thin wrapper: it exports the Ray-worker environment variables (AKernel credentials + observability env), then forwards all CLI flags verbatim to `parallel_infer_verl_kvc.py`.

`--task-config` is required — it selects the task / agent / model configuration per row. Use `--help` for the full flag list and defaults:

```bash
bash examples/llm_router/run_infer.sh --help

# Smoke test (1 sample, kv-events on)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_openyuanrong.yaml \
    --max-samples 1 --kv-events

# Full run (omit --max-samples to run the whole dataset)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_openyuanrong.yaml \
    --tensor-parallel-size 2 --n-gpus-per-node 8 --kv-events

# With mooncake cross-replica KV sharing (start the mooncake master separately)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_openyuanrong.yaml \
    --enable-mooncake --kv-events

# Ascend (vllm-ascend) backend
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_openyuanrong.yaml \
    --device ascend --enable-mooncake
```

Main CLI flags:

(1) Base configuration:

| Flag | Default | Description |
|------|------|------|
| `--model-path` | `~/models/Qwen3.5-9B` | Model |
| `--data-path` | `.../swe_bench_verified.parquet` | Dataset |
| `--task-config` | `.../task_config_openyuanrong.yaml` | Task config file |
| `--max-samples` | `-1` | Number of samples (-1 = all) |
| `--shuffle` / `--seed` | off / 42 | Shuffle + reproducible seed |
| `--prompt-length` / `--response-length` | 4096 / 131072 | Token lengths |
| `--max-model-len` | config-native ceiling | When set, prompt = max_model_len - response_length - 100 |
| `--n` | 1 | rollout.n |
| `--tensor-parallel-size` / `--n-gpus-per-node` / `--nnodes` | 4 / 8 / 1 | Parallelism |
| `--gateway-count` / `--max-concurrent-sessions` | 1 / 128 | Gateway pool / concurrency |
| `--gpu-memory-utilization` | 0.8 | vLLM GPU memory utilization |
| `--device` | `gpu` | `gpu` / `ascend` (mooncake connector class) |

(2) Routing strategy (overrides `kvcaware.yaml` defaults):

| Flag | Description |
|------|------|
| `--kv-events` | off | Enable vLLM kv-events (kvcaware load signal) |
| `--enable-mooncake` / `--mooncake-config-path` | off / `mooncake_config.json` | Cross-replica KV sharing |
| `--slow-cut` | `capacity-token-aware` |
| `--overload-mode` | `None` / `kv_cache_usage_perc` / `kv_load` |
| `--load-threshold` | Overload threshold |

