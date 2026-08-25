# AgentAware Router

## 一、动机

### 1.1 agentic RL的 rollout 负载特征

Agentic RL的 rollout 阶段，工作负载形态和传统单轮 RL rollout 差异较大：

- **多轮对话**：一个 sample 对应一个 agent 会话，单会话动辄上百轮 tool-call，单轮 prefill 又动辄上万 token。会话间轮次差异较大，有的几轮收敛、有的几十上百轮收敛。
- **强前缀复用**：同一会话的多轮请求共享一条不断增长的前缀历史。若每一轮都能命中上一轮已驻留的 KV-cache，就能跳过大量 prefill 重算；反之每轮重算则 prefill 成本线性累积。
- **KV Cache驱逐**：会话是长生命周期的，其增长的前缀 KV 只有留在原副本才有价值。但副本 KV 容量有限，单会话工作集就能吃掉一大半，引擎会驱逐旧前缀块腾位，下轮命中失败又得 prefill 重算，形成"撑满→驱逐→miss→重算"的恶性循环。
- **GPU空泡**：由于请求的上下文长度均值大，分布散，通常导致batch内或replica间存在大量的gpu空泡，进一步造成MFU低下，端到端性能变差。

### 1.2 设计目标

- **前缀命中优先**：会话的后续轮次尽量回到原副本，命中已驻留 KV，打破驱逐→miss→重算循环。
- **负载均衡迁移触发**：仅在副本确实过载时才解绑迁移，为散热付出的 prefill 重算代价须有明确判据支撑。
- **减少gpu空泡**：通过调度和多层次KVCache管理，有效缓解gpu空泡问题。

---

## 二、Agent Aware Router 详解

### 2.1 总体架构

`kvcaware` 是一个可插拔策略的路由器，注册在 `LoadBalancerRegistry` 下名为 `"kvcaware"`，由 `get_router_handle` 在 `LLMServerManager` 侧装配。核心组件：

```
KVCAwareBalancer (balancer.py)       ← 纯外壳：生命周期 + route() 委派
   ├── strategies/                   ← 评分算法（核心）
   │    ├── kvc_aware.py             ← KVCacheAwareStrategy（主策略）
   │    ├── routing.py               ← route() 加权排序
   │    ├── base.py                  ← ReplicaInfo
   │    └── registry.py              ← config类型 → 运行时策略类 注册表
   ├── store/                        ← 单例存储
   │    ├── data_store.py            ← 指标 + sticky绑定 + active_sessions
   │    ├── kv_cache_store.py        ← prefix-hash 链 + 三层命中率
   │    ├── per_replica_store.py / per_request_store.py
   ├── collectors/                   ← 采集 vLLM 信号
   │    ├── manager.py               ← 采集器生命周期
   │    ├── decoder/vllm/metrics.py  ← /metrics 轮询 → dict
   │    ├── decoder/vllm/kv.py, kv_event.py  ← kv-events zmq → 增量 block 状态
   │    └── transport/{http,zmq,callback}.py
   ├── config/                       ← KVCAwareConfig + StrategyConfig
   ├── types/                        ← SlowCut / OverloadMode / Layer / MetricKey 枚举
   └── insight/emitter.py            ← 推送到 rl-insight → Prometheus → Grafana
```

### 2.2 路由决策主流程：`score()`

每次 `acquire_server(request_id, prompt_ids)` 时，`Balancer` 调 `route()`，后者对每个策略调 `score()` 取加权排序，取 `ranking[0]`。`KVCacheAwareStrategy.score()` 的决策树：

```
1. STICKY 短路（do_shortcut=True）
   ├── 有 request_id 且有 sticky 绑定副本
   │    └── 未过载？ → HIT，返回 [STICKY_TOP_SCORE, 0, ...]（保前缀局部性）
   │        过载？ → fallback
   └── 否 → 进入 slow-cut
2. SLOW-CUT（短路失败后的兜底评分）
   └── CAPACITY_TOKEN_AWARE  → 容量门 + token 增量，离散选剩余最大
```

### 2.3 CAPACITY_TOKEN_AWARE 评分

CAPACITY_TOKEN_AWARE 在"还有余量的副本里，挑prefill命中率最高的"。具体计算公式如下：

```
avail[i]     = cap × (1 - kv_cache_usage_perc[i])                   # 纯容量可用token
need[i]      = len(prompt_ids) × (1 - gpu_hit[i])                   # 该请求去副本 i 需新算的 token
remaining[i] = avail[i] - need[i]                                   # 分配后剩余
eligible[i]  = avail[i] >= cap × (1 - capacity_reserve_threshold)   # 纯容量硬门
pick         = argmax(eligible, remaining)                          # 合格里选剩余最大
```

- prefill命中率靠 `need` 项参与排序，命中率高的副本需新算 token 少、`remaining` 大、在合格副本里胜出。
- 负载均衡首先根据纯容量硬门 `eligible`过滤，容量已满的副本被过滤掉不再分配。其次根据`avail`项参与排序，`avail`高的副本负载低，`remaining` 大。

---

## 三、实验

本示例说明基于openyuanrong，如何使用 **KV-cache-aware router**（`kvcaware`）在verl后端上运行 SWE-bench agentic 推理。

入口是 `examples/llm_router/run_infer.sh`，其中shell脚本中执行`examples/llm_router/parallel_infer_verl_kvc.py`。

### 3.1 执行路径

kvcaware 路由器挂在 verl 的 `LLMServerManager` 与多个 vLLM 副本之间，把每个会话的每一轮请求路由到合适的副本；其余链路（agent 框架、沙箱、轨迹回传）复用 uni-agent：

```text
verl LLMServerManager
    -> KVCAwareBalancer (router)  ── 路由 ──>  vLLM 副本 ×N
    -> AgentFrameworkRolloutAdapter
    -> Uni-Agent Gateway sessions
    -> Task Runner (mini-swe-agent) + sandbox
    -> TransferQueue trajectories and rewards
```

router 在第一层就介入：会话的每一轮 `acquire_server` 都由它决定落到哪个副本，副本侧的 KV-cache 占用与前缀命中又被采集回 router 作为下一轮的决策信号。

### 3.2 前置条件

1. 本仓库(含 `verl` submodule)并 `pip install -e .`,使 `uni_agent` 包(托管
   router)可解析。
2. 一个 AKernel 远程沙箱端点(`AKERNEL_SERVER_ADDRESS` / `AKERNEL_TOKEN`)。
   随附的 `task_config_openyuanrong.yaml` 通过 `sandbox.image_map` 把数据集里
   provider-agnostic 的 `swebench/**` 镜像映射到 openyuanrong SWR 仓库
   (`swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/**:v2`),
   因此普通 SWE-bench parquet 可直接使用。
3. 数据集 parquet(SWE-bench verified)。用 `--data-path` 指向任意兼容的
   parquet(可用 uni-agent 的 `examples/data_preprocess/swe_bench_verified.py` 生成)。
4. 环境变量：
    仅列出 Ray worker 内部通过 `os.environ` 读取的变量(非 CLI flag)——`run_infer.sh` 会 export 它们,调用前在 shell 里设置:

| 变量 | 默认值 | 说明 |
|-----|---------|------|
| `AKERNEL_SERVER_ADDRESS` / `AKERNEL_TOKEN` | 空 | AKernel 远程沙箱认证 |
| `AKERNEL_TUNNEL_SSL_VERIFY` | `0` | AKernel 隧道 TLS 校验(0 = 禁用) |
| `VERL_LOGGING_LEVEL` | `INFO` | verl 日志级别 |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | 沙箱内 reward 评估超时(秒) |
| `RL_INSIGHT_SERVER_URL` | `http://127.0.0.1:18080` | rl-insight 可观测性服务 |

### 3.3 运行

`run_infer.sh` 是一个薄包装:导出 Ray worker 环境变量(AKernel 凭据 + 可观测性 env),然后把所有 CLI flag 透传给 `parallel_infer_verl_kvc.py`。

`--task-config` 必填——它按行选择 task / agent / model 配置。完整 flag 列表与默认值用 `--help` 查看:

```bash
bash examples/llm_router/run_infer.sh --help

# Smoke test (1 sample, kv-events on)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_openyuanrong.yaml \
    --max-samples 1 --kv-events

# 全量运行(省略 --max-samples 即跑整个数据集)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_openyuanrong.yaml \
    --tensor-parallel-size 2 --n-gpus-per-node 8 --kv-events

# 带 mooncake 跨副本 KV 共享(mooncake master 单独起)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_openyuanrong.yaml \
    --enable-mooncake --kv-events

# Ascend（vllm-ascend）后端
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_openyuanrong.yaml \
    --device ascend --enable-mooncake
```

主要 CLI flag:

(1) 基础配置：

| Flag | 默认 | 说明 |
|------|------|------|
| `--model-path` | `~/models/Qwen3.5-9B` | 模型 |
| `--data-path` | `.../swe_bench_verified.parquet` | 数据集 |
| `--task-config` | `.../task_config_openyuanrong.yaml` | task配置文件 |
| `--max-samples` | `-1` | 样本数（-1 全部） |
| `--shuffle` / `--seed` | 关 / 42 | 打乱+可复现种子 |
| `--prompt-length` / `--response-length` | 4096 / 131072 | token 长度 |
| `--max-model-len` | config 原生上限 | 设后 prompt = max_model_len - response_length - 100 |
| `--n` | 1 | rollout.n |
| `--tensor-parallel-size` / `--n-gpus-per-node` / `--nnodes` | 4 / 8 / 1 | 并行度 |
| `--gateway-count` / `--max-concurrent-sessions` | 1 / 128 | gateway 池 / 并发 |
| `--gpu-memory-utilization` | 0.8 | vLLM 显存利用率 |
| `--device` | `gpu` | `gpu` / `ascend`（mooncake connector 类） |

(2) 路由策略（覆盖 `kvcaware.yaml` 默认值）：

| Flag | 说明 |
|------|------|
| `--kv-events` | 关 | 启用 vLLM kv-events（kvcaware 负载信号） |
| `--enable-mooncake` / `--mooncake-config-path` | 关 / `mooncake_config.json` | 跨副本 KV 共享 |
| `--slow-cut` | `capacity-token-aware` |
| `--overload-mode` | `None` / `kv_cache_usage_perc` / `kv_load` |
| `--load-threshold` | 过载阈值|
