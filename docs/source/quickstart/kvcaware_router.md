# KV-Cache-Aware

## 一、动机

### 1.1 agentic RL的 rollout 负载特征

Agentic RL的 rollout 阶段，工作负载形态和传统单轮 RL rollout 差异较大：

- **多轮对话**：一个 sample 对应一个 agent 会话，单会话动辄上百轮 tool-call，单轮 prefill 又动辄上万 token。会话间轮次差异较大，有的几轮收敛、有的几十上百轮收敛。
- **强前缀复用**：同一会话的多轮请求共享一条不断增长的前缀历史。若每一轮都能命中上一轮已驻留的 KV-cache，就能跳过大量 prefill 重算；反之每轮重算则 prefill 成本线性累积。
- **KV Cache驱逐**：会话是长生命周期的，其增长的前缀 KV 只有留在原副本才有价值。但副本 KV 容量有限，单会话工作集就能吃掉一大半，引擎会驱逐旧前缀块腾位，下轮命中失败又得 prefill 重算，形成"撑满→驱逐→miss→重算"的恶性循环。

### 1.2 router的核心挑战

把一个长生命周期会话调度到某个推理副本上，路由的本质难题是要在互相牵制的目标间做权衡：

- **prefill命中率与负载均衡的根本矛盾**：保prefill命中率要求把会话钉在原副本，负载均衡又要求把请求挪到更闲的副本——两者直接冲突。挪走会丢掉已驻留的 KV、下轮 prefill 重算；不挪则热点副本持续堆积、拖慢长尾。因此需要判断"什么时候值得为负载均衡牺牲prefill命中率"才是真正的决策。

### 1.3 设计目标

- **前缀命中优先**：会话的后续轮次尽量回到原副本，命中已驻留 KV，打破驱逐→miss→重算循环。
- **负载均衡迁移触发**：仅在副本确实过载时才解绑迁移，为散热付出的 prefill 重算代价须有明确判据支撑。

---

## 二、KV Cache Aware Router 详解

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
2. SLOW-CUT（短路失败后的兜底评分，三选一）
   ├── LEAST_INFLIGHT        → 按 active_sessions 最小，窗口随机 first-bind
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

本章说明如何用 `kvcaware` 路由器跑一轮 agentic 推理，入口是 `examples/inference/run_infer_yuanrong_kvc.sh`，其中shell脚本中实际执行`examples/inference/parallel_infer_verl_kvc.py`。

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

### 3.2 运行

```
MODEL=/path/to/Qwen3-8B
DATASET=/path/to/swe_agent/swe_bench_verified.parquet
DEPLOYMENT_CFG=/path/to/examples/inference/task_config_openyuanrong.yaml

bash examples/inference/run_infer_yuanrong_kvc.sh \
    "$MODEL" "$DATASET" "$DEPLOYMENT_CFG" \
    --tool-parser hermes --engine vllm --nnodes 1 --n-gpus-per-node 8 \
    --device gpu --tp 1 --gpu-memory-utilization 0.93 \
    --concurrency 128 --max-model-len 32768 --response-length 32768 \
    --max-samples 32 --n 8 --shuffle \
    --do-shortcut --slow-cut capacity-token-aware \
    --overload-mode kv_cache_usage_perc --load-threshold 0.9 \
    --kv-events --log-dir '/path/to/log_dir'
```

### 3.3 关键参数

完整列表用 `python examples/inference/parallel_infer_verl_kvc.py --help` 查看。基础设施与 agent 控制：

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

路由策略（覆盖 `kvcaware.yaml` 默认值）：

| Flag | 说明 |
|------|------|
| `--kv-events` | 关 | 启用 vLLM kv-events（kvcaware 负载信号） |
| `--enable-mooncake` / `--mooncake-config-path` | 关 / `mooncake_config.json` | 跨副本 KV 共享 |
| `--slow-cut` | `least-inflight` / `capacity-token-aware` |
| `--overload-mode` | `None` / `kv_cache_usage_perc` / `kv_load` / `None` |
| `--load-threshold` / `--do-shortcut` | 过载阈值 / sticky 短路开关 |

---

## 四、实验结果

### 4.1 GPU实验结果

### 4.4 结论

