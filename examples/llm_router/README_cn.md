# llm-router 推理快速开始

Last updated: 08/25/2026

## 这是什么

本示例在 verl 的 **KV-cache-aware router**（`kvcaware`）上运行 **mini-swe-agent**（黑盒）SWE-bench agentic 推理,router 由 `uni_agent.llm_router` 托管,通过 FQN 注入 verl
（`rollout.router_config_path` + `pkg://` router YAML,无需 verl 侧注册）。
会话跑在 uni-agent 的 **framework + task_runner 路径**（`uni_agent.framework.task_runner.run_task`）:vLLM 副本位于 kvcaware 路由器之后
（KV-cache 命中率 + 负载感知调度),gateway 池在 openyuanrong 远程沙箱中驱动 `mini_swe_agent` 会话,每个任务的 reward 会上报回框架。不启动 trainer。

mini-swe-agent 在沙箱**内部**从一个预构建的 tool 镜像运行（挂载在
`/opt/mini-swe-agent`），通过反向隧道（`proxy_port` + 运行时注入的 `upstream`）访问策略 gateway，因此沙箱集群与训练集群**无需互通**——只需训练侧能访问沙箱服务（API + 拉镜像）。

核心组件:
- `KVCAwareBalancer` — 路由框架,管理组件生命周期与路由决策
- `Collector` — 采集 vLLM KV 事件与 Prometheus 指标用于决策
- `Strategy` — 评分策略,综合 KV-cache 命中率与负载
- `Store` — 单例存储,缓存采集到的指标与 KV block 状态

task runner / reward / dataset 代码复用自已安装的 `uni-agent` 包
(`uni_agent.tasks`、`uni_agent.sandbox`)。

## 前置条件

1. 本仓库(含 `verl` submodule)并 `pip install -e .`,使 `uni_agent` 包(托管
   router)可解析。
2. 一个 **OpenYuanrong** 远程沙箱端点（`OPENYUANRONG_SERVER_ADDRESS` /
   `OPENYUANRONG_TOKEN`）——唯一支持反向隧道的沙箱 provider,mini-swe-agent
   的 tool-image 挂载需要它。
3. **mini-swe-agent tool 镜像**,构建并推送到沙箱服务可拉取的仓库。用兄弟
   example 构建:
   ```bash
   bash examples/mini_swe_agent/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
   ```
   随附的 `task_config_mini_swe_agent.yaml` 把
   `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest`
   挂载到 `/opt/mini-swe-agent`;若推送到别处,改 `mounts[].image_url`。
4. 数据集 parquet(SWE-bench verified)。用 `--data-path` 指向任意兼容的
   parquet(用 `python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir ~/data/swe_agent`
   生成)。schema 必须携带 `extra_info.tools_kwargs` 中的 `task` dict
   (旧格式 `tools_kwargs: {env, reward}` 会被拒绝,需重新生成)。

## 运行

`run_infer.sh` 是一个薄包装:导出 Ray worker 环境变量(OpenYuanrong 凭据 +
可观测性 env),然后把所有 CLI flag 透传给 `run_infer.py`。
`--task-config` 必填——它按行选择 task / agent / model 配置。完整 flag
列表与默认值用 `--help` 查看:

```bash
bash examples/llm_router/run_infer.sh --help

# Smoke test (1 sample, kv-events on)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_mini_swe_agent.yaml \
    --max-samples 1 --kv-events

# 全量运行(省略 --max-samples 即跑整个数据集)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_mini_swe_agent.yaml \
    --tensor-parallel-size 2 --n-gpus-per-node 8 --kv-events

# 带 mooncake 跨副本 KV 共享(mooncake master 单独起)
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_mini_swe_agent.yaml \
    --enable-mooncake --kv-events

# Ascend（vllm-ascend）后端
bash examples/llm_router/run_infer.sh \
    --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct \
    --data-path /path/to/swe_bench.parquet \
    --task-config examples/llm_router/task_config_mini_swe_agent.yaml \
    --device ascend --enable-mooncake
```

主要 CLI flag(完整列表见 `--help`):

| Flag | 默认值 | 说明 |
|------|---------|------|
| `--data-path` | `~/data/swe_agent/swe_bench_verified.parquet` | 数据集 parquet |
| `--model-path` / `--model` | `~/models/Qwen3-Coder-30B-A3B-Instruct` | 模型路径 |
| `--task-config` | (必填) | YAML task 配置(`- name:` 条目),按行驱动每个 sample |
| `--max-samples` / `--limit` | 全部 | 运行的样本数(省略即全部) |
| `--n` | `1` | 每个实例的 rollout 会话数 |
| `--shuffle` / `--seed` | 关 / `42` | 采样前打乱数据,并指定可复现的随机种子 |
| `--prompt-length` / `--response-length` | `4096` / `8192` | Token 长度 |
| `--max-model-len` | 引擎钳制 | vLLM 最大上下文长度 |
| `--tensor-parallel-size` / `--n-gpus-per-node` / `--nnodes` | `4` / `8` / `1` | 并行度 |
| `--concurrency` | env `GLOBAL_CONCURRENCY` | 最大在途 gateway 会话数 |
| `--gateway-count` / `--tool-parser` | `4` / `qwen3_coder` | Gateway 池 / 工具调用解析器 |
| `--gpu-memory-utilization` | `0.9` | vLLM GPU 显存利用率(0-1) |
| `--router-config-path` | `pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml` | 打包的 router YAML |
| `--max-num-seqs` | `256` | 每个引擎的最大并发序列数 |
| `--kv-events` | 关 | 启用 vLLM kv-events(kvcaware 负载信号) |
| `--enable-mooncake` / `--mooncake-config-path` | 关 / `mooncake_config.json` | 跨副本 KV 共享 |
| `--device` | `gpu` | `gpu` 或 `ascend`(选择 mooncake connector 类) |
| `--alpha` / `--load-threshold` / `--slow-cut` / `--overload-mode` / `--do-shortcut` | 取自 `kvcaware.yaml` | kvcaware strategy[0] 覆盖 |

## 环境变量

仅列出 Ray worker 内部通过 `os.environ` 读取的变量(非 CLI flag)——
`run_infer.sh` 会 export 它们,调用前在 shell 里设置:

| 变量 | 默认值 | 说明 |
|-----|---------|------|
| `OPENYUANRONG_SERVER_ADDRESS` / `OPENYUANRONG_TOKEN` | 空 | OpenYuanrong 远程沙箱认证 |
| `OPENYUANRONG_TUNNEL_SSL_VERIFY` | `0` | 沙箱反向隧道 TLS 校验(0 = 禁用) |
| `SANDBOX_NAME_PREFIX` | `mini-swe-` | 创建沙箱名称的前缀 |
| `VERL_LOGGING_LEVEL` | `INFO` | verl 日志级别 |
| `RL_INSIGHT_SERVER_URL` | `http://127.0.0.1:18080` | rl-insight 可观测性服务 |

## 可观测性

运行日志携带 router 的调度证据——balancer 的 `routed to server` 行、strategy
的 `score(): COMBINED` 日志、kv-events collector 指标。结束时
`run_infer.py` 打印 `inference summary` 块(mean rm_score 与
逐 session 明细),设置 `--result-path` 时会另写一个 JSON 结果文件。

## 实验矩阵

两个 driver 扫一组 sticky-vs-kvcaware 矩阵(并发 × 上下文),每次运行重试
直到日志中出现 `inference summary` 成功哨兵。两者都硬编码 `--device ascend`
（vllm-ascend）与 `--kv-events`;kvcaware 单元把 `--load-threshold` 扫过
`0.1..0.9`。

- `ascend-exps.sh` — 单节点(16 NPU,TP=4 → 4 replicas)。先改文件顶部的
  `MODEL` / `DATASET` / `MAX_SAMPLES`,再 `bash examples/llm_router/ascend-exps.sh`。
- `multi-node-ascend-exps.sh` — 6 节点(本机 = Ray head + 5 个免密 SSH worker,
  每个通过 `docker exec hgq-verl-ascend` 进入;48 NPU / TP=4 → 12 replicas)。
  每次尝试会重建并拆除 Ray 集群。先改 `WORKERS[]` 主机列表与 `MODEL` /
  `DATASET`。

`run_infer.sh` 是底层唯一入口;两个 driver 都只是循环调用它。

## 注意事项

- task runner 需要 OpenYuanrong 远程沙箱;没有它会快速失败。反向隧道
  （`proxy_port` + 运行时 `upstream`）仅 openyuanrong provider 支持——在
  其他 provider 上配置 `proxy_port` 会被显式拒绝。
- 首次运行前先用 `examples/mini_swe_agent/build_tool.sh` 构建 tool 镜像并
  推送到沙箱服务可拉取的仓库;task config 把它挂载到 `/opt/mini-swe-agent`。
- `agent.step_limit` 限制 mini-swe-agent 的轮次;`agent.run_timeout` 限制
  沙箱内 agent 进程的运行时间(秒)。让 `--max-model-len` 明显高于单 episode
  token 预算,以免长 transcript 被 vLLM 拒绝。
- 数据集 schema 必须携带 `extra_info.tools_kwargs` 中的 `task` dict(task
  runner 据此解析任务)。如果你的 parquet 早于该格式,请用
  `python -m uni_agent.tasks.swe_bench.preprocess` 重新生成。
