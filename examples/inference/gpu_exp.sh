#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# KV-cache-aware router 对照实验批处理脚本（参考 verl/gpu_exps.sh）
#
# 跑两组路由策略对照：sticky vs kvcaware-lt{0.9,0.8}，在 concurrency × context
# 双层网格上各跑一遍。每组 retry-until-success：只要日志里没出现
# "=> Mean RM Score" 就清理进程重来（保证拿到完整结果）。
#
# 与 verl/gpu_exps.sh 的适配差异：
#   - 调 examples/inference/run_infer_openyuanrong_kvc.sh（位置参数 MODEL/DATA/TASK）
#   - concurrency / log_dir 是命令行 flag，不走 sed 改 yaml（task_config 无此字段）
#   - max_steps 对应 kvc 的 max_turns，用 sed 改 task_config_openyuanrong.yaml
#
# 风格照搬 verl/gpu_exps.sh：环境变量 → 双层循环 → sticky + kvcaware 对照 → retry 循环
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export PYTHONHASHSEED=0

# ── 必填：openyuanrong 沙箱凭证（run_infer_openyuanrong_kvc.sh 也需要）─────────
export DEPLOYMENT="openyuanrong"
export AKERNEL_SERVER_ADDRESS="124.70.166.142:443"
export OPENYUANRONG_SERVER_ADDRESS="124.70.166.142:443"
export AKERNEL_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE4MTU4NTMwNTAsInJvbGUiOiJkZXZlbG9wZXIiLCJzdWIiOiJkZWZhdWx0In0.NGY4ZjkzZWZhYmE4YzkxOGIwNTdkN2VmZTQ5MTdiZWQ2MjhlMTMwYzA0OTU3NjRlMWNmNDNjZDUzYTMxNjliYw"
export OPENYUANRONG_TOKEN="${AKERNEL_TOKEN}"
export TUNNEL_SSL_VERIFY="0"

# ── 位置参数：模型 / 数据 / task config ─────────────────────────────────────
MODEL=/data1/models/Qwen/Qwen3-8B
DATASET=/data1/dataset/swe_agent/swe_bench_verified.parquet
DEPLOYMENT_CFG=/data1/yourifan/uni-agent-yuanrong/examples/inference/task_config_openyuanrong.yaml

MAX_SAMPLES=32
RES_LEN=8000
LOG_BASE=/tmp

# 结果行标记：run_infer 完成后会打印 mean rm_score 汇总，用它判定是否拿到完整结果。
TARGET="mean rm_score"

# task_config 里 max_steps 对应 kvc 的 max_turns，固定到 100（与 gpu_exps.sh 一致）。
sed -i "s|    max_steps: .*|    max_steps: 100|" "$DEPLOYMENT_CFG"

# 双层网格：concurrency × context（max_model_len）。
concurrencys=(128 96 64)
contexts=(32768 16384 8192)

for CONCURRENCY in "${concurrencys[@]}"; do
    for CONTEXT in "${contexts[@]}"; do
        # ── sticky 组：--do-shortcut --slow-cut least-inflight --overload-mode None ─────
        LOG_FILE="infer-sticky-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.log"
        while ! grep -q "$TARGET" "$LOG_FILE" 2>/dev/null; do
            pkill -9 python
            ps -aux | grep run_infer | grep -v grep | awk -F ' ' '{print$2}' | xargs -I {} kill -9 {} 2>/dev/null || true
            ray stop 2>/dev/null || true
            fuser -k /dev/nvidia* 2>/dev/null || true
            nvidia-smi
            echo "Running sticky concurrency=${CONCURRENCY} context=${CONTEXT}"
            bash examples/inference/run_infer_openyuanrong_kvc.sh \
                "$MODEL" "$DATASET" "$DEPLOYMENT_CFG" \
                --tool-parser hermes --engine vllm --nnodes 1 --n-gpus-per-node 8 \
                --device gpu --tp 1 --gpu-memory-utilization 0.93 \
                --concurrency "$CONCURRENCY" --max-model-len "$CONTEXT" --response-length 8000 \
                --max-samples "$MAX_SAMPLES" --n 8 --shuffle \
                --do-shortcut --slow-cut least-inflight --overload-mode None \
                --kv-events --log-dir "${LOG_BASE}/router-trajs/sticky-${CONCURRENCY}-${CONTEXT}" \
                > "$LOG_FILE" 2>&1
        done

        # ── kvcaware 组：--no-do-shortcut --slow-cut capacity-token-aware --overload-mode kv_cache_usage_perc ──
        lts=(0.9 0.8)
        for lt in "${lts[@]}"; do
            LOG_FILE="infer-kvcaware-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}-lt${lt}.log"
            while ! grep -q "$TARGET" "$LOG_FILE" 2>/dev/null; do
                pkill -9 python
                ps -aux | grep run_infer | grep -v grep | awk -F ' ' '{print$2}' | xargs -I {} kill -9 {} 2>/dev/null || true
                ray stop 2>/dev/null || true
                fuser -k /dev/nvidia* 2>/dev/null || true
                nvidia-smi
                echo "Running kvcaware-lt${lt} concurrency=${CONCURRENCY} context=${CONTEXT}"
                bash examples/inference/run_infer_openyuanrong_kvc.sh \
                    "$MODEL" "$DATASET" "$DEPLOYMENT_CFG" \
                    --tool-parser hermes --engine vllm --nnodes 1 --n-gpus-per-node 8 \
                    --device gpu --tp 1 --gpu-memory-utilization 0.93 \
                    --concurrency "$CONCURRENCY" --max-model-len "$CONTEXT" --response-length 8000 \
                    --max-samples "$MAX_SAMPLES" --n 8 --shuffle \
                    --no-do-shortcut --slow-cut capacity-token-aware \
                    --overload-mode kv_cache_usage_perc --load-threshold "$lt" \
                    --kv-events --log-dir "${LOG_BASE}/router-trajs/kvcaware-${CONCURRENCY}-${CONTEXT}" \
                    > "$LOG_FILE" 2>&1
            done
        done
    done
done
