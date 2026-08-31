#!/usr/bin/env bash
# Single-node Ascend (vllm-ascend) experiment matrix driver.
#
# Sweeps sticky-vs-kvcaware over (concurrency × context), retrying each run until
# the "inference summary" sentinel lands in its log. Requires an OpenYuanrong
# remote sandbox (the only reverse-tunnel provider).

set -uo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONHASHSEED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL=/path/to/Qwen/Llama3.1-8B-Instruct
DATASET=/path/to/swe_bench_train_model.parquet
MAX_SAMPLES=64
RES_LEN=8000

# OpenYuanrong sandbox creds (only reverse-tunnel provider).
: "${OPENYUANRONG_SERVER_ADDRESS:?Set OPENYUANRONG_SERVER_ADDRESS}"
: "${OPENYUANRONG_TOKEN:?Set OPENYUANRONG_TOKEN}"
export OPENYUANRONG_SERVER_ADDRESS OPENYUANRONG_TOKEN
export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
export SANDBOX_NAME_PREFIX="${SANDBOX_NAME_PREFIX:-mini-swe-}"

# rl-insight observability — start on entry, stop on exit. One-time `rl-insight server install`.
# Set VERL_RL_INSIGHT_ENABLE=0 to disable.
export VERL_RL_INSIGHT_ENABLE="${VERL_RL_INSIGHT_ENABLE:-1}"
export RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://127.0.0.1:18080}"
rl-insight server start --detach 2>/dev/null || true
trap 'rl-insight server stop 2>/dev/null || true' EXIT

TARGET="inference summary"

concurrencys=(16 24 32 128 192 256)
contexts=(16384 32768 64000 128000)

run_experiment() {
    local log_file=$1
    shift

    while ! grep -q "$TARGET" "$log_file" 2>/dev/null; do
        pkill -9 -f 'run_infer.py|ray::' || true
        ps -aux | grep run_infer.sh | grep -v grep | awk -F ' ' '{print $2}' | xargs -r -I {} kill -9 {} || true
        ray stop || true
        fuser -k /dev/davinci* || true
        npu-smi info
        bash "${REPO_ROOT}/examples/agent_aware_router/run_infer.sh" \
            --model-path "$MODEL" \
            --data-path "$DATASET" \
            --task-config "${REPO_ROOT}/examples/agent_aware_router/task_config_mini_swe_agent.yaml" \
            --device ascend \
            --n-gpus-per-node 8 \
            --tp 4 \
            --response-length "$RES_LEN" \
            --max-model-len "$CONTEXT" \
            --max-samples "$MAX_SAMPLES" \
            --n 8 \
            --shuffle \
            --concurrency "$CONCURRENCY" \
            --kv-events \
            "$@" > "$log_file" 2>&1
    done
}

for CONCURRENCY in "${concurrencys[@]}"; do
    for CONTEXT in "${contexts[@]}"; do
        LOG_FILE="infer-sticky-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.log"
        echo "Running sticky concurrency=${CONCURRENCY} context=${CONTEXT}"
        run_experiment "$LOG_FILE" \
            --slow-cut least-inflight \
            --overload-mode None

        lts=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
        for lt in "${lts[@]}"; do
            LOG_FILE="infer-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}.log"
            echo "Running kvcaware-lt${lt} concurrency=${CONCURRENCY} context=${CONTEXT}"
            run_experiment "$LOG_FILE" \
                --slow-cut capacity-token-aware \
                --overload-mode kv_cache_usage_perc \
                --load-threshold "$lt"
        done
    done
done
