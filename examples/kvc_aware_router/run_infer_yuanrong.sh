#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   bash examples/inference/run_infer_openyuanrong_kvc.sh \
#       MODEL_PATH [DATA_PATH] [TASK_CONFIG] [...extra flags...]
#
# Examples:
#   # smoke test (kvcaware router + kv-events)
#   bash examples/inference/run_infer_openyuanrong_kvc.sh \
#       /data1/models/Qwen/Qwen3-8B --limit 4 --kv-events
#
#   # full 8-GPU run with router strategy overrides
#   bash examples/inference/run_infer_openyuanrong_kvc.sh \
#       /data1/models/Qwen/Qwen3-8B \
#       --tensor-parallel-size 4 --n-gpus-per-node 8 --max-model-len 32768 \
#       --kv-events --load-threshold 0.6 --slow-cut prefix-load-aware
#
#   # with MooncakeStoreConnector (cross-replica KV sharing)
#   bash examples/inference/run_infer_openyuanrong_kvc.sh \
#       /data1/models/Qwen/Qwen3-8B --enable-mooncake --kv-events
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH=${1:?ERROR: MODEL_PATH is required}
DATA_PATH=${2:-${HOME}/data/swe_agent/swe_bench_verified.parquet}
TASK_CONFIG=${3:-${SCRIPT_DIR}/task_config_openyuanrong.yaml}
shift 3 2>/dev/null || set --  # 丢掉前 3 个；"$@" = 剩余 flag

# Pre-flight checks
for var_name in MODEL_PATH DATA_PATH TASK_CONFIG; do
    path="${!var_name}"
    if [ ! -e "$path" ]; then
        echo "ERROR: ${var_name} not found: ${path}" >&2
        exit 1
    fi
done

python examples/inference/parallel_infer_verl_kvc.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --task-config "${TASK_CONFIG}" \
    "$@"

