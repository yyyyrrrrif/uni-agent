#!/usr/bin/env bash
# Multi-node Ascend experiment driver.
#
# 6 nodes (this host = Ray head + 5 ssh workers via `docker exec hgq-verl-ascend`).
# 48 NPUs, TP=4 -> 12 replicas. rl-insight stays up for the whole matrix; the Ray
# cluster is rebuilt per attempt (clean -> ray up -> run_infer.sh, retry until
# "inference summary").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# =====================================================================
# Configuration
# =====================================================================
WORKERS=(
    "root@10.22.22.22"
    "root@10.22.22.23"
    "root@10.22.22.24"
    "root@10.22.22.25"
    "root@10.22.22.26"
)

WORKER_CONTAINER="hgq-verl-ascend"
NNODES=6                                 # head + 5 workers
N_GPUS_PER_NODE=8
TP=4                                     # 48 NPU / 4 = 12 replicas
RAY_PORT=6379
RL_INSIGHT_PORT=18080

MODEL=/root/hgq/ws/models/Llama3.1-8B-Instruct
DATASET=/root/hgq/ws/data/swe_bench_train_model.parquet
MAX_SAMPLES=64
RES_LEN=8000
GPU_MEM_UTIL=0.8

export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-26100}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# OpenYuanrong sandbox creds (only reverse-tunnel provider); passed to workers
# via `docker exec -e`.
: "${OPENYUANRONG_SERVER_ADDRESS:?Set OPENYUANRONG_SERVER_ADDRESS}"
: "${OPENYUANRONG_TOKEN:?Set OPENYUANRONG_TOKEN}"

# =====================================================================
# Helpers
# =====================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

head_ip() {
    hostname -I | awk '{print $1}'
}

worker_exec() {
    local host=$1; shift
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
        "docker exec ${WORKER_CONTAINER} $*"
}

# worker_exec + env injected via `docker exec -e` (baked into the raylets).
worker_exec_with_env() {
    local host=$1 hip=$2; shift 2
    ssh -o StrictHostKeyChecking=no "${host}" "docker exec \
        -e VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO} \
        -e VERL_RL_INSIGHT_ENABLE=1 \
        -e RL_INSIGHT_SERVER_URL=http://${hip}:${RL_INSIGHT_PORT} \
        -e OPENYUANRONG_SERVER_ADDRESS=${OPENYUANRONG_SERVER_ADDRESS} \
        -e OPENYUANRONG_TOKEN=${OPENYUANRONG_TOKEN} \
        -e OPENYUANRONG_TUNNEL_SSL_VERIFY=${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0} \
        -e SANDBOX_NAME_PREFIX=${SANDBOX_NAME_PREFIX:-mini-swe-} \
        -e HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-26100} \
        -e RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        -e PYTHONHASHSEED=0 \
        ${WORKER_CONTAINER} $*"
}

# Fan-out a command across all workers in parallel.
worker_exec_all() {
    local cmd=$1
    for w in "${WORKERS[@]}"; do
        log "  -> ${w}: ${cmd}"
        worker_exec "${w}" "${cmd}" &
    done
    wait
}

# =====================================================================
# Step 1: passwordless-SSH + container reachability check
# =====================================================================
step1_ssh_check() {
    log "=== Step 1: SSH + container reachability check (${#WORKERS[@]} workers) ==="
    local ok=1
    for w in "${WORKERS[@]}"; do
        if worker_exec "${w}" "true" 2>/dev/null; then
            log "  ok ${w} (container ${WORKER_CONTAINER})"
        else
            log "  FAIL ${w} (ssh or docker exec ${WORKER_CONTAINER} failed)"
            ok=0
        fi
    done
    [[ ${ok} -eq 1 ]] || { log "ERROR: not all workers reachable; fix ssh/docker first."; exit 1; }
}

# =====================================================================
# Per-attempt cluster teardown / bring-up (all 6 nodes)
# =====================================================================
# Cleanup order: kill driver -> ray stop -> kill stray ray:: -> fuser (davinci + :9092).
KILL_DRIVER_CMD="ps aux | grep -E 'run_infer[.]sh|run_infer[.]py' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true"
KILL_RAY_CMD="ps aux | grep -E 'ray[:][:]|raylet|gcs_server' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true"
FUSER_CMD="bash -lc 'fuser -k /dev/davinci* 2>/dev/null || true; fuser -k 9092/tcp 2>/dev/null || true'"
NODE_CLEANUP="${KILL_DRIVER_CMD}; ray stop -f 2>/dev/null || true; ${KILL_RAY_CMD}; ${FUSER_CMD}"

clean_all_nodes() {
    log "  cleaning all ${NNODES} nodes (kill by pid, ray stop, free davinci + :9092)"
    eval "${NODE_CLEANUP}"
    worker_exec_all "${NODE_CLEANUP}"
}

setup_ray_cluster() {
    log "  starting ray cluster (nnodes=${NNODES})"
    export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
    export SANDBOX_NAME_PREFIX="${SANDBOX_NAME_PREFIX:-mini-swe-}"
    export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export PYTHONHASHSEED=0
    # idle-worker reaper must be disabled at ray start --head; ray.init's
    # _system_config is ignored when connecting to a running cluster.
    ray start --head --port="${RAY_PORT}" --temp-dir=/tmp/ray_head \
        --system-config='{"idle_worker_killing_time_threshold_ms": 2147483647}'

    # Custom --temp-dir hides the cluster from bare ray.init() discovery.
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

    log "  workers joining cluster (inside ${WORKER_CONTAINER})..."
    for w in "${WORKERS[@]}"; do
        log "    -> ${w} joining"
        worker_exec_with_env "${w}" "${HEAD_IP}" \
            "ray start --address=${HEAD_IP}:${RAY_PORT} --temp-dir=/tmp/ray_worker"
        sleep 3
    done
    wait

    log "waiting for cluster to settle..."
    sleep 10
    ray status || true
    log "alive nodes: $(ray nodes 2>/dev/null | grep -c alive)"
    ray list nodes 2>/dev/null | grep -E "ALIVE" | sed -E 's/^ *[0-9]+ +([0-9a-f]+) +([0-9.]+).*/  \1 \2/'
    local nconn
    nconn=$(ray list nodes 2>/dev/null | grep -c ALIVE)
    [[ "${nconn}" -eq "${NNODES}" ]] || { log "ERROR: got ${nconn} nodes, expected ${NNODES}"; return 1; }
    log "  cluster up: ${nconn}/${NNODES} nodes alive"
}

# =====================================================================
# Step 2: rl-insight server — started once, up for the whole matrix
# =====================================================================
step2_rl_insight() {
    log "=== Step 2: rl-insight server on head (${HEAD_IP}:${RL_INSIGHT_PORT}), up for the whole matrix ==="
    export VERL_RL_INSIGHT_ENABLE=1
    export RL_INSIGHT_SERVER_URL="http://${HEAD_IP}:${RL_INSIGHT_PORT}"
    rl-insight server stop
    rl-insight server start --detach 2>/dev/null || true   # already-running is fine
    trap 'rl-insight server stop 2>/dev/null || true' EXIT
    log "rl-insight up; workers scrape via RL_INSIGHT_SERVER_URL=${RL_INSIGHT_SERVER_URL}"
}

# =====================================================================
# Step 3: ascend-exps matrix (each run_infer spans all 6 nodes)
# =====================================================================

archive_rl_insight() {
    local exp_id=${1:-}
    if [ -z "${exp_id}" ]; then
        log " (skip rl-insight archive: EXP_ID empty)"
        return 0
    fi
    local dest="${ARCHIVE_ROOT}/${exp_id}"
    mkdir -p "${dest}"
    local tgz="{dest}/rl-insight-data.tgz"
    log " archiving rl-insight data -> ${tgz}"
    if [ -d /root/.rl-insight/data ]; then
        ( cd /root/.rl-insight/ && tar czf "${tgz}" data ) 2>/dev/null \
            && log " rl-insight archived ($(du -h "${tgz}" 2>/dev/null | cut -f1))" \
            || log " WARNING: rl-insight archive failed"
    fi
}

run_experiment() {
    local log_file=$1
    shift

    local log_mtime_before=$(stat -c %Y "${log_file}" 2>/dev/null || echo 0)

    while ! grep -q "${TARGET}" "${log_file}" 2>/dev/null; do
        set +e
        clean_all_nodes
        setup_ray_cluster
        npu-smi info 2>/dev/null | tail -3 || true
        log "  running -> ${log_file}"

        bash "${REPO_ROOT}/examples/llm_router/run_infer.sh" \
            --model-path "${MODEL}" \
            --data-path "${DATASET}" \
            --task-config "${REPO_ROOT}/examples/llm_router/task_config_mini_swe_agent.yaml" \
            --device ascend \
            --nnodes "${NNODES}" \
            --n-gpus-per-node "${N_GPUS_PER_NODE}" \
            --tp "${TP}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --response-length "${RES_LEN}" \
            --max-model-len "${CONTEXT}" \
            --max-samples "${MAX_SAMPLES}" \
            --n 8 \
            --shuffle \
            --concurrency "${CONCURRENCY}" \
            --kv-events \
            "$@" > "${log_file}" 2>&1 || log "  (run failed, will retry)"
    done
    log "experiment resolved ${log_file}"
    local log_mtime_after=$(stat -c %Y "$(log_file)" 2>/dev/null || echo 0)
    if [ "${log_mtime_after}" != "${log_mtime_before}" ]; then
        archive_rl_insight "${EXP_ID}"
    else
        log " (log unchanged - resolved from previous run, skip archive)"
    fi
}

step3_matrix() {
    log "=== Step 3: ascend-exps matrix (nnodes=${NNODES}, tp=${TP}, replicas=$((NNODES*N_GPUS_PER_NODE/TP)), per-run ray cluster) ==="

    local concurrencys=(16 24 32 128 192 256)
    local contexts=(16384 32768 64000 128000)
    export TARGET="inference summary"

    for CONCURRENCY in "${concurrencys[@]}"; do
        for CONTEXT in "${contexts[@]}"; do
            local EXP_ID="infer-sticky-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}-n${NNODES}"
            local LOG_FILE="${EXP_ID}.log"
            export SWE_AGENT_TRAJECTORY_DIR="${ARCHIVE_ROOT}/${EXP_ID}/trajectories"
            export UNI_AGENT_GATEWAY_REJECTED_DIR="${ARCHIVE_ROOT}/${EXP_ID}/rejected_requests"
            mkdir -p "${SWE_AGENT_TRAJECTORY_DIR}" "${UNI_AGENT_GATEWAY_REJECTED_DIR}"
            export SWE_AGENT_DUMP_TRAJECTORIES="${EXP_ID}.traj"
            log "sticky concurrency=${CONCURRENCY} context=${CONTEXT} (traj -> ${ARCHIVE_ROOT}/${EXP_ID}/trajectories)"
            run_experiment "${LOG_FILE}" \
                --slow-cut least-inflight \
                --overload-mode None

            local lts=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
            for lt in "${lts[@]}"; do
                local EXP_ID="infer-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x8-${CONCURRENCY}x${CONTEXT}-n${NNODES}"
                local LOG_FILE="${EXP_ID}.log"
                export SWE_AGENT_TRAJECTORY_DIR="${ARCHIVE_ROOT}/${EXP_ID}/trajectories"
                export UNI_AGENT_GATEWAY_REJECTED_DIR="${ARCHIVE_ROOT}/${EXP_ID}/rejected_requests"
                mkdir -p "${SWE_AGENT_TRAJECTORY_DIR}" "${UNI_AGENT_GATEWAY_REJECTED_DIR}"
                export SWE_AGENT_DUMP_TRAJECTORIES="${EXP_ID}.traj"
                log "kvcaware-lt${lt} concurrency=${CONCURRENCY} context=${CONTEXT} (traj -> ${ARCHIVE_ROOT}/${EXP_ID}/trajectories)"
                run_experiment "${LOG_FILE}" \
                    --slow-cut capacity-token-aware \
                    --overload-mode kv_cache_usage_perc \
                    --load-threshold "${lt}"
            done
        done
    done
    log "=== matrix complete ==="
}

# =====================================================================
# Main
# =====================================================================
step1_ssh_check
HEAD_IP=$(head_ip)
export HEAD_IP
step2_rl_insight
step3_matrix
