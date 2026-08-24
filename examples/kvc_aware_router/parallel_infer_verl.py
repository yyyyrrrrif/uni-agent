"""Parallel agent inference over a verl-launched engine, through the training path.

This script is ``parallel_infer_verl.py`` fused with the KV-cache-aware router
capability from ``verl/examples/kvc_aware_router/parallel_infer.py``. It keeps the
uni-agent framework path unchanged (AgentFrameworkRolloutAdapter + gateway sessions +
``--task-config`` + TransferQueue + ``_read_rm_scores``), and layers the kvcaware router
on top so rollouts route by KV-cache occupancy and engine load instead of round-robin.

    verl LLMServerManager (vLLM / SGLang, kvcaware router)   <-- new
    ->  AgentFrameworkRolloutAdapter.generate_sequences   (fire-and-forget -> TQ)
          ->  Gateway sessions (per-session OpenAI-compatible endpoints)
          ->  uni_agent.framework.task_runner.run_task  ->  uni_agent task
    ->  per-trajectory records written to TransferQueue

The per-sample score is the trainer's own ``rm_scores`` read back from TQ: ``run_task``
(``report_reward=True``) posts the task reward to its session, and the framework writes
it as ``reward_score`` -- no external reward model. Fan-out is ``rollout.n`` (``--n``),
with no resolved/wrong-answer/timeout bucketing (just mean ``rm_scores``).

KV-cache-aware knobs added on top of the base script:

  --kv-events          vLLM kv-events zmq publisher; the kvcaware collector's load signal
                       (retained-cache occupancy). Usually paired with the router.
  --load-threshold     KVCAware strategy[0] load_threshold (overload when load > threshold).
  --slow-cut           strategy[0] slow_cut fallback scoring mode.
  --overload-mode      strategy[0] overload_mode for the sticky short-circuit.
  --do-shortcut / --no-do-shortcut   strategy[0] do_shortcut master switch.
  --alpha              strategy[0] alpha (cache vs load blend).
  --max-num-seqs       engine max concurrent sequences.
  --device             gpu/ascend (selects the mooncake connector class).
  --enable-mooncake    attach MooncakeStoreConnector for cross-replica KV sharing.
  --mooncake-config-path   mooncake config JSON (used with --enable-mooncake).

Each router strategy override falls back to ``kvcaware.yaml`` when the flag is omitted.
``--task-config`` is required (same YAML shape as ``parallel_infer_api.py``); the policy
endpoint is the gateway session, bound by the runner, not a flag.

Example (single node, 4-way tensor parallel, kvcaware router + kv-events)::

    python examples/inference/parallel_infer_verl_new.py \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
        --task-config examples/inference/task_config_openyuanrong.yaml \
        --tool-parser qwen3_coder --tensor-parallel-size 4 \
        --max-model-len 32768 --kv-events --limit 8
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import OmegaConf

import verl

try:
    import transfer_queue as tq
except ImportError:  # fall back to verl's shim (mock raises a clear error if TQ is missing)
    from verl.utils.transferqueue_utils import tq

from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
from uni_agent.tasks import TaskConfigResolver
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.llm_server import LLMServerManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 128))
PARTITION_ID = "val"

DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_RESPONSE_LENGTH = 65536
DEFAULT_PROMPT_LENGTH = 4096

# Ray's default idle-worker reaper (~10 s) kills agent workers between dispatch
# gaps, ending the job prematurely. Use a very large threshold so long-running
# agent loops are not interrupted. (Mirrors kvc_aware_router/parallel_infer.py.)
_RAY_IDLE_WORKER_TIMEOUT_MS = int(os.getenv("RAY_IDLE_WORKER_TIMEOUT_MS", str(2**30 - 1)))


def _rule(text: str = "", width: int = 50, ch: str = "-") -> str:
    """A centered-title horizontal rule."""
    if not text:
        return ch * width
    pad = max(0, width - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


def init_config(args: argparse.Namespace, *, task_configs: list[dict], served_model_name: str):
    """Compose verl's ``ppo_trainer`` config with the KV-cache-aware router and override
    the engine + framework knobs.

    This is ``parallel_infer_verl.py``'s ``init_config`` plus the kvcaware router +
    engine-kwargs injection from ``verl/examples/kvc_aware_router/parallel_infer.py``.
    The framework/TQ path (``rollout.custom.agent_framework``, ``transfer_queue``) is
    preserved untouched; only the router + engine layers are layered on.
    """
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    # Compose the kvcaware router group under ``rollout.router_config``. Without this
    # override, ``router_config.strategies`` is absent and the strategy[0] overrides
    # below have nothing to write to.
    overrides = [
        "rollout/router@actor_rollout_ref.rollout.router_config=kvcaware",
        "actor_rollout_ref.rollout.router_strategy=kvcaware",
    ]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)

    rollout = config.actor_rollout_ref.rollout

    model_cfgs = [entry.get("agent", {}).get("model", {}) for entry in task_configs]
    temperature = model_cfgs[0].get("temperature", DEFAULT_TEMPERATURE)
    top_p = model_cfgs[0].get("top_p", DEFAULT_TOP_P)
    rollout.temperature = temperature
    rollout.top_p = top_p
    rollout.val_kwargs.temperature = temperature
    rollout.val_kwargs.top_p = top_p

    # response_length = the agent's episode token budget. By default it derives from
    # the task config's max_total_tokens (the full prompt+gen context the loop may
    # consume; DEFAULT_RESPONSE_LENGTH is the fallback), keeping parity with the base
    # parallel_infer_verl.py. --response-length (when passed) overrides it explicitly,
    # mirroring kvc_aware_router/parallel_infer.py.
    # max_total_tokens = max(
    #     (m.get("max_total_tokens", DEFAULT_RESPONSE_LENGTH) for m in model_cfgs),
    #     default=DEFAULT_RESPONSE_LENGTH,
    # )
    # response_length = int(max_total_tokens)

    # Fan-out: the framework runs rollout.n gateway sessions per prompt.
    rollout.n = max(1, args.n)
    rollout.val_kwargs.n = rollout.n

    # Hardware.
    rollout.nnodes = args.nnodes
    rollout.n_gpus_per_node = args.n_gpus_per_node
    config.trainer.nnodes = args.nnodes
    config.trainer.n_gpus_per_node = args.n_gpus_per_node

    # Model + engine.
    config.actor_rollout_ref.model.path = os.path.expanduser(args.model_path)
    rollout.name = args.engine
    rollout.mode = "async"
    rollout.agent.num_workers = args.num_workers
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.gpu_memory_utilization = args.gpu_memory_utilization
    # Cap the engine's context window. The agent's per-episode token budget
    # (max_total_tokens) must be <= this; otherwise the loop keeps extending the
    # transcript past the engine limit and vLLM rejects the request with HTTP 400
    # ("Prompt length ... leaves no room to generate within the model's maximum
    # context length").
    rollout.max_model_len = args.max_model_len
    # kvcaware additions: engine concurrency + expose metrics on /metrics for the
    # kvcaware collector (which polls them as one of its load signals).
    rollout.max_num_seqs = args.max_num_seqs
    rollout.disable_log_stats = False

    # Gateway tool-call parser: the gateway decodes tool calls from raw tokens, so
    # this must match the model's chat template (the analog of vLLM's
    # --tool-call-parser, e.g. qwen3_coder for Qwen3-Coder, hermes for Qwen3).
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.format", args.tool_parser, force_add=True)

    # vLLM engine kwargs: MFU metric (always on) + optional mooncake connector / kv-events.
    vllm_kwargs: dict = {"enable_mfu_metrics": True}
    if args.enable_mooncake:
        # Cross-replica KV sharing via mooncake (config via MOONCAKE_CONFIG_PATH env, not
        # extra_config). The connector class differs by backend: GPU build uses
        # "MooncakeStoreConnector"; vllm-ascend uses "MooncakeConnectorStoreV1".
        mooncake_connector = "MooncakeConnectorStoreV1" if args.device == "ascend" else "MooncakeStoreConnector"
        vllm_kwargs["kv_transfer_config"] = {
            "kv_connector": mooncake_connector,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {},
        }
    if args.kv_events:
        # vLLM kv-events (zmq publisher) -- KVCAware router load signal (retained-cache
        # occupancy). Endpoint ports are placeholders (verl assigns ephemeral).
        vllm_kwargs["kv-events-config"] = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": "tcp://*:0",
            "replay_endpoint": "tcp://*:0",
        }
    rollout.engine_kwargs = {"vllm": vllm_kwargs}

    # Optional KVCAware router strategy[0] overrides -- each falls back to the
    # kvcaware.yaml value when the flag is omitted. The override above composes
    # router_config, so strategies is a non-empty list.
    strat0 = rollout.router_config.strategies[0]
    if args.alpha is not None:
        strat0.alpha = args.alpha
    if args.load_threshold is not None:
        strat0.load_threshold = args.load_threshold
    if args.slow_cut is not None:
        strat0.slow_cut = args.slow_cut
    if args.overload_mode is not None:
        strat0.overload_mode = args.overload_mode
    if args.do_shortcut is not None:
        strat0.do_shortcut = args.do_shortcut

    agent_framework_cfg = {
        "gateway_count": args.gateway_count,
        "agent_runners": {
            "task": {
                "runner_fqn": "uni_agent.framework.task_runner.run_task",
                "dispatch_mode": "ray_task",
                "max_concurrent_sessions": max(0, args.concurrency),
                "runner_kwargs": {
                    "task_config_path": args.task_config,
                    "model_name": served_model_name,
                    "report_reward": True,
                },
            }
        },
    }
    agent_framework_cfg["log_dir"] = args.log_dir
    OmegaConf.update(config, "actor_rollout_ref.rollout.custom.agent_framework", agent_framework_cfg, force_add=True)

    # TransferQueue carries the rollout trajectories (and their rm_scores).
    OmegaConf.update(config, "transfer_queue.enable", True, force_add=True)

    # Data.
    config.data.return_raw_chat = True
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length

    return config


def _build_prompts(samples: list, uids: list):
    """Assemble the TensorDict batch the framework's ``generate_sequences`` expects."""
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [sample.get("prompt") for sample in samples],
            "uid": list(uids),
            "tools_kwargs": [sample["extra_info"]["tools_kwargs"] for sample in samples],
        },
        non_tensor_dict={"global_steps": None, "validate": True},
    )


def _read_rm_scores(uids: list, *, partition_id: str = PARTITION_ID) -> dict:
    """Read each session's final trajectory back from TQ and score it."""
    input_uids = set(uids)
    listing = tq.kv_list() or {}
    partition = listing.get(partition_id, {}) or {}

    # (uid, session) -> (max_index, key); also collect every key we touch for cleanup.
    final: dict[tuple[str, str], tuple[int, str]] = {}
    traj_keys: list[str] = []
    uid_status: dict[str, str] = {}
    for key, tag in partition.items():
        tag = tag or {}
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            # uid-level status marker (uid has no underscores: it is a uuid4 hex-with-dashes).
            if key in input_uids:
                uid_status[key] = tag.get("status")
            continue
        uid, session, index_str = parts
        if uid not in input_uids or tag.get("status") != "success":
            continue
        try:
            index = int(index_str)
        except ValueError:
            continue
        traj_keys.append(key)
        session_key = (uid, session)
        if session_key not in final or final[session_key][0] < index:
            final[session_key] = (index, key)

    # Deterministic order so scores align with the (uid, session) they came from.
    final_items = sorted(final.items())
    final_keys = [key for _, (_, key) in final_items]
    final_sessions = [session_key for session_key, _ in final_items]

    per_uid: dict[str, list[float]] = defaultdict(list)
    scores: list[float] = []
    if final_keys:
        data = tq.kv_batch_get(keys=final_keys, partition_id=partition_id, select_fields=["rm_scores"])
        scores = [float(s) for s in data["rm_scores"].sum(dim=-1).tolist()]
        for (uid, _session), score in zip(final_sessions, scores, strict=True):
            per_uid[uid].append(score)

    uid_keys = [uid for uid in input_uids if uid in uid_status]
    return {
        "scores": scores,
        "per_uid": dict(per_uid),
        "uid_status": uid_status,
        "final_keys": final_keys,
        "traj_keys": traj_keys,
        "uid_keys": uid_keys,
    }


def _report(
    read: dict, *, wall: float, num_prompts: int, n: int, args: argparse.Namespace, served_model_name: str
) -> None:
    """Print the mean-rm_scores summary and optionally persist a JSON result file."""
    scores = read["scores"]
    per_uid = read["per_uid"]
    uid_status = read["uid_status"]

    expected = num_prompts * n
    num_scored = len(scores)
    mean_score = float(np.mean(scores)) if scores else 0.0
    # Per-prompt score = mean over that prompt's sessions; then averaged over prompts.
    prompt_means = [float(np.mean(v)) for v in per_uid.values() if v]
    mean_over_prompts = float(np.mean(prompt_means)) if prompt_means else 0.0
    failed_uids = sum(1 for status in uid_status.values() if status != "finished")

    summary = "\n".join(
        [
            "",
            _rule("inference summary"),
            f"  mean rm_score      {mean_score:>8.4f}   (over {num_scored} sessions)",
            f"  mean over prompts  {mean_over_prompts:>8.4f}   (over {len(prompt_means)} prompts)",
            f"  scored sessions    {num_scored:>4} / {expected:<4} ({num_prompts} prompts x n={n})",
            f"  failed prompts     {failed_uids:>4}",
            _rule(f"wall {wall:.1f}s"),
            "",
        ]
    )
    print(summary)

    if args.result_path:
        result_path = os.path.expanduser(args.result_path)
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        payload = {
            "model_path": os.path.expanduser(args.model_path),
            "served_model_name": served_model_name,
            "data_path": os.path.expanduser(args.data_path),
            "task_config": args.task_config,
            "n": n,
            "num_prompts": num_prompts,
            "num_scored_sessions": num_scored,
            "mean_rm_score": mean_score,
            "mean_rm_score_over_prompts": mean_over_prompts,
            "scores": scores,
            "scores_by_uid": per_uid,
        }
        with open(result_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"wrote result file to: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel agent inference over a verl-launched engine with the KV-cache-aware "
            "router (framework + TQ path)."
        )
    )

    # Input / output.
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", os.path.expanduser("~/data/swe_agent/swe_bench_verified.parquet")),
        help="Path to the input dataset (Parquet format).",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=os.path.expanduser("~/models/Qwen3-Coder-30B-A3B-Instruct"),
        help="Local model checkpoint the engine loads.",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model name sent on chat-completions requests (default: basename of --model-path).",
    )
    parser.add_argument(
        "--task-config",
        required=True,
        help="Path to a YAML task config: one ``- name: ...`` entry or a list of them (required). "
        "run_task routes each row to the entry whose 'name' matches the row's task; all agent/model "
        "knobs (sampling, max_total_tokens, max_steps, ...) come from it. The endpoint is bound to the "
        "gateway session.",
    )
    parser.add_argument(
        "--result-path",
        default=None,
        help="Optional path to write a JSON result file (mean rm_score and per-session scores).",
    )
    parser.add_argument(
        "--limit",
        "--max-samples",
        dest="limit",
        type=int,
        default=None,
        help="Only run the first N samples (smoke testing); omit for the full dataset.",
    )

    parser.add_argument(
        "--n", type=int, default=1, help="Rollout sessions per instance (rollout.n; scores average over all)."
    )

    # Sampling (mirrors kvc_aware_router/parallel_infer.py).
    parser.add_argument("--prompt-length", type=int, default=4096, help="Maximum prompt length (tokens).")
    parser.add_argument("--response-length", type=int, default=8192, help="Maximum response length (tokens).")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before slicing (--limit / --n). Aligns with fully_async data.shuffle.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle.")

    # Engine / hardware.
    parser.add_argument(
        "--engine",
        default="vllm",
        choices=["vllm", "sglang"],
        help="Inference engine backend.",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Number of agent rollout workers.")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to run the engine on.")
    parser.add_argument("--n-gpus-per-node", type=int, default=8, help="Number of GPUs per node.")
    parser.add_argument(
        "--tensor-parallel-size", "--tp", dest="tensor_parallel_size", type=int, default=4, help="Tensor parallel size."
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Engine GPU memory fraction.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("MAX_MODEL_LEN", "0")) or None,
        help=(
            "Maximum model context length (tokens) passed to the engine. Must be >= "
            "the task config's max_total_tokens; otherwise multi-step episodes crash "
            "with HTTP 400 once the transcript grows past the engine limit."
        ),
    )
    parser.add_argument(
        "--gateway-count",
        type=int,
        default=4,
        help="Number of gateway actors fronting the engine (each serves many concurrent sessions).",
    )
    parser.add_argument(
        "--tool-parser",
        default=os.getenv("TOOL_PARSER", "qwen3_coder"),
        help="Gateway tool-call parser; MUST match the model's chat template (e.g. qwen3_coder, hermes).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=GLOBAL_CONCURRENCY,
        help="Max in-flight gateway sessions for the runner (runner.max_concurrent_sessions; env GLOBAL_CONCURRENCY).",
    )
    parser.add_argument(
        "--log-dir",
        default=os.getenv("UNI_AGENT_LOG_DIR", "/tmp/uni_agent_logs"),
        help="Root directory for per-session logs and trajectories; use an empty value to disable.",
    )

    # ---- KV-cache-aware router (fused from kvc_aware_router/parallel_infer.py) ----
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences per engine.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "ascend"],
        help="Target backend: 'gpu' or 'ascend' (selects the mooncake connector class).",
    )
    parser.add_argument(
        "--enable-mooncake",
        action="store_true",
        help="Attach MooncakeStoreConnector for cross-replica KV sharing (a mooncake master must run separately).",
    )
    parser.add_argument(
        "--mooncake-config-path",
        type=str,
        default="mooncake_config.json",
        help="Path to the mooncake config JSON (used with --enable-mooncake).",
    )
    parser.add_argument(
        "--kv-events",
        action="store_true",
        help="Enable vLLM kv-events zmq publisher for retained-cache occupancy collection. "
        "Required for KVCAware router load signal and standalone collector metrics.",
    )

    # KVCAware router strategy[0] overrides (each falls back to kvcaware.yaml when omitted).
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="KVCAware strategy[0] alpha (cache vs load blend, [0,1]). Overrides kvcaware.yaml when set.",
    )
    parser.add_argument(
        "--load-threshold",
        type=float,
        default=None,
        help="KVCAware strategy[0] load_threshold (overload when load > threshold, (0,1)). "
        "Overrides kvcaware.yaml when set.",
    )
    parser.add_argument(
        "--slow-cut",
        type=str,
        choices=["locality-aware", "prefix-load-aware", "least-inflight", "capacity-token-aware"],
        default=None,
        help="KVCAware strategy[0] slow_cut fallback scoring mode. Overrides kvcaware.yaml when set.",
    )
    parser.add_argument(
        "--overload-mode",
        type=str,
        choices=["None", "kv_cache_usage_perc", "kv_load"],
        default=None,
        help="KVCAware strategy[0] overload_mode for the sticky short-circuit. Overrides kvcaware.yaml when set.",
    )
    parser.add_argument(
        "--do-shortcut",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="KVCAware strategy[0] do_shortcut master switch for the sticky short-circuit. "
        "Use --do-shortcut to enable, --no-do-shortcut to disable. Overrides kvcaware.yaml when set.",
    )

    args = parser.parse_args()

    # Mooncake connector reads MOONCAKE_CONFIG_PATH (not extra_config). Set before
    # ray.init so Ray-spawned workers inherit it.
    if args.enable_mooncake and args.mooncake_config_path:
        os.environ["MOONCAKE_CONFIG_PATH"] = os.path.expanduser(args.mooncake_config_path)

    # Disable Ray's idle-worker reaper so agent workers survive dispatch gaps
    # (default ~10 s threshold would kill them prematurely).
    ray.init(_system_config={"idle_worker_killing_time_threshold_ms": _RAY_IDLE_WORKER_TIMEOUT_MS})

    resolver = TaskConfigResolver.from_file(args.task_config)
    served_model_name = args.served_model_name or os.path.basename(os.path.expanduser(args.model_path).rstrip("/"))

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    if args.shuffle:
        logger.info("Shuffling dataset (seed=%d) before sampling", args.seed)
        dataset = dataset.shuffle(seed=args.seed)
    samples = dataset.to_list()
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected; exiting")
        return
    n = max(1, args.n)

    task_configs = list(resolver.defaults_by_name.values())

    logger.info(f"loaded {len(samples)} prompts (x n={n} sessions each) from {args.data_path}")

    # 1. TransferQueue + verl inference engine with the KV-cache-aware router.
    logger.info("initializing configuration, TransferQueue, and LLMServerManager (kvcaware router)...")
    config = init_config(args, task_configs=task_configs, served_model_name=served_model_name)
    tq.init(config.transfer_queue)
    llm_server_manager = LLMServerManager.create(config=config)

    # 2. Framework rollout adapter over the engine.
    adapter = AgentFrameworkRolloutAdapter.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
    )

    # 3. Submit the batch and wait for every trajectory to land in TQ.
    uids = [str(uuid4()) for _ in samples]
    prompts = _build_prompts(samples, uids)
    logger.info("starting inference...")
    begin_time = time.time()
    adapter.generate_sequences_and_wait(prompts)
    wall = time.time() - begin_time

    # 4. Read rm_scores back from TQ and report.
    read = _read_rm_scores(uids, partition_id=PARTITION_ID)
    _report(read, wall=wall, num_prompts=len(samples), n=n, args=args, served_model_name=served_model_name)


if __name__ == "__main__":
    main()

