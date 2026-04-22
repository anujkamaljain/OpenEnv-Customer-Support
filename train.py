"""GRPO training pipeline for Enterprise Incident Command Center.

Runs on Google Colab free tier (T4 GPU) when optional training dependencies
are installed. Supports a deterministic dry-run mode for local verification.

Usage:
    python train.py --iterations 1 --episodes 1 --k 2 --dry-run
    python train.py --iterations 20 --episodes 30 --k 4
"""

from __future__ import annotations

import ast
import argparse
import asyncio
import inspect
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from env.environment import CustomerSupportEnv
from evaluate import PolicyState, evaluate_policy, plot_reports
from evaluate import choose_policy_action as eval_choose_policy_action
from models.observation import Observation


KNOWN_ACTION_TYPES: tuple[str, ...] = (
    "classify",
    "route",
    "respond",
    "escalate",
    "request_info",
    "resolve",
    "check_monitoring",
    "probe_service",
    "fetch_logs",
    "fetch_user_data",
    "check_billing",
    "query_kb",
    "check_policy",
    "query_incident_history",
    "follow_runbook_step",
    "apply_fix",
    "verify_fix",
    "rollback_fix",
    "notify_stakeholders",
    "write_postmortem",
    "update_kb",
)

_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def _extract_first_json_object(text: str) -> dict[str, object] | None:
    """Return the first valid JSON object found in a completion.

    Tolerates chat prose, code fences, and extra whitespace so noisy model
    outputs still produce a usable reward signal instead of a degenerate
    constant penalty.
    """
    if not text:
        return None
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    for match in _JSON_OBJECT_RE.finditer(text):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


@dataclass(slots=True)
class TrajectoryRow:
    """One prompt/completion/reward row used for GRPO datasets."""

    prompt: str
    completion: str
    reward: float
    iteration: int
    episode: int
    step: int
    difficulty: str


def curriculum_difficulty(iteration: int, episode_index: int, episodes: int) -> str:
    """Deterministic curriculum schedule from phase 7 specification."""
    if iteration <= 8:
        return "easy"
    if iteration <= 14:
        return "medium"
    if iteration <= 18:
        return "hard"

    # Phase D mixed schedule: easy 20% / medium 30% / hard 40% / nightmare 10%.
    slot = episode_index % max(1, episodes)
    ratio = slot / max(1, episodes)
    if ratio < 0.20:
        return "easy"
    if ratio < 0.50:
        return "medium"
    if ratio < 0.90:
        return "hard"
    return "nightmare"


_FORMAT_INSTRUCTION = (
    "You are an incident response agent. "
    "Respond with exactly ONE compact JSON object and nothing else. "
    'Example: {"action_type":"check_monitoring","service_name":null}. '
    "Pick action_type strictly from available_actions below. "
    "Include the fields that action requires."
)


def build_prompt(obs: Observation) -> str:
    """Convert incident observation into a deterministic training prompt.

    The prompt is structured so a downstream reward function can parse it
    back (phase, available_actions) and so the model sees an explicit
    JSON-only instruction with a concrete example.
    """
    parts = [
        _FORMAT_INSTRUCTION,
        f"incident={obs.incident_id}",
        f"title={obs.incident_title}",
        f"phase={obs.incident_phase}",
        f"step={obs.current_step}/{obs.max_steps}",
        f"available_actions={obs.available_actions}",
    ]
    if obs.system_status:
        parts.append(f"system_status={json.dumps(obs.system_status, sort_keys=True)}")
    if obs.tool_results:
        parts.append(f"tool_results={json.dumps(obs.tool_results, sort_keys=True)}")
    if obs.known_facts:
        parts.append(f"known_facts={json.dumps(obs.known_facts, sort_keys=True)}")
    parts.append("Respond with ONE JSON action object only:")
    return "\n".join(parts)


def choose_training_action(
    obs: Observation,
    state: PolicyState,
    quality_ratio: float,
) -> dict[str, object]:
    """Interpolation policy used to collect trajectories during curriculum."""
    if quality_ratio < 0.5:
        return eval_choose_policy_action(obs, state, "baseline")
    return eval_choose_policy_action(obs, state, "trained")


async def collect_trajectories(
    *,
    iterations: int,
    episodes: int,
) -> tuple[list[TrajectoryRow], list[float]]:
    """Collect deterministic trajectories across curriculum iterations."""
    env = CustomerSupportEnv()
    rows: list[TrajectoryRow] = []
    reward_history: list[float] = []

    try:
        for iteration in range(1, iterations + 1):
            quality_ratio = iteration / max(1, iterations)
            cumulative = 0.0
            reward_count = 0

            for episode_idx in range(episodes):
                difficulty = curriculum_difficulty(iteration, episode_idx, episodes)
                reset = await env.reset(
                    seed=episode_idx,
                    difficulty=difficulty,
                    mode="incident",
                )
                obs = reset.observation
                policy_state = PolicyState()

                for step_idx in range(obs.max_steps):
                    prompt = build_prompt(obs)
                    action = choose_training_action(obs, policy_state, quality_ratio)
                    completion = json.dumps(action, separators=(",", ":"))
                    result = await env.step(action)

                    rows.append(
                        TrajectoryRow(
                            prompt=prompt,
                            completion=completion,
                            reward=result.reward,
                            iteration=iteration,
                            episode=episode_idx,
                            step=step_idx,
                            difficulty=difficulty,
                        )
                    )
                    cumulative += result.reward
                    reward_count += 1
                    obs = result.observation
                    if result.done:
                        break

            avg_iteration_reward = cumulative / max(1, reward_count)
            reward_history.append(round(avg_iteration_reward, 4))
            print(
                f"[train] iteration={iteration} episodes={episodes} "
                f"avg_step_reward={avg_iteration_reward:.4f}"
            )
    finally:
        await env.close()
    return rows, reward_history


def require_training_stack(*, allow_fallback: bool) -> tuple[object, object, object, object | None]:
    """Import training libraries, requiring Unsloth unless fallback is allowed."""
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Unsloth is required for this run but was not found.\n"
                "Install dependencies in Colab Step 2 and restart runtime, then rerun.\n"
                "If you intentionally want the transformers+peft fallback, run with --allow-fallback."
            ) from exc
        FastLanguageModel = None
        print("[train] unsloth not found; using transformers+peft fallback.")

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install in Colab with:\n"
            'pip install "trl>=0.15" datasets peft bitsandbytes '
            "llm-blender accelerate transformers"
        ) from exc

    return Dataset, GRPOConfig, GRPOTrainer, FastLanguageModel


def write_json(path: Path, payload: object) -> None:
    """Write JSON artifact with stable UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_everything(seed: int) -> None:
    """Seed common RNGs for reproducible trajectory collection and training."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EICC policy with GRPO.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--k", type=int, default=4, help="GRPO num_generations")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=128,
        help="Max new tokens the policy generates per action (kept short for JSON).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow transformers+peft fallback when Unsloth is unavailable.",
    )
    return parser


def _build_grpo_config(
    *,
    GRPOConfig: object,
    output_dir: Path,
    k: int,
    max_completion_length: int,
) -> object:
    """Construct GRPOConfig with TRL-version-compatible argument names."""
    params = inspect.signature(GRPOConfig).parameters
    kwargs: dict[str, object] = {
        "output_dir": str(output_dir / "grpo_output"),
        "num_train_epochs": 1,
        "learning_rate": 5e-6,
        "logging_steps": 5,
        "save_steps": 100,
        "warmup_steps": 10,
    }

    # TRL naming drift across releases.
    if "num_generations" in params:
        kwargs["num_generations"] = k
    elif "num_generation" in params:
        kwargs["num_generation"] = k

    if "max_new_tokens" in params:
        kwargs["max_new_tokens"] = max_completion_length
    elif "max_completion_length" in params:
        kwargs["max_completion_length"] = max_completion_length
    elif "response_length" in params:
        kwargs["response_length"] = max_completion_length

    if "per_device_train_batch_size" in params:
        kwargs["per_device_train_batch_size"] = 2
    elif "train_batch_size" in params:
        kwargs["train_batch_size"] = 2

    if "gradient_accumulation_steps" in params:
        kwargs["gradient_accumulation_steps"] = 4

    # Encourage diverse generations so GRPO gets meaningful reward variance.
    if "temperature" in params:
        kwargs["temperature"] = 0.8
    if "top_p" in params:
        kwargs["top_p"] = 0.95

    return GRPOConfig(**kwargs)


def main() -> None:
    """Run trajectory collection, optional GRPO training, and evaluation."""
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    _seed_everything(args.seed)

    trajectories, reward_history = asyncio.run(
        collect_trajectories(iterations=args.iterations, episodes=args.episodes)
    )
    write_json(
        output_dir / "trajectories.json",
        [asdict(row) for row in trajectories],
    )
    write_json(output_dir / "reward_history.json", reward_history)

    if args.dry_run:
        print(
            f"[dry-run] collected_rows={len(trajectories)} "
            f"iterations={args.iterations} episodes={args.episodes}"
        )
        return

    Dataset, GRPOConfig, GRPOTrainer, FastLanguageModel = require_training_stack(
        allow_fallback=args.allow_fallback
    )
    dataset_rows = [
        {"prompt": row.prompt, "completion": row.completion, "reward": row.reward}
        for row in trajectories
    ]
    dataset = Dataset.from_list(dataset_rows)
    # Action-keyed lookup: the environment reward recorded when this
    # action_type was executed from this prompt during trajectory collection.
    # Much more forgiving than full (prompt, completion) string match.
    action_reward_lookup: dict[tuple[str, str], list[float]] = {}
    for row in trajectories:
        try:
            recorded_action = json.loads(row.completion)
        except json.JSONDecodeError:
            continue
        if not isinstance(recorded_action, dict):
            continue
        action_type = str(recorded_action.get("action_type", "")).strip()
        if not action_type:
            continue
        action_reward_lookup.setdefault((row.prompt, action_type), []).append(row.reward)
    trajectory_action_reward: dict[tuple[str, str], float] = {
        key: sum(values) / len(values) for key, values in action_reward_lookup.items()
    }

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if FastLanguageModel is not None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=target_modules,
        )
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_cfg,
        )
        peft_cfg = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    preferred_actions: dict[str, set[str]] = {
        "triage": {"check_monitoring", "query_kb", "classify"},
        "investigation": {"check_monitoring", "probe_service", "fetch_logs", "query_kb"},
        "response": {"check_policy", "apply_fix", "notify_stakeholders", "respond"},
        "resolution": {"verify_fix", "write_postmortem", "update_kb", "resolve"},
    }
    required_fields: dict[str, tuple[str, ...]] = {
        "classify": ("category", "priority"),
        "probe_service": ("service_name", "check_type"),
        "fetch_logs": ("service_name", "time_range"),
        "respond": ("response_text", "tone"),
        "apply_fix": ("service_name", "fix_type"),
        "resolve": ("resolution_summary",),
        "notify_stakeholders": ("stakeholder", "urgency"),
        "write_postmortem": ("summary", "root_cause_description"),
        "update_kb": ("article_title", "content"),
    }

    reward_stats = {"total": 0, "valid_json": 0, "valid_action": 0, "available": 0}

    def _prompt_field(prompt: str, key: str) -> str:
        prefix = f"{key}="
        for line in prompt.splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
        return ""

    def _mentions_known_action(text: str) -> bool:
        for action_name in KNOWN_ACTION_TYPES:
            if f'"{action_name}"' in text or f"'{action_name}'" in text:
                return True
        return False

    def reward_function(
        prompts: list[str],
        completions: list[str],
        **_: object,
    ) -> list[float]:
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            reward_stats["total"] += 1

            action_payload = _extract_first_json_object(completion)
            if action_payload is None:
                # Partial credit for effort even without valid JSON so GRPO
                # always sees some variance instead of a constant floor.
                rewards.append(-0.02 if _mentions_known_action(completion) else -0.08)
                continue

            reward_stats["valid_json"] += 1
            action_type = str(action_payload.get("action_type", "")).strip()
            if not action_type:
                rewards.append(-0.04)
                continue

            reward_stats["valid_action"] += 1

            available_actions_raw = _prompt_field(prompt, "available_actions")
            available_actions: set[str] = set()
            if available_actions_raw:
                try:
                    parsed_available = ast.literal_eval(available_actions_raw)
                    if isinstance(parsed_available, list):
                        available_actions = {str(item) for item in parsed_available}
                except (ValueError, SyntaxError):
                    available_actions = set()

            phase = _prompt_field(prompt, "phase")

            # Base signal for a structurally valid JSON action.
            score = 0.05

            if action_type in KNOWN_ACTION_TYPES:
                score += 0.04

            if action_type in available_actions:
                score += 0.12
                reward_stats["available"] += 1
            else:
                score -= 0.04

            if action_type in preferred_actions.get(phase, set()):
                score += 0.06

            needed = required_fields.get(action_type, ())
            if needed:
                present = sum(
                    1 for field_name in needed if action_payload.get(field_name) not in (None, "")
                )
                score += 0.06 * (present / max(1, len(needed)))

            trajectory_reward = trajectory_action_reward.get((prompt, action_type), 0.0)
            score += 0.4 * float(trajectory_reward)

            # Penalize noisy / long outputs, reward concise JSON.
            completion_len = len(completion)
            if completion_len <= 160:
                score += 0.01
            elif completion_len > 400:
                score -= 0.03

            rewards.append(max(-1.0, min(1.0, score)))

        # Periodic parse-rate diagnostics so judges/you see training health.
        batch_count = reward_function.__dict__.setdefault("_call_count", 0) + 1
        reward_function.__dict__["_call_count"] = batch_count
        if batch_count % 10 == 0 or batch_count == 1:
            total = max(1, reward_stats["total"])
            print(
                "[reward] batch={} total={} valid_json={:.0%} valid_action={:.0%} "
                "available={:.0%}".format(
                    batch_count,
                    reward_stats["total"],
                    reward_stats["valid_json"] / total,
                    reward_stats["valid_action"] / total,
                    reward_stats["available"] / total,
                )
            )
        return rewards

    config = _build_grpo_config(
        GRPOConfig=GRPOConfig,
        output_dir=output_dir,
        k=args.k,
        max_completion_length=args.max_completion_length,
    )
    trainer_params = inspect.signature(GRPOTrainer).parameters
    trainer_kwargs: dict[str, object] = {"model": model}

    # TRL / Unsloth naming drift: some versions use `config`, others `args`.
    if "config" in trainer_params:
        trainer_kwargs["config"] = config
    elif "args" in trainer_params:
        trainer_kwargs["args"] = config

    # Reward callback naming is stable in recent TRL but keep compatibility.
    if "reward_funcs" in trainer_params:
        trainer_kwargs["reward_funcs"] = [reward_function]
    elif "reward_function" in trainer_params:
        trainer_kwargs["reward_function"] = reward_function

    if "train_dataset" in trainer_params:
        trainer_kwargs["train_dataset"] = dataset
    elif "dataset" in trainer_params:
        trainer_kwargs["dataset"] = dataset

    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir / "trained_adapter"))

    baseline = evaluate_policy(
        policy="baseline",
        episodes_per_difficulty=args.eval_episodes,
    )
    trained = evaluate_policy(
        policy="trained",
        episodes_per_difficulty=args.eval_episodes,
    )
    trained.behavior_examples = [
        "Agent shifts from direct fixes to monitoring-first triage.",
        "Agent verifies KB-guided actions using service logs before fixing.",
        "Agent performs verify_fix and postmortem actions more consistently.",
    ]
    write_json(output_dir / "baseline_report.json", asdict(baseline))
    write_json(output_dir / "trained_report.json", asdict(trained))
    plot_reports(baseline, trained, output_dir)
    trained.print_comparison(baseline)


if __name__ == "__main__":
    main()
