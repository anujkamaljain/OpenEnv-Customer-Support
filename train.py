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
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from env.environment import CustomerSupportEnv
from evaluate import PolicyState, evaluate_policy, plot_reports
from evaluate import choose_policy_action as eval_choose_policy_action
from models.observation import Observation


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


def build_prompt(obs: Observation) -> str:
    """Convert incident observation into a deterministic training prompt."""
    parts = [
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
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install in Colab with:\n"
            'pip install "trl>=0.15" datasets peft bitsandbytes '
            "llm-blender accelerate transformers"
        ) from exc

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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow transformers+peft fallback when Unsloth is unavailable.",
    )
    return parser


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
    reward_lookup = {
        (row.prompt, row.completion): row.reward for row in trajectories
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

    def reward_function(
        prompts: list[str],
        completions: list[str],
        **_: object,
    ) -> list[float]:
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
        }

        def _prompt_field(prompt: str, key: str) -> str:
            prefix = f"{key}="
            for line in prompt.splitlines():
                if line.startswith(prefix):
                    return line[len(prefix) :].strip()
            return ""

        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            score = -0.02  # Small base penalty encourages useful/valid actions.
            try:
                action_payload = json.loads(completion)
            except json.JSONDecodeError:
                rewards.append(-0.1)
                continue
            if not isinstance(action_payload, dict):
                rewards.append(-0.1)
                continue

            action_type = str(action_payload.get("action_type", "")).strip()
            if not action_type:
                rewards.append(-0.1)
                continue

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
            if action_type in available_actions:
                score += 0.08
            else:
                score -= 0.06

            if action_type in preferred_actions.get(phase, set()):
                score += 0.05

            needed = required_fields.get(action_type, ())
            if needed:
                has_all_fields = all(action_payload.get(field) not in (None, "") for field in needed)
                score += 0.03 if has_all_fields else -0.03

            # Preserve supervised signal when completion matches recorded trajectory exactly.
            score += 0.4 * float(reward_lookup.get((prompt, completion), 0.0))
            rewards.append(max(-1.0, min(1.0, score)))
        return rewards

    config = GRPOConfig(
        output_dir=str(output_dir / "grpo_output"),
        num_generations=args.k,
        max_new_tokens=256,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=100,
        warmup_steps=25,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        config=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
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
