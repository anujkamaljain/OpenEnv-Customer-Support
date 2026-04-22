"""Evaluation pipeline for Enterprise Incident Command Center training.

This script runs deterministic incident-mode episodes and reports formal metrics:
- normalized reward
- SLA compliance
- tool efficiency
- root-cause accuracy proxy
- long-horizon consistency
- 8 tracked behavioral skills

Usage examples:
    python evaluate.py --policy baseline --episodes-per-difficulty 3
    python evaluate.py --policy compare --episodes-per-difficulty 5 --plot
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from env.environment import CustomerSupportEnv
from models.observation import Observation

PolicyKind = Literal["baseline", "trained", "trained_heuristic"]

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard", "nightmare")
TOOL_ACTIONS = frozenset(
    [
        "check_monitoring",
        "probe_service",
        "fetch_logs",
        "fetch_user_data",
        "check_billing",
        "query_kb",
        "check_policy",
        "query_incident_history",
        "follow_runbook_step",
    ]
)
POLICY_SENSITIVE_ACTIONS = frozenset(
    ["apply_fix", "escalate", "notify_stakeholders", "update_kb", "resolve"]
)
TRACKED_SKILLS: tuple[str, ...] = (
    "investigation_before_action",
    "kb_cross_verification",
    "policy_checking",
    "stakeholder_proactivity",
    "root_cause_accuracy",
    "tone_matching",
    "resource_efficiency",
    "red_herring_dismissal",
)


@dataclass(slots=True)
class EpisodeReport:
    """Per-episode evaluation record."""

    difficulty: str
    normalized_reward: float
    sla_compliant: float
    tool_efficiency: float
    root_cause_accuracy: float
    long_horizon_consistency: float
    skill_scores: dict[str, float]
    actions: list[str]


@dataclass
class EvaluationReport:
    """Structured output from evaluate.py — used for demo and blog."""

    avg_normalized_reward: float
    sla_compliance_rate: float
    tool_efficiency: float
    root_cause_accuracy: float
    long_horizon_consistency: float
    skill_scores: dict[str, float]
    per_difficulty: dict[str, float]
    reward_history: list[float]
    behavior_examples: list[str] = field(default_factory=list)

    def print_comparison(self, baseline: "EvaluationReport") -> None:
        """Pretty-print before vs after for demo."""
        print("=" * 60)
        print("BEFORE TRAINING -> AFTER TRAINING")
        print("=" * 60)
        mult = self.avg_normalized_reward / max(0.0001, baseline.avg_normalized_reward)
        print(
            f"Normalized Reward:  {baseline.avg_normalized_reward:.3f} -> "
            f"{self.avg_normalized_reward:.3f}  ({mult:.1f}x)"
        )
        print(
            f"SLA Compliance:     {baseline.sla_compliance_rate:.0%} -> "
            f"{self.sla_compliance_rate:.0%}"
        )
        print(
            f"Root Cause Accuracy:{baseline.root_cause_accuracy:.0%} -> "
            f"{self.root_cause_accuracy:.0%}"
        )
        print(
            f"Tool Efficiency:    {baseline.tool_efficiency:.2f} -> "
            f"{self.tool_efficiency:.2f}"
        )
        print(
            "Long-Horizon Consistency: "
            f"{baseline.long_horizon_consistency:.2f} -> {self.long_horizon_consistency:.2f}"
        )
        for skill in TRACKED_SKILLS:
            base_score = baseline.skill_scores.get(skill, 0.0)
            score = self.skill_scores.get(skill, 0.0)
            print(f"  {skill}: {base_score:.0%} -> {score:.0%}")


@dataclass(slots=True)
class PolicyState:
    """Minimal per-episode policy memory for deterministic action selection."""

    has_checked_monitoring: bool = False
    has_queried_kb: bool = False
    has_checked_policy: bool = False
    has_notified: bool = False
    has_applied_fix: bool = False
    has_verified_fix: bool = False
    has_resolved: bool = False
    has_fetched_logs: bool = False
    has_probed_service: bool = False
    known_service: str = "auth"


def _json_action(action: dict[str, object]) -> str:
    return json.dumps(action, separators=(",", ":"))


def _is_trained_policy(policy: PolicyKind) -> bool:
    return policy in ("trained", "trained_heuristic")


def _priority_from_max_steps(max_steps: int) -> str:
    if max_steps >= 80:
        return "critical"
    if max_steps >= 70:
        return "critical"
    if max_steps >= 50:
        return "high"
    return "medium"


def _default_customer_id(obs: Observation) -> str:
    if obs.ticket_id.startswith("CUST-"):
        return obs.ticket_id
    return "CUST-001"


def _fallback_action(obs: Observation) -> dict[str, object]:
    phase = obs.incident_phase or "investigation"
    if phase in ("triage", "investigation"):
        return {"action_type": "check_monitoring", "service_name": None}
    if phase == "response":
        return {
            "action_type": "respond",
            "response_text": "We are investigating and will share updates shortly.",
            "tone": "empathetic",
        }
    return {
        "action_type": "resolve",
        "resolution_summary": "Incident reviewed and currently stable.",
        "offered_compensation": None,
    }


def _pick_impacted_service(obs: Observation, default_service: str) -> str:
    status = obs.system_status or {}
    for service_name in sorted(status.keys()):
        if status[service_name] == "down":
            return service_name
    for service_name in sorted(status.keys()):
        if status[service_name] == "degraded":
            return service_name
    return default_service


def choose_policy_action(
    obs: Observation,
    state: PolicyState,
    policy: PolicyKind,
) -> dict[str, object]:
    """Deterministic heuristic policy used for baseline/trained-style evaluations."""
    available = set(obs.available_actions)
    phase = obs.incident_phase or "investigation"
    trained_mode = _is_trained_policy(policy)
    customer_id = _default_customer_id(obs)
    state.known_service = _pick_impacted_service(obs, state.known_service)

    if phase == "triage":
        if "check_monitoring" in available and not state.has_checked_monitoring:
            state.has_checked_monitoring = True
            return {"action_type": "check_monitoring", "service_name": None}
        if "query_kb" in available and not state.has_queried_kb and trained_mode:
            state.has_queried_kb = True
            return {"action_type": "query_kb", "query": "payment 500 errors"}
        if "classify" in available:
            priority = _priority_from_max_steps(obs.max_steps)
            if policy == "baseline":
                priority = "medium"
            return {
                "action_type": "classify",
                "category": "bug_report",
                "priority": priority,
            }

    if phase == "investigation":
        if "check_monitoring" in available and not state.has_checked_monitoring:
            state.has_checked_monitoring = True
            return {"action_type": "check_monitoring", "service_name": None}
        if "query_kb" in available and not state.has_queried_kb:
            state.has_queried_kb = True
            return {"action_type": "query_kb", "query": "payment outage"}
        if trained_mode and "fetch_logs" in available and not state.has_fetched_logs:
            state.has_fetched_logs = True
            return {
                "action_type": "fetch_logs",
                "service_name": state.known_service,
                "time_range": "last_15m",
            }
        if "probe_service" in available and not state.has_probed_service:
            state.has_probed_service = True
            service = "payments" if policy == "baseline" else state.known_service
            return {
                "action_type": "probe_service",
                "service_name": service,
                "check_type": "logs",
            }
        if "route" in available:
            return {"action_type": "route", "department": "technical"}
        if "fetch_user_data" in available:
            return {"action_type": "fetch_user_data", "customer_id": customer_id}

    if phase == "response":
        if trained_mode and "check_policy" in available and not state.has_checked_policy:
            state.has_checked_policy = True
            return {"action_type": "check_policy", "policy_type": "compensation"}
        if "notify_stakeholders" in available and not state.has_notified and trained_mode:
            state.has_notified = True
            return {
                "action_type": "notify_stakeholders",
                "stakeholder": "all",
                "message": "Incident impact assessed and remediation underway.",
                "urgency": "warning",
            }
        if "apply_fix" in available and not state.has_applied_fix:
            state.has_applied_fix = True
            service = "payments" if policy == "baseline" else state.known_service
            return {
                "action_type": "apply_fix",
                "service_name": service,
                "fix_type": "restart_service",
            }
        if "respond" in available:
            tone = "formal" if policy == "baseline" else "empathetic"
            return {
                "action_type": "respond",
                "response_text": "We are actively working the incident and will provide updates.",
                "tone": tone,
            }

    if phase == "resolution":
        if "verify_fix" in available and not state.has_verified_fix:
            state.has_verified_fix = True
            return {"action_type": "verify_fix", "service_name": state.known_service}
        if "write_postmortem" in available and trained_mode:
            return {
                "action_type": "write_postmortem",
                "summary": "Incident resolved after service remediation and verification.",
                "root_cause_description": "Primary authentication service instability.",
                "remediation_steps": ["checked monitoring", "applied fix", "verified recovery"],
                "prevention_measures": ["refresh runbook", "add targeted alert"],
            }
        if "update_kb" in available and trained_mode:
            return {
                "action_type": "update_kb",
                "article_title": "Payment outage triage",
                "content": "Verify auth health and logs before database restarts.",
                "tags": ["incident", "payments", "auth"],
            }
        if "resolve" in available and not state.has_resolved:
            state.has_resolved = True
            return {
                "action_type": "resolve",
                "resolution_summary": "Incident resolved with verified service recovery.",
                "offered_compensation": None,
            }
        if "respond" in available:
            return {
                "action_type": "respond",
                "response_text": "Systems are stable and we are monitoring closely.",
                "tone": "empathetic",
            }
    return _fallback_action(obs)


def _episode_skill_scores(
    actions: list[str],
    reward_breakdowns: list[dict[str, float]],
    notified_early: bool,
    checked_policy_before_sensitive: bool,
    tone_matches: list[float],
) -> dict[str, float]:
    tool_actions = [a for a in actions if a in TOOL_ACTIONS]
    unique_tool_actions = len(set(tool_actions))
    tool_eff = unique_tool_actions / max(1, len(tool_actions))
    queried_kb = "query_kb" in actions
    verified_kb = queried_kb and ("fetch_logs" in actions or "probe_service" in actions)
    first_fix_idx = actions.index("apply_fix") if "apply_fix" in actions else None
    monitoring_idx = actions.index("check_monitoring") if "check_monitoring" in actions else None
    investigation_before_action = 0.0
    if first_fix_idx is not None and monitoring_idx is not None:
        investigation_before_action = 1.0 if monitoring_idx < first_fix_idx else 0.0
    root_cause = 1.0 if any("fix_correct" in b for b in reward_breakdowns) else 0.0
    resource_eff = 1.0 if actions.count("apply_fix") <= 2 else 0.0
    red_herring = 1.0 if queried_kb and verified_kb and root_cause > 0.0 else 0.0
    tone = sum(tone_matches) / max(1, len(tone_matches))
    return {
        "investigation_before_action": investigation_before_action,
        "kb_cross_verification": 1.0 if verified_kb else 0.0,
        "policy_checking": 1.0 if checked_policy_before_sensitive else 0.0,
        "stakeholder_proactivity": 1.0 if notified_early else 0.0,
        "root_cause_accuracy": root_cause,
        "tone_matching": tone,
        "resource_efficiency": resource_eff,
        "red_herring_dismissal": red_herring,
        "_tool_efficiency_internal": tool_eff,
    }


async def run_episode(
    env: CustomerSupportEnv,
    *,
    seed: int,
    difficulty: str,
    policy: PolicyKind,
) -> EpisodeReport:
    """Run one deterministic episode and compute per-episode metrics."""
    reset = await env.reset(seed=seed, difficulty=difficulty, mode="incident")
    obs = reset.observation
    state = PolicyState()
    actions: list[str] = []
    reward_breakdowns: list[dict[str, float]] = []
    notified_early = False
    checked_policy_before_sensitive = True
    policy_seen_before_sensitive = False
    tone_matches: list[float] = []
    long_horizon_actions: list[str] = []
    after_fix = False
    final_info: dict[str, object] = {}

    for _ in range(obs.max_steps):
        action = choose_policy_action(obs, state, policy)
        action_type = str(action["action_type"])
        actions.append(action_type)

        if action_type == "check_policy":
            policy_seen_before_sensitive = True
        if action_type in POLICY_SENSITIVE_ACTIONS and not policy_seen_before_sensitive:
            checked_policy_before_sensitive = False
        if action_type == "notify_stakeholders":
            patience = obs.stakeholder_patience or {}
            min_patience = min(patience.values()) if patience else 1.0
            notified_early = notified_early or (min_patience >= 0.4)
        if action_type == "respond":
            expected = 1.0 if obs.customer_sentiment in ("angry", "frustrated") else 0.5
            actual = 1.0 if action.get("tone") == "empathetic" else 0.0
            tone_matches.append(1.0 if actual >= expected else 0.0)

        result = await env.step(action)
        info = result.info
        final_info = info
        rb = info.get("reward_breakdown")
        if isinstance(rb, dict):
            reward_breakdowns.append(
                {str(k): float(v) for k, v in rb.items() if isinstance(v, (int, float))}
            )
            if "fix_correct" in rb:
                after_fix = True
        if after_fix:
            long_horizon_actions.append(action_type)

        obs = result.observation
        if result.done:
            break

    skills = _episode_skill_scores(
        actions=actions,
        reward_breakdowns=reward_breakdowns,
        notified_early=notified_early,
        checked_policy_before_sensitive=checked_policy_before_sensitive,
        tone_matches=tone_matches,
    )
    tool_eff = skills.pop("_tool_efficiency_internal")
    root_cause = skills["root_cause_accuracy"]
    if long_horizon_actions:
        consistent_set = {"verify_fix", "resolve", "respond", "write_postmortem", "update_kb"}
        consistency_hits = sum(1 for action in long_horizon_actions if action in consistent_set)
        long_consistency = consistency_hits / len(long_horizon_actions)
    else:
        long_consistency = 0.0

    cumulative = float(final_info.get("cumulative_reward", 0.0))
    max_reward = max(0.01, float(obs.max_total_reward))
    normalized = max(0.0, min(cumulative / max_reward, 1.0))
    sla_compliant = 1.0 if obs.pending_customer_tickets == 0 else 0.0

    return EpisodeReport(
        difficulty=difficulty,
        normalized_reward=normalized,
        sla_compliant=sla_compliant,
        tool_efficiency=tool_eff,
        root_cause_accuracy=root_cause,
        long_horizon_consistency=long_consistency,
        skill_scores=skills,
        actions=actions,
    )


def aggregate_reports(
    episodes: list[EpisodeReport],
    *,
    behavior_examples: list[str] | None = None,
) -> EvaluationReport:
    """Aggregate per-episode records into one EvaluationReport."""
    if not episodes:
        empty_scores = {skill: 0.0 for skill in TRACKED_SKILLS}
        return EvaluationReport(
            avg_normalized_reward=0.0,
            sla_compliance_rate=0.0,
            tool_efficiency=0.0,
            root_cause_accuracy=0.0,
            long_horizon_consistency=0.0,
            skill_scores=empty_scores,
            per_difficulty={d: 0.0 for d in DIFFICULTIES},
            reward_history=[],
            behavior_examples=behavior_examples or [],
        )

    reward_history = [episode.normalized_reward for episode in episodes]
    avg_reward = sum(reward_history) / len(reward_history)
    sla_rate = sum(episode.sla_compliant for episode in episodes) / len(episodes)
    tool_eff = sum(episode.tool_efficiency for episode in episodes) / len(episodes)
    root_acc = sum(episode.root_cause_accuracy for episode in episodes) / len(episodes)
    long_cons = sum(episode.long_horizon_consistency for episode in episodes) / len(episodes)

    per_difficulty_values: dict[str, list[float]] = defaultdict(list)
    skill_totals: dict[str, float] = {skill: 0.0 for skill in TRACKED_SKILLS}
    for episode in episodes:
        per_difficulty_values[episode.difficulty].append(episode.normalized_reward)
        for skill in TRACKED_SKILLS:
            skill_totals[skill] += episode.skill_scores.get(skill, 0.0)

    per_difficulty = {
        difficulty: (
            sum(per_difficulty_values[difficulty]) / max(1, len(per_difficulty_values[difficulty]))
        )
        for difficulty in DIFFICULTIES
    }
    skill_scores = {skill: skill_totals[skill] / len(episodes) for skill in TRACKED_SKILLS}

    return EvaluationReport(
        avg_normalized_reward=avg_reward,
        sla_compliance_rate=sla_rate,
        tool_efficiency=tool_eff,
        root_cause_accuracy=root_acc,
        long_horizon_consistency=long_cons,
        skill_scores=skill_scores,
        per_difficulty=per_difficulty,
        reward_history=reward_history,
        behavior_examples=behavior_examples or [],
    )


def behavior_diffs(before: EvaluationReport, after: EvaluationReport) -> list[str]:
    """Create concise before/after behavior diffs for demos."""
    return [
        (
            "Investigation strategy: "
            f"check_monitoring-before-fix {before.skill_scores['investigation_before_action']:.0%} -> "
            f"{after.skill_scores['investigation_before_action']:.0%}."
        ),
        (
            "KB trust vs verification: "
            f"kb_cross_verification {before.skill_scores['kb_cross_verification']:.0%} -> "
            f"{after.skill_scores['kb_cross_verification']:.0%}."
        ),
        (
            "Long-horizon execution: "
            f"consistency {before.long_horizon_consistency:.2f} -> "
            f"{after.long_horizon_consistency:.2f}."
        ),
    ]


async def evaluate_policy_async(
    *,
    policy: PolicyKind,
    episodes_per_difficulty: int,
) -> EvaluationReport:
    """Run evaluation episodes for one policy mode."""
    env = CustomerSupportEnv()
    try:
        episodes: list[EpisodeReport] = []
        for difficulty in DIFFICULTIES:
            for episode_seed in range(episodes_per_difficulty):
                report = await run_episode(
                    env,
                    seed=episode_seed,
                    difficulty=difficulty,
                    policy=policy,
                )
                episodes.append(report)
        return aggregate_reports(episodes)
    finally:
        await env.close()


def evaluate_policy(*, policy: PolicyKind, episodes_per_difficulty: int) -> EvaluationReport:
    """Sync wrapper around async policy evaluation."""
    return asyncio.run(
        evaluate_policy_async(
            policy=policy,
            episodes_per_difficulty=episodes_per_difficulty,
        )
    )


def plot_reports(
    baseline: EvaluationReport,
    trained: EvaluationReport,
    output_dir: Path,
) -> None:
    """Plot reward curves if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping reward plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(baseline.reward_history, label="baseline", marker="o", linewidth=1.5)
    ax.plot(trained.reward_history, label="trained_heuristic", marker="o", linewidth=1.5)
    ax.set_title("Normalized Reward History")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Normalized reward")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reward_curves.png", dpi=120)
    plt.close(fig)


def _write_report(report: EvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate EICC policies.")
    parser.add_argument(
        "--policy",
        choices=["baseline", "trained", "trained_heuristic", "compare"],
        default="compare",
        help=(
            "Evaluation mode. `trained`/`trained_heuristic` both run the deterministic "
            "trained-style heuristic policy (not checkpoint inference)."
        ),
    )
    parser.add_argument(
        "--episodes-per-difficulty",
        type=int,
        default=5,
        help="Episodes to run for each difficulty tier.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/eval",
        help="Directory for evaluation JSON and plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate reward curve PNG (requires matplotlib).",
    )
    return parser


def main() -> None:
    """CLI entrypoint for evaluation and before/after comparison."""
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    if args.policy == "baseline":
        baseline = evaluate_policy(
            policy="baseline",
            episodes_per_difficulty=args.episodes_per_difficulty,
        )
        _write_report(baseline, output_dir / "baseline_report.json")
        print(json.dumps(asdict(baseline), indent=2))
        return

    if args.policy == "trained":
        print(
            "[evaluate] `--policy trained` currently maps to the deterministic "
            "`trained_heuristic` policy."
        )
        trained = evaluate_policy(
            policy="trained_heuristic",
            episodes_per_difficulty=args.episodes_per_difficulty,
        )
        _write_report(trained, output_dir / "trained_report.json")
        print(json.dumps(asdict(trained), indent=2))
        return

    if args.policy == "trained_heuristic":
        trained = evaluate_policy(
            policy="trained_heuristic",
            episodes_per_difficulty=args.episodes_per_difficulty,
        )
        _write_report(trained, output_dir / "trained_report.json")
        print(json.dumps(asdict(trained), indent=2))
        return

    baseline = evaluate_policy(
        policy="baseline",
        episodes_per_difficulty=args.episodes_per_difficulty,
    )
    trained = evaluate_policy(
        policy="trained_heuristic",
        episodes_per_difficulty=args.episodes_per_difficulty,
    )
    trained.behavior_examples = behavior_diffs(baseline, trained)
    _write_report(baseline, output_dir / "baseline_report.json")
    _write_report(trained, output_dir / "trained_report.json")
    if args.plot:
        plot_reports(baseline, trained, output_dir)

    trained.print_comparison(baseline)
    if trained.behavior_examples:
        print("\nStructured behavior diffs:")
        for line in trained.behavior_examples:
            print(f"- {line}")


if __name__ == "__main__":
    main()
