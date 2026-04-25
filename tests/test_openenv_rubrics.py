"""Tests for the OpenEnv Rubric wrapper layer.

These tests verify two things:

1. The rubric API is properly used (composable, inspectable).
2. Wrapping reward dimensions as rubrics does NOT touch the env's
   actual reward computation - we can run the simulated env exactly as
   before, and the rubric tree only reads the produced ``info``.
"""

from __future__ import annotations

import pytest

from env.environment import CustomerSupportEnv
from graders.openenv_rubrics import (
    BlastRadiusRubric,
    IncidentRewardRubric,
    InvestigationBeforeActionRubric,
    JsonShapeRubric,
    KBCrossVerificationRubric,
    PhaseAvailabilityRubric,
    PolicyAwarenessRubric,
    ResourceBudgetRubric,
    RootCauseAccuracyRubric,
)
from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum


def test_rubrics_inherit_from_openenv_rubric() -> None:
    """Every wrapper must subclass openenv.core.rubrics.Rubric."""
    for klass in (
        JsonShapeRubric,
        PhaseAvailabilityRubric,
        InvestigationBeforeActionRubric,
        KBCrossVerificationRubric,
        PolicyAwarenessRubric,
        RootCauseAccuracyRubric,
        BlastRadiusRubric,
        ResourceBudgetRubric,
        IncidentRewardRubric,
    ):
        assert issubclass(klass, Rubric)


def test_incident_reward_rubric_uses_composable_containers() -> None:
    """Top-level rubric must expose Sequential + WeightedSum + Gate children."""
    tree = IncidentRewardRubric()
    assert isinstance(tree.gated_reward, Sequential)
    assert isinstance(tree.weighted_sum, WeightedSum)
    assert isinstance(tree.shape_gate, Gate)
    assert isinstance(tree.phase_gate, Gate)
    assert len(tree.weighted_sum.weights) == 6
    # Weights must sum to 1.0 (WeightedSum enforces this on init).
    assert abs(sum(tree.weighted_sum.weights) - 1.0) < 1e-6


def test_named_rubrics_walks_the_tree() -> None:
    """named_rubrics() must yield every dimension by dot-path."""
    tree = IncidentRewardRubric()
    names = {name for name, _ in tree.named_rubrics()}
    expected = {
        "investigation_before_action",
        "kb_cross_verification",
        "policy_awareness",
        "root_cause_accuracy",
        "blast_radius_safe",
        "resource_budget_respected",
        "weighted_sum",
        "shape_gate",
        "phase_gate",
        "gated_reward",
    }
    # Every expected leaf must appear somewhere in the path tree.
    for leaf in expected:
        assert any(name == leaf or name.endswith("." + leaf) for name in names), (
            f"missing rubric path: {leaf} (got {sorted(names)})"
        )


def test_json_shape_rubric_handles_dict_action() -> None:
    rubric = JsonShapeRubric()
    valid = {"action_type": "check_monitoring"}
    invalid = {"action_type": "definitely_not_a_real_action"}
    obs = {"available_actions": ["check_monitoring"]}
    assert rubric(valid, obs) == 1.0
    assert rubric(invalid, obs) == 0.0


def test_phase_availability_rubric() -> None:
    rubric = PhaseAvailabilityRubric()
    obs = {"available_actions": ["check_monitoring", "query_kb"]}
    assert rubric({"action_type": "check_monitoring"}, obs) == 1.0
    assert rubric({"action_type": "apply_fix"}, obs) == 0.0


def test_investigation_before_action_rubric_trajectory() -> None:
    rubric = InvestigationBeforeActionRubric()
    obs = {"available_actions": []}
    # check_monitoring before apply_fix -> investigated correctly.
    assert rubric({"action_type": "check_monitoring"}, obs) == 1.0
    assert rubric({"action_type": "apply_fix"}, obs) == 1.0
    # New episode: apply_fix without prior monitoring -> 0.
    rubric.reset()
    assert rubric({"action_type": "apply_fix"}, obs) == 0.0


def test_kb_cross_verification_rubric() -> None:
    rubric = KBCrossVerificationRubric()
    obs = {"available_actions": []}
    # No KB query yet -> tolerant 1.0.
    assert rubric({"action_type": "check_monitoring"}, obs) == 1.0
    # KB queried but never verified -> 0.0.
    assert rubric({"action_type": "query_kb"}, obs) == 0.0
    # Now verify by probing.
    assert rubric({"action_type": "probe_service"}, obs) == 1.0


def test_policy_awareness_rubric() -> None:
    rubric = PolicyAwarenessRubric()
    obs = {"available_actions": []}
    # policy-sensitive action without prior check_policy -> 0.0.
    assert rubric({"action_type": "apply_fix"}, obs) == 0.0
    rubric.reset()
    # check_policy first -> subsequent sensitive action passes.
    assert rubric({"action_type": "check_policy"}, obs) == 1.0
    assert rubric({"action_type": "apply_fix"}, obs) == 1.0


def test_root_cause_accuracy_reads_reward_breakdown() -> None:
    rubric = RootCauseAccuracyRubric()
    obs_correct = {
        "info": {"reward_breakdown": {"fix_correct": 0.4, "total": 0.4}},
        "available_actions": [],
    }
    obs_wrong = {
        "info": {"reward_breakdown": {"fix_wrong": -0.1, "total": -0.1}},
        "available_actions": [],
    }
    assert rubric({"action_type": "apply_fix"}, obs_correct) == 1.0
    assert rubric({"action_type": "apply_fix"}, obs_wrong) == 0.0


def test_blast_radius_rubric_detects_negative_signal() -> None:
    rubric = BlastRadiusRubric()
    safe = {
        "info": {"reward_breakdown": {"fix_correct": 0.3, "total": 0.3}},
        "available_actions": [],
    }
    risky = {
        "info": {"reward_breakdown": {"blast_radius_payments": -0.2, "total": -0.2}},
        "available_actions": [],
    }
    assert rubric({"action_type": "apply_fix"}, safe) == 1.0
    assert rubric({"action_type": "apply_fix"}, risky) == 0.0


def test_resource_budget_rubric_detects_exhaustion() -> None:
    rubric = ResourceBudgetRubric()
    ok = {"info": {"reward_breakdown": {"total": 0.05}}, "available_actions": []}
    exhausted = {
        "info": {"reward_breakdown": {"notify_budget": -0.05, "total": -0.05}},
        "available_actions": [],
    }
    assert rubric({"action_type": "notify_stakeholders"}, ok) == 1.0
    assert rubric({"action_type": "notify_stakeholders"}, exhausted) == 0.0


def test_incident_reward_rubric_short_circuits_on_invalid_shape() -> None:
    """Sequential gate forces an early 0.0 for malformed actions."""
    tree = IncidentRewardRubric()
    bad_action = {"action_type": "made_up_action"}
    obs = {"available_actions": ["check_monitoring"]}
    assert tree(bad_action, obs) == 0.0


def test_incident_reward_rubric_short_circuits_on_phase_violation() -> None:
    tree = IncidentRewardRubric()
    valid_but_wrong_phase = {"action_type": "apply_fix", "service_name": "auth", "fix_type": "restart_service"}
    obs = {"available_actions": ["check_monitoring"]}
    assert tree(valid_but_wrong_phase, obs) == 0.0


def test_incident_reward_rubric_yields_weighted_score_when_valid() -> None:
    tree = IncidentRewardRubric()
    # Build an observation that satisfies all per-dimension wrappers.
    valid_obs = {
        "available_actions": ["check_monitoring"],
        "info": {"reward_breakdown": {"check_monitoring": 0.02, "total": 0.02}},
    }
    score = tree({"action_type": "check_monitoring"}, valid_obs)
    # check_monitoring is not policy-sensitive, so policy_awareness=1.0;
    # KB not yet queried so kb_cross_verification=1.0; investigation_before_action=1.0;
    # but root_cause_accuracy=0.0 and blast_radius_safe=1.0, resource_budget=1.0.
    # WeightedSum = 0.20 + 0.15 + 0.15 + 0.0 + 0.10 + 0.10 = 0.70.
    assert score == pytest.approx(0.70, abs=1e-6)


@pytest.mark.asyncio
async def test_rubric_layer_does_not_alter_env_step_reward() -> None:
    """Critical safety check: instantiating the rubric tree must not
    perturb env.step output. The env reward is the source of truth.
    """
    env_a = CustomerSupportEnv()
    env_b = CustomerSupportEnv()
    try:
        baseline_reset = await env_a.reset(seed=0, difficulty="easy", mode="incident")
        rubric_reset = await env_b.reset(seed=0, difficulty="easy", mode="incident")
        assert baseline_reset.observation.incident_id == rubric_reset.observation.incident_id

        action = {"action_type": "check_monitoring"}
        baseline_step = await env_a.step(action)
        # Build rubric and call it on the second env's response - must
        # NOT mutate state of env_a.
        tree = IncidentRewardRubric()
        rubric_step = await env_b.step(action)
        observation_for_rubric = {
            "available_actions": list(rubric_step.observation.available_actions),
            "info": dict(rubric_step.info),
        }
        # Sequential auto-detects async context and may return a coroutine
        # when called inside a running event loop; await if needed.
        score = tree(action, observation_for_rubric)
        if hasattr(score, "__await__"):
            score = await score
        assert isinstance(score, float)
        # Env rewards must be equal regardless of rubric usage.
        assert baseline_step.reward == rubric_step.reward
    finally:
        await env_a.close()
        await env_b.close()
