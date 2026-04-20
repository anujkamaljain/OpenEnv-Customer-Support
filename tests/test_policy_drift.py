"""Tests for deterministic policy drift behavior."""

from __future__ import annotations

from env.policy_engine import PolicyChange, PolicyEngine


# =====================================================================
# Policy checks and drifts
# =====================================================================


def test_check_policy_returns_current_rules() -> None:
    engine = PolicyEngine(
        initial_policies={"refund_cap": 200, "escalation_required": False},
        drift_schedule=[],
    )
    refund = engine.check_policy("refund")
    assert refund.rules["max_refund"] == 200


def test_apply_scheduled_drifts_updates_policy_once() -> None:
    engine = PolicyEngine(
        initial_policies={"refund_cap": 200},
        drift_schedule=[
            PolicyChange(
                trigger_step=5,
                policy_type="refund",
                old_value={"max_refund": 200},
                new_value={"max_refund": 100},
                reason="CFO reduction",
            )
        ],
    )
    changes = engine.apply_scheduled_drifts(5)
    assert len(changes) == 1
    assert engine.check_policy("refund").rules["max_refund"] == 100

    changes_again = engine.apply_scheduled_drifts(5)
    assert changes_again == []


# =====================================================================
# Additional drift tests
# =====================================================================


def test_drift_before_trigger_step_has_no_effect() -> None:
    engine = PolicyEngine(
        initial_policies={"refund_cap": 200},
        drift_schedule=[
            PolicyChange(trigger_step=10, policy_type="refund", new_value={"max_refund": 50}, reason="test")
        ],
    )
    changes = engine.apply_scheduled_drifts(5)
    assert changes == []
    assert engine.check_policy("refund").rules["max_refund"] == 200


def test_multiple_drifts_at_different_steps() -> None:
    engine = PolicyEngine(
        initial_policies={},
        drift_schedule=[
            PolicyChange(trigger_step=5, policy_type="refund", new_value={"max_refund": 80}, reason="cut1"),
            PolicyChange(trigger_step=10, policy_type="refund", new_value={"max_refund": 40}, reason="cut2"),
        ],
    )
    engine.apply_scheduled_drifts(5)
    assert engine.check_policy("refund").rules["max_refund"] == 80
    engine.apply_scheduled_drifts(10)
    assert engine.check_policy("refund").rules["max_refund"] == 40


def test_check_unknown_policy_returns_empty_rules() -> None:
    engine = PolicyEngine(initial_policies={}, drift_schedule=[])
    response = engine.check_policy("unknown_type")
    assert response.rules == {}
    assert response.version == 0


def test_policy_version_increments_after_drift() -> None:
    engine = PolicyEngine(
        initial_policies={},
        drift_schedule=[
            PolicyChange(trigger_step=1, policy_type="refund", new_value={"max_refund": 100}, reason="v2"),
        ],
    )
    v1 = engine.check_policy("refund").version
    engine.apply_scheduled_drifts(1)
    v2 = engine.check_policy("refund").version
    assert v2 == v1 + 1


def test_default_policies_bootstrap() -> None:
    engine = PolicyEngine(initial_policies={}, drift_schedule=[])
    assert engine.check_policy("refund").rules["max_refund"] == 150
    assert engine.check_policy("escalation").rules["required"] is True
    assert engine.check_policy("sla").rules["enterprise_steps"] == 4
    assert engine.check_policy("compensation").rules["allow_credit"] is True
    assert engine.check_policy("communication").rules["tone"] == "empathetic"


def test_effective_since_step_updates_after_drift() -> None:
    engine = PolicyEngine(
        initial_policies={},
        drift_schedule=[
            PolicyChange(trigger_step=7, policy_type="sla", new_value={"enterprise_steps": 8}, reason="ext"),
        ],
    )
    assert engine.check_policy("sla").effective_since_step == 0
    engine.apply_scheduled_drifts(7)
    assert engine.check_policy("sla").effective_since_step == 7


def test_two_drifts_same_step_both_applied() -> None:
    engine = PolicyEngine(
        initial_policies={},
        drift_schedule=[
            PolicyChange(trigger_step=3, policy_type="refund", new_value={"max_refund": 99}, reason="a"),
            PolicyChange(trigger_step=3, policy_type="sla", new_value={"enterprise_steps": 6}, reason="b"),
        ],
    )
    changes = engine.apply_scheduled_drifts(3)
    assert len(changes) == 2
    assert engine.check_policy("refund").rules["max_refund"] == 99
    assert engine.check_policy("sla").rules["enterprise_steps"] == 6
