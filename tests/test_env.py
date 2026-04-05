"""Tests for the CustomerSupportEnv async environment (v2).

Covers: lifecycle, classification with urgency bonus, routing, SLA penalties,
business-impact multiplier, weighted keyword scoring, partial-info tickets,
full multi-step episodes, and reward bounds.
"""

import pytest

from env.environment import CustomerSupportEnv
from env.errors import EnvironmentDoneError, EnvironmentNotResetError
from models.action import ClassifyAction


# =====================================================================
# Lifecycle
# =====================================================================


@pytest.mark.asyncio
async def test_step_before_reset_raises(env: CustomerSupportEnv) -> None:
    with pytest.raises(EnvironmentNotResetError):
        await env.step({"action_type": "classify", "category": "billing", "priority": "low"})


@pytest.mark.asyncio
async def test_state_before_reset_returns_none(env: CustomerSupportEnv) -> None:
    assert await env.state() is None


@pytest.mark.asyncio
async def test_close_clears_state(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0)
    await env.close()
    assert await env.state() is None


# =====================================================================
# Reset
# =====================================================================


@pytest.mark.asyncio
async def test_reset_returns_initial_observation(env: CustomerSupportEnv) -> None:
    res = await env.reset(seed=0, difficulty="easy")
    assert res.done is False
    assert res.reward == 0.0
    obs = res.observation
    assert obs.current_step == 0
    assert obs.phase == "unclassified"
    assert "classify" in obs.available_actions
    assert obs.ticket_id == "EASY-001"
    assert obs.customer_value == "low"
    assert obs.sla_steps_remaining > 0


@pytest.mark.asyncio
async def test_deterministic_reset(env: CustomerSupportEnv) -> None:
    r1 = await env.reset(seed=42, difficulty="easy")
    r2 = await env.reset(seed=42, difficulty="easy")
    assert r1.observation.ticket_id == r2.observation.ticket_id


@pytest.mark.asyncio
async def test_different_seeds_different_tickets(env: CustomerSupportEnv) -> None:
    r1 = await env.reset(seed=0, difficulty="easy")
    r2 = await env.reset(seed=1, difficulty="easy")
    assert r1.observation.ticket_id != r2.observation.ticket_id


# =====================================================================
# Classification + Urgency bonus
# =====================================================================


@pytest.mark.asyncio
async def test_correct_classify_no_urgency(env: CustomerSupportEnv) -> None:
    """EASY-001: gold = account_access / medium → no urgency bonus."""
    await env.reset(seed=0, difficulty="easy")
    r = await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    assert r.reward == pytest.approx(0.10)
    assert r.observation.phase == "classified"


@pytest.mark.asyncio
async def test_correct_classify_with_urgency(env: CustomerSupportEnv) -> None:
    """EASY-002: gold = billing / high → +0.10 urgency bonus."""
    await env.reset(seed=1, difficulty="easy")
    r = await env.step({"action_type": "classify", "category": "billing", "priority": "high"})
    assert r.reward == pytest.approx(0.20)
    assert r.info["urgency_handled"] is True


@pytest.mark.asyncio
async def test_classify_category_only(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    r = await env.step({"action_type": "classify", "category": "account_access", "priority": "low"})
    assert r.reward == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_classify_priority_only(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    r = await env.step({"action_type": "classify", "category": "billing", "priority": "medium"})
    assert r.reward == pytest.approx(0.04)


@pytest.mark.asyncio
async def test_classify_both_wrong(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    r = await env.step({"action_type": "classify", "category": "billing", "priority": "low"})
    assert r.reward == pytest.approx(0.01)


# =====================================================================
# Routing
# =====================================================================


@pytest.mark.asyncio
async def test_correct_routing(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    r = await env.step({"action_type": "route", "department": "account"})
    assert r.reward == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_wrong_routing(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    r = await env.step({"action_type": "route", "department": "billing"})
    assert r.reward == pytest.approx(0.01)


# =====================================================================
# SLA deadline system
# =====================================================================


@pytest.mark.asyncio
async def test_sla_no_penalty_within_deadline(env: CustomerSupportEnv) -> None:
    """EASY-001 gold_priority=medium → SLA=6 steps. First steps are free."""
    await env.reset(seed=0, difficulty="easy")
    r = await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    assert r.info["sla_overage"] == 0


@pytest.mark.asyncio
async def test_sla_penalty_after_deadline(env: CustomerSupportEnv) -> None:
    """HARD-001 gold_priority=critical → SLA=3 steps.  Step index 3 is 1 over."""
    await env.reset(seed=0, difficulty="hard")
    # Steps 0-2 within SLA
    await env.step({"action_type": "classify", "category": "bug_report", "priority": "critical"})
    await env.step({"action_type": "route", "department": "technical"})
    await env.step({
        "action_type": "respond",
        "response_text": "We sincerely apologize. Our team is investigating the data loss with top priority.",
        "tone": "empathetic",
    })
    # Step 3 — beyond SLA=3 (index 3, sla_steps=3 → overage=1 → -0.02)
    r = await env.step({
        "action_type": "escalate",
        "reason": "Enterprise data loss",
        "target_team": "engineering",
    })
    # Base escalate reward is 0.15, minus SLA -0.02 = 0.13
    assert r.reward == pytest.approx(0.13)
    assert r.info["urgency_penalty_accrued"] > 0


@pytest.mark.asyncio
async def test_sla_remaining_decreases(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    sla_0 = (await env.state()).observation.sla_steps_remaining  # type: ignore[union-attr]
    await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    sla_1 = (await env.state()).observation.sla_steps_remaining  # type: ignore[union-attr]
    assert sla_1 == sla_0 - 1


# =====================================================================
# Business impact (customer_value multiplier)
# =====================================================================


@pytest.mark.asyncio
async def test_unnecessary_escalation_low_value(env: CustomerSupportEnv) -> None:
    """EASY-001: customer_value=low (1.0x) → penalty = -0.10."""
    await env.reset(seed=0, difficulty="easy")
    await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    r = await env.step({
        "action_type": "escalate",
        "reason": "testing",
        "target_team": "engineering",
    })
    assert r.reward == pytest.approx(-0.10)


@pytest.mark.asyncio
async def test_unnecessary_escalation_high_value(env: CustomerSupportEnv) -> None:
    """HARD-009: customer_value=high (1.8x) → penalty = -0.18."""
    await env.reset(seed=8, difficulty="hard")  # HARD-009 is index 8 (10 hard total, seed=8)
    obs = (await env.state()).observation  # type: ignore[union-attr]
    assert obs.ticket_id == "HARD-009"
    await env.step({"action_type": "classify", "category": "billing", "priority": "high"})
    r = await env.step({
        "action_type": "escalate",
        "reason": "testing multiplier",
        "target_team": "engineering",
    })
    assert r.reward == pytest.approx(-0.18)


# =====================================================================
# Weighted keyword scoring (forbidden patterns)
# =====================================================================


@pytest.mark.asyncio
async def test_forbidden_keyword_penalty_in_response(env: CustomerSupportEnv) -> None:
    """MED-001: forbidden = ['not a bug', 'user error']."""
    await env.reset(seed=0, difficulty="medium")
    await env.step({"action_type": "classify", "category": "bug_report", "priority": "high"})
    await env.step({"action_type": "route", "department": "technical"})
    r = await env.step({
        "action_type": "respond",
        "response_text": "This is not a bug, it is a user error.",
        "tone": "empathetic",
    })
    # Both forbidden phrases hit → penalty
    assert r.reward < 0


# =====================================================================
# Partial-info ticket (request_info)
# =====================================================================


@pytest.mark.asyncio
async def test_request_info_needed_gives_bonus(env: CustomerSupportEnv) -> None:
    """HARD-008: partial_info=true → first request_info gives +0.05."""
    await env.reset(seed=7, difficulty="hard")
    obs = (await env.state()).observation  # type: ignore[union-attr]
    assert obs.ticket_id == "HARD-008"
    await env.step({"action_type": "classify", "category": "bug_report", "priority": "high"})
    await env.step({"action_type": "route", "department": "technical"})
    r = await env.step({
        "action_type": "request_info",
        "question_to_customer": "Can you describe the exact error and which page?",
    })
    assert r.reward >= 0.05
    assert "clarification" in r.observation.history[-1].env_feedback.lower()


@pytest.mark.asyncio
async def test_request_info_repeat_penalty(env: CustomerSupportEnv) -> None:
    """Second request_info on partial_info ticket yields penalty."""
    await env.reset(seed=7, difficulty="hard")
    await env.step({"action_type": "classify", "category": "bug_report", "priority": "high"})
    await env.step({"action_type": "route", "department": "technical"})
    r1 = await env.step({
        "action_type": "request_info",
        "question_to_customer": "Details?",
    })
    r2 = await env.step({
        "action_type": "request_info",
        "question_to_customer": "More details?",
    })
    assert r1.reward > 0
    assert r2.reward < 0


@pytest.mark.asyncio
async def test_request_info_unneeded_penalty(env: CustomerSupportEnv) -> None:
    """EASY-001: partial_info=false → request_info incurs penalty."""
    await env.reset(seed=0, difficulty="easy")
    await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    await env.step({"action_type": "route", "department": "account"})
    r = await env.step({
        "action_type": "request_info",
        "question_to_customer": "Any more info?",
    })
    assert r.reward < 0


# =====================================================================
# Full episodes
# =====================================================================


@pytest.mark.asyncio
async def test_full_easy_episode(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    r = await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    assert not r.done
    r = await env.step({"action_type": "route", "department": "account"})
    assert not r.done
    r = await env.step({
        "action_type": "resolve",
        "resolution_summary": "Password reset link sent. Access to the account is now restored.",
    })
    assert r.done is True
    assert r.info["normalized_score"] > 0


@pytest.mark.asyncio
async def test_full_medium_episode(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="medium")
    await env.step({"action_type": "classify", "category": "bug_report", "priority": "high"})
    await env.step({"action_type": "route", "department": "technical"})
    r = await env.step({
        "action_type": "respond",
        "response_text": (
            "We are investigating the crash you reported when uploading files. "
            "As a workaround, please try smaller files while we fix the issue."
        ),
        "tone": "empathetic",
    })
    assert r.reward > 0
    r = await env.step({
        "action_type": "resolve",
        "resolution_summary": "Identified and deployed a fix for the upload crash.",
    })
    assert r.done is True


@pytest.mark.asyncio
async def test_full_hard_episode_with_escalation(env: CustomerSupportEnv) -> None:
    """HARD-001: enterprise data loss, requires escalation to engineering."""
    await env.reset(seed=0, difficulty="hard")
    await env.step({"action_type": "classify", "category": "bug_report", "priority": "critical"})
    await env.step({"action_type": "route", "department": "technical"})
    await env.step({
        "action_type": "respond",
        "response_text": (
            "We sincerely apologize for this disruption. Our team is investigating "
            "the data loss with top priority and will provide an update shortly."
        ),
        "tone": "empathetic",
    })
    r = await env.step({
        "action_type": "escalate",
        "reason": "Enterprise data loss requires engineering team",
        "target_team": "engineering",
    })
    assert r.observation.phase == "escalated"
    r = await env.step({
        "action_type": "resolve",
        "resolution_summary": (
            "Data has been fully restored. Root cause identified as an update "
            "migration bug. Prevention measures and monitoring deployed."
        ),
    })
    assert r.done is True
    assert r.info["normalized_score"] > 0.5


@pytest.mark.asyncio
async def test_hard_partial_info_episode(env: CustomerSupportEnv) -> None:
    """HARD-008: partial info → request_info reveals details, then respond."""
    await env.reset(seed=7, difficulty="hard")
    await env.step({"action_type": "classify", "category": "bug_report", "priority": "high"})
    await env.step({"action_type": "route", "department": "technical"})
    info_r = await env.step({
        "action_type": "request_info",
        "question_to_customer": "Can you describe the exact error?",
    })
    assert info_r.reward > 0
    # Use revealed info to write a better response
    r = await env.step({
        "action_type": "respond",
        "response_text": (
            "Thank you for the details. We are investigating the Error 500 on the "
            "dashboard when loading monthly reports. Our team is working on a fix."
        ),
        "tone": "empathetic",
    })
    assert r.reward > 0
    r = await env.step({
        "action_type": "escalate",
        "reason": "Production bug affecting enterprise users",
        "target_team": "l2_support",
    })
    r = await env.step({
        "action_type": "resolve",
        "resolution_summary": (
            "Dashboard rendering bug for monthly reports fixed and deployed. "
            "Root cause was a cache invalidation issue from the March 28th update."
        ),
    })
    assert r.done is True
    assert r.info["normalized_score"] > 0.4


# =====================================================================
# Error handling
# =====================================================================


@pytest.mark.asyncio
async def test_invalid_action(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0)
    r = await env.step({"action_type": "nonsense"})
    assert r.reward == -0.05
    assert not r.done


@pytest.mark.asyncio
async def test_wrong_phase(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0)
    r = await env.step({"action_type": "route", "department": "billing"})
    assert r.reward == -0.05
    assert r.observation.phase == "unclassified"


@pytest.mark.asyncio
async def test_step_after_done(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    await env.step({"action_type": "route", "department": "account"})
    await env.step({"action_type": "resolve", "resolution_summary": "Done."})
    with pytest.raises(EnvironmentDoneError):
        await env.step({"action_type": "classify", "category": "billing", "priority": "low"})


# =====================================================================
# Reward bounds
# =====================================================================


@pytest.mark.asyncio
async def test_per_step_reward_bounds(env: CustomerSupportEnv) -> None:
    """Every step reward must lie in [-0.25, 0.30]."""
    await env.reset(seed=0, difficulty="easy")
    for _ in range(20):
        try:
            r = await env.step({"action_type": "classify", "category": "billing", "priority": "low"})
            assert -0.25 <= r.reward <= 0.30, f"reward {r.reward} out of bounds"
            if r.done:
                break
        except EnvironmentDoneError:
            break


@pytest.mark.asyncio
async def test_max_steps_causes_done(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    result = None
    for _ in range(20):
        try:
            result = await env.step(
                {"action_type": "classify", "category": "billing", "priority": "low"}
            )
            if result.done:
                break
        except EnvironmentDoneError:
            break
    assert result is not None
    assert result.done is True


# =====================================================================
# Pydantic action input
# =====================================================================


@pytest.mark.asyncio
async def test_pydantic_model_as_action(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    action = ClassifyAction(category="account_access", priority="medium")
    r = await env.step(action)
    assert r.reward == pytest.approx(0.10)


# =====================================================================
# Repeated action penalty
# =====================================================================


@pytest.mark.asyncio
async def test_repeated_respond_penalty(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="medium")
    await env.step({"action_type": "classify", "category": "bug_report", "priority": "high"})
    await env.step({"action_type": "route", "department": "technical"})
    action = {
        "action_type": "respond",
        "response_text": "We are investigating the crash with file upload and looking for a fix.",
        "tone": "empathetic",
    }
    r1 = await env.step(action)
    r2 = await env.step(action)
    assert r2.reward < r1.reward


# =====================================================================
# State method
# =====================================================================


@pytest.mark.asyncio
async def test_state_reflects_current(env: CustomerSupportEnv) -> None:
    await env.reset(seed=0, difficulty="easy")
    s = await env.state()
    assert s is not None
    assert s.observation.phase == "unclassified"
    await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    s = await env.state()
    assert s is not None
    assert s.observation.phase == "classified"


# =====================================================================
# Dynamic MAX_TOTAL_REWARD
# =====================================================================


@pytest.mark.asyncio
async def test_max_total_reward_varies_by_ticket(env: CustomerSupportEnv) -> None:
    """Hard tickets with escalation should have higher max_total_reward."""
    r_easy = await env.reset(seed=0, difficulty="easy")
    mtr_easy = r_easy.info["max_total_reward"]

    r_hard = await env.reset(seed=0, difficulty="hard")
    mtr_hard = r_hard.info["max_total_reward"]

    assert mtr_hard > mtr_easy
