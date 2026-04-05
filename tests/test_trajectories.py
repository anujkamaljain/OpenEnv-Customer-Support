"""Best-case / worst-case trajectory simulation across all tickets.

Validates:
    1. Best-case normalized score reaches 1.0 for every ticket.
    2. Worst-case normalized score clamps to 0.0 for every ticket.
    3. Per-step rewards stay within the clamp range [-0.25, 0.30].
    4. max_total_reward matches the documented formula and is consistent
       between observation and info dict.
"""

from __future__ import annotations

import pytest

from env.environment import CustomerSupportEnv
from env.state import InternalState, compute_max_total_reward
from tasks.ticket_bank import TicketBank

_bank = TicketBank()
_ALL_TICKETS = _bank.list_tickets()

STEP_REWARD_LO = -0.25
STEP_REWARD_HI = 0.30


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _optimal_tone(ticket) -> str:
    for c in ticket.constraints:
        if "empathetic" in c.lower():
            return "empathetic"
        if "formal" in c.lower():
            return "formal"
    return "formal"


async def _run_best_case(env: CustomerSupportEnv, ticket):
    """Execute the optimal action sequence and return step rewards + final info."""
    await env.reset(seed=0)
    env._state = InternalState(ticket)

    rewards: list[float] = []

    r = await env.step({
        "action_type": "classify",
        "category": ticket.gold_category,
        "priority": ticket.gold_priority,
    })
    rewards.append(r.reward)

    r = await env.step({
        "action_type": "route",
        "department": ticket.gold_department,
    })
    rewards.append(r.reward)

    if ticket.partial_info:
        r = await env.step({
            "action_type": "request_info",
            "question_to_customer": "Can you provide more details?",
        })
        rewards.append(r.reward)

    if ticket.difficulty in ("medium", "hard"):
        kw = " ".join(ticket.response_spec.required + ticket.response_spec.optional)
        r = await env.step({
            "action_type": "respond",
            "response_text": kw or "responding to your inquiry",
            "tone": _optimal_tone(ticket),
        })
        rewards.append(r.reward)

    if ticket.requires_escalation and ticket.escalation_target:
        r = await env.step({
            "action_type": "escalate",
            "reason": "Required by policy.",
            "target_team": ticket.escalation_target,
        })
        rewards.append(r.reward)

    comp = None
    if ticket.compensation_range:
        comp = (ticket.compensation_range[0] + ticket.compensation_range[1]) / 2
    res_kw = " ".join(ticket.resolution_spec.required + ticket.resolution_spec.optional)
    r = await env.step({
        "action_type": "resolve",
        "resolution_summary": res_kw or "issue resolved",
        "offered_compensation": comp,
    })
    rewards.append(r.reward)

    return rewards, r.info, r.observation


async def _run_worst_case(env: CustomerSupportEnv, ticket):
    """Spam wrong/invalid actions until the episode terminates."""
    await env.reset(seed=0)
    env._state = InternalState(ticket)

    rewards: list[float] = []
    max_s = env._state.max_steps

    for i in range(max_s):
        r = await env.step({
            "action_type": "classify",
            "category": "general_inquiry",
            "priority": "low",
        })
        rewards.append(r.reward)
        if r.done:
            break

    return rewards, r.info, r.observation


# ------------------------------------------------------------------
# Parametrised tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket", _ALL_TICKETS, ids=lambda t: t.ticket_id)
async def test_best_case_reaches_norm_one(ticket) -> None:
    """A perfect agent must achieve normalized_score == 1.0 on every ticket."""
    env = CustomerSupportEnv(ticket_bank=_bank)
    rewards, info, obs = await _run_best_case(env, ticket)

    assert info["normalized_score"] == pytest.approx(1.0, abs=1e-3), (
        f"{ticket.ticket_id}: best norm = {info['normalized_score']:.4f} "
        f"(cum={info['cumulative_reward']:.4f}, mtr={info['max_total_reward']:.4f})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket", _ALL_TICKETS, ids=lambda t: t.ticket_id)
async def test_worst_case_clamps_to_zero(ticket) -> None:
    """An agent that does everything wrong must have normalized_score == 0."""
    env = CustomerSupportEnv(ticket_bank=_bank)
    rewards, info, obs = await _run_worst_case(env, ticket)

    assert info["normalized_score"] == pytest.approx(0.0, abs=1e-4), (
        f"{ticket.ticket_id}: worst norm = {info['normalized_score']:.4f}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket", _ALL_TICKETS, ids=lambda t: t.ticket_id)
async def test_best_case_per_step_bounds(ticket) -> None:
    """Every individual step reward must lie in [STEP_REWARD_LO, STEP_REWARD_HI]."""
    env = CustomerSupportEnv(ticket_bank=_bank)
    rewards, _, _ = await _run_best_case(env, ticket)

    for i, rw in enumerate(rewards):
        assert STEP_REWARD_LO <= rw <= STEP_REWARD_HI, (
            f"{ticket.ticket_id} step {i}: reward {rw} outside "
            f"[{STEP_REWARD_LO}, {STEP_REWARD_HI}]"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket", _ALL_TICKETS, ids=lambda t: t.ticket_id)
async def test_worst_case_per_step_bounds(ticket) -> None:
    """Even worst-case step rewards respect the clamp."""
    env = CustomerSupportEnv(ticket_bank=_bank)
    rewards, _, _ = await _run_worst_case(env, ticket)

    for i, rw in enumerate(rewards):
        assert STEP_REWARD_LO <= rw <= STEP_REWARD_HI, (
            f"{ticket.ticket_id} step {i}: reward {rw} outside "
            f"[{STEP_REWARD_LO}, {STEP_REWARD_HI}]"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket", _ALL_TICKETS, ids=lambda t: t.ticket_id)
async def test_mtr_consistent_between_obs_and_info(ticket) -> None:
    """max_total_reward must match in observation and info dict."""
    env = CustomerSupportEnv(ticket_bank=_bank)
    _, info, obs = await _run_best_case(env, ticket)

    assert obs.max_total_reward == pytest.approx(info["max_total_reward"])


@pytest.mark.asyncio
@pytest.mark.parametrize("ticket", _ALL_TICKETS, ids=lambda t: t.ticket_id)
async def test_mtr_matches_compute_function(ticket) -> None:
    """max_total_reward from InternalState matches compute_max_total_reward."""
    expected = compute_max_total_reward(ticket)
    state = InternalState(ticket)
    assert state.max_total_reward == pytest.approx(expected)


# ------------------------------------------------------------------
# Aggregate stability
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalization_range_across_all_tickets() -> None:
    """Sweep all tickets: every normalized score must remain in [0, 1]."""
    env = CustomerSupportEnv(ticket_bank=_bank)

    for ticket in _ALL_TICKETS:
        _, b_info, _ = await _run_best_case(env, ticket)
        _, w_info, _ = await _run_worst_case(env, ticket)

        assert 0.0 <= b_info["normalized_score"] <= 1.0, (
            f"{ticket.ticket_id} best norm out of [0,1]"
        )
        assert 0.0 <= w_info["normalized_score"] <= 1.0, (
            f"{ticket.ticket_id} worst norm out of [0,1]"
        )


@pytest.mark.asyncio
async def test_best_case_cumulative_equals_mtr() -> None:
    """Cumulative reward of the optimal path equals max_total_reward."""
    env = CustomerSupportEnv(ticket_bank=_bank)

    for ticket in _ALL_TICKETS:
        _, info, _ = await _run_best_case(env, ticket)
        assert info["cumulative_reward"] == pytest.approx(
            info["max_total_reward"], abs=1e-3
        ), (
            f"{ticket.ticket_id}: cum={info['cumulative_reward']:.4f} "
            f"!= mtr={info['max_total_reward']:.4f}"
        )
