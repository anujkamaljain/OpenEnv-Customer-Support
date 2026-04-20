"""Backward compatibility checks for ticket mode defaults."""

from __future__ import annotations

import pytest

from env.environment import CustomerSupportEnv


# =====================================================================
# Default mode is ticket
# =====================================================================


@pytest.mark.asyncio
async def test_reset_default_mode_is_ticket() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="easy")
    assert result.observation.mode == "ticket"
    assert result.observation.incident_id is None


@pytest.mark.asyncio
async def test_ticket_mode_core_flow_still_works() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy")
    await env.step(
        {"action_type": "classify", "category": "account_access", "priority": "medium"}
    )
    await env.step({"action_type": "route", "department": "account"})
    result = await env.step(
        {"action_type": "resolve", "resolution_summary": "Password reset completed."}
    )
    assert result.done is True
    assert result.info["normalized_score"] >= 0.0


# =====================================================================
# Extended backward compat
# =====================================================================


@pytest.mark.asyncio
async def test_ticket_mode_observation_has_no_incident_fields() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="easy")
    obs = result.observation
    assert obs.incident_phase is None
    assert obs.system_status is None
    assert obs.stakeholder_patience is None
    assert obs.active_alerts is None


@pytest.mark.asyncio
async def test_ticket_mode_info_has_normalized_score() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy")
    await env.step({"action_type": "classify", "category": "account_access", "priority": "medium"})
    await env.step({"action_type": "route", "department": "account"})
    result = await env.step({"action_type": "resolve", "resolution_summary": "Done."})
    assert "normalized_score" in result.info


@pytest.mark.asyncio
async def test_ticket_mode_all_difficulties_work() -> None:
    env = CustomerSupportEnv()
    for diff in ("easy", "medium", "hard"):
        result = await env.reset(seed=0, difficulty=diff)
        assert result.observation.mode == "ticket"
        assert result.observation.ticket_id is not None


@pytest.mark.asyncio
async def test_ticket_mode_invalid_action_returns_penalty() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy")
    result = await env.step({"action_type": "route", "department": "billing"})
    assert result.reward == pytest.approx(-0.05)


@pytest.mark.asyncio
async def test_ticket_mode_close_clears_state() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy")
    await env.close()
    assert await env.state() is None


@pytest.mark.asyncio
async def test_incident_mode_does_not_break_later_ticket_reset() -> None:
    """After incident mode, ticket reset must still work cleanly."""
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy", mode="incident")
    await env.step({"action_type": "check_monitoring"})
    result = await env.reset(seed=0, difficulty="easy", mode="ticket")
    assert result.observation.mode == "ticket"
    assert result.observation.incident_id is None
