"""Integration tests for incident-mode episode flow."""

from __future__ import annotations

import pytest

from env.environment import CustomerSupportEnv


# =====================================================================
# Full episode integration
# =====================================================================


@pytest.mark.asyncio
async def test_incident_episode_progression_and_resolution() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy", mode="incident")

    await env.step({"action_type": "check_monitoring"})
    state = await env.state()
    assert state is not None
    assert state.observation.incident_phase in ("triage", "investigation")

    await env.step({"action_type": "classify", "category": "bug_report", "priority": "medium"})
    await env.step({"action_type": "probe_service", "service_name": "auth", "check_type": "config"})
    await env.step({"action_type": "route", "department": "technical"})

    reject = await env.step(
        {"action_type": "apply_fix", "service_name": "database", "fix_type": "schema_migration"}
    )
    assert reject.reward <= 0
    assert reject.info["resource_budget"]["remaining_fix_attempts"] == 3

    await env.step(
        {"action_type": "escalate", "reason": "need specialist", "target_team": "engineering"}
    )
    apply = await env.step(
        {"action_type": "apply_fix", "service_name": "auth", "fix_type": "config_change"}
    )
    assert "fix_correct" in apply.info["reward_breakdown"] or "blast_radius" in apply.info["reward_breakdown"]

    verify = await env.step({"action_type": "verify_fix", "service_name": "auth"})
    assert "verify_fix" in verify.info["reward_breakdown"]
    assert verify.observation.active_alerts is not None

    await env.step(
        {
            "action_type": "write_postmortem",
            "summary": "Incident summary",
            "root_cause_description": "Auth token_expiry caused outage",
            "remediation_steps": ["verify auth", "apply config_change"],
            "prevention_measures": ["add token expiry alerts"],
        }
    )
    final = await env.step(
        {
            "action_type": "update_kb",
            "article_title": "Auth token expiry response",
            "content": "verify root cause and apply fix",
            "tags": ["auth", "token"],
        }
    )
    assert final.info["compliance_score"] >= 0.0
    assert final.observation.total_incident_cost is not None


# =====================================================================
# Additional episode scenarios
# =====================================================================


@pytest.mark.asyncio
async def test_incident_reset_returns_incident_observation() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="easy", mode="incident")
    obs = result.observation
    assert obs.mode == "incident"
    assert obs.incident_id is not None
    assert obs.incident_phase == "triage"
    assert obs.max_steps >= 30


@pytest.mark.asyncio
async def test_incident_medium_difficulty_resets() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="medium", mode="incident")
    assert result.observation.mode == "incident"
    assert result.observation.max_steps >= 40


@pytest.mark.asyncio
async def test_incident_hard_difficulty_resets() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="hard", mode="incident")
    assert result.observation.mode == "incident"
    assert result.observation.max_steps >= 50


@pytest.mark.asyncio
async def test_incident_nightmare_difficulty_resets() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="nightmare", mode="incident")
    assert result.observation.mode == "incident"
    assert result.observation.max_steps >= 70


@pytest.mark.asyncio
async def test_incident_known_facts_accumulate() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy", mode="incident")
    await env.step({"action_type": "check_monitoring"})
    state = await env.state()
    assert state is not None
    assert "system_status" in state.observation.known_facts


@pytest.mark.asyncio
async def test_incident_stakeholder_patience_visible() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="easy", mode="incident")
    assert result.observation.stakeholder_patience is not None
    assert "vp_engineering" in result.observation.stakeholder_patience


@pytest.mark.asyncio
async def test_incident_system_status_visible_after_monitoring() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy", mode="incident")
    step_result = await env.step({"action_type": "check_monitoring"})
    assert step_result.observation.system_status is not None


@pytest.mark.asyncio
async def test_incident_phase_gating_penalty() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy", mode="incident")
    result = await env.step(
        {"action_type": "apply_fix", "service_name": "auth", "fix_type": "restart_service"}
    )
    assert result.reward == pytest.approx(-0.05)


@pytest.mark.asyncio
async def test_incident_repeat_action_penalty() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="easy", mode="incident")
    await env.step({"action_type": "check_monitoring"})
    result = await env.step({"action_type": "check_monitoring"})
    assert result.info["reward_breakdown"]["repeat_penalty"] == pytest.approx(-0.05)
