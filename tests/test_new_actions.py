"""Tests for phase-3 action extension and incident mode dispatch."""

from __future__ import annotations

import pytest

from env.environment import CustomerSupportEnv
from models.action import (
    ACTION_CLASSES,
    ActionAdapter,
    ApplyFixAction,
    CheckBillingAction,
    CheckMonitoringAction,
    CheckPolicyAction,
    FetchLogsAction,
    FetchUserDataAction,
    FollowRunbookStepAction,
    NotifyStakeholdersAction,
    ProbeServiceAction,
    QueryIncidentHistoryAction,
    QueryKBAction,
    RollbackFixAction,
    UpdateKBAction,
    VerifyFixAction,
    WritePostmortemAction,
)


# =====================================================================
# Action schema validation
# =====================================================================


def test_new_action_validation_check_monitoring() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "check_monitoring", "service_name": "database"}
    )
    assert parsed.action_type == "check_monitoring"


def test_new_action_validation_follow_runbook_step() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "follow_runbook_step", "runbook_id": "RB-001", "step_index": 0}
    )
    assert parsed.action_type == "follow_runbook_step"


def test_new_action_validation_update_kb_defaults_tags() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "update_kb", "article_title": "A", "content": "root cause verify fix"}
    )
    assert parsed.tags == []


def test_action_classes_tuple_has_21_entries() -> None:
    assert len(ACTION_CLASSES) == 21


def test_probe_service_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "probe_service", "service_name": "auth", "check_type": "logs"}
    )
    assert isinstance(parsed, ProbeServiceAction)


def test_fetch_logs_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "fetch_logs", "service_name": "database", "time_range": "last_15m"}
    )
    assert isinstance(parsed, FetchLogsAction)


def test_apply_fix_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "apply_fix", "service_name": "auth", "fix_type": "config_change"}
    )
    assert isinstance(parsed, ApplyFixAction)


def test_verify_fix_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "verify_fix", "service_name": "auth"}
    )
    assert isinstance(parsed, VerifyFixAction)


def test_rollback_fix_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "rollback_fix", "service_name": "auth"}
    )
    assert isinstance(parsed, RollbackFixAction)


def test_notify_stakeholders_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "notify_stakeholders", "stakeholder": "vp_engineering", "message": "update", "urgency": "warning"}
    )
    assert isinstance(parsed, NotifyStakeholdersAction)


def test_write_postmortem_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "write_postmortem", "summary": "s", "root_cause_description": "r"}
    )
    assert isinstance(parsed, WritePostmortemAction)


def test_query_incident_history_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "query_incident_history", "query": "db oom"}
    )
    assert isinstance(parsed, QueryIncidentHistoryAction)


def test_check_policy_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "check_policy", "policy_type": "refund"}
    )
    assert isinstance(parsed, CheckPolicyAction)


def test_check_billing_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "check_billing", "customer_id": "CUST-A"}
    )
    assert isinstance(parsed, CheckBillingAction)


def test_fetch_user_data_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "fetch_user_data", "customer_id": "CUST-A"}
    )
    assert isinstance(parsed, FetchUserDataAction)


def test_query_kb_schema() -> None:
    parsed = ActionAdapter.validate_python(
        {"action_type": "query_kb", "query": "test"}
    )
    assert isinstance(parsed, QueryKBAction)


# =====================================================================
# Incident mode dispatch
# =====================================================================


@pytest.mark.asyncio
async def test_incident_mode_reset_and_observation_fields() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="hard", mode="incident")
    assert result.observation.mode == "incident"
    assert result.observation.incident_id is not None
    assert result.observation.incident_phase == "triage"
    assert result.observation.known_facts == {}


@pytest.mark.asyncio
async def test_incident_dispatch_tool_actions_accumulate_known_facts() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="hard", mode="incident")

    await env.step({"action_type": "check_monitoring"})
    await env.step({"action_type": "query_kb", "query": "database oom"})
    state = await env.state()
    assert state is not None
    assert state.observation.known_facts is not None
    assert "system_status" in state.observation.known_facts


@pytest.mark.asyncio
async def test_incident_dispatch_follow_runbook_step() -> None:
    env = CustomerSupportEnv()
    res = await env.reset(seed=0, difficulty="hard", mode="incident")
    runbook = res.observation.suggested_runbook
    assert runbook is not None
    step = runbook["steps"][0]["step_index"]
    result = await env.step(
        {
            "action_type": "follow_runbook_step",
            "runbook_id": runbook["runbook_id"],
            "step_index": step,
        }
    )
    assert result.info["reward_breakdown"]["follow_runbook_step"] != 0


@pytest.mark.asyncio
async def test_incident_phase_gating_blocks_wrong_action() -> None:
    env = CustomerSupportEnv()
    await env.reset(seed=0, difficulty="hard", mode="incident")
    result = await env.step({"action_type": "apply_fix", "service_name": "database", "fix_type": "memory_increase"})
    assert result.reward == -0.05


@pytest.mark.asyncio
async def test_ticket_mode_remains_default() -> None:
    env = CustomerSupportEnv()
    result = await env.reset(seed=0, difficulty="easy")
    assert result.observation.mode == "ticket"
