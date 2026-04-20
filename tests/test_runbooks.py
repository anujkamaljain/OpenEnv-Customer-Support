"""Tests for runbook suggestion and step retrieval."""

from __future__ import annotations

import pytest
from pathlib import Path

from env.runbooks import Runbook, RunbookEngine, RunbookStep


_RUNBOOKS_PATH = Path(__file__).resolve().parents[1] / "tasks" / "runbooks.json"


def _engine() -> RunbookEngine:
    return RunbookEngine.from_json(_RUNBOOKS_PATH)


# =====================================================================
# Core behavior
# =====================================================================


def test_suggest_runbook_returns_expected_match() -> None:
    engine = _engine()
    suggestion = engine.suggest_runbook("database_oom")
    assert suggestion is not None
    assert suggestion.runbook_id == "RB-001"
    assert len(suggestion.steps) == 3


def test_follow_runbook_step_returns_action_data() -> None:
    engine = _engine()
    result = engine.follow_runbook_step("RB-003", 1)
    assert result.action_type == "apply_fix"
    assert result.action_params["service_name"] == "auth"


# =====================================================================
# Extended runbook tests
# =====================================================================


def test_suggest_runbook_no_match_returns_none() -> None:
    engine = _engine()
    suggestion = engine.suggest_runbook("nonexistent_type")
    assert suggestion is None


def test_follow_runbook_step_index_zero() -> None:
    engine = _engine()
    result = engine.follow_runbook_step("RB-001", 0)
    assert result.step_index == 0
    assert result.runbook_id == "RB-001"


def test_follow_runbook_step_invalid_id_raises() -> None:
    engine = _engine()
    with pytest.raises(KeyError):
        engine.follow_runbook_step("RB-NONEXISTENT", 0)


def test_is_correct_for_incident_true() -> None:
    engine = _engine()
    assert engine.is_correct_for_incident("RB-001") is True


def test_is_correct_for_incident_false() -> None:
    engine = _engine()
    rb = engine._runbooks  # noqa: SLF001
    wrong = next((r for r in rb if not r.is_correct_for_incident), None)
    if wrong is not None:
        assert engine.is_correct_for_incident(wrong.runbook_id) is False


def test_inline_runbook_engine() -> None:
    runbook = Runbook(
        runbook_id="TEST-RB",
        title="Test Runbook",
        incident_type="test_type",
        steps=[
            RunbookStep(step_index=0, action_type="check_monitoring", action_params={}, expected_outcome="ok", description="check"),
        ],
        is_correct_for_incident=True,
    )
    engine = RunbookEngine([runbook])
    suggestion = engine.suggest_runbook("test_type")
    assert suggestion is not None
    assert suggestion.runbook_id == "TEST-RB"


def test_follow_step_returns_description() -> None:
    engine = _engine()
    result = engine.follow_runbook_step("RB-001", 0)
    assert isinstance(result.description, str)
    assert len(result.description) > 0
