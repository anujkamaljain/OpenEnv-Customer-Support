"""Runbook models and deterministic runbook selection engine."""

from __future__ import annotations

import json
from pathlib import Path
from pydantic import BaseModel, Field

JSONScalar = str | int | float | bool


class RunbookStep(BaseModel):
    """Single runbook step."""

    step_index: int = Field(ge=0)
    action_type: str
    action_params: dict[str, JSONScalar] = Field(default_factory=dict)
    expected_outcome: str
    description: str


class Runbook(BaseModel):
    """Pre-defined incident response procedure."""

    runbook_id: str
    title: str
    incident_type: str
    steps: list[RunbookStep] = Field(default_factory=list)
    is_correct_for_incident: bool
    outdated_since: str | None = None


class RunbookSuggestion(BaseModel):
    """Agent-visible runbook suggestion."""

    model_config = {"frozen": True}

    runbook_id: str
    title: str
    steps: list[RunbookStep] = Field(default_factory=list)


class RunbookStepResult(BaseModel):
    """Result of following one runbook step."""

    model_config = {"frozen": True}

    runbook_id: str
    step_index: int
    action_type: str
    action_params: dict[str, JSONScalar]
    description: str


class RunbookEngine:
    """Deterministic runbook engine."""

    def __init__(self, runbooks: list[Runbook]) -> None:
        self._runbooks = list(runbooks)

    @staticmethod
    def from_json(path: Path) -> RunbookEngine:
        """Load runbooks from JSON file."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        runbooks = [Runbook.model_validate(item) for item in raw]
        return RunbookEngine(runbooks=runbooks)

    def suggest_runbook(self, incident_type: str) -> RunbookSuggestion | None:
        """Return first matching runbook for incident type."""
        runbook = self._find_runbook(incident_type)
        if runbook is None:
            return None
        return RunbookSuggestion(
            runbook_id=runbook.runbook_id,
            title=runbook.title,
            steps=list(runbook.steps),
        )

    def follow_runbook_step(self, runbook_id: str, step_index: int) -> RunbookStepResult:
        """Return structured data for one runbook step."""
        runbook = self._get_by_id(runbook_id)
        step = next(step for step in runbook.steps if step.step_index == step_index)
        return RunbookStepResult(
            runbook_id=runbook_id,
            step_index=step.step_index,
            action_type=step.action_type,
            action_params=dict(step.action_params),
            description=step.description,
        )

    def is_correct_for_incident(self, runbook_id: str) -> bool:
        """Return whether runbook is correct for current incident."""
        return self._get_by_id(runbook_id).is_correct_for_incident

    def _find_runbook(self, incident_type: str) -> Runbook | None:
        for runbook in self._runbooks:
            if runbook.incident_type == incident_type:
                return runbook
        return None

    def _get_by_id(self, runbook_id: str) -> Runbook:
        for runbook in self._runbooks:
            if runbook.runbook_id == runbook_id:
                return runbook
        raise KeyError(f"Runbook '{runbook_id}' not found")
