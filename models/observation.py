"""Observation model exposed to the agent at each step."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Phase = Literal[
    "unclassified",
    "classified",
    "routed",
    "responding",
    "escalated",
    "resolved",
]


class ActionRecord(BaseModel):
    """Record of a single agent action and its outcome."""

    step: int
    action_taken: str
    env_feedback: str
    reward_earned: float


class Observation(BaseModel):
    """Agent-visible observation returned at every step."""

    ticket_id: str
    ticket_text: str
    customer_sentiment: Literal["angry", "frustrated", "neutral", "satisfied"]
    customer_tier: Literal["free", "pro", "enterprise"]
    category_hint: str | None = None
    history: list[ActionRecord] = Field(default_factory=list)
    pending_tickets: int = 0
    current_step: int = 0
    max_steps: int = 10
    constraints: list[str] = Field(default_factory=list)
    available_actions: list[str] = Field(default_factory=list)
    phase: Phase = "unclassified"

    # --- v2 fields ---
    sla_steps_remaining: int = 0
    customer_value: Literal["low", "medium", "high"] = "medium"
    max_total_reward: float = 1.0

    # --- v3 incident-mode extensions ---
    incident_id: str | None = None
    incident_title: str | None = None
    mode: Literal["ticket", "incident"] = "ticket"
    system_status: dict[str, str] | None = None
    active_alerts: list[str] | None = None
    tool_results: dict[str, Any] | None = None
    known_facts: dict[str, Any] | None = None
    active_policies: dict[str, Any] | None = None
    stakeholder_patience: dict[str, float] | None = None
    pending_customer_tickets: int = 0
    incident_phase: Literal["triage", "investigation", "response", "resolution"] | None = None
    suggested_runbook: dict[str, Any] | None = None
    total_incident_cost: float | None = None
