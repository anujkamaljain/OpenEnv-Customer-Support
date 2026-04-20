"""Incident scenario models for the EICC world simulation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard", "nightmare"]
ServiceName = Literal["auth", "database", "payments", "analytics", "notifications"]


class RootCause(BaseModel):
    """Root cause definition for an incident scenario."""

    model_config = {"frozen": True}

    service: ServiceName
    failure_mode: str
    fix_type: str


class RedHerring(BaseModel):
    """Misleading symptom that appears suspicious but is not causal."""

    model_config = {"frozen": True}

    service: ServiceName
    symptom: str
    actual_explanation: str
    misleading_because: str


class TicketTemplate(BaseModel):
    """Ticket payload used for initial and scheduled customer tickets."""

    model_config = {"frozen": True}

    ticket_id: str
    title: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]
    priority: Literal["low", "medium", "high", "critical"]


class DynamicTicketTrigger(BaseModel):
    """Condition under which a new ticket appears during an incident."""

    model_config = {"frozen": True}

    trigger_step: int = Field(ge=0)
    ticket: TicketTemplate


class PolicyChange(BaseModel):
    """Policy drift event applied at a deterministic step."""

    model_config = {"frozen": True}

    step: int = Field(ge=0)
    key: str
    value: str | int | float | bool


class KBArticleState(BaseModel):
    """Knowledge base article state for a scenario."""

    model_config = {"frozen": True}

    article_id: str
    title: str
    summary: str
    is_accurate: bool = True


class StakeholderConfig(BaseModel):
    """Per-stakeholder patience and decay configuration."""

    model_config = {"frozen": True}

    initial_patience: float = Field(default=1.0, ge=0.0, le=1.0)
    decay_per_step: float = Field(default=0.02, ge=0.0, le=1.0)


class CustomerProfile(BaseModel):
    """Customer profile impacted by the incident."""

    model_config = {"frozen": True}

    customer_id: str
    tier: Literal["free", "pro", "enterprise"]
    account_name: str


class IncidentScenario(BaseModel):
    """A full incident definition used as an environment episode."""

    model_config = {"frozen": True}

    incident_id: str
    title: str
    difficulty: Difficulty
    description: str
    root_causes: list[RootCause]
    cascade_chain: list[ServiceName]
    red_herrings: list[RedHerring] = Field(default_factory=list)
    affected_customer_profiles: list[CustomerProfile] = Field(default_factory=list)
    initial_tickets: list[TicketTemplate] = Field(default_factory=list)
    dynamic_ticket_schedule: list[DynamicTicketTrigger] = Field(default_factory=list)
    initial_policies: dict[str, str | int | float | bool] = Field(default_factory=dict)
    policy_drift_schedule: list[PolicyChange] = Field(default_factory=list)
    kb_articles: list[KBArticleState] = Field(default_factory=list)
    max_steps: int = Field(ge=1)
    sla_deadlines: dict[str, int] = Field(default_factory=dict)
    stakeholder_config: dict[str, StakeholderConfig] = Field(default_factory=dict)
    max_total_reward: float = Field(default=1.0, ge=0.0)
    optimal_tool_sequence: list[str] = Field(default_factory=list)
