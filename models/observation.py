"""Observation model exposed to the agent at each step."""

from __future__ import annotations

from typing import Literal

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
