"""Ticket data model with ground-truth labels for deterministic grading."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "billing",
    "bug_report",
    "feature_request",
    "account_access",
    "general_inquiry",
    "cancellation",
]

Priority = Literal["low", "medium", "high", "critical"]

Department = Literal["billing", "technical", "account", "general"]

Sentiment = Literal["angry", "frustrated", "neutral", "satisfied"]

CustomerTier = Literal["free", "pro", "enterprise"]

Difficulty = Literal["easy", "medium", "hard"]

EscalationTarget = Literal["l2_support", "engineering", "management"]

CustomerValue = Literal["low", "medium", "high"]

SLA_DEFAULTS: dict[str, int] = {
    "low": 8,
    "medium": 6,
    "high": 4,
    "critical": 3,
}

CUSTOMER_VALUE_PENALTY_MULTIPLIER: dict[str, float] = {
    "low": 1.0,
    "medium": 1.3,
    "high": 1.8,
}


class KeywordSpec(BaseModel):
    """Weighted keyword specification for grading text quality.

    *required*  — must appear; 60 % of quality score.
    *optional*  — bonus if present; 40 % of quality score.
    *forbidden* — penalty per match (``-0.03`` each).
    *min_required_hits* — hard floor; score halved when not met.
    """

    model_config = {"frozen": True}

    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    min_required_hits: int = 0


class TicketData(BaseModel):
    """Ground-truth ticket used internally by the environment."""

    model_config = {"frozen": True}

    ticket_id: str
    ticket_text: str
    customer_sentiment: Sentiment
    customer_tier: CustomerTier
    gold_category: Category
    gold_priority: Priority
    gold_department: Department

    response_spec: KeywordSpec = Field(default_factory=KeywordSpec)
    resolution_spec: KeywordSpec = Field(default_factory=KeywordSpec)

    requires_escalation: bool = False
    escalation_reason: str | None = None
    escalation_target: EscalationTarget | None = None
    compensation_range: tuple[float, float] | None = None
    constraints: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    category_hint: str | None = None

    # --- v2 fields ---
    customer_value: CustomerValue = "medium"
    sla_steps: int | None = None
    secondary_category: Category | None = None
    distractors: list[str] = Field(default_factory=list)
    partial_info: bool = False
    info_reveals: str | None = None

    @property
    def effective_sla_steps(self) -> int:
        if self.sla_steps is not None:
            return self.sla_steps
        return SLA_DEFAULTS[self.gold_priority]

    @property
    def penalty_multiplier(self) -> float:
        return CUSTOMER_VALUE_PENALTY_MULTIPLIER[self.customer_value]
