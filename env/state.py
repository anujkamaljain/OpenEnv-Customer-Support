"""Internal mutable state for a single episode."""

from __future__ import annotations

from typing import Any

from models.observation import ActionRecord, Observation, Phase
from models.ticket import TicketData

PHASE_VALID_ACTIONS: dict[Phase, frozenset[str]] = {
    "unclassified": frozenset(["classify"]),
    "classified": frozenset(["route", "escalate"]),
    "routed": frozenset(["respond", "escalate", "resolve", "request_info"]),
    "responding": frozenset(["respond", "escalate", "resolve", "request_info"]),
    "escalated": frozenset(["resolve"]),
    "resolved": frozenset(),
}

MAX_STEPS: dict[str, int] = {
    "easy": 8,
    "medium": 9,
    "hard": 10,
}


def compute_max_total_reward(ticket: TicketData) -> float:
    """Theoretical maximum reward for an optimal agent on this ticket."""
    total = 0.10  # classify base
    if ticket.gold_priority in ("critical", "high"):
        total += 0.10  # urgency bonus
    total += 0.10  # route
    if ticket.difficulty in ("medium", "hard"):
        total += 0.20  # respond
    if ticket.partial_info:
        total += 0.05  # request_info bonus
    if ticket.requires_escalation:
        total += 0.15  # escalate
    total += 0.25  # resolve
    return total


class InternalState:
    """Tracks episode progress, phase transitions, and cumulative quality scores."""

    __slots__ = (
        "ticket",
        "phase",
        "steps_taken",
        "max_steps",
        "max_total_reward",
        "actions_log",
        "cumulative_reward",
        "classification_correct",
        "routing_correct",
        "urgency_handled",
        "response_quality_score",
        "resolution_quality_score",
        "escalation_score",
        "constraints_violated",
        "done",
        "last_action_json",
        # v2 additions
        "sla_steps",
        "urgency_penalty_accrued",
        "info_requested",
    )

    def __init__(self, ticket: TicketData) -> None:
        self.ticket = ticket
        self.phase: Phase = "unclassified"
        self.steps_taken: int = 0
        self.max_steps: int = MAX_STEPS[ticket.difficulty]
        self.max_total_reward: float = compute_max_total_reward(ticket)
        self.actions_log: list[ActionRecord] = []
        self.cumulative_reward: float = 0.0

        self.classification_correct: bool | None = None
        self.routing_correct: bool | None = None
        self.urgency_handled: bool = False
        self.response_quality_score: float | None = None
        self.resolution_quality_score: float | None = None
        self.escalation_score: float | None = None
        self.constraints_violated: list[str] = []
        self.done: bool = False
        self.last_action_json: str | None = None

        # v2
        self.sla_steps: int = ticket.effective_sla_steps
        self.urgency_penalty_accrued: float = 0.0
        self.info_requested: bool = False

    # ---- helpers --------------------------------------------------------

    @property
    def available_actions(self) -> list[str]:
        return sorted(PHASE_VALID_ACTIONS[self.phase])

    def record_action(
        self, action_summary: str, feedback: str, reward: float
    ) -> None:
        self.actions_log.append(
            ActionRecord(
                step=self.steps_taken,
                action_taken=action_summary,
                env_feedback=feedback,
                reward_earned=round(reward, 4),
            )
        )
        self.cumulative_reward += reward
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.done = True

    def to_observation(self) -> Observation:
        return Observation(
            ticket_id=self.ticket.ticket_id,
            ticket_text=self.ticket.ticket_text,
            customer_sentiment=self.ticket.customer_sentiment,
            customer_tier=self.ticket.customer_tier,
            category_hint=self.ticket.category_hint,
            history=list(self.actions_log),
            pending_tickets=0,
            current_step=self.steps_taken,
            max_steps=self.max_steps,
            constraints=list(self.ticket.constraints),
            available_actions=self.available_actions,
            phase=self.phase,
            sla_steps_remaining=max(0, self.sla_steps - self.steps_taken),
            customer_value=self.ticket.customer_value,
        )

    def to_info(self) -> dict[str, Any]:
        mtr = self.max_total_reward
        return {
            "phase": self.phase,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "max_total_reward": mtr,
            "normalized_score": round(
                min(max(self.cumulative_reward / mtr, 0.0), 1.0), 4
            ),
            "classification_correct": self.classification_correct,
            "routing_correct": self.routing_correct,
            "urgency_handled": self.urgency_handled,
            "response_quality_score": self.response_quality_score,
            "resolution_quality_score": self.resolution_quality_score,
            "escalation_score": self.escalation_score,
            "constraints_violated": list(self.constraints_violated),
            "difficulty": self.ticket.difficulty,
            "sla_steps": self.sla_steps,
            "sla_overage": max(0, self.steps_taken - self.sla_steps),
            "urgency_penalty_accrued": round(self.urgency_penalty_accrued, 4),
            "customer_value": self.ticket.customer_value,
        }
