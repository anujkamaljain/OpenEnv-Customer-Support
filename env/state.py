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
    """Achievable maximum reward for an optimal agent on this ticket.

    Accounts for action rewards *and* any unavoidable SLA penalties incurred
    when the minimum optimal path exceeds the ticket's SLA deadline.
    """
    total = 0.10  # classify base
    if ticket.gold_priority in ("critical", "high"):
        total += 0.10  # urgency bonus
    total += 0.10  # route

    min_steps = 3  # classify + route + resolve (always required)
    if ticket.difficulty in ("medium", "hard"):
        total += 0.20  # respond
        min_steps += 1
    if ticket.partial_info:
        total += 0.05  # request_info bonus
        min_steps += 1
    if ticket.requires_escalation:
        total += 0.15  # escalate
        min_steps += 1
    total += 0.25  # resolve

    sla = ticket.effective_sla_steps
    for step_idx in range(min_steps):
        if step_idx >= sla:
            total -= 0.02 * (step_idx - sla + 1)

    return round(total, 4)


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
        # v3 — interpretability
        "last_reward_breakdown",
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

        # v3 — interpretability
        self.last_reward_breakdown: dict[str, float] = {}

    # ---- helpers --------------------------------------------------------

    @property
    def available_actions(self) -> list[str]:
        actions = PHASE_VALID_ACTIONS[self.phase]
        if self.info_requested:
            actions = actions - frozenset(["request_info"])
        return sorted(actions)

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
            max_total_reward=self.max_total_reward,
        )

    def to_info(self) -> dict[str, Any]:
        mtr = self.max_total_reward
        info: dict[str, Any] = {
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
            "reward_breakdown": dict(self.last_reward_breakdown),
        }
        if self.done:
            info["final_score_breakdown"] = self._compute_final_breakdown()
        return info

    def _compute_final_breakdown(self) -> dict[str, float]:
        """Episode-level weighted breakdown mirroring ``grade_episode``."""
        cls_s = 1.0 if self.classification_correct else 0.0
        rte_s = 1.0 if self.routing_correct else 0.0
        rsp_s = self.response_quality_score if self.response_quality_score is not None else 0.0
        res_s = self.resolution_quality_score if self.resolution_quality_score is not None else 0.0
        esc_s = max(0.0, self.escalation_score) if self.escalation_score is not None else 0.0
        urg_s = 1.0 if self.urgency_handled else 0.0
        eff_s = max(0.0, 1.0 - self.steps_taken / self.max_steps) if self.max_steps > 0 else 0.0

        sla_overage = max(0, self.steps_taken - self.sla_steps)
        sla_s = max(0.0, 1.0 - sla_overage * 0.2)

        constraint_pen = len(self.constraints_violated) * 0.05

        raw = (
            0.15 * cls_s
            + 0.10 * rte_s
            + 0.20 * rsp_s
            + 0.20 * res_s
            + 0.10 * esc_s
            + 0.10 * urg_s
            + 0.05 * eff_s
            + 0.10 * sla_s
        )

        return {
            "classification": round(0.15 * cls_s, 4),
            "routing": round(0.10 * rte_s, 4),
            "response_quality": round(0.20 * rsp_s, 4),
            "resolution_quality": round(0.20 * res_s, 4),
            "escalation": round(0.10 * esc_s, 4),
            "urgency": round(0.10 * urg_s, 4),
            "efficiency": round(0.05 * eff_s, 4),
            "sla_compliance": round(0.10 * sla_s, 4),
            "constraint_penalty": round(-constraint_pen, 4),
            "total": round(max(0.0, min(raw - constraint_pen, 1.0)), 4),
        }
