"""Dynamic customer queue and customer behavior modeling."""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from env.crm import CRMSystem, CustomerRecord
    from models.action import RespondAction

Sentiment = Literal["angry", "frustrated", "neutral", "satisfied"]
CustomerTier = Literal["free", "pro", "enterprise"]
CustomerValue = Literal["low", "medium", "high"]


class DynamicTicket(BaseModel):
    """Dynamically generated ticket."""

    ticket_id: str
    ticket_text: str
    customer_id: str
    customer_sentiment: Sentiment
    customer_tier: CustomerTier
    customer_value: CustomerValue
    sla_deadline: int = Field(ge=0)


class CustomerReaction(BaseModel):
    """Deterministic reaction to agent response or delay."""

    frustration_delta: float
    new_message: str
    threatens_legal: bool = False
    escalation_request: bool = False


class CustomerBehaviorModel:
    """Simulates deterministic customer reactions."""

    def react_to_response(
        self, customer: CustomerRecord, response: RespondAction
    ) -> CustomerReaction:
        """React based on frustration, tone, and apology content."""
        text = response.response_text.lower()
        if customer.frustration_level > 0.8 and response.tone != "empathetic":
            return CustomerReaction(
                frustration_delta=0.15,
                new_message="This is unacceptable. Escalate now.",
                threatens_legal=customer.tier == "enterprise",
                escalation_request=True,
            )
        if customer.frustration_level > 0.5 and "apolog" in text:
            return CustomerReaction(
                frustration_delta=-0.10,
                new_message="Thanks for the acknowledgment. Awaiting resolution.",
            )
        return CustomerReaction(
            frustration_delta=0.03 if response.tone == "concise" else -0.02,
            new_message="Received update.",
        )

    def react_to_delay(self, customer: CustomerRecord, steps_waiting: int) -> CustomerReaction:
        """Increase frustration as waiting time increases."""
        if steps_waiting > 10 and customer.tier == "enterprise":
            return CustomerReaction(
                frustration_delta=0.2,
                new_message="We are considering switching providers.",
                threatens_legal=True,
                escalation_request=True,
            )
        if steps_waiting > 6:
            return CustomerReaction(
                frustration_delta=0.1,
                new_message="Need immediate status and ETA.",
                escalation_request=customer.tier != "free",
            )
        return CustomerReaction(
            frustration_delta=0.02,
            new_message="Waiting for an update.",
        )


class CustomerQueueManager:
    """Generate deterministic dynamic tickets from world health."""

    def __init__(self, crm: CRMSystem) -> None:
        self._crm = crm
        self._ticket_counter = 0
        self._last_ticket_step_by_customer: dict[str, int] = {}

    def generate_tickets(self, world: object, step: int) -> list[DynamicTicket]:
        """Generate tickets based on service failures and customer tier."""
        health_summary = world.service_mesh.get_health_summary()  # type: ignore[attr-defined]
        if not _should_generate_from_health(health_summary):
            return []
        generated: list[DynamicTicket] = []
        for customer_id in self._crm.get_affected_customers():
            if not self._should_emit_for_customer(customer_id, step):
                continue
            generated.append(self._build_ticket(customer_id, step, health_summary))
        return generated

    def update_frustration(self, step: int) -> None:
        """Increase frustration for unresolved customers."""
        for customer_id in self._crm.get_affected_customers():
            last_step = self._last_ticket_step_by_customer.get(customer_id, step)
            waiting = step - last_step
            delta = 0.02 if waiting <= 5 else 0.06
            self._crm.update_frustration(customer_id, delta)

    def _should_emit_for_customer(self, customer_id: str, step: int) -> bool:
        profile = self._crm.fetch_user_data(customer_id)
        cadence = 2 if profile.tier == "enterprise" else 4
        return step % cadence == 0

    def _build_ticket(
        self, customer_id: str, step: int, health_summary: dict[str, str]
    ) -> DynamicTicket:
        self._ticket_counter += 1
        self._last_ticket_step_by_customer[customer_id] = step
        profile = self._crm.fetch_user_data(customer_id)
        sentiment = _sentiment_from_frustration(profile.frustration_level)
        failing_services = [name for name, health in health_summary.items() if health != "healthy"]
        issue_label = failing_services[0] if failing_services else "platform"
        ticket_id = f"DYN-{step:03d}-{self._ticket_counter:03d}"
        return DynamicTicket(
            ticket_id=ticket_id,
            ticket_text=f"{issue_label} incident still impacting account {profile.name}.",
            customer_id=customer_id,
            customer_sentiment=sentiment,
            customer_tier=profile.tier,
            customer_value=profile.value,
            sla_deadline=4 if profile.tier == "enterprise" else 8,
        )


def _should_generate_from_health(health_summary: dict[str, str]) -> bool:
    return any(health in ("degraded", "flickering", "down") for health in health_summary.values())


def _sentiment_from_frustration(level: float) -> Sentiment:
    if level >= 0.75:
        return "angry"
    if level >= 0.45:
        return "frustrated"
    if level <= 0.15:
        return "satisfied"
    return "neutral"
