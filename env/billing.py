"""Deterministic billing system simulation for enterprise incidents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PaymentStatus = Literal["current", "overdue", "failed", "disputed"]


class Invoice(BaseModel):
    """Customer invoice record."""

    invoice_id: str
    amount: float = Field(ge=0.0)
    due_step: int = Field(ge=0)
    status: Literal["open", "paid", "past_due"] = "open"


class Dispute(BaseModel):
    """Billing dispute record."""

    dispute_id: str
    amount: float = Field(ge=0.0)
    reason: str
    status: Literal["open", "closed"] = "open"


class FailedPayment(BaseModel):
    """Failed payment attempt details."""

    payment_id: str
    amount: float = Field(ge=0.0)
    reason: str


class Refund(BaseModel):
    """Historical refund entry."""

    refund_id: str
    amount: float = Field(ge=0.0)
    approved: bool
    reason: str


class BillingRecord(BaseModel):
    """Internal billing state for a customer."""

    customer_id: str
    current_balance: float
    payment_status: PaymentStatus
    pending_invoices: list[Invoice] = Field(default_factory=list)
    active_disputes: list[Dispute] = Field(default_factory=list)
    failed_payments: list[FailedPayment] = Field(default_factory=list)
    refund_history: list[Refund] = Field(default_factory=list)
    total_lifetime_value: float = 0.0


class BillingResponse(BaseModel):
    """Agent-visible billing response."""

    model_config = {"frozen": True}

    customer_id: str
    current_balance: float
    payment_status: PaymentStatus
    pending_invoices: list[Invoice] = Field(default_factory=list)
    active_disputes: list[Dispute] = Field(default_factory=list)
    failed_payments: list[FailedPayment] = Field(default_factory=list)
    refund_history: list[Refund] = Field(default_factory=list)


class RefundResult(BaseModel):
    """Outcome of a refund attempt."""

    model_config = {"frozen": True}

    approved: bool
    customer_id: str
    amount: float
    message: str


class BillingSystem:
    """Simulated billing and payment system."""

    def __init__(self, billing_data: dict[str, BillingRecord] | None = None) -> None:
        self._records: dict[str, BillingRecord] = billing_data or {}
        self._refund_cap: float = 150.0

    def set_refund_cap(self, amount: float) -> None:
        """Update active refund cap from policy engine."""
        self._refund_cap = amount

    def check_billing(self, customer_id: str) -> BillingResponse:
        """Return billing view for a customer."""
        record = self._records[customer_id]
        return BillingResponse(
            customer_id=record.customer_id,
            current_balance=record.current_balance,
            payment_status=record.payment_status,
            pending_invoices=list(record.pending_invoices),
            active_disputes=list(record.active_disputes),
            failed_payments=list(record.failed_payments),
            refund_history=list(record.refund_history),
        )

    def process_refund(self, customer_id: str, amount: float) -> RefundResult:
        """Attempt a refund based on active policy cap."""
        record = self._records[customer_id]
        if amount < 0:
            return RefundResult(
                approved=False,
                customer_id=customer_id,
                amount=amount,
                message="Refund amount must be non-negative.",
            )
        if amount > self._refund_cap:
            return RefundResult(
                approved=False,
                customer_id=customer_id,
                amount=amount,
                message=f"Refund exceeds policy cap ({self._refund_cap:.2f}).",
            )
        refund = Refund(
            refund_id=f"RF-{customer_id}-{len(record.refund_history) + 1}",
            amount=amount,
            approved=True,
            reason="incident_compensation",
        )
        record.refund_history.append(refund)
        record.current_balance = round(record.current_balance - amount, 2)
        if record.current_balance <= 0:
            record.payment_status = "current"
        return RefundResult(
            approved=True,
            customer_id=customer_id,
            amount=amount,
            message="Refund approved and recorded.",
        )
