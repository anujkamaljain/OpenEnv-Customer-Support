"""Deterministic CRM simulation for incident scenarios."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.incident import CustomerProfile

CustomerTier = Literal["free", "pro", "enterprise"]
CustomerValue = Literal["low", "medium", "high"]
AccountStatus = Literal["active", "suspended", "at_risk", "churning"]


class CustomerRecord(BaseModel):
    """Internal CRM customer record."""

    customer_id: str
    name: str
    tier: CustomerTier
    value: CustomerValue
    account_status: AccountStatus
    account_locked: bool = False
    previous_cases: int = 0
    frustration_level: float = Field(default=0.0, ge=0.0, le=1.0)
    last_contact: str | None = None
    flags: list[str] = Field(default_factory=list)
    internal_risk_score: float = Field(default=0.5, ge=0.0, le=1.0)


class CRMResponse(BaseModel):
    """Agent-visible CRM response."""

    model_config = {"frozen": True}

    customer_id: str
    name: str
    tier: CustomerTier
    value: CustomerValue
    account_status: AccountStatus
    flags: list[str] = Field(default_factory=list)
    case_history_count: int
    frustration_level: float
    account_locked: bool


class CRMSystem:
    """Simulated Customer Relationship Management system."""

    def __init__(self, customers: list[CustomerProfile]) -> None:
        self._customers: dict[str, CustomerRecord] = {}
        self._affected_customer_ids: list[str] = []
        self._load(customers)

    def _load(self, customers: list[CustomerProfile]) -> None:
        for profile in customers:
            record = CustomerRecord(
                customer_id=profile.customer_id,
                name=profile.account_name,
                tier=profile.tier,
                value=_value_for_tier(profile.tier),
                account_status="active",
                account_locked=False,
                previous_cases=_case_count_from_id(profile.customer_id),
                frustration_level=0.2 if profile.tier == "enterprise" else 0.1,
                flags=_flags_for_tier(profile.tier),
                internal_risk_score=_risk_from_id(profile.customer_id),
            )
            self._customers[record.customer_id] = record
            self._affected_customer_ids.append(record.customer_id)

    def fetch_user_data(self, customer_id: str) -> CRMResponse:
        """Return agent-visible customer data."""
        record = self._customers[customer_id]
        return CRMResponse(
            customer_id=record.customer_id,
            name=record.name,
            tier=record.tier,
            value=record.value,
            account_status=record.account_status,
            flags=list(record.flags),
            case_history_count=record.previous_cases,
            frustration_level=round(record.frustration_level, 3),
            account_locked=record.account_locked,
        )

    def get_affected_customers(self) -> list[str]:
        """List customer IDs affected by the incident."""
        return list(self._affected_customer_ids)

    def update_frustration(self, customer_id: str, delta: float) -> None:
        """Increase or decrease frustration level deterministically."""
        record = self._customers[customer_id]
        updated = max(0.0, min(1.0, record.frustration_level + delta))
        record.frustration_level = round(updated, 3)
        if updated >= 0.85:
            record.account_status = "churning"
            if "legal_escalation" not in record.flags and record.tier == "enterprise":
                record.flags.append("legal_escalation")
        elif updated >= 0.60:
            record.account_status = "at_risk"
        else:
            record.account_status = "active"


def _value_for_tier(tier: CustomerTier) -> CustomerValue:
    mapping: dict[CustomerTier, CustomerValue] = {
        "free": "low",
        "pro": "medium",
        "enterprise": "high",
    }
    return mapping[tier]


def _flags_for_tier(tier: CustomerTier) -> list[str]:
    if tier == "enterprise":
        return ["high_value", "priority_support"]
    if tier == "pro":
        return ["loyalty_program"]
    return []


def _risk_from_id(customer_id: str) -> float:
    total = sum(ord(ch) for ch in customer_id)
    return round((total % 100) / 100, 2)


def _case_count_from_id(customer_id: str) -> int:
    return sum(ord(ch) for ch in customer_id) % 12
