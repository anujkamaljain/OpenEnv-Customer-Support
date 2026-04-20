"""Tests for CRM system behavior."""

from __future__ import annotations

import pytest

from env.crm import CRMSystem, _flags_for_tier, _risk_from_id, _value_for_tier
from models.incident import CustomerProfile


def _customers() -> list[CustomerProfile]:
    return [
        CustomerProfile(customer_id="CUST-A", tier="enterprise", account_name="Atlas Bank"),
        CustomerProfile(customer_id="CUST-B", tier="pro", account_name="Blue Finch"),
        CustomerProfile(customer_id="CUST-C", tier="free", account_name="Coral Co"),
    ]


# =====================================================================
# CRM basics
# =====================================================================


def test_fetch_user_data_returns_visible_fields() -> None:
    crm = CRMSystem(_customers())
    data = crm.fetch_user_data("CUST-A")
    assert data.customer_id == "CUST-A"
    assert data.tier == "enterprise"
    assert "high_value" in data.flags


def test_get_affected_customers_is_deterministic() -> None:
    crm = CRMSystem(_customers())
    assert crm.get_affected_customers() == ["CUST-A", "CUST-B", "CUST-C"]


def test_update_frustration_changes_status() -> None:
    crm = CRMSystem(_customers())
    crm.update_frustration("CUST-A", 0.7)
    data = crm.fetch_user_data("CUST-A")
    assert data.account_status == "churning"
    assert data.frustration_level >= 0.85


# =====================================================================
# Frustration thresholds
# =====================================================================


def test_frustration_at_risk_threshold() -> None:
    """Frustration >= 0.60 should move status to 'at_risk'."""
    crm = CRMSystem(_customers())
    crm.update_frustration("CUST-B", 0.55)
    data = crm.fetch_user_data("CUST-B")
    assert data.account_status == "at_risk"


def test_frustration_stays_active_when_low() -> None:
    crm = CRMSystem(_customers())
    crm.update_frustration("CUST-C", 0.1)
    data = crm.fetch_user_data("CUST-C")
    assert data.account_status == "active"


def test_frustration_clamped_at_one() -> None:
    crm = CRMSystem(_customers())
    crm.update_frustration("CUST-C", 5.0)
    data = crm.fetch_user_data("CUST-C")
    assert data.frustration_level == pytest.approx(1.0)


def test_frustration_clamped_at_zero() -> None:
    crm = CRMSystem(_customers())
    crm.update_frustration("CUST-C", -5.0)
    data = crm.fetch_user_data("CUST-C")
    assert data.frustration_level == pytest.approx(0.0)


def test_enterprise_churning_adds_legal_flag() -> None:
    crm = CRMSystem(_customers())
    crm.update_frustration("CUST-A", 0.7)
    data = crm.fetch_user_data("CUST-A")
    assert "legal_escalation" in data.flags


def test_non_enterprise_churning_no_legal_flag() -> None:
    crm = CRMSystem(_customers())
    crm.update_frustration("CUST-B", 0.8)
    data = crm.fetch_user_data("CUST-B")
    assert "legal_escalation" not in data.flags


# =====================================================================
# Tier mapping helpers
# =====================================================================


def test_value_for_tier_enterprise() -> None:
    assert _value_for_tier("enterprise") == "high"


def test_value_for_tier_pro() -> None:
    assert _value_for_tier("pro") == "medium"


def test_value_for_tier_free() -> None:
    assert _value_for_tier("free") == "low"


def test_flags_for_enterprise() -> None:
    assert "high_value" in _flags_for_tier("enterprise")
    assert "priority_support" in _flags_for_tier("enterprise")


def test_flags_for_free_empty() -> None:
    assert _flags_for_tier("free") == []


def test_risk_from_id_deterministic() -> None:
    r1 = _risk_from_id("CUST-A")
    r2 = _risk_from_id("CUST-A")
    assert r1 == r2
    assert 0.0 <= r1 <= 1.0
