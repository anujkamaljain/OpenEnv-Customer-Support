"""Tests for BillingSystem behavior."""

from __future__ import annotations

import pytest

from env.billing import BillingRecord, BillingSystem, Dispute, FailedPayment, Invoice, Refund


def _billing_system() -> BillingSystem:
    record = BillingRecord(
        customer_id="CUST-A",
        current_balance=250.0,
        payment_status="failed",
        pending_invoices=[Invoice(invoice_id="INV-1", amount=250.0, due_step=2)],
        failed_payments=[FailedPayment(payment_id="PAY-1", amount=250.0, reason="gateway_timeout")],
        total_lifetime_value=10000.0,
    )
    return BillingSystem({"CUST-A": record})


# =====================================================================
# Billing checks and refunds
# =====================================================================


def test_check_billing_reveals_failed_payments() -> None:
    system = _billing_system()
    response = system.check_billing("CUST-A")
    assert response.payment_status == "failed"
    assert len(response.failed_payments) == 1


def test_process_refund_respects_cap() -> None:
    system = _billing_system()
    system.set_refund_cap(100.0)
    result = system.process_refund("CUST-A", 120.0)
    assert result.approved is False
    assert "policy cap" in result.message.lower() or "cap" in result.message.lower()


def test_process_refund_approved_updates_balance() -> None:
    system = _billing_system()
    result = system.process_refund("CUST-A", 50.0)
    assert result.approved is True
    response = system.check_billing("CUST-A")
    assert response.current_balance == pytest.approx(200.0)


# =====================================================================
# Edge cases
# =====================================================================


def test_refund_negative_amount_rejected() -> None:
    system = _billing_system()
    result = system.process_refund("CUST-A", -10.0)
    assert result.approved is False


def test_refund_at_cap_boundary_approved() -> None:
    system = _billing_system()
    system.set_refund_cap(100.0)
    result = system.process_refund("CUST-A", 100.0)
    assert result.approved is True


def test_refund_just_above_cap_rejected() -> None:
    system = _billing_system()
    system.set_refund_cap(100.0)
    result = system.process_refund("CUST-A", 100.01)
    assert result.approved is False


def test_refund_clears_balance_to_current() -> None:
    system = _billing_system()
    system.set_refund_cap(300.0)
    system.process_refund("CUST-A", 250.0)
    response = system.check_billing("CUST-A")
    assert response.payment_status == "current"
    assert response.current_balance == pytest.approx(0.0)


def test_multiple_refunds_accumulate() -> None:
    system = _billing_system()
    system.process_refund("CUST-A", 50.0)
    system.process_refund("CUST-A", 50.0)
    response = system.check_billing("CUST-A")
    assert response.current_balance == pytest.approx(150.0)
    assert len(response.refund_history) == 2


def test_billing_response_contains_disputes() -> None:
    record = BillingRecord(
        customer_id="CUST-D",
        current_balance=100.0,
        payment_status="disputed",
        active_disputes=[Dispute(dispute_id="DSP-1", amount=100.0, reason="unauthorized")],
    )
    system = BillingSystem({"CUST-D": record})
    response = system.check_billing("CUST-D")
    assert len(response.active_disputes) == 1
    assert response.active_disputes[0].dispute_id == "DSP-1"


def test_default_refund_cap_is_150() -> None:
    system = _billing_system()
    result = system.process_refund("CUST-A", 149.0)
    assert result.approved is True
    result_over = system.process_refund("CUST-A", 151.0)
    assert result_over.approved is False


def test_check_billing_unknown_customer_raises() -> None:
    system = _billing_system()
    with pytest.raises(KeyError):
        system.check_billing("NONEXISTENT")
