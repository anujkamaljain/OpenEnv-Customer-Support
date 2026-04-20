"""Tests for dynamic customer queue and behavior model."""

from __future__ import annotations

from types import SimpleNamespace

from env.crm import CRMSystem
from env.customers import CustomerBehaviorModel, CustomerQueueManager
from env.services import ServiceMesh
from models.action import RespondAction
from models.incident import CustomerProfile


def _crm() -> CRMSystem:
    customers = [
        CustomerProfile(customer_id="CUST-ENT-1", tier="enterprise", account_name="Atlas"),
        CustomerProfile(customer_id="CUST-PRO-1", tier="pro", account_name="Beacon"),
        CustomerProfile(customer_id="CUST-FREE-1", tier="free", account_name="Coral"),
    ]
    return CRMSystem(customers)


def _world_with_failure() -> object:
    mesh = ServiceMesh(seed=4)
    mesh.inject_failure("database", "oom")
    return SimpleNamespace(service_mesh=mesh)


def _world_healthy() -> object:
    mesh = ServiceMesh(seed=4)
    return SimpleNamespace(service_mesh=mesh)


# =====================================================================
# Dynamic tickets and reactions
# =====================================================================


def test_generate_tickets_when_services_degraded() -> None:
    crm = _crm()
    manager = CustomerQueueManager(crm)
    world = _world_with_failure()
    tickets = manager.generate_tickets(world=world, step=4)
    assert len(tickets) >= 1
    assert all(ticket.ticket_id.startswith("DYN-") for ticket in tickets)


def test_update_frustration_increases_levels() -> None:
    crm = _crm()
    manager = CustomerQueueManager(crm)
    before = crm.fetch_user_data("CUST-PRO-1").frustration_level
    manager.update_frustration(step=9)
    after = crm.fetch_user_data("CUST-PRO-1").frustration_level
    assert after > before


def test_behavior_model_reacts_to_tone_and_delay() -> None:
    crm = _crm()
    crm.update_frustration("CUST-ENT-1", 0.7)
    customer = crm._customers["CUST-ENT-1"]  # noqa: SLF001
    behavior = CustomerBehaviorModel()

    response = RespondAction(
        response_text="We are looking into this now.",
        tone="formal",
    )
    reaction = behavior.react_to_response(customer, response)
    assert reaction.frustration_delta > 0

    delay_reaction = behavior.react_to_delay(customer, steps_waiting=12)
    assert delay_reaction.threatens_legal is True


# =====================================================================
# Extended customer tests
# =====================================================================


def test_no_tickets_when_services_healthy() -> None:
    crm = _crm()
    manager = CustomerQueueManager(crm)
    world = _world_healthy()
    tickets = manager.generate_tickets(world=world, step=4)
    assert tickets == []


def test_enterprise_generates_more_frequently() -> None:
    crm = _crm()
    manager = CustomerQueueManager(crm)
    world = _world_with_failure()
    ent_tickets = []
    pro_tickets = []
    for step in range(1, 9):
        for ticket in manager.generate_tickets(world=world, step=step):
            if ticket.customer_id == "CUST-ENT-1":
                ent_tickets.append(ticket)
            elif ticket.customer_id == "CUST-PRO-1":
                pro_tickets.append(ticket)
    assert len(ent_tickets) >= len(pro_tickets)


def test_ticket_sentiment_reflects_frustration() -> None:
    crm = _crm()
    crm.update_frustration("CUST-ENT-1", 0.7)
    manager = CustomerQueueManager(crm)
    world = _world_with_failure()
    tickets = manager.generate_tickets(world=world, step=2)
    ent_tickets = [t for t in tickets if t.customer_id == "CUST-ENT-1"]
    if ent_tickets:
        assert ent_tickets[0].customer_sentiment in ("angry", "frustrated")


def test_ticket_sla_varies_by_tier() -> None:
    crm = _crm()
    manager = CustomerQueueManager(crm)
    world = _world_with_failure()
    tickets = manager.generate_tickets(world=world, step=4)
    for ticket in tickets:
        if ticket.customer_tier == "enterprise":
            assert ticket.sla_deadline == 4
        else:
            assert ticket.sla_deadline == 8


def test_behavior_model_empathetic_reduces_frustration() -> None:
    crm = _crm()
    crm.update_frustration("CUST-PRO-1", 0.5)
    customer = crm._customers["CUST-PRO-1"]  # noqa: SLF001
    behavior = CustomerBehaviorModel()

    response = RespondAction(
        response_text="We sincerely apologize for the disruption.",
        tone="empathetic",
    )
    reaction = behavior.react_to_response(customer, response)
    assert reaction.frustration_delta < 0


def test_behavior_model_short_delay_low_frustration() -> None:
    crm = _crm()
    customer = crm._customers["CUST-FREE-1"]  # noqa: SLF001
    behavior = CustomerBehaviorModel()
    reaction = behavior.react_to_delay(customer, steps_waiting=3)
    assert reaction.frustration_delta == 0.02
    assert reaction.threatens_legal is False


def test_behavior_model_medium_delay_escalation_request() -> None:
    crm = _crm()
    customer = crm._customers["CUST-PRO-1"]  # noqa: SLF001
    behavior = CustomerBehaviorModel()
    reaction = behavior.react_to_delay(customer, steps_waiting=8)
    assert reaction.escalation_request is True
    assert reaction.frustration_delta >= 0.1
