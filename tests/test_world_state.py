"""Tests for world state progression and event generation."""

from __future__ import annotations

from env.world import WorldState
from tasks.incident_bank import IncidentBank


# =====================================================================
# World initialization
# =====================================================================


def test_world_initializes_from_incident() -> None:
    """WorldState should seed mesh and initial queues from scenario."""
    incident = IncidentBank().get_incident(seed=0, difficulty="easy")
    world = WorldState(seed=13, incident=incident)

    assert world.seed == 13
    assert world.incident.incident_id == incident.incident_id
    assert len(world.support_queue) == len(incident.initial_tickets)
    assert world.steps_elapsed == 0
    assert world.total_downtime_cost == 0.0


def test_world_injects_initial_root_cause() -> None:
    """World constructor should inject all configured root causes."""
    incident = IncidentBank().get_incident(seed=0, difficulty="nightmare")
    world = WorldState(seed=22, incident=incident)
    root_services = {cause.service for cause in incident.root_causes}

    for service in root_services:
        assert world.service_mesh.services[service].is_root_cause is True


# =====================================================================
# Tick progression
# =====================================================================


def test_tick_advances_time_and_emits_events() -> None:
    """Each tick increments counters and returns deterministic events."""
    incident = IncidentBank().get_incident(seed=0, difficulty="easy")
    world = WorldState(seed=2, incident=incident)

    events = world.tick()
    assert world.steps_elapsed == 1
    assert world.incident_timer == 1
    assert len(events) > 0


def test_tick_decays_stakeholder_patience() -> None:
    """Stakeholder patience should decay each step and stay >= 0."""
    incident = IncidentBank().get_incident(seed=0, difficulty="easy")
    world = WorldState(seed=3, incident=incident)
    before = dict(world.stakeholder_patience)

    world.tick()
    after = world.stakeholder_patience
    for key in before:
        assert after[key] < before[key]
        assert after[key] >= 0.0


def test_tick_applies_dynamic_ticket_schedule_once() -> None:
    """Scheduled tickets should be emitted once when trigger step is reached."""
    incident = IncidentBank().get_incident(seed=1, difficulty="easy")
    world = WorldState(seed=9, incident=incident)
    initial_count = len(world.support_queue)

    for _ in range(4):
        world.tick()
    assert len(world.support_queue) == initial_count + 1

    world.tick()
    assert len(world.support_queue) == initial_count + 1


def test_tick_applies_policy_drift() -> None:
    """Policy values should update at configured drift steps."""
    incident = IncidentBank().get_incident(seed=0, difficulty="medium")
    world = WorldState(seed=12, incident=incident)
    assert world.policy_engine["refund_cap"] == 250

    for _ in range(12):
        world.tick()
    assert world.policy_engine["refund_cap"] == 180


def test_tick_accumulates_downtime_cost() -> None:
    """Downtime cost should increase while degraded/down services exist."""
    incident = IncidentBank().get_incident(seed=0, difficulty="easy")
    world = WorldState(seed=7, incident=incident)

    world.tick()
    first = world.total_downtime_cost
    world.tick()
    second = world.total_downtime_cost
    assert first > 0.0
    assert second >= first
