"""Tests for deterministic incident models and incident bank loading."""

from __future__ import annotations

import pytest

from models.incident import IncidentScenario
from tasks.incident_bank import IncidentBank


# =====================================================================
# Bank loading
# =====================================================================


def test_incident_bank_loads_all_tiers() -> None:
    """IncidentBank loads all four difficulty tiers from JSON."""
    bank = IncidentBank()
    assert len(bank.list_incidents("easy")) == 3
    assert len(bank.list_incidents("medium")) == 5
    assert len(bank.list_incidents("hard")) == 7
    assert len(bank.list_incidents("nightmare")) == 3


def test_incident_bank_total_count() -> None:
    """IncidentBank exposes 18 total authored scenarios."""
    bank = IncidentBank()
    assert len(bank.list_incidents()) == 18


def test_incident_selection_is_deterministic() -> None:
    """Same seed and tier always return the same incident."""
    bank = IncidentBank()
    i1 = bank.get_incident(seed=42, difficulty="hard")
    i2 = bank.get_incident(seed=42, difficulty="hard")
    assert i1.incident_id == i2.incident_id


def test_incident_selection_changes_by_seed() -> None:
    """Different seeds in same tier produce different modulo selections."""
    bank = IncidentBank()
    i1 = bank.get_incident(seed=0, difficulty="easy")
    i2 = bank.get_incident(seed=1, difficulty="easy")
    assert i1.incident_id != i2.incident_id


def test_invalid_difficulty_raises() -> None:
    """Selecting a missing difficulty raises ValueError."""
    bank = IncidentBank()
    with pytest.raises(ValueError):
        bank.get_incident(seed=0, difficulty="impossible")


def test_json_scenarios_validate_with_model() -> None:
    """Returned scenarios are strongly typed IncidentScenario instances."""
    bank = IncidentBank()
    scenario = bank.get_incident(seed=0, difficulty="nightmare")
    assert isinstance(scenario, IncidentScenario)
    assert scenario.difficulty == "nightmare"
