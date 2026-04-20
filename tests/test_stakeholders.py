"""Tests for stakeholder patience management."""

from __future__ import annotations

import pytest

from env.stakeholders import StakeholderManager


# =====================================================================
# Patience dynamics
# =====================================================================


def test_tick_decreases_patience() -> None:
    manager = StakeholderManager()
    before = manager.get_patience_levels()
    manager.tick()
    after = manager.get_patience_levels()
    assert after["vp_engineering"] < before["vp_engineering"]
    assert after["legal"] < before["legal"]
    assert after["support_lead"] < before["support_lead"]


def test_notify_restores_patience() -> None:
    manager = StakeholderManager()
    manager.tick()
    before = manager.get_patience_levels()["vp_engineering"]
    result = manager.notify("vp_engineering", "status update: incident timeline")
    after = manager.get_patience_levels()["vp_engineering"]
    assert result.accepted is True
    assert after > before


# =====================================================================
# Extended stakeholder tests
# =====================================================================


def test_patience_starts_at_one() -> None:
    manager = StakeholderManager()
    levels = manager.get_patience_levels()
    for name in ("vp_engineering", "legal", "support_lead"):
        assert levels[name] == pytest.approx(1.0)


def test_multiple_ticks_accumulate_decay() -> None:
    manager = StakeholderManager()
    for _ in range(5):
        manager.tick()
    levels = manager.get_patience_levels()
    assert levels["vp_engineering"] == pytest.approx(1.0 - 5 * 0.04)
    assert levels["legal"] == pytest.approx(1.0 - 5 * 0.02)
    assert levels["support_lead"] == pytest.approx(1.0 - 5 * 0.03)


def test_patience_clamped_at_zero() -> None:
    manager = StakeholderManager()
    for _ in range(50):
        manager.tick()
    levels = manager.get_patience_levels()
    assert levels["vp_engineering"] == pytest.approx(0.0)
    assert levels["legal"] == pytest.approx(0.0)
    assert levels["support_lead"] == pytest.approx(0.0)


def test_notify_capped_at_one() -> None:
    manager = StakeholderManager()
    manager.notify("legal", "compliance assurance and status update")
    assert manager.get_patience_levels()["legal"] == pytest.approx(1.0)


def test_tick_returns_critical_warnings_when_low() -> None:
    manager = StakeholderManager()
    for _ in range(20):
        manager.tick()
    warnings = manager.tick()
    assert any("vp_engineering" in w for w in warnings)


def test_notify_with_matching_keyword_gives_larger_boost() -> None:
    manager = StakeholderManager()
    for _ in range(5):
        manager.tick()
    before = manager.get_patience_levels()["vp_engineering"]
    manager.notify("vp_engineering", "status update on the incident")
    after_match = manager.get_patience_levels()["vp_engineering"]
    delta_match = after_match - before

    manager2 = StakeholderManager()
    for _ in range(5):
        manager2.tick()
    before2 = manager2.get_patience_levels()["vp_engineering"]
    manager2.notify("vp_engineering", "no relevant info here")
    after_no = manager2.get_patience_levels()["vp_engineering"]
    delta_no = after_no - before2

    assert delta_match > delta_no


def test_each_stakeholder_has_different_decay_rate() -> None:
    manager = StakeholderManager()
    manager.tick()
    levels = manager.get_patience_levels()
    assert levels["vp_engineering"] < levels["legal"]
    assert levels["support_lead"] < levels["legal"]
