"""Tests for deterministic service mesh behavior."""

from __future__ import annotations

import pytest

from env.services import (
    Alert,
    BLAST_RADIUS,
    SERVICE_OBSERVABILITY,
    ServiceMesh,
)


# =====================================================================
# Core mesh behavior
# =====================================================================


def test_inject_failure_cascades_to_dependents() -> None:
    """Root outage should degrade transitive dependents."""
    mesh = ServiceMesh(seed=7)
    mesh.inject_failure("database", "oom")

    summary = mesh.get_health_summary()
    assert summary["database"] == "down"
    assert summary["payments"] == "degraded"
    assert summary["analytics"] == "degraded"
    assert summary["notifications"] == "degraded"
    assert summary["auth"] == "healthy"


def test_apply_fix_root_cause_heals_chain() -> None:
    """Correct root-cause fix should heal root and dependents."""
    mesh = ServiceMesh(seed=11)
    mesh.inject_failure("auth", "config_corruption")

    result = mesh.apply_fix("auth", "config_change")
    assert result.success is True
    assert "auth" in result.healed_services
    assert mesh.get_health_summary()["auth"] == "healthy"
    assert mesh.get_health_summary()["payments"] == "healthy"


def test_apply_fix_symptom_is_temporary() -> None:
    """Fixing a symptom should not resolve root-cause state."""
    mesh = ServiceMesh(seed=3)
    mesh.inject_failure("database", "oom")

    result = mesh.apply_fix("payments", "restart_service")
    assert result.success is False
    assert mesh.services["database"].is_root_cause is True


# =====================================================================
# Advanced features
# =====================================================================


def test_topology_hidden_on_hard_and_nightmare() -> None:
    """Dependency graph must be hidden on hard/nightmare modes."""
    mesh = ServiceMesh(seed=1)
    assert mesh.get_dependencies("easy")
    assert mesh.get_dependencies("medium")
    assert mesh.get_dependencies("hard") == {}
    assert mesh.get_dependencies("nightmare") == {}


def test_observability_affects_probe_detail() -> None:
    """Low-observability services return minimal probe findings."""
    mesh = ServiceMesh(seed=8)
    high = mesh.probe_service("auth", "logs")
    low = mesh.probe_service("analytics", "logs")

    assert SERVICE_OBSERVABILITY["auth"] == "high"
    assert SERVICE_OBSERVABILITY["analytics"] == "low"
    assert len(high.findings) > len(low.findings)


def test_flickering_pattern_is_deterministic() -> None:
    """Flickering service should follow deterministic repeating pattern."""
    mesh = ServiceMesh(seed=9)
    mesh.set_flickering("payments", "connection_flap")
    observed = []
    for step in range(4):
        mesh.tick_service_health(step + 1)
        observed.append(mesh.services["payments"].health)
    assert observed == ["degraded", "healthy", "degraded", "healthy"]


def test_time_based_degradation_can_reach_down() -> None:
    """Degraded services worsen every 5 steps and eventually fail hard."""
    mesh = ServiceMesh(seed=2)
    mesh.inject_failure("database", "oom")
    mesh.services["payments"].error_rate = 0.89

    mesh.tick_service_health(5)
    assert mesh.services["payments"].health == "down"


def test_apply_wrong_fix_triggers_blast_radius() -> None:
    """Wrong fix should apply configured penalty and extra damage."""
    mesh = ServiceMesh(seed=10)
    mesh.inject_failure("auth", "rate_limiting")

    result = mesh.apply_wrong_fix("database", "schema_migration")
    assert result.penalty == pytest.approx(float(BLAST_RADIUS["schema_migration"]["penalty"]))
    assert result.cascade is True
    assert "database" in result.damaged_services


def test_generate_alerts_is_deterministic() -> None:
    """Alert stream should be reproducible for same seed and step."""
    mesh_a = ServiceMesh(seed=5)
    mesh_b = ServiceMesh(seed=5)
    mesh_a.inject_failure("database", "oom")
    mesh_b.inject_failure("database", "oom")

    alerts_a = mesh_a.generate_alerts(step=6)
    alerts_b = mesh_b.generate_alerts(step=6)
    assert [a.model_dump() for a in alerts_a] == [b.model_dump() for b in alerts_b]
    assert all(isinstance(alert, Alert) for alert in alerts_a)


def test_generate_red_herrings_is_deterministic() -> None:
    """Per-incident red herring generation should be deterministic."""
    mesh = ServiceMesh(seed=4)
    first = mesh.generate_red_herrings("HARD-012")
    second = mesh.generate_red_herrings("HARD-012")
    assert [h.model_dump() for h in first] == [h.model_dump() for h in second]
