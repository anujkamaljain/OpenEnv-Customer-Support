"""Additional cascading-failure tests for service dependency propagation."""

from __future__ import annotations

from env.services import ServiceMesh


def test_database_failure_reaches_notifications_transitively() -> None:
    """Database outage should propagate through payments to notifications."""
    mesh = ServiceMesh(seed=101)
    mesh.inject_failure("database", "oom")

    summary = mesh.get_health_summary()
    assert summary["database"] == "down"
    assert summary["payments"] == "degraded"
    assert summary["notifications"] == "degraded"


def test_root_fix_heals_transitive_chain() -> None:
    """Fixing the root service should recover all cascade-only dependents."""
    mesh = ServiceMesh(seed=202)
    mesh.inject_failure("database", "oom")

    result = mesh.apply_fix("database", "memory_increase")
    assert result.success is True
    assert set(result.healed_services) >= {"database", "payments", "analytics", "notifications"}
    assert mesh.get_health_summary()["payments"] == "healthy"
    assert mesh.get_health_summary()["notifications"] == "healthy"


def test_wrong_fix_on_root_damages_dependents_when_cascade_enabled() -> None:
    """Blast radius should degrade dependents for cascade-enabled wrong fixes."""
    mesh = ServiceMesh(seed=303)
    mesh.inject_failure("database", "oom")

    blast = mesh.apply_wrong_fix("database", "config_change")
    assert blast.cascade is True
    assert "database" in blast.damaged_services
    assert "payments" in blast.damaged_services
    assert mesh.get_health_summary()["database"] == "down"
    assert mesh.get_health_summary()["payments"] == "degraded"


def test_symptom_fix_does_not_clear_root_cause_flag() -> None:
    """Fixing a downstream symptom must not clear the actual root-cause marker."""
    mesh = ServiceMesh(seed=404)
    mesh.inject_failure("database", "oom")

    result = mesh.apply_fix("payments", "restart_service")
    assert result.success is False
    assert mesh.services["database"].is_root_cause is True


def test_deterministic_cascade_state_for_same_seed() -> None:
    """Same seed and same failure should produce identical cascade snapshots."""
    first = ServiceMesh(seed=505)
    second = ServiceMesh(seed=505)
    first.inject_failure("database", "oom")
    second.inject_failure("database", "oom")

    assert first.get_health_summary() == second.get_health_summary()


def test_auth_failure_cascades_to_payments_and_notifications() -> None:
    """Auth outage should degrade payments which degrades notifications."""
    mesh = ServiceMesh(seed=606)
    mesh.inject_failure("auth", "token_expiry")
    summary = mesh.get_health_summary()
    assert summary["auth"] == "down"
    assert summary["payments"] == "degraded"
    assert summary["notifications"] == "degraded"


def test_analytics_failure_does_not_cascade_upstream() -> None:
    """Analytics has no downstream dependents."""
    mesh = ServiceMesh(seed=707)
    mesh.inject_failure("analytics", "batch_job_runaway")
    summary = mesh.get_health_summary()
    assert summary["analytics"] == "down"
    assert summary["database"] == "healthy"
    assert summary["payments"] == "healthy"


def test_notifications_failure_is_isolated() -> None:
    """Notifications is a leaf node with no dependents."""
    mesh = ServiceMesh(seed=808)
    mesh.inject_failure("notifications", "queue_overflow")
    summary = mesh.get_health_summary()
    assert summary["notifications"] == "down"
    assert summary["payments"] == "healthy"
    assert summary["auth"] == "healthy"


def test_fix_wrong_service_does_not_clear_root() -> None:
    """Fixing a non-root service should not clear the root-cause outage."""
    mesh = ServiceMesh(seed=909)
    mesh.inject_failure("database", "oom")
    mesh.apply_fix("analytics", "restart_service")
    assert mesh.services["database"].is_root_cause is True
    assert mesh.services["database"].health == "down"


def test_monitoring_snapshot_after_failure() -> None:
    """Monitoring data should reflect injected failures."""
    mesh = ServiceMesh(seed=1010)
    mesh.inject_failure("database", "oom")
    snapshot = mesh.get_monitoring_data()
    db_record = next(r for r in snapshot.services if r.service == "database")
    assert db_record.health == "down"
    assert db_record.error_rate > 0.5


def test_blast_radius_non_cascade_fix() -> None:
    """Restart has cascade=False, so only the targeted service is damaged."""
    mesh = ServiceMesh(seed=1111)
    mesh.inject_failure("auth", "rate_limiting")
    result = mesh.apply_wrong_fix("database", "restart_service")
    assert result.cascade is False
    assert result.damaged_services == ["database"]


def test_double_failure_both_roots_marked() -> None:
    """Two independent failures should both be root causes."""
    mesh = ServiceMesh(seed=1212)
    mesh.inject_failure("auth", "config_corruption")
    mesh.inject_failure("analytics", "query_timeout")
    assert mesh.services["auth"].is_root_cause is True
    assert mesh.services["analytics"].is_root_cause is True


def test_heal_does_not_affect_independent_root() -> None:
    """Fixing one root cause should not heal another independent outage."""
    mesh = ServiceMesh(seed=1313)
    mesh.inject_failure("auth", "config_corruption")
    mesh.inject_failure("analytics", "query_timeout")
    mesh.apply_fix("auth", "config_change")
    assert mesh.services["auth"].health == "healthy"
    assert mesh.services["analytics"].health == "down"


def test_get_health_summary_returns_all_services() -> None:
    mesh = ServiceMesh(seed=1414)
    summary = mesh.get_health_summary()
    assert set(summary.keys()) == {"auth", "database", "payments", "analytics", "notifications"}
