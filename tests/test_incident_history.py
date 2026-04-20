"""Tests for incident history lookup."""

from __future__ import annotations

from pathlib import Path

from env.incident_history import HistoricalIncident, IncidentHistoryStore

_HISTORY_PATH = Path("F:/Coding/OpenEnv/tasks/history_incidents.json")


def _store() -> IncidentHistoryStore:
    return IncidentHistoryStore.from_json(_HISTORY_PATH)


# =====================================================================
# Basic queries
# =====================================================================


def test_history_query_with_service_filter() -> None:
    store = _store()
    result = store.query("database oom", service_filter="database")
    assert len(result.hits) >= 1
    assert result.hits[0].incident_id == "HIST-001"


def test_history_query_empty_when_no_match() -> None:
    store = _store()
    result = store.query("nonexistent root cause")
    assert result.hits == []


# =====================================================================
# Extended history tests
# =====================================================================


def test_query_stores_query_string() -> None:
    store = _store()
    result = store.query("payment timeout")
    assert result.query == "payment timeout"


def test_query_without_filter_searches_all() -> None:
    store = _store()
    result = store.query("database")
    all_ids = [hit.incident_id for hit in result.hits]
    assert "HIST-001" in all_ids


def test_service_filter_excludes_unrelated() -> None:
    store = _store()
    result = store.query("database", service_filter="notifications")
    for hit in result.hits:
        assert "notifications" in hit.services_affected


def test_relevance_high_for_strong_match() -> None:
    store = _store()
    result = store.query("database oom memory exhaustion", service_filter="database")
    if result.hits:
        assert result.hits[0].relevance in ("high", "medium")


def test_results_sorted_by_score_descending() -> None:
    store = _store()
    result = store.query("database")
    if len(result.hits) >= 2:
        for i in range(len(result.hits) - 1):
            assert result.hits[i].relevance != "low" or result.hits[i + 1].relevance == "low"


def test_from_json_loads_records() -> None:
    store = _store()
    result = store.query("", service_filter=None)
    assert isinstance(result.hits, list)


def test_inline_construction() -> None:
    incidents = [
        HistoricalIncident(
            incident_id="TEST-1",
            date="2026-01-01",
            title="Test Incident",
            root_cause="test failure",
            resolution="fixed test",
            services_affected=["auth"],
            is_relevant_to_current=True,
        )
    ]
    store = IncidentHistoryStore(incidents)
    result = store.query("test failure", service_filter="auth")
    assert len(result.hits) == 1
    assert result.hits[0].incident_id == "TEST-1"
