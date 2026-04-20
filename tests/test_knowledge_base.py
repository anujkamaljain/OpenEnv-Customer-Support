"""Tests for knowledge base search and persistence."""

from __future__ import annotations

from env.knowledge_base import KBArticle, KnowledgeBase, PersistentKnowledgeBase
from tasks.incident_bank import IncidentBank


def _base_articles() -> list[KBArticle]:
    return [
        KBArticle(
            article_id="KB-1",
            title="Database OOM Recovery",
            content="Identify root cause, verify metrics, then apply memory fix.",
            solution_steps=["verify metrics", "apply memory fix"],
            tags=["database", "oom", "memory"],
            last_updated="2026-04-20",
            is_accurate=True,
        ),
        KBArticle(
            article_id="KB-2",
            title="Legacy Payment Restart",
            content="Restart payments immediately.",
            solution_steps=["restart payments"],
            tags=["payments"],
            last_updated="2024-01-01",
            is_accurate=False,
            outdated_reason="Auth token failures became dominant root cause",
            correct_solution="Verify auth service and restart token cache",
        ),
    ]


# =====================================================================
# Search and update
# =====================================================================


def test_query_returns_ranked_hits() -> None:
    kb = KnowledgeBase(_base_articles())
    result = kb.query("database oom memory")
    assert len(result.hits) >= 1
    assert result.hits[0].article_id == "KB-1"


def test_update_article_marks_persistence_eligibility() -> None:
    kb = KnowledgeBase(_base_articles())
    update = kb.update_article(
        title="Auth Fix",
        content="Find root cause and verify logs before fix rollout.",
    )
    assert update.accepted_for_persistence is True


def test_persistent_kb_keeps_accepted_contributions() -> None:
    incident = IncidentBank().get_incident(seed=0, difficulty="easy")
    persistent = PersistentKnowledgeBase(_base_articles())
    persistent.record_update(
        title="Payment Diagnostic",
        content="verify root cause and apply fix after verification",
        accepted_for_persistence=True,
    )
    episode_kb = persistent.reset_for_episode(incident)
    titles = [article.title for article in episode_kb.list_articles()]
    assert "Payment Diagnostic" in titles


# =====================================================================
# Query edge cases
# =====================================================================


def test_query_no_match_returns_empty() -> None:
    kb = KnowledgeBase(_base_articles())
    result = kb.query("nonexistent topic xyz")
    assert result.hits == []


def test_query_partial_match() -> None:
    kb = KnowledgeBase(_base_articles())
    result = kb.query("payments restart")
    assert len(result.hits) >= 1
    assert any(hit.article_id == "KB-2" for hit in result.hits)


def test_query_stores_query_string() -> None:
    kb = KnowledgeBase(_base_articles())
    result = kb.query("memory fix")
    assert result.query == "memory fix"


def test_update_existing_article_by_title() -> None:
    kb = KnowledgeBase(_base_articles())
    result = kb.update_article(
        title="Database OOM Recovery",
        content="root cause verify fix updated procedure",
    )
    assert result.article_id == "KB-1"
    assert result.updated is True


def test_update_creates_new_article_when_title_not_found() -> None:
    kb = KnowledgeBase(_base_articles())
    result = kb.update_article(
        title="Brand New Topic",
        content="root cause verify fix procedure",
    )
    assert result.article_id.startswith("KB-AGENT-")
    assert result.updated is True


def test_low_quality_update_not_persisted() -> None:
    kb = KnowledgeBase(_base_articles())
    result = kb.update_article(
        title="Short",
        content="just some text without keywords",
    )
    assert result.accepted_for_persistence is False


def test_list_articles_returns_all() -> None:
    kb = KnowledgeBase(_base_articles())
    assert len(kb.list_articles()) == 2


def test_persistent_kb_rejects_low_quality() -> None:
    persistent = PersistentKnowledgeBase(_base_articles())
    persistent.record_update(
        title="Bad",
        content="garbage",
        accepted_for_persistence=False,
    )
    assert persistent.contribution_count() == 0


def test_persistent_kb_contribution_count() -> None:
    persistent = PersistentKnowledgeBase(_base_articles())
    persistent.record_update(
        title="Good Article",
        content="root cause verify fix",
        accepted_for_persistence=True,
    )
    persistent.record_update(
        title="Another Good",
        content="root cause verify fix",
        accepted_for_persistence=True,
    )
    assert persistent.contribution_count() == 2
