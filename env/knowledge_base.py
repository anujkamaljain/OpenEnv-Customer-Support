"""Knowledge base simulation with deterministic search and persistence."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from models.incident import IncidentScenario, KBArticleState

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


class KBArticle(BaseModel):
    """Knowledge base article with hidden accuracy fields."""

    article_id: str
    title: str
    content: str
    solution_steps: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    last_updated: str
    is_accurate: bool
    outdated_reason: str | None = None
    correct_solution: str | None = None


class KBSearchHit(BaseModel):
    """Single article search hit."""

    model_config = {"frozen": True}

    article_id: str
    title: str
    summary: str
    confidence: float


class KBQueryResult(BaseModel):
    """Search response from the knowledge base."""

    model_config = {"frozen": True}

    query: str
    hits: list[KBSearchHit] = Field(default_factory=list)


class KBUpdateResult(BaseModel):
    """Result of adding or updating a KB article."""

    model_config = {"frozen": True}

    article_id: str
    updated: bool
    accepted_for_persistence: bool
    message: str


class KnowledgeBase:
    """Simulated knowledge base with staleness tracking."""

    def __init__(self, articles: list[KBArticle]) -> None:
        self._articles: dict[str, KBArticle] = {article.article_id: article for article in articles}

    def query(self, search_query: str) -> KBQueryResult:
        """Search KB using deterministic keyword overlap."""
        query_tokens = _tokenize(search_query)
        hits: list[KBSearchHit] = []
        for article in self._articles.values():
            confidence = _keyword_overlap(query_tokens, _article_tokens(article))
            if confidence <= 0:
                continue
            hits.append(
                KBSearchHit(
                    article_id=article.article_id,
                    title=article.title,
                    summary=article.content[:140],
                    confidence=confidence,
                )
            )
        hits.sort(key=lambda h: (-h.confidence, h.article_id))
        return KBQueryResult(query=search_query, hits=hits)

    def update_article(self, title: str, content: str) -> KBUpdateResult:
        """Add or update an article and score persistence eligibility."""
        existing = self._find_by_title(title)
        accepted = _is_high_quality_update(content)
        if existing is not None:
            article_id = existing.article_id
            existing.content = content
            existing.last_updated = "2026-04-20"
            existing.tags = sorted(set(existing.tags + _tokenize(title)))
            return KBUpdateResult(
                article_id=article_id,
                updated=True,
                accepted_for_persistence=accepted,
                message="Article updated.",
            )
        article_id = f"KB-AGENT-{len(self._articles) + 1:03d}"
        article = KBArticle(
            article_id=article_id,
            title=title,
            content=content,
            solution_steps=[content],
            tags=_tokenize(title),
            last_updated="2026-04-20",
            is_accurate=True,
            correct_solution=content if accepted else None,
        )
        self._articles[article_id] = article
        return KBUpdateResult(
            article_id=article_id,
            updated=True,
            accepted_for_persistence=accepted,
            message="Article created.",
        )

    def list_articles(self) -> list[KBArticle]:
        """Return all KB articles."""
        return list(self._articles.values())

    def _find_by_title(self, title: str) -> KBArticle | None:
        lower = title.strip().lower()
        for article in self._articles.values():
            if article.title.strip().lower() == lower:
                return article
        return None


class PersistentKnowledgeBase:
    """KB that accumulates accepted agent contributions across episodes."""

    def __init__(self, base_articles: list[KBArticle]) -> None:
        self._base = list(base_articles)
        self._agent_contributions: list[KBArticle] = []

    def reset_for_episode(self, incident: IncidentScenario) -> KnowledgeBase:
        """Load base + scenario + accepted contributions for the next episode."""
        scenario_articles = [_from_state(article) for article in incident.kb_articles]
        combined = self._base + scenario_articles + list(self._agent_contributions)
        return KnowledgeBase(articles=combined)

    def record_update(
        self, title: str, content: str, accepted_for_persistence: bool
    ) -> None:
        """Record a correct contribution for future episodes."""
        if not accepted_for_persistence:
            return
        article = KBArticle(
            article_id=f"KB-PERSIST-{len(self._agent_contributions) + 1:03d}",
            title=title,
            content=content,
            solution_steps=[content],
            tags=_tokenize(title),
            last_updated="2026-04-20",
            is_accurate=True,
            correct_solution=content,
        )
        self._agent_contributions.append(article)

    def contribution_count(self) -> int:
        """Return number of persistent contributions."""
        return len(self._agent_contributions)


def _from_state(state: KBArticleState) -> KBArticle:
    content = state.summary
    return KBArticle(
        article_id=state.article_id,
        title=state.title,
        content=content,
        solution_steps=[content],
        tags=_tokenize(f"{state.title} {state.summary}"),
        last_updated="2026-04-20",
        is_accurate=state.is_accurate,
        outdated_reason=None,
        correct_solution=None,
    )


def _tokenize(text: str) -> list[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


def _article_tokens(article: KBArticle) -> list[str]:
    corpus = " ".join([article.title, article.content, " ".join(article.tags)])
    return _tokenize(corpus)


def _keyword_overlap(query_tokens: list[str], article_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    matches = sum(1 for token in query_tokens if token in article_tokens)
    return round(matches / len(query_tokens), 3)


def _is_high_quality_update(content: str) -> bool:
    normalized = " ".join(_tokenize(content))
    required = ("root cause", "verify", "fix")
    return all(term in normalized for term in required)
