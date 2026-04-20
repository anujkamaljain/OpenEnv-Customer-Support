"""Incident history store for deterministic past-incident lookup."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

Relevance = Literal["high", "medium", "low"]


class HistoricalIncident(BaseModel):
    """Historical incident record."""

    incident_id: str
    date: str
    title: str
    root_cause: str
    resolution: str
    services_affected: list[str] = Field(default_factory=list)
    is_relevant_to_current: bool


class HistoryHit(BaseModel):
    """Single history query hit."""

    model_config = {"frozen": True}

    incident_id: str
    date: str
    title: str
    root_cause: str
    resolution: str
    services_affected: list[str]
    relevance: Relevance


class HistoryQueryResult(BaseModel):
    """Result of querying incident history."""

    model_config = {"frozen": True}

    query: str
    hits: list[HistoryHit] = Field(default_factory=list)


class IncidentHistoryStore:
    """Database of past incidents for pattern matching."""

    def __init__(self, historical_incidents: list[HistoricalIncident]) -> None:
        self._incidents = list(historical_incidents)

    @staticmethod
    def from_json(path: Path) -> IncidentHistoryStore:
        """Load history records from JSON file."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        incidents = [HistoricalIncident.model_validate(item) for item in raw]
        return IncidentHistoryStore(incidents)

    def query(self, query: str, service_filter: str | None = None) -> HistoryQueryResult:
        """Search past incidents by keyword and optional service filter."""
        query_tokens = _tokenize(query)
        hits: list[tuple[float, HistoricalIncident]] = []
        for incident in self._incidents:
            if service_filter is not None and service_filter not in incident.services_affected:
                continue
            score = _score_incident(query_tokens, incident)
            if score <= 0:
                continue
            hits.append((score, incident))
        hits.sort(key=lambda item: (-item[0], item[1].incident_id))
        return HistoryQueryResult(query=query, hits=[_to_hit(score, incident) for score, incident in hits])


def _tokenize(text: str) -> list[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


def _score_incident(query_tokens: list[str], incident: HistoricalIncident) -> float:
    if not query_tokens:
        return 0.0
    corpus = " ".join([incident.title, incident.root_cause, incident.resolution] + incident.services_affected)
    incident_tokens = _tokenize(corpus)
    matches = sum(1 for token in query_tokens if token in incident_tokens)
    return matches / len(query_tokens)


def _to_hit(score: float, incident: HistoricalIncident) -> HistoryHit:
    relevance: Relevance = "low"
    if score >= 0.8:
        relevance = "high"
    elif score >= 0.4:
        relevance = "medium"
    return HistoryHit(
        incident_id=incident.incident_id,
        date=incident.date,
        title=incident.title,
        root_cause=incident.root_cause,
        resolution=incident.resolution,
        services_affected=list(incident.services_affected),
        relevance=relevance,
    )
