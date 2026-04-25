"""Deterministic failure curriculum scheduler for sandbox drill mode."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal

ServiceName = Literal["auth", "database", "payments", "analytics", "notifications"]
Severity = Literal["medium", "high", "critical"]

_SERVICE_FAILURES: dict[ServiceName, tuple[str, ...]] = {
    "auth": ("rate_limiting", "token_expiry", "config_corruption"),
    "database": ("oom", "connection_pool_exhaustion", "replication_lag"),
    "payments": ("gateway_timeout", "validation_errors", "idempotency_failure"),
    "analytics": ("batch_job_runaway", "query_timeout", "stale_cache"),
    "notifications": ("queue_overflow", "template_error", "rate_exceeded"),
}

_DIFFICULTY_EVENT_COUNT = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "nightmare": 4,
}


@dataclass(slots=True, frozen=True)
class DrillEvent:
    step: int
    service: ServiceName
    failure_mode: str
    severity: Severity
    deadline_step: int

    @property
    def key(self) -> str:
        return f"{self.step}:{self.service}:{self.failure_mode}"


def build_curriculum_schedule(
    *,
    seed: int,
    difficulty: str,
    max_steps: int,
) -> list[DrillEvent]:
    """Build deterministic mid-episode chaos events for drill mode."""
    count = _DIFFICULTY_EVENT_COUNT.get(difficulty, 2)
    rng = Random(seed * 31 + len(difficulty) * 17 + max_steps)
    services: list[ServiceName] = list(_SERVICE_FAILURES.keys())  # type: ignore[assignment]
    rng.shuffle(services)
    selected = services[:count]

    if max_steps < 8:
        return []
    spacing = max(3, max_steps // (count + 2))
    base_step = max(2, spacing)

    events: list[DrillEvent] = []
    for idx, service in enumerate(selected):
        modes = _SERVICE_FAILURES[service]
        mode = modes[rng.randrange(len(modes))]
        step = min(max_steps - 2, base_step + (idx * spacing))
        deadline = min(max_steps, step + max(3, spacing // 2 + 1))
        severity: Severity = "high" if idx % 2 == 0 else "medium"
        if difficulty in {"hard", "nightmare"} and idx == len(selected) - 1:
            severity = "critical"
        events.append(
            DrillEvent(
                step=step,
                service=service,
                failure_mode=mode,
                severity=severity,
                deadline_step=deadline,
            )
        )

    events.sort(key=lambda item: item.step)
    return events

