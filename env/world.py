"""World state container for the deterministic enterprise simulation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from env.services import Alert, ServiceMesh
from models.incident import DynamicTicketTrigger, IncidentScenario, TicketTemplate

EventType = Literal[
    "patience_decay",
    "ticket_arrival",
    "policy_drift",
    "downtime_cost",
    "alert_generated",
]


class Event(BaseModel):
    """Single deterministic world event emitted during a tick."""

    event_type: EventType
    message: str
    step: int


class WorldState:
    """Complete hidden state of the enterprise simulation."""

    __slots__ = (
        "seed",
        "incident",
        "service_mesh",
        "crm",
        "billing",
        "knowledge_base",
        "policy_engine",
        "support_queue",
        "resolved_tickets",
        "known_facts",
        "stakeholder_patience",
        "steps_elapsed",
        "incident_timer",
        "total_downtime_cost",
        "tools_used",
        "_tickets_emitted",
    )

    def __init__(self, seed: int, incident: IncidentScenario) -> None:
        self.seed = seed
        self.incident = incident
        self.service_mesh = ServiceMesh(seed=seed)

        # Placeholder enterprise systems for later phases.
        self.crm: dict[str, object] = {}
        self.billing: dict[str, object] = {}
        self.knowledge_base: dict[str, object] = {}
        self.policy_engine: dict[str, str | int | float | bool] = dict(incident.initial_policies)

        self.support_queue: list[TicketTemplate] = list(incident.initial_tickets)
        self.resolved_tickets: list[str] = []
        self.known_facts: dict[str, object] = {}
        self.stakeholder_patience: dict[str, float] = self._init_patience()
        self.steps_elapsed = 0
        self.incident_timer = 0
        self.total_downtime_cost = 0.0
        self.tools_used: list[str] = []
        self._tickets_emitted: set[str] = set()

        self._inject_initial_failures()

    def _init_patience(self) -> dict[str, float]:
        if self.incident.stakeholder_config:
            return {
                name: cfg.initial_patience
                for name, cfg in self.incident.stakeholder_config.items()
            }
        return {"vp": 1.0, "legal": 1.0, "support_lead": 1.0}

    def _inject_initial_failures(self) -> None:
        for cause in self.incident.root_causes:
            self.service_mesh.inject_failure(cause.service, cause.failure_mode)

    def tick(self) -> list[Event]:
        """Advance world state by one deterministic step."""
        self.steps_elapsed += 1
        self.incident_timer += 1
        events: list[Event] = []
        events.extend(self._tick_patience())
        events.extend(self._tick_ticket_schedule())
        events.extend(self._tick_policy_drift())
        self.service_mesh.tick_service_health(self.incident_timer)
        events.extend(self._tick_alerts())
        events.append(self._tick_downtime_cost())
        return events

    def _tick_patience(self) -> list[Event]:
        events: list[Event] = []
        for name, patience in self.stakeholder_patience.items():
            cfg = self.incident.stakeholder_config.get(name)
            decay = cfg.decay_per_step if cfg is not None else 0.02
            updated = max(0.0, round(patience - decay, 4))
            self.stakeholder_patience[name] = updated
            events.append(
                Event(
                    event_type="patience_decay",
                    message=f"{name} patience is now {updated:.2f}",
                    step=self.steps_elapsed,
                )
            )
        return events

    def _tick_ticket_schedule(self) -> list[Event]:
        events: list[Event] = []
        for trigger in self.incident.dynamic_ticket_schedule:
            events.extend(self._emit_ticket_if_due(trigger))
        return events

    def _emit_ticket_if_due(self, trigger: DynamicTicketTrigger) -> list[Event]:
        if self.steps_elapsed < trigger.trigger_step:
            return []
        if trigger.ticket.ticket_id in self._tickets_emitted:
            return []
        self._tickets_emitted.add(trigger.ticket.ticket_id)
        self.support_queue.append(trigger.ticket)
        return [
            Event(
                event_type="ticket_arrival",
                message=f"Ticket arrived: {trigger.ticket.ticket_id}",
                step=self.steps_elapsed,
            )
        ]

    def _tick_policy_drift(self) -> list[Event]:
        events: list[Event] = []
        for drift in self.incident.policy_drift_schedule:
            if drift.step != self.steps_elapsed:
                continue
            self.policy_engine[drift.key] = drift.value
            events.append(
                Event(
                    event_type="policy_drift",
                    message=f"Policy changed: {drift.key}={drift.value}",
                    step=self.steps_elapsed,
                )
            )
        return events

    def _tick_alerts(self) -> list[Event]:
        alerts = self.service_mesh.generate_alerts(self.steps_elapsed)
        return [self._alert_event(alert) for alert in alerts]

    def _alert_event(self, alert: Alert) -> Event:
        return Event(
            event_type="alert_generated",
            message=f"[{alert.priority}] {alert.service}: {alert.message}",
            step=self.steps_elapsed,
        )

    def _tick_downtime_cost(self) -> Event:
        down = 0
        degraded = 0
        for state in self.service_mesh.services.values():
            if state.health == "down":
                down += 1
            elif state.health in ("degraded", "flickering"):
                degraded += 1
        step_cost = float((down * 100) + (degraded * 40))
        self.total_downtime_cost = round(self.total_downtime_cost + step_cost, 2)
        return Event(
            event_type="downtime_cost",
            message=f"Downtime cost +{step_cost:.0f} (total={self.total_downtime_cost:.0f})",
            step=self.steps_elapsed,
        )


class WorldTickResult(BaseModel):
    """Optional structured wrapper for consuming world ticks."""

    events: list[Event] = Field(default_factory=list)
    steps_elapsed: int = 0
    total_downtime_cost: float = 0.0
