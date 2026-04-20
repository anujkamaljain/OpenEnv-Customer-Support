"""Deterministic service mesh simulation for cascading incident scenarios."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.incident import RedHerring

HealthState = Literal["healthy", "degraded", "flickering", "down"]
FixType = Literal[
    "restart_service",
    "memory_increase",
    "config_change",
    "schema_migration",
    "data_fix",
]
ProbeType = Literal["logs", "resources", "connections", "config"]
Priority = Literal["low", "medium", "high"]
ServiceName = Literal["auth", "database", "payments", "analytics", "notifications"]

SERVICES: dict[ServiceName, dict[str, object]] = {
    "auth": {
        "description": "Authentication & token management",
        "dependencies": [],
        "failure_modes": ["rate_limiting", "token_expiry", "config_corruption"],
    },
    "database": {
        "description": "Primary data store",
        "dependencies": [],
        "failure_modes": ["oom", "connection_pool_exhaustion", "replication_lag"],
    },
    "payments": {
        "description": "Transaction processing",
        "dependencies": ["auth", "database"],
        "failure_modes": ["gateway_timeout", "validation_errors", "idempotency_failure"],
    },
    "analytics": {
        "description": "Reporting & metrics",
        "dependencies": ["database"],
        "failure_modes": ["batch_job_runaway", "query_timeout", "stale_cache"],
    },
    "notifications": {
        "description": "Email, SMS, push notifications",
        "dependencies": ["payments"],
        "failure_modes": ["queue_overflow", "template_error", "rate_exceeded"],
    },
}

SERVICE_OBSERVABILITY: dict[ServiceName, Literal["high", "medium", "low"]] = {
    "auth": "high",
    "database": "high",
    "payments": "medium",
    "analytics": "low",
    "notifications": "low",
}

FLICKER_PATTERNS: dict[str, list[HealthState]] = {
    "intermittent_oom": ["healthy", "degraded", "healthy", "healthy", "degraded", "down"],
    "connection_flap": ["healthy", "degraded", "healthy", "degraded"],
    "gc_pressure": ["healthy", "healthy", "degraded", "healthy", "healthy", "degraded"],
}

BLAST_RADIUS: dict[FixType, dict[str, str | int | float | bool]] = {
    "restart_service": {
        "damage": "temporary_outage",
        "duration": 3,
        "cascade": False,
        "penalty": -0.08,
        "description": "Service went offline during unnecessary restart",
    },
    "memory_increase": {
        "damage": "resource_starvation",
        "duration": 0,
        "cascade": True,
        "penalty": -0.10,
        "description": "Memory reallocation starved dependent services",
    },
    "config_change": {
        "damage": "misconfiguration",
        "duration": 0,
        "cascade": True,
        "penalty": -0.12,
        "description": "Bad configuration cascaded through service mesh",
    },
    "schema_migration": {
        "damage": "data_corruption",
        "duration": 0,
        "cascade": True,
        "penalty": -0.20,
        "description": "Schema migration on wrong table corrupted data",
    },
    "data_fix": {
        "damage": "data_loss",
        "duration": 0,
        "cascade": True,
        "penalty": -0.25,
        "description": "Direct data manipulation caused data loss",
    },
}

FAILURE_FIX_MAP: dict[str, FixType] = {
    "rate_limiting": "restart_service",
    "token_expiry": "config_change",
    "config_corruption": "config_change",
    "oom": "memory_increase",
    "connection_pool_exhaustion": "restart_service",
    "replication_lag": "schema_migration",
    "gateway_timeout": "restart_service",
    "validation_errors": "config_change",
    "idempotency_failure": "data_fix",
    "batch_job_runaway": "memory_increase",
    "query_timeout": "schema_migration",
    "stale_cache": "restart_service",
    "queue_overflow": "restart_service",
    "template_error": "config_change",
    "rate_exceeded": "memory_increase",
}

HEALTH_METRICS: dict[HealthState, tuple[float, int]] = {
    "healthy": (0.01, 50),
    "degraded": (0.45, 2000),
    "flickering": (0.25, 900),
    "down": (1.0, 10000),
}


class ServiceState(BaseModel):
    """Mutable runtime state for a single service."""

    name: ServiceName
    health: HealthState = "healthy"
    root_cause: str | None = None
    error_rate: float = 0.0
    latency_ms: int = 50
    affected_by: ServiceName | None = None
    is_root_cause: bool = False
    fix_applied: bool = False
    fix_correct: bool = False
    flicker_pattern: list[HealthState] | None = None
    flicker_step_index: int = 0
    blast_until_step: int = 0
    failure_step: int = 0


class FixResult(BaseModel):
    """Result of a normal fix attempt."""

    success: bool
    service: ServiceName
    fix_type: FixType
    message: str
    healed_services: list[ServiceName] = Field(default_factory=list)


class BlastRadiusResult(BaseModel):
    """Result of a wrong-fix blast radius event."""

    service: ServiceName
    fix_type: FixType
    damaged_services: list[ServiceName] = Field(default_factory=list)
    penalty: float
    description: str
    cascade: bool


class MonitoringRecord(BaseModel):
    """Agent-visible monitoring state for one service."""

    service: ServiceName
    health: HealthState
    error_rate: float
    latency_ms: int


class MonitoringSnapshot(BaseModel):
    """Agent-visible monitoring snapshot."""

    services: list[MonitoringRecord] = Field(default_factory=list)


class ProbeResult(BaseModel):
    """Deterministic diagnostics for a service probe."""

    service: ServiceName
    check_type: ProbeType
    observability: Literal["high", "medium", "low"]
    findings: list[str] = Field(default_factory=list)


class Alert(BaseModel):
    """Single alert emitted by the service mesh."""

    source: Literal["monitoring", "pagerduty"]
    service: ServiceName
    message: str
    is_actionable: bool
    priority: Priority


class FlickeringBehavior:
    """Deterministic intermittent health state sequencer."""

    def __init__(self, pattern: list[HealthState], seed: int) -> None:
        self.pattern = list(pattern)
        self.seed = seed

    def get_current_health(self, step: int) -> HealthState:
        """Return deterministic health for the given step."""
        return self.pattern[step % len(self.pattern)]


class ServiceMesh:
    """Deterministic simulation of 5 interconnected microservices."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.step = 0
        self.services: dict[ServiceName, ServiceState] = {}
        self.dependency_graph: dict[ServiceName, list[ServiceName]] = {}
        self._reverse_graph: dict[ServiceName, list[ServiceName]] = {}
        self._init_graph()
        self._init_services()

    def _init_graph(self) -> None:
        for name, config in SERVICES.items():
            deps = [dep for dep in config["dependencies"] if isinstance(dep, str)]
            typed_deps = [dep for dep in deps if dep in SERVICES]
            self.dependency_graph[name] = typed_deps  # type: ignore[assignment]
        self._reverse_graph = {name: [] for name in SERVICES}
        for node, deps in self.dependency_graph.items():
            for dep in deps:
                self._reverse_graph[dep].append(node)

    def _init_services(self) -> None:
        for name in SERVICES:
            self.services[name] = ServiceState(name=name)

    def inject_failure(self, service: ServiceName, failure_mode: str) -> None:
        """Inject a root-cause outage and deterministically cascade it."""
        state = self.services[service]
        state.health = "down"
        state.root_cause = failure_mode
        state.is_root_cause = True
        state.fix_applied = False
        state.fix_correct = False
        state.failure_step = self.step
        state.error_rate, state.latency_ms = HEALTH_METRICS["down"]
        self._cascade_from(service)

    def _cascade_from(self, upstream: ServiceName) -> None:
        for dependent in self._reverse_graph[upstream]:
            dep_state = self.services[dependent]
            if dep_state.is_root_cause:
                continue
            if dep_state.health == "down":
                continue
            dep_state.health = "degraded"
            dep_state.affected_by = upstream
            dep_state.fix_applied = False
            dep_state.fix_correct = False
            dep_state.failure_step = self.step
            dep_state.error_rate, dep_state.latency_ms = HEALTH_METRICS["degraded"]
            self._cascade_from(dependent)

    def set_flickering(self, service: ServiceName, pattern_name: str) -> None:
        """Enable deterministic flickering behavior for a service."""
        pattern = FLICKER_PATTERNS[pattern_name]
        state = self.services[service]
        state.health = "flickering"
        state.flicker_pattern = list(pattern)
        state.flicker_step_index = 0
        state.failure_step = self.step
        state.error_rate, state.latency_ms = HEALTH_METRICS["flickering"]

    def apply_fix(self, service: ServiceName, fix_type: FixType) -> FixResult:
        """Apply a fix attempt and return deterministic outcome."""
        state = self.services[service]
        state.fix_applied = True
        expected = FAILURE_FIX_MAP.get(state.root_cause or "")
        if state.is_root_cause and expected == fix_type:
            state.fix_correct = True
            healed = self._heal_from_root(service)
            return FixResult(
                success=True,
                service=service,
                fix_type=fix_type,
                message="Root cause fixed. Recovery started.",
                healed_services=healed,
            )
        state.fix_correct = False
        if state.health in ("down", "degraded", "flickering"):
            state.health = "healthy"
            state.error_rate, state.latency_ms = HEALTH_METRICS["healthy"]
        return FixResult(
            success=False,
            service=service,
            fix_type=fix_type,
            message="Symptom fix was temporary. Service may re-degrade.",
            healed_services=[],
        )

    def _heal_from_root(self, service: ServiceName) -> list[ServiceName]:
        healed: list[ServiceName] = [service]
        root_state = self.services[service]
        root_state.health = "healthy"
        root_state.error_rate, root_state.latency_ms = HEALTH_METRICS["healthy"]
        root_state.root_cause = None
        root_state.affected_by = None
        root_state.is_root_cause = False
        for dependent in self._reverse_graph[service]:
            healed.extend(self._heal_chain(dependent))
        return healed

    def _heal_chain(self, service: ServiceName) -> list[ServiceName]:
        state = self.services[service]
        healed = [service]
        if not state.is_root_cause:
            state.health = "healthy"
            state.error_rate, state.latency_ms = HEALTH_METRICS["healthy"]
            state.affected_by = None
            state.fix_applied = False
            state.fix_correct = False
        for dependent in self._reverse_graph[service]:
            healed.extend(self._heal_chain(dependent))
        return healed

    def apply_wrong_fix(self, service: ServiceName, fix_type: FixType) -> BlastRadiusResult:
        """Apply blast-radius damage for an incorrect fix."""
        spec = BLAST_RADIUS[fix_type]
        damaged: list[ServiceName] = [service]
        target = self.services[service]
        target.health = "down"
        target.blast_until_step = self.step + int(spec["duration"])
        target.error_rate, target.latency_ms = HEALTH_METRICS["down"]
        cascade_enabled = bool(spec["cascade"])
        if cascade_enabled:
            for dependent in self._reverse_graph[service]:
                dep = self.services[dependent]
                if dep.health != "down":
                    dep.health = "degraded"
                    dep.affected_by = service
                    dep.error_rate, dep.latency_ms = HEALTH_METRICS["degraded"]
                damaged.append(dependent)
        return BlastRadiusResult(
            service=service,
            fix_type=fix_type,
            damaged_services=damaged,
            penalty=float(spec["penalty"]),
            description=str(spec["description"]),
            cascade=cascade_enabled,
        )

    def tick_service_health(self, steps_since_failure: int) -> None:
        """Progress degradation and flickering in deterministic ticks."""
        self.step += 1
        for service in self.services.values():
            self._tick_blast_recovery(service)
            self._tick_flickering(service)
            self._tick_degraded(service, steps_since_failure)

    def _tick_blast_recovery(self, service: ServiceState) -> None:
        if service.blast_until_step <= 0:
            return
        if self.step >= service.blast_until_step:
            service.blast_until_step = 0
            service.health = "degraded"
            service.error_rate, service.latency_ms = HEALTH_METRICS["degraded"]

    def _tick_flickering(self, service: ServiceState) -> None:
        if service.flicker_pattern is None:
            return
        behavior = FlickeringBehavior(service.flicker_pattern, self.seed + service.failure_step)
        service.health = behavior.get_current_health(self.step)
        service.flicker_step_index = self.step % len(service.flicker_pattern)
        service.error_rate, service.latency_ms = HEALTH_METRICS[service.health]

    def _tick_degraded(self, service: ServiceState, steps_since_failure: int) -> None:
        if service.health != "degraded":
            return
        if service.fix_applied and not service.is_root_cause:
            if (self.step - service.failure_step) >= 2:
                service.health = "degraded"
                service.error_rate, service.latency_ms = HEALTH_METRICS["degraded"]
            return
        if steps_since_failure > 0 and steps_since_failure % 5 == 0:
            service.error_rate = min(1.0, service.error_rate + 0.1)
            service.latency_ms = min(10000, service.latency_ms + 500)
            if service.error_rate >= 0.9:
                service.health = "down"
                service.error_rate, service.latency_ms = HEALTH_METRICS["down"]
                self._cascade_from(service.name)

    def get_monitoring_data(self, service: ServiceName | None = None) -> MonitoringSnapshot:
        """Return monitoring-visible health, error rate, and latency."""
        if service is not None:
            state = self.services[service]
            return MonitoringSnapshot(services=[self._record_for(state)])
        records = [self._record_for(state) for state in self.services.values()]
        return MonitoringSnapshot(services=records)

    @staticmethod
    def _record_for(state: ServiceState) -> MonitoringRecord:
        return MonitoringRecord(
            service=state.name,
            health=state.health,
            error_rate=round(state.error_rate, 3),
            latency_ms=state.latency_ms,
        )

    def probe_service(self, service: ServiceName, check_type: ProbeType) -> ProbeResult:
        """Return deterministic probe diagnostics with observability gaps."""
        level = SERVICE_OBSERVABILITY[service]
        state = self.services[service]
        findings = self._probe_findings(state, check_type, level)
        return ProbeResult(
            service=service,
            check_type=check_type,
            observability=level,
            findings=findings,
        )

    def _probe_findings(
        self,
        state: ServiceState,
        check_type: ProbeType,
        level: Literal["high", "medium", "low"],
    ) -> list[str]:
        cause = state.root_cause or "unknown"
        if level == "low":
            return [f"{state.name} responding: {'no' if state.health == 'down' else 'yes'}"]
        if check_type == "connections":
            deps = self.dependency_graph[state.name]
            return [f"connections to: {', '.join(deps) if deps else 'none'}"]
        if level == "medium":
            return [f"{check_type} check indicates instability around {state.name}"]
        return [
            f"{check_type} check for {state.name}",
            f"observed failure signature: {cause}",
            f"health={state.health} error_rate={state.error_rate:.2f}",
        ]

    def get_health_summary(self) -> dict[str, str]:
        """Return a compact health map for all services."""
        return {name: state.health for name, state in self.services.items()}

    def get_dependencies(self, difficulty: str) -> dict[str, list[str]]:
        """Return known topology by difficulty level."""
        if difficulty in ("hard", "nightmare"):
            return {}
        return {name: list(deps) for name, deps in self.dependency_graph.items()}

    def generate_red_herrings(self, incident_id: str) -> list[RedHerring]:
        """Generate deterministic per-incident red herring symptoms."""
        templates = [
            RedHerring(
                service="analytics",
                symptom="CPU at 95%",
                actual_explanation="Scheduled batch processing",
                misleading_because="High CPU often suggests runaway workloads",
            ),
            RedHerring(
                service="auth",
                symptom="token refresh latency +200ms",
                actual_explanation="Expected during key rotation window",
                misleading_because="Latency spike resembles auth incident",
            ),
            RedHerring(
                service="notifications",
                symptom="queue depth increased",
                actual_explanation="Marketing campaign burst",
                misleading_because="Queue growth resembles delivery outage",
            ),
        ]
        index_base = (self.seed + sum(ord(ch) for ch in incident_id)) % len(templates)
        return [templates[index_base], templates[(index_base + 1) % len(templates)]]

    def generate_alerts(self, step: int) -> list[Alert]:
        """Generate deterministic actionable and noise alerts."""
        alerts: list[Alert] = []
        alerts.extend(self._real_alerts())
        alerts.extend(self._cascade_noise())
        alerts.extend(self._flicker_noise(step))
        alerts.extend(self._scheduled_noise(step))
        return alerts

    def _real_alerts(self) -> list[Alert]:
        alerts: list[Alert] = []
        for service in self.services.values():
            if service.health in ("degraded", "down"):
                alerts.append(
                    Alert(
                        source="monitoring",
                        service=service.name,
                        message=f"{service.name} error rate > {service.error_rate * 100:.0f}%",
                        is_actionable=True,
                        priority="high" if service.health == "down" else "medium",
                    )
                )
        return alerts

    def _cascade_noise(self) -> list[Alert]:
        alerts: list[Alert] = []
        for service in self.services.values():
            if service.affected_by is None:
                continue
            alerts.append(
                Alert(
                    source="monitoring",
                    service=service.name,
                    message=f"{service.name} latency +{service.latency_ms - 50}ms",
                    is_actionable=False,
                    priority="medium",
                )
            )
        return alerts

    def _flicker_noise(self, step: int) -> list[Alert]:
        alerts: list[Alert] = []
        for service in self.services.values():
            if service.flicker_pattern is None:
                continue
            if step % len(service.flicker_pattern) == 0:
                alerts.append(
                    Alert(
                        source="monitoring",
                        service=service.name,
                        message=f"{service.name} status changed: healthy ↔ degraded",
                        is_actionable=False,
                        priority="low",
                    )
                )
        return alerts

    def _scheduled_noise(self, step: int) -> list[Alert]:
        alerts: list[Alert] = []
        if (self.seed + step) % 3 == 0:
            alerts.append(
                Alert(
                    source="monitoring",
                    service="analytics",
                    message="analytics batch job CPU 88% (scheduled)",
                    is_actionable=False,
                    priority="low",
                )
            )
        if (self.seed + step) % 4 == 0:
            alerts.append(
                Alert(
                    source="pagerduty",
                    service="notifications",
                    message="Auto-recovery attempted on notifications service",
                    is_actionable=False,
                    priority="medium",
                )
            )
        return alerts
