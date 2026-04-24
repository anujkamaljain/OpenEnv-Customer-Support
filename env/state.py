"""Internal mutable state for a single episode."""

from __future__ import annotations

from typing import Any, Literal

from env.customers import CustomerQueueManager
from env.incident_history import IncidentHistoryStore
from env.knowledge_base import KnowledgeBase, PersistentKnowledgeBase
from env.policy_engine import PolicyEngine
from env.runbooks import RunbookEngine
from env.stakeholders import StakeholderManager
from env.world import WorldState
from graders.investigation_grader import ACTION_COSTS, EvidenceChain, SeverityReEvaluation
from models.action import ApplyFixAction
from models.incident import IncidentScenario
from models.observation import ActionRecord, Observation, Phase
from models.ticket import TicketData

PHASE_VALID_ACTIONS: dict[Phase, frozenset[str]] = {
    "unclassified": frozenset(["classify"]),
    "classified": frozenset(["route", "escalate"]),
    "routed": frozenset(["respond", "escalate", "resolve", "request_info"]),
    "responding": frozenset(["respond", "escalate", "resolve", "request_info"]),
    "escalated": frozenset(["resolve"]),
    "resolved": frozenset(),
}

MAX_STEPS: dict[str, int] = {
    "easy": 8,
    "medium": 9,
    "hard": 10,
}

IncidentPhase = Literal["triage", "investigation", "response", "resolution"]
IncidentSeverity = Literal["medium", "high", "critical", "P0"]

INCIDENT_PHASE_VALID_ACTIONS: dict[IncidentPhase, frozenset[str]] = {
    "triage": frozenset(
        ["classify", "check_monitoring", "query_kb", "query_incident_history", "follow_runbook_step"]
    ),
    "investigation": frozenset(
        [
            "check_monitoring",
            "probe_service",
            "fetch_logs",
            "fetch_user_data",
            "check_billing",
            "query_kb",
            "check_policy",
            "query_incident_history",
            "follow_runbook_step",
            "classify",
            "route",
        ]
    ),
    "response": frozenset(
        [
            "apply_fix",
            "rollback_fix",
            "respond",
            "escalate",
            "request_info",
            "notify_stakeholders",
            "check_policy",
            "fetch_user_data",
            "check_billing",
            "query_kb",
            "follow_runbook_step",
        ]
    ),
    "resolution": frozenset(
        [
            "verify_fix",
            "resolve",
            "respond",
            "write_postmortem",
            "update_kb",
            "notify_stakeholders",
        ]
    ),
}

SEVERITY_ESCALATION: dict[int, IncidentSeverity] = {
    10: "high",
    25: "critical",
    40: "P0",
}


def compute_max_total_reward(ticket: TicketData) -> float:
    """Achievable maximum reward for an optimal agent on this ticket.

    Accounts for action rewards *and* any unavoidable SLA penalties incurred
    when the minimum optimal path exceeds the ticket's SLA deadline.
    """
    total = 0.10  # classify base
    if ticket.gold_priority in ("critical", "high"):
        total += 0.10  # urgency bonus
    total += 0.10  # route

    min_steps = 3  # classify + route + resolve (always required)
    if ticket.difficulty in ("medium", "hard"):
        total += 0.20  # respond
        min_steps += 1
    if ticket.partial_info:
        total += 0.05  # request_info bonus
        min_steps += 1
    if ticket.requires_escalation:
        total += 0.15  # escalate
        min_steps += 1
    total += 0.25  # resolve

    sla = ticket.effective_sla_steps
    for step_idx in range(min_steps):
        if step_idx >= sla:
            total -= 0.02 * (step_idx - sla + 1)

    return round(total, 4)


class InternalState:
    """Tracks episode progress, phase transitions, and cumulative quality scores."""

    __slots__ = (
        "ticket",
        "phase",
        "steps_taken",
        "max_steps",
        "max_total_reward",
        "actions_log",
        "cumulative_reward",
        "classification_correct",
        "routing_correct",
        "urgency_handled",
        "response_quality_score",
        "resolution_quality_score",
        "escalation_score",
        "constraints_violated",
        "done",
        "last_action_json",
        # v2 additions
        "sla_steps",
        "urgency_penalty_accrued",
        "info_requested",
        # v3 — interpretability
        "last_reward_breakdown",
    )

    def __init__(self, ticket: TicketData) -> None:
        self.ticket = ticket
        self.phase: Phase = "unclassified"
        self.steps_taken: int = 0
        self.max_steps: int = MAX_STEPS[ticket.difficulty]
        self.max_total_reward: float = compute_max_total_reward(ticket)
        self.actions_log: list[ActionRecord] = []
        self.cumulative_reward: float = 0.0

        self.classification_correct: bool | None = None
        self.routing_correct: bool | None = None
        self.urgency_handled: bool = False
        self.response_quality_score: float | None = None
        self.resolution_quality_score: float | None = None
        self.escalation_score: float | None = None
        self.constraints_violated: list[str] = []
        self.done: bool = False
        self.last_action_json: str | None = None

        # v2
        self.sla_steps: int = ticket.effective_sla_steps
        self.urgency_penalty_accrued: float = 0.0
        self.info_requested: bool = False

        # v3 — interpretability
        self.last_reward_breakdown: dict[str, float] = {}

    # ---- helpers --------------------------------------------------------

    @property
    def available_actions(self) -> list[str]:
        actions = PHASE_VALID_ACTIONS[self.phase]
        if self.info_requested:
            actions = actions - frozenset(["request_info"])
        return sorted(actions)

    def record_action(
        self, action_summary: str, feedback: str, reward: float
    ) -> None:
        self.actions_log.append(
            ActionRecord(
                step=self.steps_taken,
                action_taken=action_summary,
                env_feedback=feedback,
                reward_earned=round(reward, 4),
            )
        )
        self.cumulative_reward += reward
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.done = True

    def to_observation(self) -> Observation:
        return Observation(
            ticket_id=self.ticket.ticket_id,
            ticket_text=self.ticket.ticket_text,
            customer_sentiment=self.ticket.customer_sentiment,
            customer_tier=self.ticket.customer_tier,
            category_hint=self.ticket.category_hint,
            history=list(self.actions_log),
            pending_tickets=0,
            current_step=self.steps_taken,
            max_steps=self.max_steps,
            constraints=list(self.ticket.constraints),
            available_actions=self.available_actions,
            phase=self.phase,
            sla_steps_remaining=max(0, self.sla_steps - self.steps_taken),
            customer_value=self.ticket.customer_value,
            max_total_reward=self.max_total_reward,
        )

    def to_info(self) -> dict[str, Any]:
        mtr = self.max_total_reward
        info: dict[str, Any] = {
            "phase": self.phase,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "max_total_reward": mtr,
            "normalized_score": round(
                min(max(self.cumulative_reward / mtr, 0.0), 1.0), 4
            ),
            "classification_correct": self.classification_correct,
            "routing_correct": self.routing_correct,
            "urgency_handled": self.urgency_handled,
            "response_quality_score": self.response_quality_score,
            "resolution_quality_score": self.resolution_quality_score,
            "escalation_score": self.escalation_score,
            "constraints_violated": list(self.constraints_violated),
            "difficulty": self.ticket.difficulty,
            "sla_steps": self.sla_steps,
            "sla_overage": max(0, self.steps_taken - self.sla_steps),
            "urgency_penalty_accrued": round(self.urgency_penalty_accrued, 4),
            "customer_value": self.ticket.customer_value,
            "reward_breakdown": dict(self.last_reward_breakdown),
        }
        if self.done:
            info["final_score_breakdown"] = self._compute_final_breakdown()
        return info

    def _compute_final_breakdown(self) -> dict[str, float]:
        """Episode-level weighted breakdown mirroring ``grade_episode``."""
        cls_s = 1.0 if self.classification_correct else 0.0
        rte_s = 1.0 if self.routing_correct else 0.0
        rsp_s = self.response_quality_score if self.response_quality_score is not None else 0.0
        res_s = self.resolution_quality_score if self.resolution_quality_score is not None else 0.0
        esc_s = max(0.0, self.escalation_score) if self.escalation_score is not None else 0.0
        urg_s = 1.0 if self.urgency_handled else 0.0
        eff_s = max(0.0, 1.0 - self.steps_taken / self.max_steps) if self.max_steps > 0 else 0.0

        sla_overage = max(0, self.steps_taken - self.sla_steps)
        sla_s = max(0.0, 1.0 - sla_overage * 0.2)

        constraint_pen = len(self.constraints_violated) * 0.05

        raw = (
            0.15 * cls_s
            + 0.10 * rte_s
            + 0.20 * rsp_s
            + 0.20 * res_s
            + 0.10 * esc_s
            + 0.10 * urg_s
            + 0.05 * eff_s
            + 0.10 * sla_s
        )

        return {
            "classification": round(0.15 * cls_s, 4),
            "routing": round(0.10 * rte_s, 4),
            "response_quality": round(0.20 * rsp_s, 4),
            "resolution_quality": round(0.20 * res_s, 4),
            "escalation": round(0.10 * esc_s, 4),
            "urgency": round(0.10 * urg_s, 4),
            "efficiency": round(0.05 * eff_s, 4),
            "sla_compliance": round(0.10 * sla_s, 4),
            "constraint_penalty": round(-constraint_pen, 4),
            "total": round(max(0.0, min(raw - constraint_pen, 1.0)), 4),
        }


class ResourceBudget:
    """Finite resources available during an incident episode."""

    __slots__ = (
        "max_fix_attempts",
        "max_escalations",
        "max_stakeholder_notifications",
        "remaining_fix_attempts",
        "remaining_escalations",
        "remaining_notifications",
    )

    def __init__(
        self,
        max_fix_attempts: int = 3,
        max_escalations: int = 2,
        max_stakeholder_notifications: int = 5,
    ) -> None:
        self.max_fix_attempts = max_fix_attempts
        self.max_escalations = max_escalations
        self.max_stakeholder_notifications = max_stakeholder_notifications
        self.remaining_fix_attempts = max_fix_attempts
        self.remaining_escalations = max_escalations
        self.remaining_notifications = max_stakeholder_notifications

    def consume(self, resource: str) -> bool:
        """Consume one unit of a named resource if available."""
        if resource == "fix_attempt":
            if self.remaining_fix_attempts <= 0:
                return False
            self.remaining_fix_attempts -= 1
            return True
        if resource == "escalation":
            if self.remaining_escalations <= 0:
                return False
            self.remaining_escalations -= 1
            return True
        if resource == "notification":
            if self.remaining_notifications <= 0:
                return False
            self.remaining_notifications -= 1
            return True
        return False


class AuditEntry:
    """Compliance audit row for one action."""

    __slots__ = (
        "step",
        "timestamp_simulated",
        "action_type",
        "target",
        "rationale_required",
        "policy_checked",
        "compliant",
    )

    def __init__(
        self,
        *,
        step: int,
        timestamp_simulated: str,
        action_type: str,
        target: str,
        rationale_required: bool,
        policy_checked: bool,
        compliant: bool,
    ) -> None:
        self.step = step
        self.timestamp_simulated = timestamp_simulated
        self.action_type = action_type
        self.target = target
        self.rationale_required = rationale_required
        self.policy_checked = policy_checked
        self.compliant = compliant


class AuditTrail:
    """Compliance audit trail across incident actions."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def append(self, entry: AuditEntry) -> None:
        """Append a new audit entry."""
        self.entries.append(entry)

    def grade_compliance(self) -> float:
        """Return required-policy-check compliance ratio."""
        requiring = [
            entry for entry in self.entries if entry.action_type in _POLICY_SENSITIVE_ACTIONS
        ]
        if not requiring:
            return 1.0
        compliant = sum(1 for entry in requiring if entry.policy_checked and entry.compliant)
        return round(compliant / len(requiring), 4)


class ChaosEvent:
    """Mid-episode injected failure event."""

    __slots__ = ("step", "new_service", "reason", "alert_text")

    def __init__(self, step: int, new_service: str, reason: str, alert_text: str) -> None:
        self.step = step
        self.new_service = new_service
        self.reason = reason
        self.alert_text = alert_text


class ChaosInjector:
    """Inject deterministic new failures during response phase."""

    CHAOS_TRIGGERS: dict[str, dict[str, object]] = {
        "hard": {
            "trigger_step": 35,
            "probability": 0.5,
            "new_failure": {
                "service": "notifications",
                "mode": "queue_overflow",
                "reason": "Backpressure from payment retry storm",
            },
        },
        "nightmare": {
            "trigger_step": 25,
            "probability": 1.0,
            "new_failure": {
                "service": "analytics",
                "mode": "batch_job_runaway",
                "reason": "Error logging spike triggered batch reprocessing",
            },
        },
    }

    def maybe_inject(self, world: WorldState, step: int, difficulty: str) -> ChaosEvent | None:
        """Inject deterministic chaos event based on seed and step."""
        config = self.CHAOS_TRIGGERS.get(difficulty)
        if config is None:
            return None
        trigger_step = int(config["trigger_step"])
        probability = float(config["probability"])
        if step < trigger_step:
            return None
        if not self._should_trigger(world.seed, step, probability):
            return None
        failure = config["new_failure"]
        service = str(failure["service"])
        mode = str(failure["mode"])
        reason = str(failure["reason"])
        world.service_mesh.inject_failure(service, mode)
        return ChaosEvent(
            step=step,
            new_service=service,
            reason=reason,
            alert_text=f"NEW ALERT: {service} showing errors",
        )

    @staticmethod
    def _should_trigger(seed: int, step: int, probability: float) -> bool:
        threshold = int(probability * 100)
        value = (seed * 31 + step * 17) % 100
        return value < threshold


class IncidentState:
    """Tracks incident lifecycle progression and integrated world state."""

    __slots__ = (
        "incident",
        "world",
        "incident_phase",
        "triage_complete",
        "investigation_complete",
        "response_complete",
        "episode_done",
        "root_cause_identified",
        "fix_applied",
        "fix_verified",
        "tickets_resolved",
        "tools_used_sequence",
        "policies_checked",
        "kb_queried",
        "logs_checked_for",
        "postmortem_written",
        "kb_updated",
        "steps_taken",
        "max_steps",
        "cumulative_reward",
        "last_action_json",
        "last_reward_breakdown",
        "known_facts",
        "active_policies",
        "tool_results",
        "active_alerts",
        "resource_budget",
        "audit_trail",
        "current_severity",
        "severity_re_eval",
        "_pending_reclassification",
        "chaos_injector",
        "has_escalated",
        "total_action_cost",
        "evidence_chain",
        "crm",
        "billing",
        "policy_engine",
        "history_store",
        "runbook_engine",
        "stakeholder_mgr",
        "customer_queue_mgr",
        "persistent_kb",
        "knowledge_base",
        "suggested_runbook",
        "actions_log",
    )

    def __init__(
        self,
        incident: IncidentScenario,
        world: WorldState,
        *,
        crm: object,
        billing: object,
        policy_engine: PolicyEngine,
        history_store: IncidentHistoryStore,
        runbook_engine: RunbookEngine,
        stakeholder_mgr: StakeholderManager,
        customer_queue_mgr: CustomerQueueManager,
        persistent_kb: PersistentKnowledgeBase,
        knowledge_base: KnowledgeBase,
        suggested_runbook: dict[str, object] | None,
    ) -> None:
        self.incident = incident
        self.world = world
        self.incident_phase: IncidentPhase = "triage"
        self.triage_complete = False
        self.investigation_complete = False
        self.response_complete = False
        self.episode_done = False

        self.root_cause_identified = False
        self.fix_applied = False
        self.fix_verified = False
        self.tickets_resolved: list[str] = []
        self.tools_used_sequence: list[str] = []
        self.policies_checked: set[str] = set()
        self.kb_queried = False
        self.logs_checked_for: set[str] = set()
        self.postmortem_written = False
        self.kb_updated = False

        self.steps_taken = 0
        self.max_steps = incident.max_steps
        self.cumulative_reward = 0.0
        self.last_action_json: str | None = None
        self.last_reward_breakdown: dict[str, float] = {}
        self.known_facts: dict[str, object] = {}
        self.active_policies: dict[str, object] = {}
        self.tool_results: dict[str, object] | None = None
        self.active_alerts: list[str] = []

        self.resource_budget = ResourceBudget()
        self.audit_trail = AuditTrail()
        self.current_severity = _severity_for_difficulty(incident.difficulty)
        self.severity_re_eval = SeverityReEvaluation()
        self._pending_reclassification: IncidentSeverity | None = None
        self.chaos_injector = ChaosInjector()
        self.has_escalated = False
        self.total_action_cost = 0.0
        self.evidence_chain = EvidenceChain()

        self.crm = crm
        self.billing = billing
        self.policy_engine = policy_engine
        self.history_store = history_store
        self.runbook_engine = runbook_engine
        self.stakeholder_mgr = stakeholder_mgr
        self.customer_queue_mgr = customer_queue_mgr
        self.persistent_kb = persistent_kb
        self.knowledge_base = knowledge_base
        self.suggested_runbook = suggested_runbook
        self.actions_log: list[ActionRecord] = []

    @property
    def available_actions(self) -> list[str]:
        """Return actions available in the current incident phase."""
        return sorted(INCIDENT_PHASE_VALID_ACTIONS[self.incident_phase])

    def record_action(self, action_type: str, feedback: str, reward: float) -> None:
        """Record action outcome in history and counters."""
        self.actions_log.append(
            ActionRecord(
                step=self.steps_taken,
                action_taken=action_type,
                env_feedback=feedback,
                reward_earned=round(reward, 4),
            )
        )
        self.steps_taken += 1
        self.cumulative_reward += reward
        self.tools_used_sequence.append(action_type)
        self.total_action_cost = round(
            self.total_action_cost + ACTION_COSTS.get(action_type, 0), 2
        )
        if self.steps_taken >= self.max_steps:
            self.episode_done = True

    def add_audit_entry(
        self,
        *,
        action_type: str,
        target: str,
        policy_checked: bool,
        compliant: bool,
    ) -> None:
        """Append compliance audit entry for current step."""
        self.audit_trail.append(
            AuditEntry(
                step=self.steps_taken,
                timestamp_simulated=f"T+{self.steps_taken:03d}",
                action_type=action_type,
                target=target,
                rationale_required=action_type in {"apply_fix", "escalate"},
                policy_checked=policy_checked,
                compliant=compliant,
            )
        )

    def advance_phase(self) -> None:
        """Advance phase when completion conditions are met."""
        if self.incident_phase == "triage" and self.triage_complete:
            self.incident_phase = "investigation"
            return
        if self.incident_phase == "investigation" and self.investigation_complete:
            self.incident_phase = "response"
            return
        if self.incident_phase == "response" and self.response_complete:
            self.incident_phase = "resolution"

    def all_objectives_complete(self) -> bool:
        """Return True when incident is fully resolved."""
        return self.fix_verified and self.postmortem_written and self.kb_updated

    def apply_severity_auto_escalation(self) -> None:
        """Escalate severity if unresolved for long durations."""
        for step_trigger, new_severity in SEVERITY_ESCALATION.items():
            if self.steps_taken >= step_trigger and not self.response_complete:
                self.current_severity = new_severity

    def maybe_update_reclassification(self) -> None:
        """Check whether evidence implies severity reclassification."""
        reward, _, target = self.severity_re_eval.check_reclassification(
            evidence_chain=self.evidence_chain,
            current_step=self.steps_taken,
            current_severity=self.current_severity,
        )
        if reward != 0.0:
            self._pending_reclassification = target

    def mark_fix_applied(self, action: ApplyFixAction, correct: bool) -> None:
        """Mark fix state and progression."""
        self.fix_applied = True
        if correct:
            self.response_complete = True
        self.add_audit_entry(
            action_type="apply_fix",
            target=action.service_name,
            policy_checked="escalation" in self.policies_checked,
            compliant=correct,
        )

    def to_observation(self) -> Observation:
        """Build incident-mode observation payload."""
        ticket = self.world.support_queue[0] if self.world.support_queue else None
        ticket_text = (
            getattr(ticket, "body", None) or getattr(ticket, "ticket_text", None) or ""
        ) if ticket is not None else self.incident.description
        alerts = self.world.service_mesh.generate_alerts(self.steps_taken)
        self.active_alerts = [alert.message for alert in alerts]
        return Observation(
            ticket_id=ticket.ticket_id if ticket is not None else self.incident.incident_id,
            ticket_text=ticket_text,
            customer_sentiment="frustrated",
            customer_tier="enterprise",
            customer_value="high",
            category_hint=None,
            constraints=[],
            phase="responding" if self.incident_phase in ("response", "resolution") else "classified",
            available_actions=self.available_actions,
            current_step=self.steps_taken,
            max_steps=self.max_steps,
            sla_steps_remaining=max(0, self.max_steps - self.steps_taken),
            history=list(self.actions_log),
            max_total_reward=self.incident.max_total_reward,
            incident_id=self.incident.incident_id,
            incident_title=self.incident.title,
            mode="incident",
            system_status=self.world.service_mesh.get_health_summary(),
            active_alerts=list(self.active_alerts),
            tool_results=self.tool_results,
            known_facts=dict(self.known_facts),
            active_policies=dict(self.active_policies),
            stakeholder_patience=self.stakeholder_mgr.get_patience_levels(),
            pending_customer_tickets=len(self.world.support_queue),
            incident_phase=self.incident_phase,
            suggested_runbook=self.suggested_runbook,
            total_incident_cost=round(self.world.total_downtime_cost + self.total_action_cost, 2),
        )

    def to_info(self) -> dict[str, object]:
        """Build incident-mode diagnostics payload."""
        return {
            "mode": "incident",
            "incident_id": self.incident.incident_id,
            "incident_phase": self.incident_phase,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "known_facts": dict(self.known_facts),
            "active_policies": dict(self.active_policies),
            "total_incident_cost": round(self.world.total_downtime_cost + self.total_action_cost, 2),
            "reward_breakdown": dict(self.last_reward_breakdown),
            "resource_budget": {
                "remaining_fix_attempts": self.resource_budget.remaining_fix_attempts,
                "remaining_escalations": self.resource_budget.remaining_escalations,
                "remaining_notifications": self.resource_budget.remaining_notifications,
            },
            "compliance_score": self.audit_trail.grade_compliance(),
            "current_severity": self.current_severity,
        }


_POLICY_SENSITIVE_ACTIONS = frozenset(
    ["apply_fix", "escalate", "notify_stakeholders", "update_kb"]
)


def _severity_for_difficulty(difficulty: str) -> IncidentSeverity:
    if difficulty == "easy":
        return "medium"
    if difficulty == "medium":
        return "high"
    return "critical"
