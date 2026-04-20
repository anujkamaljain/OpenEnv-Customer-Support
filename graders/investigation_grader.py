"""Deterministic grading utilities for incident-mode investigation workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from env.services import BLAST_RADIUS, FAILURE_FIX_MAP
from env.world import WorldState
from graders.grader import DeterministicGrader
from models.action import (
    ApplyFixAction,
    CheckMonitoringAction,
    FollowRunbookStepAction,
    ProbeServiceAction,
    QueryIncidentHistoryAction,
    WritePostmortemAction,
)

RiskLevel = Literal["low", "medium", "high", "critical"]
Severity = Literal["low", "medium", "high", "critical"]

ACTION_COSTS: dict[str, int] = {
    "check_monitoring": 5,
    "probe_service": 15,
    "fetch_logs": 10,
    "query_kb": 2,
    "query_incident_history": 3,
    "fetch_user_data": 5,
    "check_billing": 5,
    "check_policy": 2,
    "classify": 0,
    "route": 0,
    "apply_fix": 200,
    "rollback_fix": 150,
    "verify_fix": 10,
    "respond": 20,
    "resolve": 10,
    "escalate": 500,
    "notify_stakeholders": 50,
    "write_postmortem": 100,
    "update_kb": 30,
    "follow_runbook_step": 25,
    "request_info": 15,
}

STAKEHOLDER_COMMUNICATION_REQUIREMENTS: dict[str, dict[str, object]] = {
    "vp_engineering": {
        "wants": "executive_summary",
        "required_keywords": ["impact", "timeline", "resolution", "status"],
        "forbidden_keywords": ["technical details", "stack trace", "error code"],
        "max_length": 200,
        "tone": "concise",
    },
    "legal": {
        "wants": "compliance_report",
        "required_keywords": ["SLA", "compliance", "customer impact", "data", "exposure"],
        "forbidden_keywords": ["probably", "maybe", "I think", "guess"],
        "max_length": 500,
        "tone": "formal",
    },
    "support_lead": {
        "wants": "ticket_details",
        "required_keywords": ["customer", "ticket", "affected", "workaround", "ETA"],
        "forbidden_keywords": [],
        "max_length": 300,
        "tone": "empathetic",
    },
}


class EvidenceEntry(BaseModel):
    """One evidence entry captured during investigation."""

    step: int
    source: str
    finding: str
    conclusion: str
    service: str | None = None


class EvidenceChain(BaseModel):
    """Tracks the logical chain of evidence built by the agent."""

    entries: list[EvidenceEntry] = Field(default_factory=list)

    def add_evidence(
        self,
        step: int,
        source: str,
        finding: str,
        conclusion: str,
        service: str | None = None,
    ) -> None:
        """Append deterministic evidence entry."""
        self.entries.append(
            EvidenceEntry(
                step=step,
                source=source,
                finding=finding,
                conclusion=conclusion,
                service=service,
            )
        )

    def grade_chain_coherence(self, true_root_cause: str) -> float:
        """Return coherence score in [0, 1] for the evidence chain."""
        if not self.entries:
            return 0.0
        ordered = all(
            self.entries[i].step <= self.entries[i + 1].step
            for i in range(len(self.entries) - 1)
        )
        has_monitoring = any("monitoring" in entry.source for entry in self.entries)
        has_probe = any("probe" in entry.source for entry in self.entries)
        has_root = any(
            true_root_cause.lower() in entry.conclusion.lower()
            or true_root_cause.lower() in entry.finding.lower()
            for entry in self.entries
        )
        score = 0.0
        if ordered:
            score += 0.3
        if has_monitoring and has_probe:
            score += 0.3
        if has_root:
            score += 0.4
        return round(min(score, 1.0), 4)

    def has_evidence_for(self, service: str) -> bool:
        """Return whether service has associated evidence entries."""
        lowered = service.lower()
        return any(
            (entry.service is not None and entry.service.lower() == lowered)
            or lowered in entry.finding.lower()
            or lowered in entry.conclusion.lower()
            for entry in self.entries
        )


class ApprovalResult(BaseModel):
    """CAB review decision output."""

    approved: bool
    reason: str
    penalty: float = 0.0


class ChangeAdvisoryBoard:
    """Deterministic CAB simulation for production fix approvals."""

    FIX_RISK_LEVELS: dict[str, RiskLevel] = {
        "restart_service": "low",
        "clear_cache": "low",
        "increase_timeout": "low",
        "memory_increase": "medium",
        "connection_pool": "medium",
        "rate_limit_adjust": "medium",
        "config_change": "high",
        "rollback_deployment": "high",
        "schema_migration": "critical",
        "data_fix": "critical",
    }

    def review_fix(
        self, fix: ApplyFixAction, evidence_chain: EvidenceChain, escalated: bool
    ) -> ApprovalResult:
        """Simulate CAB review rules for a proposed fix."""
        risk = self.FIX_RISK_LEVELS.get(fix.fix_type, "medium")
        if risk == "low":
            return ApprovalResult(approved=True, reason="Auto-approved (low risk)")

        has_evidence = evidence_chain.has_evidence_for(fix.service_name)
        if risk == "medium":
            if has_evidence:
                return ApprovalResult(approved=True, reason="Approved (evidence provided)")
            return ApprovalResult(
                approved=False,
                reason="REJECTED: Insufficient investigation. Probe service first.",
                penalty=-0.08,
            )

        if not has_evidence:
            return ApprovalResult(
                approved=False,
                reason="REJECTED: No diagnostic evidence. Cannot approve blind fix.",
                penalty=-0.10,
            )
        if not escalated:
            return ApprovalResult(
                approved=False,
                reason="REJECTED: High-risk change requires escalation first.",
                penalty=-0.05,
            )
        return ApprovalResult(approved=True, reason="Approved (evidence + escalation)")


class TimelineEvent(BaseModel):
    """Expected incident timeline event."""

    timestamp_simulated: str
    service: str
    event_type: Literal["root_cause", "cascade", "symptom", "red_herring"]
    description: str
    caused_by: str | None = None


class TimelineReconstructor:
    """Grade postmortem timeline reconstruction quality."""

    def grade_timeline(
        self, postmortem: WritePostmortemAction, true_timeline: list[TimelineEvent]
    ) -> float:
        """Return deterministic timeline score."""
        if not true_timeline:
            return 0.0
        text = " ".join(postmortem.remediation_steps).lower()
        first_root = true_timeline[0]
        score = 0.0
        if first_root.service.lower() in text:
            score += 0.05
        ordered_mentions = sum(
            1 for event in true_timeline if event.service.lower() in text
        )
        if ordered_mentions >= max(1, len(true_timeline) // 2):
            score += 0.05
        if any(event.event_type == "red_herring" and event.service.lower() in text for event in true_timeline):
            score -= 0.03
        return round(max(-0.03, min(score, 0.10)), 4)


class SeverityTrigger(BaseModel):
    """Condition requiring severity reclassification."""

    condition: str
    discoverable_via: str
    step_discoverable_after: int
    new_severity: Severity
    reason: str


class SeverityReEvaluation:
    """Evaluate whether severity should be reclassified from evidence."""

    def __init__(self) -> None:
        self.severity_triggers: list[SeverityTrigger] = [
            SeverityTrigger(
                condition="enterprise_customer_affected",
                discoverable_via="fetch_user_data",
                step_discoverable_after=4,
                new_severity="critical",
                reason="Enterprise customer affected",
            ),
            SeverityTrigger(
                condition="staging_only_confirmed",
                discoverable_via="probe_service",
                step_discoverable_after=10,
                new_severity="low",
                reason="Staging-only scope",
            ),
            SeverityTrigger(
                condition="data_exposure_detected",
                discoverable_via="fetch_logs",
                step_discoverable_after=6,
                new_severity="critical",
                reason="Potential data exposure in logs",
            ),
        ]

    def check_reclassification(
        self,
        evidence_chain: EvidenceChain,
        current_step: int,
        current_severity: Severity,
    ) -> tuple[float, str, Severity]:
        """Return (reward, message, severity_after_check)."""
        target = self._triggered_severity(evidence_chain, current_step)
        if target is None:
            return 0.0, "No severity change required.", current_severity
        if target == current_severity:
            return -0.05, "Severity not updated despite trigger evidence.", current_severity
        return 0.05, "Severity reclassified based on new evidence.", target

    def _triggered_severity(
        self, evidence_chain: EvidenceChain, current_step: int
    ) -> Severity | None:
        for trigger in self.severity_triggers:
            if current_step < trigger.step_discoverable_after:
                continue
            if any(trigger.discoverable_via in entry.source for entry in evidence_chain.entries):
                return trigger.new_severity
        return None


class InvestigationGrader:
    """Grades investigation and remediation behavior deterministically."""

    def __init__(self) -> None:
        self._text_grader = DeterministicGrader()

    def grade_monitoring_check(
        self, action: CheckMonitoringAction, world: WorldState
    ) -> tuple[float, str, dict[str, float]]:
        """Grade monitoring checks on affected vs healthy services."""
        affected = _affected_services(world)
        if action.service_name is None:
            return 0.03, "Monitoring overview checked.", {"monitoring_check": 0.03}
        if action.service_name in affected:
            return 0.05, "Checked affected service.", {"monitoring_check": 0.05}
        return -0.02, "Checked healthy service.", {"monitoring_check": -0.02}

    def grade_probe(
        self, action: ProbeServiceAction, world: WorldState
    ) -> tuple[float, str, dict[str, float]]:
        """Grade probe relevance and probe type quality."""
        affected = _affected_services(world)
        if action.service_name not in affected:
            return -0.02, "Probe on healthy service.", {"probe": -0.02}
        useful = _recommended_probe_type(action.service_name)
        if action.check_type == useful:
            return 0.05, "Probe used best check type.", {"probe": 0.05}
        return 0.03, "Probe used non-optimal check type.", {"probe": 0.03}

    def grade_root_cause_discovery(
        self, known_facts: dict[str, object], world: WorldState
    ) -> float:
        """Return bonus when known facts include true root causes."""
        if not world.incident.root_causes:
            return 0.0
        discovered = str(known_facts).lower()
        matches = all(
            cause.service in discovered or cause.failure_mode in discovered
            for cause in world.incident.root_causes
        )
        return 0.15 if matches else 0.0

    def grade_red_herring_handling(
        self, action_history: list[str], world: WorldState
    ) -> float:
        """Reward dismissal and penalize acting on red herrings."""
        lowered = " ".join(action_history).lower()
        for red in world.incident.red_herrings:
            if red.service in lowered and "apply_fix" in lowered:
                return -0.05
        for red in world.incident.red_herrings:
            if red.service in lowered and "dismiss" in lowered:
                return 0.05
        return 0.0

    def grade_kb_cross_verification(
        self, kb_queried: bool, logs_checked: bool, world: WorldState
    ) -> float:
        """Reward cross-verification behavior after KB use."""
        if kb_queried and logs_checked:
            return 0.05
        has_outdated = any(not article.is_accurate for article in world.incident.kb_articles)
        if kb_queried and not logs_checked and has_outdated:
            return -0.05
        return 0.0

    def grade_fix(
        self, action: ApplyFixAction, world: WorldState
    ) -> tuple[float, str, dict[str, float]]:
        """Grade fix targeting accuracy."""
        root_by_service = {cause.service: cause.failure_mode for cause in world.incident.root_causes}
        if action.service_name in root_by_service:
            expected = FAILURE_FIX_MAP.get(root_by_service[action.service_name], "")
            if action.fix_type == expected:
                return 0.15, "Correct fix on root cause service.", {"fix": 0.15}
            return 0.05, "Partial fix: right service, wrong method.", {"fix": 0.05}
        return -0.10, "Wrong fix target.", {"fix": -0.10}

    def grade_verify(
        self, action_service: str, world: WorldState
    ) -> tuple[float, str, dict[str, float]]:
        """Grade fix verification outcomes."""
        health = world.service_mesh.get_health_summary().get(action_service, "unknown")
        if health == "healthy":
            return 0.08, "Verification confirms recovery.", {"verify": 0.08}
        return 0.03, "Verification attempted; service still unhealthy.", {"verify": 0.03}

    def grade_postmortem(
        self, action: WritePostmortemAction, world: WorldState
    ) -> tuple[float, str, dict[str, float]]:
        """Grade postmortem quality by deterministic keyword checks."""
        root_text = action.root_cause_description.lower()
        rem_text = " ".join(action.remediation_steps).lower()
        prev_text = " ".join(action.prevention_measures).lower()

        root_hit = any(cause.failure_mode in root_text or cause.service in root_text for cause in world.incident.root_causes)
        rem_hit = len(action.remediation_steps) > 0
        prev_hit = len(action.prevention_measures) > 0
        score = 0.04
        score += 0.03 if root_hit else 0.0
        score += 0.02 if rem_hit else 0.0
        score += 0.01 if prev_hit else 0.0
        if "red herring" in rem_text and world.incident.red_herrings:
            score += 0.0
        if "red herring" in prev_text and world.incident.red_herrings:
            score += 0.0
        return round(min(score, 0.10), 4), "Postmortem graded.", {"postmortem": round(min(score, 0.10), 4)}

    def grade_kb_update(
        self, article_title: str, content: str, world: WorldState
    ) -> tuple[float, str, dict[str, float]]:
        """Grade KB update correctness against root cause."""
        text = f"{article_title} {content}".lower()
        correct = any(cause.failure_mode in text or cause.service in text for cause in world.incident.root_causes)
        if correct:
            return 0.05, "KB update contains correct root-cause data.", {"kb_update": 0.05}
        return -0.05, "KB update appears incorrect.", {"kb_update": -0.05}

    def grade_runbook_decision(
        self,
        action: FollowRunbookStepAction,
        world: WorldState,
        runbook_correct: bool,
    ) -> tuple[float, str, dict[str, float]]:
        """Grade whether runbook following/deviation was appropriate."""
        if runbook_correct:
            return 0.05, "Followed correct runbook.", {"runbook": 0.05}
        return -0.08, "Followed incorrect/outdated runbook.", {"runbook": -0.08}

    def grade_runbook_deviation(self, runbook_correct: bool) -> tuple[float, str, dict[str, float]]:
        """Grade explicit deviation from runbook."""
        if runbook_correct:
            return -0.03, "Deviation from correct runbook.", {"runbook": -0.03}
        return 0.05, "Correctly deviated from bad runbook.", {"runbook": 0.05}

    def grade_incident_history_query(
        self, action: QueryIncidentHistoryAction, world: WorldState
    ) -> tuple[float, str, dict[str, float]]:
        """Grade usefulness of incident history lookup query."""
        query = action.query.lower()
        relevant = any(cause.service in query or cause.failure_mode in query for cause in world.incident.root_causes)
        if relevant:
            return 0.03, "Relevant historical incident query.", {"incident_history": 0.03}
        return -0.02, "History query returned low relevance.", {"incident_history": -0.02}

    def grade_flickering_detection(
        self, known_facts: dict[str, object], world: WorldState
    ) -> float:
        """Grade detection of intermittent flickering services."""
        flickering = {
            name
            for name, service in world.service_mesh.services.items()
            if service.flicker_pattern is not None
        }
        if not flickering:
            return 0.0
        facts = str(known_facts).lower()
        detected = any(name in facts and "flicker" in facts for name in flickering)
        return 0.08 if detected else -0.05

    @staticmethod
    def grade_tool_diminishing_returns(
        action_type: str,
        params_signature: str,
        seen_calls: set[str],
    ) -> float:
        """Return penalty for repeated same tool call with same parameters."""
        key = f"{action_type}:{params_signature}"
        if key in seen_calls:
            return -0.02
        seen_calls.add(key)
        return 0.0

    @staticmethod
    def grade_cost_penalty(total_incident_cost: float) -> float:
        """Return subtle cumulative cost penalty."""
        return round(-(total_incident_cost / 500.0) * 0.01, 4)

    @staticmethod
    def grade_alert_triage(investigated_actionable: bool, dismissed_noise: bool) -> float:
        """Grade alert triage decisions for actionable vs noise alerts."""
        if investigated_actionable:
            return 0.03
        if dismissed_noise:
            return 0.02
        if not investigated_actionable and not dismissed_noise:
            return -0.02
        return 0.0

    @staticmethod
    def grade_blast_radius(
        wrong_fix: bool, approved: bool, risk_level: RiskLevel
    ) -> float:
        """Grade blast radius impact for wrong approved fixes."""
        if not wrong_fix:
            return 0.0
        if not approved:
            return -0.08
        penalties: dict[RiskLevel, float] = {
            "low": -0.08,
            "medium": -0.10,
            "high": -0.12,
            "critical": -0.20,
        }
        return penalties[risk_level]


def _affected_services(world: WorldState) -> set[str]:
    return {
        name
        for name, state in world.service_mesh.services.items()
        if state.health in ("degraded", "flickering", "down")
    }


def _recommended_probe_type(service_name: str) -> str:
    recommendations = {
        "database": "resources",
        "auth": "config",
        "payments": "connections",
        "analytics": "logs",
        "notifications": "logs",
    }
    return recommendations.get(service_name, "logs")
