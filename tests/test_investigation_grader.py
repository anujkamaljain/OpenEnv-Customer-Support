"""Tests for deterministic incident-mode grading utilities."""

from __future__ import annotations

import pytest

from env.world import WorldState
from graders.grader import DeterministicGrader
from graders.investigation_grader import (
    ACTION_COSTS,
    ChangeAdvisoryBoard,
    EvidenceChain,
    InvestigationGrader,
    SeverityReEvaluation,
    TimelineEvent,
    TimelineReconstructor,
)
from models.action import (
    ApplyFixAction,
    CheckMonitoringAction,
    FollowRunbookStepAction,
    ProbeServiceAction,
    QueryIncidentHistoryAction,
    WritePostmortemAction,
)
from tasks.incident_bank import IncidentBank


def _world(seed: int = 0, difficulty: str = "hard") -> WorldState:
    incident = IncidentBank().get_incident(seed=seed, difficulty=difficulty)
    return WorldState(seed=seed, incident=incident)


# =====================================================================
# Investigation grading
# =====================================================================


def test_grade_monitoring_check_rewards_affected_service() -> None:
    grader = InvestigationGrader()
    world = _world()
    score, _, parts = grader.grade_monitoring_check(
        CheckMonitoringAction(service_name="database"), world
    )
    assert score == pytest.approx(0.05)
    assert parts["monitoring_check"] == pytest.approx(0.05)


def test_grade_probe_rewards_best_check_type() -> None:
    grader = InvestigationGrader()
    world = _world()
    score, _, _ = grader.grade_probe(
        ProbeServiceAction(service_name="database", check_type="resources"), world
    )
    assert score == pytest.approx(0.05)


def test_grade_fix_distinguishes_correct_vs_wrong() -> None:
    grader = InvestigationGrader()
    world = _world(seed=0, difficulty="easy")

    ok_score, _, _ = grader.grade_fix(
        ApplyFixAction(service_name="auth", fix_type="config_change"), world
    )
    bad_score, _, _ = grader.grade_fix(
        ApplyFixAction(service_name="analytics", fix_type="restart_service"), world
    )
    assert ok_score == pytest.approx(0.15)
    assert bad_score == pytest.approx(-0.10)


def test_grade_kb_cross_verification_penalizes_blind_trust() -> None:
    grader = InvestigationGrader()
    world = _world(seed=0, difficulty="hard")
    score = grader.grade_kb_cross_verification(
        kb_queried=True, logs_checked=False, world=world
    )
    assert score == pytest.approx(-0.05)


def test_grade_runbook_decision_for_wrong_runbook() -> None:
    grader = InvestigationGrader()
    world = _world()
    score, _, _ = grader.grade_runbook_decision(
        FollowRunbookStepAction(runbook_id="RB-002", step_index=0),
        world,
        runbook_correct=False,
    )
    assert score == pytest.approx(-0.08)


def test_grade_incident_history_query() -> None:
    grader = InvestigationGrader()
    world = _world()
    score, _, _ = grader.grade_incident_history_query(
        QueryIncidentHistoryAction(query="database oom", service_filter="database"),
        world,
    )
    assert score == pytest.approx(0.03)


def test_grade_postmortem_and_timeline() -> None:
    grader = InvestigationGrader()
    world = _world(seed=0, difficulty="easy")
    postmortem = WritePostmortemAction(
        summary="Incident summary",
        root_cause_description="auth token_expiry triggered failures",
        remediation_steps=["auth recovered", "payments stabilized"],
        prevention_measures=["add monitoring"],
    )
    score, _, _ = grader.grade_postmortem(postmortem, world)
    assert 0.0 <= score <= 0.10

    reconstructor = TimelineReconstructor()
    timeline_score = reconstructor.grade_timeline(
        postmortem,
        [
            TimelineEvent(
                timestamp_simulated="02:47",
                service="auth",
                event_type="root_cause",
                description="Auth token cache issue",
            ),
            TimelineEvent(
                timestamp_simulated="02:50",
                service="payments",
                event_type="cascade",
                description="Payment failures",
            ),
        ],
    )
    assert -0.03 <= timeline_score <= 0.10


# =====================================================================
# Advanced feature helpers
# =====================================================================


def test_evidence_chain_and_cab_review() -> None:
    chain = EvidenceChain()
    chain.add_evidence(
        step=1,
        source="probe_service",
        finding="database memory at 99%",
        conclusion="database oom likely root cause",
        service="database",
    )
    coherence = chain.grade_chain_coherence("oom")
    assert coherence > 0.0
    assert chain.has_evidence_for("database") is True

    cab = ChangeAdvisoryBoard()
    approved = cab.review_fix(
        ApplyFixAction(service_name="database", fix_type="memory_increase"),
        chain,
        escalated=False,
    )
    assert approved.approved is True

    rejected = cab.review_fix(
        ApplyFixAction(service_name="database", fix_type="schema_migration"),
        EvidenceChain(),
        escalated=False,
    )
    assert rejected.approved is False
    assert rejected.penalty == pytest.approx(-0.10)


def test_severity_re_evaluation() -> None:
    evaluator = SeverityReEvaluation()
    chain = EvidenceChain()
    chain.add_evidence(
        step=5,
        source="fetch_user_data",
        finding="enterprise customer impacted",
        conclusion="severity should be critical",
        service="payments",
    )
    reward, _, new_severity = evaluator.check_reclassification(
        evidence_chain=chain,
        current_step=5,
        current_severity="high",
    )
    assert reward == pytest.approx(0.05)
    assert new_severity == "critical"


def test_cost_dict_and_episode_grade() -> None:
    assert ACTION_COSTS["apply_fix"] == 200
    assert ACTION_COSTS["escalate"] == 500

    score = DeterministicGrader.grade_incident_episode(
        root_cause_identified=True,
        fix_effective=True,
        customer_scores=[0.8, 1.0],
        tool_efficiency=0.9,
        sla_compliance_rate=0.75,
        stakeholder_satisfaction=0.8,
        policy_compliance_rate=1.0,
        postmortem_quality=0.7,
        kb_contribution_quality=0.6,
    )
    assert 0.0 <= score <= 1.0


# =====================================================================
# Extended grading coverage
# =====================================================================


def test_evidence_chain_empty_coherence_is_zero() -> None:
    chain = EvidenceChain()
    assert chain.grade_chain_coherence("anything") == pytest.approx(0.0)


def test_evidence_chain_has_evidence_for_missing_service() -> None:
    chain = EvidenceChain()
    assert chain.has_evidence_for("database") is False


def test_cab_rejects_high_risk_without_escalation() -> None:
    chain = EvidenceChain()
    chain.add_evidence(step=1, source="probe", finding="x", conclusion="y", service="database")
    cab = ChangeAdvisoryBoard()
    result = cab.review_fix(
        ApplyFixAction(service_name="database", fix_type="schema_migration"),
        chain,
        escalated=False,
    )
    assert result.approved is False


def test_cab_approves_low_risk_without_evidence() -> None:
    cab = ChangeAdvisoryBoard()
    result = cab.review_fix(
        ApplyFixAction(service_name="database", fix_type="restart_service"),
        EvidenceChain(),
        escalated=False,
    )
    assert result.approved is True


def test_severity_re_eval_no_change_when_no_evidence() -> None:
    evaluator = SeverityReEvaluation()
    chain = EvidenceChain()
    reward, _, target = evaluator.check_reclassification(
        evidence_chain=chain,
        current_step=0,
        current_severity="medium",
    )
    assert reward == pytest.approx(0.0)
    assert target == "medium"


def test_grade_kb_cross_verification_rewards_verification() -> None:
    grader = InvestigationGrader()
    world = _world(seed=0, difficulty="hard")
    score = grader.grade_kb_cross_verification(
        kb_queried=True, logs_checked=True, world=world
    )
    assert score >= 0.0


def test_grade_monitoring_check_unknown_service() -> None:
    grader = InvestigationGrader()
    world = _world()
    score, _, _ = grader.grade_monitoring_check(
        CheckMonitoringAction(service_name=None), world
    )
    assert score >= 0.0


def test_grade_fix_wrong_service_penalized() -> None:
    grader = InvestigationGrader()
    world = _world(seed=0, difficulty="easy")
    score, _, _ = grader.grade_fix(
        ApplyFixAction(service_name="notifications", fix_type="restart_service"), world
    )
    assert score < 0


def test_grade_postmortem_empty_fields() -> None:
    grader = InvestigationGrader()
    world = _world(seed=0, difficulty="easy")
    postmortem = WritePostmortemAction(
        summary="short",
        root_cause_description="unknown",
    )
    score, _, _ = grader.grade_postmortem(postmortem, world)
    assert score >= 0.0


def test_grade_runbook_decision_correct_runbook_positive() -> None:
    grader = InvestigationGrader()
    world = _world()
    score, _, _ = grader.grade_runbook_decision(
        FollowRunbookStepAction(runbook_id="RB-001", step_index=0),
        world,
        runbook_correct=True,
    )
    assert score == pytest.approx(0.05)


def test_grade_incident_episode_all_zeros() -> None:
    score = DeterministicGrader.grade_incident_episode(
        root_cause_identified=False,
        fix_effective=False,
        customer_scores=[],
        tool_efficiency=0.0,
        sla_compliance_rate=0.0,
        stakeholder_satisfaction=0.0,
        policy_compliance_rate=0.0,
        postmortem_quality=0.0,
        kb_contribution_quality=0.0,
    )
    assert score == pytest.approx(0.0)


def test_grade_incident_episode_perfect() -> None:
    score = DeterministicGrader.grade_incident_episode(
        root_cause_identified=True,
        fix_effective=True,
        customer_scores=[1.0, 1.0, 1.0],
        tool_efficiency=1.0,
        sla_compliance_rate=1.0,
        stakeholder_satisfaction=1.0,
        policy_compliance_rate=1.0,
        postmortem_quality=1.0,
        kb_contribution_quality=1.0,
    )
    assert score == pytest.approx(1.0)


def test_timeline_reconstructor_empty() -> None:
    reconstructor = TimelineReconstructor()
    postmortem = WritePostmortemAction(
        summary="nothing happened",
        root_cause_description="no root cause",
    )
    score = reconstructor.grade_timeline(postmortem, [])
    assert -0.05 <= score <= 0.10


def test_action_costs_all_positive() -> None:
    for action_type, cost in ACTION_COSTS.items():
        assert cost >= 0, f"{action_type} has negative cost"
