"""Action models using a discriminated union keyed on ``action_type``."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field, TypeAdapter


class ClassifyAction(BaseModel):
    """Classify the ticket's category and priority."""

    action_type: Literal["classify"] = "classify"
    category: Literal[
        "billing",
        "bug_report",
        "feature_request",
        "account_access",
        "general_inquiry",
        "cancellation",
    ]
    priority: Literal["low", "medium", "high", "critical"]


class RouteAction(BaseModel):
    """Route the ticket to a department."""

    action_type: Literal["route"] = "route"
    department: Literal["billing", "technical", "account", "general"]


class RespondAction(BaseModel):
    """Send a response to the customer."""

    action_type: Literal["respond"] = "respond"
    response_text: str = Field(min_length=1, max_length=2000)
    tone: Literal["formal", "empathetic", "concise"] = "formal"


class EscalateAction(BaseModel):
    """Escalate the ticket to a specialized team."""

    action_type: Literal["escalate"] = "escalate"
    reason: str = Field(min_length=1, max_length=500)
    target_team: Literal["l2_support", "engineering", "management"]


class ResolveAction(BaseModel):
    """Resolve and close the ticket."""

    action_type: Literal["resolve"] = "resolve"
    resolution_summary: str = Field(min_length=1, max_length=2000)
    offered_compensation: float | None = None


class RequestInfoAction(BaseModel):
    """Request additional information from the customer."""

    action_type: Literal["request_info"] = "request_info"
    question_to_customer: str = Field(min_length=1, max_length=1000)


# === Investigation actions ===


class CheckMonitoringAction(BaseModel):
    """Query service health from monitoring."""

    action_type: Literal["check_monitoring"] = "check_monitoring"
    service_name: str | None = None


class ProbeServiceAction(BaseModel):
    """Run deeper diagnostics for one service."""

    action_type: Literal["probe_service"] = "probe_service"
    service_name: str
    check_type: Literal["logs", "resources", "connections", "config"]


class FetchLogsAction(BaseModel):
    """Fetch logs for a service and time range."""

    action_type: Literal["fetch_logs"] = "fetch_logs"
    service_name: str
    time_range: Literal["last_5m", "last_15m", "last_1h"] = "last_15m"


# === Enterprise tool actions ===


class FetchUserDataAction(BaseModel):
    """Retrieve customer record from CRM."""

    action_type: Literal["fetch_user_data"] = "fetch_user_data"
    customer_id: str


class CheckBillingAction(BaseModel):
    """Retrieve customer billing state."""

    action_type: Literal["check_billing"] = "check_billing"
    customer_id: str


class QueryKBAction(BaseModel):
    """Search knowledge base articles."""

    action_type: Literal["query_kb"] = "query_kb"
    query: str


class CheckPolicyAction(BaseModel):
    """Read currently active policy values."""

    action_type: Literal["check_policy"] = "check_policy"
    policy_type: Literal["refund", "escalation", "sla", "compensation", "communication"]


class QueryIncidentHistoryAction(BaseModel):
    """Search historical incidents for similar patterns."""

    action_type: Literal["query_incident_history"] = "query_incident_history"
    query: str
    service_filter: str | None = None


# === Remediation actions ===


class ApplyFixAction(BaseModel):
    """Attempt a service-level remediation."""

    action_type: Literal["apply_fix"] = "apply_fix"
    service_name: str
    fix_type: str


class VerifyFixAction(BaseModel):
    """Verify service health after remediation."""

    action_type: Literal["verify_fix"] = "verify_fix"
    service_name: str


class RollbackFixAction(BaseModel):
    """Rollback a previously attempted fix."""

    action_type: Literal["rollback_fix"] = "rollback_fix"
    service_name: str


class NotifyStakeholdersAction(BaseModel):
    """Send incident update to stakeholders."""

    action_type: Literal["notify_stakeholders"] = "notify_stakeholders"
    stakeholder: Literal["vp_engineering", "legal", "support_lead", "all"]
    message: str = Field(min_length=1, max_length=2000)
    urgency: Literal["info", "warning", "critical"]


class FollowRunbookStepAction(BaseModel):
    """Execute one runbook step."""

    action_type: Literal["follow_runbook_step"] = "follow_runbook_step"
    runbook_id: str
    step_index: int


class WritePostmortemAction(BaseModel):
    """Write incident postmortem summary."""

    action_type: Literal["write_postmortem"] = "write_postmortem"
    summary: str = Field(min_length=1, max_length=3000)
    root_cause_description: str = Field(min_length=1, max_length=2000)
    remediation_steps: list[str] = Field(default_factory=list)
    prevention_measures: list[str] = Field(default_factory=list)


class UpdateKBAction(BaseModel):
    """Add or update KB content."""

    action_type: Literal["update_kb"] = "update_kb"
    article_title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=list)


Action = Annotated[
    Union[
        ClassifyAction,
        RouteAction,
        RespondAction,
        EscalateAction,
        ResolveAction,
        RequestInfoAction,
        CheckMonitoringAction,
        ProbeServiceAction,
        FetchLogsAction,
        FetchUserDataAction,
        CheckBillingAction,
        QueryKBAction,
        CheckPolicyAction,
        QueryIncidentHistoryAction,
        ApplyFixAction,
        VerifyFixAction,
        RollbackFixAction,
        NotifyStakeholdersAction,
        FollowRunbookStepAction,
        WritePostmortemAction,
        UpdateKBAction,
    ],
    Discriminator("action_type"),
]

ActionAdapter: TypeAdapter[Action] = TypeAdapter(Action)  # type: ignore[arg-type]

ACTION_CLASSES = (
    ClassifyAction,
    RouteAction,
    RespondAction,
    EscalateAction,
    ResolveAction,
    RequestInfoAction,
    CheckMonitoringAction,
    ProbeServiceAction,
    FetchLogsAction,
    FetchUserDataAction,
    CheckBillingAction,
    QueryKBAction,
    CheckPolicyAction,
    QueryIncidentHistoryAction,
    ApplyFixAction,
    VerifyFixAction,
    RollbackFixAction,
    NotifyStakeholdersAction,
    FollowRunbookStepAction,
    WritePostmortemAction,
    UpdateKBAction,
)
