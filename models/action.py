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


Action = Annotated[
    Union[
        ClassifyAction,
        RouteAction,
        RespondAction,
        EscalateAction,
        ResolveAction,
        RequestInfoAction,
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
)
