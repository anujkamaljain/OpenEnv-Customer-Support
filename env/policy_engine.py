"""Deterministic policy engine with step-based drift support."""

from __future__ import annotations

from pydantic import BaseModel, Field

JSONScalar = str | int | float | bool
JSONValue = JSONScalar | dict[str, JSONScalar]


class PolicyState(BaseModel):
    """Current policy state for one policy type."""

    policy_type: str
    rules: dict[str, JSONScalar] = Field(default_factory=dict)
    effective_since_step: int = 0
    version: int = 1


class PolicyChange(BaseModel):
    """Scheduled policy drift definition."""

    trigger_step: int = Field(ge=0)
    policy_type: str
    old_value: dict[str, JSONScalar] = Field(default_factory=dict)
    new_value: dict[str, JSONScalar] = Field(default_factory=dict)
    reason: str
    announced: bool = False


class PolicyResponse(BaseModel):
    """Agent-visible policy query response."""

    model_config = {"frozen": True}

    policy_type: str
    rules: dict[str, JSONScalar] = Field(default_factory=dict)
    effective_since_step: int
    version: int


class PolicyEngine:
    """Manages enterprise policies that can change mid-incident."""

    def __init__(
        self, initial_policies: dict[str, JSONValue], drift_schedule: list[PolicyChange]
    ) -> None:
        self._policies: dict[str, PolicyState] = _bootstrap_policies(initial_policies)
        self._drift_schedule: list[PolicyChange] = list(drift_schedule)
        self._applied_drifts: set[int] = set()

    def check_policy(self, policy_type: str) -> PolicyResponse:
        """Return the currently active policy for a type."""
        policy = self._policies.get(policy_type)
        if policy is None:
            return PolicyResponse(
                policy_type=policy_type,
                rules={},
                effective_since_step=0,
                version=0,
            )
        return PolicyResponse(
            policy_type=policy.policy_type,
            rules=dict(policy.rules),
            effective_since_step=policy.effective_since_step,
            version=policy.version,
        )

    def apply_scheduled_drifts(self, current_step: int) -> list[PolicyChange]:
        """Apply all drifts scheduled for current step."""
        applied: list[PolicyChange] = []
        for index, change in enumerate(self._drift_schedule):
            if change.trigger_step != current_step:
                continue
            if index in self._applied_drifts:
                continue
            self._apply_change(change, current_step)
            self._applied_drifts.add(index)
            applied.append(change)
        return applied

    def _apply_change(self, change: PolicyChange, current_step: int) -> None:
        current = self._policies.get(change.policy_type)
        current_version = current.version if current is not None else 0
        self._policies[change.policy_type] = PolicyState(
            policy_type=change.policy_type,
            rules=dict(change.new_value),
            effective_since_step=current_step,
            version=current_version + 1,
        )


def _bootstrap_policies(initial_policies: dict[str, JSONValue]) -> dict[str, PolicyState]:
    policies: dict[str, PolicyState] = {
        "refund": PolicyState(policy_type="refund", rules={"max_refund": 150}, version=1),
        "escalation": PolicyState(policy_type="escalation", rules={"required": True}, version=1),
        "sla": PolicyState(policy_type="sla", rules={"enterprise_steps": 4}, version=1),
        "compensation": PolicyState(policy_type="compensation", rules={"allow_credit": True}, version=1),
        "communication": PolicyState(policy_type="communication", rules={"tone": "empathetic"}, version=1),
    }
    for key, value in initial_policies.items():
        _apply_initial_key(policies, key, value)
    return policies


def _apply_initial_key(
    policies: dict[str, PolicyState], key: str, value: JSONValue
) -> None:
    if key == "refund_cap":
        policies["refund"].rules["max_refund"] = _to_scalar(value, default=150)
        return
    if key == "escalation_required":
        policies["escalation"].rules["required"] = _to_scalar(value, default=True)
        return
    if key == "sla_extension_steps":
        policies["sla"].rules["extension_steps"] = _to_scalar(value, default=0)
        return
    if isinstance(value, dict):
        scalar_rules = {rule_key: _to_scalar(rule_value, default="") for rule_key, rule_value in value.items()}
        policies[key] = PolicyState(policy_type=key, rules=scalar_rules, version=1)
        return
    policies[key] = PolicyState(policy_type=key, rules={"value": _to_scalar(value, default="")}, version=1)


def _to_scalar(value: JSONValue, default: JSONScalar) -> JSONScalar:
    if isinstance(value, (str, int, float, bool)):
        return value
    return default
