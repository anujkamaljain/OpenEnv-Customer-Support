"""OpenEnv Rubric API wrappers for EICC reward dimensions.

This module exposes the EICC reward signal as composable
:class:`openenv.core.rubrics.Rubric` instances so external tooling
(`Rubric.children()`, hooks, `state_dict`, dot-path lookup) works on
our incident environment.

Design principles
-----------------
- **Zero behavior drift**: the canonical per-step reward continues to
  be produced inside :mod:`env.environment`. These rubrics introspect
  the resulting ``info`` payload and re-derive each dimension. Running
  the rubric tree never changes the env's reward.
- **Composable**: scoring is built from small, single-purpose ``Rubric``
  subclasses combined via ``WeightedSum``, ``Sequential``, and ``Gate``
  exactly as recommended by RFC 004.
- **Inspectable**: ``EICCRewardRubric().named_rubrics()`` yields a flat
  view of every dimension so judges and reviewers can audit weights
  without reading reward dispatch code.

Each rubric accepts ``(action, observation)`` where ``observation`` is a
dict-like step payload that includes the ``info`` object emitted by
:meth:`CustomerSupportEnv.step`. ``observation`` may also be a
:class:`StepResult` or a Pydantic ``Observation`` — the wrappers tolerate
all three for ergonomic re-use from ``evaluate.py``.
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics import Rubric, Sequential, Gate, WeightedSum

from models.action import Action, ActionAdapter
from models.step_result import StepResult


# ---------------------------------------------------------------------------
# Helpers — extract action_type and info dict from heterogeneous inputs.
# ---------------------------------------------------------------------------


def _action_payload(action: Any) -> dict[str, Any]:
    """Return the action as a normalized dict regardless of input shape."""
    if isinstance(action, dict):
        return action
    if isinstance(action, Action):  # type: ignore[arg-type]
        return action.model_dump(exclude_none=True)
    try:
        parsed = ActionAdapter.validate_python(action)
        return parsed.model_dump(exclude_none=True)
    except Exception:
        return {}


def _action_type(action: Any) -> str:
    payload = _action_payload(action)
    value = payload.get("action_type", "")
    return str(value) if value is not None else ""


def _observation_dict(observation: Any) -> dict[str, Any]:
    """Coerce StepResult / Observation / dict into a flat dict."""
    if isinstance(observation, dict):
        return observation
    if isinstance(observation, StepResult):
        obs = observation.observation.model_dump()
        obs["info"] = dict(observation.info)
        obs["reward"] = observation.reward
        obs["done"] = observation.done
        return obs
    if hasattr(observation, "model_dump"):
        try:
            return observation.model_dump()  # type: ignore[no-any-return]
        except Exception:
            return {}
    return {}


def _info_dict(observation: Any) -> dict[str, Any]:
    obs = _observation_dict(observation)
    info = obs.get("info") if isinstance(obs.get("info"), dict) else {}
    return info if isinstance(info, dict) else {}


def _reward_breakdown(observation: Any) -> dict[str, float]:
    info = _info_dict(observation)
    rb = info.get("reward_breakdown")
    if not isinstance(rb, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in rb.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _step_reward(observation: Any) -> float:
    obs = _observation_dict(observation)
    reward = obs.get("reward", None)
    if isinstance(reward, (int, float)):
        return float(reward)
    rb = _reward_breakdown(observation)
    return float(rb.get("total", 0.0))


# ---------------------------------------------------------------------------
# Per-dimension rubrics. Each maps a single reward concern to ``[0, 1]``.
# ---------------------------------------------------------------------------


class JsonShapeRubric(Rubric):
    """1.0 if the action parses as a valid OpenEnv action, else 0.0."""

    def forward(self, action: Any, observation: Any) -> float:
        try:
            ActionAdapter.validate_python(_action_payload(action))
            return 1.0
        except Exception:
            return 0.0


class PhaseAvailabilityRubric(Rubric):
    """1.0 when ``action_type`` is in ``observation.available_actions``."""

    def forward(self, action: Any, observation: Any) -> float:
        atype = _action_type(action)
        obs = _observation_dict(observation)
        available = obs.get("available_actions") or []
        if not isinstance(available, list):
            return 0.0
        return 1.0 if atype in available else 0.0


class InvestigationBeforeActionRubric(Rubric):
    """Trajectory-aware: 1.0 if ``check_monitoring`` happened before any
    ``apply_fix`` in this episode (so far). Returns 1.0 if no fix was
    attempted yet (not applicable).
    """

    def __init__(self) -> None:
        super().__init__()
        # Use object.__setattr__ to avoid the parent's child-rubric
        # auto-registration trying to coerce a dict/list into Rubric.
        object.__setattr__(self, "_history", [])

    def reset(self) -> None:
        object.__setattr__(self, "_history", [])

    def forward(self, action: Any, observation: Any) -> float:
        atype = _action_type(action)
        history: list[str] = self._history  # type: ignore[assignment]
        history.append(atype)
        if atype == "apply_fix":
            return 1.0 if "check_monitoring" in history[:-1] else 0.0
        return 1.0


class KBCrossVerificationRubric(Rubric):
    """1.0 once the agent has both queried KB and probed/log-fetched at
    least one service in the same episode; tolerant before that.
    """

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_history", [])

    def reset(self) -> None:
        object.__setattr__(self, "_history", [])

    def forward(self, action: Any, observation: Any) -> float:
        history: list[str] = self._history  # type: ignore[assignment]
        history.append(_action_type(action))
        kb = "query_kb" in history
        verified = kb and ("probe_service" in history or "fetch_logs" in history)
        if not kb:
            return 1.0
        return 1.0 if verified else 0.0


class PolicyAwarenessRubric(Rubric):
    """1.0 if ``check_policy`` ran before any policy-sensitive action."""

    POLICY_SENSITIVE = frozenset(
        {"apply_fix", "escalate", "notify_stakeholders", "update_kb", "resolve"}
    )

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_history", [])
        object.__setattr__(self, "_policy_seen", False)

    def reset(self) -> None:
        object.__setattr__(self, "_history", [])
        object.__setattr__(self, "_policy_seen", False)

    def forward(self, action: Any, observation: Any) -> float:
        atype = _action_type(action)
        history: list[str] = self._history  # type: ignore[assignment]
        history.append(atype)
        if atype == "check_policy":
            object.__setattr__(self, "_policy_seen", True)
            return 1.0
        if atype in self.POLICY_SENSITIVE:
            return 1.0 if self._policy_seen else 0.0
        return 1.0


class RootCauseAccuracyRubric(Rubric):
    """1.0 if env reward breakdown contains a ``fix_correct`` signal."""

    def forward(self, action: Any, observation: Any) -> float:
        rb = _reward_breakdown(observation)
        return 1.0 if any("fix_correct" in key for key in rb.keys()) else 0.0


class BlastRadiusRubric(Rubric):
    """1.0 if no ``blast_radius`` penalty appears for this step, else 0.0.

    Wrong fixes record a negative ``blast_radius_*`` term in
    ``reward_breakdown``; their absence means the agent did not damage
    additional services on this step.
    """

    def forward(self, action: Any, observation: Any) -> float:
        rb = _reward_breakdown(observation)
        for key, value in rb.items():
            if "blast" in key and value < 0.0:
                return 0.0
        return 1.0


class ResourceBudgetRubric(Rubric):
    """1.0 if the step did not trigger a resource-budget exhaustion."""

    def forward(self, action: Any, observation: Any) -> float:
        rb = _reward_breakdown(observation)
        for key, value in rb.items():
            if "budget" in key and value < 0.0:
                return 0.0
        return 1.0


# ---------------------------------------------------------------------------
# Composite rubric — the EICC reward as an inspectable tree.
# ---------------------------------------------------------------------------


class IncidentRewardRubric(Rubric):
    """Composable EICC incident reward.

    Tree shape::

        IncidentRewardRubric (Sequential gate -> WeightedSum)
        ├── shape_gate      (Gate(JsonShape, threshold=1.0))
        ├── phase_gate      (Gate(PhaseAvailability, threshold=1.0))
        └── weighted_sum
            ├── investigation_before_action     (weight 0.20)
            ├── kb_cross_verification           (weight 0.15)
            ├── policy_awareness                (weight 0.15)
            ├── root_cause_accuracy             (weight 0.30)
            ├── blast_radius_safe               (weight 0.10)
            └── resource_budget_respected       (weight 0.10)

    Use ``rubric.reset()`` between episodes to clear trajectory state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.investigation_before_action = InvestigationBeforeActionRubric()
        self.kb_cross_verification = KBCrossVerificationRubric()
        self.policy_awareness = PolicyAwarenessRubric()
        self.root_cause_accuracy = RootCauseAccuracyRubric()
        self.blast_radius_safe = BlastRadiusRubric()
        self.resource_budget_respected = ResourceBudgetRubric()

        self.weighted_sum = WeightedSum(
            [
                self.investigation_before_action,
                self.kb_cross_verification,
                self.policy_awareness,
                self.root_cause_accuracy,
                self.blast_radius_safe,
                self.resource_budget_respected,
            ],
            weights=[0.20, 0.15, 0.15, 0.30, 0.10, 0.10],
        )

        self.shape_gate = Gate(JsonShapeRubric(), threshold=1.0)
        self.phase_gate = Gate(PhaseAvailabilityRubric(), threshold=1.0)

        # Sequential applies fail-fast gating: invalid JSON or wrong-phase
        # actions short-circuit to 0.0 before the weighted sum is computed.
        self.gated_reward = Sequential(
            self.shape_gate,
            self.phase_gate,
            self.weighted_sum,
        )

    def forward(self, action: Any, observation: Any) -> float:
        return self.gated_reward(action, observation)

    def reset(self) -> None:
        for child in self.rubrics():
            child.reset()


__all__ = [
    "JsonShapeRubric",
    "PhaseAvailabilityRubric",
    "InvestigationBeforeActionRubric",
    "KBCrossVerificationRubric",
    "PolicyAwarenessRubric",
    "RootCauseAccuracyRubric",
    "BlastRadiusRubric",
    "ResourceBudgetRubric",
    "IncidentRewardRubric",
]
