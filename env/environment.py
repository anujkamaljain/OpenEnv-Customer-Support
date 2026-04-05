"""Async customer support environment conforming to the OpenEnv contract.

v2 — multi-objective rewards, SLA deadlines, business-impact multipliers,
weighted keyword grading, and significantly harder tasks.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from env.errors import EnvironmentDoneError, EnvironmentNotResetError
from env.state import PHASE_VALID_ACTIONS, InternalState
from graders.grader import DeterministicGrader
from models.action import (
    ACTION_CLASSES,
    Action,
    ActionAdapter,
    ClassifyAction,
    EscalateAction,
    RequestInfoAction,
    RespondAction,
    ResolveAction,
    RouteAction,
)
from models.step_result import StepResult
from tasks.ticket_bank import TicketBank

_ConcreteAction = (
    ClassifyAction
    | RouteAction
    | RespondAction
    | EscalateAction
    | ResolveAction
    | RequestInfoAction
)


class CustomerSupportEnv:
    """Production-grade async environment for customer support triage.

    Usage::

        env  = CustomerSupportEnv()
        res  = await env.reset(seed=0, difficulty="easy")
        while not res.done:
            res = await env.step(agent.act(res.observation))
        await env.close()
    """

    def __init__(self, ticket_bank: TicketBank | None = None) -> None:
        self._bank = ticket_bank or TicketBank()
        self._state: InternalState | None = None
        self._grader = DeterministicGrader()

    # ==================================================================
    # Public async API
    # ==================================================================

    async def reset(
        self,
        seed: int = 0,
        difficulty: str | None = None,
    ) -> StepResult:
        """Start a new episode with a deterministically selected ticket."""
        ticket = self._bank.get_ticket(seed=seed, difficulty=difficulty)
        self._state = InternalState(ticket)
        return StepResult(
            observation=self._state.to_observation(),
            reward=0.0,
            done=False,
            info=self._state.to_info(),
        )

    async def step(self, action: dict[str, Any] | Action) -> StepResult:  # type: ignore[type-arg]
        """Apply *action*, return ``(observation, reward, done, info)``."""
        state = self._require_active_state()

        # --- parse -------------------------------------------------------
        parsed = self._safe_parse(action)
        if parsed is None:
            return self._penalty(state, "invalid_parse", "Action failed schema validation.")

        # --- repeat detection --------------------------------------------
        action_json = parsed.model_dump_json(exclude_none=True)
        repeat_pen = -0.05 if action_json == state.last_action_json else 0.0
        state.last_action_json = action_json

        # --- phase gate --------------------------------------------------
        action_type: str = parsed.action_type  # type: ignore[union-attr]
        valid = PHASE_VALID_ACTIONS[state.phase]
        if action_type not in valid:
            return self._penalty(
                state,
                action_type,
                f"Action '{action_type}' not valid in phase '{state.phase}'. "
                f"Valid: {sorted(valid)}",
            )

        # --- dispatch & reward -------------------------------------------
        reward, feedback = self._dispatch(state, parsed)
        reward += repeat_pen

        # --- SLA penalty -------------------------------------------------
        sla_pen = self._grader.sla_penalty(state.steps_taken, state.sla_steps)
        reward += sla_pen

        parts = [feedback]
        if repeat_pen < 0:
            parts.append(f"Repeat penalty: {repeat_pen:+.2f}.")
        if sla_pen < 0:
            state.urgency_penalty_accrued += abs(sla_pen)
            over = state.steps_taken - state.sla_steps + 1
            parts.append(f"SLA exceeded ({over} over): {sla_pen:+.2f}.")
        feedback = " ".join(parts)

        # --- clamp -------------------------------------------------------
        reward = max(-0.25, min(reward, 0.30))

        state.record_action(action_type, feedback, reward)

        if action_type == "resolve":
            state.phase = "resolved"
            state.done = True
            self._finalize_escalation_score(state)

        return StepResult(
            observation=state.to_observation(),
            reward=round(reward, 4),
            done=state.done,
            info=state.to_info(),
        )

    async def state(self) -> StepResult | None:
        """Return the current observation without advancing the episode."""
        if self._state is None:
            return None
        return StepResult(
            observation=self._state.to_observation(),
            reward=0.0,
            done=self._state.done,
            info=self._state.to_info(),
        )

    async def close(self) -> None:
        """Release resources and reset internal state."""
        self._state = None

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _require_active_state(self) -> InternalState:
        if self._state is None:
            raise EnvironmentNotResetError("Call reset() before step().")
        if self._state.done:
            raise EnvironmentDoneError("Episode ended. Call reset() for a new one.")
        return self._state

    def _safe_parse(self, action: dict[str, Any] | _ConcreteAction) -> _ConcreteAction | None:  # type: ignore[type-arg]
        if isinstance(action, dict):
            try:
                return ActionAdapter.validate_python(action)  # type: ignore[return-value]
            except (ValidationError, ValueError):
                return None
        if isinstance(action, ACTION_CLASSES):
            return action
        return None

    def _penalty(
        self, state: InternalState, label: str, feedback: str
    ) -> StepResult:
        reward = -0.05
        state.record_action(label, feedback, reward)
        return StepResult(
            observation=state.to_observation(),
            reward=reward,
            done=state.done,
            info=state.to_info(),
        )

    @staticmethod
    def _finalize_escalation_score(state: InternalState) -> None:
        """Set escalation_score when it was never explicitly determined."""
        if state.escalation_score is not None:
            return
        if state.ticket.requires_escalation:
            state.escalation_score = 0.0  # missed required escalation
        else:
            state.escalation_score = 1.0  # correctly never escalated

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, state: InternalState, action: _ConcreteAction) -> tuple[float, str]:
        if isinstance(action, ClassifyAction):
            return self._on_classify(state, action)
        if isinstance(action, RouteAction):
            return self._on_route(state, action)
        if isinstance(action, RespondAction):
            return self._on_respond(state, action)
        if isinstance(action, EscalateAction):
            return self._on_escalate(state, action)
        if isinstance(action, ResolveAction):
            return self._on_resolve(state, action)
        if isinstance(action, RequestInfoAction):
            return self._on_request_info(state, action)
        return -0.05, "Unrecognised action type."

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_classify(
        self, state: InternalState, action: ClassifyAction
    ) -> tuple[float, str]:
        ticket = state.ticket
        cat_ok = action.category == ticket.gold_category
        pri_ok = action.priority == ticket.gold_priority
        state.classification_correct = cat_ok and pri_ok
        state.phase = "classified"

        if cat_ok and pri_ok:
            reward, feedback = 0.10, "Correct classification."
        elif cat_ok:
            reward = 0.06
            feedback = f"Category correct; expected priority '{ticket.gold_priority}'."
        elif pri_ok:
            reward = 0.04
            feedback = f"Priority correct; expected category '{ticket.gold_category}'."
        else:
            reward = 0.01
            feedback = (
                f"Incorrect. Expected '{ticket.gold_category}' / '{ticket.gold_priority}'."
            )

        if pri_ok and ticket.gold_priority in ("critical", "high"):
            reward += 0.10
            state.urgency_handled = True
            feedback += " Urgency correctly identified (+0.10)."

        return reward, feedback

    def _on_route(
        self, state: InternalState, action: RouteAction
    ) -> tuple[float, str]:
        correct = action.department == state.ticket.gold_department
        state.routing_correct = correct
        state.phase = "routed"
        if correct:
            return 0.10, "Routed to the correct department."
        return 0.01, f"Incorrect routing; expected '{state.ticket.gold_department}'."

    def _on_respond(
        self, state: InternalState, action: RespondAction
    ) -> tuple[float, str]:
        ticket = state.ticket
        state.phase = "responding"

        quality, forbidden_pen = self._grader.weighted_keyword_score(
            action.response_text, ticket.response_spec
        )
        base_reward = round(quality * 0.20, 4)

        forbidden_pen *= ticket.penalty_multiplier
        tone_pen = self._eval_tone_constraint(state, action.tone)
        tone_pen *= ticket.penalty_multiplier

        reward = base_reward + forbidden_pen + tone_pen
        state.response_quality_score = quality

        parts = [f"Response quality: {quality:.0%}."]
        if forbidden_pen < 0:
            parts.append(f"Forbidden pattern penalty: {forbidden_pen:+.3f}.")
        if tone_pen < 0:
            parts.append("Tone constraint violated.")
        return round(reward, 4), " ".join(parts)

    def _on_escalate(
        self, state: InternalState, action: EscalateAction
    ) -> tuple[float, str]:
        ticket = state.ticket
        state.phase = "escalated"

        if not ticket.requires_escalation:
            pen = round(-0.10 * ticket.penalty_multiplier, 4)
            state.escalation_score = 0.0
            return pen, "Unnecessary escalation; ticket does not require it."

        team_ok = (
            ticket.escalation_target is not None
            and action.target_team == ticket.escalation_target
        )
        if team_ok:
            state.escalation_score = 1.0
            return 0.15, "Correctly escalated to the right team."

        state.escalation_score = 0.3
        return 0.05, (
            f"Escalation needed but target should be '{ticket.escalation_target}'."
        )

    def _on_resolve(
        self, state: InternalState, action: ResolveAction
    ) -> tuple[float, str]:
        ticket = state.ticket

        quality, forbidden_pen = self._grader.weighted_keyword_score(
            action.resolution_summary, ticket.resolution_spec
        )
        comp_score = self._grader.compensation_accuracy(
            action.offered_compensation, ticket.compensation_range
        )
        constraint_pen = self._eval_resolve_constraints(state, action)

        # quality(0..1) * 0.15 + comp(0..1) * 0.05 + base 0.05 → max 0.25
        raw = quality * 0.15 + comp_score * 0.05 + 0.05
        penalties = (forbidden_pen + constraint_pen) * ticket.penalty_multiplier

        reward = round(max(0.0, min(raw + penalties, 0.25)), 4)
        state.resolution_quality_score = quality

        parts = [f"Resolution quality: {quality:.0%}."]
        if forbidden_pen < 0:
            parts.append(f"Forbidden pattern penalty: {forbidden_pen:+.3f}.")
        if constraint_pen < 0:
            parts.append(f"Constraint penalty: {constraint_pen:+.3f}.")
        if comp_score < 1.0 and ticket.compensation_range is not None:
            parts.append("Compensation outside optimal range.")
        return reward, " ".join(parts)

    def _on_request_info(
        self, state: InternalState, action: RequestInfoAction
    ) -> tuple[float, str]:
        ticket = state.ticket

        if ticket.partial_info and not state.info_requested:
            state.info_requested = True
            reveal = ticket.info_reveals or "No additional details available."
            return 0.05, f"Customer clarification: {reveal}"

        pen = round(-0.03 * ticket.penalty_multiplier, 4)
        if state.info_requested:
            return pen, "No new information. Customer already provided clarification."
        return pen, "Information requested. No additional details available (simulated)."

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eval_tone_constraint(state: InternalState, tone: str) -> float:
        """Return raw penalty (caller applies business-impact multiplier)."""
        for c in state.ticket.constraints:
            cl = c.lower()
            if "empathetic" in cl and tone != "empathetic":
                state.constraints_violated.append(c)
                return -0.05
            if "formal" in cl and tone != "formal":
                state.constraints_violated.append(c)
                return -0.05
        return 0.0

    def _eval_resolve_constraints(
        self, state: InternalState, action: ResolveAction
    ) -> float:
        """Return raw penalty (caller applies business-impact multiplier)."""
        ticket = state.ticket
        penalty = 0.0

        for constraint in ticket.constraints:
            cl = constraint.lower()

            if "refund" in cl and action.offered_compensation is not None:
                if self._grader.check_refund_constraint(
                    constraint, action.offered_compensation
                ):
                    state.constraints_violated.append(constraint)
                    penalty -= 0.05

            if "escalat" in cl and ticket.requires_escalation:
                if state.phase != "escalated":
                    already = any(
                        "escalat" in v.lower() for v in state.constraints_violated
                    )
                    if not already:
                        state.constraints_violated.append(constraint)
                        penalty -= 0.05

        return penalty
