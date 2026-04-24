"""Async customer support environment conforming to the OpenEnv contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from env.billing import BillingRecord, BillingSystem, FailedPayment, Invoice
from env.crm import CRMSystem
from env.customers import CustomerQueueManager
from env.errors import EnvironmentDoneError, EnvironmentNotResetError
from env.incident_history import IncidentHistoryStore
from env.knowledge_base import KBArticle, PersistentKnowledgeBase
from env.policy_engine import PolicyChange as EnginePolicyChange
from env.policy_engine import PolicyEngine
from env.runbooks import RunbookEngine
from env.state import INCIDENT_PHASE_VALID_ACTIONS, PHASE_VALID_ACTIONS, IncidentState, InternalState
from env.stakeholders import StakeholderManager
from env.world import WorldState
from graders.grader import DeterministicGrader
from graders.investigation_grader import ChangeAdvisoryBoard
from models.action import ACTION_CLASSES
from models.action import (
    Action,
    ActionAdapter,
    ApplyFixAction,
    CheckBillingAction,
    CheckMonitoringAction,
    CheckPolicyAction,
    ClassifyAction,
    EscalateAction,
    FetchLogsAction,
    FetchUserDataAction,
    FollowRunbookStepAction,
    NotifyStakeholdersAction,
    ProbeServiceAction,
    QueryIncidentHistoryAction,
    QueryKBAction,
    RequestInfoAction,
    RollbackFixAction,
    RespondAction,
    ResolveAction,
    RouteAction,
    UpdateKBAction,
    VerifyFixAction,
    WritePostmortemAction,
)
from models.incident import IncidentScenario
from models.step_result import StepResult
from tasks.incident_bank import IncidentBank
from tasks.ticket_bank import TicketBank

_TicketAction = (
    ClassifyAction
    | RouteAction
    | RespondAction
    | EscalateAction
    | ResolveAction
    | RequestInfoAction
)

_IncidentPhase = Literal["triage", "investigation", "response", "resolution"]

# Quality multiplier applied to keyword scores when the agent skips
# request_info on a partial-info ticket.  Creates a speed-vs-quality
# trade-off: skipping saves a step (less SLA risk) but caps quality.
_INFO_SKIP_QUALITY_FACTOR = 0.6


class _IncidentRuntime:
    """Incident mode runtime state for phase-3 action dispatch."""

    __slots__ = (
        "seed",
        "incident",
        "world",
        "crm",
        "billing",
        "policy_engine",
        "history_store",
        "runbook_engine",
        "stakeholder_mgr",
        "customer_queue_mgr",
        "persistent_kb",
        "knowledge_base",
        "incident_phase",
        "steps_taken",
        "max_steps",
        "done",
        "known_facts",
        "active_policies",
        "tool_results",
        "active_alerts",
        "suggested_runbook",
        "actions_log",
        "cumulative_reward",
        "last_action_json",
        "last_reward_breakdown",
        "severity",
    )

    def __init__(
        self,
        *,
        seed: int,
        incident: IncidentScenario,
        world: WorldState,
        crm: CRMSystem,
        billing: BillingSystem,
        policy_engine: PolicyEngine,
        history_store: IncidentHistoryStore,
        runbook_engine: RunbookEngine,
        stakeholder_mgr: StakeholderManager,
        customer_queue_mgr: CustomerQueueManager,
        persistent_kb: PersistentKnowledgeBase,
    ) -> None:
        self.seed = seed
        self.incident = incident
        self.world = world
        self.crm = crm
        self.billing = billing
        self.policy_engine = policy_engine
        self.history_store = history_store
        self.runbook_engine = runbook_engine
        self.stakeholder_mgr = stakeholder_mgr
        self.customer_queue_mgr = customer_queue_mgr
        self.persistent_kb = persistent_kb
        self.knowledge_base = persistent_kb.reset_for_episode(incident)
        self.incident_phase: _IncidentPhase = "triage"
        self.steps_taken = 0
        self.max_steps = incident.max_steps
        self.done = False
        self.known_facts: dict[str, Any] = {}
        self.active_policies: dict[str, Any] = {}
        self.tool_results: dict[str, Any] | None = None
        self.active_alerts: list[str] = []
        self.suggested_runbook = _suggested_runbook(runbook_engine, incident)
        self.actions_log = []
        self.cumulative_reward = 0.0
        self.last_action_json: str | None = None
        self.last_reward_breakdown: dict[str, float] = {}
        self.severity: str | None = None

    @property
    def available_actions(self) -> list[str]:
        return sorted(INCIDENT_PHASE_VALID_ACTIONS[self.incident_phase])

    def record_action(self, action_type: str, feedback: str, reward: float) -> None:
        from models.observation import ActionRecord

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
        if self.steps_taken >= self.max_steps:
            self.done = True

    def to_observation(self):
        from models.observation import Observation

        ticket = self.world.support_queue[0] if self.world.support_queue else None
        ticket_text = (
            getattr(ticket, "body", None) or getattr(ticket, "ticket_text", None) or ""
        ) if ticket is not None else self.incident.description
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
            total_incident_cost=self.world.total_downtime_cost,
        )

    def to_info(self) -> dict[str, Any]:
        return {
            "mode": "incident",
            "incident_id": self.incident.incident_id,
            "incident_phase": self.incident_phase,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "known_facts": dict(self.known_facts),
            "active_policies": dict(self.active_policies),
            "total_incident_cost": self.world.total_downtime_cost,
            "reward_breakdown": dict(self.last_reward_breakdown),
        }


class CustomerSupportEnv:
    """Production-grade async environment for customer support triage.

    Usage::

        env  = CustomerSupportEnv()
        res  = await env.reset(seed=0, difficulty="easy")
        while not res.done:
            res = await env.step(agent.act(res.observation))
        await env.close()
    """

    def __init__(
        self,
        ticket_bank: TicketBank | None = None,
        incident_bank: IncidentBank | None = None,
    ) -> None:
        self._bank = ticket_bank or TicketBank()
        self._incident_bank = incident_bank or IncidentBank()
        self._state: InternalState | None = None
        self._incident_state: IncidentState | None = None
        self._mode: Literal["ticket", "incident"] = "ticket"
        self._grader = DeterministicGrader()
        self._cab = ChangeAdvisoryBoard()

    # ==================================================================
    # Public async API
    # ==================================================================

    async def reset(
        self,
        seed: int = 0,
        difficulty: str | None = None,
        mode: Literal["ticket", "incident"] = "ticket",
    ) -> StepResult:
        """Start a new episode in ticket or incident mode."""
        self._mode = mode
        self._incident_state = None
        self._state = None
        if mode == "incident":
            return self._reset_incident(seed=seed, difficulty=difficulty)

        ticket = self._bank.get_ticket(seed=seed, difficulty=difficulty)
        self._state = InternalState(ticket)
        return StepResult(
            observation=self._state.to_observation(),
            reward=0.0,
            done=False,
            info=self._state.to_info(),
        )

    def _reset_incident(self, seed: int, difficulty: str | None) -> StepResult:
        incident = self._incident_bank.get_incident(seed=seed, difficulty=difficulty)
        world = WorldState(seed=seed, incident=incident)
        crm = CRMSystem(incident.affected_customer_profiles)
        billing = BillingSystem(_build_billing_records(incident))
        history_store = IncidentHistoryStore.from_json(
            Path(__file__).resolve().parents[1] / "tasks" / "history_incidents.json"
        )
        runbook_engine = RunbookEngine.from_json(
            Path(__file__).resolve().parents[1] / "tasks" / "runbooks.json"
        )
        stakeholder_mgr = StakeholderManager()
        customer_queue_mgr = CustomerQueueManager(crm=crm)
        persistent_kb = PersistentKnowledgeBase(base_articles=_base_kb_articles())
        policy_engine = PolicyEngine(
            initial_policies=dict(incident.initial_policies),
            drift_schedule=_convert_policy_schedule(incident),
        )
        knowledge_base = persistent_kb.reset_for_episode(incident)
        self._incident_state = IncidentState(
            incident=incident,
            world=world,
            crm=crm,
            billing=billing,
            policy_engine=policy_engine,
            history_store=history_store,
            runbook_engine=runbook_engine,
            stakeholder_mgr=stakeholder_mgr,
            customer_queue_mgr=customer_queue_mgr,
            persistent_kb=persistent_kb,
            knowledge_base=knowledge_base,
            suggested_runbook=_suggested_runbook(runbook_engine, incident),
        )
        return StepResult(
            observation=self._incident_state.to_observation(),
            reward=0.0,
            done=False,
            info=self._incident_state.to_info(),
        )

    async def step(self, action: dict[str, Any] | Action) -> StepResult:  # type: ignore[type-arg]
        """Apply *action*, return ``(observation, reward, done, info)``."""
        if self._mode == "incident":
            return self._step_incident(action)

        state = self._require_active_state_ticket()

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
        reward, feedback, breakdown = self._dispatch_ticket(state, parsed)
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

        # --- reward breakdown --------------------------------------------
        breakdown["repeat_penalty"] = repeat_pen
        breakdown["sla_penalty"] = sla_pen
        breakdown["total"] = round(reward, 4)
        state.last_reward_breakdown = breakdown

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
        if self._mode == "incident":
            if self._incident_state is None:
                return None
            return StepResult(
                observation=self._incident_state.to_observation(),
                reward=0.0,
                done=self._incident_state.episode_done,
                info=self._incident_state.to_info(),
            )
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
        self._incident_state = None
        self._mode = "ticket"

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _require_active_state_ticket(self) -> InternalState:
        if self._state is None:
            raise EnvironmentNotResetError("Call reset() before step().")
        if self._state.done:
            raise EnvironmentDoneError("Episode ended. Call reset() for a new one.")
        return self._state

    def _safe_parse(self, action: dict[str, Any] | Action) -> Action | None:  # type: ignore[type-arg]
        if isinstance(action, dict):
            try:
                return ActionAdapter.validate_python(action)  # type: ignore[return-value]
            except (ValidationError, ValueError):
                return None
        if isinstance(action, ACTION_CLASSES):
            return action
        return None

    def _penalty(self, state: InternalState, label: str, feedback: str) -> StepResult:
        reward = -0.05
        state.last_reward_breakdown = {"penalty": reward, "total": reward}
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

    _DispatchResult = tuple[float, str, dict[str, float]]

    def _dispatch_ticket(self, state: InternalState, action: Action) -> _DispatchResult:
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
        return -0.05, "Unrecognised action type.", {"penalty": -0.05}

    def _step_incident(self, action: dict[str, Any] | Action) -> StepResult:  # type: ignore[type-arg]
        runtime = self._require_active_incident_state()
        parsed = self._safe_parse(action)
        if parsed is None:
            return self._incident_penalty(runtime, "invalid_parse", "Action failed schema validation.")

        action_json = parsed.model_dump_json(exclude_none=True)
        repeat_pen = -0.05 if action_json == runtime.last_action_json else 0.0
        runtime.last_action_json = action_json

        action_type: str = parsed.action_type  # type: ignore[union-attr]
        if action_type not in runtime.available_actions:
            return self._incident_penalty(
                runtime,
                action_type,
                f"Action '{action_type}' not valid in incident phase '{runtime.incident_phase}'.",
            )

        reward, feedback, breakdown = self._dispatch_incident(runtime, parsed)
        reward += repeat_pen
        reward = max(-0.25, min(reward, 0.30))

        breakdown["repeat_penalty"] = repeat_pen
        breakdown["total"] = round(reward, 4)
        runtime.last_reward_breakdown = breakdown
        runtime.record_action(action_type, feedback, reward)
        self._tick_incident_runtime(runtime)
        return StepResult(
            observation=runtime.to_observation(),
            reward=round(reward, 4),
            done=runtime.episode_done,
            info=runtime.to_info(),
        )

    def _require_active_incident_state(self) -> IncidentState:
        if self._incident_state is None:
            raise EnvironmentNotResetError("Call reset(mode='incident') before step().")
        if self._incident_state.episode_done:
            raise EnvironmentDoneError("Episode ended. Call reset() for a new one.")
        return self._incident_state

    def _incident_penalty(
        self, runtime: IncidentState, action_label: str, feedback: str
    ) -> StepResult:
        reward = -0.05
        runtime.last_reward_breakdown = {"penalty": reward, "total": reward}
        runtime.record_action(action_label, feedback, reward)
        self._tick_incident_runtime(runtime)
        return StepResult(
            observation=runtime.to_observation(),
            reward=reward,
            done=runtime.episode_done,
            info=runtime.to_info(),
        )

    def _tick_incident_runtime(self, runtime: IncidentState) -> None:
        world_events = runtime.world.tick()
        runtime.policy_engine.apply_scheduled_drifts(runtime.steps_taken)
        new_tickets = runtime.customer_queue_mgr.generate_tickets(
            runtime.world, runtime.steps_taken
        )
        runtime.world.support_queue.extend(new_tickets)
        runtime.customer_queue_mgr.update_frustration(runtime.steps_taken)
        runtime.stakeholder_mgr.tick()

        runtime.active_alerts = [
            event.message for event in world_events if event.event_type == "alert_generated"
        ]
        runtime.apply_severity_auto_escalation()
        runtime.maybe_update_reclassification()

        chaos = runtime.chaos_injector.maybe_inject(
            runtime.world, runtime.steps_taken, runtime.incident.difficulty
        )
        if chaos is not None:
            runtime.active_alerts.append(chaos.alert_text)
            runtime.known_facts["chaos_event"] = {
                "step": chaos.step,
                "service": chaos.new_service,
                "reason": chaos.reason,
            }

        runtime.advance_phase()
        if runtime.steps_taken >= runtime.max_steps or runtime.all_objectives_complete():
            runtime.episode_done = True

    def _dispatch_incident(self, runtime: IncidentState, action: Action) -> _DispatchResult:
        if isinstance(action, ClassifyAction):
            return self._on_incident_classify(runtime, action)
        if isinstance(action, RouteAction):
            return self._on_incident_route(runtime, action)
        if isinstance(action, RespondAction):
            return self._on_incident_respond(runtime, action)
        if isinstance(action, EscalateAction):
            return self._on_incident_escalate(runtime, action)
        if isinstance(action, ResolveAction):
            return self._on_incident_resolve(runtime, action)
        if isinstance(action, RequestInfoAction):
            return self._on_incident_request_info(runtime, action)
        if isinstance(action, CheckMonitoringAction):
            return self._on_check_monitoring(runtime, action)
        if isinstance(action, ProbeServiceAction):
            return self._on_probe_service(runtime, action)
        if isinstance(action, FetchLogsAction):
            return self._on_fetch_logs(runtime, action)
        if isinstance(action, FetchUserDataAction):
            return self._on_fetch_user_data(runtime, action)
        if isinstance(action, CheckBillingAction):
            return self._on_check_billing(runtime, action)
        if isinstance(action, QueryKBAction):
            return self._on_query_kb(runtime, action)
        if isinstance(action, CheckPolicyAction):
            return self._on_check_policy(runtime, action)
        if isinstance(action, QueryIncidentHistoryAction):
            return self._on_query_incident_history(runtime, action)
        if isinstance(action, ApplyFixAction):
            return self._on_apply_fix(runtime, action)
        if isinstance(action, VerifyFixAction):
            return self._on_verify_fix(runtime, action)
        if isinstance(action, RollbackFixAction):
            return self._on_rollback_fix(runtime, action)
        if isinstance(action, NotifyStakeholdersAction):
            return self._on_notify_stakeholders(runtime, action)
        if isinstance(action, FollowRunbookStepAction):
            return self._on_follow_runbook_step(runtime, action)
        if isinstance(action, WritePostmortemAction):
            return self._on_write_postmortem(runtime, action)
        if isinstance(action, UpdateKBAction):
            return self._on_update_kb(runtime, action)
        return -0.05, "Unrecognised incident action type.", {"penalty": -0.05}

    def _on_incident_classify(
        self, runtime: IncidentState, action: ClassifyAction
    ) -> _DispatchResult:
        severity = _severity_for_difficulty(runtime.incident.difficulty)
        correct = action.priority == severity
        runtime.current_severity = action.priority  # type: ignore[assignment]
        runtime.triage_complete = True
        reward = 0.1 if correct else 0.02
        return reward, "Incident severity classified.", {"classification": reward}

    def _on_incident_route(self, runtime: IncidentState, action: RouteAction) -> _DispatchResult:
        runtime.investigation_complete = True
        reward = 0.08 if action.department == "technical" else 0.02
        return reward, "Incident routed to response team.", {"routing": reward}

    def _on_incident_respond(
        self, runtime: IncidentState, action: RespondAction
    ) -> _DispatchResult:
        reward = 0.05 if action.tone == "empathetic" else 0.01
        return reward, "Customer communication sent.", {"respond": reward}

    def _on_incident_escalate(
        self, runtime: IncidentState, action: EscalateAction
    ) -> _DispatchResult:
        if not runtime.resource_budget.consume("escalation"):
            return -0.05, "Escalation budget exhausted.", {"escalate": -0.05}
        runtime.has_escalated = True
        runtime.add_audit_entry(
            action_type="escalate",
            target=action.target_team,
            policy_checked="escalation" in runtime.policies_checked,
            compliant=True,
        )
        reward = 0.06 if action.target_team in ("engineering", "management") else 0.03
        return reward, "Incident escalated.", {"escalate": reward}

    def _on_incident_resolve(self, runtime: IncidentState, action: ResolveAction) -> _DispatchResult:
        runtime.response_complete = True
        runtime.fix_verified = True
        if runtime.world.support_queue:
            resolved = runtime.world.support_queue.pop(0)
            runtime.tickets_resolved.append(resolved.ticket_id)
        runtime.add_audit_entry(
            action_type="resolve",
            target="ticket",
            policy_checked=True,
            compliant=True,
        )
        return 0.12, "Incident marked resolved.", {"resolve": 0.12}

    def _on_incident_request_info(
        self, runtime: IncidentState, action: RequestInfoAction
    ) -> _DispatchResult:
        return 0.03, "Additional customer information requested.", {"request_info": 0.03}

    def _on_check_monitoring(
        self, runtime: IncidentState, action: CheckMonitoringAction
    ) -> _DispatchResult:
        snapshot = runtime.world.service_mesh.get_monitoring_data(action.service_name)
        data = snapshot.model_dump()
        runtime.tool_results = {"check_monitoring": data}
        runtime.known_facts["system_status"] = runtime.world.service_mesh.get_health_summary()
        runtime.evidence_chain.add_evidence(
            step=runtime.steps_taken,
            source="check_monitoring",
            finding=str(data),
            conclusion="monitoring reviewed",
            service=action.service_name,
        )
        if action.service_name is None:
            runtime.triage_complete = True
        return 0.02, "Monitoring data retrieved.", {"check_monitoring": 0.02}

    def _on_probe_service(
        self, runtime: IncidentState, action: ProbeServiceAction
    ) -> _DispatchResult:
        result = runtime.world.service_mesh.probe_service(action.service_name, action.check_type)
        data = result.model_dump()
        runtime.tool_results = {"probe_service": data}
        runtime.known_facts[f"probe:{action.service_name}"] = data
        runtime.evidence_chain.add_evidence(
            step=runtime.steps_taken,
            source="probe_service",
            finding=str(data),
            conclusion=f"probe completed for {action.service_name}",
            service=action.service_name,
        )
        return 0.03, "Service probe completed.", {"probe_service": 0.03}

    def _on_fetch_logs(self, runtime: IncidentState, action: FetchLogsAction) -> _DispatchResult:
        state = runtime.world.service_mesh.services[action.service_name]
        logs = [
            f"{action.service_name} status={state.health}",
            f"{action.service_name} error_rate={state.error_rate:.2f}",
            f"time_range={action.time_range}",
        ]
        runtime.tool_results = {"fetch_logs": {"service": action.service_name, "entries": logs}}
        runtime.known_facts[f"logs:{action.service_name}"] = logs
        runtime.logs_checked_for.add(action.service_name)
        runtime.evidence_chain.add_evidence(
            step=runtime.steps_taken,
            source="fetch_logs",
            finding="; ".join(logs),
            conclusion=f"log context gathered for {action.service_name}",
            service=action.service_name,
        )
        return 0.02, "Logs retrieved.", {"fetch_logs": 0.02}

    def _on_fetch_user_data(
        self, runtime: IncidentState, action: FetchUserDataAction
    ) -> _DispatchResult:
        data = runtime.crm.fetch_user_data(action.customer_id).model_dump()
        runtime.tool_results = {"fetch_user_data": data}
        runtime.known_facts[f"customer:{action.customer_id}"] = data
        return 0.02, "CRM data fetched.", {"fetch_user_data": 0.02}

    def _on_check_billing(
        self, runtime: IncidentState, action: CheckBillingAction
    ) -> _DispatchResult:
        data = runtime.billing.check_billing(action.customer_id).model_dump()
        runtime.tool_results = {"check_billing": data}
        runtime.known_facts[f"billing:{action.customer_id}"] = data
        return 0.02, "Billing data fetched.", {"check_billing": 0.02}

    def _on_query_kb(self, runtime: IncidentState, action: QueryKBAction) -> _DispatchResult:
        result = runtime.knowledge_base.query(action.query).model_dump()
        runtime.tool_results = {"query_kb": result}
        runtime.known_facts[f"kb:{action.query}"] = result
        runtime.kb_queried = True
        return 0.02, "Knowledge base queried.", {"query_kb": 0.02}

    def _on_check_policy(
        self, runtime: IncidentState, action: CheckPolicyAction
    ) -> _DispatchResult:
        response = runtime.policy_engine.check_policy(action.policy_type).model_dump()
        runtime.active_policies[action.policy_type] = response
        runtime.policies_checked.add(action.policy_type)
        runtime.tool_results = {"check_policy": response}
        return 0.02, "Policy checked.", {"check_policy": 0.02}

    def _on_query_incident_history(
        self, runtime: IncidentState, action: QueryIncidentHistoryAction
    ) -> _DispatchResult:
        result = runtime.history_store.query(action.query, action.service_filter).model_dump()
        runtime.tool_results = {"query_incident_history": result}
        runtime.known_facts[f"history:{action.query}"] = result
        return 0.03, "Incident history queried.", {"query_incident_history": 0.03}

    def _on_apply_fix(self, runtime: IncidentState, action: ApplyFixAction) -> _DispatchResult:
        if not runtime.resource_budget.consume("fix_attempt"):
            return -0.05, "Fix attempts exhausted", {"apply_fix": -0.05}

        approval = self._cab.review_fix(
            fix=action,
            evidence_chain=runtime.evidence_chain,
            escalated=runtime.has_escalated,
        )
        if not approval.approved:
            runtime.resource_budget.remaining_fix_attempts += 1
            runtime.tool_results = {
                "apply_fix": {"cab_rejected": True, "reason": approval.reason}
            }
            runtime.add_audit_entry(
                action_type="apply_fix",
                target=action.service_name,
                policy_checked="escalation" in runtime.policies_checked,
                compliant=False,
            )
            return approval.penalty, approval.reason, {"cab_rejected": 1.0}

        result = runtime.world.service_mesh.apply_fix(action.service_name, action.fix_type)
        runtime.tool_results = {"apply_fix": result.model_dump()}
        runtime.fix_applied = True
        runtime.mark_fix_applied(action, result.success)
        if result.success:
            runtime.response_complete = True
            return 0.15, "Fix applied successfully!", {"fix_correct": 1.0}

        blast = runtime.world.service_mesh.apply_wrong_fix(action.service_name, action.fix_type)
        runtime.tool_results["apply_fix"]["blast_radius"] = blast.model_dump()
        return blast.penalty, blast.description, {"blast_radius": 1.0}

    def _on_verify_fix(self, runtime: IncidentState, action: VerifyFixAction) -> _DispatchResult:
        health = runtime.world.service_mesh.get_health_summary().get(action.service_name, "unknown")
        runtime.tool_results = {"verify_fix": {"service_name": action.service_name, "health": health}}
        runtime.fix_verified = health == "healthy"
        if runtime.fix_verified:
            runtime.response_complete = True
        reward = 0.05 if health == "healthy" else -0.02
        return reward, "Fix verification completed.", {"verify_fix": reward}

    def _on_rollback_fix(
        self, runtime: IncidentState, action: RollbackFixAction
    ) -> _DispatchResult:
        service = runtime.world.service_mesh.services[action.service_name]
        if service.fix_applied and not service.fix_correct:
            service.health = "degraded"
            runtime.tool_results = {"rollback_fix": {"service_name": action.service_name, "rolled_back": True}}
            return 0.02, "Rolled back a bad fix.", {"rollback_fix": 0.02}
        if service.fix_correct:
            service.health = "degraded"
            runtime.tool_results = {"rollback_fix": {"service_name": action.service_name, "rolled_back": True}}
            return -0.03, "Rolled back a correct fix.", {"rollback_fix": -0.03}
        runtime.tool_results = {"rollback_fix": {"service_name": action.service_name, "rolled_back": False}}
        return -0.01, "No fix to roll back.", {"rollback_fix": -0.01}

    def _on_notify_stakeholders(
        self, runtime: IncidentState, action: NotifyStakeholdersAction
    ) -> _DispatchResult:
        if not runtime.resource_budget.consume("notification"):
            return -0.05, "Stakeholder notification budget exhausted.", {"notify_stakeholders": -0.05}
        targets = (
            ["vp_engineering", "legal", "support_lead"]
            if action.stakeholder == "all"
            else [action.stakeholder]
        )
        results = [runtime.stakeholder_mgr.notify(target, action.message).model_dump() for target in targets]
        runtime.tool_results = {"notify_stakeholders": results}
        runtime.add_audit_entry(
            action_type="notify_stakeholders",
            target=action.stakeholder,
            policy_checked="communication" in runtime.policies_checked,
            compliant=True,
        )
        return 0.03, "Stakeholders notified.", {"notify_stakeholders": 0.03}

    def _on_follow_runbook_step(
        self, runtime: IncidentState, action: FollowRunbookStepAction
    ) -> _DispatchResult:
        step = runtime.runbook_engine.follow_runbook_step(action.runbook_id, action.step_index)
        is_correct = runtime.runbook_engine.is_correct_for_incident(action.runbook_id)
        runtime.tool_results = {"follow_runbook_step": step.model_dump()}
        reward = 0.03 if is_correct else -0.08
        return reward, "Runbook step executed.", {"follow_runbook_step": reward}

    def _on_write_postmortem(
        self, runtime: IncidentState, action: WritePostmortemAction
    ) -> _DispatchResult:
        has_root_cause = bool(action.root_cause_description.strip())
        has_remediation = len(action.remediation_steps) > 0
        has_prevention = len(action.prevention_measures) > 0
        quality = 0.03 + (0.01 if has_root_cause else 0.0) + (0.01 if has_remediation else 0.0) + (0.01 if has_prevention else 0.0)
        runtime.tool_results = {"write_postmortem": {"quality": quality}}
        runtime.postmortem_written = True
        return quality, "Postmortem recorded.", {"write_postmortem": quality}

    def _on_update_kb(self, runtime: IncidentState, action: UpdateKBAction) -> _DispatchResult:
        result = runtime.knowledge_base.update_article(action.article_title, action.content)
        runtime.persistent_kb.record_update(
            title=action.article_title,
            content=action.content,
            accepted_for_persistence=result.accepted_for_persistence,
        )
        runtime.tool_results = {"update_kb": result.model_dump()}
        runtime.kb_updated = True
        reward = 0.05 if result.accepted_for_persistence else 0.01
        return reward, result.message, {"update_kb": reward}

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _on_classify(
        self, state: InternalState, action: ClassifyAction
    ) -> _DispatchResult:
        ticket = state.ticket
        cat_ok = action.category == ticket.gold_category
        pri_ok = action.priority == ticket.gold_priority
        state.classification_correct = cat_ok and pri_ok
        state.phase = "classified"

        if cat_ok and pri_ok:
            base, feedback = 0.10, "Correct classification."
        elif cat_ok:
            base = 0.06
            feedback = f"Category correct; expected priority '{ticket.gold_priority}'."
        elif pri_ok:
            base = 0.04
            feedback = f"Priority correct; expected category '{ticket.gold_category}'."
        else:
            base = 0.01
            feedback = (
                f"Incorrect. Expected '{ticket.gold_category}' / '{ticket.gold_priority}'."
            )

        urgency_bonus = 0.0
        if pri_ok and ticket.gold_priority in ("critical", "high"):
            urgency_bonus = 0.10
            state.urgency_handled = True
            feedback += " Urgency correctly identified (+0.10)."

        reward = base + urgency_bonus
        breakdown = {"classification": base, "urgency_bonus": urgency_bonus}
        return reward, feedback, breakdown

    def _on_route(
        self, state: InternalState, action: RouteAction
    ) -> _DispatchResult:
        correct = action.department == state.ticket.gold_department
        state.routing_correct = correct
        state.phase = "routed"
        if correct:
            return 0.10, "Routed to the correct department.", {"routing": 0.10}
        return 0.01, f"Incorrect routing; expected '{state.ticket.gold_department}'.", {"routing": 0.01}

    def _on_respond(
        self, state: InternalState, action: RespondAction
    ) -> _DispatchResult:
        ticket = state.ticket
        state.phase = "responding"

        quality, forbidden_pen = self._grader.weighted_keyword_score(
            action.response_text, ticket.response_spec
        )

        info_factor = 1.0
        if ticket.partial_info and not state.info_requested:
            info_factor = _INFO_SKIP_QUALITY_FACTOR
            quality *= info_factor

        base_reward = round(quality * 0.20, 4)

        forbidden_pen = round(forbidden_pen * ticket.penalty_multiplier, 4)
        tone_pen = self._eval_tone_constraint(state, action.tone)
        tone_pen = round(tone_pen * ticket.penalty_multiplier, 4)

        reward = base_reward + forbidden_pen + tone_pen
        state.response_quality_score = quality

        parts = [f"Response quality: {quality:.0%}."]
        if info_factor < 1.0:
            parts.append(
                f"Incomplete information (factor {info_factor:.1f}): "
                "customer clarification was not gathered."
            )
        if forbidden_pen < 0:
            parts.append(f"Forbidden pattern penalty: {forbidden_pen:+.3f}.")
        if tone_pen < 0:
            parts.append("Tone constraint violated.")

        breakdown: dict[str, float] = {
            "response_quality": base_reward,
            "incomplete_info_factor": info_factor,
            "forbidden_penalty": forbidden_pen,
            "tone_penalty": tone_pen,
        }
        return round(reward, 4), " ".join(parts), breakdown

    def _on_escalate(
        self, state: InternalState, action: EscalateAction
    ) -> _DispatchResult:
        ticket = state.ticket
        state.phase = "escalated"

        if not ticket.requires_escalation:
            pen = round(-0.10 * ticket.penalty_multiplier, 4)
            state.escalation_score = 0.0
            return pen, "Unnecessary escalation; ticket does not require it.", {"escalation": pen}

        team_ok = (
            ticket.escalation_target is not None
            and action.target_team == ticket.escalation_target
        )
        if team_ok:
            state.escalation_score = 1.0
            return 0.15, "Correctly escalated to the right team.", {"escalation": 0.15}

        state.escalation_score = 0.3
        return 0.05, (
            f"Escalation needed but target should be '{ticket.escalation_target}'."
        ), {"escalation": 0.05}

    def _on_resolve(
        self, state: InternalState, action: ResolveAction
    ) -> _DispatchResult:
        ticket = state.ticket

        quality, forbidden_pen = self._grader.weighted_keyword_score(
            action.resolution_summary, ticket.resolution_spec
        )

        info_factor = 1.0
        if ticket.partial_info and not state.info_requested:
            info_factor = _INFO_SKIP_QUALITY_FACTOR
            quality *= info_factor

        comp_score = self._grader.compensation_accuracy(
            action.offered_compensation, ticket.compensation_range
        )
        constraint_pen = self._eval_resolve_constraints(state, action)

        quality_r = round(quality * 0.15, 4)
        comp_r = round(comp_score * 0.05, 4)
        base_r = 0.05
        raw = quality_r + comp_r + base_r
        forbidden_scaled = round(forbidden_pen * ticket.penalty_multiplier, 4)
        constraint_scaled = round(constraint_pen * ticket.penalty_multiplier, 4)

        reward = round(max(0.0, min(raw + forbidden_scaled + constraint_scaled, 0.25)), 4)
        state.resolution_quality_score = quality

        parts = [f"Resolution quality: {quality:.0%}."]
        if info_factor < 1.0:
            parts.append(
                f"Incomplete information (factor {info_factor:.1f}): "
                "customer clarification was not gathered."
            )
        if forbidden_pen < 0:
            parts.append(f"Forbidden pattern penalty: {forbidden_pen:+.3f}.")
        if constraint_pen < 0:
            parts.append(f"Constraint penalty: {constraint_pen:+.3f}.")
        if comp_score < 1.0 and ticket.compensation_range is not None:
            parts.append("Compensation outside optimal range.")

        breakdown: dict[str, float] = {
            "resolution_quality": quality_r,
            "incomplete_info_factor": info_factor,
            "compensation": comp_r,
            "base": base_r,
            "forbidden_penalty": forbidden_scaled,
            "constraint_penalty": constraint_scaled,
        }
        return reward, " ".join(parts), breakdown

    def _on_request_info(
        self, state: InternalState, action: RequestInfoAction
    ) -> _DispatchResult:
        ticket = state.ticket

        if ticket.partial_info and not state.info_requested:
            state.info_requested = True
            reveal = ticket.info_reveals or "No additional details available."
            return 0.05, f"Customer clarification: {reveal}", {"info_bonus": 0.05}

        pen = round(-0.03 * ticket.penalty_multiplier, 4)
        if state.info_requested:
            return pen, "No new information. Customer already provided clarification.", {"info_penalty": pen}
        return pen, "Information requested. No additional details available (simulated).", {"info_penalty": pen}

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


def _severity_for_difficulty(difficulty: str) -> str:
    if difficulty == "easy":
        return "medium"
    if difficulty == "medium":
        return "high"
    return "critical"


def _build_billing_records(incident: IncidentScenario) -> dict[str, BillingRecord]:
    records: dict[str, BillingRecord] = {}
    for customer in incident.affected_customer_profiles:
        balance = 500.0 if customer.tier == "enterprise" else 150.0
        status: Literal["current", "overdue", "failed", "disputed"] = "failed"
        records[customer.customer_id] = BillingRecord(
            customer_id=customer.customer_id,
            current_balance=balance,
            payment_status=status,
            pending_invoices=[
                Invoice(
                    invoice_id=f"INV-{customer.customer_id}",
                    amount=balance,
                    due_step=2,
                    status="open",
                )
            ],
            failed_payments=[
                FailedPayment(
                    payment_id=f"PAY-{customer.customer_id}",
                    amount=balance,
                    reason="incident_related_failure",
                )
            ],
            total_lifetime_value=10000.0 if customer.tier == "enterprise" else 3000.0,
        )
    return records


def _convert_policy_schedule(incident: IncidentScenario) -> list[EnginePolicyChange]:
    changes: list[EnginePolicyChange] = []
    for item in incident.policy_drift_schedule:
        policy_type = "refund" if item.key == "refund_cap" else (
            "escalation" if item.key == "escalation_required" else "communication"
        )
        key_name = "max_refund" if policy_type == "refund" else (
            "required" if policy_type == "escalation" else item.key
        )
        changes.append(
            EnginePolicyChange(
                trigger_step=item.step,
                policy_type=policy_type,
                old_value={},
                new_value={key_name: item.value},
                reason=f"Scheduled drift for {item.key}",
            )
        )
    return changes


def _base_kb_articles() -> list[KBArticle]:
    return [
        KBArticle(
            article_id="KB-BASE-001",
            title="Database OOM Recovery",
            content="verify root cause and apply fix safely",
            solution_steps=["verify", "apply memory fix"],
            tags=["database", "oom"],
            last_updated="2026-04-20",
            is_accurate=True,
        )
    ]


def _suggested_runbook(runbook_engine: RunbookEngine, incident: IncidentScenario) -> dict[str, Any] | None:
    incident_type = "payment_500"
    first = incident.root_causes[0] if incident.root_causes else None
    if first is not None:
        if first.service == "database" and first.failure_mode == "oom":
            incident_type = "database_oom"
        elif first.service == "auth" and first.failure_mode == "rate_limiting":
            incident_type = "auth_rate_limiting"
    suggestion = runbook_engine.suggest_runbook(incident_type)
    if suggestion is None:
        return None
    return suggestion.model_dump()
