"""Sandbox-backed environment adapter.

This class preserves the OpenEnv-facing API (`reset/step/state/close`) while
running selected incident actions against a live container cluster.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from env.environment import CustomerSupportEnv
from models.action import Action, ActionAdapter
from models.step_result import StepResult
from sandbox.env_adapter.bridge import (
    SandboxBridge,
    SandboxBridgeError,
    SandboxConnectionError,
    SandboxValidationError,
)
from sandbox.env_adapter.drill import DrillEvent, build_curriculum_schedule
from tasks.incident_bank import IncidentBank

ServiceName = Literal["auth", "database", "payments", "analytics", "notifications"]
_KNOWN_SERVICES: tuple[ServiceName, ...] = (
    "auth",
    "database",
    "payments",
    "analytics",
    "notifications",
)


def _service_map(base_url: str) -> dict[str, str]:
    root = base_url.rstrip("/")
    return {
        "auth": f"{root}:5001",
        "database": f"{root}:5002",
        "payments": f"{root}:5003",
        "analytics": f"{root}:5004",
        "notifications": f"{root}:5005",
    }


class SandboxEnv:
    """Drop-in environment that augments incident actions with live cluster I/O."""

    def __init__(
        self,
        *,
        cluster_base_url: str | None = None,
        chaos_url: str | None = None,
        timeout_s: float = 2.0,
    ) -> None:
        base = (cluster_base_url or os.environ.get("OPENENV_SANDBOX_CLUSTER_URL") or "http://localhost").rstrip("/")
        chaos = (chaos_url or os.environ.get("OPENENV_SANDBOX_CHAOS_URL") or f"{base}:6660").rstrip("/")
        self._sim_env = CustomerSupportEnv()
        self._incident_bank = IncidentBank()
        self._bridge = SandboxBridge(
            services=_service_map(base),
            chaos_url=chaos,
            timeout_s=timeout_s,
        )
        self._sandbox_enabled = True
        self._mode: Literal["ticket", "incident"] = "ticket"
        self._last_seed = 0
        self._last_difficulty: str | None = None
        self._drill_state: DrillState | None = None

    async def reset(
        self,
        seed: int = 0,
        difficulty: str | None = None,
        mode: Literal["ticket", "incident"] = "ticket",
        drill_mode: bool = False,
        drill_seed: int | None = None,
    ) -> StepResult:
        self._mode = mode
        self._last_seed = seed
        self._last_difficulty = difficulty
        result = await self._sim_env.reset(seed=seed, difficulty=difficulty, mode=mode)
        if mode != "incident":
            return result

        sandbox_info = await self._run_live_reset(seed=seed, difficulty=difficulty)
        if drill_mode:
            drill_seed_value = seed if drill_seed is None else drill_seed
            drill_events = build_curriculum_schedule(
                seed=drill_seed_value,
                difficulty=difficulty or "medium",
                max_steps=result.observation.max_steps,
            )
            self._drill_state = DrillState(
                enabled=True,
                seed=drill_seed_value,
                difficulty=difficulty or "medium",
                schedule=drill_events,
            )
            sandbox_info["drill"] = self._drill_state.to_payload()
        else:
            self._drill_state = None
        self._annotate_result(result, sandbox_info=sandbox_info)
        return result

    async def step(self, action: dict[str, Any] | Action) -> StepResult:  # type: ignore[type-arg]
        result = await self._sim_env.step(action)
        if self._mode != "incident":
            return result

        sandbox_info = await self._run_live_action(action)
        if self._drill_state is not None and self._drill_state.enabled:
            current_step = int(result.observation.current_step)
            drill_meta = await self._advance_drill(current_step=current_step)
            sandbox_info["drill"] = drill_meta
        self._annotate_result(result, sandbox_info=sandbox_info)
        return result

    async def state(self) -> StepResult | None:
        result = await self._sim_env.state()
        if result is None or self._mode != "incident":
            return result
        try:
            snapshot = await self._run_io(self._bridge.chaos_status)
        except SandboxBridgeError as exc:
            snapshot = {"status": "unavailable", "error": str(exc)}
        state_payload: dict[str, Any] = {"state_snapshot": snapshot}
        if self._drill_state is not None and self._drill_state.enabled:
            state_payload["drill"] = self._drill_state.to_payload()
        self._annotate_result(result, sandbox_info=state_payload)
        return result

    async def close(self) -> None:
        # Best-effort cleanup so stale injected failures do not leak
        # between manual local test runs.
        if self._mode == "incident":
            try:
                await self._run_io(self._bridge.chaos_clear_all)
            except Exception:
                pass
        self._drill_state = None
        await self._sim_env.close()

    async def _run_live_reset(self, *, seed: int, difficulty: str | None) -> dict[str, Any]:
        try:
            clear_resp = await self._run_io(self._bridge.chaos_clear_all)
            incident = self._incident_bank.get_incident(seed=seed, difficulty=difficulty)
            injected: list[dict[str, str]] = []
            skipped: list[dict[str, str]] = []
            for cause in incident.root_causes:
                try:
                    await self._run_io(self._bridge.chaos_inject, cause.service, cause.failure_mode)
                    injected.append({"service": cause.service, "failure_mode": cause.failure_mode})
                except SandboxValidationError as exc:
                    # Skip injections that the chaos controller rejects
                    # (e.g. failure_mode not mapped to that service) instead
                    # of disabling sandbox mode entirely.
                    skipped.append(
                        {
                            "service": cause.service,
                            "failure_mode": cause.failure_mode,
                            "reason": exc.detail,
                        }
                    )
            await asyncio.sleep(0.2)
            status = await self._run_io(self._bridge.chaos_status)
            self._sandbox_enabled = True
            payload: dict[str, Any] = {
                "backend": "live_cluster",
                "incident_id": incident.incident_id,
                "cleared": clear_resp.get("status", "ok"),
                "injected_failures": injected,
                "chaos_status": status,
            }
            if skipped:
                payload["skipped_failures"] = skipped
            return payload
        except SandboxConnectionError as exc:
            self._sandbox_enabled = False
            return {"backend": "sim_fallback", "error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive fallback path
            self._sandbox_enabled = False
            return {"backend": "sim_fallback", "error": str(exc)}

    async def _run_live_action(self, action: dict[str, Any] | Action) -> dict[str, Any]:
        payload = self._normalize_action(action)
        action_type = str(payload.get("action_type", ""))
        if not self._sandbox_enabled:
            return {"backend": "sim_fallback", "skipped_action": action_type}

        try:
            if action_type == "check_monitoring":
                service_name = payload.get("service_name")
                if isinstance(service_name, str) and service_name:
                    health = await self._run_io(self._bridge.check_health, service_name)
                    return {"backend": "live_cluster", "check_monitoring": {"service": service_name, "health": health}}
                snapshots: dict[str, Any] = {}
                for service in _KNOWN_SERVICES:
                    snapshots[service] = await self._run_io(self._bridge.check_health, service)
                return {"backend": "live_cluster", "check_monitoring": {"services": snapshots}}

            if action_type == "probe_service":
                service_name = str(payload.get("service_name", ""))
                if not service_name:
                    return {"backend": "live_cluster", "probe_service": {"error": "missing service_name"}}
                health = await self._run_io(self._bridge.check_health, service_name)
                metrics = await self._run_io(self._bridge.fetch_metrics_text, service_name)
                return {
                    "backend": "live_cluster",
                    "probe_service": {
                        "service": service_name,
                        "check_type": payload.get("check_type"),
                        "health": health,
                        "metrics_excerpt": metrics.splitlines()[:12],
                    },
                }

            if action_type == "fetch_logs":
                service_name = str(payload.get("service_name", ""))
                if not service_name:
                    return {"backend": "live_cluster", "fetch_logs": {"error": "missing service_name"}}
                lines = 30 if payload.get("time_range") == "last_5m" else 80
                logs = await self._run_io(self._bridge.fetch_logs, service_name, lines=lines)
                return {"backend": "live_cluster", "fetch_logs": logs}

            if action_type == "apply_fix":
                service_name = str(payload.get("service_name", ""))
                fix_type = str(payload.get("fix_type", "restart_service"))
                if not service_name:
                    return {"backend": "live_cluster", "apply_fix": {"error": "missing service_name"}}
                verification = await self._run_io(self._bridge.chaos_verify_fix, service_name, fix_type)
                return {"backend": "live_cluster", "apply_fix": verification}

            if action_type == "verify_fix":
                service_name = str(payload.get("service_name", ""))
                if not service_name:
                    return {"backend": "live_cluster", "verify_fix": {"error": "missing service_name"}}
                health = await self._run_io(self._bridge.check_health, service_name)
                return {"backend": "live_cluster", "verify_fix": health}

            if action_type == "rollback_fix":
                service_name = str(payload.get("service_name", ""))
                if not service_name:
                    return {"backend": "live_cluster", "rollback_fix": {"error": "missing service_name"}}
                cleared = await self._run_io(self._bridge.chaos_clear, service_name)
                return {"backend": "live_cluster", "rollback_fix": cleared}

            return {"backend": "live_cluster", "note": f"action '{action_type}' uses simulated backend only"}
        except SandboxValidationError as exc:
            # Validation errors (4xx) are recoverable - the agent picked a bad
            # input. Keep sandbox enabled so subsequent actions can still hit
            # the live cluster.
            return {
                "backend": "live_cluster",
                "action_type": action_type,
                "validation_error": exc.detail,
                "status_code": exc.status_code,
            }
        except SandboxConnectionError as exc:
            # Cluster-level failure: degrade to simulated backend for the
            # remainder of the episode so eval keeps progressing.
            self._sandbox_enabled = False
            return {"backend": "sim_fallback", "action_type": action_type, "error": str(exc)}
        except Exception as exc:  # defensive catch-all
            self._sandbox_enabled = False
            return {"backend": "sim_fallback", "action_type": action_type, "error": str(exc)}

    async def _advance_drill(self, *, current_step: int) -> dict[str, Any]:
        if self._drill_state is None:
            return {"enabled": False}

        injected_now: list[dict[str, Any]] = []
        for event in self._drill_state.schedule:
            if event.step > current_step:
                continue
            if event.key in self._drill_state.injected_steps:
                continue
            try:
                await self._run_io(self._bridge.chaos_inject, event.service, event.failure_mode)
                self._drill_state.injected_steps[event.key] = current_step
                injected_now.append(
                    {
                        "step": current_step,
                        "service": event.service,
                        "failure_mode": event.failure_mode,
                        "severity": event.severity,
                    }
                )
            except SandboxConnectionError as exc:
                # Cluster unreachable while injecting - drop drill mode but
                # leave already-injected events in place.
                self._drill_state.errors.append(
                    f"inject_unreachable:{event.service}:{event.failure_mode}:{exc}"
                )
            except SandboxBridgeError as exc:
                self._drill_state.errors.append(
                    f"inject_failed:{event.service}:{event.failure_mode}:{exc}"
                )
            except Exception as exc:
                self._drill_state.errors.append(
                    f"inject_failed:{event.service}:{event.failure_mode}:{exc}"
                )

        for event in self._drill_state.schedule:
            if event.key not in self._drill_state.injected_steps:
                continue
            if event.key in self._drill_state.resolved_steps:
                continue
            try:
                health = await self._run_io(self._bridge.check_health, event.service)
            except SandboxBridgeError as exc:
                self._drill_state.errors.append(
                    f"health_failed:{event.service}:{event.failure_mode}:{exc}"
                )
                continue
            except Exception as exc:
                self._drill_state.errors.append(
                    f"health_failed:{event.service}:{event.failure_mode}:{exc}"
                )
                continue
            if str(health.get("status", "")) == "healthy":
                self._drill_state.resolved_steps[event.key] = current_step

        payload = self._drill_state.to_payload()
        payload["injected_now"] = injected_now
        return payload

    async def _run_io(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        return await asyncio.to_thread(fn, *args, **kwargs)

    @staticmethod
    def _normalize_action(action: dict[str, Any] | Action) -> dict[str, Any]:  # type: ignore[type-arg]
        if isinstance(action, dict):
            parsed = ActionAdapter.validate_python(action)
            return parsed.model_dump(exclude_none=True)
        return action.model_dump(exclude_none=True)

    @staticmethod
    def _annotate_result(result: StepResult, *, sandbox_info: dict[str, Any]) -> None:
        result.info["sandbox"] = sandbox_info
        if result.observation.tool_results is None:
            result.observation.tool_results = {}
        if isinstance(result.observation.tool_results, dict):
            result.observation.tool_results["sandbox_live"] = sandbox_info


@dataclass(slots=True)
class DrillState:
    """Runtime drill scheduler + score tracking."""

    enabled: bool
    seed: int
    difficulty: str
    schedule: list[DrillEvent] = field(default_factory=list)
    injected_steps: dict[str, int] = field(default_factory=dict)
    resolved_steps: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        total = len(self.schedule)
        resolved = len(self.resolved_steps)
        on_time = 0
        for event in self.schedule:
            r_step = self.resolved_steps.get(event.key)
            if r_step is not None and r_step <= event.deadline_step:
                on_time += 1
        score = 0.0
        if total > 0:
            score = ((resolved / total) * 0.6) + ((on_time / total) * 0.4)
        return {
            "enabled": self.enabled,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "total_events": total,
            "resolved_events": resolved,
            "on_time_resolved_events": on_time,
            "drill_score": round(score, 4),
            "events": [asdict(event) for event in self.schedule],
            "injected_steps": dict(self.injected_steps),
            "resolved_steps": dict(self.resolved_steps),
            "errors": list(self.errors[-8:]),
        }

