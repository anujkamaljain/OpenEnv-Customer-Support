"""FastAPI server exposing the CustomerSupportEnv over HTTP.

Endpoints:
    GET  /              — browser debug UI (HTML)
    GET  /docs          — interactive OpenAPI (Swagger UI, default theme)
    POST /reset         — start a new episode
    POST /step          — apply an action
    GET  /state         — read current state without advancing
    POST /close         — release episode resources
    POST /inference     — run the full LLM inference loop
    GET  /health        — liveness probe (JSON)
"""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import redirect_stdout
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from env.environment import CustomerSupportEnv
from models.action import Action
from sandbox.env_adapter import SandboxEnv

_OPENAPI_TAGS = [
    {
        "name": "Environment",
        "description": (
            "Episode lifecycle for the customer-support RL environment: "
            "call **reset** before **step**; use **state** to inspect without advancing."
        ),
    },
    {
        "name": "Interface",
        "description": "Browser UI for manual debugging (HTML, not JSON).",
    },
    {
        "name": "System",
        "description": "Health checks and optional batch inference (requires server env configuration).",
    },
]

app = FastAPI(
    title="OpenEnv Enterprise Incident Command Center",
    version="2.0.0",
    openapi_tags=_OPENAPI_TAGS,
    description=(
        "Deterministic enterprise incident simulation for [OpenEnv](https://github.com/open-env). "
        "Supports backward-compatible ticket mode and a full incident command lifecycle "
        "with enterprise tools, policy drift, and multi-step remediation."
    ),
)

_DEFAULT_SESSION_ID = "default"
_API_KEY = os.environ.get("OPENENV_API_KEY", "").strip()
_RATE_LIMIT_PER_MIN = int(os.environ.get("OPENENV_RATE_LIMIT_PER_MIN", "120"))
_AUDIT_LOG_PATH = os.environ.get("OPENENV_AUDIT_LOG_PATH", "").strip()
_AUTH_EXEMPT_PATHS = frozenset(["/", "/docs", "/openapi.json", "/redoc", "/health"])
EnvLike = CustomerSupportEnv | SandboxEnv
_USE_SANDBOX = os.environ.get("OPENENV_SANDBOX", "false").strip().lower() in {"1", "true", "yes", "on"}
_SANDBOX_CLUSTER_URL = os.environ.get("OPENENV_SANDBOX_CLUSTER_URL", "http://localhost").strip()
_SANDBOX_CHAOS_URL = os.environ.get("OPENENV_SANDBOX_CHAOS_URL", "http://localhost:6660").strip()


def _create_env() -> EnvLike:
    if _USE_SANDBOX:
        return SandboxEnv(cluster_base_url=_SANDBOX_CLUSTER_URL, chaos_url=_SANDBOX_CHAOS_URL)
    return CustomerSupportEnv()


_SESSION_ENVS: dict[str, EnvLike] = {_DEFAULT_SESSION_ID: _create_env()}
_RATE_LIMIT_WINDOWS: dict[str, deque[float]] = defaultdict(deque)


def _resolve_session_id(
    request_value: str | None,
    header_value: str | None,
) -> str:
    session_id = (request_value or header_value or _DEFAULT_SESSION_ID).strip()
    return session_id or _DEFAULT_SESSION_ID


def _audit_event(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    if _AUDIT_LOG_PATH:
        path = Path(_AUDIT_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


async def _get_env(session_id: str, *, create_if_missing: bool = True) -> EnvLike:
    env = _SESSION_ENVS.get(session_id)
    if env is None and create_if_missing:
        env = _create_env()
        _SESSION_ENVS[session_id] = env
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id '{session_id}'")
    return env


async def _close_session(session_id: str) -> None:
    env = _SESSION_ENVS.get(session_id)
    if env is None:
        return
    await env.close()
    if session_id != _DEFAULT_SESSION_ID:
        _SESSION_ENVS.pop(session_id, None)


@app.middleware("http")
async def security_and_audit_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    started = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    req_api_key = request.headers.get("X-API-Key", "")
    req_session_id = request.headers.get("X-Session-ID", _DEFAULT_SESSION_ID)
    status_code = 500

    try:
        if _API_KEY and request.url.path not in _AUTH_EXEMPT_PATHS:
            if req_api_key != _API_KEY:
                status_code = 401
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        if _RATE_LIMIT_PER_MIN > 0 and request.url.path not in _AUTH_EXEMPT_PATHS:
            now = time.monotonic()
            rate_key = req_api_key or client_host
            window = _RATE_LIMIT_WINDOWS[rate_key]
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= _RATE_LIMIT_PER_MIN:
                status_code = 429
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
            window.append(now)

        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        _audit_event(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "elapsed_ms": elapsed_ms,
                "client": client_host,
                "session_id": req_session_id,
            }
        )


# ── request / response schemas ───────────────────────────────────────────────


class ResetRequest(BaseModel):
    """Start a new episode; same inputs yield the same deterministic scenario."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"seed": 0, "difficulty": "easy", "mode": "ticket"},
                {"seed": 0, "difficulty": "easy", "mode": "incident"},
                {"seed": 0, "difficulty": "nightmare", "mode": "incident"},
                {
                    "seed": 0,
                    "difficulty": "hard",
                    "mode": "incident",
                    "drill_mode": True,
                    "drill_seed": 7,
                },
                {"seed": 0, "difficulty": None, "mode": "ticket"},
            ]
        }
    )

    seed: int = Field(
        default=0,
        description="Index into the ticket pool (modulo pool size). With `difficulty` omitted, indexes the combined bank.",
    )
    mode: Literal["ticket", "incident"] = Field(
        default="ticket",
        description="Episode mode. `ticket` preserves legacy triage behavior; `incident` enables incident lifecycle simulation.",
    )
    difficulty: Literal["easy", "medium", "hard", "nightmare"] | None = Field(
        default=None,
        description=(
            "Difficulty filter. Ticket mode supports easy/medium/hard; "
            "incident mode supports easy/medium/hard/nightmare."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Optional episode session key. If omitted, uses the default shared session.",
    )
    drill_mode: bool = Field(
        default=False,
        description=(
            "Sandbox-only: enable deterministic failure curriculum drill events "
            "during incident episodes."
        ),
    )
    drill_seed: int | None = Field(
        default=None,
        description="Sandbox-only deterministic seed override for drill schedule.",
    )


class StepRequest(BaseModel):
    """Apply one agent action. Must match the current phase (see `observation.phase` and `available_actions`)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": {
                    "action_type": "classify",
                    "category": "general_inquiry",
                    "priority": "medium",
                }
            }
        }
    )

    action: Action = Field(
        description=(
            "Discriminated union on `action_type`: classify, route, respond, escalate, request_info, "
            "resolve, check_monitoring, probe_service, fetch_logs, fetch_user_data, check_billing, "
            "query_kb, check_policy, query_incident_history, follow_runbook_step, apply_fix, verify_fix, "
            "rollback_fix, notify_stakeholders, write_postmortem, update_kb. "
            "Wrong shape -> **422**; wrong phase -> **200** with a penalty in `reward`."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Optional episode session key. Must match a prior reset session to avoid cross-episode overlap.",
    )


class EnvResponse(BaseModel):
    """Observation and reward after reset, step, or state."""

    observation: dict[str, Any] | None = Field(
        description="Ticket + phase + history. `null` only from GET /state when reset has not been called yet.",
    )
    reward: float = Field(description="Reward for the last transition; 0 on reset and on GET /state.")
    done: bool = Field(description="True after a terminal resolve (episode finished).")
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostics (e.g. difficulty, last feedback, reward breakdown).",
    )


class InferenceResponse(BaseModel):
    """Output from running the bundled LLM inference script once."""

    stdout: str = Field(description="Captured stdout from `inference.run()` (validator-style log lines).")
    score: float = Field(description="Parsed from the final `[END]` line when present; else 0.0.")
    success: bool = Field(description="Parsed from the final `[END]` line when present; else false.")


# ── helpers ──────────────────────────────────────────────────────────────────


def _result_to_dict(result: Any, *, session_id: str | None = None) -> dict[str, Any]:
    info = dict(result.info)
    if session_id is not None:
        info["session_id"] = session_id
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": info,
    }


_DEBUG_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Customer Support Command Center</title>
  <style>
    :root {
      --bg: #141517;
      --surface: #25262b;
      --border: #373a40;
      --text: #e9ecef;
      --muted: #868e96;
      --accent: #4c6ef5;
      --exec: #e8590c;
      --exec-hover: #fd7e14;
      --hint-bg: #1b4332;
      --hint-border: #2b8a3e;
      --card-blue: #1864ab;
      --card-orange: #d9480f;
      --card-green: #2b8a3e;
      --card-grey: #495057;
      --danger: #fa5252;
      --mono: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
    }
    * { box-sizing: border-box; }
    html { color-scheme: dark; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    .wrap { max-width: 56rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
    h1 { font-size: 1.35rem; font-weight: 700; margin: 0 0 0.25rem; letter-spacing: -0.02em; }
    .sub { color: var(--muted); font-size: 0.875rem; margin: 0 0 1.25rem; }
    .hint-bar {
      background: var(--hint-bg);
      border: 1px solid var(--hint-border);
      color: #b2f2bb;
      font-size: 0.8rem;
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      margin-bottom: 1rem;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.15rem;
      margin-bottom: 1rem;
    }
    .panel h2 {
      margin: 0 0 0.75rem;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
    }
    label { font-size: 0.75rem; color: var(--muted); display: block; margin-bottom: 0.3rem; }
    label .hint { font-weight: 400; color: #5c636a; }
    select, input[type="text"], input[type="number"], textarea {
      width: 100%;
      padding: 0.5rem 0.55rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #1a1b1e;
      color: var(--text);
      font-size: 0.85rem;
    }
    select {
      cursor: pointer;
      appearance: auto;
      min-height: 2.25rem;
    }
    textarea { font-family: var(--mono); font-size: 0.8rem; min-height: 4.5rem; resize: vertical; }
    /* Dark scrollbars — match theme (textarea JSON preview, pre blocks, timeline) */
    textarea,
    pre {
      scrollbar-width: thin;
      scrollbar-color: #5c636a #141517;
    }
    textarea::-webkit-scrollbar,
    pre::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    textarea::-webkit-scrollbar-corner,
    pre::-webkit-scrollbar-corner {
      background: #141517;
    }
    textarea::-webkit-scrollbar-track,
    pre::-webkit-scrollbar-track {
      background: #141517;
      border-radius: 4px;
    }
    textarea::-webkit-scrollbar-thumb,
    pre::-webkit-scrollbar-thumb {
      background: #5c636a;
      border-radius: 4px;
      border: 2px solid #141517;
    }
    textarea::-webkit-scrollbar-thumb:hover,
    pre::-webkit-scrollbar-thumb:hover {
      background: #868e96;
    }
    /* Number inputs: drop default light steppers (seed still editable) */
    input[type="number"] {
      -moz-appearance: textfield;
      appearance: textfield;
    }
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }
    .form-grid {
      display: grid;
      gap: 0.75rem;
      margin-top: 0.75rem;
    }
    @media (min-width: 560px) {
      .form-grid.cols-2 { grid-template-columns: 1fr 1fr; }
    }
    .field-group { display: none; }
    .field-group.active { display: block; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
    button {
      border: none;
      padding: 0.55rem 1rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    button.exec { background: var(--exec); color: #fff; }
    button.exec:hover:not(:disabled) { background: var(--exec-hover); }
    button.exec:disabled {
      background: #495057;
      color: #868e96;
      cursor: not-allowed;
      opacity: 0.65;
    }
    button.secondary { background: #495057; color: #fff; }
    button.secondary:hover { background: #5c636a; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.65rem;
    }
    @media (min-width: 720px) { .metrics { grid-template-columns: repeat(4, 1fr); } }
    .metric {
      border-radius: 8px;
      padding: 0.75rem 0.85rem;
      color: #fff;
    }
    .metric .m-label { font-size: 0.65rem; text-transform: uppercase; opacity: 0.9; letter-spacing: 0.06em; }
    .metric .m-val { font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
    .metric.blue { background: linear-gradient(135deg, var(--card-blue), #1c7ed6); }
    .metric.orange { background: linear-gradient(135deg, var(--card-orange), #e8590c); }
    .metric.green { background: linear-gradient(135deg, var(--card-green), #37b24d); }
    .metric.grey { background: linear-gradient(135deg, #495057, #6c757d); }
    pre {
      margin: 0;
      padding: 0.65rem 0.75rem;
      background: #1a1b1e;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 0.72rem;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .tag { font-size: 0.72rem; color: var(--muted); margin-bottom: 0.25rem; }
    #status { font-size: 0.8rem; min-height: 1.25rem; margin: 0.5rem 0; }
    #status.err { color: var(--danger); }

    /* Fixed toasts — errors visible without scrolling past the timeline */
    #toast-root {
      position: fixed;
      top: 0.75rem;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10000;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 0.45rem;
      max-width: min(38rem, calc(100vw - 1.5rem));
      pointer-events: none;
    }
    .toast {
      pointer-events: auto;
      margin: 0;
      padding: 0.65rem 1rem;
      border-radius: 8px;
      font-size: 0.82rem;
      line-height: 1.45;
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.5);
      border: 1px solid rgba(250, 82, 82, 0.45);
      background: #3b1219;
      color: #ffc9c9;
      opacity: 0;
      transform: translateY(-0.4rem);
      transition: opacity 0.22s ease, transform 0.22s ease;
      cursor: pointer;
      word-break: break-word;
    }
    .toast.toast-visible {
      opacity: 1;
      transform: translateY(0);
    }
    .toast:focus {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .json-preview label { margin-top: 0.75rem; }
    #actionJson { min-height: 5rem; opacity: 0.92; }
    footer { margin-top: 1.25rem; font-size: 0.72rem; color: var(--muted); }
    footer a { color: var(--accent); }

    /* Current ticket (reference-style card) */
    h2.section-title {
      font-size: 1rem;
      font-weight: 700;
      color: #f8f9fa;
      margin: 0 0 0.5rem;
      letter-spacing: -0.01em;
    }
    .ticket-panel {
      background: #1e1f23 !important;
      border-color: #2c2e33 !important;
      padding: 1rem 1.1rem 1.1rem !important;
    }
    .ticket-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin-bottom: 0.85rem;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.28rem 0.65rem;
      border-radius: 5px;
      background: #2f3138;
      border: 1px solid #3d4049;
      font-size: 0.72rem;
      color: #dee2e6;
    }
    .pill kbd {
      font-family: inherit;
      font-weight: 600;
      color: #adb5bd;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .pill span.val { font-weight: 600; color: #fff; }
    .pill.mode-ticket { background: #243d56; border-color: #2f5f87; }
    .pill.mode-incident { background: #4a2532; border-color: #7f344a; }
    .pill.phase-unclassified { background: #3f3f46; border-color: #5c5f66; }
    .pill.phase-classified { background: #2b3a67; border-color: #3f58a1; }
    .pill.phase-routed { background: #2f4c33; border-color: #3f7b48; }
    .pill.phase-responding { background: #4f4123; border-color: #8c6d30; }
    .pill.phase-escalated { background: #4a2b2b; border-color: #8b3f3f; }
    .pill.phase-resolved { background: #2f4f3a; border-color: #3f8f58; }
    .pill.phase-triage { background: #3d2e58; border-color: #654094; }
    .pill.phase-investigation { background: #274d5a; border-color: #2f7f96; }
    .pill.phase-response { background: #5a4220; border-color: #9c7431; }
    .pill.phase-resolution { background: #25523c; border-color: #2f8a5b; }
    .ticket-msg-wrap {
      background: #121214;
      border: 1px solid #2c2e33;
      border-radius: 6px;
      padding: 0.75rem 0.9rem 1rem;
    }
    .ticket-msg-label {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #868e96;
      margin-bottom: 0.5rem;
    }
    .ticket-msg-body {
      font-size: 0.95rem;
      line-height: 1.55;
      color: #f1f3f5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .ticket-msg-body.empty { color: #5c636a; font-style: italic; }

    /* Action timeline */
    .timeline-panel {
      background: #252830 !important;
      border-color: #3d4150 !important;
    }
    .timeline-scroll {
      max-height: 18rem;
      overflow-y: auto;
      overflow-x: hidden;
      scrollbar-gutter: stable;
      scrollbar-width: thin;
      scrollbar-color: #5c636a #141517;
    }
    .timeline-scroll::-webkit-scrollbar { width: 8px; }
    .timeline-scroll::-webkit-scrollbar-track { background: #141517; border-radius: 4px; }
    .timeline-scroll::-webkit-scrollbar-thumb {
      background: #5c636a;
      border-radius: 4px;
      border: 2px solid #141517;
    }
    .timeline-scroll::-webkit-scrollbar-thumb:hover { background: #868e96; }
    .timeline-list { display: flex; flex-direction: column; gap: 0.5rem; }
    .timeline-item {
      background: #121214;
      border: 1px solid #2c2e33;
      border-radius: 6px;
      padding: 0.65rem 0.85rem;
      overflow: hidden;
    }
    .timeline-row {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 1rem;
    }
    .tl-main { min-width: 0; flex: 1; }
    .tl-step-num {
      font-weight: 700;
      color: #e9ecef;
      font-size: 0.82rem;
    }
    .tl-action-name {
      font-size: 0.82rem;
      color: #ced4da;
      font-weight: 500;
    }
    .tl-action-chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      font-size: 0.62rem;
      padding: 0.1rem 0.45rem;
      margin-left: 0.45rem;
      border: 1px solid transparent;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
      color: #f8f9fa;
    }
    .tl-action-chip.family-diagnose { background: #21354d; border-color: #345a84; }
    .tl-action-chip.family-respond { background: #4b351a; border-color: #8e6026; }
    .tl-action-chip.family-resolve { background: #1e4c35; border-color: #2f8a5b; }
    .tl-action-chip.family-governance { background: #47264d; border-color: #7e4090; }
    .tl-action-chip.family-legacy { background: #3f3f46; border-color: #5e6068; }
    .tl-reward {
      font-size: 0.85rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      color: #fff;
      flex-shrink: 0;
    }
    .tl-reward.neg { color: #ff8787; }
    .tl-reward.pos { color: #8ce99a; }
    .tl-feedback {
      margin-top: 0.45rem;
      padding-top: 0.45rem;
      border-top: 1px solid #2c2e33;
      font-size: 0.72rem;
      line-height: 1.4;
      color: #868e96;
    }
    .timeline-empty {
      text-align: center;
      padding: 1.25rem 0.75rem;
      color: #868e96;
      font-size: 0.82rem;
      background: #121214;
      border-radius: 6px;
      border: 1px dashed #3d4049;
    }

    .page-head {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      justify-content: space-between;
      gap: 0.75rem;
      margin-bottom: 0.25rem;
    }
    .page-head-text { flex: 1; min-width: 12rem; }
    .page-head-text h1 { margin-bottom: 0.25rem; }
    .page-head-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      align-items: center;
    }
    a.btn-mini {
      display: inline-block;
      padding: 0.35rem 0.65rem;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
      text-decoration: none;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #fdfd96;
      color: #111;
      border: none;
      cursor: pointer;
      font-family: inherit;
      line-height: 1.2;
    }
    a.btn-mini:hover { filter: brightness(1.06); }
    a.btn-mini:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .quick-presets {
      margin-top: 0.85rem;
      border: 1px dashed #3f444f;
      border-radius: 8px;
      padding: 0.65rem 0.75rem;
      background: #1a1b1e;
    }
    .preset-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-top: 0.3rem;
    }
    .preset-btn {
      border: 1px solid #3d4150;
      background: #2a2d35;
      color: #e9ecef;
      padding: 0.28rem 0.55rem;
      border-radius: 999px;
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: lowercase;
      letter-spacing: 0.01em;
      cursor: pointer;
    }
    .preset-btn:hover:not(:disabled) {
      border-color: #5c7cfa;
      background: #30374a;
    }
    .preset-btn:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
  </style>
</head>
<body>
  <div id="toast-root" aria-live="assertive" aria-relevant="additions"></div>
  <div class="wrap">
    <header class="page-head">
      <div class="page-head-text">
        <h1>Enterprise Incident Command Center</h1>
        <p class="sub">Ticket triage + incident operations in one console ·
          <a href="/health" style="color:var(--accent)">/health</a></p>
      </div>
      <div class="page-head-actions">
        <a class="btn-mini" href="/docs" target="_blank" rel="noopener noreferrer">Docs</a>
      </div>
    </header>

    <div class="metrics panel" style="padding:0.85rem;margin-bottom:1rem;background:#1a1b1e;border-style:dashed">
      <div class="metric blue">
        <div class="m-label">Last step reward</div>
        <div class="m-val" id="mLastReward">—</div>
      </div>
      <div class="metric orange">
        <div class="m-label">Total reward</div>
        <div class="m-val" id="mTotalReward">—</div>
      </div>
      <div class="metric green">
        <div class="m-label">Score</div>
        <div class="m-val" id="mScore">—</div>
      </div>
      <div class="metric grey">
        <div class="m-label">Status</div>
        <div class="m-val" id="mStatus" style="font-size:1rem">—</div>
      </div>
    </div>

    <h2 class="section-title">Current Context</h2>
    <div class="panel ticket-panel">
      <div class="ticket-pills" id="ticketPills"></div>
      <div class="ticket-msg-wrap">
        <div class="ticket-msg-label">Customer Message</div>
        <div class="ticket-msg-body empty" id="ticketBody">Reset or load state to show the active ticket.</div>
      </div>
    </div>

    <div class="panel">
      <div class="hint-bar">
        Choose <strong>mode</strong> on reset: <strong>ticket</strong> for legacy triage or <strong>incident</strong> for EICC workflow.
        In sandbox mode, optional <strong>drill mode</strong> injects deterministic mid-episode failures.
        Use <strong>Get state</strong> to inspect <code>available_actions</code>; unavailable actions are auto-disabled after reset/step.
      </div>
      <h2>Step-by-step action</h2>
      <label for="actionType">Action type <span class="hint">(matches POST /step JSON)</span></label>
      <select id="actionType">
        <optgroup label="Ticket Actions">
          <option value="classify">classify</option>
          <option value="route">route</option>
          <option value="respond">respond</option>
          <option value="escalate">escalate</option>
          <option value="request_info">request_info</option>
          <option value="resolve">resolve</option>
        </optgroup>
        <optgroup label="Incident Investigation">
          <option value="check_monitoring">check_monitoring</option>
          <option value="probe_service">probe_service</option>
          <option value="fetch_logs">fetch_logs</option>
          <option value="fetch_user_data">fetch_user_data</option>
          <option value="check_billing">check_billing</option>
          <option value="query_kb">query_kb</option>
          <option value="check_policy">check_policy</option>
          <option value="query_incident_history">query_incident_history</option>
          <option value="follow_runbook_step">follow_runbook_step</option>
        </optgroup>
        <optgroup label="Incident Response & Resolution">
          <option value="apply_fix">apply_fix</option>
          <option value="verify_fix">verify_fix</option>
          <option value="rollback_fix">rollback_fix</option>
          <option value="notify_stakeholders">notify_stakeholders</option>
          <option value="write_postmortem">write_postmortem</option>
          <option value="update_kb">update_kb</option>
        </optgroup>
      </select>

      <div id="fgClassify" class="field-group form-grid cols-2">
        <div>
          <label for="clsCategory">Category</label>
          <select id="clsCategory">
            <option value="billing">billing</option>
            <option value="bug_report">bug_report</option>
            <option value="feature_request">feature_request</option>
            <option value="account_access">account_access</option>
            <option value="general_inquiry" selected>general_inquiry</option>
            <option value="cancellation">cancellation</option>
          </select>
        </div>
        <div>
          <label for="clsPriority">Priority</label>
          <select id="clsPriority">
            <option value="low">low</option>
            <option value="medium" selected>medium</option>
            <option value="high">high</option>
            <option value="critical">critical</option>
          </select>
        </div>
      </div>

      <div id="fgRoute" class="field-group form-grid">
        <div>
          <label for="rteDept">Department</label>
          <select id="rteDept">
            <option value="billing">billing</option>
            <option value="technical">technical</option>
            <option value="account">account</option>
            <option value="general" selected>general</option>
          </select>
        </div>
      </div>

      <div id="fgRespond" class="field-group form-grid">
        <div>
          <label for="rspTone">Tone</label>
          <select id="rspTone">
            <option value="formal" selected>formal</option>
            <option value="empathetic">empathetic</option>
            <option value="concise">concise</option>
          </select>
        </div>
        <div style="grid-column:1/-1">
          <label for="rspText">Response text</label>
          <textarea id="rspText" spellcheck="false">Thank you for contacting us. We are reviewing your request.</textarea>
        </div>
      </div>

      <div id="fgEscalate" class="field-group form-grid cols-2">
        <div>
          <label for="escTeam">Target team</label>
          <select id="escTeam">
            <option value="l2_support" selected>l2_support</option>
            <option value="engineering">engineering</option>
            <option value="management">management</option>
          </select>
        </div>
        <div style="grid-column:1/-1">
          <label for="escReason">Reason</label>
          <textarea id="escReason" spellcheck="false">Requires specialist review per policy.</textarea>
        </div>
      </div>

      <div id="fgRequestInfo" class="field-group form-grid">
        <div>
          <label for="reqQ">Question to customer</label>
          <textarea id="reqQ" spellcheck="false">Could you share your account email and approximate time of the issue?</textarea>
        </div>
      </div>

      <div id="fgResolve" class="field-group form-grid">
        <div style="grid-column:1/-1">
          <label for="rsvSummary">Resolution summary</label>
          <textarea id="rsvSummary" spellcheck="false">Issue reviewed and resolved to customer satisfaction.</textarea>
        </div>
        <div>
          <label for="rsvComp">Offered compensation <span class="hint">(optional, empty = omit)</span></label>
          <input type="number" id="rsvComp" step="any" placeholder="e.g. 29.99" />
        </div>
      </div>

      <div id="fgCheckMonitoring" class="field-group form-grid">
        <div>
          <label for="monService">Service name <span class="hint">(optional)</span></label>
          <select id="monService">
            <option value="">all services</option>
            <option value="auth">auth</option>
            <option value="database">database</option>
            <option value="payments">payments</option>
            <option value="analytics">analytics</option>
            <option value="notifications">notifications</option>
          </select>
        </div>
      </div>

      <div id="fgProbeService" class="field-group form-grid cols-2">
        <div>
          <label for="probeService">Service name</label>
          <select id="probeService">
            <option value="auth">auth</option>
            <option value="database">database</option>
            <option value="payments" selected>payments</option>
            <option value="analytics">analytics</option>
            <option value="notifications">notifications</option>
          </select>
        </div>
        <div>
          <label for="probeType">Check type</label>
          <select id="probeType">
            <option value="logs">logs</option>
            <option value="resources">resources</option>
            <option value="connections" selected>connections</option>
            <option value="config">config</option>
          </select>
        </div>
      </div>

      <div id="fgFetchLogs" class="field-group form-grid cols-2">
        <div>
          <label for="logsService">Service name</label>
          <select id="logsService">
            <option value="auth">auth</option>
            <option value="database">database</option>
            <option value="payments" selected>payments</option>
            <option value="analytics">analytics</option>
            <option value="notifications">notifications</option>
          </select>
        </div>
        <div>
          <label for="logsRange">Time range</label>
          <select id="logsRange">
            <option value="last_5m">last_5m</option>
            <option value="last_15m" selected>last_15m</option>
            <option value="last_1h">last_1h</option>
          </select>
        </div>
      </div>

      <div id="fgFetchUserData" class="field-group form-grid">
        <div>
          <label for="userCustomerId">Customer ID</label>
          <input type="text" id="userCustomerId" value="cust-acme-001" />
        </div>
      </div>

      <div id="fgCheckBilling" class="field-group form-grid">
        <div>
          <label for="billingCustomerId">Customer ID</label>
          <input type="text" id="billingCustomerId" value="cust-acme-001" />
        </div>
      </div>

      <div id="fgQueryKb" class="field-group form-grid">
        <div>
          <label for="kbQuery">KB query</label>
          <textarea id="kbQuery" spellcheck="false">payment timeout after auth token refresh</textarea>
        </div>
      </div>

      <div id="fgCheckPolicy" class="field-group form-grid">
        <div>
          <label for="policyType">Policy type</label>
          <select id="policyType">
            <option value="refund">refund</option>
            <option value="escalation">escalation</option>
            <option value="sla" selected>sla</option>
            <option value="compensation">compensation</option>
            <option value="communication">communication</option>
          </select>
        </div>
      </div>

      <div id="fgQueryIncidentHistory" class="field-group form-grid cols-2">
        <div style="grid-column:1/-1">
          <label for="historyQuery">History query</label>
          <textarea id="historyQuery" spellcheck="false">payments failing after auth token cache expiry</textarea>
        </div>
        <div>
          <label for="historyService">Service filter <span class="hint">(optional)</span></label>
          <select id="historyService">
            <option value="">none</option>
            <option value="auth">auth</option>
            <option value="database">database</option>
            <option value="payments" selected>payments</option>
            <option value="analytics">analytics</option>
            <option value="notifications">notifications</option>
          </select>
        </div>
      </div>

      <div id="fgFollowRunbookStep" class="field-group form-grid cols-2">
        <div>
          <label for="runbookId">Runbook ID</label>
          <input type="text" id="runbookId" value="rb-payment-cascade" />
        </div>
        <div>
          <label for="runbookStepIdx">Step index</label>
          <input type="number" id="runbookStepIdx" value="0" step="1" />
        </div>
      </div>

      <div id="fgApplyFix" class="field-group form-grid cols-2">
        <div>
          <label for="fixService">Service name</label>
          <select id="fixService">
            <option value="auth">auth</option>
            <option value="database">database</option>
            <option value="payments" selected>payments</option>
            <option value="analytics">analytics</option>
            <option value="notifications">notifications</option>
          </select>
        </div>
        <div>
          <label for="fixType">Fix type</label>
          <input type="text" id="fixType" value="clear_token_cache" />
        </div>
      </div>

      <div id="fgVerifyFix" class="field-group form-grid">
        <div>
          <label for="verifyService">Service name</label>
          <select id="verifyService">
            <option value="auth">auth</option>
            <option value="database">database</option>
            <option value="payments" selected>payments</option>
            <option value="analytics">analytics</option>
            <option value="notifications">notifications</option>
          </select>
        </div>
      </div>

      <div id="fgRollbackFix" class="field-group form-grid">
        <div>
          <label for="rollbackService">Service name</label>
          <select id="rollbackService">
            <option value="auth">auth</option>
            <option value="database">database</option>
            <option value="payments" selected>payments</option>
            <option value="analytics">analytics</option>
            <option value="notifications">notifications</option>
          </select>
        </div>
      </div>

      <div id="fgNotifyStakeholders" class="field-group form-grid cols-2">
        <div>
          <label for="notifyStakeholder">Stakeholder</label>
          <select id="notifyStakeholder">
            <option value="vp_engineering" selected>vp_engineering</option>
            <option value="legal">legal</option>
            <option value="support_lead">support_lead</option>
            <option value="all">all</option>
          </select>
        </div>
        <div>
          <label for="notifyUrgency">Urgency</label>
          <select id="notifyUrgency">
            <option value="info">info</option>
            <option value="warning" selected>warning</option>
            <option value="critical">critical</option>
          </select>
        </div>
        <div style="grid-column:1/-1">
          <label for="notifyMessage">Message</label>
          <textarea id="notifyMessage" spellcheck="false">Investigation in progress. Mitigation underway and customer impact is being contained.</textarea>
        </div>
      </div>

      <div id="fgWritePostmortem" class="field-group form-grid cols-2">
        <div style="grid-column:1/-1">
          <label for="pmSummary">Summary</label>
          <textarea id="pmSummary" spellcheck="false">Payments requests failed intermittently due to stale auth token cache, triggering cascading retries.</textarea>
        </div>
        <div style="grid-column:1/-1">
          <label for="pmRootCause">Root cause description</label>
          <textarea id="pmRootCause" spellcheck="false">Auth token cache was not invalidated after key rotation, causing downstream authorization failures.</textarea>
        </div>
        <div>
          <label for="pmRemediation">Remediation steps <span class="hint">(comma or newline separated)</span></label>
          <textarea id="pmRemediation" spellcheck="false">flush token cache, restart auth workers, verify success metrics</textarea>
        </div>
        <div>
          <label for="pmPrevention">Prevention measures <span class="hint">(comma or newline separated)</span></label>
          <textarea id="pmPrevention" spellcheck="false">add key-rotation canary, alert on auth mismatch, runbook update</textarea>
        </div>
      </div>

      <div id="fgUpdateKb" class="field-group form-grid cols-2">
        <div style="grid-column:1/-1">
          <label for="kbTitle">Article title</label>
          <input type="text" id="kbTitle" value="Fixing payment failures after auth key rotation" />
        </div>
        <div style="grid-column:1/-1">
          <label for="kbContent">Article content</label>
          <textarea id="kbContent" spellcheck="false">If payment errors spike after key rotation, check auth token cache consistency, flush cache, and verify service health before rollback.</textarea>
        </div>
        <div style="grid-column:1/-1">
          <label for="kbTags">Tags <span class="hint">(comma separated)</span></label>
          <input type="text" id="kbTags" value="payments, auth, incident-response" />
        </div>
      </div>

      <div class="json-preview">
        <label for="actionJson">Request body preview <code>{"action": …}</code></label>
        <textarea id="actionJson" readonly spellcheck="false"></textarea>
      </div>

      <div class="form-grid cols-2" style="margin-top:0.85rem">
        <div>
          <label for="resetMode">Reset · mode</label>
          <select id="resetMode">
            <option value="ticket" selected>ticket</option>
            <option value="incident">incident</option>
          </select>
        </div>
        <div>
          <label for="resetDifficulty">Reset · difficulty</label>
          <select id="resetDifficulty">
            <option value="">Any</option>
            <option value="easy">easy only</option>
            <option value="medium">medium only</option>
            <option value="hard">hard only</option>
            <option value="nightmare">nightmare (incident only)</option>
          </select>
        </div>
        <div>
          <label for="resetSeed">Reset · seed</label>
          <input type="number" id="resetSeed" value="0" step="1" />
        </div>
        <div>
          <label for="resetDrillMode">Reset · drill mode (sandbox)</label>
          <select id="resetDrillMode">
            <option value="false" selected>disabled</option>
            <option value="true">enabled</option>
          </select>
        </div>
        <div>
          <label for="resetDrillSeed">Reset · drill seed <span class="hint">(optional)</span></label>
          <input type="number" id="resetDrillSeed" step="1" placeholder="uses reset seed if empty" />
        </div>
      </div>

      <div class="btn-row">
        <button type="button" class="exec" id="btnStep">Execute step</button>
        <button type="button" class="secondary" id="btnReset">Reset</button>
        <button type="button" class="secondary" id="btnState">Get state</button>
      </div>
      <div class="quick-presets">
        <div class="tag">quick actions for current phase</div>
        <div class="preset-row" id="presetRow">
          <button type="button" class="preset-btn" disabled>reset to load suggestions</button>
        </div>
      </div>
    </div>

    <h2 class="section-title">Action Timeline</h2>
    <div class="panel timeline-panel">
      <div class="timeline-scroll">
        <div class="timeline-list" id="timelineList">
          <div class="timeline-empty">No actions yet. Run <strong>Reset</strong>, then <strong>Execute step</strong>.</div>
        </div>
      </div>
    </div>

    <p id="status"></p>

    <div class="panel">
      <div class="tag">observation</div>
      <pre id="outObs">—</pre>
    </div>
    <div class="panel grid" style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
      <div>
        <div class="tag">reward (response)</div>
        <pre id="outReward">—</pre>
      </div>
      <div>
        <div class="tag">done</div>
        <pre id="outDone">—</pre>
      </div>
    </div>
    <div class="panel">
      <div class="tag">info</div>
      <pre id="outInfo">—</pre>
    </div>

    <footer>Built from dropdowns → <code>POST /step</code>. Toggle action type to edit fields.</footer>
  </div>
  <script>
(function () {
  const $ = (id) => document.getElementById(id);
  const toastRoot = $("toast-root");
  const status = $("status");
  const outObs = $("outObs");
  const outReward = $("outReward");
  const outDone = $("outDone");
  const outInfo = $("outInfo");
  const mLast = $("mLastReward");
  const mTotal = $("mTotalReward");
  const mScore = $("mScore");
  const mStat = $("mStatus");
  const presetRow = $("presetRow");

  let totalReward = 0;
  let episodeActive = false;
  let sessionId = null;

  function syncStepButton() {
    $("btnStep").disabled = !episodeActive;
  }

  function pretty(obj) {
    return JSON.stringify(obj, null, 2);
  }

  function requestHeaders(includeJson) {
    const headers = {};
    if (includeJson) headers["Content-Type"] = "application/json";
    if (sessionId) headers["X-Session-ID"] = sessionId;
    return headers;
  }

  function parseListInput(raw) {
    if (!raw) return [];
    return raw
      .split(/\\n|,/)
      .map((part) => part.trim())
      .filter(Boolean);
  }

  function buildAction() {
    const t = $("actionType").value;
    switch (t) {
      case "classify":
        return {
          action_type: "classify",
          category: $("clsCategory").value,
          priority: $("clsPriority").value,
        };
      case "route":
        return { action_type: "route", department: $("rteDept").value };
      case "respond":
        return {
          action_type: "respond",
          response_text: $("rspText").value.trim() || " ",
          tone: $("rspTone").value,
        };
      case "escalate":
        return {
          action_type: "escalate",
          reason: $("escReason").value.trim() || "Escalation.",
          target_team: $("escTeam").value,
        };
      case "request_info":
        return {
          action_type: "request_info",
          question_to_customer: $("reqQ").value.trim() || "Please provide more details.",
        };
      case "resolve": {
        const raw = $("rsvComp").value.trim();
        let comp = null;
        if (raw !== "") {
          const n = Number(raw);
          comp = Number.isFinite(n) ? n : null;
        }
        const o = {
          action_type: "resolve",
          resolution_summary: $("rsvSummary").value.trim() || "Resolved.",
        };
        if (comp !== null) o.offered_compensation = comp;
        return o;
      }
      case "check_monitoring": {
        const service = $("monService").value;
        const payload = { action_type: "check_monitoring" };
        if (service) payload.service_name = service;
        return payload;
      }
      case "probe_service":
        return {
          action_type: "probe_service",
          service_name: $("probeService").value,
          check_type: $("probeType").value,
        };
      case "fetch_logs":
        return {
          action_type: "fetch_logs",
          service_name: $("logsService").value,
          time_range: $("logsRange").value,
        };
      case "fetch_user_data":
        return {
          action_type: "fetch_user_data",
          customer_id: $("userCustomerId").value.trim() || "cust-acme-001",
        };
      case "check_billing":
        return {
          action_type: "check_billing",
          customer_id: $("billingCustomerId").value.trim() || "cust-acme-001",
        };
      case "query_kb":
        return {
          action_type: "query_kb",
          query: $("kbQuery").value.trim() || "payment timeout",
        };
      case "check_policy":
        return {
          action_type: "check_policy",
          policy_type: $("policyType").value,
        };
      case "query_incident_history": {
        const payload = {
          action_type: "query_incident_history",
          query: $("historyQuery").value.trim() || "payments incident",
        };
        const service = $("historyService").value;
        if (service) payload.service_filter = service;
        return payload;
      }
      case "follow_runbook_step": {
        const stepRaw = parseInt($("runbookStepIdx").value, 10);
        return {
          action_type: "follow_runbook_step",
          runbook_id: $("runbookId").value.trim() || "rb-payment-cascade",
          step_index: Number.isFinite(stepRaw) ? stepRaw : 0,
        };
      }
      case "apply_fix":
        return {
          action_type: "apply_fix",
          service_name: $("fixService").value,
          fix_type: $("fixType").value.trim() || "clear_token_cache",
        };
      case "verify_fix":
        return {
          action_type: "verify_fix",
          service_name: $("verifyService").value,
        };
      case "rollback_fix":
        return {
          action_type: "rollback_fix",
          service_name: $("rollbackService").value,
        };
      case "notify_stakeholders":
        return {
          action_type: "notify_stakeholders",
          stakeholder: $("notifyStakeholder").value,
          message: $("notifyMessage").value.trim() || "Incident update in progress.",
          urgency: $("notifyUrgency").value,
        };
      case "write_postmortem":
        return {
          action_type: "write_postmortem",
          summary: $("pmSummary").value.trim() || "Incident summary pending.",
          root_cause_description: $("pmRootCause").value.trim() || "Root cause pending.",
          remediation_steps: parseListInput($("pmRemediation").value),
          prevention_measures: parseListInput($("pmPrevention").value),
        };
      case "update_kb":
        return {
          action_type: "update_kb",
          article_title: $("kbTitle").value.trim() || "Incident update",
          content: $("kbContent").value.trim() || "Pending content.",
          tags: parseListInput($("kbTags").value),
        };
      default:
        return { action_type: "classify", category: "general_inquiry", priority: "medium" };
    }
  }

  function syncJsonPreview() {
    $("actionJson").value = pretty(buildAction());
  }

  function showFieldGroups() {
    const t = $("actionType").value;
    document.querySelectorAll(".field-group").forEach((el) => {
      el.classList.toggle("active", el.id === "fg" + capitalize(t));
    });
    syncJsonPreview();
  }

  function capitalize(s) {
    return s.split("_").map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join("");
  }

  ["actionType", "clsCategory", "clsPriority", "rteDept", "rspTone", "rspText",
   "escTeam", "escReason", "reqQ", "rsvSummary", "rsvComp", "monService",
   "probeService", "probeType", "logsService", "logsRange", "userCustomerId",
   "billingCustomerId", "kbQuery", "policyType", "historyQuery", "historyService",
   "runbookId", "runbookStepIdx", "fixService", "fixType", "verifyService",
   "rollbackService", "notifyStakeholder", "notifyMessage", "notifyUrgency",
   "pmSummary", "pmRootCause", "pmRemediation", "pmPrevention", "kbTitle",
   "kbContent", "kbTags", "resetDrillMode", "resetDrillSeed"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("input", syncJsonPreview);
    if (el) el.addEventListener("change", syncJsonPreview);
  });
  $("actionType").addEventListener("change", showFieldGroups);

  function actionSummary(actionTaken) {
    if (!actionTaken) return "—";
    try {
      const o = JSON.parse(actionTaken);
      if (o && typeof o.action_type === "string") return o.action_type;
    } catch (e) { /* ignore */ }
    return actionTaken.length > 56 ? actionTaken.slice(0, 53) + "…" : actionTaken;
  }

  function phaseClass(phase) {
    if (!phase) return "";
    return "phase-" + String(phase).toLowerCase();
  }

  function extractActionType(actionTaken) {
    if (!actionTaken) return "";
    try {
      const parsed = JSON.parse(actionTaken);
      if (parsed && typeof parsed.action_type === "string") return parsed.action_type;
    } catch (e) { /* ignore */ }
    return String(actionTaken).trim();
  }

  function actionFamily(actionType) {
    const diagnose = new Set([
      "check_monitoring", "probe_service", "fetch_logs", "fetch_user_data", "check_billing",
      "query_kb", "check_policy", "query_incident_history", "follow_runbook_step",
    ]);
    const respond = new Set(["apply_fix", "rollback_fix", "notify_stakeholders", "respond", "escalate", "request_info"]);
    const resolve = new Set(["verify_fix", "resolve", "write_postmortem", "update_kb"]);
    if (diagnose.has(actionType)) return "diagnose";
    if (respond.has(actionType)) return "respond";
    if (resolve.has(actionType)) return "resolve";
    return "legacy";
  }

  function renderPresets(obs) {
    if (!presetRow) return;
    presetRow.innerHTML = "";
    const fallback = ["classify", "route", "respond", "resolve"];
    let suggestions = fallback;
    if (obs && obs.mode === "incident") {
      const byPhase = {
        triage: ["check_monitoring", "query_kb", "query_incident_history", "follow_runbook_step"],
        investigation: ["probe_service", "fetch_logs", "check_policy", "fetch_user_data"],
        response: ["apply_fix", "notify_stakeholders", "rollback_fix", "respond"],
        resolution: ["verify_fix", "resolve", "write_postmortem", "update_kb"],
      };
      suggestions = byPhase[obs.incident_phase] || byPhase.triage;
    } else if (obs) {
      const byTicketPhase = {
        unclassified: ["classify"],
        classified: ["route", "escalate"],
        routed: ["respond", "request_info", "resolve"],
        responding: ["respond", "request_info", "resolve"],
        escalated: ["resolve"],
      };
      suggestions = byTicketPhase[obs.phase] || fallback;
    }

    const available = obs && Array.isArray(obs.available_actions)
      ? new Set(obs.available_actions)
      : null;
    suggestions.forEach((action) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-btn";
      btn.textContent = action;
      btn.disabled = available ? !available.has(action) : false;
      btn.addEventListener("click", () => {
        const selector = $("actionType");
        selector.value = action;
        showFieldGroups();
      });
      presetRow.appendChild(btn);
    });
  }

  function renderTicket(obs, info) {
    const body = $("ticketBody");
    const pills = $("ticketPills");
    pills.innerHTML = "";
    if (!obs) {
      body.textContent = "No observation loaded.";
      body.classList.add("empty");
      return;
    }
    body.classList.remove("empty");
    body.textContent = obs.ticket_text != null ? obs.ticket_text : "";

    function addPill(label, value, variant) {
      if (value === undefined || value === null || value === "") return;
      const p = document.createElement("div");
      p.className = "pill" + (variant ? " " + variant : "");
      const k = document.createElement("kbd");
      k.textContent = label + ":";
      const v = document.createElement("span");
      v.className = "val";
      v.textContent = String(value);
      p.appendChild(k);
      p.appendChild(v);
      pills.appendChild(p);
    }

    if (obs.mode) addPill("Mode", obs.mode, "mode-" + obs.mode);
    if (info && info.difficulty) addPill("Difficulty", String(info.difficulty));
    const impact = (obs.customer_value || "").toString().toUpperCase();
    addPill("Impact", impact);
    addPill("ID", obs.ticket_id);
    addPill("Tier", obs.customer_tier);
    addPill("Sentiment", obs.customer_sentiment);
    if (obs.mode === "incident" && obs.incident_phase) {
      addPill("Incident Phase", obs.incident_phase, phaseClass(obs.incident_phase));
    } else {
      addPill("Phase", obs.phase, phaseClass(obs.phase));
    }
    if (typeof obs.sla_steps_remaining === "number") {
      addPill("SLA steps left", String(obs.sla_steps_remaining));
    }
    if (obs.mode === "incident" && obs.system_status) {
      let degraded = 0;
      let down = 0;
      let flickering = 0;
      Object.values(obs.system_status).forEach((state) => {
        if (state === "degraded") degraded += 1;
        if (state === "down") down += 1;
        if (state === "flickering") flickering += 1;
      });
      addPill("Service Alerts", "D:" + degraded + " F:" + flickering + " X:" + down);
    }
    if (obs.category_hint) addPill("Hint", obs.category_hint);
  }

  function renderTimeline(obs) {
    const list = $("timelineList");
    list.innerHTML = "";
    if (!obs || !Array.isArray(obs.history) || obs.history.length === 0) {
      const empty = document.createElement("div");
      empty.className = "timeline-empty";
      empty.innerHTML = "No actions yet. Run <strong>Reset</strong>, then <strong>Execute step</strong>.";
      list.appendChild(empty);
      return;
    }
    obs.history.forEach(function (h) {
      const item = document.createElement("div");
      item.className = "timeline-item";

      const row = document.createElement("div");
      row.className = "timeline-row";

      const main = document.createElement("div");
      main.className = "tl-main";
      const sn = document.createElement("span");
      sn.className = "tl-step-num";
      sn.textContent = "Step " + h.step + ": ";
      const an = document.createElement("span");
      an.className = "tl-action-name";
      const actionType = extractActionType(h.action_taken);
      an.textContent = actionType || actionSummary(h.action_taken);
      main.appendChild(sn);
      main.appendChild(an);
      const chip = document.createElement("span");
      chip.className = "tl-action-chip family-" + actionFamily(actionType);
      chip.textContent = actionFamily(actionType);
      main.appendChild(chip);

      const rw = document.createElement("div");
      rw.className = "tl-reward";
      const r = Number(h.reward_earned);
      rw.textContent = (r >= 0 ? "+" : "") + r.toFixed(2);
      if (r < 0) rw.classList.add("neg");
      else if (r > 0) rw.classList.add("pos");

      row.appendChild(main);
      row.appendChild(rw);
      item.appendChild(row);

      if (h.env_feedback) {
        const fb = document.createElement("div");
        fb.className = "tl-feedback";
        fb.textContent = h.env_feedback;
        item.appendChild(fb);
      }

      list.appendChild(item);
    });
  }

  function syncActionAvailability(obs) {
    const select = $("actionType");
    if (!select) return;
    const available = obs && Array.isArray(obs.available_actions)
      ? new Set(obs.available_actions)
      : null;
    Array.from(select.options).forEach((opt) => {
      if (!opt.value) return;
      opt.disabled = available ? !available.has(opt.value) : false;
    });
    if (available && !available.has(select.value)) {
      const fallback = Array.from(select.options).find((opt) => opt.value && !opt.disabled);
      if (fallback) {
        select.value = fallback.value;
      }
    }
    showFieldGroups();
  }

  function syncDifficultyByMode() {
    const mode = $("resetMode").value;
    const difficulty = $("resetDifficulty");
    const drillMode = $("resetDrillMode");
    const drillSeed = $("resetDrillSeed");
    const nightmare = difficulty.querySelector('option[value="nightmare"]');
    if (nightmare) nightmare.disabled = mode !== "incident";
    if (mode !== "incident" && difficulty.value === "nightmare") {
      difficulty.value = "";
    }
    if (drillMode) {
      drillMode.disabled = mode !== "incident";
      if (mode !== "incident") {
        drillMode.value = "false";
      }
    }
    if (drillSeed) {
      drillSeed.disabled = mode !== "incident";
      if (mode !== "incident") {
        drillSeed.value = "";
      }
    }
  }

  function showPayload(data, opts) {
    const obs = data.observation;
    outObs.textContent = obs == null ? "null" : pretty(obs);
    outReward.textContent = String(data.reward);
    outDone.textContent = String(data.done);
    outInfo.textContent = pretty(data.info != null ? data.info : {});
    if (data.info && typeof data.info.session_id === "string") {
      sessionId = data.info.session_id;
    }

    renderTicket(obs, data.info);
    renderTimeline(obs);
    syncActionAvailability(obs);
    renderPresets(obs);

    mLast.textContent = Number(data.reward).toFixed(2);
    if (opts && opts.addStepReward && typeof data.reward === "number") {
      totalReward += data.reward;
    }
    mTotal.textContent = totalReward.toFixed(2);
    const ns = data.info && typeof data.info.normalized_score === "number"
      ? data.info.normalized_score
      : null;
    mScore.textContent = ns == null ? "—" : Math.round(ns * 100) + "%";
    mStat.textContent = data.done ? "DONE" : "RUNNING";
  }

  function showErrorToast(msg) {
    if (!msg || !toastRoot) return;
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = msg;
    el.setAttribute("role", "alert");
    el.tabIndex = 0;
    el.title = "Click to dismiss";
    toastRoot.appendChild(el);
    requestAnimationFrame(function () {
      el.classList.add("toast-visible");
    });
    function dismiss() {
      el.classList.remove("toast-visible");
      setTimeout(function () {
        el.remove();
      }, 280);
    }
    const t = window.setTimeout(dismiss, 8500);
    el.addEventListener("click", function () {
      window.clearTimeout(t);
      dismiss();
    });
  }

  function setStatus(msg, isErr) {
    status.textContent = msg || "";
    status.className = isErr ? "err" : "";
    if (isErr && msg) showErrorToast(msg);
  }

  async function parseJsonResponse(res) {
    const text = await res.text();
    try {
      return { ok: res.ok, data: JSON.parse(text), raw: text };
    } catch {
      return { ok: res.ok, data: null, raw: text };
    }
  }

  async function doReset() {
    setStatus("POST /reset …");
    const seedRaw = parseInt($("resetSeed").value, 10);
    const seed = Number.isFinite(seedRaw) ? seedRaw : 0;
    const diff = $("resetDifficulty").value;
    const drillEnabled = $("resetDrillMode").value === "true";
    const drillSeedRaw = $("resetDrillSeed").value.trim();
    const drillSeed = drillSeedRaw === "" ? null : parseInt(drillSeedRaw, 10);
    if (!sessionId) {
      sessionId = "ui-" + Math.random().toString(36).slice(2, 10);
    }
    const payload = { seed: seed, mode: $("resetMode").value, session_id: sessionId };
    if (diff) payload.difficulty = diff;
    if (payload.mode === "incident" && drillEnabled) {
      payload.drill_mode = true;
      if (Number.isFinite(drillSeed)) payload.drill_seed = drillSeed;
    }
    const res = await fetch("/reset", {
      method: "POST",
      headers: requestHeaders(true),
      body: JSON.stringify(payload),
    });
    const { ok, data, raw } = await parseJsonResponse(res);
    if (!ok || !data) {
      setStatus((data && data.detail) || raw || res.statusText, true);
      return;
    }
    totalReward = 0;
    episodeActive = true;
    syncStepButton();
    showPayload(data, {});
    mLast.textContent = "0.00";
    mTotal.textContent = "0.00";
    setStatus("POST /reset → " + res.status);
  }

  async function doState() {
    setStatus("GET /state …");
    const res = await fetch("/state", { method: "GET", headers: requestHeaders(false) });
    const { ok, data, raw } = await parseJsonResponse(res);
    if (!ok || !data) {
      setStatus((data && data.detail) || raw || res.statusText, true);
      return;
    }
    if (data.observation != null) {
      episodeActive = !data.done;
    }
    syncStepButton();
    showPayload(data, {});
    setStatus("GET /state → " + res.status);
  }

  async function doStep() {
    if (!episodeActive) {
      setStatus("Call Reset before executing a step.", true);
      return;
    }
    const action = buildAction();
    syncJsonPreview();
    setStatus("POST /step …");
    const res = await fetch("/step", {
      method: "POST",
      headers: requestHeaders(true),
      body: JSON.stringify({ action: action, session_id: sessionId }),
    });
    const { ok, data, raw } = await parseJsonResponse(res);
    if (!ok || !data) {
      const detail = data && (data.detail || data.message);
      setStatus(String(detail || raw || res.statusText), true);
      return;
    }
    if (data.done) {
      episodeActive = false;
      syncStepButton();
    }
    showPayload(data, { addStepReward: true });
    setStatus("POST /step → " + res.status);
  }

  $("btnReset").addEventListener("click", () => { doReset().catch((e) => setStatus(String(e), true)); });
  $("btnState").addEventListener("click", () => { doState().catch((e) => setStatus(String(e), true)); });
  $("btnStep").addEventListener("click", () => { doStep().catch((e) => setStatus(String(e), true)); });
  $("resetMode").addEventListener("change", () => {
    syncDifficultyByMode();
    syncJsonPreview();
  });

  syncDifficultyByMode();
  showFieldGroups();
  syncStepButton();
})();
  </script>
</body>
</html>
"""


# ── routes ───────────────────────────────────────────────────────────────────


@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["Interface"],
    summary="Web Command Center",
)
def ui() -> HTMLResponse:
    """Browser debug UI (HTML). Use **GET /** in the browser; API clients should use JSON endpoints below."""
    return HTMLResponse(content=_DEBUG_UI_HTML)


@app.get("/health", tags=["System"], summary="Health check")
async def health() -> dict[str, str]:
    """Return `{\"status\": \"ok\"}` when the server is up."""
    return {"status": "ok"}


@app.post(
    "/reset",
    response_model=EnvResponse,
    tags=["Environment"],
    summary="Reset episode",
    response_description="Initial observation, zero reward, `done` false.",
    responses={
        400: {"description": "Bad request (e.g. empty ticket pool for the given filter)."},
    },
)
async def reset(
    req: ResetRequest | None = None,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
) -> dict[str, Any]:
    """Start a new episode with a deterministically selected ticket. Body is optional (`seed=0`, all difficulties)."""
    if req is None:
        req = ResetRequest()
    session_id = _resolve_session_id(req.session_id, x_session_id)
    env = await _get_env(session_id, create_if_missing=True)
    try:
        if isinstance(env, SandboxEnv):
            result = await env.reset(
                seed=req.seed,
                difficulty=req.difficulty,
                mode=req.mode,
                drill_mode=req.drill_mode,
                drill_seed=req.drill_seed,
            )
        else:
            result = await env.reset(seed=req.seed, difficulty=req.difficulty, mode=req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _result_to_dict(result, session_id=session_id)


@app.post(
    "/step",
    response_model=EnvResponse,
    tags=["Environment"],
    summary="Step environment",
    responses={
        422: {"description": "Action JSON does not match the documented action schemas."},
    },
)
async def step(
    req: StepRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
) -> dict[str, Any]:
    """Apply one agent action. Invalid **phase** actions still return HTTP 200 with a penalty in `reward`."""
    session_id = _resolve_session_id(req.session_id, x_session_id)
    env = await _get_env(session_id, create_if_missing=True)
    try:
        result = await env.step(req.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _result_to_dict(result, session_id=session_id)


@app.get(
    "/state",
    response_model=EnvResponse,
    tags=["Environment"],
    summary="Get state",
)
async def state(
    session_id: str | None = None,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
) -> dict[str, Any]:
    """Read the current observation without advancing time. If `reset` was never called, `observation` is `null`."""
    resolved_session_id = _resolve_session_id(session_id, x_session_id)
    env = await _get_env(resolved_session_id, create_if_missing=True)
    result = await env.state()
    if result is None:
        return {
            "observation": None,
            "reward": 0.0,
            "done": False,
            "info": {"session_id": resolved_session_id},
        }
    return _result_to_dict(result, session_id=resolved_session_id)


@app.post("/close", tags=["Environment"], summary="Close episode")
async def close(
    session_id: str | None = None,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
) -> dict[str, str]:
    """Release the current episode; call **reset** before stepping again."""
    resolved_session_id = _resolve_session_id(session_id, x_session_id)
    await _close_session(resolved_session_id)
    return {"status": "closed", "session_id": resolved_session_id}


@app.post(
    "/inference",
    response_model=InferenceResponse,
    tags=["System"],
    summary="Run inference benchmark",
)
async def inference_endpoint() -> dict[str, Any]:
    """Execute the bundled LLM inference loop once. Requires server-side env (e.g. `HF_TOKEN`); see project README."""
    import inference

    buf = io.StringIO()
    with redirect_stdout(buf):
        await inference.run()

    stdout = buf.getvalue()
    score = 0.0
    success = False
    for line in stdout.strip().splitlines():
        if line.startswith("[END]"):
            for part in line.split():
                if part.startswith("score="):
                    score = float(part.split("=")[1])
                if part.startswith("success="):
                    success = part.split("=")[1] == "true"

    return {"stdout": stdout, "score": score, "success": success}


# ── entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    import uvicorn

    workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=max(1, workers))


if __name__ == "__main__":
    main()
