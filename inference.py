"""OpenEnv hackathon inference script.

Runs an LLM agent against the CustomerSupportEnv, emitting validator-exact
stdout lines.  All output is deterministic given the same model responses.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from typing import Any

from openai import AsyncOpenAI

from env.environment import CustomerSupportEnv

# ── configuration ────────────────────────────────────────────────────────────

TASK_NAME = os.environ.get("TASK_NAME", "customer_support_triage")
BENCHMARK = os.environ.get("BENCHMARK", "customer_support_triage")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MAX_STEPS = 10
RUN_MODE = os.environ.get("OPENENV_MODE", "ticket").strip().lower()

# Comma-separated subset of easy,medium,hard — default runs full hackathon baseline.
def _episode_difficulties() -> list[str]:
    default_order = ("easy", "medium", "hard")
    raw = os.environ.get("OPENENV_DIFFICULTIES", "")
    if not raw.strip():
        return list(default_order)
    want = {x.strip().lower() for x in raw.split(",") if x.strip()}
    picked = [d for d in default_order if d in want]
    return picked if picked else list(default_order)


EPISODE_DIFFICULTIES: list[str] = _episode_difficulties()

# ── valid enum values (for clamping bad model output) ────────────────────────

_CATEGORIES = frozenset(
    ["billing", "bug_report", "feature_request", "account_access",
     "general_inquiry", "cancellation"]
)
_PRIORITIES = frozenset(["low", "medium", "high", "critical"])
_DEPARTMENTS = frozenset(["billing", "technical", "account", "general"])
_ESCALATION_TARGETS = frozenset(["l2_support", "engineering", "management"])
_TONES = frozenset(["formal", "empathetic", "concise"])
_CHECK_TYPES = frozenset(["logs", "resources", "connections", "config"])
_TIME_RANGES = frozenset(["last_5m", "last_15m", "last_1h"])
_POLICY_TYPES = frozenset(["refund", "escalation", "sla", "compensation", "communication"])
_STAKEHOLDERS = frozenset(["vp_engineering", "legal", "support_lead", "all"])
_URGENCIES = frozenset(["info", "warning", "critical"])

# ── fallback actions per phase ───────────────────────────────────────────────

_FALLBACK: dict[str, dict[str, Any]] = {
    "unclassified": {
        "action_type": "classify",
        "category": "general_inquiry",
        "priority": "medium",
    },
    "classified": {
        "action_type": "route",
        "department": "general",
    },
    "routed": {
        "action_type": "resolve",
        "resolution_summary": "Issue has been reviewed and resolved.",
    },
    "responding": {
        "action_type": "resolve",
        "resolution_summary": "Issue has been reviewed and resolved.",
    },
    "escalated": {
        "action_type": "resolve",
        "resolution_summary": "Issue has been reviewed and resolved after escalation.",
    },
    "resolved": {
        "action_type": "resolve",
        "resolution_summary": "Resolved.",
    },
}

_INCIDENT_FALLBACK: dict[str, dict[str, Any]] = {
    "triage": {
        "action_type": "check_monitoring",
        "service_name": None,
    },
    "investigation": {
        "action_type": "check_monitoring",
        "service_name": None,
    },
    "response": {
        "action_type": "respond",
        "response_text": "We are actively investigating and will share updates shortly.",
        "tone": "empathetic",
    },
    "resolution": {
        "action_type": "resolve",
        "resolution_summary": "Issue reviewed and currently stable.",
        "offered_compensation": None,
    },
}

# ── stdout helpers ───────────────────────────────────────────────────────────


def _emit_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _emit_step(
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    done_s = "true" if done else "false"
    err_s = error[:200] if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_s} error={err_s}",
        flush=True,
    )


def _emit_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    success_s = "true" if success else "false"
    rewards_s = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_s} steps={steps} "
        f"score={score:.3f} rewards={rewards_s}",
        flush=True,
    )


# ── observation → prompt ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert customer support agent. You interact with a support \
environment by emitting ONE JSON action per turn.

RULES:
- Respond with ONLY a single JSON object — no markdown, no explanation.
- The JSON must contain "action_type" and the required fields for that type.
- Available action_type values and their schemas:

  classify   → {"action_type":"classify","category":"<cat>","priority":"<pri>"}
               category: billing|bug_report|feature_request|account_access|general_inquiry|cancellation
               priority: low|medium|high|critical

  route      → {"action_type":"route","department":"<dept>"}
               department: billing|technical|account|general

  respond    → {"action_type":"respond","response_text":"<text>","tone":"<tone>"}
               tone: formal|empathetic|concise

  escalate   → {"action_type":"escalate","reason":"<reason>","target_team":"<team>"}
               target_team: l2_support|engineering|management

  resolve    → {"action_type":"resolve","resolution_summary":"<summary>","offered_compensation":<num|null>}

  request_info → {"action_type":"request_info","question_to_customer":"<question>"}

- Follow the phase order: classify → route → (optional: request_info, respond, escalate) → resolve
- Only use actions listed in "available_actions".
- Do NOT repeat request_info — once clarification is gathered it disappears from available_actions.
- Pay attention to constraints and customer sentiment.
- Be efficient — avoid unnecessary steps.
"""

_INCIDENT_SYSTEM_PROMPT = """\
You are an expert enterprise incident commander. You are managing a critical
incident at a fintech company with 5 interconnected services.

YOUR MISSION: Diagnose the root cause, fix the issue, handle affected customers,
and write a post-mortem — all under time pressure.

INCIDENT PHASES:
1. TRIAGE — Assess severity, check monitoring, classify the incident
2. INVESTIGATION — Probe services, fetch logs, query KB, identify root cause
3. RESPONSE — Apply fix, handle customers, notify stakeholders, check policies
4. RESOLUTION — Verify fix, resolve tickets, write post-mortem, update KB

AVAILABLE TOOLS (use JSON actions):
- check_monitoring: {"action_type":"check_monitoring","service_name":"payments"}
- probe_service: {"action_type":"probe_service","service_name":"auth","check_type":"logs"}
- fetch_logs: {"action_type":"fetch_logs","service_name":"database","time_range":"last_15m"}
- query_kb: {"action_type":"query_kb","query":"payment 500 errors"}
- fetch_user_data: {"action_type":"fetch_user_data","customer_id":"CUST-001"}
- check_billing: {"action_type":"check_billing","customer_id":"CUST-001"}
- check_policy: {"action_type":"check_policy","policy_type":"refund"}
- apply_fix: {"action_type":"apply_fix","service_name":"database","fix_type":"restart_service"}
- verify_fix: {"action_type":"verify_fix","service_name":"database"}
- notify_stakeholders: {"action_type":"notify_stakeholders","stakeholder":"vp_engineering","message":"...","urgency":"warning"}
- respond: {"action_type":"respond","response_text":"...","tone":"empathetic"}
- resolve: {"action_type":"resolve","resolution_summary":"...","offered_compensation":null}
- write_postmortem: {"action_type":"write_postmortem","summary":"...","root_cause_description":"...","remediation_steps":["..."],"prevention_measures":["..."]}
- update_kb: {"action_type":"update_kb","article_title":"...","content":"...","tags":["..."]}

CRITICAL RULES:
- ALWAYS check_monitoring before diagnosing
- ALWAYS verify KB information against logs (KB may be outdated!)
- ALWAYS check_policy before offering compensation (policies can change!)
- Keep stakeholders informed — patience decreases every step
- Prioritize enterprise customers (higher SLA, higher value)
- Only ONE JSON action per turn — no extra text
"""


def _obs_to_user_message(obs: Any) -> str:
    parts = [
        f"Ticket ID: {obs.ticket_id}",
        f"Phase: {obs.phase}",
        f"Step: {obs.current_step}/{obs.max_steps}",
        f"SLA steps remaining: {obs.sla_steps_remaining}",
        f"Customer value: {obs.customer_value}",
        f"Customer sentiment: {obs.customer_sentiment}",
        f"Customer tier: {obs.customer_tier}",
        f"Available actions: {obs.available_actions}",
    ]
    if obs.constraints:
        parts.append(f"Constraints: {obs.constraints}")
    if obs.category_hint:
        parts.append(f"Category hint: {obs.category_hint}")
    parts.append(f"\nTicket text:\n{obs.ticket_text}")
    if obs.history:
        parts.append("\nHistory:")
        for h in obs.history:
            parts.append(
                f"  step {h.step}: {h.action_taken} → {h.env_feedback} "
                f"(reward: {h.reward_earned:+.2f})"
            )
    return "\n".join(parts)


def _format_alert_line(alert: str) -> str:
    lowered = alert.lower()
    if "[high]" in lowered:
        return f"🔴 {alert}"
    if "[medium]" in lowered:
        return f"🟡 {alert}"
    if "[low]" in lowered:
        return f"🟢 {alert}"
    return f"⚪ {alert}"


def _incident_obs_to_user_message(obs: Any) -> str:
    """Convert incident observation to compact action-focused prompt."""
    parts = [
        f"=== INCIDENT: {obs.incident_title or obs.incident_id or 'Unknown'} ===",
        f"Phase: {obs.incident_phase}",
        f"Step: {obs.current_step}/{obs.max_steps}",
        f"Available actions: {obs.available_actions}",
    ]

    if getattr(obs, "active_alerts", None):
        parts.append("\nALERTS:")
        for alert in obs.active_alerts[:20]:
            parts.append(f"  {_format_alert_line(alert)}")

    if getattr(obs, "system_status", None):
        parts.append(f"\nSYSTEM STATUS: {json.dumps(obs.system_status, sort_keys=True)}")

    if getattr(obs, "stakeholder_patience", None):
        parts.append(f"\nSTAKEHOLDER PATIENCE: {obs.stakeholder_patience}")

    if getattr(obs, "pending_customer_tickets", 0) > 0:
        parts.append(f"\nPENDING CUSTOMER TICKETS: {obs.pending_customer_tickets}")

    if getattr(obs, "total_incident_cost", None) is not None:
        parts.append(f"\nTOTAL INCIDENT COST: ${obs.total_incident_cost}")

    if getattr(obs, "suggested_runbook", None):
        parts.append(f"\nSUGGESTED RUNBOOK: {json.dumps(obs.suggested_runbook)}")

    if getattr(obs, "known_facts", None):
        parts.append(f"\nKNOWN FACTS: {json.dumps(obs.known_facts, sort_keys=True)}")

    if getattr(obs, "tool_results", None):
        parts.append(f"\nLAST TOOL RESULT: {json.dumps(obs.tool_results, sort_keys=True)}")

    if getattr(obs, "ticket_text", None):
        parts.append(f"\nCURRENT TICKET:\n{obs.ticket_text}")

    history = list(getattr(obs, "history", []) or [])
    if history:
        parts.append("\nHISTORY (last 5):")
        for h in history[-5:]:
            parts.append(f"  step {h.step}: {h.action_taken} -> {h.env_feedback}")
        if len(history) > 5:
            parts.append(f"\nEarlier actions summarized in known facts ({len(history)-5} omitted).")
    return "\n".join(parts)


# ── model interaction ────────────────────────────────────────────────────────


def _build_client() -> AsyncOpenAI:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")
    return AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


async def _query_model(
    client: AsyncOpenAI,
    messages: list[dict[str, str]],
) -> str:
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()


# ── action parsing ───────────────────────────────────────────────────────────


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _clamp_val(val: Any, allowed: frozenset[str], default: str) -> str:
    s = str(val).strip().lower().replace(" ", "_")
    return s if s in allowed else default


def _sanitise_action(
    raw_dict: dict[str, Any],
    phase: str,
    mode: str = "ticket",
    incident_phase: str = "investigation",
) -> dict[str, Any]:
    action_type = str(raw_dict.get("action_type", "")).strip().lower()

    if action_type == "classify":
        return {
            "action_type": "classify",
            "category": _clamp_val(
                raw_dict.get("category", "general_inquiry"),
                _CATEGORIES, "general_inquiry",
            ),
            "priority": _clamp_val(
                raw_dict.get("priority", "medium"),
                _PRIORITIES, "medium",
            ),
        }

    if action_type == "route":
        return {
            "action_type": "route",
            "department": _clamp_val(
                raw_dict.get("department", "general"),
                _DEPARTMENTS, "general",
            ),
        }

    if action_type == "respond":
        text = str(raw_dict.get("response_text", "I am looking into your issue."))
        tone = _clamp_val(raw_dict.get("tone", "formal"), _TONES, "formal")
        return {
            "action_type": "respond",
            "response_text": text[:2000] or "I am looking into your issue.",
            "tone": tone,
        }

    if action_type == "escalate":
        return {
            "action_type": "escalate",
            "reason": str(raw_dict.get("reason", "Requires specialist review."))[:500]
                      or "Requires specialist review.",
            "target_team": _clamp_val(
                raw_dict.get("target_team", "l2_support"),
                _ESCALATION_TARGETS, "l2_support",
            ),
        }

    if action_type == "resolve":
        summary = str(raw_dict.get("resolution_summary", "Issue resolved."))[:2000]
        comp = raw_dict.get("offered_compensation")
        if comp is not None:
            try:
                comp = float(comp)
            except (TypeError, ValueError):
                comp = None
        return {
            "action_type": "resolve",
            "resolution_summary": summary or "Issue resolved.",
            "offered_compensation": comp,
        }

    if action_type == "request_info":
        q = str(raw_dict.get("question_to_customer",
                             "Could you provide more details?"))[:1000]
        return {
            "action_type": "request_info",
            "question_to_customer": q or "Could you provide more details?",
        }

    if action_type == "check_monitoring":
        service = raw_dict.get("service_name")
        return {
            "action_type": "check_monitoring",
            "service_name": None if service in (None, "", "all") else str(service),
        }

    if action_type == "probe_service":
        return {
            "action_type": "probe_service",
            "service_name": str(raw_dict.get("service_name", "payments"))[:100],
            "check_type": _clamp_val(raw_dict.get("check_type", "logs"), _CHECK_TYPES, "logs"),
        }

    if action_type == "fetch_logs":
        return {
            "action_type": "fetch_logs",
            "service_name": str(raw_dict.get("service_name", "payments"))[:100],
            "time_range": _clamp_val(raw_dict.get("time_range", "last_15m"), _TIME_RANGES, "last_15m"),
        }

    if action_type == "fetch_user_data":
        return {
            "action_type": "fetch_user_data",
            "customer_id": str(raw_dict.get("customer_id", "CUST-001"))[:100],
        }

    if action_type == "check_billing":
        return {
            "action_type": "check_billing",
            "customer_id": str(raw_dict.get("customer_id", "CUST-001"))[:100],
        }

    if action_type == "query_kb":
        return {
            "action_type": "query_kb",
            "query": str(raw_dict.get("query", "incident root cause"))[:500] or "incident root cause",
        }

    if action_type == "check_policy":
        return {
            "action_type": "check_policy",
            "policy_type": _clamp_val(raw_dict.get("policy_type", "refund"), _POLICY_TYPES, "refund"),
        }

    if action_type == "apply_fix":
        return {
            "action_type": "apply_fix",
            "service_name": str(raw_dict.get("service_name", "payments"))[:100],
            "fix_type": str(raw_dict.get("fix_type", "restart_service"))[:100],
        }

    if action_type == "verify_fix":
        return {
            "action_type": "verify_fix",
            "service_name": str(raw_dict.get("service_name", "payments"))[:100],
        }

    if action_type == "rollback_fix":
        return {
            "action_type": "rollback_fix",
            "service_name": str(raw_dict.get("service_name", "payments"))[:100],
        }

    if action_type == "notify_stakeholders":
        message = str(raw_dict.get("message", "Incident update: investigation in progress."))[:2000]
        return {
            "action_type": "notify_stakeholders",
            "stakeholder": _clamp_val(raw_dict.get("stakeholder", "all"), _STAKEHOLDERS, "all"),
            "message": message or "Incident update: investigation in progress.",
            "urgency": _clamp_val(raw_dict.get("urgency", "warning"), _URGENCIES, "warning"),
        }

    if action_type == "write_postmortem":
        remediation = raw_dict.get("remediation_steps", [])
        prevention = raw_dict.get("prevention_measures", [])
        rem_list = [str(x)[:300] for x in remediation] if isinstance(remediation, list) else []
        prev_list = [str(x)[:300] for x in prevention] if isinstance(prevention, list) else []
        return {
            "action_type": "write_postmortem",
            "summary": str(raw_dict.get("summary", "Incident summary"))[:3000] or "Incident summary",
            "root_cause_description": str(raw_dict.get("root_cause_description", "Root cause under investigation"))[:2000] or "Root cause under investigation",
            "remediation_steps": rem_list,
            "prevention_measures": prev_list,
        }

    if action_type == "update_kb":
        tags = raw_dict.get("tags", [])
        tag_list = [str(x)[:50] for x in tags] if isinstance(tags, list) else []
        return {
            "action_type": "update_kb",
            "article_title": str(raw_dict.get("article_title", "Incident update"))[:300] or "Incident update",
            "content": str(raw_dict.get("content", "verify root cause and apply fix"))[:4000] or "verify root cause and apply fix",
            "tags": tag_list,
        }

    if action_type == "query_incident_history":
        service_filter = raw_dict.get("service_filter")
        return {
            "action_type": "query_incident_history",
            "query": str(raw_dict.get("query", "similar incidents"))[:500] or "similar incidents",
            "service_filter": None if service_filter in (None, "") else str(service_filter)[:100],
        }

    if action_type == "follow_runbook_step":
        step = raw_dict.get("step_index", 0)
        try:
            step_val = int(step)
        except (TypeError, ValueError):
            step_val = 0
        return {
            "action_type": "follow_runbook_step",
            "runbook_id": str(raw_dict.get("runbook_id", "RB-001"))[:100],
            "step_index": max(0, step_val),
        }

    if mode == "incident":
        return dict(_INCIDENT_FALLBACK.get(incident_phase, _INCIDENT_FALLBACK["investigation"]))
    return dict(_FALLBACK.get(phase, _FALLBACK["routed"]))


def _fallback_action(obs: Any) -> dict[str, Any]:
    mode = str(getattr(obs, "mode", "ticket") or "ticket")
    if mode == "incident":
        phase = str(getattr(obs, "incident_phase", "investigation") or "investigation")
        return dict(_INCIDENT_FALLBACK.get(phase, _INCIDENT_FALLBACK["investigation"]))
    phase = str(getattr(obs, "phase", "routed") or "routed")
    return dict(_FALLBACK.get(phase, _FALLBACK["routed"]))


def _action_to_str(action: dict[str, Any]) -> str:
    return json.dumps(action, separators=(",", ":"))


# ── main loop ────────────────────────────────────────────────────────────────


async def _run_one_episode(
    env: CustomerSupportEnv,
    client: AsyncOpenAI,
    difficulty: str,
) -> None:
    """One full episode: [START] … [STEP]* … [END] for a single difficulty."""
    rewards: list[float] = []
    steps = 0
    score = 0.0
    success = False

    _emit_start(difficulty)

    messages: list[dict[str, str]] = []

    try:
        mode = "incident" if RUN_MODE == "incident" else "ticket"
        result = await env.reset(seed=0, difficulty=difficulty, mode=mode)
        obs = result.observation
        is_incident = getattr(obs, "mode", "ticket") == "incident"
        system_prompt = _INCIDENT_SYSTEM_PROMPT if is_incident else _SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}]
        episode_cap = obs.max_steps if is_incident else MAX_STEPS

        for step_idx in range(episode_cap):
            user_msg = _incident_obs_to_user_message(obs) if is_incident else _obs_to_user_message(obs)
            messages.append({"role": "user", "content": user_msg})
            if len(messages) > 1 + 20:
                messages = [messages[0]] + messages[-20:]

            error: str | None = None
            try:
                raw_text = await _query_model(client, messages)
                raw_dict = _extract_json(raw_text)
                if raw_dict is not None:
                    action = _sanitise_action(
                        raw_dict,
                        str(getattr(obs, "phase", "routed")),
                        mode=str(getattr(obs, "mode", "ticket") or "ticket"),
                        incident_phase=str(getattr(obs, "incident_phase", "investigation") or "investigation"),
                    )
                else:
                    action = _fallback_action(obs)
                    error = "JSON parse failed; used fallback action"
            except Exception as exc:
                action = _fallback_action(obs)
                error = str(exc)[:200]

            action_str = _action_to_str(action)

            step_num = step_idx + 1

            try:
                result = await env.step(action)
            except Exception as exc:
                _emit_step(step_num, action_str, 0.0, True, str(exc)[:200])
                steps = step_num
                break

            reward = result.reward
            done = result.done
            rewards.append(reward)
            steps = step_num

            _emit_step(step_num, action_str, reward, done, error)

            messages.append(
                {"role": "assistant", "content": action_str}
            )

            if done:
                score = result.info.get("normalized_score", 0.0)
                break

            obs = result.observation

        else:
            score = result.info.get("normalized_score", 0.0)

    except Exception:
        error_msg = traceback.format_exc().splitlines()[-1][:200]
        if not rewards:
            _emit_step(1, "{}", 0.0, True, error_msg)
            steps = max(steps, 1)

    success = score > 0.1
    _emit_end(success, steps, score, rewards)


async def run() -> None:
    env = CustomerSupportEnv()
    try:
        try:
            client = _build_client()
        except Exception:
            _emit_start(TASK_NAME)
            error_msg = traceback.format_exc().splitlines()[-1][:200]
            _emit_step(1, "{}", 0.0, True, error_msg[:200])
            _emit_end(False, 1, 0.0, [])
            return

        for difficulty in EPISODE_DIFFICULTIES:
            await _run_one_episode(env, client, difficulty)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run())
