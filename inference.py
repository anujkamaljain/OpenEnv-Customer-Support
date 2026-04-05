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


def _sanitise_action(raw_dict: dict[str, Any], phase: str) -> dict[str, Any]:
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

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
    ]

    try:
        result = await env.reset(seed=0, difficulty=difficulty)
        obs = result.observation

        for step_idx in range(MAX_STEPS):
            user_msg = _obs_to_user_message(obs)
            messages.append({"role": "user", "content": user_msg})

            error: str | None = None
            try:
                raw_text = await _query_model(client, messages)
                raw_dict = _extract_json(raw_text)
                if raw_dict is not None:
                    action = _sanitise_action(raw_dict, obs.phase)
                else:
                    action = dict(_FALLBACK.get(obs.phase, _FALLBACK["routed"]))
                    error = "JSON parse failed; used fallback action"
            except Exception as exc:
                action = dict(_FALLBACK.get(obs.phase, _FALLBACK["routed"]))
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
