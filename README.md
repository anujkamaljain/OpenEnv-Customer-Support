---
title: OpenEnv Customer Support
emoji: 🌍
colorFrom: purple
colorTo: indigo
sdk: docker
tags:
  - openenv
app_port: 7860
pinned: false
license: mit
short_description: Real-world customer support triage environment
---

# OpenEnv: Customer Support Triage

A production-grade reinforcement learning environment for evaluating AI agents on real-world customer support workflows. Built for the [OpenEnv](https://github.com/open-env) framework.

Agents must classify, route, respond to, escalate, and resolve customer tickets under time pressure, policy constraints, and incomplete information — mirroring the decision-making complexity of a human support team.

**Live deployment:** [anuj2209-openenv-customer-support.hf.space](https://anuj2209-openenv-customer-support.hf.space)

---

## Why This Matters

Most LLM benchmarks test isolated capabilities: summarization, QA, or tool use. Real support work is a **multi-step, multi-objective optimization problem** where agents must balance competing pressures:

- **Speed vs. quality** — gathering information improves responses but risks SLA violations
- **Policy compliance vs. customer satisfaction** — refund caps conflict with angry customers demanding full refunds
- **Signal vs. noise** — critical security incidents can be buried in routine-sounding tickets

This environment makes those trade-offs explicit and measurable.

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-stage workflow** | `classify → route → (respond / escalate / request_info) → resolve` |
| **SLA deadlines** | Per-ticket step limits based on urgency; escalating penalties for overruns |
| **Business impact weighting** | Errors on high-value enterprise customers carry heavier penalties |
| **Multi-objective reward** | 8 weighted components: classification, routing, response, resolution, escalation, urgency, efficiency, SLA |
| **Deterministic grading** | No LLMs in the loop — reproducible scores via weighted keyword matching with diversity controls |
| **Async-native** | All public methods are `async`; runs in any async inference loop |
| **Pydantic v2 schemas** | Typed observations, actions, and results with validation |

---

## Environment Design

### State Machine

The environment is a deterministic finite state machine with six phases:

```
unclassified ──classify──► classified ──route──► routed ──┬── respond ──► responding ──┐
                               │                          ├── request_info ─────────────┤
                               │                          ├── escalate ──► escalated ───┤
                               └── escalate ──► escalated ─┘                            │
                                                                               resolve ─┘
                                                                                  │
                                                                                  ▼
                                                                              resolved
```

Each phase gates which actions are valid. Attempting an out-of-phase action returns a penalty (`-0.05`) without crashing the environment.

### Phase Transitions

| Phase | Valid Actions |
|---|---|
| `unclassified` | `classify` |
| `classified` | `route`, `escalate` |
| `routed` | `respond`, `escalate`, `resolve`, `request_info` |
| `responding` | `respond`, `escalate`, `resolve`, `request_info` |
| `escalated` | `resolve` |
| `resolved` | *(terminal)* |

### Episode Termination

An episode ends when the agent issues `resolve` or exhausts the step budget (8–10 steps depending on difficulty).

---

## Action Space

All actions are Pydantic v2 models using a discriminated union on `action_type`:

| Action | Fields | Purpose |
|---|---|---|
| `classify` | `category`, `priority` | Assign ticket category and urgency level |
| `route` | `department` | Forward to the correct department |
| `respond` | `response_text`, `tone` | Draft a customer-facing response |
| `escalate` | `reason`, `target_team` | Escalate to L2, engineering, or management |
| `resolve` | `resolution_summary`, `offered_compensation` | Close the ticket with a summary |
| `request_info` | `question_to_customer` | Ask the customer for clarification |

**Categories:** `billing`, `bug_report`, `feature_request`, `account_access`, `general_inquiry`, `cancellation`

**Priorities:** `low`, `medium`, `high`, `critical`

**Departments:** `billing`, `technical`, `account`, `general`

---

## Observation Space

Each step returns an `Observation` with the following fields:

| Field | Type | Description |
|---|---|---|
| `ticket_id` | `str` | Unique identifier |
| `ticket_text` | `str` | Customer's original message |
| `customer_sentiment` | `angry \| frustrated \| neutral \| satisfied` | Emotional state |
| `customer_tier` | `free \| pro \| enterprise` | Account tier |
| `customer_value` | `low \| medium \| high` | Business impact weight |
| `category_hint` | `str \| null` | Hint for easy tasks; `null` for hard |
| `constraints` | `list[str]` | Policy constraints (e.g. "do not offer refund > $50") |
| `phase` | `Phase` | Current state machine phase |
| `available_actions` | `list[str]` | Valid actions in this phase |
| `current_step` / `max_steps` | `int` | Step budget tracking |
| `sla_steps_remaining` | `int` | Steps before SLA penalties begin |
| `history` | `list[ActionRecord]` | Full action-reward history |
| `max_total_reward` | `float` | Achievable maximum for normalization |

---

## Tasks

### Easy — Classification + Routing + Resolution

- 5 tickets with clear categories and direct customer requests
- `category_hint` provided; no ambiguity
- No escalation required, no policy constraints
- **SLA:** 5 steps | **Max steps:** 8
- Optimal path: 3 steps (`classify → route → resolve`)

### Medium — Classification + Response + Resolution

- 5 tickets requiring thoughtful responses judged by keyword coverage
- Mixed sentiments; tone constraints active
- Compensation ranges with boundary conditions
- **SLA:** 5–6 steps | **Max steps:** 9
- Optimal path: 4 steps (`classify → route → respond → resolve`)

### Hard — Multi-Turn Reasoning Under Pressure

- 10 tickets designed to stress-test agent reasoning:

| Trap Type | Example |
|---|---|
| **Ambiguity** | Ticket mentions both billing and a UI bug — agent must identify root cause as `bug_report`, not `billing` |
| **Misleading sentiment** | Customer writes calmly about unauthorized account access — agent must still flag as `critical` |
| **Policy constraints** | Customer demands $499 full refund; policy caps at $150 — agent must offer partial compensation |
| **Partial information** | Vague bug report ("something isn't working") — agent must `request_info` before responding |
| **Distractor signals** | Ticket mixes three issues; agent must prioritize security over minor billing discrepancy |
| **Escalation traps** | Some tickets require escalation; others don't — wrong call in either direction is penalized |

- No `category_hint` — the agent must infer everything
- **SLA:** 4–6 steps | **Max steps:** 10
- Optimal path: 5–6 steps

---

## Reward Function

### Per-Step Rewards

| Signal | Reward | Condition |
|---|---|---|
| Correct classification | `+0.10` | Category and priority both correct |
| Partial classification | `+0.01 – +0.06` | One of category/priority correct |
| Urgency bonus | `+0.10` | Correctly identifying `high`/`critical` priority |
| Correct routing | `+0.10` | Department matches ground truth |
| Response quality | `0 – +0.20` | Weighted keyword coverage of required/optional terms |
| Correct escalation | `+0.15` | Escalated to the right team when required |
| Unnecessary escalation | `-0.10×` | Multiplied by business impact factor |
| Information gathering | `+0.05` | First `request_info` on partial-info tickets |
| Resolution quality | `0 – +0.25` | Keyword coverage + compensation accuracy |
| Repeated action | `-0.05` | Exact duplicate of previous action |
| Invalid/out-of-phase | `-0.05` | Action doesn't match current phase |
| SLA overage | `-0.02 × steps_over` | Accumulating penalty per step past deadline |

All per-step rewards are clamped to `[-0.25, +0.30]`.

### Speed vs. Quality Trade-Off

On tickets with `partial_info = true`:

- **Skip `request_info`:** Response and resolution quality scores are multiplied by `0.6` — faster but capped
- **Use `request_info`:** Full quality available, but the extra step risks SLA penalties

This creates a genuine strategic dilemma with no universally correct answer.

### Final Episode Score

When `done = true`, the `info` dict includes a `final_score_breakdown`:

| Component | Weight |
|---|---|
| Classification | 15% |
| Routing | 10% |
| Response quality | 20% |
| Resolution quality | 20% |
| Escalation | 10% |
| Urgency handling | 10% |
| Efficiency | 5% |
| SLA compliance | 10% |

The `normalized_score` (∈ [0, 1]) divides cumulative reward by `max_total_reward`, which accounts for unavoidable SLA penalties on the optimal path.

---

## Example Trajectory

A sample episode on an easy billing ticket (`EASY-002`: duplicate charge, frustrated Pro customer):

```
Step 1: classify  → category=billing, priority=high
        reward=+0.20  (classification +0.10, urgency bonus +0.10)
        feedback="Correct classification. Urgency correctly identified (+0.10)."

Step 2: route     → department=billing
        reward=+0.10
        feedback="Routed to the correct department."

Step 3: resolve   → resolution_summary="Duplicate charge refund processed...", compensation=29.99
        reward=+0.25
        feedback="Resolution quality: 100%."

Episode complete: normalized_score=0.982
```

Total: 3 steps, 0 SLA penalties, all objectives met.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)

### Local Installation

```bash
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t openenv-customer-support .
docker run -p 7860:7860 -e HF_TOKEN=your_token openenv-customer-support
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Hugging Face API token for model inference |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use via HF router |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible API endpoint |
| `TASK_NAME` | `customer_support_triage` | Task identifier for validator output |
| `OPENENV_DIFFICULTIES` | `easy,medium,hard` | Subset/order of baseline episodes (comma-separated) |
| `BENCHMARK` | `customer_support_triage` | `env=` field in `[START]` lines (matches `openenv.yaml` `name`) |

---

## Usage

### HTTP API

The server exposes endpoints on port `7860`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Browser **Command Center** (HTML debug UI) |
| `GET` | `/docs` | Interactive **OpenAPI** reference (Swagger UI; same contract as this section) |
| `GET` | `/health` | Liveness probe — JSON `{"status": "ok"}` |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Apply one action |
| `GET` | `/state` | Read observation without advancing |
| `POST` | `/close` | Clear episode state |
| `POST` | `/inference` | Run the bundled LLM loop once (requires `HF_TOKEN` on the server) |

#### POST /reset

Starts a new episode. Body is **optional** — all fields have defaults.

```bash
# With parameters
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 0, "difficulty": "easy"}'

# With empty body (defaults: seed=0, difficulty=null → picks from all tickets)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# With no body at all (same defaults)
curl -X POST http://localhost:7860/reset
```

**Request body fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `0` | Index into the ticket pool (modulo pool size); with `difficulty` omitted, indexes the full combined bank |
| `difficulty` | `easy` \| `medium` \| `hard` \| omitted / `null` | `null` | Only these three literals are accepted; omit or `null` to sample across all difficulties. Other strings are rejected with **422** (Swagger’s generic `"string"` example is invalid). |

**Errors:** **400** if the ticket bank cannot satisfy the request (e.g. empty pool for a filter). **422** if JSON does not match the schema.

**Sample response:**

```json
{
  "observation": {
    "ticket_id": "EASY-001",
    "ticket_text": "Hi, I forgot my password and can't log in...",
    "customer_sentiment": "neutral",
    "customer_tier": "free",
    "customer_value": "low",
    "category_hint": "account_access",
    "phase": "unclassified",
    "available_actions": ["classify"],
    "current_step": 0,
    "max_steps": 8,
    "sla_steps_remaining": 6,
    "constraints": [],
    "history": [],
    "max_total_reward": 0.45
  },
  "reward": 0.0,
  "done": false,
  "info": {
    "phase": "unclassified",
    "steps_taken": 0,
    "normalized_score": 0.0,
    "max_total_reward": 0.45,
    "difficulty": "easy"
  }
}
```

#### POST /step

Applies an action. The body must be `{"action": { ... }}` where `action` matches one of the Pydantic action models (discriminated union on `action_type` — same shapes as in [Action Space](#action-space)).

**Validation vs. phase errors (HTTP):**

- **422** — JSON does not match any action schema (unknown `action_type`, missing required fields, wrong enum values).
- **200** — Request is valid JSON, but the environment may apply a **penalty** (`reward` negative) if the action is illegal for the current phase (same behavior as the Python `CustomerSupportEnv`).

**All 6 action types:**

```bash
# 1. classify
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "classify",
      "category": "billing",
      "priority": "high"
    }
  }'

# 2. route
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "route",
      "department": "billing"
    }
  }'

# 3. respond
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "respond",
      "response_text": "We sincerely apologize for the duplicate charge. We are processing your refund immediately.",
      "tone": "empathetic"
    }
  }'

# 4. escalate
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "escalate",
      "reason": "Enterprise customer experiencing data loss requires engineering investigation.",
      "target_team": "engineering"
    }
  }'

# 5. resolve
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "resolve",
      "resolution_summary": "Duplicate charge refund of $29.99 has been processed. Credit will appear within 3-5 business days.",
      "offered_compensation": 29.99
    }
  }'

# 6. request_info
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "request_info",
      "question_to_customer": "Could you share which specific page shows the error and any error messages you see?"
    }
  }'
```

**Sample response:**

```json
{
  "observation": {
    "ticket_id": "EASY-002",
    "phase": "classified",
    "available_actions": ["escalate", "route"],
    "current_step": 1,
    "sla_steps_remaining": 3,
    "history": [
      {
        "step": 0,
        "action_taken": "classify",
        "env_feedback": "Correct classification. Urgency correctly identified (+0.10).",
        "reward_earned": 0.2
      }
    ]
  },
  "reward": 0.2,
  "done": false,
  "info": {
    "normalized_score": 0.3636,
    "reward_breakdown": {
      "classification": 0.1,
      "urgency_bonus": 0.1,
      "repeat_penalty": 0.0,
      "sla_penalty": 0.0,
      "total": 0.2
    }
  }
}
```

#### GET /state

Returns the current observation without advancing the episode. If **`POST /reset`** has never been called (or after **`POST /close`**), the response is still **200** with `observation: null`, `reward: 0.0`, `done: false`, and `info: {}`.

```bash
curl http://localhost:7860/state
```

#### POST /close

Clears internal episode state on the server. Call **`POST /reset`** before **`POST /step`** again.

```bash
curl -X POST http://localhost:7860/close
```

#### GET /health

```bash
curl http://localhost:7860/health
```

#### POST /inference

Runs `inference.run()` in-process (same script as `python inference.py`). Requires **`HF_TOKEN`** and related settings on the server. Response JSON: `stdout` (captured logs), `score`, `success` (parsed from the final `[END]` line when present).

```bash
curl -X POST http://localhost:7860/inference
```

### Python API

```python
from env import CustomerSupportEnv

env = CustomerSupportEnv()
result = await env.reset(seed=0, difficulty="hard")

while not result.done:
    action = agent.act(result.observation)
    result = await env.step(action)

print(result.info["normalized_score"])
await env.close()
```

### Run Inference

```bash
python inference.py
```

Output follows the OpenEnv validator format:

```
[START] task=customer_support_triage env=openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"classify",...} reward=0.20 done=false error=null
[STEP] step=2 action={"action_type":"route",...} reward=0.10 done=false error=null
[STEP] step=3 action={"action_type":"resolve",...} reward=0.25 done=true error=null
[END] success=true steps=3 score=0.982 rewards=0.20,0.10,0.25
```

### Validate

```bash
openenv validate
```

---

## Deploying to Hugging Face Spaces

### 1. Create a new Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and select:

- **SDK:** Docker
- **Hardware:** CPU Basic (2 vCPU, 16 GB) — the environment is lightweight
- **Visibility:** Public or Private

### 2. Push the code

```bash
git remote add space https://huggingface.co/spaces/Anuj2209/openenv-customer-support
git push space main
```

Or upload all files through the HF web interface.

### 3. Set the `HF_TOKEN` secret

In your Space's **Settings > Repository secrets**, add:

| Name | Value |
|---|---|
| `HF_TOKEN` | Your Hugging Face API token (from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)) |

This is only needed if you use the `/inference` endpoint. The environment endpoints (`/reset`, `/step`, `/state`) work without it.

### 4. Verify deployment

Once the Space is running (build takes ~30 seconds), test it:

```bash
# Health check (JSON)
curl https://anuj2209-openenv-customer-support.hf.space/health

# Optional: open Command Center (HTML) or API docs in a browser
# https://anuj2209-openenv-customer-support.hf.space/
# https://anuj2209-openenv-customer-support.hf.space/docs

# Reset
curl -X POST https://anuj2209-openenv-customer-support.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 0, "difficulty": "easy"}'
```

### How it works

- The Dockerfile creates a non-root `user` (UID 1000) as required by HF Spaces
- FastAPI serves on `0.0.0.0:7860` — the port HF Spaces exposes
- Startup is instant: no model loading, no GPU, no heavy initialization
- The environment is fully self-contained — all ticket data is bundled in the image

---

## Baseline Results

| Difficulty | Model | Score | Steps | Notes |
|---|---|---|---|---|
| Easy | Qwen2.5-72B-Instruct | ~0.95 | 3 | Near-optimal on straightforward tickets |
| Medium | Qwen2.5-72B-Instruct | ~0.85 | 4–5 | Response keyword coverage is the bottleneck |
| Hard | Qwen2.5-72B-Instruct | ~0.70 | 5–7 | Policy traps and ambiguity cause partial-credit |

Scores are `normalized_score` values (cumulative reward / max achievable reward). A score of `1.0` represents a theoretically perfect agent.

---

## Project Structure

```
├── env/
│   ├── environment.py      # Core async environment
│   ├── state.py             # Internal state machine + reward tracking
│   └── errors.py            # Custom exceptions
├── models/
│   ├── action.py            # Action discriminated union (Pydantic v2)
│   ├── observation.py       # Observation schema
│   ├── ticket.py            # TicketData + KeywordSpec
│   └── step_result.py       # Uniform return type
├── graders/
│   └── grader.py            # Deterministic keyword + compensation grading
├── tasks/
│   ├── ticket_bank.py       # Ticket loading + deterministic selection
│   └── tickets/
│       ├── easy.json         # 5 tickets
│       ├── medium.json       # 5 tickets
│       └── hard.json         # 10 tickets
├── server/
│   └── app.py               # FastAPI server
├── tests/                    # 216 tests (pytest + pytest-asyncio)
├── inference.py              # LLM agent loop (validator-compliant)
├── openenv.yaml              # Environment manifest
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

```
216 passed in ~1s
```

Tests cover: state transitions, reward bounds, grader determinism, keyword scoring robustness, SLA penalties, trajectory simulations (best/worst case), trade-off mechanics, and inference output format compliance.

---

## License

MIT

