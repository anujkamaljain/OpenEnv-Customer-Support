# OpenEnv — Customer Support Triage Environment

Async, deterministic RL environment for customer support inbox triage and resolution.

## Quick start

```bash
pip install -e ".[dev]"
```

```python
from env import CustomerSupportEnv

env = CustomerSupportEnv()

result = await env.reset(seed=0, difficulty="hard")

while not result.done:
    action = agent.act(result.observation)
    result = await env.step(action)
    print(result.reward, result.observation.phase)

print(f"Final score: {result.info['normalized_score']:.2%}")
await env.close()
```

## Structure

```
env/            Async environment (reset / step / state / close)
models/         Pydantic v2 schemas (Observation, Action, StepResult, TicketData, KeywordSpec)
tasks/          Ticket bank + JSON ticket data (5 easy / 5 medium / 10 hard)
graders/        Deterministic grader (weighted keywords, SLA, multi-objective scoring)
tests/          pytest-asyncio test suite (63 tests)
```

## Difficulty tiers

| Tier   | Tickets | Optimal steps | Key challenge                             |
|--------|---------|---------------|-------------------------------------------|
| Easy   | 5       | 3-4           | Classification + routing                  |
| Medium | 5       | 4-6           | Response quality + tone constraints       |
| Hard   | 10      | 5-9           | Ambiguity, SLA pressure, policy traps     |

## Reward design

### Per-action rewards

| Action        | Correct           | Partial          | Wrong / Unnecessary |
|---------------|-------------------|------------------|---------------------|
| `classify`    | +0.10             | +0.04 to +0.06   | +0.01               |
| urgency bonus | +0.10 (high/crit) | —                | —                   |
| `route`       | +0.10             | —                | +0.01               |
| `respond`     | 0 to +0.20        | (keyword score)  | forbidden: -0.03/ea |
| `escalate`    | +0.15             | +0.05 (wrong tm) | -0.10 * biz_mult    |
| `resolve`     | 0 to +0.25        | (keyword+comp)   | forbidden: -0.03/ea |
| `request_info`| +0.05 (needed)    | —                | -0.03 * biz_mult    |

### Per-step modifiers

| Modifier              | Value                                    |
|-----------------------|------------------------------------------|
| Repeat penalty        | -0.05                                    |
| Wrong phase           | -0.05                                    |
| SLA penalty           | -0.02 * (steps over deadline)            |
| Per-step clamp        | [-0.25, +0.30]                           |

### Business impact multiplier

Negative action-level penalties are scaled by customer value:

| Customer value | Multiplier |
|----------------|------------|
| low            | 1.0x       |
| medium         | 1.3x       |
| high           | 1.8x       |

## SLA / deadline system

Each ticket has an SLA step budget derived from priority:

| Priority | SLA steps | Meaning                              |
|----------|-----------|--------------------------------------|
| critical | 3         | Must resolve in 3 steps or face penalty |
| high     | 4         | —                                    |
| medium   | 6         | —                                    |
| low      | 8         | —                                    |

Every step beyond the SLA incurs an increasing penalty (-0.02, -0.04, -0.06, ...).

## Grader (deterministic, no LLM)

Keyword scoring uses weighted `KeywordSpec`:
- **Required** keywords (60% weight) — must appear
- **Optional** keywords (40% weight) — bonus
- **Forbidden** patterns — penalty per match (-0.03)
- **min_required_hits** — hard floor; score halved when not met

Episode-level scoring (8 dimensions, weights sum to 1.0):

| Dimension              | Weight |
|------------------------|--------|
| Classification         | 15%    |
| Routing                | 10%    |
| Response quality       | 20%    |
| Resolution quality     | 20%    |
| Escalation correctness | 10%    |
| Urgency handling       | 10%    |
| Step efficiency        | 5%     |
| SLA compliance         | 10%    |

## Hard ticket challenges

| ID       | Challenge                                                  |
|----------|------------------------------------------------------------|
| HARD-006 | Ambiguous category (billing symptoms but bug root cause)   |
| HARD-007 | Conflicting signals (calm tone, critical security issue)   |
| HARD-008 | Partial information (must request_info before responding)  |
| HARD-009 | Policy trap (customer demands $499 refund, cap is $150)    |
| HARD-010 | Multi-issue priority conflict (3 issues, security buried)  |

## Running tests

```bash
pytest -v
```
