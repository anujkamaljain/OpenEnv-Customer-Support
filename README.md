---
title: Enterprise Incident Command Center
emoji: 🚨
colorFrom: purple
colorTo: red
sdk: docker
tags:
  - openenv
  - rl
  - incident-response
app_port: 7860
pinned: false
license: mit
short_description: Enterprise incident response world model for OpenEnv
---

# OpenEnv: Enterprise Incident Command Center (EICC)

EICC is a deterministic OpenEnv environment for enterprise incident response.  
It extends the original customer-support triage workflow with incident-mode world modeling, cascading microservice failures, enterprise tools, policy drift, and long-horizon action planning.

- Live Space: [anuj2209-openenv-customer-support.hf.space](https://anuj2209-openenv-customer-support.hf.space)
- API style: async `reset()` / `step()` / `state()` / `close()`
- Backward compatible ticket mode + new incident mode

## What Is New in v2

- 21 total actions (6 ticket + 15 incident actions)
- 4 incident phases: triage, investigation, response, resolution
- 5-service dependency mesh with cascading failure behavior
- 8 enterprise subsystems (monitoring, CRM, billing, KB, policy, incident history, runbooks, stakeholders)
- Incident scenarios across easy/medium/hard/nightmare tiers
- Deterministic grading and simulation (seed-based, no stochastic graders)
- Training pipeline artifacts (`train.py`, `evaluate.py`, `train_notebook.ipynb`)

## Architecture

```text
AUTH -> PAYMENTS -> NOTIFICATIONS
   \      |
    \     v
   DATABASE <- ANALYTICS

Enterprise tools:
  CRM | Billing | KB | Policy | Incident History | Runbooks
Human layer:
  Dynamic customer queue + stakeholder patience
```

Detailed pitch/demo diagram: `demo/architecture_diagram.md`

## Action Reference (21)

### Ticket mode (legacy 6)
- `classify`
- `route`
- `respond`
- `escalate`
- `resolve`
- `request_info`

### Incident mode (15 new)
- `check_monitoring`
- `probe_service`
- `fetch_logs`
- `fetch_user_data`
- `check_billing`
- `query_kb`
- `check_policy`
- `query_incident_history`
- `follow_runbook_step`
- `apply_fix`
- `verify_fix`
- `rollback_fix`
- `notify_stakeholders`
- `write_postmortem`
- `update_kb`

## Incident Scenario Tiers

| Tier | Count | Steps | Customers | Typical Characteristics |
|---|---:|---:|---:|---|
| easy | 3 | 40 | 2-3 | single failure, clear root cause |
| medium | 5 | 50 | 4-6 | cascading issues, red herrings, one policy drift |
| hard | 7 | 70 | 8-12 | deeper cascades, outdated KB, multiple drifts |
| nightmare | 3 | 80 | 10-15 | compound faults, high noise, maximum complexity |

## Training Pipeline

- `train.py`: curriculum + trajectory collection + optional GRPO training hooks
- `evaluate.py`: formal evaluation metrics + before/after comparison output
- `train_notebook.ipynb`: Colab-ready notebook flow

Expected workflow:

```bash
python train.py --iterations 1 --episodes 1 --k 2 --dry-run
python evaluate.py --policy compare --episodes-per-difficulty 5 --plot
```

The compare mode emits baseline vs trained reports and structured behavior diffs.

## Updated Baseline Results Table

| Mode | Difficulty Mix | Typical Episode Length | Notes |
|---|---|---:|---|
| ticket | easy/medium/hard | 3-10 | stable legacy baseline |
| incident (baseline policy) | all tiers | 40-80 | low score without systematic investigation |
| incident (trained-style policy) | all tiers | 40-80 | improved root-cause and policy-check behavior |

## HTTP API

Endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /close`
- `GET /health`
- `POST /inference`

Optional production headers:
- `X-Session-ID`: isolate concurrent episodes per client/session
- `X-API-Key`: required only when server is configured with `OPENENV_API_KEY`

Incident reset example:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"mode":"incident","difficulty":"easy","seed":0}'
```

## Setup

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Reproducible runtime dependencies are pinned in `requirements.lock`.

## Deployment (Hugging Face Spaces)

```bash
docker build -t eicc .
docker run -p 7860:7860 eicc
curl http://localhost:7860/reset -X POST \
  -H "Content-Type: application/json" \
  -d '{"mode":"incident","difficulty":"easy"}'
```

Container defaults to 1 worker and includes a healthcheck on `/health`.
This environment keeps episode state in-process, so multi-worker deployment
without a shared external state backend can cause reset/step session drift.

## Demo Assets

- Reward curve image: `demo/reward_curve.svg`
- Incident walkthrough: `demo/incident_resolution_walkthrough.md`
- Architecture diagram: `demo/architecture_diagram.md`
- 2-minute video script: `demo/video_script_2min.md`

## Repository Layout (high-level)

```text
env/           # world + systems + environment dispatch/state machine
models/        # pydantic action/observation/incident schemas
graders/       # deterministic ticket + incident graders
tasks/         # ticket bank + incident bank + scenario data
tests/         # legacy + incident test suite
train.py       # phase-7 training pipeline
evaluate.py    # formal metrics and comparison script
openenv.yaml   # OpenEnv manifest (ticket + incident tasks)
```

## Determinism Contract

- Seeded scenario selection
- Seeded service-mesh and alert behavior
- No LLM judge in scoring
- No wall-clock/time-based randomness

## License

MIT
