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
It extends the original customer-support triage workflow into a multi-app incident command simulation featuring cascading microservice failures, policy drift, long-horizon reasoning, and structured tool use.

- Live Hugging Face Space: [anuj2209-openenv-customer-support.hf.space](https://anuj2209-openenv-customer-support.hf.space)
- OpenEnv API style: async `reset()` / `step()` / `state()` / `close()`
- Backward-compatible legacy ticket mode + new incident mode
- Theme: `#3.1 Professional Tasks` + Scaler AI Labs multi-app enterprise sub-theme

## TL;DR

- **Problem:** Enterprise incident response is a multi-app, partially observable, long-horizon task. Most benchmarks don't capture it.
- **Environment:** 21 typed actions across 4 incident phases, 5 interdependent services, 8 enterprise subsystems, 18 hand-crafted scenarios, deterministic scoring.
- **Training pipeline:** GRPO on Qwen2.5-3B via Unsloth + HF TRL, end-to-end with the running environment (not a static dataset).
- **Evaluation pipeline:** deterministic episodes across difficulty tiers with normalized + raw reward, SLA, root-cause, long-horizon consistency, and 8 tracked behavioral skills; supports checkpoint-based adapter evaluation.

## How Judges Can Reproduce (quick path)

1. Open `train_notebook.ipynb` in Colab (T4 GPU).
2. Run cells in order:
   - Step 1: clone repo.
   - Step 2: install deps.
   - Step 3: dry-run sanity: `python train.py --iterations 1 --episodes 1 --k 2 --dry-run`.
   - Step 4: baseline evaluation.
   - Step 5: GRPO training (produces `artifacts/train/trained_adapter/`).
   - Step 6: baseline vs trained-checkpoint comparison with `--plot`.
   - Steps 7-8: inspect reports and reward curves.
   - Steps 9-10 (optional): 2-seed reproducibility + mean ± std summary.

All outputs are written to `artifacts/...` and can be downloaded directly from the Colab file browser.

## Environment Design

### Architecture

```text
AUTH -> PAYMENTS -> NOTIFICATIONS
   \      |
    \     v
   DATABASE <- ANALYTICS

Enterprise tools:
  Monitoring | CRM | Billing | KB | Policy | Incident History | Runbooks
Human layer:
  Dynamic customer queue + stakeholder patience
```

Detailed diagram: `demo/architecture_diagram.md`.

### Action space (21 actions)

Ticket mode (legacy 6): `classify`, `route`, `respond`, `escalate`, `resolve`, `request_info`.

Incident mode (15 new):
- Investigation: `check_monitoring`, `probe_service`, `fetch_logs`, `fetch_user_data`, `check_billing`, `query_kb`, `check_policy`, `query_incident_history`, `follow_runbook_step`
- Response: `apply_fix`, `rollback_fix`, `notify_stakeholders`
- Resolution: `verify_fix`, `write_postmortem`, `update_kb`

Actions are phase-gated: the agent sees only valid actions for the current phase via `available_actions` in the observation.

### Incident scenario tiers

| Tier | Count | Steps | Customers | Typical Characteristics |
|---|---:|---:|---:|---|
| easy | 3 | 40 | 2-3 | single failure, clear root cause |
| medium | 5 | 50 | 4-6 | cascading issues, red herrings, one policy drift |
| hard | 7 | 70 | 8-12 | deeper cascades, outdated KB, multiple drifts |
| nightmare | 3 | 80 | 10-15 | compound faults, high noise, maximum complexity |

### Partial observability

Hidden from agent (must be discovered): true root cause, full dependency graph on hard+, KB accuracy, current policy values, internal customer risk, red-herring symptoms.

Visible: alert text, tool results, accumulated `known_facts`, stakeholder patience, SLA/step counter, phase-restricted `available_actions`.

## Reward Design

Per-step deterministic rewards shaped by:

- action validity (structurally valid action for current phase)
- investigation-before-action
- KB cross-verification (query_kb followed by evidence-gathering)
- policy-awareness before sensitive actions
- root-cause correctness of `apply_fix`
- customer tone matching against sentiment
- stakeholder proactivity before patience decays
- resource budgets (max fix attempts, max escalations)

No LLM judge is used. All reward logic is in the repo (see `graders/` and `env/`).

## Training Pipeline

- `train.py`
  - Collects trajectories from the live OpenEnv environment across a curriculum (easy → nightmare).
  - Runs GRPO (Unsloth + HF TRL) on Qwen2.5-3B with LoRA adapters.
  - Uses a structured reward with parse-rate diagnostics printed during training.
  - Saves the trained adapter to `artifacts/train/trained_adapter/`.
  - After training, attempts checkpoint-based evaluation; falls back to heuristic policy with a visible log line if checkpoint eval can't run.
- `evaluate.py`
  - Runs deterministic incident episodes per difficulty tier.
  - Reports normalized reward, raw cumulative reward, SLA compliance, root-cause accuracy, long-horizon consistency, and 8 behavioral skill scores.
  - Plots a dual-panel reward curve (normalized + raw) so the baseline is not hidden by clamping.
- `train_notebook.ipynb`
  - Colab-first flow with dry-run, baseline eval, training, compare eval, plotting, and optional two-seed reproducibility cells.

Key CLI examples:

```bash
python train.py --iterations 1 --episodes 1 --k 2 --dry-run
python evaluate.py --policy compare --episodes-per-difficulty 5 --plot --output-dir artifacts/eval
```

Checkpoint-based evaluation (recommended for final submission):

```bash
python evaluate.py --policy compare \
  --compare-trained-policy trained_checkpoint \
  --checkpoint-dir artifacts/train/trained_adapter \
  --checkpoint-base-model Qwen/Qwen2.5-3B-Instruct \
  --episodes-per-difficulty 5 --plot --output-dir artifacts/eval_compare_ckpt
```

Two-seed reproducibility:

```bash
python train.py --iterations 4 --episodes 8 --k 4 --seed 7 --output-dir artifacts/train_seed7
python train.py --iterations 4 --episodes 8 --k 4 --seed 17 --output-dir artifacts/train_seed17
```

## Evaluation Metrics

Per-episode:
- Normalized reward (clamped to `[0, 1]`)
- Raw cumulative reward (can be negative)
- SLA compliance (pending tickets == 0 at end)
- Root-cause accuracy (did at least one correct `apply_fix` land)
- Long-horizon consistency (post-fix actions stay in the consistent set)
- 8 tracked skills: investigation-before-action, KB cross-verification, policy checking, stakeholder proactivity, root-cause accuracy, tone matching, resource efficiency, red-herring dismissal

Aggregates:
- `avg_normalized_reward`
- `avg_raw_reward`
- Per-difficulty averages

## Results

This repository ships the full training and evaluation pipeline. Numerical results are produced by running the pipeline; we don't hard-code numbers into this README.

When you run the full Colab notebook end-to-end on a T4:
- Training output: `artifacts/train/trained_adapter/` (trained LoRA adapter)
- Evaluation output: `artifacts/eval/reward_curves.png`, `baseline_report.json`, `trained_report.json`

For judges, the recommended evidence flow is:
1. Run Step 5 (training) in `train_notebook.ipynb`.
2. Run Step 6 with `--compare-trained-policy trained_checkpoint`.
3. Inspect `reward_curves.png` and `trained_report.json` in `artifacts/eval/`.

> Note on compare semantics: if you run `--policy compare` without `--compare-trained-policy trained_checkpoint`, the trained side is a deterministic trained-style heuristic policy (clearly labeled in logs). Use the checkpoint option above for strict before/after evidence from the actual trained adapter.

## HTTP API (OpenEnv-compatible)

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

Example:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"mode":"incident","difficulty":"easy","seed":0}'
```

## Setup (local)

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

The container defaults to 1 worker with a healthcheck on `/health`. Episode state is in-process, so horizontal scaling requires an external session store.

## Determinism Contract

- Seeded scenario selection and service-mesh behavior
- No LLM judge in scoring
- No wall-clock/time-based randomness
- Same `(seed, difficulty, mode)` triple deterministically yields the same scenario

## Demo Assets

- Architecture diagram: `demo/architecture_diagram.md`
- Incident walkthrough: `demo/incident_resolution_walkthrough.md`
- Reward curve sketch (illustrative): `demo/reward_curve.svg`
- 3-minute pitch script: `demo/pitch_script_3min.md`
- 2-minute video script: `demo/video_script_2min.md`
- Mini-blog draft: `demo/hf_mini_blog.md`
- Real evaluation outputs are produced by `evaluate.py` into `--output-dir` (default `artifacts/eval`).

## Public Submission Links

The links below are filled in at the time of final submission; treat `TODO_*` as placeholders until they are replaced with live URLs:

- Hugging Face Space: [anuj2209-openenv-customer-support.hf.space](https://anuj2209-openenv-customer-support.hf.space)
- Hugging Face mini-blog: `TODO_ADD_HF_BLOG_URL`
- YouTube demo (<2 min): `TODO_ADD_YOUTUBE_URL`
- Optional slides: `TODO_ADD_SLIDES_URL`

## Hackathon Submission Checklist

- [x] OpenEnv-based environment with manifest: `openenv.yaml`
- [x] Training script + Colab notebook using TRL/Unsloth: `train.py`, `train_notebook.ipynb`
- [x] Hugging Face Space deployment link in README
- [x] README motivates the problem, explains how the env works, and describes results flow
- [x] Evaluation pipeline emits baseline + trained-side reports and reward curves
- [x] Checkpoint-based evaluation path (`--policy trained_checkpoint`) is available
- [ ] Publish a Hugging Face mini-blog (<2 min read) and add URL in **Public Submission Links**
- [ ] Publish a <2 min YouTube demo and add URL in **Public Submission Links**
- [ ] Commit one real run snapshot (plot + reports) before submission

## Repository Layout

```text
env/           # world + systems + environment dispatch/state machine
models/        # pydantic action/observation/incident schemas
graders/       # deterministic ticket + incident graders
tasks/         # ticket bank + incident bank + scenario data
tests/         # legacy + incident test suite
train.py       # GRPO training pipeline
evaluate.py    # deterministic evaluation + before/after comparison
openenv.yaml   # OpenEnv manifest
```

## License

MIT
