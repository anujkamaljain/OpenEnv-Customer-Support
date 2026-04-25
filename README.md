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

> **Theme:** `#3.1 World Modeling / Professional Tasks`
> **Tagline:** *When everything is on fire, can your agent keep its cool?*

EICC is a deterministic OpenEnv environment that trains LLM agents to behave like a senior SRE / incident commander: diagnose cascading microservice failures, cross-verify potentially outdated knowledge bases, coordinate tools and stakeholders, and execute multi-step remediation — all under SLA pressure and partial observability.

- **Live Hugging Face Space:** [anuj2209-openenv-customer-support.hf.space](https://anuj2209-openenv-customer-support.hf.space)
- **OpenEnv API:** async `reset()` / `step()` / `state()` / `close()` + HTTP server
- **Backward-compatible ticket mode** + new multi-app **incident mode**

---

## For Judges: 60-Second Walkthrough

1. **Problem (why it matters):** Most LLM benchmarks test isolated skills. Real enterprise incidents require partial observability, causal reasoning, long-horizon execution, and tool orchestration.
2. **Environment:** 21 typed actions across 4 incident phases, 5-service causal mesh, 8 enterprise tools, 18 hand-crafted scenarios, deterministic reward (no LLM judge).
3. **Training:** GRPO on Qwen2.5-3B via Unsloth + HF TRL in Colab. Reward includes strict JSON-shape, phase-availability, investigation-before-action, KB cross-verification, policy awareness, and trajectory-grounded action values. Concise-output and cap-hit penalties prevent reward hacking.
4. **Evidence:** `evaluate.py` produces dual-panel reward curves (normalized + raw), per-difficulty breakdown, root-cause accuracy, 8 tracked behavioral skills, and a machine-readable `trained_report.json` that records `policy_used` (`trained_checkpoint` vs `trained_heuristic` fallback) so judges can see exactly which policy produced each run.

### Judging criteria mapping (40 / 30 / 20 / 10)

| Criterion (weight) | How EICC scores |
|---|---|
| **Environment Innovation (40%)** | Partial-observability enterprise world model with causal service mesh, policy drift, outdated KBs, dynamic ticket arrivals, stakeholder patience, CAB approval gate, and blast-radius on wrong fixes. 11 explicit anti-shortcut mechanisms. |
| **Storytelling (30%)** | Single-page README TL;DR, 3-min pitch + 2-min video scripts in `demo/`, architecture diagram, incident walkthrough, and HF mini-blog draft. |
| **Reward Improvement (20%)** | `reward_curves.png` (dual-panel: normalized clamped + raw that exposes negative baselines), `baseline_report.json` vs `trained_report.json`, 8 behavioral skill diffs. |
| **Reward & Training Pipeline (10%)** | GRPO with compact-JSON / single-JSON / cap-hit / multi-JSON / extra-text penalties; trajectory-grounded action-reward lookup; structured diagnostics every N batches with unhealthy-signal warnings. |

---

## Quick Reproduction (Colab)

Open `train_notebook.ipynb` and run in order:

| Step | Purpose | Typical time |
|---|---|---|
| 1 | Clone repo | < 10 s |
| 2 | Install deps (Unsloth + TRL + peft + bitsandbytes) | ~3 min |
| 3 | Dry-run sanity: `python train.py --dry-run` | < 30 s |
| 4 | Baseline evaluation (writes `artifacts/eval/baseline_report.json`) | ~1 min |
| 5a | Quick-mode GRPO training (`--iterations 10 --episodes 15 --k 2`) | ~1.5–2 hrs on T4 |
| 5b | Full-mode GRPO training (`--iterations 20 --episodes 30 --k 4`) | ~6–8 hrs on T4 |
| 6 | Post-training compare: `evaluate.py --policy compare --compare-trained-policy trained_checkpoint` | ~5–15 min |
| 7–8 | Inspect reports and plot | instant |
| 9–10 | (Optional) two-seed reproducibility | 2× Step 5 time |

Artifacts land in `artifacts/...` and can be downloaded from the Colab file browser.  
Step 5 also writes an internal checkpoint-eval snapshot under `artifacts/train/checkpoint_eval/`; Step 6 writes judge-facing compare outputs under `artifacts/eval/`.

---

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

| Tier | Count | Steps | Customers | Characteristics |
|---|---:|---:|---:|---|
| easy | 3 | 40 | 2–3 | single failure, clear root cause |
| medium | 5 | 50 | 4–6 | cascading issues, red herrings, one policy drift |
| hard | 7 | 70 | 8–12 | deeper cascades, outdated KB, multiple drifts |
| nightmare | 3 | 80 | 10–15 | compound faults, high noise, maximum complexity |

### Partial observability

**Hidden from agent (must be discovered):** true root cause, full dependency graph on hard+, KB accuracy, current policy values, internal customer risk, red-herring symptoms.

**Visible:** alert text, tool results, accumulated `known_facts`, stakeholder patience, SLA/step counter, phase-restricted `available_actions`.

---

## Reward Design

Per-step deterministic rewards shaped by:

- structural validity + phase availability of the action
- investigation-before-action, KB cross-verification, policy-awareness
- root-cause correctness of `apply_fix` (with CAB approval gate + blast-radius penalty on wrong fixes)
- customer tone matching against sentiment
- stakeholder proactivity before patience decays
- resource budgets (max fix attempts, escalations, notifications)

Training-time reward adds strict output-shape penalties (single JSON, no extra prose, cap-hit penalty) so the agent is pushed toward compact, parseable actions rather than noisy prose.

No LLM judge is used. All reward logic is in the repo (`graders/` and `env/`).

---

## Training Pipeline

- `train.py` — collects trajectories from the live environment across a curriculum (easy → nightmare), runs GRPO (Unsloth + TRL) on Qwen2.5-3B with LoRA, saves adapter to `artifacts/train/trained_adapter/`, then runs baseline vs trained evaluation with dual-panel reward plots.
- `evaluate.py` — deterministic incident episodes per difficulty tier; reports normalized / raw reward, SLA compliance, root-cause accuracy, long-horizon consistency, 8 behavioral skills, and `policy_used` provenance.
- `train_notebook.ipynb` — Colab-first flow with dry-run, baseline eval, training (quick or full), compare eval, plotting, and optional two-seed reproducibility.

### Key CLI examples

```bash
# Dry-run sanity (local, no GPU needed)
python train.py --iterations 1 --episodes 1 --k 2 --dry-run

# Quick training (Colab T4, ~1.5-2 hrs)
python train.py --iterations 10 --episodes 15 --k 2 --max-completion-length 96

# Full training (Colab T4, ~6-8 hrs)
python train.py --iterations 20 --episodes 30 --k 4 --max-completion-length 128

# Heuristic compare (fast evidence)
python evaluate.py --policy compare --episodes-per-difficulty 5 --plot --output-dir artifacts/eval

# Checkpoint-based compare (use after a real training run)
python evaluate.py --policy compare \
  --compare-trained-policy trained_checkpoint \
  --checkpoint-dir artifacts/train/trained_adapter \
  --checkpoint-base-model Qwen/Qwen2.5-3B-Instruct \
  --episodes-per-difficulty 5 --plot --output-dir artifacts/eval
```

---

## Evaluation Metrics

Per-episode:
- Normalized reward (clamped to `[0, 1]`)
- Raw cumulative reward (can be negative, exposed explicitly)
- SLA compliance (pending tickets == 0 at end)
- Root-cause accuracy (at least one correct `apply_fix` landed)
- Long-horizon consistency (post-fix actions stay in the consistent set)
- 8 tracked skills: investigation-before-action, KB cross-verification, policy checking, stakeholder proactivity, root-cause accuracy, tone matching, resource efficiency, red-herring dismissal

Aggregates: `avg_normalized_reward`, `avg_raw_reward`, per-difficulty averages, `policy_used`, `episodes_per_difficulty`.

---

## Results

This repository ships the full training and evaluation pipeline. Real numbers are produced by running the notebook; we don't hardcode claims into this README.

When you run the full Colab notebook end-to-end on a T4:
- Training output: `artifacts/train/trained_adapter/`
- Evaluation output: `artifacts/eval/reward_curves.png`, `baseline_report.json`, `trained_report.json`

> **`policy_used` field:** both `baseline_report.json` and `trained_report.json` include `policy_used` so you can confirm whether trained-side numbers came from the **real Qwen LoRA checkpoint** (`trained_checkpoint`) or the **deterministic heuristic fallback** (`trained_heuristic`). We keep this explicit in logs/reports for transparency.

---

## HTTP API (OpenEnv-compatible)

Endpoints:

- `POST /reset`, `POST /step`, `GET /state`, `POST /close`
- `GET /health`, `POST /inference`

Optional production headers:
- `X-Session-ID`: isolate concurrent episodes per client/session
- `X-API-Key`: required only when the server is configured with `OPENENV_API_KEY`

Example:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"mode":"incident","difficulty":"easy","seed":0}'
```

---

## Setup (local)

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Reproducible runtime dependencies are pinned in `requirements.lock`.

---

## Deployment (Hugging Face Spaces)

```bash
docker build -t eicc .
docker run -p 7860:7860 eicc
curl http://localhost:7860/reset -X POST \
  -H "Content-Type: application/json" \
  -d '{"mode":"incident","difficulty":"easy"}'
```

The container defaults to 1 worker with a healthcheck on `/health`. Episode state is in-process, so horizontal scaling requires an external session store.

---

## Determinism Contract

- Seeded scenario selection and service-mesh behavior
- No LLM judge in scoring
- No wall-clock / time-based randomness
- Same `(seed, difficulty, mode)` triple deterministically yields the same scenario
- Greedy decode (`do_sample=False`) for checkpoint evaluation

---

## Demo Assets

- Architecture diagram: `demo/architecture_diagram.md`
- Incident walkthrough: `demo/incident_resolution_walkthrough.md`
- 3-minute pitch script: `demo/pitch_script_3min.md`
- 2-minute video script: `demo/video_script_2min.md`
- HF mini-blog draft: `demo/hf_mini_blog.md`
- Reward curve PNG is produced by `evaluate.py` into `--output-dir` (default `artifacts/eval/reward_curves.png`)

---

## Public Submission Links

> These links are finalized at submission time. Replace placeholders as soon as assets are published.

- Hugging Face Space: [anuj2209-openenv-customer-support.hf.space](https://anuj2209-openenv-customer-support.hf.space)
- Hugging Face mini-blog: *pending publish — draft in `demo/hf_mini_blog.md`*
- YouTube demo (<2 min): *pending publish — script in `demo/video_script_2min.md`*
- Optional slide deck: *pending*

---

## Hackathon Submission Checklist

- [x] OpenEnv-based environment with manifest (`openenv.yaml`)
- [x] Training script + Colab notebook using Unsloth + HF TRL (`train.py`, `train_notebook.ipynb`)
- [x] Hugging Face Space deployment + link in README
- [x] README motivates the problem, explains the env, and describes the results flow
- [x] Evaluation pipeline emits baseline + trained reports and a dual-panel reward curve
- [x] Checkpoint-based evaluation path (`--policy trained_checkpoint`) + heuristic fallback with `policy_used` provenance
- [x] HF mini-blog draft committed (`demo/hf_mini_blog.md`)
- [x] Pitch and video scripts committed (`demo/pitch_script_3min.md`, `demo/video_script_2min.md`)
- [ ] Publish the HF mini-blog and paste URL in **Public Submission Links**
- [ ] Publish the <2 min YouTube demo and paste URL in **Public Submission Links**
- [ ] Commit one real run snapshot (plot + reports) before final submission

---

## Repository Layout

```text
env/           # world + systems + environment dispatch/state machine
models/        # pydantic action/observation/incident schemas
graders/       # deterministic ticket + incident graders
tasks/         # ticket bank + incident bank + scenario data
tests/         # full backward-compat + incident test suite (404 tests)
train.py       # GRPO training pipeline
evaluate.py    # deterministic evaluation + before/after comparison + plotting
openenv.yaml   # OpenEnv manifest
```

---

## License

MIT
