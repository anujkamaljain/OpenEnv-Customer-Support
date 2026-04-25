# EICC: Training Incident-Response Agents in OpenEnv

Enterprise incident response is a world-modeling task, not a single-step QA task.  
Agents must diagnose causality across systems, update beliefs as tools return evidence, and execute long-horizon workflows under policy and SLA constraints.

## What we built

EICC (Enterprise Incident Command Center) extends OpenEnv customer support into a realistic incident world:

- **21 typed actions** across triage, investigation, response, resolution
- **5-service dependency mesh** with cascading failures
- **8 enterprise subsystems** (monitoring, CRM, billing, KB, policy, history, runbooks, stakeholders)
- **partially observable state** (root cause hidden, evidence must be collected)
- **deterministic rewards + seeded reproducibility**

## Why it is useful

This environment teaches capabilities current LLM agents often lack in production:

1. Investigate before acting
2. Cross-verify uncertain evidence
3. Coordinate tools + communication in multi-step loops
4. Maintain consistent internal state over long episodes

## Training + Evaluation

We train with GRPO (Unsloth + TRL) in Colab and evaluate:

- normalized reward
- raw cumulative reward (to expose negative baselines)
- root-cause accuracy
- long-horizon consistency
- structured behavior diffs
- explicit policy provenance via `policy_used` (`trained_checkpoint` vs `trained_heuristic`)

Quick local smoke:

```bash
python train.py --iterations 1 --episodes 1 --k 2 --dry-run
python evaluate.py --policy compare --episodes-per-difficulty 5 --plot
```

Checkpoint-based evaluation:

```bash
python evaluate.py --policy compare \
  --compare-trained-policy trained_checkpoint \
  --checkpoint-dir artifacts/train/trained_adapter \
  --checkpoint-base-model Qwen/Qwen2.5-3B-Instruct \
  --episodes-per-difficulty 5 --plot --output-dir artifacts/eval
```

## Reproducibility note

Run at least two seeds and report mean ± std on key metrics for final submission.

---

HF Space: https://huggingface.co/spaces/Anuj2209/openenv-customer-support  
GitHub: https://github.com/anujkamaljain/OpenEnv-Customer-Support  
Video (<2 min): TODO_ADD_YOUTUBE_URL
