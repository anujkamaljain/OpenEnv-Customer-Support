# 2-Minute Video Script

## 0:00 - 0:18 Hook

"It is 3 AM. Payments are failing, alerts are noisy, and stakeholders are escalating.  
Can an AI agent resolve this without guessing?"

## 0:18 - 0:45 Problem

"Most AI benchmarks test isolated skills.  
Real enterprise incidents require partial observability, tool usage, policy-aware decisions, and long-horizon execution."

## 0:45 - 1:15 Environment

"We built EICC on OpenEnv:
- 5 cascading services,
- 8 enterprise systems,
- 21 typed actions across triage, investigation, response, and resolution.

The agent cannot see root cause directly. It must query tools and update beliefs."

## 1:15 - 1:43 Training + Evidence

"We train with GRPO in Colab using Unsloth + TRL, then compare baseline vs trained policy.
We track normalized reward, raw cumulative reward, root-cause accuracy, and long-horizon consistency."

"Result snapshot: baseline tends to accumulate negative raw reward, while trained policy shifts positive and improves root-cause handling."

"We also log `policy_used` for transparency, so judges can verify whether a run used the checkpoint policy or a guarded fallback."

"And we now publish a Sim→Sandbox transfer report to show how much trained behavior carries over from simulation to live container-backed infrastructure."

"We also run a deterministic drill mode where new failures appear mid-episode, and track a drill score for recovery quality."

## 1:43 - 2:00 Close

"EICC is deterministic, reproducible, and deployed on Hugging Face Spaces.  
It is a practical benchmark for training AI incident commanders in multi-app enterprise workflows."
