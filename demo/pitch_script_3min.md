## 3-Minute Pitch Script (Top-tier judges)

### 0:00-0:25 — Hook

"It is 3 AM. Payments are failing. Alerts are noisy. Customers are escalating.  
Most AI agents fail here because they guess actions instead of building a world model."

"We built the **Enterprise Incident Command Center (EICC)** on OpenEnv to train agents for this exact failure mode."

### 0:25-0:55 — Problem

"Most benchmarks test isolated skills. Real enterprise operations require:
- partial observability,
- tool-driven diagnosis,
- policy-aware decisions,
- and multi-step execution under pressure."

"Our question: can an LLM learn to investigate first, update beliefs, and resolve incidents causally instead of shortcutting?"

### 0:55-1:35 — Environment

"EICC simulates a fintech incident world:
- 5 interdependent services with cascading failures,
- 8 enterprise systems: monitoring, CRM, billing, KB, policy, incident history, runbooks, stakeholders,
- 21 typed actions across triage, investigation, response, and resolution,
- deterministic rewards and reproducible seeds."

"The agent cannot see root cause directly. It must call tools, interpret outcomes, and coordinate fixes and communications."

### 1:35-2:20 — Training + Evidence

"We train with GRPO using Unsloth/TRL in Colab.  
Pipeline: collect trajectories from the environment, train adapter, then evaluate baseline vs trained policy."

"We report:
- normalized reward,
- raw cumulative reward (including negative baseline episodes),
- root-cause accuracy,
- long-horizon consistency,
- and behavior diffs."

"Key behavior shift: trained policy checks monitoring and policy earlier, verifies evidence, and executes resolution steps more consistently."

### 2:20-2:45 — Why this is novel

"This is not a toy grid world. It is a partially observable, multi-app enterprise workflow where naive reward hacking fails.  
The reward model is dense, structured, and tied to causal outcomes, not a single terminal success bit."

### 2:45-3:00 — Close

"EICC aligns directly with OpenEnv Professional Tasks and Scaler's multi-app enterprise theme.  
It is reproducible, deployable on Hugging Face Spaces, and built to train agents that can reason under operational uncertainty."

"If judges want, we can run a live reset-step-state episode now and show the trained policy trace."
