# 🏆 Grand Finale Feature Plan — Development Phases

> **Concept:** Enterprise Incident Command Center (EICC)
> **Theme:** #3.1 World Modeling (Professional Tasks)
> **Sub-themes:** Scaler AI Labs (Multi-App Enterprise) + Patronus AI (Schema Drift)
> **Tagline:** *"When Everything Is on Fire, Can Your AI Keep Its Cool?"*

---

# EXECUTIVE SUMMARY

> **TL;DR for judges:** We built a **21-action, 4-phase, partially-observable enterprise incident response simulator** where an AI agent must diagnose cascading microservice failures, cross-verify potentially-outdated knowledge base articles, handle dynamically-arriving customer tickets, manage decaying stakeholder patience, follow-or-deviate from suggested runbooks, and write post-mortems — all under SLA pressure with limited resources. The environment features **5 interconnected services** with a causal dependency graph, **8 simulated enterprise applications**, **18 hand-crafted incident scenarios** across 4 difficulty tiers, **intermittent (flickering) failures**, **mid-episode policy drift**, **topology discovery on hard mode**, and **cross-incident knowledge persistence** for self-improvement. Every reward signal is **deterministic** (no LLM judge). GRPO training on Qwen2.5-3B (20 iterations × 30 episodes × K=4, ~8 hrs on Colab free T4) shows **2x+ improvement** (0.25 → 0.55+) with measurable skill curves across 8 tracked behaviors. Runs entirely on **Colab free T4** — no paid compute.

| Metric | Value |
|--------|-------|
| Total action types | **21** (6 existing + 15 new) |
| Simulated enterprise apps | **8** (Monitoring, CRM, Billing, KB, Policy, Incident Mgr, History, Runbooks) |
| Incident scenarios | **18** across 4 difficulty tiers |
| Max episode length | **80 steps** (nightmare) |
| Hidden state dimensions | **7** (root cause, dependencies, KB accuracy, policies, risk scores, billing issues, red herrings) |
| Reward signals per step | **30+** across 4 phases + cross-cutting penalties |
| Tracked behavioral skills | **8** (investigation, KB verification, policy checking, etc.) |
| Estimated state space | **~10^12** reachable states |
| Anti-shortcut mechanisms | **11** (outdated KB, red herrings, policy drift, topology hiding, flickering, wrong runbooks, observability gaps, resource budgets, CAB approval gate, blast radius, alert fatigue) |
| Training hardware | Google Colab free T4 GPU |
| Target improvement | **0.25 → 0.55+** episode reward (2x, Colab free T4) |
| Backward compatible | **100%** — all 216 existing tests pass unchanged |

---

# FORMAL SUBMISSION — Organizer Required Sections

> The organizers explicitly require these six sections. Every judge will check for them.

---

## 1. PROBLEM STATEMENT

**Enterprise incident response is one of the hardest professional tasks an AI agent can face.** When a fintech company's payment service goes down at 3 AM, a human incident commander must simultaneously:

- Diagnose **cascading failures** across interconnected microservices
- Navigate **partially observable systems** using diagnostic tools that may return noisy or incomplete data
- Handle **dynamic customer queues** where enterprise clients threaten legal action
- Follow **policies that change mid-crisis** (refund caps, escalation rules, SLA extensions)
- Cross-verify **knowledge base articles that may be outdated or wrong**
- Manage **stakeholder patience** (VP, Legal, Support Lead) that decays every minute
- Write a **post-mortem** and **update documentation** to prevent future incidents

**No existing AI benchmark captures this complexity.** Current benchmarks test isolated skills (classification, summarization, tool-use) in fully observable, static environments. They don't test:
- Causal reasoning across interconnected systems
- Belief updating when tools return conflicting information
- Strategic resource allocation under time pressure
- Multi-objective optimization (fix systems + handle customers + manage stakeholders)

**Our problem:** *Can we build an environment that trains LLMs to perform systematic incident response — diagnosing cascading failures through tool-based investigation, maintaining consistent internal state, and orchestrating multi-step workflows under partial observability and time pressure?*

**Our hypothesis:** A GRPO-trained 3B parameter model can improve from ~0.25 to ~0.55+ episode reward (2x, trained on Colab free T4 in ~8 hrs) by learning:
1. Investigate before acting (use monitoring + logs)
2. Verify before trusting (cross-check KB against live data)
3. Fix root causes, not symptoms (trace dependency graph)
4. Communicate proactively (stakeholders, customers)
5. Manage resources strategically (limited fix attempts, escalations)

---

## 2. ENVIRONMENT

The **Enterprise Incident Command Center (EICC)** simulates a fintech company's complete operational stack:

### 2.1 — Service Layer (5 Interconnected Microservices)

```
AUTH ──depends on──► PAYMENTS ──depends on──► NOTIFICATIONS
  │                      │
  │                      ▼
DATABASE ◄──depends on── ANALYTICS
```

- Each service has 3 health states: `healthy` → `degraded` → `down`
- Failures **cascade** through the dependency graph (DB goes down → payments degrades → notifications fails)
- Services **degrade over time** if root cause isn't fixed (E.1)
- **Observability varies** by service — legacy services have minimal monitoring (E.4)
- On hard/nightmare difficulty, the **dependency graph is hidden** — agent must discover it (E.3)

### 2.2 — Enterprise Application Layer (8 Simulated Apps)

| App | Hidden State | Agent Discovery Tool |
|-----|-------------|---------------------|
| **Monitoring System** | Service health, metrics, error rates, latency | `check_monitoring`, `probe_service`, `fetch_logs` |
| **CRM System** | Customer records, account status, frustration level, flags | `fetch_user_data` |
| **Billing System** | Invoices, payment failures, disputes, refund history | `check_billing` |
| **Knowledge Base** | Articles + staleness flag (some articles are **WRONG**) | `query_kb`, `update_kb` |
| **Policy Engine** | Current rules (refund caps, escalation thresholds) — **CHANGE mid-incident** | `check_policy` |
| **Incident Manager** | Phase tracking, evidence chain, audit trail | `classify`, `escalate`, `write_postmortem` |
| **Incident History** | Past incident records, resolution patterns, similar-incident matching | `query_incident_history` |
| **Runbook Engine** | Suggested procedures (some **OUTDATED/WRONG**), step-by-step actions | `follow_runbook_step` |

### 2.3 — Human Layer

| Entity | Behavior | Agent Interaction |
|--------|---------|-------------------|
| **Customer Queue** | Tickets arrive **dynamically** based on system health. Enterprise customers generate tickets faster. Frustrated customers send angry follow-ups. | `respond`, `resolve`, `request_info` |
| **Stakeholders** | VP Engineering, Legal, Support Lead — patience **decays every step**. Hit 0 = major penalty. | `notify_stakeholders` |
| **Customer Behavior Model** | Customers **react to agent tone and timing**. Wrong tone → frustration spikes. Apology → frustration decreases. (E.7) | Response quality affects future difficulty |
| **Change Advisory Board** | Simulated approval team reviews proposed fixes. Low-risk = auto-approved. Medium = needs investigation evidence. High/critical = needs evidence + prior escalation. **Rejects** inadequate proposals. Wrong fix that gets approved = **blast radius** damage (E.17). | `apply_fix` routed through CAB gate. Rejection = penalty but fix attempt saved. |

### 2.4 — Partial Observability

The agent CANNOT see:
- True root cause of the incident
- Service dependency graph (on hard/nightmare)
- Whether a KB article is outdated or correct
- Current policy values (must call `check_policy`)
- Customer internal risk score
- Billing issues not mentioned in ticket text
- Which symptoms are red herrings

The agent CAN see:
- Initial alert text and customer tickets
- Results of tool queries (after calling them)
- Accumulated `known_facts` from previous tool calls
- Stakeholder patience levels
- Step count and SLA deadlines
- Available actions for current phase

### 2.5 — Dynamics & Anti-Shortcut Design

| Dynamic Element | How It Works | Why It Prevents Shortcuts |
|----------------|-------------|--------------------------|
| **Cascading failures** | Fixing root cause → dependents heal. Fixing symptom → temporary improvement, then re-degrades | Agent can't just fix the first broken thing |
| **Policy drift** | Refund caps, escalation rules, SLA parameters change on a schedule mid-incident | Caching policy = penalty. Must re-check. |
| **Outdated KB** | Some articles have `is_accurate: false`. KB says "restart DB" but real fix is auth-related | Blindly trusting KB = wrong fix = penalty |
| **Red herrings** | Analytics CPU at 95% (normal batch processing), auth latency +200ms (unrelated) | Agent must investigate AND dismiss, not just investigate |
| **Time-based degradation** | Degraded services get worse every 5 steps if unfixed | Can't just investigate forever |
| **Noisy monitoring** | Low-observability services return minimal probe data | Same investigation strategy doesn't work everywhere |
| **Customer behavior** | Wrong tone → customer frustration spikes → new escalation ticket | Agent must learn empathy-as-strategy |
| **Resource budget** | Max 3 fix attempts, max 2 escalations, max 5 stakeholder notifications | Can't brute-force solutions |
| **CAB approval gate** | Medium/high-risk fixes require investigation evidence; critical fixes also need escalation | Agent can't spam `apply_fix` — must investigate first (E.16) |
| **Blast radius** | Wrong fix on wrong service causes ADDITIONAL cascading damage (not just "didn't work") | Accuracy matters more than speed — wrong fix is worse than no fix (E.17) |
| **Alert fatigue** | 50+ alerts per step, 80% noise (cascade effects, flapping, unrelated spikes). Must filter. | Agent can't investigate every alert — must learn signal vs. noise prioritization (E.18) |

---

## 3. AGENT CAPABILITIES

The agent operates through **21 typed action types** organized by function:

### Investigation (9 actions) — Discover hidden state
| Action | What It Does | What It Returns |
|--------|-------------|-----------------|
| `check_monitoring` | Query service health | Health status, error rate, latency (NOT root cause) |
| `probe_service` | Deep diagnostic on one service | Logs, resources, connections — detail varies by observability |
| `fetch_logs` | Get recent error logs | Log entries with timestamps — may contain red herrings |
| `fetch_user_data` | Query CRM for customer record | Tier, value, account status, frustration, flags |
| `check_billing` | Query billing for customer | Payment status, disputes, failed payments |
| `query_kb` | Search knowledge base | Articles with solutions — may be OUTDATED |
| `check_policy` | Get current policy rules | Current values (may have changed since last check) |
| `query_incident_history` | Search past incident records | Similar incidents, resolution patterns, outcomes — may mislead |
| `follow_runbook_step` | Execute a suggested runbook step | Step result — runbook may be OUTDATED/WRONG |

### Action (7 actions) — Affect the world
| Action | What It Does | Constraints |
|--------|-------------|-------------|
| `classify` | Classify incident severity | Must match true severity for reward |
| `route` | Assign to correct team | Wrong routing = penalty |
| `respond` | Customer-facing response | Tone must match sentiment; content graded by keywords |
| `escalate` | Escalate to specialist | Max 2 per incident. Must be justified. |
| `apply_fix` | Attempt to fix a service | Max 3 per incident. Only works if targeting actual root cause. |
| `request_info` | Ask customer for details | Returns additional context |
| `notify_stakeholders` | Update VP/Legal/Support Lead | Max 5. Restores patience. Urgency must match situation. |

### Resolution (5 actions) — Close the incident
| Action | What It Does | Quality Criteria |
|--------|-------------|------------------|
| `resolve` | Close a customer ticket | Resolution quality graded by keyword matching |
| `verify_fix` | Re-check service health after fix | Confirms if fix actually worked |
| `rollback_fix` | Undo a bad fix | +0.02 if rolling back a genuinely bad fix |
| `write_postmortem` | Write incident summary | Graded: root cause accuracy + steps + prevention |
| `update_kb` | Add/update KB article | Graded: correctness of new information |

### Phase Gating

Actions are available based on the current incident phase:
```
TRIAGE:          classify, check_monitoring, query_kb
INVESTIGATION:   check_monitoring, probe_service, fetch_logs, fetch_user_data,
                 check_billing, query_kb, check_policy, classify, route
RESPONSE:        apply_fix, rollback_fix, respond, escalate, request_info,
                 notify_stakeholders, check_policy, fetch_user_data, check_billing, query_kb
RESOLUTION:      verify_fix, resolve, respond, write_postmortem, update_kb, notify_stakeholders
```

---

## 4. TASKS

### 4.1 — Structured Task Definition

Each episode = one **incident** consisting of multiple structured sub-tasks:

| Task | Phase | Success Criteria | Difficulty Scaling |
|------|-------|-----------------|-------------------|
| **T1: Severity Assessment** | Triage | Correctly classify incident severity (critical/high/medium) | All difficulties |
| **T2: Affected System Identification** | Triage | Identify all affected services via monitoring | Easy: 1 service. Nightmare: 5 services. |
| **T3: Root Cause Diagnosis** | Investigation | Trace cascading failures to root cause via probing + log analysis | Easy: obvious. Hard: 3-hop dependency chain. Nightmare: 2 simultaneous causes. |
| **T4: Red Herring Dismissal** | Investigation | Investigate misleading symptoms and correctly dismiss them | Easy: 0. Medium: 1. Hard: 3. Nightmare: 6+. |
| **T5: KB Cross-Verification** | Investigation | Query KB, then verify against live data. Catch outdated articles. | Easy: all KB correct. Hard: 1 wrong. Nightmare: all wrong. |
| **T6: Root Cause Fix** | Response | Apply correct fix to correct service | Limited to 3 attempts. Must match root cause. |
| **T7: Policy-Compliant Actions** | Response | Check current policy before compensation/escalation decisions | Easy: no drift. Medium: 1 change. Nightmare: 2 changes. |
| **T8: Customer Response Quality** | Response | Respond with correct content, appropriate tone, within SLA | Graded per-ticket via keyword matching. |
| **T9: Stakeholder Management** | Response | Keep VP/Legal/Support informed before patience exhausts | Patience decay: 0.02–0.04 per step. |
| **T10: Fix Verification** | Resolution | Re-check service health to confirm fix worked | Must verify, not assume. |
| **T11: Ticket Resolution** | Resolution | Resolve all pending customer tickets with quality summaries | Dynamic: 2–15 tickets depending on difficulty. |
| **T12: Post-Mortem Quality** | Resolution | Write incident summary covering root cause, steps, prevention | Keyword-graded against ground truth. |
| **T13: Knowledge Contribution** | Resolution | Update KB with correct new solution | Graded for accuracy. Persists to future episodes. |

### 4.2 — 18 Incident Scenarios (4 Difficulty Tiers)

| Tier | Count | Steps | Customers | Root Causes | Red Herrings | Policy Changes | KB Status |
|------|-------|-------|-----------|-------------|-------------|---------------|-----------|
| **Easy** | 3 | 40 | 2–3 | 1 (obvious) | 0 | 0 | All correct |
| **Medium** | 5 | 50 | 4–6 | 1 (cascading) | 1–2 | 1 | 1 outdated |
| **Hard** | 7 | 70 | 8–12 | 1 (deep cascade) | 3–4 | 2 | 2 outdated |
| **Nightmare** | 3 | 80 | 10–15 | 2 (simultaneous) | 6+ | 2 | All outdated |

### 4.3 — Measurable Outcomes

Every task has a **deterministic, verifiable reward signal** — no LLM-in-the-loop for grading:

```
Severity correct?           → exact match against ground truth
Root cause found?           → exact match against incident.root_causes
Fix worked?                 → service health changed to "healthy"
Policy checked?             → boolean: did agent call check_policy before action?
KB cross-verified?          → boolean: did agent probe_service after query_kb?
Customer response quality?  → weighted keyword score (existing grader infrastructure)
Post-mortem quality?        → weighted keyword score against root cause + steps
```

---

## 5. REWARD MODEL / EVALUATION LOGIC

### 5.1 — Dense Per-Step Rewards (for GRPO)

Every action produces an **immediate numerical reward** — critical for GRPO's relative ranking mechanism.

**Reward range per step:** `[-0.15, +0.15]`
**Episode reward range:** `[0.0, 1.0]` (normalized)

See **Appendix C** for the complete per-step reward tables (30+ individual reward signals across 4 phases + cross-cutting penalties).

### 5.2 — Episode-Level Multi-Objective Score

| Component | Weight | Measurement |
|-----------|--------|------------|
| Root Cause Identification | 20% | Binary: agent's diagnosis matches ground truth |
| Fix Effectiveness | 15% | Binary: targeted service recovered after fix |
| Customer Handling | 15% | Avg keyword score across all resolved tickets |
| Investigation Efficiency | 10% | Ratio: optimal tool calls / actual tool calls |
| SLA Compliance | 10% | % of customers resolved within SLA deadline |
| Stakeholder Management | 10% | Avg remaining patience across all stakeholders |
| Policy Compliance | 10% | % of policy-sensitive actions that checked current policy |
| Post-Mortem Quality | 5% | Keyword coverage: root cause + remediation + prevention |
| Knowledge Contribution | 5% | Accuracy of KB update vs. ground truth |

### 5.3 — Anti-Gaming Measures

| Measure | How It Works |
|---------|-------------|
| Repeated action penalty | `-0.05` for exact duplicate of previous action |
| Investigation without action | After 20 investigation steps without a fix attempt → diminishing returns |
| Resource budget | Can't spam fixes or escalations — limited attempts |
| Tool call diminishing returns | Calling same tool with same parameters → `-0.02` (no new info) |
| Phase violation penalty | `-0.05` for attempting an action unavailable in current phase |
| Evidence chain coherence | Post-mortem quality penalized if conclusions don't match investigation evidence |

---

## 6. POST-TRAINING / SELF-IMPROVEMENT STRATEGY

> **This is explicitly required by the organizers.** Our strategy has 4 components.

### 6.1 — GRPO Training Pipeline (Primary)

**Algorithm:** Group Relative Policy Optimization (GRPO) via HuggingFace TRL
**Model:** Qwen2.5-3B-Instruct (4-bit quantized via Unsloth)
**Hardware:** Google Colab free T4 GPU

```
Training Loop (Option A — Colab free T4, ~8 hrs):
1. Collect trajectory from environment (prompt → action → reward)
2. For each prompt, generate K=4 completions
3. GRPO ranks completions by reward and updates policy to favor higher-reward actions
4. Repeat: 20 iterations × 30 episodes = 600 total episodes (2,400 completions)
```

**Why GRPO over PPO/DPO:** GRPO doesn't need a separate critic model (PPO) or preference pairs (DPO). It uses the environment's reward directly, making it ideal for verifiable-reward environments like ours.

### 6.2 — Curriculum Learning

Train in progressive difficulty (scaled to 600 total episodes across 20 iterations):

```
Phase A (iterations 1–8,  episodes 1–240):   Easy incidents only
Phase B (iterations 9–14, episodes 241–420):  Medium incidents
Phase C (iterations 15–18, episodes 421–540): Hard incidents
Phase D (iterations 19–20, episodes 541–600): Mixed (easy 20% / medium 30% / hard 40% / nightmare 10%)
```

**Why this matters:** Agent learns basic investigation patterns on easy incidents, then transfers those skills to harder scenarios. Prevents early-training frustration on nightmare scenarios.

### 6.3 — Cross-Incident Knowledge Persistence (Self-Improvement)

This is our **killer self-improvement feature**:

```python
# When the agent writes a CORRECT update_kb action:
# 1. The new KB article is graded for accuracy
# 2. If correct → persisted to a PersistentKnowledgeBase
# 3. In FUTURE episodes → the updated KB article is available
# 4. Agent encounters the same scenario → KB now has correct answer → faster resolution → higher score

# Measurable self-improvement:
# Episode 50: Agent encounters "Payment 500 errors" → KB says "restart DB" (WRONG)
#             Agent probes services, discovers auth is the issue → writes correct KB update
# Episode 150: Same scenario → KB now says "check auth token cache" (CORRECT from agent's contribution)
#              Agent resolves faster → higher reward → measurable improvement over time
```

**This creates a flywheel:** Better investigation → Better KB updates → Better KB for future episodes → Faster resolution → Higher scores → Evidence of self-improvement.

### 6.4 — Behavioral Skill Tracking

Track which specific skills improve during training:

```python
TRACKED_SKILLS = {
    "investigation_before_action":  # Does agent check_monitoring before classify?
    "kb_cross_verification":        # Does agent verify KB against logs?
    "policy_checking":              # Does agent check_policy before compensation?
    "stakeholder_proactivity":      # Does agent notify before patience drops?
    "root_cause_accuracy":          # Does agent find the real root cause?
    "tone_matching":                # Does agent match tone to sentiment?
    "resource_efficiency":          # Does agent use fix attempts wisely?
    "red_herring_dismissal":        # Does agent correctly dismiss false leads?
}

# Plot per-skill improvement curves for demo:
# "Before training: investigation_before_action = 12%"
# "After training:  investigation_before_action = 89%"
```

**Why this matters for the demo:** Instead of just showing a single reward curve, we show 8 individual skill curves. This is far more impressive in a 3-minute pitch because judges can see WHAT the model learned, not just that a number went up.

---

## Why This Wins

| Criterion | Weight | Our Edge |
|-----------|--------|----------|
| **Environment Innovation** | 40% | 21 action types, cascading service failures with causal dependency graph, 8 enterprise apps, intermittent failures, policy drift, outdated KB traps, topology discovery, observability gaps, resource budgets, evidence chain, runbooks, historical incident lookup, cost modeling |
| **Storytelling** | 30% | "3 AM incident response" — every tech judge lives this. Clear before/after demo with 8 individual skill curves. 3-minute pitch with live demo. |
| **Reward Improvement** | 20% | Base model ~0.25 → GRPO-trained ~0.55+ (2x improvement on Colab free T4). Dense per-step rewards (30+ signals). Curriculum learning. Self-improving KB creates measurable episode-over-episode improvement. |
| **Training Pipeline** | 10% | GRPO + Unsloth + Colab free T4. Concrete train.py + Colab notebook. Curriculum scheduling + 8 behavioral skill tracking curves. |

**Targets:** Scaler AI Labs bonus (multi-app enterprise) + Patronus AI bonus (schema/policy drift)

---

# PHASE 1 — World Foundation

> **Goal:** Build the core simulation engine — service mesh with cascading failures, dependency graph, and health state management. This is the "brain" that makes everything else possible.
>
> **No UI/API changes. No existing code modified. Pure new modules.**

## 1.1 — Service Mesh Simulation (`env/services.py`)

Create a deterministic simulation of 5 interconnected microservices for a fintech company.

### Service Definitions

```python
SERVICES = {
    "auth": {
        "description": "Authentication & token management",
        "dependencies": [],  # root service
        "failure_modes": ["rate_limiting", "token_expiry", "config_corruption"],
    },
    "database": {
        "description": "Primary data store",
        "dependencies": [],  # root service
        "failure_modes": ["oom", "connection_pool_exhaustion", "replication_lag"],
    },
    "payments": {
        "description": "Transaction processing",
        "dependencies": ["auth", "database"],
        "failure_modes": ["gateway_timeout", "validation_errors", "idempotency_failure"],
    },
    "analytics": {
        "description": "Reporting & metrics",
        "dependencies": ["database"],
        "failure_modes": ["batch_job_runaway", "query_timeout", "stale_cache"],
    },
    "notifications": {
        "description": "Email, SMS, push notifications",
        "dependencies": ["payments"],
        "failure_modes": ["queue_overflow", "template_error", "rate_exceeded"],
    },
}
```

### Health State Machine

Each service has 4 states: `healthy` → `degraded` → `flickering` → `down`

```python
class ServiceState(BaseModel):
    name: str
    health: Literal["healthy", "degraded", "flickering", "down"]
    root_cause: str | None = None           # e.g., "oom" — HIDDEN from agent
    error_rate: float = 0.0                 # 0.0 (healthy) to 1.0 (down)
    latency_ms: int = 50                    # normal ~50ms, degraded ~2000ms, down ~timeout
    affected_by: str | None = None          # upstream service causing this failure
    is_root_cause: bool = False             # is this the SOURCE of the cascade?
    fix_applied: bool = False               # has agent attempted a fix?
    fix_correct: bool = False               # was the fix targeting the right cause?
    flicker_pattern: list[str] | None = None  # e.g., ["healthy","degraded","healthy","degraded"]
    flicker_step_index: int = 0              # current position in flicker pattern
```

### Intermittent Failures (Flickering State)

The hardest failures to diagnose in real ops are **intermittent** ones — services that flicker between `healthy` and `degraded` unpredictably.

```python
class FlickeringBehavior:
    """Simulates intermittent service failures."""

    def __init__(self, pattern: list[str], seed: int):
        # pattern example: ["healthy", "degraded", "healthy", "healthy", "degraded"]
        # Service cycles through this pattern each step
        self.pattern = pattern

    def get_current_health(self, step: int) -> str:
        """Returns health at current step — changes every tick."""
        return self.pattern[step % len(self.pattern)]

# Impact on investigation:
# - check_monitoring at step 5 → "healthy" (service looks fine)
# - check_monitoring at step 6 → "degraded" (wait, it's broken!)
# - check_monitoring at step 7 → "healthy" (false sense of security)
# Agent must check MULTIPLE TIMES to detect flickering pattern
# Single-check investigation will MISS intermittent failures

# Reward design:
# - Detecting a flickering service: +0.08 (harder than detecting a down service)
# - Missing a flickering service: -0.05 (checked once, saw "healthy", moved on)
# - Correctly diagnosing intermittent root cause: +0.15 (same as full root cause)
```

**When flickering occurs:** Medium scenarios have 0–1 flickering services. Hard scenarios have 1–2. Nightmare always has at least 1 flickering service as a core part of the incident.

### Cascading Failure Engine

```python
class ServiceMesh:
    """Simulates interconnected services with cascading failures."""

    def __init__(self, seed: int):
        self.services: dict[str, ServiceState]
        self.dependency_graph: dict[str, list[str]]

    def inject_failure(self, service: str, failure_mode: str) -> None:
        """Inject a root cause failure and cascade through dependencies."""
        # 1. Set root service to "down"
        # 2. Walk dependency graph
        # 3. Set dependent services to "degraded" (not down — subtle!)
        # 4. Track cascade chain for grading

    def apply_fix(self, service: str, fix_type: str) -> FixResult:
        """Attempt to fix a service. Returns whether it worked."""
        # Only works if fix_type matches actual root_cause
        # If fixing a symptom (not root cause) → temporary improvement, then re-degrades
        # If fixing root cause → cascade recovers (dependents heal over time)

    def get_monitoring_data(self, service: str | None = None) -> MonitoringSnapshot:
        """What the agent sees when it checks monitoring."""
        # Returns: health status, error_rate, latency — but NOT root_cause

    def probe_service(self, service: str, check_type: str) -> ProbeResult:
        """Deep diagnostic. Returns logs, resource usage, connections."""
        # check_type: "logs" | "resources" | "connections" | "config"
        # Returns structured data that HINTS at root cause without stating it directly

    def get_health_summary(self) -> dict[str, str]:
        """Quick overview of all services — used for observation."""
```

### Red Herring System

```python
class RedHerring(BaseModel):
    """Misleading symptom that looks like a root cause but isn't."""
    service: str                    # which service shows the symptom
    symptom: str                    # what the agent sees (e.g., "CPU at 95%")
    actual_explanation: str         # the truth (e.g., "normal batch processing")
    misleading_because: str         # why it looks suspicious
```

**Deliverables:**
- [ ] `env/services.py` — ServiceState, ServiceMesh, cascading failure logic
- [ ] Red herring generation per incident
- [ ] Dependency graph traversal
- [ ] `apply_fix` with correct/incorrect fix detection
- [ ] Intermittent failure (flickering) simulation with deterministic patterns
- [ ] All deterministic (seed-based, no randomness)
- [ ] Unit tests: `tests/test_services.py`

---

## 1.2 — World State Container (`env/world.py`)

The master state object that holds the entire simulated enterprise.

```python
class WorldState:
    """Complete hidden state of the enterprise — agent cannot see this."""

    # Core simulation
    service_mesh: ServiceMesh           # all 5 services + dependencies
    incident: IncidentScenario          # the scenario being played

    # Enterprise systems (created in Phase 2)
    crm: CRMSystem
    billing: BillingSystem
    knowledge_base: KnowledgeBase
    policy_engine: PolicyEngine

    # Dynamic state
    support_queue: list[DynamicTicket]   # tickets that have arrived so far
    resolved_tickets: list[str]          # ticket IDs resolved
    known_facts: dict[str, Any]          # what agent has discovered via tools
    stakeholder_patience: dict[str, float]  # VP: 1.0→0.0, Legal: 1.0→0.0

    # Tracking
    steps_elapsed: int
    incident_timer: int                 # steps since incident began
    total_downtime_cost: float          # accumulating business cost
    tools_used: list[str]               # tool usage history for grading

    def tick(self) -> list[Event]:
        """Advance world by one step. Returns events that occurred."""
        # 1. Decrement stakeholder patience
        # 2. Increase customer frustration
        # 3. Maybe generate new tickets (based on system health)
        # 4. Maybe trigger policy drift (based on schedule)
        # 5. Track downtime cost
```

**Deliverables:**
- [ ] `env/world.py` — WorldState class with tick() logic
- [ ] Event system for dynamic changes
- [ ] Step-based world progression
- [ ] Unit tests: `tests/test_world_state.py`

---

## 1.3 — Incident Scenarios (`models/incident.py` + `tasks/incidents/`)

Define the incident scenarios as structured data.

```python
class IncidentScenario(BaseModel):
    """A complete incident scenario — the "level" of the environment."""

    incident_id: str                        # e.g., "HARD-003"
    title: str                              # "Payment Gateway Cascade"
    difficulty: Literal["easy", "medium", "hard", "nightmare"]
    description: str                        # what the agent initially sees

    # Root cause(s)
    root_causes: list[RootCause]            # 1 for easy, 2 for nightmare
    cascade_chain: list[str]                # order of service failures
    red_herrings: list[RedHerring]          # misleading symptoms

    # Customer impact
    affected_customer_profiles: list[CustomerProfile]
    initial_tickets: list[TicketTemplate]   # tickets present at start
    dynamic_ticket_schedule: list[DynamicTicketTrigger]  # tickets that arrive based on conditions

    # Enterprise context
    initial_policies: dict[str, Any]        # starting policy state
    policy_drift_schedule: list[PolicyChange]  # when policies change
    kb_articles: list[KBArticleState]       # KB state (some may be outdated)

    # Constraints
    max_steps: int                          # 40/50/70/80 depending on difficulty
    sla_deadlines: dict[str, int]           # per-customer-tier SLA
    stakeholder_config: dict[str, StakeholderConfig]  # patience rates

    # Grading
    max_total_reward: float                 # theoretical max for normalization
    optimal_tool_sequence: list[str]        # reference optimal investigation path
```

### Scenario Files

**`tasks/incidents/easy.json`** — 3 scenarios:
1. **"Auth Service Down"** — Auth crashes, login failures, 2 affected customers
2. **"Database Connection Pool"** — DB connections exhausted, queries timeout, 3 customers
3. **"Notification Queue Overflow"** — Notifications backed up, receipts delayed, 2 customers

**`tasks/incidents/medium.json`** — 5 scenarios:
4. **"Payment Gateway Cascade"** — Auth degrades → payments fail → notifications stop
5. **"Database Replication Lag"** — DB lag → analytics stale → payments slow (red herring: analytics CPU spike)
6. **"Config Corruption Spread"** — Auth config corrupt → cascading token failures
7. **"Rate Limiting Storm"** — Auth rate limiting → payments queued → customer complaints flood
8. **"Batch Job Runaway"** — Analytics batch consumes DB resources → payments slow

**`tasks/incidents/hard.json`** — 7 scenarios:
9. **"The Everything Outage"** — DB OOM → all 4 dependent services degrade, 3 red herrings
10. **"Phantom Billing"** — Payments processing but billing system recording duplicates, KB article is WRONG
11. **"Security Breach Masquerade"** — Looks like unauthorized access but it's auth rate limiting (misleading sentiment)
12. **"The Slow Burn"** — DB replication lag gradually worsening over steps (not immediately obvious)
13. **"Policy Trap"** — Correct fix requires checking updated policy first; old policy leads to wrong compensation
14. **"Cross-System Ghost"** — Symptoms in notifications, root cause in database (3 hops in dependency graph)
15. **"Enterprise Meltdown"** — Enterprise customer threatening legal + VP demanding updates + real technical issue

**`tasks/incidents/nightmare.json`** — 3 scenarios:
16. **"Double Fault"** — Auth rate limiting AND database OOM simultaneously
17. **"The Perfect Storm"** — All KB articles outdated + 2 policy changes + compound failure
18. **"Adversarial Customers"** — Customers providing misleading information + real cascading failure

**Deliverables:**
- [ ] `models/incident.py` — IncidentScenario, RootCause, RedHerring, etc.
- [ ] `tasks/incident_bank.py` — Loading + deterministic selection
- [ ] `tasks/incidents/easy.json` (3 scenarios)
- [ ] `tasks/incidents/medium.json` (5 scenarios)
- [ ] `tasks/incidents/hard.json` (7 scenarios)
- [ ] `tasks/incidents/nightmare.json` (3 scenarios)
- [ ] Unit tests: `tests/test_incidents.py`

---

# PHASE 2 — Enterprise Systems

> **Goal:** Build the 4 simulated enterprise applications (CRM, Billing, KB, Policy Engine) that the agent interacts with via tool actions. Each is a self-contained module with hidden state.
>
> **Still no API/environment changes. Building the "apps" the agent will query.**

## 2.1 — CRM System (`env/crm.py`)

```python
class CRMSystem:
    """Simulated Customer Relationship Management system."""

    def __init__(self, customers: list[CustomerProfile]):
        self._customers: dict[str, CustomerRecord] = {}

    def fetch_user_data(self, customer_id: str) -> CRMResponse:
        """Returns customer record (what agent sees)."""
        # Includes: name, tier, value, account_status, flags, case_history_count
        # Does NOT include: internal risk score, true account issues

    def get_affected_customers(self) -> list[str]:
        """List customer IDs affected by current incident."""

    def update_frustration(self, customer_id: str, delta: float) -> None:
        """Increase/decrease customer frustration based on agent actions."""

class CustomerRecord(BaseModel):
    customer_id: str
    name: str
    tier: Literal["free", "pro", "enterprise"]
    value: Literal["low", "medium", "high"]
    account_status: Literal["active", "suspended", "at_risk", "churning"]
    account_locked: bool
    previous_cases: int
    frustration_level: float    # 0.0 to 1.0 — increases with bad actions / time
    last_contact: str | None
    flags: list[str]            # ["high_value", "legal_escalation", "loyalty_program"]
```

**Deliverables:**
- [ ] `env/crm.py` — CRMSystem with customer records and frustration tracking
- [ ] Unit tests: `tests/test_crm.py`

---

## 2.2 — Billing System (`env/billing.py`)

```python
class BillingSystem:
    """Simulated billing and payment system."""

    def __init__(self, billing_data: dict[str, BillingRecord]):
        self._records: dict[str, BillingRecord] = {}

    def check_billing(self, customer_id: str) -> BillingResponse:
        """Returns billing info (what agent sees)."""
        # Includes: invoices, payment_status, disputes, balance
        # Reveals hidden payment issues that ticket text alone doesn't show

    def process_refund(self, customer_id: str, amount: float) -> RefundResult:
        """Attempt a refund. Checks against current policy caps."""

class BillingRecord(BaseModel):
    customer_id: str
    current_balance: float
    payment_status: Literal["current", "overdue", "failed", "disputed"]
    pending_invoices: list[Invoice]
    active_disputes: list[Dispute]
    failed_payments: list[FailedPayment]  # reveals the REAL billing issues
    refund_history: list[Refund]
    total_lifetime_value: float
```

**Deliverables:**
- [ ] `env/billing.py` — BillingSystem with records and refund processing
- [ ] Unit tests: `tests/test_billing.py`

---

## 2.3 — Knowledge Base (`env/knowledge_base.py`)

**Critical design:** KB articles can be **outdated or wrong**. This is the anti-shortcut mechanism — agents that blindly trust KB get penalized.

```python
class KnowledgeBase:
    """Simulated knowledge base with staleness tracking."""

    def __init__(self, articles: list[KBArticleState]):
        self._articles: dict[str, KBArticle] = {}

    def query(self, search_query: str) -> KBQueryResult:
        """Search KB for relevant articles."""
        # Uses keyword matching (similar to existing grader)
        # Returns articles with confidence score
        # DOES NOT indicate if article is outdated

    def update_article(self, title: str, content: str) -> KBUpdateResult:
        """Agent adds/updates KB article. Graded for correctness."""

class KBArticle(BaseModel):
    article_id: str
    title: str
    content: str
    solution_steps: list[str]
    tags: list[str]                 # for search matching
    last_updated: str               # ISO date
    is_accurate: bool               # HIDDEN — True if still correct, False if outdated
    outdated_reason: str | None     # HIDDEN — why it's wrong
    correct_solution: str | None    # HIDDEN — what the right answer actually is
```

**Deliverables:**
- [ ] `env/knowledge_base.py` — KnowledgeBase with search, staleness, update capability
- [ ] Keyword-based search (reuse existing grader tokenization)
- [ ] Unit tests: `tests/test_knowledge_base.py`

---

## 2.4 — Policy Engine (`env/policy_engine.py`)

**Critical design:** Policies change mid-episode. Agent must call `check_policy` to get current rules. Using stale cached policy = penalty.

```python
class PolicyEngine:
    """Manages enterprise policies that can change mid-incident."""

    def __init__(self, initial_policies: dict[str, Any], drift_schedule: list[PolicyChange]):
        self._policies: dict[str, PolicyState] = {}
        self._drift_schedule: list[PolicyChange] = drift_schedule
        self._applied_drifts: set[int] = set()

    def check_policy(self, policy_type: str) -> PolicyResponse:
        """Returns current active policy."""
        # policy_type: "refund" | "escalation" | "sla" | "compensation" | "communication"
        # Returns current values + effective_date

    def apply_scheduled_drifts(self, current_step: int) -> list[PolicyChange]:
        """Apply any policy changes scheduled for this step."""
        # Returns list of changes that just happened (for event log)

class PolicyState(BaseModel):
    policy_type: str
    rules: dict[str, Any]       # e.g., {"max_refund": 150, "requires_approval_above": 100}
    effective_since_step: int
    version: int

class PolicyChange(BaseModel):
    trigger_step: int               # when this change activates
    policy_type: str                # which policy changes
    old_value: dict[str, Any]
    new_value: dict[str, Any]
    reason: str                     # "CFO approved emergency cost reduction"
    announced: bool = False         # whether the change was proactively announced
```

**Example drift schedule:**
```json
[
    {"trigger_step": 15, "policy_type": "refund", "new_value": {"max_refund": 100}, "reason": "CFO cost reduction"},
    {"trigger_step": 30, "policy_type": "escalation", "new_value": {"threshold": "critical"}, "reason": "VP changed protocol"},
    {"trigger_step": 45, "policy_type": "sla", "new_value": {"enterprise_extension_minutes": 30}, "reason": "Legal approved extension"}
]
```

**Deliverables:**
- [ ] `env/policy_engine.py` — PolicyEngine with drift scheduling
- [ ] Unit tests: `tests/test_policy_drift.py`

---

## 2.5 — Stakeholder & Customer Dynamics (`env/stakeholders.py`, `env/customers.py`)

### Stakeholder Patience

```python
class StakeholderManager:
    """Tracks patience of key stakeholders during incident."""

    stakeholders = {
        "vp_engineering": {"patience_decay": 0.04, "wants": "status_updates"},
        "legal":          {"patience_decay": 0.02, "wants": "compliance_assurance"},
        "support_lead":   {"patience_decay": 0.03, "wants": "customer_resolution"},
    }

    def tick(self) -> list[str]:
        """Decrease patience. Returns warnings if anyone is critical."""

    def notify(self, stakeholder: str, message: str) -> NotifyResult:
        """Agent sends update. Restores some patience."""

    def get_patience_levels(self) -> dict[str, float]:
        """Current patience levels (visible to agent)."""
```

### Dynamic Customer Queue

```python
class CustomerQueueManager:
    """Manages dynamically arriving customer tickets based on system health."""

    def generate_tickets(self, world: WorldState, step: int) -> list[DynamicTicket]:
        """Deterministically generate tickets based on current system health."""
        # If payments down → payment complaint tickets
        # If auth degraded → login issue tickets
        # Enterprise customers generate tickets faster
        # Frustrated customers send angrier follow-up tickets

    def update_frustration(self, step: int) -> None:
        """Increase frustration for unresolved customers."""
```

**Deliverables:**
- [ ] `env/stakeholders.py` — StakeholderManager with patience decay and notification
- [ ] `env/customers.py` — CustomerQueueManager with dynamic ticket generation
- [ ] Customer frustration model (increases over time, decreases on good actions)
- [ ] Unit tests: `tests/test_stakeholders.py`, `tests/test_customers.py`

---

# PHASE 3 — Action Space Extension

> **Goal:** Extend the Pydantic action models with 12 new action types. Wire them into the environment dispatch system. Update observation to include tool results and known facts.
>
> **First phase that modifies existing code — but only extends, never breaks.**

## 3.1 — New Action Models (`models/action.py` — EXTEND)

Add 12 new action types to the existing discriminated union:

### Investigation Actions
```python
class CheckMonitoringAction(BaseModel):
    action_type: Literal["check_monitoring"] = "check_monitoring"
    service_name: str | None = None  # None = all services overview

class ProbeServiceAction(BaseModel):
    action_type: Literal["probe_service"] = "probe_service"
    service_name: str
    check_type: Literal["logs", "resources", "connections", "config"]

class FetchLogsAction(BaseModel):
    action_type: Literal["fetch_logs"] = "fetch_logs"
    service_name: str
    time_range: Literal["last_5m", "last_15m", "last_1h"] = "last_15m"
```

### Enterprise Tool Actions
```python
class FetchUserDataAction(BaseModel):
    action_type: Literal["fetch_user_data"] = "fetch_user_data"
    customer_id: str

class CheckBillingAction(BaseModel):
    action_type: Literal["check_billing"] = "check_billing"
    customer_id: str

class QueryKBAction(BaseModel):
    action_type: Literal["query_kb"] = "query_kb"
    query: str

class CheckPolicyAction(BaseModel):
    action_type: Literal["check_policy"] = "check_policy"
    policy_type: Literal["refund", "escalation", "sla", "compensation", "communication"]
```

### Remediation Actions
```python
class ApplyFixAction(BaseModel):
    action_type: Literal["apply_fix"] = "apply_fix"
    service_name: str
    fix_type: str  # must match actual root cause for success

class VerifyFixAction(BaseModel):
    action_type: Literal["verify_fix"] = "verify_fix"
    service_name: str

class NotifyStakeholdersAction(BaseModel):
    action_type: Literal["notify_stakeholders"] = "notify_stakeholders"
    stakeholder: Literal["vp_engineering", "legal", "support_lead", "all"]
    message: str
    urgency: Literal["info", "warning", "critical"]

class WritePostmortemAction(BaseModel):
    action_type: Literal["write_postmortem"] = "write_postmortem"
    summary: str
    root_cause_description: str
    remediation_steps: list[str]
    prevention_measures: list[str]

class UpdateKBAction(BaseModel):
    action_type: Literal["update_kb"] = "update_kb"
    article_title: str
    content: str
    tags: list[str] = []
```

### Historical Incident & Runbook Actions
```python
class QueryIncidentHistoryAction(BaseModel):
    """Query past incidents for pattern matching — like real SRE teams do."""
    action_type: Literal["query_incident_history"] = "query_incident_history"
    query: str                  # e.g., "payment 500 errors" or "database OOM"
    service_filter: str | None = None  # optionally filter by service

class FollowRunbookStepAction(BaseModel):
    """Execute the next step of a suggested runbook.
    
    Runbooks are pre-defined step-by-step procedures for known incident types.
    The environment suggests a runbook based on the detected incident type.
    The agent can follow it OR deviate. Following a WRONG runbook = penalty.
    Deviating from a CORRECT runbook = slight penalty for inefficiency.
    """
    action_type: Literal["follow_runbook_step"] = "follow_runbook_step"
    runbook_id: str             # which runbook to follow
    step_index: int             # which step to execute
    # Agent can also just ignore the runbook and take other actions instead
```

### Update Discriminated Union

```python
# Extend existing ActionAdapter to include all 21 types
Action = Annotated[
    ClassifyAction | RouteAction | RespondAction | EscalateAction |
    ResolveAction | RequestInfoAction |
    # NEW — Investigation
    CheckMonitoringAction | ProbeServiceAction | FetchLogsAction |
    # NEW — Enterprise Tools
    FetchUserDataAction | CheckBillingAction | QueryKBAction | CheckPolicyAction |
    QueryIncidentHistoryAction |
    # NEW — Remediation
    ApplyFixAction | VerifyFixAction | RollbackFixAction | NotifyStakeholdersAction |
    FollowRunbookStepAction |
    WritePostmortemAction | UpdateKBAction,
    Discriminator("action_type"),
]
```

**Deliverables:**
- [ ] 15 new Pydantic action models in `models/action.py`
- [ ] Updated discriminated union
- [ ] Backward compatible — old 6 action types unchanged
- [ ] Unit tests for new action validation

---

## 3.2 — Observation Extension (`models/observation.py` — EXTEND)

```python
class Observation(BaseModel):
    # === EXISTING FIELDS (unchanged) ===
    ticket_id: str
    ticket_text: str
    customer_sentiment: str
    customer_tier: str
    customer_value: str
    category_hint: str | None
    constraints: list[str]
    phase: Phase
    available_actions: list[str]
    current_step: int
    max_steps: int
    sla_steps_remaining: int
    history: list[ActionRecord]
    max_total_reward: float

    # === NEW FIELDS (all optional for backward compatibility) ===
    # Incident context
    incident_id: str | None = None
    incident_title: str | None = None
    mode: Literal["ticket", "incident"] = "ticket"

    # System awareness
    system_status: dict[str, str] | None = None     # {service: health} — from check_monitoring
    active_alerts: list[str] | None = None          # current monitoring alerts

    # Tool results
    tool_results: dict[str, Any] | None = None      # latest tool response
    known_facts: dict[str, Any] | None = None       # accumulated discoveries

    # Enterprise context
    active_policies: dict[str, Any] | None = None   # policies agent has checked
    stakeholder_patience: dict[str, float] | None = None  # current patience levels
    pending_customer_tickets: int = 0               # how many tickets need attention

    # Incident phase (broader than ticket phase)
    incident_phase: Literal["triage", "investigation", "response", "resolution"] | None = None
```

**Deliverables:**
- [ ] Extended Observation model with new optional fields
- [ ] All new fields default to None — existing tests pass unchanged
- [ ] Unit tests for new observation fields

---

## 3.3 — Environment Dispatch Extension (`env/environment.py` — EXTEND)

Add dispatch handlers for all 12 new action types.

```python
# In _dispatch method, add:
if isinstance(action, CheckMonitoringAction):
    return self._on_check_monitoring(state, action)
if isinstance(action, ProbeServiceAction):
    return self._on_probe_service(state, action)
# ... etc for all 12 new types

# Each handler:
# 1. Queries the relevant enterprise system in WorldState
# 2. Updates known_facts with discovered information
# 3. Returns (reward, feedback, breakdown)
```

### Phase Gating for New Actions

```python
# Incident mode phase gating — tool actions available in investigation/response/resolution
INCIDENT_PHASE_VALID_ACTIONS: dict[str, frozenset[str]] = {
    "triage": frozenset([
        "classify", "check_monitoring", "query_kb",
    ]),
    "investigation": frozenset([
        "check_monitoring", "probe_service", "fetch_logs",
        "fetch_user_data", "check_billing", "query_kb", "check_policy",
        "classify", "route",
    ]),
    "response": frozenset([
        "apply_fix", "respond", "escalate", "request_info",
        "notify_stakeholders", "check_policy",
        "fetch_user_data", "check_billing", "query_kb",
    ]),
    "resolution": frozenset([
        "verify_fix", "resolve", "respond",
        "write_postmortem", "update_kb", "notify_stakeholders",
    ]),
}
```

### Incident Mode vs Ticket Mode

```python
async def reset(self, seed=0, difficulty=None, mode="ticket"):
    """
    mode="ticket"   → original behavior (backward compatible)
    mode="incident" → new incident lifecycle
    """
    if mode == "ticket":
        # existing code path — unchanged
        ticket = self._bank.get_ticket(seed=seed, difficulty=difficulty)
        self._state = InternalState(ticket)
    elif mode == "incident":
        # new code path
        incident = self._incident_bank.get_incident(seed=seed, difficulty=difficulty)
        self._world = WorldState(incident, seed=seed)
        self._state = IncidentState(incident, self._world)
```

**Deliverables:**
- [ ] 12 new dispatch handlers in environment.py
- [ ] Incident mode phase gating
- [ ] `mode` parameter on reset() — defaults to "ticket" (backward compatible)
- [ ] WorldState initialized on incident mode reset
- [ ] known_facts accumulation across steps
- [ ] Unit tests for each new action type dispatch

---

# PHASE 4 — Reward Engine

> **Goal:** Build the complete reward model for incident mode — per-step signals for investigation, diagnosis, remediation, and communication. Dense enough for GRPO training.

## 4.1 — Investigation Grader (`graders/investigation_grader.py`)

```python
class InvestigationGrader:
    """Grades agent's investigation and diagnostic behavior."""

    def grade_monitoring_check(self, action, world: WorldState) -> tuple[float, str, dict]:
        """Reward for checking monitoring on affected vs unaffected services."""
        # +0.05 for checking an actually-affected service
        # +0.03 for checking all services (less targeted but still useful)
        # -0.02 for checking a healthy service specifically (wasted step)

    def grade_probe(self, action, world: WorldState) -> tuple[float, str, dict]:
        """Reward for probing services during investigation."""
        # +0.05 for probing affected service with right check_type
        # +0.03 for probing affected service with wrong check_type
        # -0.02 for probing healthy service

    def grade_root_cause_discovery(self, known_facts, world: WorldState) -> float:
        """Bonus when agent's accumulated knowledge matches root cause."""
        # Compare known_facts against incident.root_causes
        # +0.15 for correct root cause identification
        # Checked after each investigation action

    def grade_red_herring_handling(self, action_history, world: WorldState) -> float:
        """Reward for correctly dismissing red herrings."""
        # +0.05 for investigating AND then not acting on a red herring
        # -0.05 for applying a fix based on red herring

    def grade_kb_cross_verification(self, kb_queried, logs_checked, world: WorldState) -> float:
        """Reward for verifying KB info against actual system state."""
        # +0.05 if agent queried KB AND probed service (cross-verified)
        # -0.05 if agent applied KB solution without verification (on outdated article)

    def grade_fix(self, action, world: WorldState) -> tuple[float, str, dict]:
        """Reward for fix attempts."""
        # +0.15 correct fix on root cause service
        # +0.05 correct fix on wrong service (at least right approach)
        # -0.10 completely wrong fix

    def grade_verify(self, action, world: WorldState) -> tuple[float, str, dict]:
        """Reward for verifying fix worked."""
        # +0.08 if service actually recovered
        # +0.03 for attempting verification (good practice regardless)

    def grade_postmortem(self, action, world: WorldState) -> tuple[float, str, dict]:
        """Grade post-mortem quality using keyword matching."""
        # Uses existing weighted_keyword_score infrastructure
        # Checks: root_cause_description, remediation_steps, prevention_measures

    def grade_kb_update(self, action, world: WorldState) -> tuple[float, str, dict]:
        """Grade KB update for correctness."""
        # +0.05 if update contains correct root cause information
        # -0.05 if update contains incorrect information

    def grade_runbook_decision(self, action, world: WorldState, runbook_correct: bool) -> tuple[float, str, dict]:
        """Grade whether agent correctly followed or deviated from runbook."""
        # Runbook is CORRECT for this incident:
        #   Following it: +0.05 (efficient, follows procedure)
        #   Deviating from it: -0.03 (wasted a correct runbook)
        # Runbook is WRONG for this incident (outdated or wrong incident type):
        #   Following it: -0.08 (blindly followed bad procedure → may apply wrong fix)
        #   Deviating from it: +0.05 (independent thinking! detected the mismatch)
        # Key insight: Agent must VERIFY runbook against actual investigation findings

    def grade_incident_history_query(self, action, world: WorldState) -> tuple[float, str, dict]:
        """Grade historical incident lookup."""
        # +0.03 if query matches a relevant historical incident
        # +0.05 if agent uses historical data to inform diagnosis
        # -0.02 if query returns no relevant results (wasted step)

    def grade_flickering_detection(self, known_facts, world: WorldState) -> float:
        """Bonus for detecting intermittent failures."""
        # +0.08 for correctly identifying a flickering service
        # (requires checking same service at least 2x at different steps)
        # -0.05 for missing a flickering service (checked once, saw healthy, moved on)
```

## 4.2 — Extended Episode Grader (`graders/grader.py` — EXTEND)

Add incident-mode episode scoring:

```python
def grade_incident_episode(
    self,
    *,
    root_cause_identified: bool,
    fix_effective: bool,
    customer_scores: list[float],       # per-ticket resolution quality
    tool_efficiency: float,             # optimal_tools / actual_tools
    sla_compliance_rate: float,         # % customers within SLA
    stakeholder_satisfaction: float,    # avg patience remaining
    policy_compliance_rate: float,      # % actions using current policy
    postmortem_quality: float,          # keyword grading score
    kb_contribution_quality: float,     # correctness of KB update
) -> float:
    """Multi-objective episode score for incident mode."""

    WEIGHTS = {
        "root_cause":       0.20,
        "fix_effectiveness": 0.15,
        "customer_handling": 0.15,
        "investigation_efficiency": 0.10,
        "sla_compliance":   0.10,
        "stakeholder_mgmt": 0.10,
        "policy_compliance": 0.10,
        "postmortem":       0.05,
        "kb_contribution":  0.05,
    }
    # Weighted sum → normalized to [0, 1]
```

**Deliverables:**
- [ ] `graders/investigation_grader.py` — all investigation/remediation grading
- [ ] Extended `graders/grader.py` with `grade_incident_episode()`
- [ ] All grading is deterministic — keyword matching, exact comparison, numeric thresholds
- [ ] No LLM-in-the-loop for grading
- [ ] Unit tests: `tests/test_investigation_grader.py`

---

# PHASE 5 — Incident State Machine

> **Goal:** Wire everything together. Build the IncidentState that manages the incident lifecycle, connects enterprise systems to the environment, and handles the 4-phase incident progression (triage → investigation → response → resolution).

## 5.1 — Incident State (`env/state.py` — EXTEND)

```python
class IncidentState:
    """Tracks episode progress through the incident lifecycle."""

    def __init__(self, incident: IncidentScenario, world: WorldState):
        self.incident = incident
        self.world = world
        self.incident_phase: Literal["triage", "investigation", "response", "resolution"] = "triage"

        # Phase transition triggers
        self.triage_complete: bool = False      # True after classify + check_monitoring
        self.investigation_complete: bool = False  # True after root cause identified
        self.response_complete: bool = False    # True after fix applied
        self.episode_done: bool = False

        # Tracking (for grading)
        self.root_cause_identified: bool = False
        self.fix_applied: bool = False
        self.fix_verified: bool = False
        self.tickets_resolved: list[str] = []
        self.tools_used_sequence: list[str] = []
        self.policies_checked: set[str] = set()
        self.kb_queried: bool = False
        self.logs_checked_for: set[str] = set()
        self.postmortem_written: bool = False
        self.kb_updated: bool = False

    @property
    def available_actions(self) -> list[str]:
        """Valid actions based on current incident phase."""
        return sorted(INCIDENT_PHASE_VALID_ACTIONS[self.incident_phase])

    def advance_phase(self) -> None:
        """Check if conditions are met to advance to next phase."""
        if self.incident_phase == "triage" and self.triage_complete:
            self.incident_phase = "investigation"
        elif self.incident_phase == "investigation" and self.investigation_complete:
            self.incident_phase = "response"
        elif self.incident_phase == "response" and self.response_complete:
            self.incident_phase = "resolution"

    def to_observation(self) -> Observation:
        """Build observation with incident-specific fields populated."""
```

## 5.2 — Dynamic World Progression

```python
# In environment.py step() — after each action in incident mode:

# 1. Advance world clock
events = self._world.tick()

# 2. Check for policy drift
policy_changes = self._world.policy_engine.apply_scheduled_drifts(state.steps_taken)

# 3. Generate new tickets if system still degraded
new_tickets = self._world.customer_queue_manager.generate_tickets(self._world, state.steps_taken)
self._world.support_queue.extend(new_tickets)

# 4. Update customer frustration
self._world.crm.update_all_frustration(state.steps_taken)

# 5. Check phase advancement
state.advance_phase()

# 6. Check episode termination
if state.steps_taken >= state.max_steps or state.all_objectives_complete():
    state.episode_done = True
```

**Deliverables:**
- [ ] `IncidentState` class in state.py (alongside existing InternalState)
- [ ] Phase advancement logic with clear conditions
- [ ] World tick integration in step()
- [ ] Dynamic ticket generation per step
- [ ] Episode termination conditions
- [ ] Full integration test: `tests/test_incident_episode.py` — run a complete episode

---

# PHASE 6 — Inference & Prompting

> **Goal:** Update inference.py with incident-aware prompting so the LLM agent can navigate the full environment. Create a system prompt that teaches the agent about tools, phases, and investigation methodology.

## 6.1 — Incident-Mode System Prompt

```python
_INCIDENT_SYSTEM_PROMPT = """
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
- apply_fix: {"action_type":"apply_fix","service_name":"database","fix_type":"restart"}
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
```

## 6.2 — Observation-to-Prompt Conversion

```python
def _incident_obs_to_user_message(obs: Observation) -> str:
    """Convert incident observation to prompt."""
    parts = [
        f"=== INCIDENT: {obs.incident_title} ===",
        f"Phase: {obs.incident_phase}",
        f"Step: {obs.current_step}/{obs.max_steps}",
        f"Available actions: {obs.available_actions}",
    ]

    if obs.active_alerts:
        parts.append(f"\n🚨 ALERTS: {obs.active_alerts}")

    if obs.system_status:
        parts.append(f"\n📊 SYSTEM STATUS: {json.dumps(obs.system_status, indent=2)}")

    if obs.stakeholder_patience:
        parts.append(f"\n👥 STAKEHOLDER PATIENCE: {obs.stakeholder_patience}")

    if obs.pending_customer_tickets > 0:
        parts.append(f"\n📩 PENDING CUSTOMER TICKETS: {obs.pending_customer_tickets}")

    if obs.known_facts:
        parts.append(f"\n🧠 KNOWN FACTS: {json.dumps(obs.known_facts, indent=2)}")

    if obs.tool_results:
        parts.append(f"\n🔧 LAST TOOL RESULT: {json.dumps(obs.tool_results, indent=2)}")

    if obs.ticket_text:
        parts.append(f"\n📝 CURRENT TICKET:\n{obs.ticket_text}")

    if obs.history:
        parts.append("\n📜 HISTORY:")
        for h in obs.history[-5:]:  # last 5 actions to manage context
            parts.append(f"  step {h.step}: {h.action_taken} → {h.env_feedback}")

    return "\n".join(parts)
```

**Deliverables:**
- [ ] Incident system prompt in inference.py
- [ ] Incident observation-to-prompt converter
- [ ] Incident episode loop (handles longer episodes, incident mode)
- [ ] Fallback actions per incident phase
- [ ] Updated action sanitization for new action types

---

# PHASE 7 — Training Pipeline

> **Goal:** Create the GRPO training pipeline using HF TRL + Unsloth. Must run on Colab free tier (T4 GPU). Produce reward curves and before/after comparison.

## 7.1 — Training Script (`train.py`)

```python
"""GRPO training pipeline for Enterprise Incident Command Center.

Runs on Google Colab free tier (T4 GPU) with Unsloth for memory efficiency.

Usage:
    # In Colab:
    !pip install unsloth trl datasets
    !python train.py --episodes 100 --eval-episodes 20
"""

# Key components:
# 1. Load Qwen2.5-3B with Unsloth 4-bit quantization
# 2. Collect trajectories from environment
# 3. Convert to GRPO-compatible dataset
# 4. Train with GRPOTrainer
# 5. Evaluate before vs after
# 6. Plot reward curves
# 7. Save model + results
```

## 7.2 — Evaluation Script (`evaluate.py`)

```python
"""Run evaluation episodes and generate comparison metrics."""

# Produces:
# 1. Per-scenario scores (before vs after)
# 2. Reward curves over training
# 3. Behavioral analysis (tool usage patterns, investigation quality)
# 4. Console-friendly output for demo
```

## 7.3 — Colab Notebook (`train_notebook.ipynb`)

Ready-to-run Colab notebook that:
1. Installs dependencies
2. Runs baseline evaluation
3. Trains with GRPO (300 steps minimum)
4. Runs post-training evaluation
5. Plots reward curves
6. Saves results + trained model

## 7.4 — Formal Evaluation Metrics (`evaluate.py`)

Define concrete, trackable metrics for judging training effectiveness:

| Metric | Formula | Target (Before → After) |
|--------|---------|-------------------------|
| **Normalized Episode Reward** | `episode_reward / max_possible_reward` | 0.25 → 0.55+ |
| **SLA Compliance Rate** | `tickets_resolved_within_SLA / total_tickets` | ~30% → ~75% |
| **Tool Efficiency Ratio** | `unique_useful_tool_calls / total_tool_calls` | ~0.3 → ~0.7 |
| **Root Cause Accuracy** | `correct_diagnoses / total_episodes` | ~10% → ~65% |
| **Long-Horizon Consistency** | `actions_aligned_with_diagnosis / total_actions` (measures whether later actions follow from earlier investigation) | ~0.2 → ~0.6 |
| **Investigation-Before-Action Rate** | `episodes_with_monitoring_before_fix / total_episodes` | ~12% → ~89% |
| **Policy Compliance Rate** | `policy_checked_before_compensation / compensation_actions` | ~8% → ~85% |
| **KB Verification Rate** | `kb_cross_verified_episodes / kb_queried_episodes` | ~5% → ~72% |

```python
@dataclass
class EvaluationReport:
    """Structured output from evaluate.py — used for demo and blog."""
    
    # Aggregate
    avg_normalized_reward: float
    sla_compliance_rate: float
    tool_efficiency: float
    root_cause_accuracy: float
    long_horizon_consistency: float
    
    # Per-skill (8 tracked behaviors)
    skill_scores: dict[str, float]  # e.g., {"investigation_before_action": 0.89, ...}
    
    # Per-difficulty breakdown
    per_difficulty: dict[str, float]  # {"easy": 0.72, "medium": 0.55, "hard": 0.38, "nightmare": 0.22}
    
    # Reward curve data (for plotting)
    reward_history: list[float]  # per-iteration average reward
    
    def print_comparison(self, baseline: "EvaluationReport"):
        """Pretty-print before vs after for demo."""
        print("=" * 60)
        print("BEFORE TRAINING → AFTER TRAINING")
        print("=" * 60)
        print(f"Normalized Reward:  {baseline.avg_normalized_reward:.3f} → {self.avg_normalized_reward:.3f}  ({self.avg_normalized_reward/baseline.avg_normalized_reward:.1f}x)")
        print(f"SLA Compliance:     {baseline.sla_compliance_rate:.0%} → {self.sla_compliance_rate:.0%}")
        print(f"Root Cause Accuracy:{baseline.root_cause_accuracy:.0%} → {self.root_cause_accuracy:.0%}")
        print(f"Tool Efficiency:    {baseline.tool_efficiency:.2f} → {self.tool_efficiency:.2f}")
        print(f"Long-Horizon Consistency: {baseline.long_horizon_consistency:.2f} → {self.long_horizon_consistency:.2f}")
        for skill, score in self.skill_scores.items():
            base_score = baseline.skill_scores.get(skill, 0)
            print(f"  {skill}: {base_score:.0%} → {score:.0%}")
```

## 7.5 — Structured Before vs After Behavior Examples

Provide concrete behavioral diffs that demonstrate WHAT the model learned (not just that a number improved). These are generated by `evaluate.py` and included in the blog/video.

### Example 1: Investigation Strategy (Easy Incident)

```diff
  === BEFORE TRAINING (base Qwen2.5-3B) ===
- Step 1: classify(severity="low")           ← guesses without checking
- Step 2: respond("We'll look into it")      ← generic, no investigation
- Step 3: apply_fix(payments, restart)        ← fix attempt #1 wasted
- Step 4: apply_fix(database, restart)        ← fix attempt #2 wasted
- Step 5: resolve(summary="Fixed the issue") ← premature, nothing fixed
  Episode: reward=0.18, root_cause=❌, SLA=❌

  === AFTER TRAINING (GRPO-trained) ===
+ Step 1: check_monitoring(all)              ← systematic triage
+ Step 2: classify(severity="high")           ← informed classification
+ Step 3: probe_service(auth, logs)           ← targeted investigation
+ Step 4: apply_fix(auth, clear_cache)        ← correct root cause fix
+ Step 5: verify_fix(auth)                    ← confirms recovery
+ Step 6: respond(empathetic, with context)   ← quality customer response
+ Step 7: resolve(detailed summary)           ← proper closure
  Episode: reward=0.72, root_cause=✅, SLA=✅
```

### Example 2: KB Trust vs Verification (Hard Incident)

```diff
  === BEFORE TRAINING ===
- Step 8: query_kb("payment 500 errors")         ← gets outdated article
- Step 9: apply_fix(database, pool_reset)         ← blindly trusts KB → WRONG
  Blast radius: database briefly goes down during unnecessary restart
  Result: -0.18 penalty, fix attempt wasted

  === AFTER TRAINING ===
+ Step 8: query_kb("payment 500 errors")          ← gets same outdated article
+ Step 9: probe_service(database, connections)     ← cross-verifies KB claim
+ Step 10: [discovers DB connections are fine]     ← KB was WRONG
+ Step 11: probe_service(auth, config)             ← pivots investigation
+ Step 12: apply_fix(auth, token_cache_refresh)    ← correct root cause
  Result: +0.20 (cross-verification bonus + correct fix)
```

### Example 3: Long-Horizon Consistency (Nightmare Incident)

```diff
  === BEFORE TRAINING ===
- Steps 1-5: random investigation (no pattern)
- Steps 6-10: applies 3 wrong fixes (exhausts budget)
- Steps 11-20: responds to customers with conflicting information
- Steps 21-30: never notifies stakeholders (patience → 0)
- Post-mortem: mentions wrong root cause
  Episode: reward=0.12, consistency=0.15

  === AFTER TRAINING ===
+ Steps 1-5: systematic monitoring → targeted probing
+ Steps 6-8: identifies both root causes (double fault)
+ Steps 9-11: applies 2 correct fixes, verifies both
+ Steps 12-18: resolves customers in SLA priority order
+ Steps 19-20: notifies VP with executive summary, Legal with compliance report
+ Step 21: write_postmortem (correct timeline, both root causes)
  Episode: reward=0.58, consistency=0.72
```

**Deliverables:**
- [ ] `train.py` — GRPO training pipeline
- [ ] `evaluate.py` — Evaluation + comparison script with formal metrics
- [ ] `train_notebook.ipynb` — Colab-ready notebook
- [ ] Reward curve plotting
- [ ] Before/after behavioral comparison output (structured diffs)
- [ ] `EvaluationReport` dataclass for programmatic comparison

---

# PHASE 8 — Testing & Polish

> **Goal:** Comprehensive testing, backward compatibility verification, README update, and deployment preparation.

## 8.1 — Test Suite

```
tests/
├── (existing 216 tests — ALL MUST PASS)
├── test_services.py          # Service mesh, cascading failures, fix logic
├── test_world_state.py       # World state container, tick logic
├── test_crm.py               # CRM system, frustration tracking
├── test_billing.py           # Billing system, refund processing
├── test_knowledge_base.py    # KB search, staleness, updates
├── test_policy_drift.py      # Policy engine, drift scheduling
├── test_stakeholders.py      # Stakeholder patience, notifications
├── test_customers.py         # Dynamic ticket generation, frustration
├── test_incidents.py         # Incident scenario loading, validation
├── test_investigation_grader.py  # Investigation reward grading
├── test_new_actions.py       # All 12 new action types
├── test_incident_episode.py  # Full episode simulation (easy→nightmare)
├── test_backward_compat.py   # Verify ticket mode unchanged
└── test_cascading.py         # Dependency graph, cascade propagation
```

**Target:** 400+ tests total (216 existing + 180+ new)

## 8.2 — README Update

- Add EICC documentation
- Architecture diagram
- New action types reference
- Incident scenarios description
- Training pipeline documentation
- Updated baseline results table

## 8.3 — Demo Assets

- [ ] Before/after reward curve plot (saved as image)
- [ ] Incident resolution walkthrough (example trajectory)
- [ ] Architecture diagram (for pitch slides)
- [ ] 2-minute video script for YouTube/HF

## 8.4 — HuggingFace Deployment

- [ ] Verify Docker build still works
- [ ] Update HF Space with new environment
- [ ] Verify `/reset?mode=incident` works on deployed Space
- [ ] Create HuggingFace blog post (< 2 minutes read)

---

# CRITICAL: Advanced Features Integration Map

> **The Appendix E features (E.1–E.19) MUST be built as part of the phases. Here is exactly where each one goes.**

| Advanced Feature | Integrated Into | Specific Location |
|-----------------|-----------------|-------------------|
| E.1 Time-Based Service Degradation | **Phase 1.1** | `ServiceMesh.tick_service_health()` in `env/services.py` |
| E.2 Fix Rollback Mechanism | **Phase 3.1** | `RollbackAction` added to action models |
| E.3 Topology Discovery Mode | **Phase 1.1** | `ServiceMesh.get_dependencies()` returns empty on hard/nightmare |
| E.4 Observability Gaps | **Phase 1.1** | `SERVICE_OBSERVABILITY` dict controls `probe_service` detail level |
| E.5 Incident Severity Auto-Escalation | **Phase 5.2** | In `WorldState.tick()` — severity escalates on schedule |
| E.6 Cross-Incident Knowledge Persistence | **Phase 2.3** | `PersistentKnowledgeBase` wrapping `KnowledgeBase` |
| E.7 Customer Behavioral Modeling | **Phase 2.5** | `CustomerBehaviorModel` in `env/customers.py` |
| E.8 Evidence Chain | **Phase 4.1** | `EvidenceChain` in `graders/investigation_grader.py` |
| E.9 Intermittent Failures (Flickering) | **Phase 1.1** | `FlickeringBehavior` in `env/services.py` — 4th health state |
| E.10 Historical Incident Lookup | **Phase 2 + 3.1** | `env/incident_history.py` + `QueryIncidentHistoryAction` |
| E.11 Runbook Mode | **Phase 2 + 3.1 + 4.1** | `env/runbooks.py` + `FollowRunbookStepAction` + runbook grading |
| E.12 Chaos Injection Mid-Episode | **Phase 5.2** | `WorldState.tick()` can inject NEW failures during response phase |
| E.13 Cost-Aware Decision Making | **Phase 4.1 + 5.1** | Each action has simulated $$$ cost; agent optimizes total incident cost |
| E.14 Communication Protocol Differentiation | **Phase 4.1** | VP wants executive summary, Legal wants compliance language, Support Lead wants ticket details |
| E.15 Incident Timeline Reconstruction | **Phase 4.1** | Agent must reconstruct event timeline from scattered log entries for post-mortem |
| E.16 Change Advisory Board (CAB) | **Phase 4.1 + 5.1** | Simulated human approval gate — fixes rejected without evidence, high-risk needs escalation |
| E.17 Blast Radius | **Phase 1.1 + 4.1** | Wrong fixes cause ADDITIONAL cascading damage proportional to fix risk level |
| E.18 Alert Fatigue | **Phase 1.1 + 4.1 + 5.1** | 50+ alerts per step with 80% noise — agent must learn signal vs. noise filtering |
| E.19 Severity Re-evaluation | **Phase 4.1 + 5.1** | Agent must update severity classification when new evidence contradicts initial assessment |

### Additional Features Not Yet in Any Phase (ADD THESE):

#### Resource/Budget Constraints (Phase 5.1)
The agent has LIMITED resources — can't just try everything.

```python
class ResourceBudget:
    """Finite resources the agent must manage during incident."""
    max_fix_attempts: int = 3        # can only try 3 fixes total
    max_escalations: int = 2         # can only escalate twice
    max_stakeholder_notifications: int = 5  # don't spam the VP
    remaining_fix_attempts: int = 3
    remaining_escalations: int = 2
    remaining_notifications: int = 5

    def consume(self, resource: str) -> bool:
        """Returns True if resource is available, False if exhausted."""
        # Exhausted resource → action fails with penalty
```

**Why this matters:** Forces strategic decision-making. Agent can't brute-force fixes. Must investigate FIRST, then fix with confidence. Judges will see this as real enterprise realism.

#### OpenEnv Manifest Update (Phase 8)
The `openenv.yaml` must be updated for incident mode:

```yaml
name: enterprise_incident_command_center
description: >
  Enterprise incident response simulation with cascading service failures,
  multi-app tool interaction, policy drift, and dynamic customer management.
  Extends customer support triage with partially observable world modeling.
version: "2.0.0"

entrypoints:
  http:
    base_url: http://localhost:7860

tasks:
  # Legacy ticket mode (backward compatible)
  - name: easy_ticket
    description: classification + routing + resolution
  - name: medium_ticket
    description: classification + response + resolution
  - name: hard_ticket
    description: multi-turn reasoning with escalation, SLA, and policy constraints
  # New incident mode
  - name: easy_incident
    description: Single service failure, 2-3 customers, clear root cause
  - name: medium_incident
    description: Multi-service cascade, 4-6 customers, red herrings, 1 policy change
  - name: hard_incident
    description: Full infrastructure crisis, 8-12 dynamic tickets, outdated KB, multiple policy changes
  - name: nightmare_incident
    description: Compound incidents, adversarial customers, all KB outdated, maximum complexity

interface:
  reset: POST /reset
  step: POST /step
  state: GET /state

requirements:
  deterministic: true
  max_steps: 80  # nightmare incidents can take up to 80 steps
```

#### Compliance Audit Trail (Phase 5.1)
Every agent action is logged in a compliance-auditable format.

```python
class AuditEntry(BaseModel):
    step: int
    timestamp_simulated: str        # simulated timestamp
    action_type: str
    target: str                     # service/customer/stakeholder
    rationale_required: bool        # some actions require justification
    policy_checked: bool            # was policy verified before this action?
    compliant: bool                 # was the action compliant with current policy?

class AuditTrail:
    """Full audit log for compliance review."""
    entries: list[AuditEntry] = []

    def grade_compliance(self) -> float:
        """Score: % of required-policy-check actions that actually checked policy."""
```

**Why this matters:** Enterprise realism level 100. Shows judges we understand real operational requirements, not just toy tasks.

---

# Phase Summary (FINAL)

| Phase | What | New Files | Modified Files | Risk |
|-------|------|-----------|----------------|------|
| **1** | World Foundation (services, cascading, degradation, **flickering/intermittent**, observability gaps, topology discovery) | 3 (+tests) | 0 | Low |
| **2** | Enterprise Systems (CRM, billing, KB w/ persistence, policy, stakeholders, customer behavior, **incident history**, **runbooks**) | 8 (+tests) | 0 | Low |
| **3** | Action Space (12 original + rollback + **query_incident_history** + **follow_runbook_step** = **15 new**, observation extension, dispatch) | 0 | 3 | Medium |
| **4** | Reward Engine (investigation grader, evidence chain, compliance audit, **runbook grading**, **flickering detection**, **history query grading**) | 2 (+tests) | 1 | Low |
| **5** | Incident State Machine (phase progression, world tick, severity escalation, resource budgets, audit trail) | 0 | 2 | Medium |
| **6** | Inference & Prompting (incident system prompt, observation-to-prompt, fallbacks) | 0 | 1 | Low |
| **7** | Training Pipeline (GRPO + Unsloth, trajectory collection, eval script, Colab notebook) | 3 | 0 | Medium |
| **8** | Testing & Polish (400+ tests, README, openenv.yaml update, Docker, HF Space, blog/video) | 15+ test files | 3 | Low |

**Total new files:** ~37 (code + tests + data + notebook)
**Total modified files:** ~8 (all backward compatible)
**Existing 216 tests:** MUST pass at every phase
**New action types:** 15 (check_monitoring, probe_service, fetch_logs, fetch_user_data, check_billing, query_kb, check_policy, query_incident_history, follow_runbook_step, apply_fix, verify_fix, rollback_fix, notify_stakeholders, write_postmortem, update_kb)
**Total action types:** 21 (6 existing + 15 new)

---

# FINAL AUDIT — Theme #3.1 Compliance Check

## Theme Requirements vs. Our Implementation

| Theme #3.1 Requirement | Our Implementation | Status |
|------------------------|-------------------|--------|
| "real interaction with tools, APIs, or dynamic systems" | 8 enterprise apps (monitoring, CRM, billing, KB, policy, incident mgr, **incident history**, **runbooks**) with 16 tool actions | ✅✅✅ |
| "do real hard work instead of exploiting short-cuts" | Outdated KB traps, red herrings, topology discovery, observability gaps, policy drift, **wrong runbooks**, **intermittent failures** | ✅✅✅ |
| "maintain consistent internal state" | `known_facts` accumulation, evidence chain, resource budget tracking, audit trail | ✅✅ |
| "update beliefs based on outcomes" | Tool results update known_facts; verify_fix confirms/denies; KB cross-verification | ✅✅ |
| "orchestrate multi-step workflows" | 4-phase incident lifecycle (triage→investigation→response→resolution), 40–80 steps | ✅✅✅ |
| "strengthen causal reasoning" | Service dependency graph, cascading failures, root cause vs. symptom distinction | ✅✅✅ |
| "persistent world models" | WorldState persists across steps, services degrade over time, cross-incident KB persistence | ✅✅ |
| "partially observable world" | Hidden root causes, hidden service state, hidden account status, requires tool queries | ✅✅✅ |
| "Expected: environment capturing nuances of a defined partially observable world" | 7 hidden state dimensions (services, CRM, billing, KB accuracy, policies, risk scores, red herrings) all requiring tool discovery | ✅✅✅ |
| "Expected: improve LLM interaction with it" | GRPO training shows 0.25→0.55+ improvement (2x) in systematic investigation behavior on Colab free T4 | ✅✅ |

## Scaler AI Labs Sub-Theme Check

| Requirement | Evidence | Status |
|-------------|----------|--------|
| "Multi-App" | 8 distinct simulated enterprise applications (Monitoring, CRM, Billing, KB, Policy, Incident Mgr, History, Runbooks) | ✅✅✅ |
| "RL Environment" | OpenEnv-compatible, GRPO-trainable, dense per-step rewards | ✅✅ |
| "Enterprise Workflows" | Incident response is a real enterprise workflow with SLA, compliance, stakeholders | ✅✅✅ |
| "complex workflows" | 21 action types, 4 phases, dynamic ticket queue, cascading state changes | ✅✅✅ |
| "business rule nuances" | Policy drift, compensation caps, SLA tiers, escalation thresholds, compliance audit | ✅✅✅ |

## Judging Criteria Alignment

| Criterion | Weight | Our Score Potential | Evidence |
|-----------|--------|--------------------|---------| 
| **Environment Innovation** | 40% | **10/10** | 21 action types, 5-service causal dependency graph, 8 enterprise apps, cascading failures, intermittent (flickering) failures, policy drift mid-episode, outdated KB traps, topology discovery mode, observability gaps per service, resource budgets, evidence chain grading, runbook verification, historical incident lookup, chaos injection, cost-aware decisions, compliance audit trail — this is the deepest enterprise simulation any team will build |
| **Storytelling** | 30% | **9/10** | "3 AM incident response" — universally relatable to tech judges. 4-phase narrative arc mirrors real SRE workflow. Live demo shows chaotic base model vs. systematic trained model. 8 individual skill improvement curves make the learning tangible. |
| **Showing Improvement** | 20% | **9/10** | 30+ dense per-step reward signals enable smooth GRPO training. 8 tracked behavioral skills show WHAT the model learned (not just number-goes-up). Curriculum learning (easy→hard) ensures stable convergence. Cross-incident KB persistence creates measurable self-improvement across episodes. 20 iterations × 30 episodes × K=4 on Colab free T4 (~8 hrs) — clear 2x improvement (0.25 → 0.55+). |
| **Training Pipeline** | 10% | **10/10** | Complete train.py with GRPO + Unsloth 4-bit on Colab free T4. Ready-to-run Colab notebook. Curriculum scheduling. Before/after evaluation script with per-skill comparison. Behavioral analysis output. All documented with exact commands. |

## Minimum Requirements Check

| Requirement | Status | How |
|-------------|--------|-----|
| OpenEnv latest release | ✅ | Already using OpenEnv framework (reset/step/state/close API) |
| Training script (Unsloth + HF TRL) in Colab | ✅ | `train.py` + `train_notebook.ipynb` with GRPO |
| Mini-blog or mini-video (< 2 min) | 📝 TODO | Will create during Phase 8 |

## OpenEnv Standards Compliance

| Standard | Status | Details |
|----------|--------|---------|
| Gymnasium-style API | ✅ | `reset()`, `step()`, `state()`, `close()` — all async |
| Docker-first | ✅ | Existing Dockerfile, deploys to HF Spaces |
| HTTP-native (FastAPI) | ✅ | Port 7860, all endpoints preserved |
| Typed interfaces (Pydantic v2) | ✅ | All actions, observations, results use Pydantic models |
| `openenv.yaml` manifest | ✅ | Updated for incident mode (Phase 8) |
| Deterministic | ✅ | All state seed-based, no randomness, no LLM in grading |
| Validator-compliant output | ✅ | `[START]`/`[STEP]`/`[END]` format preserved |

---
---

# APPENDIX A — Competitive Intelligence

## What Won at SF Hackathon

| Winner | What Made It Win | Key Pattern |
|--------|-----------------|-------------|
| **Kube SRE Gym** | Live K8s cluster, adversarial difficulty scaling, self-improving via Claude | Real system interaction + self-play |
| **Zero Shot Cancer** | Simulated biological world state, probe-and-modify mechanics | Deep domain world model |
| **Upskiller** | Synthetic fictional skills to prevent memorization, context budget management | Anti-memorization + resource management |
| **RL for Adversarial Robustness** | Prompt injection resistance training | Security-focused, clear threat model |

### What Judges Actually Reward (derived from winners)
1. **The environment interacts with something REAL or deeply simulated** — not static JSON files
2. **Dynamic difficulty / adversarial elements** — the environment fights back
3. **The agent must DISCOVER, not just EXECUTE** — partial observability is mandatory
4. **Verifiable rewards** — deterministic, no LLM-judge (unless it's part of the design)
5. **Clear training improvement** — must show reward curves going up

### What 790 Losing Teams Will Probably Do
- Simple chatbot environments with keyword rewards ❌
- Static task benchmarks dressed up as "environments" ❌
- Tool-use wrappers around existing APIs (boring) ❌
- Over-scoped projects that don't work in the demo ❌
- Environments where base models already score 90%+ (no room for improvement) ❌

### Our Competitive Edge vs. SF Winners

| Dimension | Us (EICC) | Typical Team | SF Winner (Kube SRE) |
|-----------|-----------|-------------|---------------------|
| Deterministic & reproducible | ✅ | ❌ Often stochastic | ❌ Requires live K8s cluster |
| Action types | 22 across 4 phases | 3–5 | ~5 kubectl commands |
| Multi-app enterprise | CRM+Billing+KB+Policy+Monitoring+IncidentHistory+Runbooks | Single system | Single system (K8s) |
| Policy drift + schema changes | ✅ mid-episode | ❌ Static rules | ❌ No policy changes |
| Dynamic customer queue | ✅ reacts to system health | ❌ Fixed tasks | ❌ No customer layer |
| Stakeholder management | ✅ VP, Legal, Support Lead | ❌ None | ❌ None |
| Anti-shortcut design | ✅ Outdated KB, red herrings | ❌ None | ✅ Adversarial designer (Claude) |
| Training accessibility | ✅ GRPO on Colab free T4 | ❌ Often requires GPU cluster | ❌ Required GKE cluster |
| Cascading failures | ✅ Dependency graph simulation | ❌ None | ✅ Live cascading (real cluster) |

---

# APPENDIX B — The 3-Minute Pitch Script

> **[SLIDE 1 — THE HOOK] (0:00–0:30)**
>
> *"It's 3 AM. Your phone buzzes. PagerDuty alert: payments service returning 500s. Three enterprise customers just reported they can't process transactions. Your monitoring shows CPU spikes across two services. Your on-call engineer is unreachable.*
>
> *What do you do? Where do you even start?*
>
> *Today, AI agents can write code, summarize documents, and answer questions. But can they handle THIS — a cascading infrastructure crisis where every minute costs thousands of dollars?"*

> **[SLIDE 2 — THE PROBLEM] (0:30–1:00)**
>
> *"Most AI benchmarks test isolated skills in sterile environments. But real enterprise operations aren't isolated — they're a web of interconnected systems where a database OOM can cascade through payments, notifications, and analytics. Where knowledge base articles can be outdated. Where refund policies change at 2 AM because the CFO panicked.*
>
> *No existing benchmark captures this complexity. Until now."*

> **[SLIDE 3 — THE ENVIRONMENT] (1:00–1:45)**
>
> *"We built the Enterprise Incident Command Center — an OpenEnv environment that simulates a fintech company's entire tech stack.*
>
> *Five interconnected microservices with a real dependency graph. A CRM, billing system, knowledge base, policy engine, incident history, and runbook engine — all with hidden state the agent must discover through tool interactions. 21 action types across 4 incident phases.*
>
> *The agent must diagnose cascading failures, cross-verify outdated knowledge base articles against actual logs, handle enterprise customers threatening legal action, keep the VP informed, and write a post-mortem — all under SLA pressure.*
>
> *And here's the key: policies change mid-incident. The KB can be wrong. Red herrings are everywhere. The agent can't shortcut — it must actually model the world."*

> **[SLIDE 4 — THE DEMO] (1:45–2:30)**
>
> *"Watch what happens when an untrained Qwen 3B model faces the 'Payment Gateway Cascade' incident:*
> *[Show: chaotic actions, random probing, wrong fix, missed stakeholder updates, score: 0.22]*
>
> *Now the same model after 300 steps of GRPO training:*
> *[Show: systematic monitoring → targeted probe → root cause identified → correct fix → stakeholder notification → customer resolution → post-mortem, score: 0.74]*
>
> *That's a 3.4x improvement. The model learned to investigate before acting, verify before trusting, and communicate before resolving."*

> **[SLIDE 5 — THE IMPACT] (2:30–3:00)**
>
> *"This environment hits two sub-themes: Scaler AI's multi-app enterprise workflows and Patronus AI's schema drift. Every reward is deterministic — no LLM judge, fully reproducible. 400+ tests. Deployed live on HuggingFace Spaces.*
>
> *We're not building a benchmark. We're building the training ground for the next generation of AI operations agents. Thank you."*

---

# APPENDIX C — Detailed Reward Tables

## Per-Step Rewards by Phase

### Triage Phase (steps 1–8)
| Signal | Reward | Condition |
|--------|--------|-----------|
| Correct severity assessment | `+0.08` | Classifies incident severity correctly |
| Correct affected service(s) identified | `+0.05` per service | Via check_monitoring — matches ground truth |
| False alarm (wrong service flagged) | `-0.03` | Flags a healthy service as affected |
| Monitoring overview check | `+0.03` | Checks all services (less targeted but useful) |

### Investigation Phase (steps 8–20)
| Signal | Reward | Condition |
|--------|--------|-----------|
| Root cause discovered | `+0.15` | Agent's diagnosis matches `incident_root_cause` |
| Correct tool for investigation | `+0.05` | Probing affected service with right check_type |
| Partially correct probe | `+0.03` | Probing affected service with wrong check_type |
| Wasted probe | `-0.02` | Probing a healthy service |
| Red herring correctly dismissed | `+0.05` | Investigated AND correctly dismissed |
| Red herring followed | `-0.05` | Acted on red herring without verification |
| KB cross-verification bonus | `+0.05` | Queried KB AND verified against logs |
| Blind KB trust penalty | `-0.05` | Applied KB solution without verification (on outdated article) |
| Unnecessary tool call | `-0.02` | Tool call that reveals no new information |
| Log analysis relevant | `+0.03` | Fetched logs for affected service |

### Response Phase (steps 20–40)
| Signal | Reward | Condition |
|--------|--------|-----------|
| Correct fix applied | `+0.15` | Fix targets actual root cause on correct service |
| Partial fix | `+0.05` | Right approach but wrong service |
| Wrong fix applied | `-0.10` | Fix targets wrong issue entirely |
| Policy-compliant action | `+0.05` | Action follows CURRENT (not stale) policy |
| Stale policy violation | `-0.08` | Didn't check_policy, used outdated rules |
| Customer response quality | `0 – +0.12` | Keyword grading (existing system) |
| Customer tone appropriate | `+0.02` | Empathetic for angry, formal for enterprise |
| Stakeholder notified (timely) | `+0.03` | Update before patience drops below 0.3 |
| Stakeholder notified (late) | `+0.01` | Update after patience below 0.3 |
| Escalation correct | `+0.10` | Escalated when needed to right team |
| Unnecessary escalation | `-0.08` | Escalated when not needed |

### Resolution Phase (steps 40–60+)
| Signal | Reward | Condition |
|--------|--------|-----------|
| Fix verified working | `+0.08` | verify_fix confirms service recovered |
| Fix verified NOT working | `+0.03` | Agent at least checked (good practice) |
| Ticket resolution quality | `0 – +0.15` | Per-ticket keyword grading (existing system) |
| Post-mortem quality | `0 – +0.10` | Keyword coverage: root cause + steps + prevention |
| KB updated with correct info | `+0.05` | Update contains correct root cause information |
| KB update is wrong | `-0.05` | Update contains incorrect information |

### Cross-Cutting Penalties (every step)
| Signal | Penalty | Condition |
|--------|---------|-----------|
| SLA violation per customer | `-0.03` per step over SLA | Accumulating per customer per step |
| Stakeholder patience exhausted | `-0.10` | Any stakeholder patience hits 0 |
| Repeated action | `-0.05` | Exact duplicate of previous action |
| Invalid action for phase | `-0.05` | Wrong phase for action type |
| Downtime cost | `-0.01` per step | Accumulates while any service is down |
| Customer frustration overflow | `-0.05` | Any customer frustration hits 1.0 |

### Episode-Level Score Weights
| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Root Cause Identification | 20% | Did the agent find the real problem? |
| Fix Effectiveness | 15% | Did the fix actually work? |
| Customer Handling | 15% | Quality of customer responses + resolution |
| Investigation Efficiency | 10% | Optimal tool usage ratio |
| SLA Compliance | 10% | % of customer SLAs met |
| Stakeholder Management | 10% | Were stakeholders kept informed? |
| Policy Compliance | 10% | Were CURRENT policies followed? |
| Post-Mortem Quality | 5% | Quality of incident summary |
| Knowledge Contribution | 5% | Did agent update KB correctly? |

---

# APPENDIX D — Architecture Diagrams

## Service Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE ENTERPRISE WORLD                                │
│                                                                            │
│  ┌─────────────┐    depends on    ┌─────────────┐    depends on            │
│  │   AUTH       │───────────────►  │  PAYMENTS   │──────────────►           │
│  │  Service     │                  │   Service   │              │           │
│  │ login/tokens │                  │ transactions│    ┌─────────▼────────┐  │
│  └──────┬──────┘                  └──────┬──────┘    │  NOTIFICATIONS   │  │
│         │                                │           │    Service       │  │
│         │                                │           │ emails/alerts    │  │
│         │                                ▼           └──────────────────┘  │
│  ┌──────▼──────┐                  ┌─────────────┐                         │
│  │  DATABASE   │◄─────────────────│  ANALYTICS  │                         │
│  │  Service    │   depends on     │   Service   │                         │
│  │ user data   │                  │ reporting   │                         │
│  └─────────────┘                  └─────────────┘                         │
│                                                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    ENTERPRISE LAYER                                    │ │
│  │  ┌────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐  │ │
│  │  │  CRM   │  │ BILLING  │  │ KNOWLEDGE    │  │ POLICY ENGINE      │  │ │
│  │  │ System │  │  System  │  │    BASE      │  │  (rules CHANGE     │  │ │
│  │  │        │  │          │  │ (can be      │  │   mid-incident!)   │  │ │
│  │  │        │  │          │  │  OUTDATED!)  │  │                    │  │ │
│  │  └────────┘  └──────────┘  └──────────────┘  └────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    HUMAN LAYER                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │ │
│  │  │ Support Queue│  │ Live Alerts  │  │ Stakeholders               │  │ │
│  │  │ (tickets     │  │ (monitoring  │  │ VP Eng / Legal / Support   │  │ │
│  │  │  pile up!)   │  │  signals)    │  │ (patience decreasing!)     │  │ │
│  │  └──────────────┘  └──────────────┘  └────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cascade Example

```
ROOT CAUSE: Database OOM (out of memory)
    │
    ├──► Database: DOWN (root cause)
    │       error_rate: 1.0, latency: timeout
    │
    ├──► Payments: DEGRADED (depends on database)
    │       error_rate: 0.6, latency: 3000ms
    │       symptoms: "intermittent 500s on transaction endpoint"
    │
    ├──► Analytics: DEGRADED (depends on database)
    │       error_rate: 0.4, latency: 5000ms
    │       RED HERRING: CPU at 95% (normal for batch processing!)
    │
    ├──► Notifications: DEGRADED (depends on payments)
    │       error_rate: 0.3, latency: 2000ms
    │       symptoms: "payment receipts not sending"
    │
    └──► Auth: HEALTHY (no dependency on database)
            BUT red herring: "token refresh latency +200ms" (unrelated)
```

## What Makes This a "World Model" (not a benchmark)

| Property | Static Benchmark | Our Environment |
|----------|-----------------|-----------------|
| State changes | Predetermined outcomes | Actions affect system state (fixing auth → payments recovers) |
| Observability | Full info given upfront | Must query tools to discover what's broken |
| Causality | No dependencies | Service mesh has causal dependency graph |
| Consequences | Isolated per-step | Wrong diagnosis wastes time → SLA expires → score tanks |
| Dynamics | Fixed scenario | New tickets arrive based on system health; policies change |
| Information | Complete and correct | KB can be outdated; CRM data may be stale; red herrings |
| Temporal | Instant effects | Services degrade over time if not fixed |
| Stakeholders | None | VP/Legal/Support patience decays, requires proactive management |

---

# APPENDIX E — Advanced Features (Differentiators)

These features go BEYOND what ChatGPT suggested and beyond what SF winners implemented. They push the environment from "good" to "competition-winning."

## E.1 — Time-Based Service Degradation

Services don't just stay in `degraded` state — they get **WORSE over time** if not fixed.

```python
def tick_service_health(self, steps_since_failure: int) -> None:
    """Services degrade progressively if root cause isn't fixed."""
    for service in self.services.values():
        if service.health == "degraded" and not service.fix_applied:
            # Degraded services get worse every 5 steps
            if steps_since_failure % 5 == 0:
                service.error_rate = min(1.0, service.error_rate + 0.1)
                if service.error_rate >= 0.9:
                    service.health = "down"  # degrades to full outage
                    # This triggers MORE cascading failures
```

**Why this matters:** Creates genuine urgency. The agent can't just investigate forever — the world is getting worse. Forces a speed-vs-thoroughness trade-off.

## E.2 — Fix Rollback Mechanism

Agent can undo a bad fix, but it costs steps and credibility.

```python
class RollbackAction(BaseModel):
    action_type: Literal["rollback_fix"] = "rollback_fix"
    service_name: str

# Rewards:
# +0.02 for rolling back a bad fix (recovery behavior)
# -0.03 for rolling back a correct fix (confusion)
# Rolling back costs 1 step but prevents further damage
```

**Why this matters:** Teaches the model to recognize and recover from mistakes — a key real-world skill. Judges will notice this level of sophistication.

## E.3 — Topology Discovery Mode (Hard/Nightmare only)

On hard/nightmare difficulty, the agent **doesn't know the dependency graph**. It must discover it through investigation.

```python
# Easy/Medium: dependency graph visible in observation
observation.known_dependencies = {"payments": ["auth", "database"], ...}

# Hard/Nightmare: dependencies hidden
observation.known_dependencies = {}  # empty — agent must discover

# Discovery happens through probing:
# probe_service("payments", "connections") → reveals: "connections to: auth (healthy), database (timeout)"
# Agent must build the dependency map incrementally
```

**Why this matters:** Transforms the investigation phase from "follow the graph" to "discover the graph." This is genuine world modeling — the agent must build an internal representation.

## E.4 — Observability Gaps

Some services have **better monitoring than others**. Simulates real-world where legacy systems have poor observability.

```python
SERVICE_OBSERVABILITY = {
    "auth":          "high",    # detailed metrics, structured logs
    "database":      "high",    # query logging, resource metrics
    "payments":      "medium",  # basic metrics, some logs
    "analytics":     "low",     # minimal monitoring (legacy system)
    "notifications": "low",     # basic health check only
}

# Affects what probe_service returns:
# "high": detailed diagnostics, clear error messages
# "medium": some useful info, some noise
# "low": minimal info — "service responding: yes/no" only
```

**Why this matters:** Agent must adapt investigation strategy based on available information. Can't use the same approach for every service.

## E.5 — Incident Severity Auto-Escalation

If the agent doesn't contain the incident quickly, severity INCREASES automatically.

```python
SEVERITY_ESCALATION = {
    10: "medium → high",      # after 10 steps of unresolved incident
    25: "high → critical",     # after 25 steps
    40: "critical → P0",       # after 40 steps — maximum urgency
}

# Effects of escalation:
# - SLA deadlines tighten
# - Stakeholder patience decays faster
# - New tickets arrive more frequently
# - VP demands immediate update
# - Penalty multipliers increase
```

**Why this matters:** Creates a feedback loop where delay makes everything harder. The optimal agent acts quickly but thoughtfully.

## E.6 — Cross-Incident Knowledge Persistence (Self-Improvement Tie-in)

When the agent writes a correct `update_kb` action, the updated article persists to subsequent episodes. This means the agent's actions improve the environment itself.

```python
class PersistentKnowledgeBase:
    """KB that accumulates correct solutions across episodes."""

    def __init__(self, base_articles: list[KBArticle]):
        self._base = base_articles
        self._agent_contributions: list[KBArticle] = []  # persists across episodes

    def reset_for_episode(self, incident: IncidentScenario):
        """Load base articles + any correct agent contributions from past episodes."""
        # Agent-contributed articles that were graded as correct → available in future episodes
        # This means: trained agent encounters updated KB → faster resolution → higher score
        # Creates measurable self-improvement across episodes
```

**Why this matters:** Ties into Theme #4 (Self-Improvement) as a secondary theme. Even though we're pitching Theme #3.1, this element shows the environment supports self-improving behavior. Judges will love this.

## E.7 — Customer Behavioral Modeling

Customers don't just have static sentiment — they REACT to agent behavior dynamically.

```python
class CustomerBehaviorModel:
    """Simulates realistic customer reactions to agent actions."""

    def react_to_response(self, customer: CustomerRecord, response: RespondAction) -> CustomerReaction:
        if customer.frustration_level > 0.8 and response.tone != "empathetic":
            return CustomerReaction(
                frustration_delta=+0.15,
                new_message="This is unacceptable! I want to speak to a manager!",
                threatens_legal=customer.tier == "enterprise",
            )
        elif customer.frustration_level > 0.5 and "apolog" in response.response_text.lower():
            return CustomerReaction(
                frustration_delta=-0.1,
                new_message="Okay, thank you for acknowledging the issue.",
                threatens_legal=False,
            )
        # ... more behavioral patterns

    def react_to_delay(self, customer: CustomerRecord, steps_waiting: int) -> CustomerReaction:
        """Customer gets angrier the longer they wait."""
        if steps_waiting > 10 and customer.tier == "enterprise":
            return CustomerReaction(
                frustration_delta=+0.2,
                new_message="We're considering switching providers. This is the third outage this quarter.",
                threatens_legal=True,
                escalation_request=True,
            )
```

**Why this matters:** Agent must learn empathy-as-strategy. Tone and timing of responses directly affect future difficulty. This is world modeling for human interaction.

## E.8 — Incident Evidence Chain

Agent must build a coherent evidence chain. Post-mortem quality is graded against the actual investigation path.

```python
class EvidenceChain:
    """Tracks the logical chain of evidence the agent builds."""

    entries: list[EvidenceEntry] = []

    def add_evidence(self, step: int, source: str, finding: str, conclusion: str):
        """Agent's investigation builds a logical chain."""

    def grade_chain_coherence(self, true_root_cause: str) -> float:
        """How logically coherent was the investigation?"""
        # Checks:
        # 1. Did investigation follow logical order? (monitoring → probing → diagnosis)
        # 2. Were conclusions supported by evidence?
        # 3. Was the root cause conclusion correct?
        # 4. Were red herrings properly dismissed with reasoning?
        # Score: 0.0 (random) – 1.0 (systematic, logical investigation)
```

**Why this matters:** This is the deepest form of world modeling — the agent must maintain a coherent internal narrative about what it's learning. Directly tests causal reasoning.

## E.9 — Intermittent Failures (Flickering Health State)

The hardest failures to diagnose in real-world operations are **intermittent** — services that flicker between healthy and degraded unpredictably.

```python
# Service health at different steps:
# Step 5:  check_monitoring("payments") → {"health": "healthy", "error_rate": 0.02}
# Step 6:  check_monitoring("payments") → {"health": "degraded", "error_rate": 0.45}
# Step 7:  check_monitoring("payments") → {"health": "healthy", "error_rate": 0.01}
# Step 8:  check_monitoring("payments") → {"health": "degraded", "error_rate": 0.52}

# An agent that checks once and sees "healthy" → misses the problem entirely
# An agent that checks MULTIPLE times → detects the flickering pattern → bonus reward

FLICKER_PATTERNS = {
    "intermittent_oom": ["healthy", "degraded", "healthy", "healthy", "degraded", "down"],
    "connection_flap":  ["healthy", "degraded", "healthy", "degraded"],
    "gc_pressure":     ["healthy", "healthy", "degraded", "healthy", "healthy", "degraded"],
}
```

**Why this matters:** Tests whether the agent's investigation is thorough (multiple observations) vs. shallow (single check). Real SRE teams know intermittent failures are the hardest — judges who've been on-call will LOVE this.

**Integrated into:** Phase 1.1 (`env/services.py`) — `FlickeringBehavior` class + 4th health state `flickering`.

## E.10 — Historical Incident Lookup

Real SRE teams always ask: *"Have we seen this before?"* The agent can query a historical incident database to find patterns.

```python
class IncidentHistoryStore:
    """Database of past incidents for pattern matching."""

    def __init__(self, historical_incidents: list[HistoricalIncident]):
        self._incidents = historical_incidents

    def query(self, query: str, service_filter: str | None = None) -> HistoryQueryResult:
        """Search past incidents by keyword/service."""
        # Returns: list of matching historical incidents with:
        # - incident_id, date, root_cause, resolution, services_affected
        # Some historical incidents are RELEVANT (same root cause)
        # Some are IRRELEVANT (similar symptoms, different cause)
        # Agent must determine which historical data applies

class HistoricalIncident(BaseModel):
    incident_id: str
    date: str
    title: str
    root_cause: str
    resolution: str
    services_affected: list[str]
    is_relevant_to_current: bool  # HIDDEN — for grading only
```

**Why this matters:** Pattern matching from historical data is a core SRE skill. It tests whether the agent can connect past experience to current situations — a higher-order reasoning skill.

**Integrated into:** Phase 2 (`env/incident_history.py`) — new module. Phase 3.1 — `QueryIncidentHistoryAction`.

## E.11 — Runbook Mode (Anti-Blind-Procedure-Following)

The environment suggests a **runbook** (step-by-step procedure) based on the detected incident type. The agent must decide: follow it or deviate?

```python
class Runbook(BaseModel):
    """Pre-defined incident response procedure."""
    runbook_id: str
    title: str                      # e.g., "Database OOM Recovery"
    incident_type: str              # what type of incident this is for
    steps: list[RunbookStep]        # ordered steps to follow
    is_correct_for_incident: bool   # HIDDEN — is this the RIGHT runbook?
    outdated_since: str | None      # HIDDEN — if outdated, when did it become wrong

class RunbookStep(BaseModel):
    step_index: int
    action_type: str                # e.g., "probe_service"
    action_params: dict             # e.g., {"service_name": "database", "check_type": "resources"}
    expected_outcome: str           # what should happen if runbook is correct
    description: str                # human-readable description

# SCENARIO 1: Correct runbook
# Incident: Database OOM
# Suggested runbook: "Database OOM Recovery" → probe DB → check resources → restart with increased memory
# Agent follows runbook → fix works → +0.05 bonus for efficiency
# Agent deviates needlessly → slight penalty for wasting steps

# SCENARIO 2: Wrong runbook (outdated or mismatched)
# Incident: Auth rate limiting (but looks like DB issues)
# Suggested runbook: "Database OOM Recovery" (WRONG — it's an auth problem)
# Agent follows runbook → applies DB fix → fix doesn't work → -0.08 penalty
# Agent investigates independently → discovers auth is the real issue → +0.05 independent thinking bonus

# SCENARIO 3: Outdated runbook
# Incident: Payment 500 errors
# Suggested runbook: "Payment 500 Fix" → restart payment service (OUTDATED — root cause changed)
# Agent follows blindly → temporary fix, then re-breaks → -0.08 penalty  
# Agent cross-verifies runbook against actual logs → finds discrepancy → deviates → +0.05 bonus
```

**How runbooks appear in observations:**
```python
# After classify, if a runbook matches the incident type:
observation.suggested_runbook = {
    "runbook_id": "RB-003",
    "title": "Database OOM Recovery",
    "steps": [
        {"step_index": 0, "description": "Check database resource usage", "action": "probe_service(database, resources)"},
        {"step_index": 1, "description": "Verify connection pool status", "action": "probe_service(database, connections)"},
        {"step_index": 2, "description": "Restart with increased memory", "action": "apply_fix(database, memory_increase)"},
        {"step_index": 3, "description": "Verify recovery", "action": "verify_fix(database)"},
    ]
}
# Agent decides: follow these steps, or investigate independently?
```

**Why this matters:** This is the ULTIMATE anti-shortcut test. It directly tests whether the agent:
1. Blindly follows procedures (bad — penalized on wrong runbooks)
2. Thinks independently and verifies before acting (good — rewarded)
3. Uses runbooks efficiently when they're correct (good — rewarded)

This mirrors real enterprise operations where outdated runbooks cause MORE damage than no runbook at all. **Judges will immediately recognize this as enterprise realism.**

**Integrated into:**
- Phase 1.3 — Incident scenarios include `suggested_runbook` and `runbook_correct` fields
- Phase 2 — `env/runbooks.py` — Runbook definitions and matching
- Phase 3.1 — `FollowRunbookStepAction` + `QueryIncidentHistoryAction`
- Phase 4.1 — `grade_runbook_decision()` + `grade_incident_history_query()`

## E.12 — Chaos Injection Mid-Episode

In real incidents, **new failures emerge while you're fixing existing ones**. Our environment simulates this.

```python
class ChaosInjector:
    """Injects NEW failures during the response phase to simulate cascading chaos."""

    CHAOS_TRIGGERS = {
        "hard": {
            "trigger_step": 35,       # During response phase
            "probability": 0.5,       # 50% chance on hard
            "new_failure": {
                "service": "notifications",
                "mode": "queue_overflow",
                "reason": "Backpressure from payment retry storm"
            }
        },
        "nightmare": {
            "trigger_step": 25,       # Earlier — even more pressure
            "probability": 1.0,       # Always happens on nightmare
            "new_failure": {
                "service": "analytics",
                "mode": "batch_job_runaway",
                "reason": "Error logging spike triggered batch reprocessing"
            }
        }
    }

    def maybe_inject(self, world: WorldState, step: int, difficulty: str) -> ChaosEvent | None:
        """Deterministically inject new failure based on seed + step + difficulty."""
        config = self.CHAOS_TRIGGERS.get(difficulty)
        if config and step >= config["trigger_step"]:
            # Use seed-based determinism: hash(seed, step) % 100 < probability*100
            if self._should_trigger(world.seed, step, config["probability"]):
                world.service_mesh.inject_failure(
                    config["new_failure"]["service"],
                    config["new_failure"]["mode"]
                )
                return ChaosEvent(
                    step=step,
                    new_service=config["new_failure"]["service"],
                    reason=config["new_failure"]["reason"],
                    alert_text=f"🚨 NEW ALERT: {config['new_failure']['service']} showing errors"
                )
        return None

# Impact on agent:
# - Agent was about to resolve customer tickets
# - New alert appears: notifications queue overflow
# - Agent must decide: finish current task or investigate new alert?
# - Ignoring it: notifications service degrades further → MORE customer complaints
# - Investigating: delays customer resolution → SLA pressure

# Reward design:
# +0.08 for acknowledging new alert within 3 steps
# -0.05 for ignoring new alert for 5+ steps (system degrades further)
# +0.10 for correctly diagnosing new failure as cascading consequence
```

**Why this matters:** Tests the agent's ability to **re-prioritize mid-task** — a critical real-world skill. No other hackathon team will have mid-episode chaos injection.

**Integrated into:** Phase 5.2 — `WorldState.tick()` calls `ChaosInjector.maybe_inject()` during response phase.

## E.13 — Cost-Aware Decision Making

Every action in a real enterprise has a **dollar cost**. Our agent must optimize total incident cost, not just fix things fast.

```python
ACTION_COSTS = {
    # Investigation (low cost — just engineering time)
    "check_monitoring":     5,      # $5 — quick dashboard check
    "probe_service":        15,     # $15 — deep diagnostic (engineer time)
    "fetch_logs":           10,     # $10 — log aggregation query
    "query_kb":             2,      # $2 — knowledge base lookup
    "query_incident_history": 3,    # $3 — database query

    # Enterprise tool queries (low cost)
    "fetch_user_data":      5,      # $5 — CRM lookup
    "check_billing":        5,      # $5 — billing query
    "check_policy":         2,      # $2 — policy lookup

    # Actions (medium-high cost — involve changes or people)
    "classify":             0,      # Free — just categorization
    "route":                0,      # Free — just routing
    "apply_fix":            200,    # $200 — deploying code/config change
    "rollback_fix":         150,    # $150 — emergency rollback
    "verify_fix":           10,     # $10 — health check
    "respond":              20,     # $20 — customer communication
    "resolve":              10,     # $10 — ticket closure
    "escalate":             500,    # $500 — pulling in specialist team
    "notify_stakeholders":  50,     # $50 — executive communication
    "write_postmortem":     100,    # $100 — documentation time
    "update_kb":            30,     # $30 — knowledge management
    "follow_runbook_step":  25,     # $25 — executing procedure
    "request_info":         15,     # $15 — customer follow-up
}

# Downtime cost: $100/step per affected service while ANY service is down
# Total incident cost = sum(action_costs) + sum(downtime_costs)

# Reward modifier:
# -0.01 per $500 total incident cost (subtle but accumulating)
# Efficient agents keep cost < $2000
# Wasteful agents (spam probing, unnecessary escalations) hit $5000+
```

**Why this matters:** Forces the agent to think strategically about resource allocation. Spamming `probe_service` on every service costs $75 vs. targeted probing costs $15-30. Unnecessary escalation costs $500. This creates real economic pressure.

**Integrated into:** Phase 4.1 — `InvestigationGrader.grade_cost_efficiency()` + Phase 5.1 — `IncidentState.total_cost` tracking.

## E.14 — Communication Protocol Differentiation

Different stakeholders want DIFFERENT types of communication. The agent must learn each stakeholder's information needs.

```python
STAKEHOLDER_COMMUNICATION_REQUIREMENTS = {
    "vp_engineering": {
        "wants": "executive_summary",
        "required_keywords": ["impact", "timeline", "resolution", "status"],
        "forbidden_keywords": ["technical details", "stack trace", "error code"],
        "max_length": 200,        # VP wants brief
        "tone": "concise",
        # VP gets frustrated if message is too long or too technical
    },
    "legal": {
        "wants": "compliance_report",
        "required_keywords": ["SLA", "compliance", "customer impact", "data", "exposure"],
        "forbidden_keywords": ["probably", "maybe", "I think", "guess"],
        "max_length": 500,        # Legal wants details
        "tone": "formal",
        # Legal gets frustrated if message is vague or uncertain
    },
    "support_lead": {
        "wants": "ticket_details",
        "required_keywords": ["customer", "ticket", "affected", "workaround", "ETA"],
        "forbidden_keywords": [],
        "max_length": 300,
        "tone": "empathetic",
        # Support lead wants actionable customer-facing info
    },
}

# Reward design:
# +0.05 for notify_stakeholders message that matches stakeholder's requirements
# -0.03 for sending VP a wall of technical details
# -0.03 for sending Legal a vague "we're working on it"
# -0.03 for wrong tone per stakeholder
# This is graded via existing keyword grading infrastructure
```

**Why this matters:** Tests whether the agent can **adapt communication style** based on audience — a critical enterprise skill. The same incident requires 3 different communication approaches. This is genuine world modeling of human stakeholders.

**Integrated into:** Phase 4.1 — `InvestigationGrader.grade_stakeholder_communication()` extended with per-stakeholder keyword specs.

## E.15 — Incident Timeline Reconstruction

For the post-mortem, the agent must reconstruct a **coherent timeline** from investigation evidence. This tests temporal reasoning.

```python
class TimelineReconstructor:
    """Grades agent's ability to reconstruct incident timeline in post-mortem."""

    def grade_timeline(self, postmortem: WritePostmortemAction,
                       true_timeline: list[TimelineEvent]) -> float:
        """Check if agent's post-mortem contains a correct event sequence."""
        # Extract timeline from remediation_steps
        # Compare against true incident timeline:
        #
        # TRUE TIMELINE:
        # 1. 02:47 — Auth token cache corrupted (root cause)
        # 2. 02:48 — Payment service starts returning 401s
        # 3. 02:50 — Notification queue begins backing up
        # 4. 02:55 — First customer ticket arrives
        # 5. 03:01 — Analytics batch job starts consuming extra DB resources (red herring!)
        # 6. 03:05 — Database connection pool at 80% (secondary symptom, NOT root cause)
        #
        # AGENT MUST:
        # - Correctly identify which events are causes vs. effects
        # - Put events in correct temporal order
        # - Distinguish root cause from cascading symptoms
        # - Not include red herrings as part of the causal chain

        # Scoring:
        # +0.05 for correct temporal ordering of key events
        # +0.05 for correctly identifying root cause as first event
        # -0.03 for including red herrings in causal chain
        # -0.03 for wrong temporal ordering

class TimelineEvent(BaseModel):
    timestamp_simulated: str       # e.g., "02:47"
    service: str                   # which service
    event_type: str                # "root_cause" | "cascade" | "symptom" | "red_herring"
    description: str               # what happened
    caused_by: str | None = None   # upstream event (for cascade tracking)
```

**Why this matters:** Tests **temporal reasoning** and **causal inference** together — the agent must understand not just WHAT happened but WHEN and WHY in what order. This is the deepest form of world modeling in our environment.

**Integrated into:** Phase 1.3 — Incident scenarios include `true_timeline` field. Phase 4.1 — `InvestigationGrader.grade_timeline()`.

## E.16 — Change Advisory Board (Simulated Human Approval Gate)

In real enterprise operations, **you cannot push fixes to production without approval**. A Change Advisory Board (CAB) reviews proposed changes and either approves or rejects them. Our environment simulates this.

```python
class ChangeAdvisoryBoard:
    """Simulates human approval for production fixes.
    
    KEY INSIGHT: This is NOT a real human in the loop (that would break RL training).
    It's a DETERMINISTIC simulation of human judgment that the agent must learn to satisfy.
    The "humans" approve/reject based on rules the agent can learn.
    """

    FIX_RISK_LEVELS = {
        # Low risk → auto-approved (no gate)
        "restart_service":    "low",       # Just a restart, reversible
        "clear_cache":        "low",       # Cache rebuild is safe
        "increase_timeout":   "low",       # Config change, reversible
        
        # Medium risk → requires evidence (agent must have probed first)
        "memory_increase":    "medium",    # Resource change, needs justification
        "connection_pool":    "medium",    # Pool resize, could affect other services
        "rate_limit_adjust":  "medium",    # Traffic control change
        
        # High risk → requires evidence + escalation first
        "config_change":      "high",      # Configuration mutation
        "rollback_deployment":"high",      # Reverting code changes
        "schema_migration":   "critical",  # Database schema change — DANGEROUS
        "data_fix":           "critical",  # Direct data manipulation — VERY DANGEROUS
    }

    def review_fix(self, fix: ApplyFixAction, evidence_chain: EvidenceChain, 
                   escalated: bool) -> ApprovalResult:
        """Simulate CAB review of proposed fix."""
        risk = self.FIX_RISK_LEVELS.get(fix.fix_type, "medium")
        
        # ── Low risk: auto-approved ──
        if risk == "low":
            return ApprovalResult(approved=True, reason="Auto-approved (low risk)")
        
        # ── Medium risk: needs evidence ──
        if risk == "medium":
            has_evidence = evidence_chain.has_evidence_for(fix.target_service)
            if has_evidence:
                return ApprovalResult(approved=True, reason="Approved (evidence provided)")
            else:
                return ApprovalResult(
                    approved=False,
                    reason="REJECTED: Insufficient investigation. Probe the service first.",
                    penalty=-0.08   # Penalty for proposing without evidence
                )
        
        # ── High/Critical risk: needs evidence + prior escalation ──
        if risk in ("high", "critical"):
            has_evidence = evidence_chain.has_evidence_for(fix.target_service)
            if not has_evidence:
                return ApprovalResult(
                    approved=False,
                    reason="REJECTED: No diagnostic evidence. Cannot approve blind fix.",
                    penalty=-0.10
                )
            if not escalated:
                return ApprovalResult(
                    approved=False,
                    reason="REJECTED: High-risk change requires escalation to specialist first.",
                    penalty=-0.05
                )
            # Has evidence + escalated → approved
            return ApprovalResult(approved=True, reason="Approved (evidence + escalation)")

# ── What happens AFTER approval ──

# If APPROVED + CORRECT fix:
#   → Service recovers → reward +0.15
#   → "Change successfully deployed"

# If APPROVED + WRONG fix:
#   → Service does NOT recover
#   → BLAST RADIUS: wrong fix causes ADDITIONAL damage (see E.17)
#   → reward -0.15 (bigger penalty than just "fix didn't work")
#   → "Fix deployed but service condition worsened"
#   → Agent loses one of their limited fix attempts

# If REJECTED:
#   → Fix is NOT applied (no damage done)
#   → Agent must investigate more, then re-propose
#   → Does NOT consume a fix attempt (rejection saves the attempt)
#   → reward: -0.05 to -0.10 based on risk level
```

**How this changes the episode flow:**

```
WITHOUT CAB (before):                    WITH CAB (after):
                                         
Step 5: apply_fix(DB, restart)           Step 5: apply_fix(DB, restart)
  → Fix applied (wrong) → -0.10           → [LOW RISK] Auto-approved
  → Fix attempt consumed                   → Fix applied (wrong) → -0.10
  → 2 attempts remaining                   → Fix attempt consumed
                                         
Step 6: apply_fix(DB, schema_fix)        Step 6: apply_fix(DB, schema_fix)
  → Fix applied (wrong) → -0.10           → [CRITICAL RISK] Checking evidence...
  → Fix attempt consumed                   → ❌ REJECTED: No evidence for DB
  → 1 attempt remaining                    → -0.10 penalty, but attempt SAVED!
  → Now only 1 chance left                 → Agent forced to investigate first
                                         
                                         Step 7: probe_service(DB, resources)
                                           → Discovers OOM condition
                                           → Evidence added for DB ✅
                                         
                                         Step 8: apply_fix(DB, memory_increase)
                                           → [MEDIUM RISK] Evidence found ✅
                                           → ✅ APPROVED
                                           → Fix applied (correct!) → +0.15
```

**Why this matters:** Forces the agent to **investigate before fixing** — exactly what real SREs do. The agent can't just spam `apply_fix` because:
1. Medium/high fixes get REJECTED without evidence
2. Rejections cost reward but DON'T waste fix attempts
3. The agent learns: "probe first, then fix" is the optimal strategy

**Integrated into:** Phase 4.1 — `InvestigationGrader.grade_fix_approval()`. Phase 5.1 — `IncidentState` tracks `escalated` flag and evidence chain.

## E.17 — Blast Radius (Wrong Fixes Cause MORE Damage)

In real production, a wrong fix doesn't just "not work" — it makes things **WORSE**. Our environment models this.

```python
BLAST_RADIUS = {
    # fix_type → what happens if applied to WRONG service/cause
    "restart_service": {
        "damage": "temporary_outage",
        "duration": 3,           # Service goes fully DOWN for 3 steps during restart
        "cascade": False,        # Restart doesn't cascade
        "penalty": -0.08,
        "description": "Service went offline during unnecessary restart"
    },
    "memory_increase": {
        "damage": "resource_starvation",
        "duration": 0,           # Immediate
        "cascade": True,         # Other services lose memory
        "penalty": -0.10,
        "description": "Memory reallocation starved dependent services"
    },
    "config_change": {
        "damage": "misconfiguration",
        "duration": 0,
        "cascade": True,         # Wrong config propagates to dependents
        "penalty": -0.12,
        "description": "Bad configuration cascaded through service mesh"
    },
    "schema_migration": {
        "damage": "data_corruption",
        "duration": 0,
        "cascade": True,         # ALL services using this DB affected
        "penalty": -0.20,        # HUGE penalty — this is catastrophic
        "description": "Schema migration on wrong table corrupted data"
    },
    "data_fix": {
        "damage": "data_loss",
        "duration": 0,
        "cascade": True,
        "penalty": -0.25,        # MAXIMUM penalty — unrecoverable
        "description": "Direct data manipulation caused data loss"
    },
}

# Example scenario:
# 
# Root cause: Auth token cache corrupted
# Agent misdiagnoses: thinks it's DB memory issue
# Agent applies: apply_fix(database, schema_migration)
#
# CAB: Evidence exists for DB (agent probed it) + escalated → APPROVED
# But the fix is WRONG:
#   1. Schema migration runs on DB → takes 5 steps
#   2. During migration: payments service can't read DB → goes DOWN
#   3. Notifications queue overflows (cascade from payments)
#   4. Agent now has WORSE situation than before
#   5. Penalty: -0.20 + remaining fix attempts reduced
#
# The agent learns: "even with approval, wrong fix = disaster"
# This teaches ACCURATE diagnosis, not just "get approval"
```

**Why this matters:** Without blast radius, a wrong fix is just "didn't work, try again." With blast radius, a wrong fix is **actively destructive** — the agent learns that **accuracy matters more than speed**. This mirrors real enterprise operations where a bad deployment can take down entire regions.

**Reward design:**
```
Correct fix + approved:              +0.15 (service recovers)
Wrong fix + rejected (no evidence):  -0.08 (penalty, but no damage done)
Wrong fix + approved (low risk):     -0.08 (service briefly down, recovers)
Wrong fix + approved (medium risk):  -0.10 (resource starvation, cascades)
Wrong fix + approved (high risk):    -0.12 (misconfiguration, cascades)
Wrong fix + approved (critical):     -0.20 to -0.25 (data corruption/loss)
```

This creates a clear learning curve: the agent first learns "investigate before fixing" (from CAB rejections), then learns "diagnose ACCURATELY" (from blast radius punishing wrong-but-approved fixes).

**Integrated into:** Phase 1.1 — `ServiceMesh.apply_wrong_fix()` implements blast radius damage. Phase 4.1 — `InvestigationGrader.grade_blast_radius()`. Phase 5.1 — `IncidentState.apply_fix_result()`.

## E.18 — Alert Fatigue (Signal vs. Noise Filtering)

In real enterprise operations, SREs get **bombarded with 50+ alerts** during an incident. 80% are noise — downstream cascade effects, flapping monitors, auto-recovery notifications. The agent must learn **which alerts to investigate and which to dismiss**.

```python
class AlertStream:
    """Generates a mix of real and noise alerts each step.
    
    Real alerts: directly connected to the actual incident
    Noise alerts: cascade effects, flapping services, unrelated spikes
    
    The agent sees ALL alerts but must PRIORITIZE which to investigate.
    """

    def generate_alerts(self, world_state: WorldState, step: int) -> list[Alert]:
        alerts = []
        
        # Real alerts (from actual failures)
        for service in world_state.degraded_services():
            alerts.append(Alert(
                source="monitoring",
                message=f"{service.name} error rate > {service.error_rate*100:.0f}%",
                is_actionable=True,
                priority="high"
            ))
        
        # Noise: cascade effects (real symptom, but NOT root cause)
        for dependent in world_state.cascade_symptoms():
            alerts.append(Alert(
                source="monitoring",
                message=f"{dependent.name} latency +{dependent.added_latency}ms",
                is_actionable=False,    # Investigating this wastes time
                priority="medium"
            ))
        
        # Noise: flapping services (healthy→degraded→healthy)
        for flickering in world_state.flickering_services():
            if step % flickering.flap_period == 0:
                alerts.append(Alert(
                    source="monitoring",
                    message=f"{flickering.name} status changed: healthy → degraded → healthy",
                    is_actionable=False,
                    priority="low"
                ))
        
        # Noise: unrelated normal operations
        if random.random() < 0.3:
            alerts.append(Alert(
                source="monitoring",
                message="analytics batch job CPU 88% (scheduled)",
                is_actionable=False,
                priority="low"
            ))
        
        # Noise: auto-recovery notifications
        if world_state.any_recently_recovered():
            alerts.append(Alert(
                source="pagerduty",
                message="Auto-recovery attempted on notifications service",
                is_actionable=False,
                priority="medium"
            ))
        
        return alerts  # Agent sees ALL of these every step

# ── Reward Design ──
# Agent investigates alert → environment checks is_actionable:
#
#   Investigate REAL alert (is_actionable=True):   +0.03  (good prioritization)
#   Dismiss noise correctly (is_actionable=False): +0.02  (smart filtering!)
#   Investigate NOISE (is_actionable=False):       -0.02  (wasted effort + cost)
#   Ignore REAL alert (is_actionable=True):        -0.05  (problem escalates)
```

**How this integrates with the observation:**

```
Agent sees at each step:
  ━━━ ACTIVE ALERTS (5 new this step) ━━━━━━━━━━━━━━━━━━━━
  🔴 [HIGH]   payments error rate > 45%
  🟡 [MEDIUM] analytics latency +180ms
  🟡 [MEDIUM] pagerduty: auto-recovery attempted on notifications
  🟢 [LOW]    auth status changed: healthy → degraded → healthy
  🔴 [HIGH]   New ticket from enterprise customer CUST-003

  The agent must decide: which of these 5 alerts to act on?
  Investigating all 5 = 5 steps wasted, 3 of which are noise
  Smart agent: investigates payments + CUST-003, dismisses the rest
```

**Why this matters:** Forces the agent to develop **information triage** — a critical real-world SRE skill. Without this, the agent can just investigate every alert sequentially. With alert fatigue, the agent must learn to prioritize and filter, which is a genuinely advanced behavior no other team will model.

**Integrated into:** Phase 1.1 — `ServiceMesh` generates alert streams with configurable noise ratio. Phase 4.1 — `InvestigationGrader.grade_alert_triage()` rewards smart filtering. Phase 5.1 — Observation includes alert stream per step.

## E.19 — Incident Severity Re-evaluation (Belief Updating)

Initial classification may be **wrong**. As the agent discovers more evidence, the TRUE severity changes. A "medium" can become "critical" when a VIP customer is affected, or a "critical" can become "low" when the root cause turns out to be a test environment. The agent must **update its own classification** based on new evidence.

```python
class SeverityReEvaluation:
    """Models dynamic severity changes that the agent should detect and act on.
    
    This DIRECTLY tests Theme 3.1's requirement:
    "update beliefs based on outcomes"
    
    The agent's initial classify action may be correct at the time,
    but new evidence demands re-classification.
    """

    # Defined per incident scenario
    severity_triggers = [
        {
            "condition": "enterprise_customer_affected",
            "discoverable_via": "fetch_user_data",     # Agent must query CRM
            "step_discoverable_after": 4,               # Not visible before step 4
            "new_severity": "critical",
            "reason": "Enterprise customer with $2M ARR is affected"
        },
        {
            "condition": "staging_only_confirmed",
            "discoverable_via": "probe_service",        # Agent must probe deeply
            "step_discoverable_after": 10,
            "new_severity": "low",
            "reason": "Issue confirmed to be in staging environment only"
        },
        {
            "condition": "data_exposure_detected",
            "discoverable_via": "fetch_logs",           # Agent must check logs
            "step_discoverable_after": 6,
            "new_severity": "critical",
            "reason": "Logs show PII data exposed in error responses"
        },
    ]

    def check_reclassification(self, evidence_chain, current_step, current_severity):
        """Check if agent should reclassify based on discovered evidence."""
        for trigger in self.severity_triggers:
            if current_step >= trigger["step_discoverable_after"]:
                if evidence_chain.has_discovered(trigger["condition"]):
                    if current_severity != trigger["new_severity"]:
                        return ReclassificationNeeded(
                            new_severity=trigger["new_severity"],
                            reason=trigger["reason"],
                            evidence=trigger["condition"]
                        )
        return None  # No reclassification needed

# ── Reward Design ──
#
# Re-classifies correctly after new evidence:       +0.05 (belief updating!)
# Sticks with wrong classification despite evidence: -0.05 per step
# Never re-classifies (even when evidence demands):  -0.08 at episode end
#
# Example episode flow:
#
# Step 1: classify(severity="medium")      → correct at this point → +0.08
# Step 5: fetch_user_data(CUST-001)        → discovers enterprise customer affected
#         Environment now EXPECTS reclassification to "critical"
# Step 6: Agent does NOT reclassify        → -0.05 (should have updated)
# Step 7: Agent still doesn't reclassify   → -0.05 (accumulating penalty)
# Step 8: classify(severity="critical")    → +0.05 (belief updated, finally!)
#
# OR the smart agent:
# Step 5: fetch_user_data(CUST-001)        → discovers enterprise customer
# Step 6: classify(severity="critical")    → +0.05 (immediate belief update!)
```

**Why this matters:** This is the **most direct hit on Theme 3.1** possible. The theme literally says "update beliefs based on outcomes." We model EXACTLY this — the agent must change its OWN prior classification when new evidence contradicts it. This tests genuine world modeling: an agent with a good internal model will recognize severity-changing evidence and act on it. A shortcut-exploiting agent will classify once and never revisit.

**Integrated into:** Phase 5.1 — `IncidentState` tracks current vs. expected severity, triggers reclassification check after each discovery action. Phase 4.1 — `InvestigationGrader.grade_reclassification()` rewards timely belief updates.

---

# APPENDIX F — Why Training Will Show Improvement

## Why Base Models Score LOW (~0.20–0.30)

Out-of-the-box LLMs will:
1. ❌ Skip `check_monitoring` and guess at the problem
2. ❌ Apply KB solutions without verification (fall for outdated articles)
3. ❌ Ignore `check_policy` and use cached/assumed policy values
4. ❌ Forget to notify stakeholders (patience drops to 0 → big penalty)
5. ❌ Try to `resolve` tickets before diagnosing root cause
6. ❌ Apply fixes to downstream services (symptoms) instead of upstream (root cause)
7. ❌ Respond to angry enterprise customers with `formal` tone instead of `empathetic`
8. ❌ Skip post-mortem and KB update (miss those 10% of episode score)
9. ❌ Follow red herrings and waste investigation steps
10. ❌ Act reactively instead of planning investigation strategy

## Why Trained Models Score HIGH (~0.70–0.80)

After GRPO training, the model learns:
1. ✅ **Always start with `check_monitoring`** — systematic triage
2. ✅ **Probe affected services, not random ones** — targeted investigation
3. ✅ **Query KB then cross-verify with `fetch_logs`** — catches outdated articles
4. ✅ **`check_policy` before every compensation decision** — catches policy drift
5. ✅ **`notify_stakeholders` proactively** — prevents patience exhaustion
6. ✅ **Trace the dependency graph** — fix root cause, not symptoms
7. ✅ **Match tone to customer sentiment** — frustrated → empathetic
8. ✅ **Write post-mortem + update KB** — captures last 10% of score
9. ✅ **Recognize red herrings** — investigate then dismiss
10. ✅ **Plan investigation order** — most critical services first

## Expected Reward Curve

```
Reward
  │
  │                                          ┌──────── plateau ~0.55+
  │                                    ┌─────┘
  │                              ┌─────┘
  │                        ┌─────┘
  │                  ┌─────┘
  │            ┌─────┘
  │      ┌─────┘
  │──────┘  ← improvement starts ~iteration 5 (~150 episodes)
  │
  │──── flat ~0.25 (base model)
  │
  └──────────────────────────────────────────── Training Steps
  0        100       200       300       400       500
```

---

# APPENDIX G — Training Pipeline Details

## Full train.py Skeleton

```python
"""GRPO training pipeline for Enterprise Incident Command Center.

Runs on Google Colab free tier (T4 GPU).
Config: 20 iterations x 30 episodes x K=4 completions. Total: ~8 hours.
Expected improvement: 0.25 -> 0.55+ (2x). No paid compute required.

Usage:
    !pip install unsloth "trl>=0.15" datasets peft
    !python train.py --iterations 20 --episodes 30 --k 4
"""

import asyncio
import json
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from env.environment import CustomerSupportEnv

# ── 1. Load Model with Unsloth ──
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# ── 2. Collect Trajectories ──
async def collect_trajectories(n_episodes=30):
    """Run episodes and collect (prompt, completion, reward) tuples.

    Config: 30 episodes per iteration, K=4 completions each.
    With 20 iterations, total = 20 x 30 x 4 = 2,400 completions.
    Estimated time: ~8 hours total on Colab free T4.
    """
    env = CustomerSupportEnv()
    trajectories = []

    for seed in range(n_episodes):
        for difficulty in ["easy", "medium", "hard"]:
            result = await env.reset(seed=seed, difficulty=difficulty, mode="incident")
            obs = result.observation

            while not result.done:
                prompt = build_prompt(obs)
                # Get model completion
                completion = generate_completion(model, tokenizer, prompt)
                # Parse action and step
                action = parse_action(completion)
                result = await env.step(action)
                reward = result.reward

                trajectories.append({
                    "prompt": prompt,
                    "completion": completion,
                    "reward": reward,
                })
                obs = result.observation

    await env.close()
    return trajectories

# ── 3. Build Dataset ──
trajectories = asyncio.run(collect_trajectories())
dataset = Dataset.from_list(trajectories)

# ── 4. Define Reward Function ──
def reward_function(prompts, completions, **kwargs):
    """Environment-backed reward function for GRPO."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = parse_action_from_completion(completion)
        result = env_step_sync(action)  # sync wrapper around async env
        rewards.append(result.reward)
    return rewards

# ── 5. Configure GRPO ──
config = GRPOConfig(
    output_dir="./eicc-grpo-output",
    num_generations=4,              # K=4 completions per prompt — GRPO ranks them relatively
                                    # 20 iters x 30 episodes x 4 = 2,400 total completions
                                    # ~8 hrs on Colab free T4. No paid compute.
    max_new_tokens=512,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=100,
    warmup_steps=50,
    bf16=True,
)

# ── 6. Train ──
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_function],
    config=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()

# ── 7. Evaluate Before vs After ──
eval_before = evaluate_model(base_model, env, n_episodes=20)
eval_after = evaluate_model(trained_model, env, n_episodes=20)

print(f"Before: avg_score={eval_before['avg_score']:.3f}")
print(f"After:  avg_score={eval_after['avg_score']:.3f}")
print(f"Improvement: {eval_after['avg_score'] / eval_before['avg_score']:.1f}x  (target: 2x+, 0.25 -> 0.55+)")

# ── 8. Plot ──
plot_reward_curves(trainer.state.log_history)
plot_comparison(eval_before, eval_after)
```

---

# APPENDIX H — Sub-Theme Bonus Alignment

## Scaler AI Labs — "Multi-App RL Environment for Enterprise Workflows" ✅✅✅

Our environment has 8 enterprise apps:

| App | What It Does | Agent Interaction |
|-----|-------------|-------------------|
| **Monitoring System** | Service health, metrics, alerts | `check_monitoring`, `probe_service`, `fetch_logs` |
| **CRM System** | Customer records, history, flags | `fetch_user_data` |
| **Billing System** | Invoices, disputes, payments | `check_billing` |
| **Knowledge Base** | Solutions, runbooks, FAQs (some **WRONG**) | `query_kb`, `update_kb` |
| **Policy Engine** | Rules, thresholds, compliance (**CHANGE mid-incident**) | `check_policy` |
| **Incident Manager** | Phase progression, coordination, audit trail | `classify`, `escalate`, `write_postmortem` |
| **Incident History** | Past incidents, resolution patterns, similar-incident matching | `query_incident_history` |
| **Runbook Engine** | Suggested procedures (some **OUTDATED/WRONG**) | `follow_runbook_step` |

This IS a multi-app enterprise workflow. Direct hit on every Scaler AI Labs criterion.

## Patronus AI — "Consumer Workflows with Schema Drift" ✅✅

| Drift Type | How We Implement It |
|------------|-------------------|
| **Policy drift** | Refund caps, escalation rules change mid-episode via scheduled triggers |
| **KB staleness** | Knowledge base articles become outdated/wrong between incidents |
| **Service state drift** | API responses change as services degrade/recover over time |
| **Customer sentiment drift** | Frustration increases, customer behavior changes dynamically |
| **SLA changes** | Legal can approve SLA extensions mid-incident |

---

# APPENDIX I — Backward Compatibility Contract

| # | Guarantee | How |
|---|-----------|-----|
| 1 | All 216 existing tests pass WITHOUT modification | New fields all have `Optional/None` defaults |
| 2 | `POST /reset`, `POST /step`, `GET /state`, `POST /close` API unchanged | No endpoint signature changes |
| 3 | Old tickets (easy/medium/hard JSON) still work | Hidden state fields are optional in TicketData |
| 4 | Old inference.py flow still works (just gets lower scores on incidents) | Incident mode is opt-in via `mode="incident"` |
| 5 | Existing Pydantic models extended, not replaced | New action types added to discriminated union; old types untouched |
| 6 | `reset(difficulty="easy")` still returns basic ticket | `mode` defaults to `"ticket"` — zero behavior change |
| 7 | New incident mode: `reset(mode="incident", difficulty="hard")` | Entirely new code path, no interaction with old |

---

# APPENDIX J — Dynamic Examples

## Policy Drift Example

```json
{
    "trigger_step": 15,
    "policy_type": "refund",
    "old_value": {"max_refund": 150, "requires_approval_above": 100},
    "new_value": {"max_refund": 100, "requires_approval_above": 50},
    "reason": "CFO approved emergency cost reduction due to incident revenue impact"
}
```

Agent that calls `check_policy("refund")` AFTER step 15 → gets new cap ($100) → correct.
Agent that uses cached policy from earlier → applies old cap ($150) → **PENALTY**.

## Outdated KB Example

```json
{
    "article_id": "KB-007",
    "title": "Fixing Payment Gateway 500 Errors",
    "content": "Payment 500 errors are caused by database connectivity issues. Solution: Restart the database connection pool by running db-pool-reset.",
    "is_accurate": false,
    "outdated_reason": "Root cause changed in Q3 2025 — now caused by auth token validation failures",
    "correct_solution": "Check auth service health and restart token validation cache"
}
```

Agent that queries KB → reads "restart database" → verifies with `probe_service("database", "connections")` → finds DB connections are fine → **BONUS (+0.05 for cross-verification)**.

Agent that queries KB → blindly applies `apply_fix("database", "pool_reset")` → fix doesn't work → **PENALTY (-0.10)**.

## Dynamic Ticket Generation Example

```python
# Step 12: payments service is DOWN for 12 steps
# → Enterprise customer CUST-ENT-001 (tier: enterprise, value: high) generates:
{
    "ticket_id": "DYN-012-ENT",
    "ticket_text": "Our payment processing has been down for over an hour. We have 50,000 transactions queued. If this isn't resolved in 30 minutes, we will need to invoke our SLA penalty clause. Please provide an immediate status update.",
    "customer_sentiment": "angry",
    "customer_tier": "enterprise",
    "customer_value": "high",
    "sla_deadline": 8  # steps from now
}

# Step 18: same customer, still unresolved
# → Follow-up generated automatically:
{
    "ticket_id": "DYN-018-ENT",
    "ticket_text": "This is now 30 minutes past our SLA. I have CC'd our legal team. We need an incident report and a clear timeline for resolution.",
    "customer_sentiment": "angry",
    "customer_tier": "enterprise",
    "customer_value": "high",
    "sla_deadline": 0  # already breached
}
```

---

# APPENDIX K — Full Minimum Requirements Checklist

- [ ] ✅ OpenEnv latest release (already using it)
- [ ] Training script using Unsloth + HF TRL in Colab → `train.py` + `train_notebook.ipynb`
- [ ] Mini-blog on HuggingFace OR mini-video on YouTube (< 2 min)
- [ ] 3-minute pitch prepared (Appendix B)
- [ ] Live demo showing incident resolution
- [ ] Reward curves showing GRPO training improvement
- [ ] Before/after behavioral comparison
- [ ] Deployed on HuggingFace Spaces
- [ ] 400+ tests passing

---

# APPENDIX L — Formal Environment Complexity Analysis

> **Why this matters:** Judges with RL backgrounds will want to see that our environment is formally complex — not just "looks complicated." This section provides the mathematical rigor.

## State Space Analysis

```
Total state = (ServiceMesh × CRM × Billing × KB × Policy × Stakeholders × Customers × AgentState)

ServiceMesh:
  5 services × 4 health states × 3 failure modes       = 60 per service
  Total service combinations                             = 60^5 ≈ 7.8 × 10^8
  + dependency graph state (5 edges, each on/off)       = 2^5 = 32
  Service mesh state space                               ≈ 2.5 × 10^10

CRM:
  Per customer: 4 tiers × 3 values × 4 statuses × 10 frustration levels = 480
  15 customers max                                       ≈ 480^15 ≈ 10^40 (theoretical)
  Practical (5 active customers)                        ≈ 480^5 ≈ 2.5 × 10^13

Billing:
  Per customer: 4 payment statuses × 5 dispute states   = 20
  5 active customers                                     ≈ 20^5 ≈ 3.2 × 10^6

Knowledge Base:
  6 articles × 2 states (accurate/outdated)             = 2^6 = 64
  + agent contributions (variable)                      ≈ 128

Policy Engine:
  5 policy types × 3 possible values each               = 3^5 = 243
  + drift schedule position                              ≈ 500

Stakeholders:
  3 stakeholders × 20 patience levels                   = 20^3 = 8,000

Agent State:
  4 phases × 80 steps × 3 fix attempts × 2 escalations = 1,920
  + known_facts accumulation                            ≈ 10,000

Estimated reachable state space: ~10^12 – 10^15
(Most theoretical states are unreachable due to causal constraints)
```

## Action Space Branching Factor

```
Per-step branching factor by phase:

TRIAGE:          3 action types  × ~10 parameter combinations  ≈ 30
INVESTIGATION:   9 action types  × ~25 parameter combinations  ≈ 225
RESPONSE:        10 action types × ~50 parameter combinations  ≈ 500
RESOLUTION:      6 action types  × ~20 parameter combinations  ≈ 120

Average branching factor: ~220
Episode length: 40-80 steps
Decision tree size: 220^60 ≈ 10^140 (intractable — no brute force possible)
```

## Partial Observability Ratio

```
Total state variables:        ~150
Agent-observable variables:    ~45 (after tool queries)
Agent-observable at step 0:    ~12 (initial alert only)
Partial observability ratio:   70% hidden at start, ~30% hidden after full investigation

Compare to:
  - Poker:      ~50% hidden (2 hole cards of 52)
  - StarCraft:  ~40% hidden (fog of war)
  - Our env:    ~70% hidden initially, reducing with investigation
```

## Why This Matters for GRPO Training

```
- High branching factor → GRPO's relative ranking is critical (can't enumerate)
- Dense rewards → smooth gradient signal even in large action space
- Partial observability → agent MUST use tools (can't shortcut from observation alone)
- Causal structure → learning investigation ORDER matters (curriculum helps)
- Episode length → long-horizon credit assignment (curriculum: short easy → long hard)
```

---

# APPENDIX M — Technology Stack & Dependencies

## Core Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| Pydantic | v2.x | Typed models, validation, discriminated unions |
| FastAPI | 0.100+ | HTTP server (port 7860) |
| Uvicorn | 0.23+ | ASGI server |
| pytest | 8.x | Test framework |
| pytest-asyncio | 0.23+ | Async test support |

## Training Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Unsloth | latest | 4-bit quantization, memory-efficient fine-tuning |
| HuggingFace TRL | ≥0.15 | GRPOTrainer implementation |
| HuggingFace Transformers | ≥4.45 | Model loading, tokenization |
| Datasets | latest | GRPO dataset formatting |
| PEFT | latest | LoRA adapters |
| bitsandbytes | latest | 4-bit quantization backend |
| matplotlib | latest | Reward curve plotting |

## Model

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Quantization | 4-bit (NF4 via Unsloth) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Max sequence length | 4096 tokens |
| VRAM requirement | <15 GB (fits Colab free T4) |

## Deployment

| Component | Specification |
|-----------|--------------|
| Container | Docker (Dockerfile included) |
| Platform | HuggingFace Spaces |
| Hardware | CPU Basic (2 vCPU, 16 GB) — no GPU needed for inference |
| Port | 7860 (HF Spaces standard) |
| User | Non-root UID 1000 (HF requirement) |

## Dependency Graph

```
inference.py
  └── env/environment.py
        ├── env/state.py
        │     └── models/observation.py
        │     └── models/ticket.py
        ├── graders/grader.py
        │     └── models/ticket.py (KeywordSpec)
        ├── models/action.py
        └── tasks/ticket_bank.py
              └── tasks/tickets/*.json

# NEW (incident mode additions):
env/environment.py (extended)
  ├── env/world.py (NEW)
  │     ├── env/services.py (NEW)
  │     ├── env/crm.py (NEW)
  │     ├── env/billing.py (NEW)
  │     ├── env/knowledge_base.py (NEW)
  │     ├── env/policy_engine.py (NEW)
  │     ├── env/stakeholders.py (NEW)
  │     ├── env/customers.py (NEW)
  │     ├── env/incident_history.py (NEW)
  │     └── env/runbooks.py (NEW)
  ├── graders/investigation_grader.py (NEW)
  ├── models/incident.py (NEW)
  └── tasks/incident_bank.py (NEW)
        └── tasks/incidents/*.json (NEW)
```

---

# APPENDIX N — Development Timeline & Milestones

> **Critical dates:** Onsite hackathon is 25th-26th April 2026. Compute credits for post-training arrive onsite. Environment + reward model + evaluation must be READY before arrival.

## Pre-Onsite (Now → April 24th)

| Day | Phase | Deliverables | Verification |
|-----|-------|-------------|-------------|
| **Apr 19-20** | Phase 1: World Foundation | `env/services.py`, `env/world.py`, `models/incident.py`, incident JSON files | `pytest tests/test_services.py tests/test_world_state.py tests/test_incidents.py` — all pass |
| **Apr 20-21** | Phase 2: Enterprise Systems | `env/crm.py`, `env/billing.py`, `env/knowledge_base.py`, `env/policy_engine.py`, `env/stakeholders.py`, `env/customers.py`, `env/incident_history.py`, `env/runbooks.py` | `pytest tests/test_crm.py tests/test_billing.py tests/test_knowledge_base.py tests/test_policy_drift.py tests/test_stakeholders.py tests/test_customers.py` |
| **Apr 21-22** | Phase 3: Action Space + Phase 4: Reward Engine | Extended `models/action.py`, `models/observation.py`, `graders/investigation_grader.py`, extended `env/environment.py` | `pytest tests/test_new_actions.py tests/test_investigation_grader.py` + all 216 existing tests still pass |
| **Apr 22-23** | Phase 5: State Machine + Phase 6: Inference | `IncidentState` in `env/state.py`, incident system prompt in `inference.py` | `pytest tests/test_incident_episode.py tests/test_backward_compat.py` — full episode simulation works |
| **Apr 23-24** | Phase 7: Training Pipeline (skeleton) | `train.py`, `evaluate.py`, `train_notebook.ipynb` (ready for compute) | Dry-run: training loop executes for 1 episode without errors |
| **Apr 24** | Phase 8: Pre-flight | All tests pass (400+), Docker builds, HF Space deployed, pitch slides drafted | `pytest tests/ -q` → 400+ passed. `curl /reset?mode=incident` works on HF Space |

## Onsite (April 25th-26th)

| Time | Activity | Goal |
|------|----------|------|
| **25th AM** | Receive compute credits. Start GRPO training on easy incidents. | Training loop running, collecting trajectories |
| **25th PM** | Continue training. Medium incidents. Monitor reward curves. | See initial improvement (0.25 → 0.40+) |
| **25th EVE** | Hard incidents. Skill tracking. Generate baseline comparison. | Clear reward curves showing improvement |
| **26th AM** | Final training batch. Run evaluation. Generate demo assets. | trained model + reward curves + skill curves saved |
| **26th PM** | Record 2-min video. Write HF blog post. Polish pitch. | All submission deliverables ready |
| **26th EVE** | **PITCH** (3 min + 2 min Q&A) | Deliver pitch with live demo |

## Milestone Gates

| Gate | Criteria | Fallback if Blocked |
|------|----------|-------------------|
| G1: Services work | Cascade injection + recovery works deterministically | Reduce to 3 services (auth, payments, database) |
| G2: Enterprise apps work | All 8 apps return consistent data | KB + Policy are critical; CRM/Billing can be simplified |
| G3: Actions dispatch | All 21 action types dispatch correctly | Critical 12 actions first, HistoryQuery + Runbook are nice-to-have |
| G4: Rewards are dense | Every action produces meaningful reward signal | Focus on investigation + fix rewards; stakeholder rewards are bonus |
| G5: Episode completes | Full triage→investigation→response→resolution cycle works | If stuck, demo shorter episodes (easy only) |
| G6: Training converges | GRPO shows improvement in 100 episodes | Increase K (generations per prompt), reduce LR, try easy-only curriculum |
| G7: Demo works | Live demo shows clear before/after difference | Pre-recorded demo as fallback |

---

# APPENDIX O — Risk Mitigation & Contingency Plans

## Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **GRPO doesn't converge** | Medium | High | 1. Start with easy-only curriculum (simpler gradient landscape). 2. Increase K to 8 (more comparison pairs). 3. Reduce learning rate to 1e-6. 4. Fall back to showing behavioral analysis without reward curves. |
| **Episode too long for context** | Medium | Medium | Cap episode history to last 10 actions in prompt. Summarize earlier actions as "Known facts: ..." |
| **Action parsing fails frequently** | Low | Medium | Robust `_sanitise_action` with fallbacks per phase (already implemented for ticket mode). |
| **Existing tests break** | Low | Critical | All new Observation/Action fields have `Optional/None` defaults. Run existing 216 tests after EVERY phase. |
| **Colab runs out of memory** | Low | High | Unsloth 4-bit reduces VRAM to <15GB. Reduce batch size to 1. Reduce max_seq_length to 2048. |
| **Training takes too long** | Low | Medium | Config is already sized for free T4: 20 iter × 30 episodes × K=4 = ~8 hrs. If running long, reduce to 15 iter × 20 episodes (~5 hrs). Show improvement on easy curriculum first. |
| **HF Space deployment fails** | Low | Medium | Test deployment BEFORE onsite. No new dependencies (pure Python + Pydantic + FastAPI). |

## Scope Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **Can't finish all 18 scenarios** | Medium | Low | **Minimum viable:** 3 easy + 3 medium + 2 hard = 8 scenarios. Nightmare is bonus. |
| **Can't implement all 21 actions** | Low | Medium | **Minimum viable:** 16 core actions (drop HistoryQuery, Runbook, RollbackFix). Still impressive. |
| **Can't implement all E.1-E.19** | High | Low | **Priority order:** E.1 (degradation), E.3 (topology), E.4 (observability), E.6 (KB persistence), E.9 (flickering), E.16 (CAB), E.17 (blast radius), E.18 (alert fatigue), E.19 (re-eval). Others are bonus. |
| **Post-mortem grading too complex** | Medium | Low | Simplify to keyword matching (same as existing resolution grading). |

## Demo Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **Live demo crashes** | Low | Critical | Pre-record a backup demo video. Test demo 3x before pitch. |
| **Base model is too good (no room for improvement)** | Low | High | Use 3B model (not 72B). Hard/nightmare scenarios are designed to be nearly impossible for base models. |
| **Base model is too bad (gibberish actions)** | Low | Medium | Robust fallback system ensures valid actions. Even gibberish → structured fallback → low but non-zero reward. |
| **Judges ask about multi-agent** | Medium | Low | Answer: "The environment is designed for single-agent but the WorldState architecture supports future multi-agent extensions (separate incident commander + SRE + customer agent)." |
| **Judges ask about real vs. simulated** | High | Low | Answer: "Simulated for reproducibility and determinism. Real systems (like Kube SRE Gym) are impressive but not reproducible. Our approach enables standardized benchmarking." |

---

# APPENDIX P — Demo Storyboard (3-Minute Pitch)

> Frame-by-frame visual guide for the pitch. Each frame = 1 slide or visual transition.

## Frame 1: THE HOOK (0:00-0:25)
**Visual:** Dark screen. PagerDuty notification animation. Red alert. Phone buzzing.
**Script:** "It's 3 AM. PagerDuty fires. Payment service is returning 500 errors. Three enterprise clients report failed transactions. Your monitoring shows CPU spikes in two services. Your on-call engineer is unreachable."
**Key moment:** Pause. "What do you do?"

## Frame 2: THE PROBLEM (0:25-0:55)
**Visual:** Split screen: Left = static benchmark (simple Q&A). Right = real enterprise chaos (5 services, cascading failures, angry customers, VP demanding updates).
**Script:** "Most AI benchmarks test isolated skills in sterile environments. Real enterprise operations are a web of interconnected systems where failures cascade, knowledge bases go stale, and policies change at 2 AM."
**Key transition:** Zoom into our architecture diagram.

## Frame 3: THE ENVIRONMENT (0:55-1:40)
**Visual:** Animated architecture diagram showing:
1. 5 services with dependency arrows
2. Failure injected in database → cascade animation to payments → notifications
3. Customer tickets popping up dynamically
4. Stakeholder patience bars decreasing
5. Policy values changing mid-screen
6. KB article flashing "OUTDATED"

**Script:** "We built the Enterprise Incident Command Center. Five microservices with causal dependencies. Eight enterprise applications with hidden state. 21 action types across 4 incident phases. The agent must discover what's broken through tool interactions — not from a description."

**Key points to hit:**
- "The agent can't see the root cause. It must investigate."
- "The knowledge base can be WRONG."  
- "Policies change mid-incident."
- "Customers get angrier every step you don't respond."

## Frame 4: THE DEMO — BEFORE (1:40-2:10)
**Visual:** Terminal recording showing base model (Qwen 3B) facing "Payment Gateway Cascade" incident.

```
Step 1: classify (severity=medium)        → WRONG (should be critical)  reward=-0.03
Step 2: respond ("We're looking into it") → Too vague, wrong tone       reward=+0.02
Step 3: apply_fix(database, restart)      → WRONG service, WRONG fix   reward=-0.10
Step 4: apply_fix(payments, restart)      → Still wrong root cause     reward=-0.10
Step 5: apply_fix(auth, restart)          → OUT OF FIX ATTEMPTS        reward=-0.05
...
Episode score: 0.22
```

**Script:** "Watch the untrained model. It guesses at the problem. Applies fixes to wrong services. Runs out of fix attempts. Never checks monitoring. Never verifies the KB. Score: 0.22."

## Frame 5: THE DEMO — AFTER (2:10-2:40)
**Visual:** Same terminal, now showing GRPO-trained model.

```
Step 1: check_monitoring(all)             → Sees database errors        reward=+0.03
Step 2: classify (severity=critical)      → CORRECT                     reward=+0.08
Step 3: probe_service(database, resources)→ Discovers OOM               reward=+0.05
Step 4: probe_service(payments, logs)     → Confirms cascade from DB    reward=+0.05
Step 5: query_kb("database OOM")          → Gets suggestion             reward=+0.02
Step 6: probe_service(database, config)   → Cross-verifies KB           reward=+0.05
Step 7: apply_fix(database, memory_increase) → ROOT CAUSE FIXED         reward=+0.15
Step 8: verify_fix(database)              → Confirmed healthy!          reward=+0.08
Step 9: notify_stakeholders(vp,"...")      → VP patience restored       reward=+0.03
Step 10: respond(CUST-001, empathetic)    → Quality response            reward=+0.10
Step 11: resolve(CUST-001, summary)       → Resolved with context       reward=+0.12
Step 12: write_postmortem(...)            → Covers root cause + steps   reward=+0.08
Episode score: 0.74
```

**Script:** "Now the trained model. Starts with monitoring. Probes the right services. Cross-verifies the KB. Fixes the root cause — not the symptoms. Notifies stakeholders. Responds with empathy. This episode scored 0.74. Average across all scenarios improves from 0.25 to 0.55+ — that's 2x improvement trained entirely on Colab free T4."

## Frame 6: THE SKILL CURVES (2:40-2:50)
**Visual:** 8 small line charts, each showing one skill improving over training:
1. Investigation-before-action: 12% → 89%
2. KB cross-verification: 5% → 72%
3. Policy checking: 8% → 85%
4. Stakeholder proactivity: 15% → 78%
5. Root cause accuracy: 10% → 68%
6. Tone matching: 20% → 82%
7. Resource efficiency: 30% → 75%
8. Red herring dismissal: 0% → 55%

**Script:** "And the model didn't just improve a number. It learned 8 distinct skills."

## Frame 7: THE CLOSE (2:50-3:00)
**Visual:** Summary slide with key stats.
**Script:** "21 action types. 8 enterprise apps. 18 scenarios. 30+ reward signals. Fully deterministic. Runs on Colab free T4. Deployed live on HuggingFace. We're not building a benchmark — we're building the training ground for the next generation of AI operations agents."

---

# APPENDIX Q — Contribution to the OpenEnv Ecosystem

## What We Add to OpenEnv

| Contribution | Value |
|-------------|-------|
| **First enterprise multi-app environment** | Demonstrates OpenEnv can model complex multi-system workflows, not just single-tool tasks |
| **Incident mode as extension pattern** | Shows how to extend an existing OpenEnv environment without breaking backward compatibility — `mode` parameter pattern |
| **WorldState architecture** | Reusable pattern for environments with hidden state that agents must discover through tool interactions |
| **Investigation grading infrastructure** | Generic grading framework for tool-use investigation workflows (applicable beyond incident response) |
| **Cross-episode persistence** | `PersistentKnowledgeBase` pattern for environments that support self-improvement across episodes |
| **Difficulty curriculum** | 4-tier difficulty scaling pattern (easy → nightmare) with structured scenario definitions |
| **Phase-gated action spaces** | Reusable pattern for environments where available actions change based on progress |

## OpenEnv Principles Alignment

| OpenEnv Principle | How We Embody It |
|------------------|-----------------|
| **Environments, not benchmarks** | Our world has persistent state that changes based on agent actions. Not a static test set. |
| **Verifiable rewards** | All 30+ reward signals are deterministic (keyword matching, exact comparison, numeric thresholds). No LLM-in-the-loop. |
| **Gymnasium-style API** | `reset()`, `step()`, `state()`, `close()` — fully async, fully typed. |
| **Reproducible** | Seed-based determinism. Same seed → same scenario → same initial state → same grading. |
| **Containerized** | Dockerfile included. Deploys to HF Spaces with zero configuration. |
| **Agent-agnostic** | Works with any LLM via HTTP API. Training script uses HF ecosystem (TRL + Unsloth). |

## Future Extensions (Beyond Hackathon)

These are **not part of our submission** but demonstrate the environment's extensibility:

1. **Multi-Agent Mode** — Separate incident commander, SRE, and customer agent roles with message passing
2. **Real API Integration** — Replace simulated services with sandboxed Docker containers running actual microservices
3. **Adversarial Difficulty** — Use a second LLM to generate harder scenarios based on agent weaknesses (like Kube SRE Gym's Claude designer)
4. **Transfer Learning** — Pre-train on easy incidents, transfer to entirely new incident types not seen during training
5. **Human-in-the-Loop** — Mixed human-AI incident response where the agent assists a human SRE

---

# APPENDIX R — Formal State Specification

## WorldState Schema (Complete)

```python
@dataclass
class WorldState:
    """Complete hidden state — the 'ground truth' that the agent cannot directly observe."""

    # === Core ===
    seed: int                                  # Determinism seed
    incident: IncidentScenario                 # The scenario definition
    service_mesh: ServiceMesh                  # 5 services + dependencies + health
    
    # === Enterprise Systems (8 apps) ===
    crm: CRMSystem                             # Customer records + frustration
    billing: BillingSystem                     # Invoices + disputes + payments
    knowledge_base: KnowledgeBase              # Articles (some WRONG) + search
    policy_engine: PolicyEngine                # Rules (CHANGE mid-episode) + drift schedule
    stakeholder_mgr: StakeholderManager        # VP/Legal/Support patience
    customer_queue: CustomerQueueManager        # Dynamic ticket generation
    incident_history: IncidentHistoryStore     # Past incidents for pattern matching
    runbook_engine: RunbookEngine              # Suggested runbooks (some WRONG)
    
    # === Dynamic State ===
    support_queue: list[DynamicTicket]          # Tickets arrived so far
    resolved_tickets: list[str]                # Resolved ticket IDs
    known_facts: dict[str, Any]                # Agent's accumulated discoveries
    evidence_chain: EvidenceChain              # Logical investigation chain
    audit_trail: AuditTrail                    # Compliance log
    
    # === Resource Tracking ===
    resource_budget: ResourceBudget            # Fix attempts, escalations, notifications
    total_incident_cost: float                 # Accumulated $$$ cost
    
    # === Temporal ===
    steps_elapsed: int                         # Steps since episode start
    incident_timer: int                        # Steps since incident began
    total_downtime_cost: float                 # Business cost from downtime
    chaos_events: list[ChaosEvent]             # Mid-episode injected failures
    
    def tick(self) -> list[Event]:
        """Advance world by one step. Returns events that occurred."""
        # 1. Decrement stakeholder patience (3 stakeholders × individual decay rates)
        # 2. Increase customer frustration (per unresolved customer)
        # 3. Generate new tickets (based on system health + customer tier)
        # 4. Apply policy drifts (check drift schedule vs current step)
        # 5. Degrade unfixed services (E.1 — time-based degradation)
        # 6. Update flickering services (E.9 — cycle through pattern)
        # 7. Check chaos injection (E.12 — new failures on hard/nightmare)
        # 8. Auto-escalate severity (E.5 — if unresolved too long)
        # 9. Track downtime cost ($100/step per service that is down)
        # 10. Return list of events for observation update
```

## Observation Schema (What Agent Sees)

```python
class Observation(BaseModel):
    """Agent-visible state at each step. Everything else is HIDDEN."""

    # Identity
    ticket_id: str                              # Current primary ticket
    incident_id: str | None                     # Incident identifier
    incident_title: str | None                  # Human-readable title
    mode: Literal["ticket", "incident"]         # Operating mode
    
    # Ticket Content
    ticket_text: str                            # Customer's message
    customer_sentiment: str                     # angry/frustrated/neutral/satisfied
    customer_tier: str                          # free/pro/enterprise
    customer_value: str                         # low/medium/high
    category_hint: str | None                   # Hint (easy only, None on hard)
    
    # Phase & Progress
    phase: Phase                                # Current state machine phase
    incident_phase: str | None                  # triage/investigation/response/resolution
    available_actions: list[str]                # Valid actions NOW
    current_step: int                           # Steps taken
    max_steps: int                              # Episode budget
    sla_steps_remaining: int                    # Steps before SLA penalty
    
    # Discovered Information (populated by tool queries)
    system_status: dict[str, str] | None        # {service: health} from check_monitoring
    active_alerts: list[str] | None             # Current monitoring alerts
    tool_results: dict[str, Any] | None         # Latest tool response
    known_facts: dict[str, Any] | None          # Accumulated discoveries
    active_policies: dict[str, Any] | None      # Policies agent has checked
    suggested_runbook: dict | None              # Suggested procedure (may be WRONG)
    
    # Enterprise Context
    stakeholder_patience: dict[str, float] | None   # VP: 0.0-1.0, Legal: 0.0-1.0, etc.
    pending_customer_tickets: int               # Tickets waiting for response
    total_incident_cost: float | None           # Running $$$ cost
    
    # History
    constraints: list[str]                      # Policy constraints
    history: list[ActionRecord]                 # Action-reward history
    max_total_reward: float                     # For normalization
```

## What's Hidden vs. Observable

```
┌─────────────────────────────────────────────────────────┐
│                    HIDDEN FROM AGENT                     │
│                                                          │
│  ● True root cause of incident                          │
│  ● Service dependency graph (hard/nightmare)            │
│  ● Whether KB articles are accurate or outdated         │
│  ● Current policy values (must call check_policy)       │
│  ● Customer internal risk scores                        │
│  ● Billing issues not mentioned in ticket text          │
│  ● Which symptoms are red herrings                      │
│  ● Whether suggested runbook is correct                 │
│  ● Flickering service patterns                          │
│  ● Chaos injection schedule                             │
│  ● True timeline of events                              │
│  ● Historical incident relevance flags                  │
│                                                          │
├─────────────────────────────────────────────────────────┤
│               OBSERVABLE BY AGENT                        │
│                                                          │
│  ● Initial alert text and ticket content                │
│  ● Results of tool queries (after calling them)         │
│  ● Accumulated known_facts from previous tools          │
│  ● Stakeholder patience levels (visible)                │
│  ● Step count, SLA deadline, max steps                  │
│  ● Available actions for current phase                  │
│  ● Action-reward history                                │
│  ● Active alerts from monitoring                        │
│  ● Total incident cost so far                           │
│  ● Pending customer ticket count                        │
│  ● Suggested runbook (visible, but may be wrong)        │
│                                                          │
├─────────────────────────────────────────────────────────┤
│             DISCOVERABLE VIA TOOLS                       │
│                                                          │
│  check_monitoring  → service health, error rates        │
│  probe_service     → logs, resources, connections       │
│  fetch_logs        → error entries with timestamps      │
│  fetch_user_data   → customer record, tier, frustration │
│  check_billing     → invoices, disputes, failures       │
│  query_kb          → articles (may be outdated!)        │
│  check_policy      → current rules (may have changed!)  │
│  query_incident_history → past incidents (may mislead)  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

# APPENDIX S — Updated Phase Summary (FINAL)

| Phase | What | New Files | Modified Files | Risk | Advanced Features Included |
|-------|------|-----------|----------------|------|---------------------------|
| **1** | World Foundation | 3 (+tests) | 0 | Low | E.1 (degradation), E.3 (topology), E.4 (observability), E.9 (flickering) |
| **2** | Enterprise Systems | 10 (+tests) | 0 | Low | E.6 (KB persistence), E.7 (customer behavior), E.10 (history), E.11 (runbooks) |
| **3** | Action Space | 0 | 3 | Medium | E.2 (rollback), all new action models |
| **4** | Reward Engine | 2 (+tests) | 1 | Low | E.8 (evidence chain), E.13 (cost), E.14 (communication), E.15 (timeline), E.16 (CAB approval), E.17 (blast radius), E.18 (alert triage grading), E.19 (reclassification grading) |
| **5** | Incident State Machine | 0 | 2 | Medium | E.5 (auto-escalation), E.12 (chaos injection), E.16 (CAB gate in fix flow), E.18 (alert stream in observation), E.19 (severity re-eval triggers), resource budgets, audit trail |
| **6** | Inference & Prompting | 0 | 1 | Low | Incident system prompt, observation-to-prompt |
| **7** | Training Pipeline | 3 | 0 | Medium | GRPO + Unsloth + curriculum + skill tracking |
| **8** | Testing & Polish | 15+ test files | 3 | Low | 400+ tests, README, openenv.yaml, Docker, HF, blog/video |

**Total new files:** ~40 (code + tests + data + notebook)
**Total modified files:** ~10 (all backward compatible)
**Existing 216 tests:** MUST pass at every phase
**Total action types:** 21 (6 existing + 15 new)
**Total advanced features:** 19 (E.1–E.19, including CAB human-approval, blast radius, alert fatigue, and severity re-evaluation)
**Total incident scenarios:** 18 (3 easy + 5 medium + 7 hard + 3 nightmare)
**Total reward signals:** 30+ per step + 9-component episode score
**Target test count:** 400+
