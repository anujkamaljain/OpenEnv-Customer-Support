# It's 3 AM. Payments are failing. Can your AI keep its cool?

There is a moment in every on-call shift that you remember.
Ours was a Tuesday at 3:14 AM. Pager screaming. Auth service flapping.
Payments queue backing up. Stakeholders on the bridge asking the same
question every two minutes: *"What is happening?"*

We did what every SRE does. Opened five tabs. Pulled metrics. Tailed
logs. Compared against an outdated runbook. Pinged the database team.
Tried a restart. It sort of worked. We patched it, wrote a postmortem
the next morning, and life moved on.

A year later we started building agents on top of LLMs and noticed
something uncomfortable: the models that could solve LeetCode in a few
seconds would *fall apart* the moment they had to hold a multi-step
investigation in their head. They would jump to a fix before reading
the alert. They would trust the knowledge base when the knowledge base
was wrong. They would tell the customer "we're working on it" while
the database was actively melting.

That gap is what this project is about.

---

## The thing we built

It is called **EICC** - *Enterprise Incident Command Center*. It is an
**OpenEnv environment** that turns a real on-call shift into a training
ground for LLM agents. Not a quiz, not a benchmark with a single right
answer - a world that punches back.

You wake the agent up with a customer ticket. Behind that ticket is a
mesh of five microservices that can fail and cascade into each other,
eight enterprise tools the agent has to learn how to use (monitoring,
CRM, billing, knowledge base, policy engine, incident history, runbooks,
stakeholder manager), and a clock that does not stop just because the
agent is confused.

The agent cannot see the truth. The root cause is hidden. The KB is
sometimes outdated. The policies have drifted since the last training
data. New tickets keep arriving while it is still investigating the
first one. If it shouts "I have fixed it!" at the wrong moment,
patience drops, customers escalate, and the reward turns negative.

Every action is one of **21 typed actions** across four phases:
*triage*, *investigation*, *response*, *resolution*. Reward are given
by our RL engine to the LLM. The same seed always produces the same
scenario, so we can actually measure whether the agent got better.

---

## Why this is a "Theme #3.1 World Modeling - Professional Tasks" submission

The hackathon brief for sub-theme #3.1 asked for environments that
*require real interaction with tools, APIs, or dynamic systems where the
model is expected to do real hard work instead of exploiting short-cuts*,
where agents have to *maintain consistent internal state, update beliefs
based on outcomes, and orchestrate multi-step workflows*, with the goal
of *strengthening causal reasoning and persistent world models*. Expected
outcome: *an environment capturing nuances of a defined partially
observable world and improving LLM interaction with it.*

That is the brief we designed to. EICC is not a single-turn benchmark.
The root cause is hidden. The KB lies sometimes. Policies drift. The
service mesh has real causal structure - hit the wrong node and the
blame propagates. The agent has to call tools, read the returns,
update what it thinks is going on, and pick the next action - over
and over, for up to 80 steps in the hardest tier. There are 11
explicit anti-shortcut mechanisms baked into the reward (phase gating,
investigation-before-action, KB cross-verification, blast-radius
penalties, CAB approvals for risky changes, resource budgets, tone
matching, ...) so the model cannot game its way to a good score.

And the "real systems" requirement is taken literally in Mode 3: the
exact same agent, the exact same action API, but routed to a live
5-service Docker cluster with a chaos controller. When the agent says
"restart auth," something actually restarts.

---

## Three ways to play

Most environments give you one mode. EICC gives you three, because
training and demoing have different needs.

1. **Ticket mode** is the most basic one: classify, route, respond,
   resolve. It exists for backward compatibility and as a sanity ramp.

2. **Mock environment** is where training happens, but it is mock
   because it tries to replicate the database, auth service, etc.
   using python functions - no real scenarios. It is a deterministic
   simulation of the whole world: services, tools, customers,
   stakeholder patience. It is fast, free, and reproducible. We train
   GRPO on top of `Qwen2.5-3B-Instruct` here using Unsloth + TRL.

3. **VM environment** is our USP. It replicates the environment as
   close as possible to real-life conditions. The same `/reset` and
   `/step` API, but now backed by a **real Docker cluster of five
   microservices** plus a chaos controller. When the agent says
   *"restart the auth service,"* something actually restarts. When it
   says *"verify_fix,"* we hit a real `/health` endpoint. Sim-to-sandbox
   transfer scoring tells us how much of what was learned in the cheap
   simulation actually carries over to live infrastructure. There is a
   deterministic **drill mode** that injects fresh failures mid-episode
   so we can score recovery quality, not just lucky first hits.

That last bit - the same agent, scored on simulation and on a live
cluster, with a number that says *"this much of your learning
transferred"* - is the part we are most proud of.

---

## What the agent has to learn

The reward is deliberately mean. It rewards behavior that real
incident commanders use, not behavior that looks confident:

- *Investigate before acting.* Apply a fix without checking
  monitoring, you lose points.
- *Cross-verify the KB.* Trust an outdated runbook, blast radius hits
  customers, you lose points.
- *Respect policy drift.* Push a risky change without a CAB approval
  gate, you lose points.
- *Match tone.* Be cheerful at a furious enterprise customer, you
  lose points.
- *Don't spam stakeholders.* Notify before you have facts, patience
  drops faster.
- *Keep the JSON clean.* Long, rambling generations get penalized.

There are eight tracked behavioral skills behind the curtain:
investigation-before-action, KB cross-verification, policy checking,
stakeholder proactivity, root-cause accuracy, tone matching, resource
efficiency, and red-herring dismissal. The judges (and you, in the
notebook) can read all eight per run, side by side, baseline vs trained.

---

## Baseline vs trained: who is on each side of the curve

Every plot in this submission has two lines. They look like just
"before" and "after," but the actual identity of each side matters.

The **baseline** is the **untrained** `Qwen2.5-3B-Instruct` walking
into the incident world cold. No exposure to our reward function, no
LoRA adapter, nothing - just the off-the-shelf instruction-tuned model
trying to fight a fire it has never seen. Most of the time it does
what you would expect a smart-but-naive engineer to do: jump to a fix,
miss a step, send a confident customer email while the database is
still on fire.

The **trained** side is the **same model**, same prompts, same
scenarios, same seeds - only the weights are different. The LoRA
adapter on top has been updated by GRPO using the rewards from the
environment in this repo. Nothing else changed. So when the trained
line sits above the baseline line, that gap is *exactly* the behavior
our environment taught the model. Not a different model. Not a
different prompt. Just learning.

We keep that distinction visible in the artifacts on purpose. Every
`trained_report.json` carries a `policy_used` field
(`trained_checkpoint` if the LoRA adapter loaded, `trained_heuristic`
as a guarded fallback). If a number ever looks too good, you can read
one line of JSON and find out which policy actually produced it.
No magic, no asterisks.

---

## What the curves actually show

After training, three plots fall out - one for *easy*, one for
*medium*, one for *hard*. Each plot has two lines, baseline and
trained. The gap between the two lines is the part that matters: it
is the behavior the model picked up that the base model did not have.

You can open the actual files we shipped under [`results/`](./results/)
in the repo. `results/simple/` is the Mock-environment run.
`results/sandbox/` is the same trained checkpoint scored again on the
live container cluster. `results/training/reward_history.json` is the
training-time reward signal from Phase 1. No re-running required -
just click and read.

---

## The thing we learned building this

The first version of the env was too easy. The agent learned to
shortcut: skip investigation, guess the fix, take the points, move on.
So we added partial observability. Then it learned to spam tools.
So we added a resource budget. Then it learned to be polite to
everyone. So we added tone matching against actual sentiment. Each
shortcut closed produced a better agent.

That is what an environment is supposed to do. The design of the
world *is* the curriculum.

---

## Try it yourself

The HF Space is live. The training notebook walks through both the
Mock lane and the VM lane on a Colab GPU. Reproducing the headline
numbers takes about an hour on an A10. Pull the repo, open the
notebook, follow the steps. The first time the trained reward curve
crosses the baseline curve on the *hard* difficulty plot, you'll know
exactly why we built this.

---

*Built on OpenEnv. Trained with GRPO via Unsloth + TRL. Deployed on
Hugging Face Spaces. Source and notebook are public - links in the
repo README.*
