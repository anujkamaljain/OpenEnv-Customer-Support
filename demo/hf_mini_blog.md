# EICC in 2 Minutes: Training AI for Incident Response

Enterprise incident response is a hard world-modeling problem: partial observability, cascading dependencies, policy drift, and communication pressure all happen at once.

EICC extends OpenEnv customer-support triage into a full incident command simulation:

- 21 actions across triage, investigation, response, and resolution
- 5-service dependency mesh with cascading failures
- 8 enterprise systems (monitoring, CRM, billing, KB, policy, history, runbooks, stakeholders)
- deterministic scoring and seeded reproducibility

Why this matters:

- Agents must investigate before acting, not memorize scripts.
- Agents must update beliefs as evidence changes.
- Training improvements are measurable via formal metrics and behavior diffs.

You can run dry-run training locally:

```bash
python train.py --iterations 1 --episodes 1 --k 2 --dry-run
```

And compare behavior:

```bash
python evaluate.py --policy compare --episodes-per-difficulty 5 --plot
```

EICC is designed as a reusable OpenEnv pattern for enterprise workflows where tool use, causality, and long-horizon reasoning all matter.
