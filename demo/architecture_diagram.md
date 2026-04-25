# EICC Architecture Diagram

```text
                         ENTERPRISE INCIDENT COMMAND CENTER

  Services (causal mesh):
    AUTH --------> PAYMENTS --------> NOTIFICATIONS
      \               |
       \              v
        -------> DATABASE <-------- ANALYTICS

  Enterprise systems:
    - Monitoring (health, alerts, logs)
    - CRM (customer profile + risk/frustration)
    - Billing (invoices, payment failures, disputes)
    - Knowledge Base (may contain outdated guidance)
    - Policy Engine (drifts over time)
    - Incident History (similar incidents + outcomes)
    - Runbook Engine (suggested procedures)
    - Stakeholder Manager (patience and comms impact)

  Agent loop:
    Observation -> Action -> Environment transition -> Reward -> Next observation
    (evaluation logs `policy_used` for checkpoint/fallback provenance)

  Dual-backend evaluation:
    Simulated backend (official deterministic score)
    Sandbox backend (live container cluster)
    -> transfer_report.json quantifies sim->sandbox skill transfer
    -> optional drill mode injects deterministic mid-episode failures
       and logs drill_score for response quality under fresh disruptions

  Incident phases:
    TRIAGE -> INVESTIGATION -> RESPONSE -> RESOLUTION
```
