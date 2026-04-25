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

  Incident phases:
    TRIAGE -> INVESTIGATION -> RESPONSE -> RESOLUTION
```
