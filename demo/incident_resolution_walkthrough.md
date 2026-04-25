# Incident Resolution Walkthrough

## Scenario

- Incident: payment processing failures
- Difficulty: easy
- Goal: show deterministic triage to resolution path

## Example Trajectory

1. `check_monitoring` (all services)  
   - Finds elevated error rate in `database` and degraded `payments`.
2. `classify` (`priority=critical`)  
   - Correctly sets urgency for customer/business impact.
3. `probe_service` (`database`, `resources`)  
   - Confirms resource exhaustion signal.
4. `apply_fix` (`database`, `<root-cause-aligned fix_type>`)  
   - Fix type is selected from gathered evidence (not hardcoded).
5. `verify_fix` (`database`)  
   - Service returns `healthy`; dependent services recover.
6. `notify_stakeholders` (`all`)  
   - Patience restored before decay threshold.
7. `respond` (`empathetic`)  
   - Customer-facing update with clear status.
8. `resolve`  
   - Ticket closure with verified remediation summary.
9. `write_postmortem`  
   - Captures root cause, remediation, prevention measures.

## Evaluation Transparency

- In report artifacts, `policy_used` explicitly records whether behavior came from
  `trained_checkpoint` or `trained_heuristic`.
- Guarded fallback is intentional and visible, so benchmark numbers are auditable.
- New `transfer_report.json` compares simulated-vs-sandbox gains, so we can show
  whether trained behavior actually transfers to live infrastructure actions.

## Why This Matters

- Demonstrates investigation-before-action behavior.
- Shows policy of fixing root cause, not symptom.
- Preserves deterministic grading and reproducibility.
- Drill mode can inject fresh failures mid-episode, so we also test recovery
  quality under changing conditions via deterministic `drill_score`.
