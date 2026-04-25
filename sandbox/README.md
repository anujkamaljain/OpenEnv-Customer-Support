# Sandbox Mode (Live Cluster)

This directory contains a production-like sandbox cluster used for **evaluation/demo**.
Training remains on the deterministic simulated environment.

## Start cluster

```bash
docker compose -f sandbox/docker-compose.yml up --build -d
```

Services:

- `auth` -> `http://localhost:5001`
- `database` -> `http://localhost:5002`
- `payments` -> `http://localhost:5003`
- `analytics` -> `http://localhost:5004`
- `notifications` -> `http://localhost:5005`
- `chaos-controller` -> `http://localhost:6660`
- `prometheus` -> `http://localhost:9090`

## Quick sanity checks

```bash
curl http://localhost:5001/health
curl http://localhost:6660/chaos/status
curl -X POST http://localhost:6660/chaos/inject -H "Content-Type: application/json" -d "{\"service\":\"auth\",\"failure_mode\":\"rate_limiting\"}"
curl http://localhost:5001/health
curl -X POST http://localhost:6660/chaos/verify_fix -H "Content-Type: application/json" -d "{\"service\":\"auth\",\"fix_type\":\"restart_service\"}"
curl http://localhost:5001/health
```

## Use from OpenEnv server

Set environment flags and start the API:

```bash
set OPENENV_SANDBOX=true
set OPENENV_SANDBOX_CLUSTER_URL=http://localhost
set OPENENV_SANDBOX_CHAOS_URL=http://localhost:6660
python -m server.app
```

Now `/reset` + `/step` continue using the same action schema, but incident actions
include live sandbox data under `info.sandbox` and `observation.tool_results.sandbox_live`.

## Drill mode (failure curriculum)

Drill mode injects deterministic mid-episode failures using `drill_mode` on reset:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"mode":"incident","difficulty":"hard","seed":0,"drill_mode":true,"drill_seed":7}'
```

Per-step drill telemetry is available in:

- `info.sandbox.drill`
- `observation.tool_results.sandbox_live.drill`

## Automated smoke test

After starting both cluster and API server:

```bash
python sandbox/smoke_test.py --base-url http://localhost --api-url http://localhost:7860
```

For a full Windows-focused walkthrough, see:

- `sandbox/LOCAL_TEST_RUNBOOK.md`

