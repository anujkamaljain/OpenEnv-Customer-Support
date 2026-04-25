# Local Test Runbook (Windows + PowerShell)

This guide is intentionally explicit so you can run a full sandbox validation
without guessing any steps.

---

## 0) Prerequisites

Install and verify:

- Docker Desktop (running)
- Python 3.11+
- Git
- `curl.exe` (bundled with Windows 10+; do NOT rely on the `curl` alias which on
  Windows PowerShell 5.x is mapped to `Invoke-WebRequest` and does not accept
  POSIX-style flags like `-X POST -d ...`)

PowerShell checks:

```powershell
docker --version
docker compose version
python --version
git --version
curl.exe --version
```

If Docker is not running, start Docker Desktop first.

---

## 1) Open PowerShell in repo root

```powershell
cd F:\Coding\OpenEnv
```

Confirm:

```powershell
ls
```

You should see `sandbox`, `server`, `env`, `evaluate.py`.

---

## 2) Build and start sandbox cluster

```powershell
docker compose -f "sandbox/docker-compose.yml" up --build -d
```

Check containers:

```powershell
docker compose -f "sandbox/docker-compose.yml" ps
```

---

## 3) Validate service health endpoints

```powershell
curl.exe http://localhost:5001/health
curl.exe http://localhost:5002/health
curl.exe http://localhost:5003/health
curl.exe http://localhost:5004/health
curl.exe http://localhost:5005/health
curl.exe http://localhost:6660/health
curl.exe http://localhost:6660/chaos/status
```

Expected: JSON responses for all endpoints.

---

## 4) Create Python env and install project deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
```

---

## 5) Start OpenEnv API in sandbox mode

Use a **new PowerShell window** for the API server:

```powershell
cd F:\Coding\OpenEnv
.\.venv\Scripts\Activate.ps1
$env:OPENENV_SANDBOX="true"
$env:OPENENV_SANDBOX_CLUSTER_URL="http://localhost"
$env:OPENENV_SANDBOX_CHAOS_URL="http://localhost:6660"
python -m server.app
```

Keep this running.

---

## 6) API smoke test in another PowerShell window

```powershell
cd F:\Coding\OpenEnv
.\.venv\Scripts\Activate.ps1
python sandbox\smoke_test.py --base-url http://localhost --api-url http://localhost:7860
```

Expected final line: `Sandbox smoke test passed.`

---

## 7) Manual step-by-step API checks

### 7.1 Reset incident

```powershell
$bodyObj = @{
  mode = "incident"
  difficulty = "easy"
  seed = 0
}
$body = $bodyObj | ConvertTo-Json -Compress
$body

Invoke-RestMethod -Method Post -Uri "http://localhost:7860/reset" `
  -Headers @{ "X-Session-ID" = "demo-session" } `
  -ContentType "application/json" `
  -Body $body
```

### 7.1b Reset incident with drill mode

```powershell
$bodyObj = @{
  mode = "incident"
  difficulty = "hard"
  seed = 0
  drill_mode = $true
  drill_seed = 7
}
$body = $bodyObj | ConvertTo-Json -Compress
$body

Invoke-RestMethod -Method Post -Uri "http://localhost:7860/reset" `
  -Headers @{ "X-Session-ID" = "drill-session" } `
  -ContentType "application/json" `
  -Body $body
```

### 7.2 Check monitoring

```powershell
$bodyObj = @{
  action = @{
    action_type = "check_monitoring"
  }
}
$body = $bodyObj | ConvertTo-Json -Compress
$body

Invoke-RestMethod -Method Post -Uri "http://localhost:7860/step" `
  -Headers @{ "X-Session-ID" = "demo-session" } `
  -ContentType "application/json" `
  -Body $body
```

### 7.3 Fetch logs

```powershell
$bodyObj = @{
  action = @{
    action_type = "fetch_logs"
    service_name = "auth"
    time_range = "last_5m"
  }
}
$body = $bodyObj | ConvertTo-Json -Compress
$body

Invoke-RestMethod -Method Post -Uri "http://localhost:7860/step" `
  -Headers @{ "X-Session-ID" = "demo-session" } `
  -ContentType "application/json" `
  -Body $body
```

### 7.4 Apply + verify fix

```powershell
$applyBodyObj = @{
  action = @{
    action_type = "apply_fix"
    service_name = "auth"
    fix_type = "restart_service"
  }
}
$applyBody = $applyBodyObj | ConvertTo-Json -Compress
$applyBody

Invoke-RestMethod -Method Post -Uri "http://localhost:7860/step" `
  -Headers @{ "X-Session-ID" = "demo-session" } `
  -ContentType "application/json" `
  -Body $applyBody

$verifyBodyObj = @{
  action = @{
    action_type = "verify_fix"
    service_name = "auth"
  }
}
$verifyBody = $verifyBodyObj | ConvertTo-Json -Compress
$verifyBody

Invoke-RestMethod -Method Post -Uri "http://localhost:7860/step" `
  -Headers @{ "X-Session-ID" = "demo-session" } `
  -ContentType "application/json" `
  -Body $verifyBody
```

### 7.5 Confirm sandbox payload is attached

In responses, check:

- `info.sandbox`
- `observation.tool_results.sandbox_live`

---

## 8) Sandbox evaluation runs

Use two profiles:

- Smoke check (`--episodes-per-difficulty 1`) for quick verification.
- Final reporting (`--episodes-per-difficulty 5` or `10`) for stable metrics.

### 8.1 Smoke check (fast)

```powershell
cd F:\Coding\OpenEnv
.\.venv\Scripts\Activate.ps1
python evaluate.py --policy compare --episodes-per-difficulty 1 --sandbox --output-dir artifacts/eval_sandbox
```

### 8.2 Final sandbox report (recommended)

```powershell
python evaluate.py --policy compare --compare-trained-policy trained_checkpoint --checkpoint-dir artifacts/train/trained_adapter --checkpoint-base-model Qwen/Qwen2.5-3B-Instruct --episodes-per-difficulty 5 --sandbox --output-dir artifacts/eval_sandbox
```

For publication-quality numbers, use `--episodes-per-difficulty 10`.

`trained_checkpoint` requires a completed training run that produced
`artifacts/train/trained_adapter`.

### 8.3 Final transfer benchmark (sim + sandbox in one run)

```powershell
python evaluate.py --policy compare --compare-trained-policy trained_checkpoint --checkpoint-dir artifacts/train/trained_adapter --checkpoint-base-model Qwen/Qwen2.5-3B-Instruct --episodes-per-difficulty 5 --transfer-report --output-dir artifacts/eval_transfer
```

### 8.4 Final transfer benchmark with drill mode

```powershell
python evaluate.py --policy compare --compare-trained-policy trained_checkpoint --checkpoint-dir artifacts/train/trained_adapter --checkpoint-base-model Qwen/Qwen2.5-3B-Instruct --episodes-per-difficulty 5 --transfer-report --sandbox-drill-mode --sandbox-drill-seed 7 --output-dir artifacts/eval_transfer_drill
```

### 8.5 Mentor-friendly plots (easy/medium/hard stage curves)

Run compare with plotting enabled (use `7` for your requested 6-7 stages):

```powershell
python evaluate.py --policy compare --compare-trained-policy trained_checkpoint --checkpoint-dir artifacts/train/trained_adapter --checkpoint-base-model Qwen/Qwen2.5-3B-Instruct --episodes-per-difficulty 7 --plot --sandbox --output-dir artifacts/eval_sandbox
```

Generated files:

- `artifacts/eval_sandbox/reward_curve_easy.png`
- `artifacts/eval_sandbox/reward_curve_medium.png`
- `artifacts/eval_sandbox/reward_curve_hard.png`

---

## 9) Run tests

```powershell
python -m pytest tests/test_sandbox_env.py -q
python -m pytest tests/test_server_api.py -q
```

---

## 10) Shutdown + cleanup

Stop server with `Ctrl+C` in API window.

Then:

```powershell
docker compose -f "sandbox/docker-compose.yml" down
```

Optional full cleanup (remove images/volumes):

```powershell
docker compose -f "sandbox/docker-compose.yml" down -v --rmi local
```

---

## Troubleshooting

- `connection refused` on ports 500x/6660: cluster not running -> run compose up.
- `sandbox info missing` in `/step`: ensure `OPENENV_SANDBOX=true` in server window.
- `Docker not available`: start Docker Desktop and retry.
- Health is `down` right after reset: expected if failure injected; use `apply_fix` then `verify_fix`.
- PowerShell request parsing issues: use `Invoke-RestMethod` examples in section 7
  (recommended over `curl`/`curl.exe` on Windows).
- Port already in use (5001-5005, 6660, 7860): stop conflicting processes or change
  the host port mapping in `sandbox/docker-compose.yml`.

