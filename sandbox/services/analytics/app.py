"""Analytics sandbox service."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections import deque
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

app = FastAPI(title="sandbox-analytics", version="0.1.0")

DB_URL = os.environ.get("DB_URL", "http://database:5002").rstrip("/")
STARTED = time.time()
ACTIVE_FAILURE: str | None = None
LOGS: deque[str] = deque(maxlen=500)


class ChaosInjectRequest(BaseModel):
    failure_mode: str
    severity: str = "high"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(level: str, message: str, **extra: Any) -> None:
    payload = {"ts": _now(), "service": "analytics", "level": level, "message": message, **extra}
    LOGS.append(str(payload))


def _db_status() -> str:
    try:
        req = urllib.request.Request(f"{DB_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return "down"
    status = str(payload.get("status", "down"))
    return status if status in {"healthy", "degraded", "down"} else "down"


def _status() -> str:
    if ACTIVE_FAILURE in {"batch_job_runaway", "query_timeout"}:
        return "down"
    if ACTIVE_FAILURE in {"stale_cache"}:
        return "degraded"
    db = _db_status()
    if db == "down":
        return "down"
    if db == "degraded":
        return "degraded"
    return "healthy"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "service": "analytics",
        "status": _status(),
        "failure_mode": ACTIVE_FAILURE,
        "depends_on": {"database": _db_status()},
        "uptime_s": int(time.time() - STARTED),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    status = _status()
    health_value = {"healthy": 1.0, "degraded": 0.5, "down": 0.0}[status]
    error_rate = 0.01 if status == "healthy" else (0.2 if status == "degraded" else 0.9)
    latency = 0.05 if status == "healthy" else (0.55 if status == "degraded" else 2.5)
    return (
        f'service_health{{service="analytics"}} {health_value}\n'
        f'service_error_rate{{service="analytics"}} {error_rate}\n'
        f'service_latency_seconds{{service="analytics",quantile="0.99"}} {latency}\n'
    )


@app.get("/logs")
def logs(lines: int = 50) -> dict[str, Any]:
    size = max(1, min(lines, 500))
    return {"service": "analytics", "entries": list(LOGS)[-size:]}


@app.post("/chaos/inject")
def inject(req: ChaosInjectRequest) -> dict[str, Any]:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = req.failure_mode
    _log("WARN", "failure injected", failure_mode=req.failure_mode, severity=req.severity)
    return {"status": "ok", "service": "analytics", "failure_mode": ACTIVE_FAILURE}


@app.post("/chaos/clear")
def clear() -> dict[str, Any]:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = None
    _log("INFO", "failure cleared")
    return {"status": "ok", "service": "analytics", "failure_mode": ACTIVE_FAILURE}


@app.get("/report")
def report() -> dict[str, Any]:
    status = _status()
    if status == "down":
        _log("ERROR", "report generation failed", reason="service down")
        return {"ok": False, "error": "analytics_down"}
    _log("INFO", "report generated", status=status)
    return {"ok": True, "status": status, "generated_at": _now()}

