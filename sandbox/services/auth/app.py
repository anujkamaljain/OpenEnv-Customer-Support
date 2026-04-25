"""Auth sandbox service."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

app = FastAPI(title="sandbox-auth", version="0.1.0")

STARTED = time.time()
ACTIVE_FAILURE: str | None = None
LOGS: deque[str] = deque(maxlen=500)


class ChaosInjectRequest(BaseModel):
    failure_mode: str
    severity: str = "high"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(level: str, message: str, **extra: Any) -> None:
    payload = {"ts": _now(), "service": "auth", "level": level, "message": message, **extra}
    LOGS.append(str(payload))


def _status() -> str:
    if ACTIVE_FAILURE in {"config_corruption", "oom"}:
        return "down"
    if ACTIVE_FAILURE in {"rate_limiting", "token_expiry"}:
        return "degraded"
    return "healthy"


@app.get("/health")
def health() -> dict[str, Any]:
    return {"service": "auth", "status": _status(), "failure_mode": ACTIVE_FAILURE, "uptime_s": int(time.time() - STARTED)}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    status = _status()
    health_value = {"healthy": 1.0, "degraded": 0.5, "down": 0.0}[status]
    error_rate = 0.01 if status == "healthy" else (0.25 if status == "degraded" else 0.95)
    latency = 0.04 if status == "healthy" else (0.35 if status == "degraded" else 2.0)
    return (
        f'service_health{{service="auth"}} {health_value}\n'
        f'service_error_rate{{service="auth"}} {error_rate}\n'
        f'service_latency_seconds{{service="auth",quantile="0.99"}} {latency}\n'
    )


@app.get("/logs")
def logs(lines: int = 50) -> dict[str, Any]:
    size = max(1, min(lines, 500))
    return {"service": "auth", "entries": list(LOGS)[-size:]}


@app.post("/chaos/inject")
def inject(req: ChaosInjectRequest) -> dict[str, Any]:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = req.failure_mode
    _log("WARN", "failure injected", failure_mode=req.failure_mode, severity=req.severity)
    return {"status": "ok", "service": "auth", "failure_mode": ACTIVE_FAILURE}


@app.post("/chaos/clear")
def clear() -> dict[str, Any]:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = None
    _log("INFO", "failure cleared")
    return {"status": "ok", "service": "auth", "failure_mode": ACTIVE_FAILURE}


@app.post("/login")
def login() -> dict[str, Any]:
    status = _status()
    if status == "down":
        _log("ERROR", "login failed", reason="service down")
        return {"ok": False, "error": "service_down"}
    if ACTIVE_FAILURE == "rate_limiting":
        _log("WARN", "login throttled")
        return {"ok": False, "error": "rate_limited"}
    return {"ok": True, "token": "sandbox-token"}


@app.get("/validate")
def validate() -> dict[str, Any]:
    if _status() == "down":
        _log("ERROR", "token validation failed", reason="service down")
        return {"ok": False, "error": "service_down"}
    if ACTIVE_FAILURE == "token_expiry":
        _log("WARN", "token validation failed", reason="forced expiry")
        return {"ok": False, "error": "expired"}
    return {"ok": True}

