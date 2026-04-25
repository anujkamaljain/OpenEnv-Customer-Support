"""Notifications sandbox service."""

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

app = FastAPI(title="sandbox-notifications", version="0.1.0")

PAYMENTS_URL = os.environ.get("PAYMENTS_URL", "http://payments:5003").rstrip("/")
STARTED = time.time()
ACTIVE_FAILURE: str | None = None
LOGS: deque[str] = deque(maxlen=500)
QUEUE_DEPTH = 0


class ChaosInjectRequest(BaseModel):
    failure_mode: str
    severity: str = "high"


class SendRequest(BaseModel):
    customer_id: str
    message: str


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(level: str, message: str, **extra: Any) -> None:
    payload = {"ts": _now(), "service": "notifications", "level": level, "message": message, **extra}
    LOGS.append(str(payload))


def _payments_status() -> str:
    try:
        req = urllib.request.Request(f"{PAYMENTS_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return "down"
    status = str(payload.get("status", "down"))
    return status if status in {"healthy", "degraded", "down"} else "down"


def _status() -> str:
    if ACTIVE_FAILURE in {"template_error", "queue_overflow"}:
        return "down"
    if ACTIVE_FAILURE in {"rate_exceeded"}:
        return "degraded"
    pay = _payments_status()
    if pay == "down":
        return "down"
    if pay == "degraded":
        return "degraded"
    return "healthy"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "service": "notifications",
        "status": _status(),
        "failure_mode": ACTIVE_FAILURE,
        "queue_depth": QUEUE_DEPTH,
        "depends_on": {"payments": _payments_status()},
        "uptime_s": int(time.time() - STARTED),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    status = _status()
    health_value = {"healthy": 1.0, "degraded": 0.5, "down": 0.0}[status]
    error_rate = 0.01 if status == "healthy" else (0.4 if status == "degraded" else 0.99)
    latency = 0.03 if status == "healthy" else (0.9 if status == "degraded" else 2.8)
    return (
        f'service_health{{service="notifications"}} {health_value}\n'
        f'service_error_rate{{service="notifications"}} {error_rate}\n'
        f'service_latency_seconds{{service="notifications",quantile="0.99"}} {latency}\n'
        f'queue_depth{{service="notifications"}} {QUEUE_DEPTH}\n'
    )


@app.get("/logs")
def logs(lines: int = 50) -> dict[str, Any]:
    size = max(1, min(lines, 500))
    return {"service": "notifications", "entries": list(LOGS)[-size:]}


@app.post("/chaos/inject")
def inject(req: ChaosInjectRequest) -> dict[str, Any]:
    global ACTIVE_FAILURE, QUEUE_DEPTH
    ACTIVE_FAILURE = req.failure_mode
    if req.failure_mode == "queue_overflow":
        QUEUE_DEPTH = 100_000
    _log("WARN", "failure injected", failure_mode=req.failure_mode, severity=req.severity)
    return {"status": "ok", "service": "notifications", "failure_mode": ACTIVE_FAILURE}


@app.post("/chaos/clear")
def clear() -> dict[str, Any]:
    global ACTIVE_FAILURE, QUEUE_DEPTH
    ACTIVE_FAILURE = None
    QUEUE_DEPTH = 0
    _log("INFO", "failure cleared")
    return {"status": "ok", "service": "notifications", "failure_mode": ACTIVE_FAILURE}


@app.post("/send")
def send(req: SendRequest) -> dict[str, Any]:
    global QUEUE_DEPTH
    status = _status()
    if status == "down":
        _log("ERROR", "send failed", reason="service down", customer_id=req.customer_id)
        return {"ok": False, "error": "notifications_down"}
    QUEUE_DEPTH = max(0, QUEUE_DEPTH - 1)
    _log("INFO", "notification sent", customer_id=req.customer_id, size=len(req.message))
    return {"ok": True, "queued": status == "degraded"}

