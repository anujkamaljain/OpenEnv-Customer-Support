"""Payments sandbox service."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

app = FastAPI(title="sandbox-payments", version="0.1.0")

AUTH_URL = os.environ.get("AUTH_URL", "http://auth:5001").rstrip("/")
DB_URL = os.environ.get("DB_URL", "http://database:5002").rstrip("/")
STARTED = time.time()
ACTIVE_FAILURE: str | None = None
LOGS: deque[str] = deque(maxlen=500)


class ChaosInjectRequest(BaseModel):
    failure_mode: str
    severity: str = "high"


class PayRequest(BaseModel):
    transaction_id: str
    amount: float


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(level: str, message: str, **extra: Any) -> None:
    payload = {"ts": _now(), "service": "payments", "level": level, "message": message, **extra}
    LOGS.append(str(payload))


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=1.5) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1.5) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _upstream_status(service_url: str) -> str:
    try:
        health = _get_json(f"{service_url}/health")
    except Exception:
        return "down"
    status = str(health.get("status", "down"))
    return status if status in {"healthy", "degraded", "down"} else "down"


def _status() -> str:
    if ACTIVE_FAILURE in {"gateway_timeout", "validation_errors"}:
        return "down"
    if ACTIVE_FAILURE in {"idempotency_failure", "rate_exceeded"}:
        return "degraded"
    auth_status = _upstream_status(AUTH_URL)
    db_status = _upstream_status(DB_URL)
    if auth_status == "down" or db_status == "down":
        return "down"
    if auth_status == "degraded" or db_status == "degraded":
        return "degraded"
    return "healthy"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "service": "payments",
        "status": _status(),
        "failure_mode": ACTIVE_FAILURE,
        "depends_on": {"auth": _upstream_status(AUTH_URL), "database": _upstream_status(DB_URL)},
        "uptime_s": int(time.time() - STARTED),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    status = _status()
    health_value = {"healthy": 1.0, "degraded": 0.5, "down": 0.0}[status]
    error_rate = 0.02 if status == "healthy" else (0.35 if status == "degraded" else 0.99)
    latency = 0.06 if status == "healthy" else (0.8 if status == "degraded" else 4.0)
    return (
        f'service_health{{service="payments"}} {health_value}\n'
        f'service_error_rate{{service="payments"}} {error_rate}\n'
        f'service_latency_seconds{{service="payments",quantile="0.99"}} {latency}\n'
    )


@app.get("/logs")
def logs(lines: int = 50) -> dict[str, Any]:
    size = max(1, min(lines, 500))
    return {"service": "payments", "entries": list(LOGS)[-size:]}


@app.post("/chaos/inject")
def inject(req: ChaosInjectRequest) -> dict[str, Any]:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = req.failure_mode
    _log("WARN", "failure injected", failure_mode=req.failure_mode, severity=req.severity)
    return {"status": "ok", "service": "payments", "failure_mode": ACTIVE_FAILURE}


@app.post("/chaos/clear")
def clear() -> dict[str, Any]:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = None
    _log("INFO", "failure cleared")
    return {"status": "ok", "service": "payments", "failure_mode": ACTIVE_FAILURE}


@app.post("/pay")
def pay(req: PayRequest) -> dict[str, Any]:
    status = _status()
    if status == "down":
        _log("ERROR", "payment failed", reason="service unavailable", txn=req.transaction_id)
        return {"ok": False, "error": "service_unavailable"}
    if ACTIVE_FAILURE == "validation_errors":
        _log("ERROR", "payment validation failed", txn=req.transaction_id)
        return {"ok": False, "error": "validation_error"}
    try:
        auth_valid = _get_json(f"{AUTH_URL}/validate")
    except urllib.error.URLError:
        return {"ok": False, "error": "auth_unreachable"}
    if not auth_valid.get("ok"):
        return {"ok": False, "error": "auth_rejected"}
    _post_json(f"{DB_URL}/query", {"key": req.transaction_id})
    _log("INFO", "payment accepted", txn=req.transaction_id, amount=req.amount)
    return {"ok": True, "transaction_id": req.transaction_id}

