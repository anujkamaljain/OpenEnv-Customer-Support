"""Database sandbox service."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

app = FastAPI(title="sandbox-database", version="0.1.0")

STARTED = time.time()
ACTIVE_FAILURE: str | None = None
LOGS: deque[str] = deque(maxlen=500)
DATA: dict[str, dict[str, Any]] = {"txn-001": {"status": "ok", "amount": 42.0}}


class ChaosInjectRequest(BaseModel):
    failure_mode: str
    severity: str = "high"


class QueryRequest(BaseModel):
    key: str


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(level: str, message: str, **extra: Any) -> None:
    payload = {"ts": _now(), "service": "database", "level": level, "message": message, **extra}
    LOGS.append(str(payload))


def _status() -> str:
    if ACTIVE_FAILURE in {"oom", "query_timeout"}:
        return "down"
    if ACTIVE_FAILURE in {"connection_pool_exhaustion", "replication_lag"}:
        return "degraded"
    return "healthy"


@app.get("/health")
def health() -> dict[str, Any]:
    return {"service": "database", "status": _status(), "failure_mode": ACTIVE_FAILURE, "uptime_s": int(time.time() - STARTED)}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    status = _status()
    health_value = {"healthy": 1.0, "degraded": 0.5, "down": 0.0}[status]
    error_rate = 0.01 if status == "healthy" else (0.3 if status == "degraded" else 0.98)
    latency = 0.03 if status == "healthy" else (0.45 if status == "degraded" else 3.5)
    return (
        f'service_health{{service="database"}} {health_value}\n'
        f'service_error_rate{{service="database"}} {error_rate}\n'
        f'service_latency_seconds{{service="database",quantile="0.99"}} {latency}\n'
    )


@app.get("/logs")
def logs(lines: int = 50) -> dict[str, Any]:
    size = max(1, min(lines, 500))
    return {"service": "database", "entries": list(LOGS)[-size:]}


@app.post("/chaos/inject")
def inject(req: ChaosInjectRequest) -> dict[str, Any]:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = req.failure_mode
    _log("WARN", "failure injected", failure_mode=req.failure_mode, severity=req.severity)
    return {"status": "ok", "service": "database", "failure_mode": ACTIVE_FAILURE}


@app.post("/chaos/clear")
def clear() -> dict[str, Any]:
    global ACTIVE_FAILURE
    ACTIVE_FAILURE = None
    _log("INFO", "failure cleared")
    return {"status": "ok", "service": "database", "failure_mode": ACTIVE_FAILURE}


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    if _status() == "down":
        _log("ERROR", "query failed", reason="database down")
        return {"ok": False, "error": "database_down"}
    if ACTIVE_FAILURE == "replication_lag":
        _log("WARN", "serving stale record", key=req.key)
        return {"ok": True, "stale": True, "record": DATA.get(req.key)}
    return {"ok": True, "record": DATA.get(req.key)}

