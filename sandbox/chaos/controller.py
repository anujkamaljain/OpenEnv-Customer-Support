"""Chaos controller for sandbox services.

This service orchestrates failure injection and fix verification by calling
per-service chaos hooks over HTTP.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from failure_modes import ALLOWED_FIX_TYPES, FAILURE_SPECS

app = FastAPI(title="EICC Sandbox Chaos Controller", version="0.1.0")

_DEFAULT_ENDPOINTS = {
    "auth": "http://auth:5001",
    "database": "http://database:5002",
    "payments": "http://payments:5003",
    "analytics": "http://analytics:5004",
    "notifications": "http://notifications:5005",
}

_SERVICE_ENDPOINTS = _DEFAULT_ENDPOINTS.copy()
if os.environ.get("SERVICE_ENDPOINTS_JSON"):
    try:
        loaded = json.loads(os.environ["SERVICE_ENDPOINTS_JSON"])
        if isinstance(loaded, dict):
            _SERVICE_ENDPOINTS.update({str(k): str(v) for k, v in loaded.items()})
    except json.JSONDecodeError:
        pass

_ACTIVE_FAILURES: dict[str, dict[str, Any]] = {}


class InjectRequest(BaseModel):
    service: str
    failure_mode: str
    severity: str = Field(default="high")


class ServiceRequest(BaseModel):
    service: str


class VerifyFixRequest(BaseModel):
    service: str
    fix_type: str


def _endpoint(service: str) -> str:
    url = _SERVICE_ENDPOINTS.get(service)
    if url is None:
        raise HTTPException(status_code=404, detail=f"Unknown service '{service}'")
    return url.rstrip("/")


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return _read_json(req)


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET")
    return _read_json(req)


def _read_json(req: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=2.0) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Upstream HTTP {exc.code}: {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream unavailable: {exc.reason}") from exc

    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from upstream: {payload[:120]}") from exc
    if not isinstance(obj, dict):
        raise HTTPException(status_code=502, detail="Upstream payload must be a JSON object")
    return obj


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chaos/inject")
def inject_failure(req: InjectRequest) -> dict[str, Any]:
    if req.failure_mode not in FAILURE_SPECS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported failure_mode '{req.failure_mode}'.",
        )
    expected_service = FAILURE_SPECS[req.failure_mode].service
    if req.service != expected_service:
        raise HTTPException(
            status_code=400,
            detail=(
                f"failure_mode '{req.failure_mode}' is mapped to service "
                f"'{expected_service}', got '{req.service}'."
            ),
        )
    service_url = _endpoint(req.service)
    response = _post_json(
        f"{service_url}/chaos/inject",
        {"failure_mode": req.failure_mode, "severity": req.severity},
    )
    _ACTIVE_FAILURES[req.service] = {"mode": req.failure_mode, "severity": req.severity}
    return {"status": "injected", "service": req.service, "result": response}


@app.post("/chaos/clear")
def clear_service(req: ServiceRequest) -> dict[str, Any]:
    service_url = _endpoint(req.service)
    response = _post_json(f"{service_url}/chaos/clear", {})
    _ACTIVE_FAILURES.pop(req.service, None)
    return {"status": "cleared", "service": req.service, "result": response}


@app.post("/chaos/clear_all")
def clear_all() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for service, service_url in _SERVICE_ENDPOINTS.items():
        try:
            results[service] = _post_json(f"{service_url.rstrip('/')}/chaos/clear", {})
        except HTTPException as exc:
            results[service] = {"error": exc.detail}
    _ACTIVE_FAILURES.clear()
    return {"status": "cleared_all", "services": results}


@app.get("/chaos/status")
def status() -> dict[str, Any]:
    return {
        "active_failures": [
            {"service": service, "mode": data["mode"], "severity": data["severity"]}
            for service, data in sorted(_ACTIVE_FAILURES.items())
        ],
        "supported_failures": sorted(FAILURE_SPECS.keys()),
        "supported_fixes": sorted(ALLOWED_FIX_TYPES),
    }


@app.post("/chaos/verify_fix")
def verify_fix(req: VerifyFixRequest) -> dict[str, Any]:
    service_url = _endpoint(req.service)
    if req.fix_type not in ALLOWED_FIX_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported fix_type '{req.fix_type}'. "
                f"Supported: {sorted(ALLOWED_FIX_TYPES)}"
            ),
        )
    # Keep fix handling deterministic and safe: any supported fix clears
    # the active failure on the target service.
    _post_json(f"{service_url}/chaos/clear", {})
    _ACTIVE_FAILURES.pop(req.service, None)
    health = _get_json(f"{service_url}/health")
    return {
        "service": req.service,
        "fix_type": req.fix_type,
        "fixed": health.get("status") == "healthy",
        "health": health,
    }

