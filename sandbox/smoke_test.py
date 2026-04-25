"""Local smoke test for sandbox cluster + OpenEnv sandbox mode.

Usage:
  python sandbox/smoke_test.py --base-url http://localhost
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET", headers=headers or {})
    with urllib.request.urlopen(req, timeout=5.0) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=8.0) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _assert_status(name: str, payload: dict[str, Any]) -> None:
    status = payload.get("status")
    if status not in {"healthy", "degraded", "down", "ok"}:
        raise RuntimeError(f"{name}: unexpected status payload: {payload}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sandbox smoke test")
    parser.add_argument("--base-url", default="http://localhost", help="Base URL for services")
    parser.add_argument("--api-url", default="http://localhost:7860", help="OpenEnv API base URL")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    services = {
        "auth": f"{base}:5001",
        "database": f"{base}:5002",
        "payments": f"{base}:5003",
        "analytics": f"{base}:5004",
        "notifications": f"{base}:5005",
    }
    chaos_url = f"{base}:6660"

    print("[1/5] Checking service health endpoints...")
    for name, url in services.items():
        health = _get_json(f"{url}/health")
        _assert_status(name, health)
        print(f"  - {name}: {health.get('status')}")

    print("[2/5] Injecting + verifying chaos action...")
    inject = _post_json(
        f"{chaos_url}/chaos/inject",
        {"service": "auth", "failure_mode": "rate_limiting", "severity": "high"},
    )
    print(f"  - inject: {inject.get('status')}")
    verify = _post_json(
        f"{chaos_url}/chaos/verify_fix",
        {"service": "auth", "fix_type": "restart_service"},
    )
    print(f"  - verify fixed: {verify.get('fixed')}")

    print("[3/5] Resetting OpenEnv incident episode in sandbox mode...")
    session_headers = {"X-Session-ID": "smoke-test-session"}
    reset = _post_json(
        f"{args.api_url.rstrip('/')}/reset",
        {"mode": "incident", "difficulty": "easy", "seed": 0},
        headers=session_headers,
    )
    if "observation" not in reset:
        raise RuntimeError(f"Unexpected /reset response: {reset}")
    print("  - reset ok")

    print("[4/5] Running monitoring + log actions...")
    step1 = _post_json(
        f"{args.api_url.rstrip('/')}/step",
        {"action": {"action_type": "check_monitoring"}},
        headers=session_headers,
    )
    step2 = _post_json(
        f"{args.api_url.rstrip('/')}/step",
        {"action": {"action_type": "fetch_logs", "service_name": "auth", "time_range": "last_5m"}},
        headers=session_headers,
    )
    if "sandbox" not in (step1.get("info") or {}):
        raise RuntimeError("Expected sandbox info in step response.")
    if "sandbox" not in (step2.get("info") or {}):
        raise RuntimeError("Expected sandbox info in step response.")
    print("  - step actions ok")

    print("[5/5] Closing session...")
    _post_json(f"{args.api_url.rstrip('/')}/close", {}, headers=session_headers)
    print("Sandbox smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error: {exc}") from exc

