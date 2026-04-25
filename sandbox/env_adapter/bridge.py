"""HTTP bridge utilities for the VM sandbox cluster.

The bridge intentionally uses only stdlib networking so the core environment
does not gain new runtime dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class SandboxBridgeError(RuntimeError):
    """Raised when sandbox cluster communication fails."""


class SandboxValidationError(SandboxBridgeError):
    """Raised when the sandbox cluster rejects a request as invalid (4xx).

    These errors should NOT disable sandbox mode for the rest of the episode -
    they represent recoverable input-quality issues from the agent.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class SandboxConnectionError(SandboxBridgeError):
    """Raised when the sandbox cluster is unreachable or returns 5xx.

    These errors indicate infrastructure issues and should disable sandbox mode
    for the rest of the episode (graceful fallback to simulated behavior).
    """


@dataclass(slots=True, frozen=True)
class ServiceEndpoint:
    """A named service endpoint in the sandbox cluster."""

    name: str
    url: str


class SandboxBridge:
    """Small synchronous HTTP client for sandbox service APIs."""

    def __init__(
        self,
        *,
        services: dict[str, str],
        chaos_url: str,
        timeout_s: float = 2.0,
    ) -> None:
        self._services = {name: ServiceEndpoint(name=name, url=url.rstrip("/")) for name, url in services.items()}
        self._chaos_url = chaos_url.rstrip("/")
        self._timeout_s = timeout_s

    def service_url(self, service_name: str) -> str:
        endpoint = self._services.get(service_name)
        if endpoint is None:
            raise SandboxBridgeError(f"Unknown service '{service_name}'")
        return endpoint.url

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, method="GET")
        return self._read_json(req)

    def post_json(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload or {}, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(
            url,
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        return self._read_json(req)

    def check_health(self, service_name: str) -> dict[str, Any]:
        return self.get_json(f"{self.service_url(service_name)}/health")

    def fetch_logs(self, service_name: str, *, lines: int = 50) -> dict[str, Any]:
        return self.get_json(
            f"{self.service_url(service_name)}/logs",
            params={"lines": str(max(1, min(lines, 500)))},
        )

    def fetch_metrics_text(self, service_name: str) -> str:
        req = urllib.request.Request(f"{self.service_url(service_name)}/metrics", method="GET")
        return self._read_text(req)

    def chaos_inject(self, service_name: str, failure_mode: str) -> dict[str, Any]:
        return self.post_json(
            f"{self._chaos_url}/chaos/inject",
            {"service": service_name, "failure_mode": failure_mode},
        )

    def chaos_clear(self, service_name: str) -> dict[str, Any]:
        return self.post_json(f"{self._chaos_url}/chaos/clear", {"service": service_name})

    def chaos_clear_all(self) -> dict[str, Any]:
        return self.post_json(f"{self._chaos_url}/chaos/clear_all")

    def chaos_status(self) -> dict[str, Any]:
        return self.get_json(f"{self._chaos_url}/chaos/status")

    def chaos_verify_fix(self, service_name: str, fix_type: str) -> dict[str, Any]:
        return self.post_json(
            f"{self._chaos_url}/chaos/verify_fix",
            {"service": service_name, "fix_type": fix_type},
        )

    def _read_json(self, request: urllib.request.Request) -> dict[str, Any]:
        raw = self._read_text(request)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SandboxBridgeError(
                f"Non-JSON response from {request.full_url}: {raw[:200]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise SandboxBridgeError(
                f"Expected object from {request.full_url}, got {type(parsed).__name__}"
            )
        return parsed

    def _read_text(self, request: urllib.request.Request) -> str:
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                payload = response.read().decode("utf-8", errors="replace")
                return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if 400 <= exc.code < 500:
                raise SandboxValidationError(exc.code, detail[:300]) from exc
            raise SandboxConnectionError(
                f"HTTP {exc.code} from {request.full_url}: {detail[:300]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SandboxConnectionError(
                f"Failed request to {request.full_url}: {exc.reason}"
            ) from exc

