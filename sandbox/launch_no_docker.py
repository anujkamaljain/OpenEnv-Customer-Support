"""Launch the sandbox cluster without Docker.

Useful for environments where Docker is unavailable (e.g. Hugging Face Spaces).
Each service runs as an in-process uvicorn server in a background thread, all
bound to localhost on the same ports the SandboxBridge expects (5001-5005,
6660 for the chaos controller).

Usage:
    python -m sandbox.launch_no_docker

The script blocks until interrupted with Ctrl+C. Run sandbox evaluations from a
separate shell while this process is alive.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import uvicorn

SANDBOX_DIR = Path(__file__).resolve().parent

SERVICES: list[tuple[str, int, Path]] = [
    ("auth", 5001, SANDBOX_DIR / "services" / "auth" / "app.py"),
    ("database", 5002, SANDBOX_DIR / "services" / "database" / "app.py"),
    ("payments", 5003, SANDBOX_DIR / "services" / "payments" / "app.py"),
    ("analytics", 5004, SANDBOX_DIR / "services" / "analytics" / "app.py"),
    ("notifications", 5005, SANDBOX_DIR / "services" / "notifications" / "app.py"),
]
CHAOS_NAME = "chaos-controller"
CHAOS_PORT = 6660
CHAOS_PATH = SANDBOX_DIR / "chaos" / "controller.py"


def _load_app(name: str, app_file: Path):
    """Load the FastAPI ``app`` object from a service file by absolute path."""
    spec = importlib.util.spec_from_file_location(f"sandbox_app_{name}", str(app_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load app for service '{name}' at {app_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The chaos controller imports `failure_modes` as a sibling. Add its parent
    # to sys.path so that import works without Docker's WORKDIR semantics.
    parent_str = str(app_file.parent)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
    spec.loader.exec_module(module)
    if not hasattr(module, "app"):
        raise RuntimeError(f"Service '{name}' module has no `app` attribute")
    return module.app


def _serve(name: str, app, port: int) -> Callable[[], None]:
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _runner() -> None:
        # uvicorn manages its own loop; signal handlers are no-ops in threads.
        server.config.setup_event_loop()
        server.run()

    thread = threading.Thread(target=_runner, name=f"sandbox-{name}", daemon=True)
    thread.start()
    return server.should_exit.set if hasattr(server, "should_exit") else (lambda: None)


def main() -> None:
    # Match the chaos-controller's expected default service URLs (localhost).
    os.environ.setdefault(
        "SERVICE_ENDPOINTS_JSON",
        '{"auth":"http://127.0.0.1:5001",'
        '"database":"http://127.0.0.1:5002",'
        '"payments":"http://127.0.0.1:5003",'
        '"analytics":"http://127.0.0.1:5004",'
        '"notifications":"http://127.0.0.1:5005"}',
    )

    print("[sandbox-no-docker] starting services in-process...")
    stop_callbacks: list[Callable[[], None]] = []

    for name, port, app_file in SERVICES:
        app = _load_app(name, app_file)
        stop_callbacks.append(_serve(name, app, port))
        print(f"  - {name} -> http://127.0.0.1:{port}")

    chaos_app = _load_app(CHAOS_NAME, CHAOS_PATH)
    stop_callbacks.append(_serve(CHAOS_NAME, chaos_app, CHAOS_PORT))
    print(f"  - {CHAOS_NAME} -> http://127.0.0.1:{CHAOS_PORT}")

    # Give servers a moment to bind before declaring readiness.
    time.sleep(1.5)
    print("[sandbox-no-docker] all services listening (Ctrl+C to stop).")

    stop_event = threading.Event()

    def _handle_signal(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (AttributeError, ValueError):
        pass

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    finally:
        print("[sandbox-no-docker] shutting down...")
        for cb in stop_callbacks:
            try:
                cb()
            except Exception:
                pass


if __name__ == "__main__":
    main()
