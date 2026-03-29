#!/usr/bin/env python3
"""
Single-command startup script for the Essay Writing Agent system.

Usage:
    python run_app.py

This script:
 - Starts the FastAPI backend on port 8000
 - Starts the React/Vite frontend on port 3000
 - Opens the browser at http://localhost:3000
"""

import os
import signal
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))

# Seconds to wait before opening the browser so both servers have time to start
BROWSER_OPEN_DELAY: int = int(os.getenv("BROWSER_OPEN_DELAY", "3"))
# Seconds between process health-checks in the main loop
HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "5"))
# Seconds to wait for backend/frontend to become reachable during startup
STARTUP_TIMEOUT: int = int(os.getenv("STARTUP_TIMEOUT", "60"))
# Seconds to wait for a process to exit gracefully before sending SIGKILL
SHUTDOWN_TIMEOUT: int = int(os.getenv("SHUTDOWN_TIMEOUT", "15"))


def check_frontend_deps() -> bool:
    """Return True if node_modules already installed."""
    return (FRONTEND_DIR / "node_modules").exists()


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_http(url: str, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            time.sleep(0.5)
    return False


def open_browser(url: str) -> None:
    browser_cmd = os.getenv("BROWSER", "").strip()
    if browser_cmd:
        try:
            subprocess.Popen([*shlex.split(browser_cmd), url])
            return
        except Exception:
            pass

    try:
        webbrowser.open(url)
    except Exception:
        print(f"[run_app] Could not open browser automatically. Visit {url}")


def run() -> None:
    procs: list[subprocess.Popen] = []
    backend_url = f"http://127.0.0.1:{BACKEND_PORT}/api/health"
    frontend_url = f"http://127.0.0.1:{FRONTEND_PORT}"

    def _cleanup(sig=None, frame=None) -> None:  # noqa: ANN001
        print("\n[run_app] Shutting down…")
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                print(f"[run_app] Process {p.pid} did not stop in time; killing.")
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # ── Backend ──────────────────────────────────────────────────────────────
    if is_port_open("127.0.0.1", BACKEND_PORT):
        if wait_for_http(backend_url, 3):
            print(f"[run_app] Reusing existing backend on port {BACKEND_PORT}.")
        else:
            print(f"[run_app] Port {BACKEND_PORT} is already in use by another process.")
            print("[run_app] Stop that process or change BACKEND_PORT before retrying.")
            sys.exit(1)
    else:
        print(f"[run_app] Starting backend on port {BACKEND_PORT}…")
        backend_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(BACKEND_PORT),
                "--reload",
            ],
            cwd=BACKEND_DIR,
            start_new_session=True,
        )
        procs.append(backend_proc)

    # ── Frontend ─────────────────────────────────────────────────────────────
    if not check_frontend_deps():
        print("[run_app] Installing frontend dependencies (first run)…")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)

    if is_port_open("127.0.0.1", FRONTEND_PORT):
        if wait_for_http(frontend_url, 3):
            print(f"[run_app] Reusing existing frontend on port {FRONTEND_PORT}.")
        else:
            print(f"[run_app] Port {FRONTEND_PORT} is already in use by another process.")
            print("[run_app] Stop that process or change FRONTEND_PORT before retrying.")
            sys.exit(1)
    else:
        print(f"[run_app] Starting frontend on port {FRONTEND_PORT}…")
        frontend_proc = subprocess.Popen(
            [
                "npm",
                "run",
                "dev",
                "--",
                "--host",
                "0.0.0.0",
                "--port",
                str(FRONTEND_PORT),
            ],
            cwd=FRONTEND_DIR,
            start_new_session=True,
        )
        procs.append(frontend_proc)

    url = f"http://localhost:{FRONTEND_PORT}"
    backend_ready = wait_for_http(backend_url, STARTUP_TIMEOUT)
    frontend_ready = wait_for_http(frontend_url, STARTUP_TIMEOUT)

    if not backend_ready or not frontend_ready:
        print("[run_app] Startup timed out before the app became reachable. Shutting down.")
        _cleanup()

    time.sleep(BROWSER_OPEN_DELAY)
    print(f"[run_app] Opening {url}")
    open_browser(url)

    print("[run_app] System running. Keep this terminal open while using the app.")
    print("[run_app] Press Ctrl+C only when you want to stop both backend and frontend.")
    try:
        while True:
            # Shut down cleanly if either process dies unexpectedly
            for p in list(procs):
                if p.poll() is not None:
                    print(f"[run_app] Process {p.pid} exited unexpectedly. Shutting down.")
                    _cleanup()  # raises SystemExit; not re-entered
            time.sleep(HEALTH_CHECK_INTERVAL)
    except KeyboardInterrupt:
        _cleanup()


if __name__ == "__main__":
    run()
