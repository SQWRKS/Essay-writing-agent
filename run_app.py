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
import subprocess
import sys
import time
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


def check_frontend_deps() -> bool:
    """Return True if node_modules already installed."""
    return (FRONTEND_DIR / "node_modules").exists()


def run() -> None:
    procs: list[subprocess.Popen] = []

    def _cleanup(sig=None, frame=None) -> None:  # noqa: ANN001
        print("\n[run_app] Shutting down…")
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # ── Backend ──────────────────────────────────────────────────────────────
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

    print(f"[run_app] Starting frontend on port {FRONTEND_PORT}…")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        start_new_session=True,
    )
    procs.append(frontend_proc)

    # ── Open browser after a short delay ─────────────────────────────────────
    time.sleep(BROWSER_OPEN_DELAY)
    url = f"http://localhost:{FRONTEND_PORT}"
    print(f"[run_app] Opening {url}")
    try:
        webbrowser.open(url)
    except Exception:
        print(f"[run_app] Could not open browser automatically. Visit {url}")

    print("[run_app] System running. Press Ctrl+C to stop.")
    try:
        while True:
            # Restart either process if it dies unexpectedly
            for p in list(procs):
                if p.poll() is not None:
                    print(f"[run_app] Process {p.pid} exited unexpectedly.")
            time.sleep(HEALTH_CHECK_INTERVAL)
    except KeyboardInterrupt:
        _cleanup()


if __name__ == "__main__":
    run()
