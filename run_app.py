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
import socket
import signal
import subprocess
import sys
import time
import webbrowser
import ctypes
from urllib import error, request
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))
BACKEND_RELOAD = os.getenv("BACKEND_RELOAD", "0") == "1"

# Seconds to wait before opening the browser so both servers have time to start
BROWSER_OPEN_DELAY: int = int(os.getenv("BROWSER_OPEN_DELAY", "3"))
# Seconds between process health-checks in the main loop
HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "5"))
# Seconds to wait for a process to exit gracefully before sending SIGKILL
SHUTDOWN_TIMEOUT: int = int(os.getenv("SHUTDOWN_TIMEOUT", "15"))

PR_SET_PDEATHSIG = 1


def check_frontend_deps() -> bool:
    """Return True if node_modules already installed."""
    return (FRONTEND_DIR / "node_modules").exists()


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if a TCP port is already bound on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def get_port_owners(port: int) -> list[str]:
    """Return process lines from lsof for processes listening on a port."""
    try:
        output = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
        )
    except Exception:
        return []

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    return lines[1:]


def url_responds(url: str, timeout: float = 2.0) -> bool:
    """Return True when the given HTTP endpoint responds successfully."""
    try:
        with request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except (error.URLError, TimeoutError, OSError, ValueError):
        return False


def existing_stack_is_healthy() -> bool:
    """Return True when both the backend and frontend are already serving."""
    backend_url = f"http://127.0.0.1:{BACKEND_PORT}/api/health"
    frontend_url = f"http://127.0.0.1:{FRONTEND_PORT}"
    return url_responds(backend_url) and url_responds(frontend_url)


def _set_parent_death_signal() -> None:
    """Ask Linux to send SIGTERM to this child if the launcher dies."""
    if not sys.platform.startswith("linux"):
        return

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _child_setup() -> None:
    """Start subprocesses in a fresh session and tie them to the parent lifetime."""
    os.setsid()
    _set_parent_death_signal()


def run() -> None:
    procs: list[subprocess.Popen] = []
    frontend_url = f"http://localhost:{FRONTEND_PORT}"

    # Fail fast with a helpful message if either required port is occupied.
    occupied_ports = [
        (name, port)
        for name, port in (("backend", BACKEND_PORT), ("frontend", FRONTEND_PORT))
        if is_port_in_use(port)
    ]
    if occupied_ports:
        if len(occupied_ports) == 2 and existing_stack_is_healthy():
            print("[run_app] Backend and frontend are already running.")
            print(f"[run_app] Visit {frontend_url}")
            try:
                webbrowser.open(frontend_url)
            except Exception:
                pass
            return

        for name, port in occupied_ports:
            print(f"[run_app] Cannot start {name}: port {port} is already in use.")
            owners = get_port_owners(port)
            if owners:
                print("[run_app] Listening processes:")
                for owner in owners:
                    print(f"[run_app]   {owner}")
        print("[run_app] Stop the process above or choose different ports via env vars.")
        sys.exit(1)

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
    print(f"[run_app] Starting backend on port {BACKEND_PORT}…")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(BACKEND_PORT),
    ]
    if BACKEND_RELOAD:
        backend_cmd.append("--reload")
    if os.name == "posix":
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=BACKEND_DIR,
            preexec_fn=_child_setup,
        )
    else:
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=BACKEND_DIR,
            start_new_session=True,
        )
    procs.append(backend_proc)

    # Give backend a moment to fail fast (e.g., config/runtime issues) before
    # starting frontend so we avoid a half-started system.
    time.sleep(1)
    if backend_proc.poll() is not None:
        print("[run_app] Backend exited during startup. See backend logs above.")
        sys.exit(1)

    # ── Frontend ─────────────────────────────────────────────────────────────
    if not check_frontend_deps():
        print("[run_app] Installing frontend dependencies (first run)…")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)

    print(f"[run_app] Starting frontend on port {FRONTEND_PORT}…")
    frontend_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "0.0.0.0",
        "--port",
        str(FRONTEND_PORT),
    ]
    if os.name == "posix":
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=FRONTEND_DIR,
            preexec_fn=_child_setup,
        )
    else:
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=FRONTEND_DIR,
            start_new_session=True,
        )
    procs.append(frontend_proc)

    # ── Open browser after a short delay ─────────────────────────────────────
    time.sleep(BROWSER_OPEN_DELAY)
    print(f"[run_app] Opening {frontend_url}")
    try:
        webbrowser.open(frontend_url)
    except Exception:
        print(f"[run_app] Could not open browser automatically. Visit {frontend_url}")

    print("[run_app] System running. Press Ctrl+C to stop.")
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
