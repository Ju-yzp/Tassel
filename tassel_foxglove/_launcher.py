"""Foxglove Studio launcher — bridge, HTTP server, deep-link orchestration."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import yaml

from ._config import generate_layout
from ._env import bridge_defaults, foxglove_binary, load_env


def _kill_existing_bridge(port: int) -> None:
    """Kill any foxglove_bridge process or port occupant."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "foxglove_bridge"],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            for pid in result.stdout.strip().split():
                print(f"  Killing old foxglove_bridge (PID {pid}) …")
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.5)
    except FileNotFoundError:
        pass

    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """Poll until a TCP server is listening on host:port."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def _find_free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _QuietHTTPHandler(SimpleHTTPRequestHandler):
    """Serves static files without logging every request."""

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        pass


def _start_layout_server(layout_path: Path, project_dir: Path) -> tuple[HTTPServer, int, str]:
    """Start a throwaway HTTP server to serve the layout file.

    Returns (server, port, url_path).
    """
    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port),
                        lambda *a, **kw: _QuietHTTPHandler(*a, directory=str(project_dir), **kw))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if not _wait_for_port("127.0.0.1", port, timeout=3.0):
        print(f"  Warning: layout server failed to start on port {port}", file=sys.stderr)

    try:
        rel = layout_path.resolve().relative_to(project_dir.resolve())
    except ValueError:
        rel = layout_path.resolve().relative_to(Path.cwd())
    url = str(rel)

    print(f"  Layout server on http://127.0.0.1:{port}/{url}")
    return server, port, url


def launch(
    config_path: str | Path,
    *,
    port: int | None = None,
    env_path: str | Path | None = None,
    no_studio: bool = False,
    no_bridge: bool = False,
    project_dir: str | Path | None = None,
) -> None:
    """One-shot launch of the full Foxglove visualization stack.

    Parameters
    ----------
    config_path : Path to a YAML panel config (panels + optional bridge section).
    port        : Override bridge WebSocket port.
    env_path    : Override environment config file.
    no_studio   : Skip launching Foxglove Studio.
    no_bridge   : Skip starting foxglove_bridge.
    project_dir : Project root (for HTTP server document root).  Defaults to
                  the parent of the tassel_foxglove package.
    """
    load_env(env_path)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    if project_dir is None:
        project_dir = Path(__file__).resolve().parent.parent
    else:
        project_dir = Path(project_dir)

    # ── load YAML ──────────────────────────────────────────────────────────
    with open(config_path) as f:
        root = yaml.safe_load(f)

    panels = root.get("panels", [])
    if not panels:
        print(f"'{config_path}' contains no 'panels' list", file=sys.stderr)
        sys.exit(1)

    bridge_cfg = {**bridge_defaults(), **root.get("bridge", {})}
    bridge_port = port or bridge_cfg.get("port", 8765)
    grid_cols = root.get("grid_cols", load_env().get("layout", {}).get("default_grid_cols", 2))

    # ── generate layout ────────────────────────────────────────────────────
    layout = generate_layout(panels, grid_cols)
    layout_name = config_path.stem + ".foxglove.json"
    layout_path = config_path.with_suffix(".foxglove.json")
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"  Layout → {layout_path}  ({len(panels)} panels, {grid_cols} cols)")

    # ── layout HTTP server ─────────────────────────────────────────────────
    layout_server, layout_port, layout_rel = None, None, None
    if not no_studio:
        layout_server, layout_port, layout_rel = _start_layout_server(layout_path, project_dir)

    # ── foxglove_bridge ────────────────────────────────────────────────────
    bridge_proc = None
    if not no_bridge:
        _kill_existing_bridge(bridge_port)

        ros_args = ["ros2", "run", "foxglove_bridge", "foxglove_bridge",
                    "--ros-args", "-p", f"port:={bridge_port}",
                    "-p", f"address:={bridge_cfg.get('address', '0.0.0.0')}"]
        if bridge_cfg.get("topic_whitelist"):
            ros_args.extend(["-p", f"topic_whitelist:=[{','.join(bridge_cfg['topic_whitelist'])}]"])
        if bridge_cfg.get("tls"):
            ros_args.extend(["-p", "tls:=true"])
        if bridge_cfg.get("use_compression"):
            ros_args.extend(["-p", "use_compression:=true"])

        print(f"  Starting foxglove_bridge on ws://localhost:{bridge_port} …")
        try:
            bridge_proc = subprocess.Popen(
                ros_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("  ros2 command not found — is the ROS 2 environment sourced?",
                  file=sys.stderr)
            sys.exit(1)

        if not _wait_for_port("127.0.0.1", bridge_port, timeout=10.0):
            print("  foxglove_bridge failed to start within 10 s", file=sys.stderr)
            bridge_proc.terminate()
            sys.exit(1)

        print(f"  foxglove_bridge listening on ws://localhost:{bridge_port}")

    # ── foxglove-studio ────────────────────────────────────────────────────
    if not no_studio:
        foxglove = foxglove_binary()
        if foxglove is None:
            print("  foxglove-studio not found", file=sys.stderr)
            sys.exit(1)

        deep_link = (
            f"foxglove://open"
            f"?ds=foxglove-websocket&ds.url=ws://localhost:{bridge_port}"
            f"&layoutUrl=http://127.0.0.1:{layout_port}/{layout_rel}"
        )
        print(f"  Opening Foxglove Studio …")
        subprocess.Popen([str(foxglove), deep_link],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)

    # ── info ───────────────────────────────────────────────────────────────
    print()
    print(f"  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  Foxglove 已启动                                          ║")
    print(f"  ║                                                        ║")
    if not no_bridge:
        print(f"  ║  Bridge:      ws://localhost:{bridge_port:<5}                          ║")
    print(f"  ║  Layout:      {layout_name:<40}  ║")
    if no_studio:
        print(f"  ║  Studio not launched (--no-studio).                     ║")
    if no_bridge:
        print(f"  ║  Bridge not launched (--no-bridge).                     ║")
    print(f"  ║  Press Ctrl+C to stop.                                  ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    print()

    # ── wait ───────────────────────────────────────────────────────────────
    if no_bridge and no_studio:
        return

    try:
        if bridge_proc:
            bridge_proc.wait()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n  Shutting down …")
        if layout_server:
            layout_server.shutdown()
        if bridge_proc:
            bridge_proc.terminate()
            try:
                bridge_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge_proc.kill()
        print("  Done.")
