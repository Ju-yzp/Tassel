#!/usr/bin/env python3
"""Start foxglove_bridge and open the configured Tassel layout."""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import yaml


def load_config(path: Path) -> tuple[dict, Path, dict]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    studio = config.get("studio", {})
    layout_value = studio.get("layout_file")
    if not layout_value:
        raise ValueError("studio.layout_file is required")
    layout_path = Path(layout_value)
    if not layout_path.is_absolute():
        layout_path = path.parent / layout_path
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    required_keys = {
        "configById", "globalVariables", "userNodes", "playbackConfig", "layout"
    }
    if not required_keys.issubset(layout):
        raise ValueError(f"invalid Foxglove layout: {layout_path}")
    apply_viewer_config(config, layout)
    return config, layout_path, layout


def apply_viewer_config(config: dict, layout: dict) -> None:
    viewer = config.get("viewer", {})
    config_by_id = layout["configById"]
    scene = config_by_id["3D!tasselmain"]
    path = scene["topics"]["/vo/path"]
    cloud = scene["topics"]["/landmarks"]
    plot = config_by_id["Plot!tasselcostreduction"]

    path["color"] = str(viewer.get("path_color", path.get("color", "#ef4444")))
    path["lineWidth"] = float(viewer.get("path_line_width", path.get("lineWidth", 0.005)))
    cloud["pointSize"] = float(viewer.get("point_size", cloud.get("pointSize", 1.75)))
    plot["followingViewWidth"] = float(
        viewer.get("cost_window_seconds", plot.get("followingViewWidth", 15))
    )


def layouts_directory() -> Path:
    root = Path.home() / ".config" / "Foxglove" / "studio-datastores"
    candidates = sorted(root.glob("layouts-remote-*"))
    if candidates:
        return candidates[0]
    local = root / "layouts-local"
    local.mkdir(parents=True, exist_ok=True)
    return local


def install_layout(layout: dict, name: str, source: Path) -> tuple[str, Path]:
    layout_payload = json.dumps(layout, sort_keys=True, separators=(",", ":"))
    digest_input = f"{source.resolve()}\n{layout_payload}".encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    layout_id = f"lay_{digest}"
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    entry = {
        "id": layout_id,
        "name": name,
        "permission": "CREATOR_WRITE",
        "baseline": {"data": layout, "savedAt": now},
        # Project layouts are local-only. "tracked" makes Foxglove delete the
        # cache entry when no layout with this ID exists on its cloud service.
        "syncInfo": {"status": "exempt"},
    }
    destination = layouts_directory() / layout_id
    destination.write_text(json.dumps(entry, ensure_ascii=True), encoding="utf-8")
    return layout_id, destination


def port_is_open(address: str, port: int) -> bool:
    host = "127.0.0.1" if address in ("0.0.0.0", "::") else address
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def start_bridge(config: dict) -> subprocess.Popen | None:
    bridge = config.get("bridge", {})
    port = int(bridge.get("port", 8765))
    address = str(bridge.get("address", "127.0.0.1"))
    if port_is_open(address, port):
        print(f"Using bridge already listening on ws://{address}:{port}")
        return None

    command = [
        "ros2", "run", "foxglove_bridge", "foxglove_bridge", "--ros-args",
        "-p", f"port:={port}", "-p", f"address:={address}",
    ]
    whitelist = bridge.get("topic_whitelist", [])
    if whitelist:
        command.extend(["-p", "topic_whitelist:=[" + ",".join(whitelist) + "]"])
    process = subprocess.Popen(command)
    for _ in range(50):
        if process.poll() is not None:
            raise RuntimeError("foxglove_bridge exited during startup")
        if port_is_open(address, port):
            print(f"Bridge listening on ws://{address}:{port}")
            return process
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError("foxglove_bridge did not open its port within 5 seconds")


def open_studio(config: dict, layout_id: str) -> None:
    bridge = config.get("bridge", {})
    studio = config.get("studio", {})
    port = int(bridge.get("port", 8765))
    address = str(bridge.get("address", "127.0.0.1"))
    connect_host = "127.0.0.1" if address in ("0.0.0.0", "::") else address
    executable = str(studio.get("executable", "foxglove-studio"))
    resolved = shutil.which(executable) if not Path(executable).is_absolute() else executable
    if not resolved or not Path(resolved).exists():
        raise FileNotFoundError(f"Foxglove Studio not found: {executable}")
    query = urlencode({
        "ds": "foxglove-websocket",
        "ds.url": f"ws://{connect_host}:{port}",
        "layoutId": layout_id,
    })
    environment = dict(os.environ)
    environment.pop("ELECTRON_RUN_AS_NODE", None)
    process = subprocess.Popen(
        [str(resolved), f"foxglove://open?{query}"],
        env=environment,
    )
    time.sleep(1.0)
    if process.poll() not in (None, 0):
        raise RuntimeError(f"Foxglove Studio exited with code {process.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="config/foxglove.yaml")
    parser.add_argument("--no-studio", action="store_true")
    parser.add_argument("--no-bridge", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config, layout_path, layout = load_config(config_path)
    layout_name = str(config.get("studio", {}).get("layout_name", layout_path.stem))
    layout_id, installed_path = install_layout(layout, layout_name, layout_path)
    print(f"Layout {layout_name!r} installed as {layout_id}")
    print(f"Layout store: {installed_path}")
    if args.generate_only:
        return

    bridge_process = None
    try:
        if not args.no_bridge:
            bridge_process = start_bridge(config)
        if not args.no_studio:
            open_studio(config, layout_id)
            print("Foxglove opened with the configured layout")
        if bridge_process is not None:
            bridge_process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if bridge_process is not None and bridge_process.poll() is None:
            bridge_process.terminate()
            bridge_process.wait(timeout=5)


if __name__ == "__main__":
    main()
