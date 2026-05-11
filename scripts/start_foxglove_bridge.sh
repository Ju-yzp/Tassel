#!/usr/bin/env bash
# ── Foxglove Bridge Launcher ───────────────────────────────────────────────
# 1. Reads  config/stereo_vio_viewer.yaml  (or $1)
# 2. Generates  .foxglove.json  layout
# 3. Starts  foxglove_bridge  on  ws://localhost:8765
# 4. Launches  foxglove-studio  with the generated layout
#
# Prefer  scripts/launch_foxglove.py  for the one-click deep-link workflow.
# This script is kept as a simple, dependency-light alternative.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONFIG="${1:-config/stereo_vio_viewer.yaml}"
BRIDGE_PORT="${2:-8765}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ── 1. check dependencies ────────────────────────────────────────────────
if ! python3 -c "import yaml" 2>/dev/null; then
    echo "  PyYAML not found →  pip install pyyaml"
    exit 1
fi

if ! ros2 pkg prefix foxglove_bridge &>/dev/null; then
    echo "  foxglove_bridge not found →  sudo apt install ros-${ROS_DISTRO:-humble}-foxglove-bridge"
    exit 1
fi

# ── 2. generate Foxglove layout ───────────────────────────────────────────
echo "── Reading: $CONFIG"
if [[ "$CONFIG" == *.yaml ]]; then
    LAYOUT="${CONFIG%.yaml}.foxglove.json"
else
    LAYOUT="${CONFIG%.yml}.foxglove.json"
fi
python3 "$SCRIPT_DIR/generate_foxglove_layout.py" "$CONFIG" -o "$LAYOUT"

# ── 3. kill any old foxglove_bridge on the same port ──────────────────────
OLD_PID=$(pgrep -f "foxglove_bridge" 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    echo "── Killing old foxglove_bridge (PID $OLD_PID) …"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

# ── 4. start foxglove_bridge ──────────────────────────────────────────────
echo "── Starting foxglove_bridge on ws://localhost:${BRIDGE_PORT} …"
trap 'echo; echo "── Shutting down …"; kill %1 2>/dev/null; exit 0' INT TERM

ros2 run foxglove_bridge foxglove_bridge \
    --ros-args -p port:="${BRIDGE_PORT}" &

sleep 2

# ── 5. launch foxglove-studio with deep-link ──────────────────────────────
LAYOUT_NAME="$(basename "$LAYOUT")"
PROJECT_ABS="$(realpath "$PROJECT_DIR")"

# Start a tiny HTTP server to serve the layout file for the deep-link
LAYOUT_PORT=0
python3 -c "
import socket, threading, http.server, os, sys
os.chdir('$PROJECT_ABS')

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
port = s.getsockname()[1]
s.close()
print(port, end='')
" 2>/dev/null || {
    echo "  Warning: could not find free port for layout server"
    LAYOUT_PORT=""
}

if [ -n "$LAYOUT_PORT" ]; then
    # serve layout in background
    python3 -c "
import http.server, os, threading
os.chdir('$PROJECT_ABS')

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

s = http.server.HTTPServer(('127.0.0.1', $LAYOUT_PORT), Handler)
t = threading.Thread(target=s.serve_forever, daemon=True)
t.start()
import time; time.sleep(999999)
" &
    SERVER_PID=$!

    DEEP_LINK="foxglove://open?ds=foxglove-websocket&ds.url=ws://localhost:${BRIDGE_PORT}&layoutUrl=http://127.0.0.1:${LAYOUT_PORT}/${LAYOUT_NAME}"
    echo "── Opening Foxglove Studio …"
    foxglove-studio "$DEEP_LINK" &
else
    # fallback: file:// URL
    LAYOUT_ABS="$(realpath "$LAYOUT")"
    echo "── Opening Foxglove Studio with layout: ${LAYOUT_ABS}"
    foxglove-studio "file://${LAYOUT_ABS}" &
fi

# ── 6. ready ──────────────────────────────────────────────────────────────
echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  Foxglove Studio 已启动                                  ║"
echo "  ║                                                        ║"
echo "  ║  Bridge:  ws://localhost:${BRIDGE_PORT}                          ║"
echo "  ║  Layout:  ${LAYOUT}"
echo "  ║                                                        ║"
echo "  ║  Press Ctrl+C to stop bridge.                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

wait
