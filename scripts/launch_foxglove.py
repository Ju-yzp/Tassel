#!/usr/bin/env python3
"""Launch Foxglove Studio with auto-generated layout — one command.

Thin CLI wrapper around tassel_foxglove.launch().
The real logic lives in the tassel_foxglove package.

Usage:
  python3 scripts/launch_foxglove.py [config.yaml]
  python3 scripts/launch_foxglove.py config/example_viewer.yaml -p 9090
  TASSEL_FOXGLOVE_ENV=my_env.yaml python3 scripts/launch_foxglove.py ...
"""

import argparse
import sys
from pathlib import Path

# allow running from repo root without PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tassel_foxglove import launch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Launch Foxglove Studio with auto-generated layout")
    parser.add_argument(
        "config", nargs="?", default="config/example_viewer.yaml",
        help="YAML panel config")
    parser.add_argument(
        "-p", "--port", type=int, default=None,
        help="Bridge WebSocket port (default: from env.yaml or 8765)")
    parser.add_argument("--no-studio", action="store_true",
                        help="Skip launching Foxglove Studio")
    parser.add_argument("--no-bridge", action="store_true",
                        help="Skip starting foxglove_bridge")
    parser.add_argument(
        "-e", "--env", default=None,
        help="Environment config file (default: TASSEL_FOXGLOVE_ENV or built-in)")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    launch(
        args.config,
        port=args.port,
        env_path=args.env,
        no_studio=args.no_studio,
        no_bridge=args.no_bridge,
        project_dir=project_dir,
    )


if __name__ == "__main__":
    main()
