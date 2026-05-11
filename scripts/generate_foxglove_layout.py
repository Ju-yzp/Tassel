#!/usr/bin/env python3
"""Generate Foxglove Studio layout JSON from a YAML panel config.

Thin CLI wrapper around tassel_foxglove.generate_layout().
Panel types, config, and layout logic live in the tassel_foxglove package.

Usage:
  python3 scripts/generate_foxglove_layout.py [config.yaml] [-o layout.json] [-c COLS]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not found.  Install with:  pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# allow running from repo root without PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tassel_foxglove import generate_layout  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Generate Foxglove Studio layout JSON from YAML panel config")
    parser.add_argument(
        "config", nargs="?", default="config/example_viewer.yaml",
        help="YAML panel config (default: config/example_viewer.yaml)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: <config_stem>.foxglove.json)")
    parser.add_argument("-c", "--cols", type=int, default=None,
                        help="Grid columns (default: from YAML or 2)")
    args = parser.parse_args()

    with open(args.config) as f:
        root = yaml.safe_load(f)

    panels = root.get("panels", [])
    if not panels:
        print(f"'{args.config}' contains no 'panels' list", file=sys.stderr)
        sys.exit(1)

    cols = args.cols or root.get("grid_cols", 2)
    layout = generate_layout(panels, cols)

    if args.output:
        out = args.output
    else:
        cfg = Path(args.config)
        out = str(cfg.parent / f"{cfg.stem}.foxglove.json")
    with open(out, "w") as f:
        json.dump(layout, f, indent=2)

    print(f"Foxglove layout → {out}  ({len(panels)} panels, {cols} cols)")


if __name__ == "__main__":
    main()
