"""Tassel Foxglove — one-click Foxglove Studio visualization for ROS 2.

Usage as a library
──────────────────
    from tassel_foxglove import launch, generate_layout

    # generate layout JSON only
    layout = generate_layout(panels, grid_cols=2)

    # full launch: bridge + layout + Studio
    launch("config/example_viewer.yaml")

Usage via scripts
─────────────────
    python3 scripts/generate_foxglove_layout.py config/example_viewer.yaml
    python3 scripts/launch_foxglove.py config/example_viewer.yaml

Environment config
──────────────────
Copy  tassel_foxglove/env.yaml  to your project, edit it, then either:

    export TASSEL_FOXGLOVE_ENV=my_env.yaml
    python3 scripts/launch_foxglove.py config/example_viewer.yaml

Or pass it directly:

    from tassel_foxglove import launch
    launch("config.yaml", env_path="my_env.yaml")
"""

from ._config import FOXGLOVE_PANEL, WIDE_PANELS, generate_layout, panel_config
from ._env import bridge_defaults, foxglove_binary, layout_defaults, load_env
from ._launcher import launch

__all__ = [
    "FOXGLOVE_PANEL",
    "WIDE_PANELS",
    "bridge_defaults",
    "foxglove_binary",
    "generate_layout",
    "launch",
    "layout_defaults",
    "load_env",
    "panel_config",
]
