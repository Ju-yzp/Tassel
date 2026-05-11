"""Environment configuration — loads from env.yaml with sensible defaults."""

import os
from pathlib import Path

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent
_BUILTIN_ENV = _PACKAGE_DIR / "env.yaml"

# defaults used when no env.yaml is present
_DEFAULTS = {
    "foxglove": {
        "studio_path": "auto",
        "bridge": {
            "port": 8765,
            "address": "0.0.0.0",
            "tls": False,
            "use_compression": False,
        },
    },
    "layout": {
        "default_grid_cols": 2,
    },
}

_env_cache: dict | None = None


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Merge overlay into base recursively, returning a new dict."""
    result = base.copy()
    for k, v in overlay.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_env(env_path: str | Path | None = None) -> dict:
    """Load environment config, merging user overrides onto built-in defaults.

    Resolution order (later wins):
      1. hardcoded defaults
      2. built-in env.yaml (tassel_foxglove/env.yaml)
      3. user-supplied env file (env_path parameter)
      4. TASSEL_FOXGLOVE_ENV environment variable (if set and env_path is None)

    Returns the merged config dict.  Result is cached after the first call.
    """
    global _env_cache
    if _env_cache is not None and env_path is None:
        return _env_cache

    cfg = _DEFAULTS.copy()

    if _BUILTIN_ENV.exists():
        with open(_BUILTIN_ENV) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    if env_path is None:
        env_path = os.environ.get("TASSEL_FOXGLOVE_ENV", "")

    if env_path:
        p = Path(env_path)
        if p.exists():
            with open(p) as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
        else:
            raise FileNotFoundError(f"env file not found: {env_path}")

    _env_cache = cfg
    return cfg


def foxglove_binary() -> Path | None:
    """Resolve the foxglove-studio binary path.

    Returns the configured path (if not "auto"), or searches PATH and common
    install locations.
    """
    env = load_env()
    explicit = env.get("foxglove", {}).get("studio_path", "auto")
    if explicit and explicit != "auto":
        p = Path(explicit)
        return p if p.is_file() else None

    candidates = [
        "/usr/bin/foxglove-studio",
        "/usr/local/bin/foxglove-studio",
        "/opt/Foxglove/foxglove-studio",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return Path(c)
    for d in os.environ.get("PATH", "").split(":"):
        fp = Path(d) / "foxglove-studio"
        if fp.is_file():
            return fp
    return None


def bridge_defaults() -> dict:
    """Return the bridge config dict with port, address, tls, compression."""
    return load_env().get("foxglove", {}).get("bridge", {})


def layout_defaults() -> dict:
    """Return the layout defaults dict."""
    return load_env().get("layout", {})
