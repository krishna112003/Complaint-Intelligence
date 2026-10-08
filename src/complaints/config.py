"""Centralised configuration loader.

Loads all YAML configs from configs/, applies the global seed, and exposes
typed access via module-level functions. All config values are read-only
after loading. No magic numbers should exist in code — they live here.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ── Internal state ────────────────────────────────────────────────────────────
_configs: dict[str, dict[str, Any]] = {}
_loaded = False


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def load_configs(configs_dir: Path | None = None) -> None:
    """Load all YAML config files and apply the global seed.

    Idempotent — calling twice is safe but re-reads from disk.
    """
    global _configs, _loaded

    configs_dir = configs_dir or CONFIGS_DIR
    _configs.clear()

    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        key = yaml_file.stem  # e.g. "data", "models", "eval", "llm"
        _configs[key] = _load_yaml(yaml_file)
        logger.debug("Loaded config: %s (%d keys)", key, len(_configs[key]))

    # Load .env file if present
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug("Loaded .env from %s", env_path)

    _loaded = True

    # Apply global seed
    seed = get_seed()
    _apply_seed(seed)
    logger.info("Global seed set to %d", seed)


def _apply_seed(seed: int) -> None:
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (optional — only if installed and used)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass


def _ensure_loaded() -> None:
    """Auto-load configs on first access."""
    if not _loaded:
        load_configs()


# ── Public accessors ──────────────────────────────────────────────────────────


def get(config_name: str, key: str, default: Any = None) -> Any:
    """Get a value from a named config file.

    Example: get("data", "seed") → 42
    """
    _ensure_loaded()
    return _configs.get(config_name, {}).get(key, default)


def get_section(config_name: str) -> dict[str, Any]:
    """Get the entire dict for a config file (e.g. 'data', 'models')."""
    _ensure_loaded()
    return _configs.get(config_name, {})


def get_seed() -> int:
    """Return the global seed (from data.yaml, falls back to 42)."""
    _ensure_loaded()
    # Try data config first, then any config with a seed key
    for name in ("data", "models", "eval", "llm"):
        cfg = _configs.get(name, {})
        if "seed" in cfg:
            return int(cfg["seed"])
    return 42


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    return PROJECT_ROOT


def get_env(key: str, required: bool = False) -> str | None:
    """Read an environment variable, optionally failing loudly if missing."""
    _ensure_loaded()
    value = os.environ.get(key)
    if required and not value:
        raise OSError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the entire project."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    # Quiet noisy libraries
    for lib in ("urllib3", "httpx", "httpcore", "filelock", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)
