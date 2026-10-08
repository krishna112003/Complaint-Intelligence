"""MLflow experiment tracking helpers.

Provides structured run tracking, parameter/metric logging, artifact saving,
and automatic tagging with git commit hash and data split hashes.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from complaints.config import get

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Return the current short git commit hash, or 'unknown' if not in git."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for lineage tracking."""
    if not filepath.exists():
        return "file_not_found"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()[:12]


def setup_mlflow(experiment_name: str = "complaint-intelligence") -> str:
    """Configure MLflow tracking URI and experiment."""
    tracking_uri = str(get("tracking", "uri", "sqlite:///mlruns/mlflow.db"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured: URI=%s, experiment=%s", tracking_uri, experiment_name)
    return tracking_uri


def start_tracked_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    split_file: Path | None = None,
) -> mlflow.ActiveRun:
    """Start an MLflow run enriched with git commit and dataset lineage tags."""
    setup_mlflow()
    run_tags = {
        "git_commit": get_git_commit_hash(),
        "run_name": run_name,
    }
    if split_file:
        run_tags["split_file_hash"] = compute_file_hash(split_file)
    if tags:
        run_tags.update(tags)

    return mlflow.start_run(run_name=run_name, tags=run_tags)


def log_metrics_and_params(
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: list[Path] | None = None,
) -> None:
    """Log parameters, metrics, and artifact files to the active MLflow run."""
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if artifacts:
        for art in artifacts:
            if art.exists():
                mlflow.log_artifact(str(art))
