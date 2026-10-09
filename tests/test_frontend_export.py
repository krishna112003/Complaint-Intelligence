"""Unit tests for frontend data exports and schema conformance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from complaints.config import get_project_root


@pytest.fixture
def data_dir() -> Path:
    root = get_project_root()
    d = root / "frontend" / "public" / "data"
    assert d.is_dir(), (
        f"Frontend data directory {d} does not exist. Run make export-frontend first."
    )
    return d


def test_exported_json_files_exist(data_dir: Path):
    """Ensure all 9 required frontend JSON files are present."""
    required_files = [
        "summary.json",
        "pipeline.json",
        "models.json",
        "pairwise_tests.json",
        "calibration.json",
        "bias.json",
        "volume.json",
        "examples.json",
        "gate.json",
    ]
    for rf in required_files:
        p = data_dir / rf
        assert p.is_file(), f"Missing required exported file: {rf}"
        assert p.stat().st_size > 50, (
            f"Exported file {rf} is suspiciously small ({p.stat().st_size} bytes)"
        )


def test_summary_json_schema(data_dir: Path):
    """Verify summary.json structure and numbers."""
    with open(data_dir / "summary.json") as f:
        data = json.load(f)
    assert data["bronze_row_count"] > 100000
    assert data["gold_row_count"] > 10000
    assert data["class_count"] == 10
    assert "train" in data["split_counts"]
    assert "cost_disclaimer" in data


def test_models_json_schema_and_cost_label(data_dir: Path):
    """Verify models.json contains bootstrap CIs, pareto indicators, and hypothetical cost labeling."""
    with open(data_dir / "models.json") as f:
        data = json.load(f)
    assert "models" in data
    assert len(data["models"]) >= 5
    for m in data["models"]:
        assert "name" in m
        assert "macro_f1" in m
        assert "ci_low" in m
        assert "ci_high" in m
        assert "cost_label" in m
        assert "reference pricing" in m["cost_label"]
        assert m["ci_low"] <= m["macro_f1"] <= m["ci_high"]


def test_pairwise_tests_schema(data_dir: Path):
    """Verify pairwise_tests.json contains McNemar and Holm-adjusted p-values."""
    with open(data_dir / "pairwise_tests.json") as f:
        data = json.load(f)
    assert "pairs" in data
    assert len(data["pairs"]) >= 10
    for p in data["pairs"]:
        assert "model_a" in p
        assert "model_b" in p
        assert "delta_f1" in p
        assert "mcnemar_stat" in p
        assert "mcnemar_p_value" in p
        assert "adjusted_p_value" in p


def test_calibration_schema(data_dir: Path):
    """Verify calibration.json contains 10 bins per model and valid ECE."""
    with open(data_dir / "calibration.json") as f:
        data = json.load(f)
    assert "models" in data
    for _name, m in data["models"].items():
        assert "ece" in m
        assert "bins" in m
        assert len(m["bins"]) == 10


def test_bias_schema(data_dir: Path):
    """Verify bias.json contains all 4 probes and before/after example pairs."""
    with open(data_dir / "bias.json") as f:
        data = json.load(f)
    assert "probes" in data
    assert "name_gender" in data["probes"]
    assert "name_ethnicity" in data["probes"]
    assert "location" in data["probes"]
    assert "typo" in data["probes"]
    assert "examples" in data
    assert len(data["examples"]) >= 4


def test_volume_schema(data_dir: Path):
    """Verify volume.json contains monthly trends, product breakdown, and rolling averages."""
    with open(data_dir / "volume.json") as f:
        data = json.load(f)
    assert "products" in data
    assert len(data["products"]) == 10
    assert "by_product" in data
    assert "summary_table" in data
    assert len(data["summary_table"]) == 10
    assert "overall_monthly" in data
    assert len(data["overall_monthly"]) > 0
    for s in data["summary_table"]:
        assert "product" in s
        assert "total_volume" in s
        assert "latest_rolling_3m_avg" in s


def test_examples_schema(data_dir: Path):
    """Verify examples.json contains gold set narratives and model predictions."""
    with open(data_dir / "examples.json") as f:
        data = json.load(f)
    assert "examples" in data
    assert len(data["examples"]) >= 30
    for ex in data["examples"]:
        assert "complaint_id" in ex
        assert "narrative" in ex
        assert "true_label" in ex
        assert "model_predictions" in ex
        assert "llm_extraction" in ex


def test_gate_schema(data_dir: Path):
    """Verify gate.json contains pass/fail status and all 4 guardrails."""
    with open(data_dir / "gate.json") as f:
        data = json.load(f)
    assert "passed" in data
    assert "checks" in data
    assert len(data["checks"]) == 4
