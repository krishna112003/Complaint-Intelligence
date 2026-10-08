"""Smoke tests to verify repository setup, configuration loading, and fake LLM client."""

from __future__ import annotations

import numpy as np

from complaints.config import get, get_project_root, get_section, get_seed, load_configs
from complaints.llm.client import FakeLLMClient
from complaints.models.base import PREDICTION_COLUMNS, Timer


def test_config_loader():
    """Verify that YAML configs load without errors and default values exist."""
    load_configs()
    seed = get_seed()
    assert seed == 42
    assert get("data", "seed") == 42
    assert get("llm", "active_provider") == "gemini"

    data_cfg = get_section("data")
    assert "sample_size" in data_cfg
    assert data_cfg["sample_size"] == 60000

    root = get_project_root()
    assert (root / "configs").is_dir()


def test_seed_determinism():
    """Verify numpy RNG is seeded by config loader."""
    load_configs()
    val1 = np.random.rand()
    load_configs()
    val2 = np.random.rand()
    assert val1 == val2


def test_fake_llm_client():
    """Verify that FakeLLMClient operates deterministically without network calls."""
    client = FakeLLMClient(default_response='{"issue_category": "Credit Reporting"}')
    assert client.model_name == "fake-model-v1"

    messages = [{"role": "user", "content": "Sample complaint test"}]
    resp = client.complete(messages)

    assert resp.text == '{"issue_category": "Credit Reporting"}'
    assert resp.input_tokens > 0
    assert resp.cost_usd == 0.0
    assert client.call_count == 1


def test_prediction_schema_columns():
    """Verify the shared prediction schema columns definition."""
    required = [
        "complaint_id",
        "y_true",
        "y_pred",
        "confidence",
        "latency_ms",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "model_name",
        "run_id",
    ]
    assert required == PREDICTION_COLUMNS


def test_timer_context_manager():
    """Verify Timer measures elapsed time correctly."""
    import time

    with Timer() as timer:
        time.sleep(0.01)
    assert timer.elapsed_ms >= 5.0
