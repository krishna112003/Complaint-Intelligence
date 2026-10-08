"""Unit tests for Phase 9 regression gate and CI threshold enforcement."""

from __future__ import annotations

from complaints.eval.gate import generate_gate_report, run_gate_checks


def test_gate_passes_when_all_thresholds_met() -> None:
    """Test that a healthy candidate passes all 4 guardrails."""
    baseline = {"baseline_macro_f1": 0.60, "baseline_cost_per_1k": 0.02}
    candidate = {
        "macro_f1": 0.595,  # 0.005 drop <= 0.02 limit
        "schema_valid_rate": 0.99,  # >= 0.98 limit
        "cost_per_1k": 0.022,  # 10% increase <= 20% limit
        "bias_flip_rate": 0.02,  # <= 0.05 limit
    }
    passed, checks = run_gate_checks(candidate, baseline_metrics=baseline)
    assert passed is True
    assert len(checks) == 4
    assert all(c["status"] == "PASS" for c in checks)


def test_gate_fails_on_macro_f1_drop() -> None:
    """Test that an F1 drop exceeding threshold triggers a failure."""
    baseline = {"baseline_macro_f1": 0.60, "baseline_cost_per_1k": 0.0}
    candidate = {
        "macro_f1": 0.55,  # 0.05 drop > 0.02 limit
        "schema_valid_rate": 1.0,
    }
    passed, checks = run_gate_checks(candidate, baseline_metrics=baseline)
    assert passed is False
    f1_check = next(c for c in checks if c["name"] == "Macro-F1 Quality Drop")
    assert f1_check["status"] == "FAIL"


def test_gate_fails_on_schema_invalidity() -> None:
    """Test that schema valid rate below 0.98 triggers a failure."""
    baseline = {"baseline_macro_f1": 0.60, "baseline_cost_per_1k": 0.0}
    candidate = {
        "macro_f1": 0.60,
        "schema_valid_rate": 0.94,  # < 0.98 limit
    }
    passed, checks = run_gate_checks(candidate, baseline_metrics=baseline)
    assert passed is False
    schema_check = next(c for c in checks if c["name"] == "JSON Schema Validity")
    assert schema_check["status"] == "FAIL"


def test_gate_fails_on_cost_increase() -> None:
    """Test that cost increase > 20% triggers a failure."""
    baseline = {"baseline_macro_f1": 0.60, "baseline_cost_per_1k": 0.05}
    candidate = {
        "macro_f1": 0.60,
        "cost_per_1k": 0.08,  # +60% > 20% limit
    }
    passed, checks = run_gate_checks(candidate, baseline_metrics=baseline)
    assert passed is False
    cost_check = next(c for c in checks if c["name"] == "Inference Cost Ceiling")
    assert cost_check["status"] == "FAIL"


def test_gate_fails_on_bias_flip_rate() -> None:
    """Test that bias flip rate > 5% triggers a failure."""
    candidate = {
        "macro_f1": 0.60,
        "bias_flip_rate": 0.12,  # 12% > 5% limit
    }
    passed, checks = run_gate_checks(candidate)
    assert passed is False
    bias_check = next(c for c in checks if c["name"] == "Bias / Robustness Flip Rate")
    assert bias_check["status"] == "FAIL"


def test_generate_gate_report_structure(tmp_path) -> None:
    """Test gate report markdown generation."""
    checks = [
        {
            "name": "Macro-F1 Quality Drop",
            "status": "PASS",
            "candidate_val": 0.60,
            "baseline_val": 0.60,
            "threshold": "Drop <= 2%",
            "detail": "No drop",
        }
    ]
    report_file = tmp_path / "gate_report.md"
    generate_gate_report(True, checks, output_path=report_file)
    assert report_file.exists()
    content = report_file.read_text()
    assert "LLMOps Regression Gate Report: [PASSED]" in content
    assert "| Macro-F1 Quality Drop | `PASS` |" in content
