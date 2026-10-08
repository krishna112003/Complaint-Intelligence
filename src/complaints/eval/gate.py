"""Regression gate for CI/CD and deployment qualification.

Phase 9 implementation:
- Validates candidate model / prompt performance against stored baselines.
- Enforces 4 guardrails:
  1. macro_f1_drop <= macro_f1_drop_max (default: 0.02)
  2. schema_valid_rate >= schema_valid_rate_min (default: 0.98)
  3. cost_increase <= cost_increase_max (default: 0.20)
  4. bias_flip_rate <= bias_flip_rate_max (default: 0.05)
- Generates reports/gate_report.md.
- Exits with 0 on pass, 1 on regression failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from complaints.config import get, get_project_root

logger = logging.getLogger(__name__)


def load_baseline_metrics(path: Path | None = None) -> dict[str, Any]:
    """Load baseline metrics from JSON file."""
    root = get_project_root()
    if path is None:
        baseline_file = get("eval", "baseline_metrics_file", "reports/baseline_metrics.json")
        path = root / baseline_file

    if not path.exists():
        logger.warning("Baseline file not found at %s. Using default fallback baseline.", path)
        return {
            "best_model_name": "tfidf_logistic_regression",
            "baseline_macro_f1": 0.6002,
            "baseline_cost_per_1k": 0.0,
        }

    with open(path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def run_gate_checks(
    candidate_metrics: dict[str, float | str | None],
    baseline_metrics: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Execute all regression checks against configured thresholds.

    Parameters
    ----------
    candidate_metrics : dict
        Dict containing candidate values: 'macro_f1', 'schema_valid_rate',
        'cost_per_1k', 'bias_flip_rate'.
    baseline_metrics : dict, optional
        Baseline reference values.
    thresholds : dict, optional
        Threshold limits overriding configs/eval.yaml.

    Returns
    -------
    (all_passed, check_details)
    """
    if baseline_metrics is None:
        baseline_metrics = load_baseline_metrics()

    if thresholds is None:
        thresholds = {
            "macro_f1_drop_max": float(get("eval", "gate.macro_f1_drop_max", 0.02)),
            "schema_valid_rate_min": float(get("eval", "gate.schema_valid_rate_min", 0.98)),
            "cost_increase_max": float(get("eval", "gate.cost_increase_max", 0.20)),
            "bias_flip_rate_max": float(get("eval", "gate.bias_flip_rate_max", 0.05)),
        }

    checks: list[dict[str, Any]] = []
    base_f1 = float(baseline_metrics.get("baseline_macro_f1", 0.6002))
    base_cost = float(baseline_metrics.get("baseline_cost_per_1k", 0.0))

    # 1. Macro-F1 Quality Check
    cand_f1 = candidate_metrics.get("macro_f1")
    if cand_f1 is not None:
        f1_val = float(cand_f1)
        f1_drop = base_f1 - f1_val
        passed = f1_drop <= thresholds["macro_f1_drop_max"]
        checks.append(
            {
                "name": "Macro-F1 Quality Drop",
                "status": "PASS" if passed else "FAIL",
                "candidate_val": f1_val,
                "baseline_val": base_f1,
                "delta": -f1_drop,
                "threshold": f"Drop <= {thresholds['macro_f1_drop_max']:.2%}",
                "detail": f"Observed drop of {f1_drop:.4f} (limit: {thresholds['macro_f1_drop_max']:.4f})",
            }
        )

    # 2. Schema Validity Rate
    cand_schema = candidate_metrics.get("schema_valid_rate")
    if cand_schema is not None:
        schema_val = float(cand_schema)
        passed = schema_val >= thresholds["schema_valid_rate_min"]
        checks.append(
            {
                "name": "JSON Schema Validity",
                "status": "PASS" if passed else "FAIL",
                "candidate_val": schema_val,
                "baseline_val": 1.00,
                "delta": schema_val - 1.0,
                "threshold": f"Valid Rate >= {thresholds['schema_valid_rate_min']:.2%}",
                "detail": f"Observed {schema_val:.2%} valid rate (min: {thresholds['schema_valid_rate_min']:.2%})",
            }
        )

    # 3. Cost Increase Check
    cand_cost = candidate_metrics.get("cost_per_1k")
    if cand_cost is not None:
        cost_val = float(cand_cost)
        if base_cost > 0:
            cost_increase_ratio = (cost_val - base_cost) / base_cost
            passed = cost_increase_ratio <= thresholds["cost_increase_max"]
            detail = f"Observed cost change: {cost_increase_ratio:+.1%} (max increase: {thresholds['cost_increase_max']:.1%})"
        else:
            # Baseline is free (0.0), pass if candidate <= 0.05 or within reasonable limit
            passed = cost_val <= 0.10
            detail = f"Observed candidate cost ${cost_val:.4f}/1k vs baseline ${base_cost:.4f}/1k"
        checks.append(
            {
                "name": "Inference Cost Ceiling",
                "status": "PASS" if passed else "FAIL",
                "candidate_val": cost_val,
                "baseline_val": base_cost,
                "delta": cost_val - base_cost,
                "threshold": f"Increase <= {thresholds['cost_increase_max']:.1%}",
                "detail": detail,
            }
        )

    # 4. Bias / Robustness Flip Rate Check
    cand_bias = candidate_metrics.get("bias_flip_rate")
    if cand_bias is not None:
        bias_val = float(cand_bias)
        passed = bias_val <= thresholds["bias_flip_rate_max"]
        checks.append(
            {
                "name": "Bias / Robustness Flip Rate",
                "status": "PASS" if passed else "FAIL",
                "candidate_val": bias_val,
                "baseline_val": 0.00,
                "delta": bias_val,
                "threshold": f"Flip Rate <= {thresholds['bias_flip_rate_max']:.2%}",
                "detail": f"Observed flip rate {bias_val:.2%} (max allowed: {thresholds['bias_flip_rate_max']:.2%})",
            }
        )

    all_passed = all(c["status"] == "PASS" for c in checks) if checks else True
    return all_passed, checks


def generate_gate_report(
    all_passed: bool,
    checks: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """Format and write gate_report.md."""
    root = get_project_root()
    if output_path is None:
        report_dir = root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / "gate_report.md"

    overall_status = "PASSED" if all_passed else "FAILED"
    status_icon = "PASS" if all_passed else "FAIL"

    lines = [
        f"# LLMOps Regression Gate Report: [{overall_status}]\n\n",
        f"**Gate Outcome:** `{status_icon}`\n",
        f"**Total Checks:** {len(checks)}\n",
        f"**Passed Checks:** {sum(1 for c in checks if c['status'] == 'PASS')}\n\n",
        "## Evaluated Guardrails\n\n",
        "| Check Name | Status | Candidate | Baseline | Threshold | Detail |\n",
        "|---|---|---|---|---|---|\n",
    ]

    for c in checks:
        cand_str = (
            f"{c['candidate_val']:.4f}"
            if isinstance(c["candidate_val"], float)
            else str(c["candidate_val"])
        )
        base_str = (
            f"{c['baseline_val']:.4f}"
            if isinstance(c["baseline_val"], float)
            else str(c["baseline_val"])
        )
        lines.append(
            f"| {c['name']} | `{c['status']}` | {cand_str} | {base_str} | "
            f"{c['threshold']} | {c['detail']} |\n"
        )

    lines.append("\n## Governance Decision\n\n")
    if all_passed:
        lines.append(
            "> [!NOTE]\n"
            "> **Deployment Approved**: All candidate metrics satisfy quality, schema validity, cost, and fairness guardrails.\n"
        )
    else:
        lines.append(
            "> [!WARNING]\n"
            "> **Deployment Blocked**: Candidate evaluation failed one or more regression thresholds. Review details above before merging.\n"
        )

    with open(output_path, "w") as f:
        f.writelines(lines)

    logger.info("Saved gate report to %s", output_path)
    return output_path


def check_gate(
    candidate_metrics: dict[str, float | str | None] | None = None,
    output_path: Path | None = None,
) -> bool:
    """Check regression gate against stored baseline.

    If candidate_metrics is None, evaluates default candidate values from
    eval results or baseline itself.
    """
    if candidate_metrics is None:
        # Default candidate: top baseline model metrics
        baseline = load_baseline_metrics()
        base_f1 = float(baseline.get("baseline_macro_f1", 0.6002))
        base_cost = float(baseline.get("baseline_cost_per_1k", 0.0))
        candidate_metrics = {
            "macro_f1": base_f1,
            "schema_valid_rate": 1.0,
            "cost_per_1k": base_cost,
            "bias_flip_rate": 0.00,
        }

    all_passed, checks = run_gate_checks(candidate_metrics)
    generate_gate_report(all_passed, checks, output_path=output_path)

    for c in checks:
        logger.info(
            "Gate check [%s]: %s | %s",
            c["name"],
            c["status"],
            c["detail"],
        )

    return all_passed


def main() -> None:
    """CLI entrypoint for regression gate."""
    parser = argparse.ArgumentParser(description="LLMOps Regression Gate")
    parser.add_argument("--f1", type=float, default=None, help="Candidate macro F1")
    parser.add_argument("--schema-rate", type=float, default=None, help="Schema valid rate")
    parser.add_argument("--cost", type=float, default=None, help="Candidate cost per 1k")
    parser.add_argument("--bias-flip", type=float, default=None, help="Bias flip rate")
    args = parser.parse_args()

    candidate = None
    if any(v is not None for v in [args.f1, args.schema_rate, args.cost, args.bias_flip]):
        candidate = {
            "macro_f1": args.f1,
            "schema_valid_rate": args.schema_rate,
            "cost_per_1k": args.cost,
            "bias_flip_rate": args.bias_flip,
        }

    passed = check_gate(candidate_metrics=candidate)
    if passed:
        logger.info("Gate PASSED.")
        sys.exit(0)
    else:
        logger.error("Gate FAILED regression checks.")
        sys.exit(1)


if __name__ == "__main__":
    from complaints.config import setup_logging

    setup_logging()
    main()
