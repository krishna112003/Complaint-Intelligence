"""Unit tests for statistical significance tests with hand-calculated examples."""

from __future__ import annotations

import numpy as np

from complaints.eval.bootstrap import compute_bootstrap_ci, compute_paired_bootstrap_difference
from complaints.eval.calibration import compute_expected_calibration_error
from complaints.eval.stats_tests import holm_bonferroni_correction, mcnemar_test, wilcoxon_test


def test_mcnemar_exact_binomial_hand_calculated():
    """Verify McNemar exact binomial test on small discordant counts.

    Scenario:
    y_true:   10 items [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred_a: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (A gets 10 right)
    y_pred_b: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] (B gets 2 right, 8 wrong)

    Discordant pairs:
    b (A right, B wrong) = 8
    c (A wrong, B right) = 0
    Total discordant = 8 (< 25 -> exact binomial)
    Two-sided p-value for k=8, n=8 under p=0.5: 2 * (1/2)^8 = 2/256 = 0.0078125
    """
    y_true = ["1"] * 10
    y_pred_a = ["1"] * 10
    y_pred_b = ["1", "1", "0", "0", "0", "0", "0", "0", "0", "0"]

    res = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert res["method"] == "exact_binomial"
    assert res["contingency_table"]["a_only"] == 8
    assert res["contingency_table"]["b_only"] == 0
    assert np.isclose(res["p_value"], 0.0078125)
    assert res["significant_at_05"] is True


def test_mcnemar_chi_squared_hand_calculated():
    """Verify McNemar asymptotic chi-squared test with Edwards continuity correction.

    Suppose b = 30, c = 10 (discordant = 40 >= 25).
    statistic = (|30 - 10| - 1)^2 / (30 + 10) = 19^2 / 40 = 361 / 40 = 9.025
    """
    # Construct synthetic predictions
    y_true = ["1"] * 50
    # 30 cases where A is right, B is wrong
    pred_a_1 = ["1"] * 30
    pred_b_1 = ["0"] * 30
    # 10 cases where A is wrong, B is right
    pred_a_2 = ["0"] * 10
    pred_b_2 = ["1"] * 10
    # 10 cases where both are right
    pred_a_3 = ["1"] * 10
    pred_b_3 = ["1"] * 10

    ya = pred_a_1 + pred_a_2 + pred_a_3
    yb = pred_b_1 + pred_b_2 + pred_b_3

    res = mcnemar_test(y_true, ya, yb)
    assert res["method"] == "chi_squared_edwards"
    assert res["contingency_table"]["a_only"] == 30
    assert res["contingency_table"]["b_only"] == 10
    assert np.isclose(res["statistic"], 9.025)
    assert res["p_value"] < 0.01
    assert res["significant_at_05"] is True


def test_holm_bonferroni_correction_hand_calculated():
    """Verify Holm-Bonferroni step-down correction against known values.

    Input p-values: [0.01, 0.04, 0.03]
    m = 3 tests.
    Sorted:
      Rank 0 (p = 0.01): adjusted = min(1.0, 0.01 * 3) = 0.03
      Rank 1 (p = 0.03): adjusted = min(1.0, 0.03 * 2) = 0.06 -> max(0.03, 0.06) = 0.06
      Rank 2 (p = 0.04): adjusted = min(1.0, 0.04 * 1) = 0.04 -> max(0.06, 0.04) = 0.06
    Original indices:
      p=0.01 -> 0.03
      p=0.04 -> 0.06
      p=0.03 -> 0.06
    """
    p_vals = [0.01, 0.04, 0.03]
    adj = holm_bonferroni_correction(p_vals)

    assert np.isclose(adj[0]["adjusted_p"], 0.03)
    assert np.isclose(adj[1]["adjusted_p"], 0.06)
    assert np.isclose(adj[2]["adjusted_p"], 0.06)
    assert adj[0]["significant_at_05"] is True
    assert adj[1]["significant_at_05"] is False
    assert adj[2]["significant_at_05"] is False


def test_wilcoxon_identical_distributions():
    """Verify Wilcoxon test returns p=1.0 for identical scores."""
    scores = [1.0, 2.0, 3.0, 4.0]
    res = wilcoxon_test(scores, scores)
    assert res["p_value"] == 1.0
    assert res["significant_at_05"] is False


def test_bootstrap_ci_coverage():
    """Verify bootstrap CI bounds are consistent."""
    y_true = ["A"] * 50 + ["B"] * 50
    y_pred = ["A"] * 40 + ["B"] * 10 + ["A"] * 10 + ["B"] * 40

    boot = compute_bootstrap_ci(y_true, y_pred, n_resamples=200, seed=42)
    assert boot["ci_lower"] <= boot["point_estimate"] <= boot["ci_upper"]
    assert boot["std_err"] > 0.0


def test_paired_bootstrap_difference():
    """Verify paired bootstrap difference bounds on better vs worse model."""
    y_true = ["A"] * 100
    y_pred_good = ["A"] * 90 + ["B"] * 10  # 90% correct
    y_pred_bad = ["A"] * 50 + ["B"] * 50  # 50% correct

    diff = compute_paired_bootstrap_difference(
        y_true, y_pred_good, y_pred_bad, n_resamples=200, seed=42
    )
    assert diff["diff_point"] > 0.0
    assert diff["ci_lower"] > 0.0
    assert diff["excludes_zero"] is True


def test_calibration_ece_hand_calculated():
    """Verify ECE calculation on a simple 2-bin scenario.

    Bin 1: 2 samples with confidence 0.2, both incorrect -> acc=0, conf=0.2, diff=0.2, weight=2/4
    Bin 2: 2 samples with confidence 0.8, both correct -> acc=1.0, conf=0.8, diff=0.2, weight=2/4
    Expected ECE = 0.5 * 0.2 + 0.5 * 0.2 = 0.20
    """
    y_true = ["A", "A", "A", "A"]
    y_pred = ["B", "B", "A", "A"]  # first 2 wrong, last 2 right
    confs = [0.2, 0.2, 0.8, 0.8]

    ece_res = compute_expected_calibration_error(y_true, y_pred, confs, n_bins=2)
    assert np.isclose(ece_res["ece"], 0.20)
