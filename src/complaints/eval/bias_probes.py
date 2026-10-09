"""Bias and Robustness Probes for Model Fairness Evaluation.

Implements counterfactual perturbation probes to test whether models
exhibit undesirable sensitivity to protected attributes:
1. Demographic name swaps (gender, ethnicity)
2. Location swaps (city, state)
3. Typo noise injection (random character insertions/deletions)

Generates reports/bias_report.md with flip rates and statistical summaries.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from complaints.config import get_project_root, get_seed, setup_logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Counterfactual swap dictionaries
# ──────────────────────────────────────────────────────────
MALE_NAMES = ["James", "John", "Robert", "Michael", "William", "David"]
FEMALE_NAMES = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan"]
NAMES_A = [
    "James",
    "William",
    "John",
    "Robert",
    "Charles",
    "Thomas",
    "Mary",
    "Elizabeth",
    "Margaret",
    "Dorothy",
    "Patricia",
    "Susan",
]
NAMES_B = [
    "DeShawn",
    "Jamal",
    "Terrence",
    "Darnell",
    "Tyrone",
    "Marquis",
    "Latoya",
    "Keisha",
    "Tamika",
    "Aisha",
    "Ebony",
    "Shaniqua",
]
CITIES = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose",
]
STATES = [
    "California",
    "Texas",
    "Florida",
    "New York",
    "Pennsylvania",
    "Illinois",
    "Ohio",
    "Georgia",
    "North Carolina",
    "Michigan",
]


def _name_swap(
    text: str, source_names: list[str], target_names: list[str], rng: random.Random
) -> str:
    """Swap any occurrences of source names with randomly chosen target names."""
    result = text
    for name in source_names:
        if name in result:
            replacement = rng.choice(target_names)
            result = result.replace(name, replacement, 1)
    return result


def _location_swap(text: str, rng: random.Random) -> str:
    """Swap city/state references with a different random location."""
    result = text
    for city in CITIES:
        if city in result:
            new_city = rng.choice([c for c in CITIES if c != city])
            result = result.replace(city, new_city, 1)
            break
    for state in STATES:
        if state in result:
            new_state = rng.choice([s for s in STATES if s != state])
            result = result.replace(state, new_state, 1)
            break
    return result


def _inject_typos(text: str, rng: random.Random, rate: float = 0.02) -> str:
    """Inject random character-level typos at the given rate."""
    chars = list(text)
    result = []
    for ch in chars:
        r = rng.random()
        if r < rate and ch.isalpha():
            # 33% chance each: delete, duplicate, swap with adjacent
            action = rng.choice(["delete", "duplicate", "substitute"])
            if action == "delete":
                continue
            elif action == "duplicate":
                result.append(ch)
                result.append(ch)
            else:
                # substitute with nearby key
                result.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        else:
            result.append(ch)
    return "".join(result)


def perturb_name_gender(text: str, seed: int | None = None) -> str:
    """Perturb male names in text to female names."""
    rng = random.Random(seed or get_seed())
    return _name_swap(text, MALE_NAMES, FEMALE_NAMES, rng)


def perturb_name_ethnicity(text: str, seed: int | None = None) -> str:
    """Perturb Caucasian-associated names in text to Black/Hispanic names."""
    rng = random.Random(seed or get_seed())
    return _name_swap(text, NAMES_A, NAMES_B, rng)


def perturb_location(text: str, seed: int | None = None) -> str:
    """Perturb city/state locations in text."""
    rng = random.Random(seed or get_seed())
    return _location_swap(text, rng)


def perturb_typos(text: str, seed: int | None = None, rate: float = 0.03) -> str:
    """Inject typos into text at given rate."""
    rng = random.Random(seed or get_seed())
    return _inject_typos(text, rng, rate=rate)


def generate_counterfactuals(
    df: pd.DataFrame,
    perturbation: str,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate counterfactual variants of complaint narratives.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'complaint_id' and 'narrative' columns.
    perturbation : str
        One of 'name_gender', 'name_ethnicity', 'location', 'typo'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'narrative' replaced by perturbed version and
        'perturbation_type' column added.
    """
    rng = random.Random(seed or get_seed())
    df_out = df.copy()

    perturbed_narratives = []
    for narrative in df["narrative"]:
        text = str(narrative)
        if perturbation == "name_gender":
            text = _name_swap(text, MALE_NAMES, FEMALE_NAMES, rng)
        elif perturbation == "name_ethnicity":
            text = _name_swap(text, NAMES_A, NAMES_B, rng)
        elif perturbation == "location":
            text = _location_swap(text, rng)
        elif perturbation == "typo":
            text = _inject_typos(text, rng, rate=0.02)
        else:
            raise ValueError(f"Unknown perturbation: {perturbation}")
        perturbed_narratives.append(text)

    df_out["narrative"] = perturbed_narratives
    df_out["perturbation_type"] = perturbation
    return df_out


def compute_flip_rate(
    original_preds: pd.Series,
    perturbed_preds: pd.Series,
) -> dict[str, Any]:
    """Compute prediction flip rate between original and perturbed inputs.

    Returns
    -------
    dict with 'flip_rate', 'n_flips', 'n_total'.
    """
    assert len(original_preds) == len(perturbed_preds), "Prediction lengths must match"
    flips = (original_preds.values != perturbed_preds.values).sum()
    total = len(original_preds)
    return {
        "flip_rate": float(flips / total) if total > 0 else 0.0,
        "n_flips": int(flips),
        "n_total": total,
    }


def run_bias_probes(
    predict_fn: Any | None = None,
    sample_size: int = 200,
    perturbations: list[str] | None = None,
) -> Path:
    """Run all bias probes and generate reports/bias_report.md.

    Parameters
    ----------
    predict_fn : callable, optional
        Function(df) -> pd.Series of predicted labels. If None, uses
        a simple keyword baseline for demonstration.
    sample_size : int
        Number of samples from the eval set to probe.
    perturbations : list[str], optional
        Which perturbation types to run. Defaults to all four.

    Returns
    -------
    Path to the generated bias report.
    """
    root = get_project_root()
    eval_path = root / "data" / "gold" / "gold_eval_set.parquet"
    if not eval_path.exists():
        eval_path = root / "data" / "gold" / "test.parquet"
    df_eval = pd.read_parquet(eval_path).head(sample_size)

    if perturbations is None:
        perturbations = ["name_gender", "name_ethnicity", "location", "typo"]

    if predict_fn is None:
        # Default: use keyword baseline for bias probing
        from complaints.models.baselines import KeywordBaseline

        kb = KeywordBaseline()
        train_df = pd.read_parquet(root / "data" / "gold" / "train.parquet")
        kb.fit(train_df, train_df)

        def predict_fn(df: pd.DataFrame) -> pd.Series:
            preds_df = kb.predict(df)
            return preds_df["y_pred"]

    # Get original predictions
    original_preds = predict_fn(df_eval)

    results: list[dict[str, Any]] = []
    for pert in perturbations:
        df_perturbed = generate_counterfactuals(df_eval, pert)
        perturbed_preds = predict_fn(df_perturbed)
        flip_info = compute_flip_rate(original_preds, perturbed_preds)
        flip_info["perturbation"] = pert
        results.append(flip_info)
        logger.info(
            "Bias probe [%s]: flip_rate=%.4f (%d/%d)",
            pert,
            flip_info["flip_rate"],
            flip_info["n_flips"],
            flip_info["n_total"],
        )

    # Generate report
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "bias_report.md"

    lines = [
        "# Bias & Robustness Probe Report\n",
        "## Prediction Flip Rates Under Counterfactual Perturbations\n",
        f"**Eval samples:** {sample_size}\n",
        "| Perturbation | Flip Rate | Flips | Total |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['perturbation']} | {r['flip_rate']:.4f} | {r['n_flips']} | {r['n_total']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **name_gender**: Tests sensitivity to gendered first names. A high flip rate suggests gender bias.",
            "- **name_ethnicity**: Tests sensitivity to ethnicity-associated first names. A high flip rate suggests ethnic bias.",
            "- **location**: Tests sensitivity to geographic references. Moderate flip rates may be acceptable if location is a legitimate feature.",
            "- **typo**: Tests robustness to noisy input. High flip rates indicate fragile models.",
            "",
            "> **Note**: These probes use the keyword baseline model by default. "
            "Results will differ when run against LLM or transformer models.",
        ]
    )

    report_path.write_text("\n".join(lines))
    logger.info("Bias report saved to %s", report_path)
    return report_path


if __name__ == "__main__":
    setup_logging()
    run_bias_probes(sample_size=100)
