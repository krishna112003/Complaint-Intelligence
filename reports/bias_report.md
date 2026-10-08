# Bias & Robustness Probe Report

## Prediction Flip Rates Under Counterfactual Perturbations

**Eval samples:** 100

| Perturbation | Flip Rate | Flips | Total |
|---|---|---|---|
| name_gender | 0.0000 | 0 | 100 |
| name_ethnicity | 0.0000 | 0 | 100 |
| location | 0.0000 | 0 | 100 |
| typo | 0.0800 | 8 | 100 |

## Interpretation

- **name_gender**: Tests sensitivity to gendered first names. A high flip rate suggests gender bias.
- **name_ethnicity**: Tests sensitivity to ethnicity-associated first names. A high flip rate suggests ethnic bias.
- **location**: Tests sensitivity to geographic references. Moderate flip rates may be acceptable if location is a legitimate feature.
- **typo**: Tests robustness to noisy input. High flip rates indicate fragile models.

> **Note**: These probes use the keyword baseline model by default. Results will differ when run against LLM or transformer models.