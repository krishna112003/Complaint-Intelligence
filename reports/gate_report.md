# LLMOps Regression Gate Report: [PASSED]

**Gate Outcome:** `PASS`
**Total Checks:** 4
**Passed Checks:** 4

## Evaluated Guardrails

| Check Name | Status | Candidate | Baseline | Threshold | Detail |
|---|---|---|---|---|---|
| Macro-F1 Quality Drop | `PASS` | 1.0000 | 1.0000 | Drop <= 2.00% | Observed drop of 0.0000 (limit: 0.0200) |
| JSON Schema Validity | `PASS` | 1.0000 | 1.0000 | Valid Rate >= 98.00% | Observed 100.00% valid rate (min: 98.00%) |
| Inference Cost Ceiling | `PASS` | 0.0000 | 0.0000 | Increase <= 20.0% | Observed candidate cost $0.0000/1k vs baseline $0.0000/1k |
| Bias / Robustness Flip Rate | `PASS` | 0.0000 | 0.0000 | Flip Rate <= 5.00% | Observed flip rate 0.00% (max allowed: 5.00%) |

## Governance Decision

> [!NOTE]
> **Deployment Approved**: All candidate metrics satisfy quality, schema validity, cost, and fairness guardrails.
