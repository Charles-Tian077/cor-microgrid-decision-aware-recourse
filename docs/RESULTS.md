# Verified reference results

The numerical tables below are recomputed from the supplied daily records. All policies share the same dates and inherited charge. Timing fields are excluded from equality checks.

| Policy | Cost (CNY) | Adjustment (CNY) | Emergency cost (CNY) | Emergency (kWh) | Emergency days |
|---|---:|---:|---:|---:|---:|
| no_update | 4,390,480.89 | 0.00 | 232,382.51 | 37,365.291842 | 40 |
| half_gain | 4,304,804.70 | 31,775.64 | 114,930.68 | 19,572.715506 | 35 |
| full_gain | 4,280,800.67 | 54,620.49 | 68,081.80 | 11,859.595170 | 27 |
| no_reserve | 4,281,151.55 | 54,206.33 | 68,846.83 | 12,018.496938 | 28 |

## Artifact-to-result mapping

| Evidence | Primary artifact |
|---|---|
| Forecast MAE/RMSE | `results/reference/expanded/forecast_comparison.csv` |
| Candidate selection | `results/reference/expanded/policy_validation_summary.csv`, `policy_selection.json` |
| Main comparison / reserve ablation | `results/reference/expanded/policy_test_daily.csv`, `policy_test_summary.csv` |
| Paired inference | `results/reference/expanded/paired_test_comparison.json`; independently recomputed in `validation/verification/verification.json` |
| 334-day continuation | `results/reference/initial/full_daily.csv` |
| Stress experiments | `results/reference/initial/stress_daily.csv` |
| GRU fitting | `results/reference/expanded/gru_training_history.csv`, `gru_manifest.json` |
| Reconstructed weights | `models/regenerated_cpu/*.pt`, `checkpoint_provenance.json` |
| Rerun evidence | `validation/*/comparison.json` |

## Scope of the successful tests

- Frozen four-policy test: 368 policy-days, 2,208 LP subproblems.
- Expanded validation plus test: 1,196 + 368 policy-days, 9,384 LP subproblems.
- Initial grid and continuous replay: 1,012 + 668 policy-days, 10,080 LP subproblems.
- Stress comparison: 920 policy-days, 5,520 LP subproblems.
- Three-seed CPU retraining regenerated six checkpoints; both validation and test inference were checked.
- End-to-end replay with the reconstructed predictions preserves the selected policies and main test results.
- MATLAB sources and original figure artifacts are supplied, but current regeneration was blocked before script execution by MathWorks service error 5001. Numerical reproduction does not require MATLAB.

These are successful reproduction checks of one dataset and protocol, not additional independent experimental replications.
