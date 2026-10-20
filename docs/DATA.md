# Data and provenance

The archive starts from the supplied, already-prepared 2026 CUMCM Problem C benchmark export. It does not claim an independently collected field dataset or reconstruct the competition organizers' original observations from another source.

| Input | Rows | Key | Role |
|---|---:|---|---|
| `03_观测_10min.csv.gz` | 52,560 | date, slot_index | Load/PV power and realized settlement price |
| `04_评估日历.csv.gz` | 334 | date | Evaluation and chronological-partition calendar |
| `05_继承输入_Q2.csv.gz` | 48,096 | date, slot_index | Fixed inherited purchases, storage reference, day-ahead quantities |
| `06_发布时刻输入.csv.gz` | 120,240 | date, release_hour, slot_index | Release-specific net-load target, reserve and decision prices |
| `07_光伏预报_10min.csv.gz` | 120,240 | date, release_hour, slot_index | Original released photovoltaic forecasts |
| `10_ML训练表_10min.csv.gz` | 48,096 | date, slot_index | Prepared features and separate prediction targets |

The consolidated workbook contains descriptions, a field dictionary, configuration, data-quality information, source provenance and auxiliary experiments. The CSV.GZ and NPZ exports preserve machine-readable precision. `data/export/导出清单与校验.csv` is the source export manifest; `SOURCE_MANIFEST.json` records the actual copied artifacts without modifying their contents.

Prepared feature precision follows the unchanged upstream `dataset_lib.py`: same-slot historical price means are rounded to four decimal places and the 28-day population standard deviation to six. Independent feature checks apply that documented export rounding after recomputing the strictly historical quantities. They do not alter the supplied CSV values.

## Time and units

- Local operating dates are used consistently; there is no cross-timezone join.
- Slot indices are 1–144: slot 1 is 00:00–00:10, slot 144 is 23:50–24:00.
- `load_kw` and `pv_kw` are power. Multiply by `1/6 h` before using energy balances.
- Stored energy, grid purchases, charging, discharge, emergency and surplus are in kWh.
- Prices are CNY/kWh. The prediction floor of 0.05 is not a floor on the observed price used for settlement.
- January is warm-up. Development is February–June (150 days), validation July–September (92), and test October–December (92).

## Information availability

Target columns must not be used as explanatory variables. The active price predictors use only the 16 explicitly listed features in `train_gpu.py` / `cor_q4_experiment.py`; the matched-history tree adds 28 strictly preceding same-slot prices. Later release columns can be missing before release by design. Do not forward-fill these from future values. Development rows in `gru_predictions.csv` are intentionally unscored and contain missing GRU predictions; validation and test predictions must be finite.

The release table covers remaining horizons of 144, 108, 72 and 36 periods per day, totaling 360 rows per day. Historical dates must precede the target day and the latest observed slot must not exceed the release boundary. Independent checks reconstruct all price lags and historical mean/std features directly from observations.

## Auxiliary exports

Scenario and earlier experiment tables are retained for completeness and provenance. They are not additional independent observations, and the current remaining-day LP does not optimize a CVaR term or use all exported scenarios. The original dataset description records local crossings in upstream quantile trajectories; that limitation must be retained when using those trajectories as fixed-quantile scenarios. It does not turn the active empirical reserve into a formal coverage guarantee.
