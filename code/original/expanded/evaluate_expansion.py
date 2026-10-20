"""Frozen, physically sequential policy comparison for COR question four.

Validation chooses only the predeclared price-fusion weight. All policies
start each evaluation block from the same inherited Q2 SOC and use identical
observations, settlement rules, and release-time forecasts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def metrics(frame: pd.DataFrame) -> dict:
    return {
        "days": len(frame),
        "total_cost_yuan": float(frame.total_cost_yuan.sum()),
        "plan_cost_yuan": float(frame.plan_cost_yuan.sum()),
        "adjustment_cost_yuan": float(frame.adjustment_cost_yuan.sum()),
        "emergency_cost_yuan": float(frame.emergency_cost_yuan.sum()),
        "emergency_kwh": float(frame.emergency_kwh.sum()),
        "emergency_days": int(frame.emergency_day.sum()),
        "adjusted_grid_kwh": float(frame.adjusted_grid_kwh.sum()),
        "solver_calls": int(frame.solver_calls.sum()),
        "solver_time_sec": float(frame.solver_time_sec.sum()),
        "max_balance_residual_kwh": float(frame.max_balance_residual_kwh.max()),
        "final_soc_kwh": float(frame.soc_final_kwh.iloc[-1]),
    }


def run_policy(dates, label, alpha, gain, reserve, tables, initial_soc, engine):
    rows = []
    soc = float(initial_soc)
    for i, date in enumerate(dates):
        row = engine.one_day(date, alpha, gain, reserve, soc, tables, label)
        if not np.isfinite(row["total_cost_yuan"]) or row["max_balance_residual_kwh"] > 1e-5:
            raise AssertionError(f"Invalid policy replay on {date}")
        rows.append(row)
        soc = row["soc_final_kwh"]
        if (i + 1) % 25 == 0:
            print(f"{label}: {i+1}/{len(dates)}", flush=True)
    return pd.DataFrame(rows)


def paired_ci(candidate: pd.DataFrame, baseline: pd.DataFrame, seed: int) -> dict:
    a = candidate.set_index("date").sort_index()
    b = baseline.set_index("date").sort_index()
    if len(a) != 92 or not a.index.equals(b.index):
        raise AssertionError("Frozen test pairing failed")
    diff = (a.total_cost_yuan - b.total_cost_yuan).to_numpy(float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    block = 7
    starts = rng.integers(0, n, size=(4000, int(np.ceil(n / block))))
    ids = (starts[:, :, None] + np.arange(block)) % n
    means = diff[ids.reshape(4000, -1)[:, :n]].mean(axis=1)
    return {
        "candidate_minus_baseline_total_yuan": float(diff.sum()),
        "candidate_minus_baseline_mean_daily_yuan": float(diff.mean()),
        "paired_7day_block_bootstrap_95ci_mean_daily_yuan":
            np.quantile(means, [0.025, 0.975]).tolist(),
        "days_candidate_lower_by_over_1_yuan": int((diff < -1).sum()),
        "days_candidate_higher_by_over_1_yuan": int((diff > 1).sum()),
        "days_near_tie": int((np.abs(diff) <= 1).sum()),
    }


def matched_history_hgb(obs, ml, features):
    """Strong tree baseline with the same 28-day same-slot history as GRU."""
    o = obs.sort_values(["date", "slot_index"])
    m = ml.sort_values(["date", "slot_index"]).reset_index(drop=True)
    calendar = np.sort(o.date.unique())
    dates = {d: i for i, d in enumerate(calendar)}
    di = m.date.map(dates).to_numpy(int)
    slot = m.slot_index.to_numpy(int) - 1
    if di.min() < 28:
        raise ValueError("Insufficient causal history")
    prices = o.price_yuan_per_kwh.to_numpy(float).reshape(len(calendar), 144)
    lag = prices[di[:, None] - np.arange(28, 0, -1)[None, :], slot[:, None]]
    x = np.concatenate((m[features].to_numpy(float), lag), axis=1)
    y = m.price_yuan_per_kwh.to_numpy(float)
    split = m.split.to_numpy()
    pred = np.full(len(m), np.nan)
    params = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                  min_samples_leaf=30, l2_regularization=1.0,
                  early_stopping=False, random_state=20261018)
    for train_names, destination in ((("development",), "validation"),
                                      (("development", "validation"), "test")):
        model = HistGradientBoostingRegressor(**params)
        model.fit(x[np.isin(split, train_names)], y[np.isin(split, train_names)])
        pred[split == destination] = np.maximum(model.predict(x[split == destination]), 0.05)
    if not np.isfinite(pred[np.isin(split, ["validation", "test"])]).all():
        raise AssertionError("History HGB predictions missing")
    return m[["date", "slot_index"]].assign(hgb_history_price=pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gpu-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["COR_DATA_DIR"] = str(args.data.resolve())
    import cor_q4_experiment as engine

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    obs, q2, releases, ml, dates = engine.read_data()
    ml_extra = pd.read_csv(args.data / "10_ML训练表_10min.csv.gz",
                           usecols=["date", "slot_index", "split", "price_yuan_per_kwh",
                                    "price_fc_r00_yuan_per_kwh"])
    gru = pd.read_csv(args.gpu_results / "gru_predictions.csv")
    if gru.duplicated(["date", "slot_index"]).any() or len(gru) != 334 * 144:
        raise ValueError("GPU forecast key mismatch")
    joined = ml_extra.merge(gru[["date", "slot_index", "gru_price", "ensemble_std"]],
                            on=["date", "slot_index"], validate="one_to_one")
    hgb = engine.make_price_predictions(ml)
    joined = joined.merge(hgb, on=["date", "slot_index"], validate="one_to_one")
    matched = matched_history_hgb(obs, ml, engine.FEATURES)
    joined = joined.merge(matched, on=["date", "slot_index"], validate="one_to_one")
    if len(joined) != 334 * 144:
        raise AssertionError("Forecast join changed sample count")
    eval_rows = joined[joined.split.isin(["validation", "test"])].copy()
    forecast_metrics = []
    for split, subset in eval_rows.groupby("split"):
        actual = subset.price_yuan_per_kwh.to_numpy(float)
        for label, col in (("causal_shrinkage", "price_fc_r00_yuan_per_kwh"),
                           ("hist_gradient_boosting", "hgb_price"),
                           ("hgb_same_28day_history", "hgb_history_price"),
                           ("gru_3seed_ensemble", "gru_price")):
            pred = subset[col].to_numpy(float)
            if not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite forecast {split}/{label}")
            forecast_metrics.append({"split": split, "forecaster": label,
                "slots": len(actual), "mae_yuan_per_kwh": float(np.mean(np.abs(actual - pred))),
                "rmse_yuan_per_kwh": float(np.sqrt(np.mean((actual - pred) ** 2)))})
    pd.DataFrame(forecast_metrics).to_csv(args.out / "forecast_comparison.csv", index=False)

    val_dates = [d for d in dates if "2025-07-01" <= d <= "2025-09-30"]
    test_dates = [d for d in dates if "2025-10-01" <= d <= "2025-12-31"]
    if len(val_dates) != 92 or len(test_dates) != 92:
        raise AssertionError("Unexpected chronological split")
    frame_for = {
        "causal": joined[["date", "slot_index", "hgb_price"]].copy(),
        "hgb": joined[["date", "slot_index", "hgb_price"]].copy(),
        "hgb_history": joined[["date", "slot_index", "hgb_history_price"]].rename(
            columns={"hgb_history_price": "hgb_price"}).copy(),
        "gru": joined[["date", "slot_index", "gru_price"]].rename(columns={"gru_price": "hgb_price"}).copy(),
    }
    tables = {key: engine.prep_index(obs, q2, releases, frame) for key, frame in frame_for.items()}
    first_states = {}
    for split, ds in (("validation", val_dates), ("test", test_dates)):
        state = q2.loc[(q2.date == ds[0]) & (q2.slot_index == 1), "soc_start_kwh"]
        if len(state) != 1:
            raise AssertionError(f"Missing inherited initial SOC for {split}")
        first_states[split] = float(state.iloc[0])
    predeclared = [
        ("no_update", "causal", 0.0, 0.0, 1.0),
        ("half_gain", "causal", 0.0, 0.5, 1.0),
        ("full_gain", "causal", 0.0, 1.0, 1.0),
        ("no_reserve", "causal", 0.0, 1.0, 0.0),
    ]
    for model in ("hgb", "hgb_history", "gru"):
        for alpha in (0.25, 0.5, 1.0):
            predeclared.append((f"{model}_alpha_{alpha:g}", model, alpha, 1.0, 1.0))
    val_outputs = []
    for label, model, alpha, gain, reserve in predeclared:
        daily = run_policy(val_dates, label, alpha, gain, reserve,
                           tables[model], first_states["validation"], engine)
        daily["forecaster"] = model
        val_outputs.append(daily)
    validation = pd.concat(val_outputs, ignore_index=True)
    validation.to_csv(args.out / "policy_validation_daily.csv", index=False)
    validation_summary = pd.DataFrame([{"policy": name, **metrics(part)}
                  for name, part in validation.groupby("split", sort=False)])
    validation_summary.to_csv(args.out / "policy_validation_summary.csv", index=False)
    cost = validation_summary.set_index("policy").total_cost_yuan
    selected = {}
    for model in ("hgb", "hgb_history", "gru"):
        choices = ["full_gain"] + [f"{model}_alpha_{a:g}" for a in (0.25, 0.5, 1.0)]
        selected[model] = min(choices, key=lambda x: cost[x])
    (args.out / "policy_selection.json").write_text(json.dumps({
        "selection_split": "2025-07-01 to 2025-09-30",
        "test_split": "2025-10-01 to 2025-12-31",
        "same_inherited_q2_initial_soc_kwh": first_states,
        "predeclared_validation_candidates": [x[0] for x in predeclared],
        "selected_price_fusion_policies": selected,
        "note": "Test labels are not consulted in selection; previous study had inspected these months, so this is retrospective, not pristine external validation."
    }, indent=2), encoding="utf-8")
    to_test = predeclared[:4] + [next(x for x in predeclared if x[0] == name)
                                 for name in sorted(set(selected.values())) if name != "full_gain"]
    test_outputs = []
    for label, model, alpha, gain, reserve in to_test:
        daily = run_policy(test_dates, label, alpha, gain, reserve,
                           tables[model], first_states["test"], engine)
        daily["forecaster"] = model
        test_outputs.append(daily)
    test = pd.concat(test_outputs, ignore_index=True)
    test.to_csv(args.out / "policy_test_daily.csv", index=False)
    test_summary = pd.DataFrame([{"policy": name, **metrics(part)}
                  for name, part in test.groupby("split", sort=False)])
    test_summary.to_csv(args.out / "policy_test_summary.csv", index=False)
    paired = {}
    base = test[test.split == "half_gain"]
    for i, label in enumerate(test_summary.policy):
        if label != "half_gain":
            paired[label] = paired_ci(test[test.split == label], base, 20261018 + i)
    (args.out / "paired_test_comparison.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    monthly = test.assign(month=test.date.str[:7]).groupby(["month", "split"], as_index=False).agg(
        days=("date", "nunique"), cost_yuan=("total_cost_yuan", "sum"),
        emergency_kwh=("emergency_kwh", "sum"), emergency_days=("emergency_day", "sum"))
    monthly.to_csv(args.out / "policy_test_monthly.csv", index=False)
    manifest = {"elapsed_sec": time.perf_counter() - started,
                "input_rows": {"observations": len(obs), "q2": len(q2), "releases": len(releases), "ml": len(ml)},
                "forecast_metrics": forecast_metrics,
                "selected": selected,
                "test_days": 92,
                "max_test_balance_residual_kwh": float(test.max_balance_residual_kwh.max())}
    (args.out / "expansion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
