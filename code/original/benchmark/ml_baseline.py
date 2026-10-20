# -*- coding: utf-8 -*-
"""Learned day-ahead forecasting baselines for the microgrid dataset.

Purpose: give the paper an *independent, reproducible learning baseline* next to
the causal shrinkage forecaster used by the optimisation model, and quantify
whether a learned mapping from the same causal information set helps.

Protocol (identical to the optimisation experiments, no leakage):
    development 2025-02-01 .. 2025-06-30   150 days -> train
    validation  2025-07-01 .. 2025-09-30    92 days -> model selection
    test        2025-10-01 .. 2025-12-31    92 days -> one frozen evaluation
The test comparison uses models refitted on development+validation only.

Targets: day-ahead hourly-slot values of price (元/kWh), net load (kW),
load (kW) and PV (kW). Features are restricted to information available at
00:00 of the target day (plus the operational PV forecast release 0 for PV /
net-load tasks).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")   # avoid OpenMP oversubscription on 20 vCPU
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataset_lib as DL  # noqa: E402

BASE = HERE.parent
DATA = BASE / "数据"
OUT = BASE / "ML基线"
SEED = 20261017
NT = 144

CAL = ["month", "weekday", "is_weekend", "is_holiday", "is_makeup_workday",
       "sin_slot", "cos_slot", "sin_doy", "cos_doy"]

TASKS = {
    "price": {
        "target": "price_yuan_per_kwh",
        "features": CAL + ["price_fixed_yuan_per_kwh",
                           "price_lag_1d", "price_lag_7d", "price_lag_14d", "price_lag_28d",
                           "price_same_slot_mean_7d", "price_same_slot_mean_28d",
                           "price_same_slot_std_28d"],
        "baseline_col": None,
        "baseline_name": "causal_shrinkage_35d_14d",
        "clip_low": 0.05,
    },
    "net_load": {
        "target": "net_load_kw",
        "features": CAL + ["load_lag_1d", "load_lag_7d", "load_same_slot_mean_7d",
                           "load_same_slot_mean_28d", "pv_lag_1d", "pv_same_slot_mean_7d",
                           "net_lag_1d", "net_lag_7d", "net_same_slot_mean_7d",
                           "net_same_slot_mean_28d", "pv_fc_r00_kw"],
        "baseline_col": "net_lag_1d",
        "baseline_name": "persistence_24h",
        "clip_low": None,
    },
    "load": {
        "target": "load_kw",
        "features": CAL + ["load_lag_1d", "load_lag_7d", "load_same_slot_mean_7d",
                           "load_same_slot_mean_28d"],
        "baseline_col": "load_lag_1d",
        "baseline_name": "persistence_24h",
        "clip_low": 0.0,
    },
    "pv": {
        "target": "pv_kw",
        "features": CAL + ["pv_lag_1d", "pv_lag_7d", "pv_same_slot_mean_7d",
                           "pv_same_slot_mean_28d", "pv_fc_r00_kw"],
        "baseline_col": "pv_fc_r00_kw",
        "baseline_name": "operational_pv_forecast_r00",
        "clip_low": 0.0,
    },
}

# model-based references that are already part of the dataset, used as extra
# (not "learned") comparators where meaningful
EXTRA_BASELINES = {
    "net_load": {"load_forecast_kw": "q2_day_ahead_load_forecast",
                 "net_forecast_kw": "q2_day_ahead_net_forecast"},
    "load": {"load_forecast_kw": "q2_day_ahead_load_forecast"},
    "price": {"price_fc_r00_yuan_per_kwh": "causal_shrinkage_release_0"},
}


def load_data():
    z = np.load(DATA / "inputs.npz")
    dates = pd.DatetimeIndex(z["dates"])
    inputs = {
        "dates": dates, "load": z["load"], "pv": z["pv"], "price": z["price"],
        "fixed_price": z["fixed_price"],
        "pv_release": {r: z[f"pv_{r}"] for r in (0, 6, 12, 18)},
    }
    inh = pd.read_csv(DATA / "inherited_q2.csv", parse_dates=["date"])
    q2 = pd.read_csv(DATA / "派生" / "q2_forecast_inputs.csv", parse_dates=["date"])
    rel = pd.read_csv(DATA / "released_forecast_inputs.csv", parse_dates=["date"])
    return inputs, inh, q2, rel


def model_zoo():
    return {
        "ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0, random_state=SEED,
            early_stopping=False,
        ),
    }


def evaluate(frame, y_true, y_pred, target, model, split, rows):
    err = y_pred - y_true
    rows.append({
        "target": target, "model": model, "split": split, "n": int(len(err)),
        "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mbe": float(np.mean(err)),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    })


def daily_metrics(frame, y_true, y_pred, target, model):
    d = pd.DataFrame({"date": frame["date"].to_numpy(), "e": np.abs(y_pred - y_true),
                      "e2": (y_pred - y_true) ** 2})
    g = d.groupby("date").agg(mae=("e", "mean"), rmse=("e2", lambda x: math.sqrt(x.mean())),
                              n=("e", "size")).reset_index()
    g["target"], g["model"] = target, model
    return g


def dm_p(a, b):
    """Diebold-Mariano p-value on daily absolute-error series (A vs B)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return math.nan
    mean_d = d.mean()
    var = np.mean((d - mean_d) ** 2)
    if var <= 0:
        return math.nan
    stat = mean_d / math.sqrt(var / n)
    return float(2 * (1 - stats.norm.cdf(abs(stat))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, inh, q2, rel = load_data()
    table = DL.build_ml_table(
        inputs["dates"], inputs["load"], inputs["pv"], inputs["price"],
        inputs["fixed_price"], inputs["pv_release"], inh, q2, rel,
        [c for c in inh.columns if c.startswith("net_")],
    )
    table["date"] = pd.to_datetime(table["date"])
    splits = {s: table[table.split == s] for s in ["development", "validation", "test"]}
    summary_rows, daily_frames, selection = [], [], {}

    for target_key, spec in TASKS.items():
        target = spec["target"]
        feats = spec["features"]
        y = table[target].to_numpy(float)
        best = None
        val_scores = {}
        for name, model in model_zoo().items():
            model.fit(splits["development"][feats], splits["development"][target])
            pred_val = model.predict(splits["validation"][feats])
            if spec["clip_low"] is not None:
                pred_val = np.maximum(pred_val, spec["clip_low"])
            val_mae = float(np.mean(np.abs(pred_val - splits["validation"][target].to_numpy(float))))
            val_scores[name] = val_mae
            evaluate(splits["validation"], splits["validation"][target].to_numpy(float),
                     pred_val, target_key, name, "validation", summary_rows)
            daily_frames.append(daily_metrics(splits["validation"], splits["validation"][target].to_numpy(float),
                                              pred_val, target_key, name))
        best = min(val_scores, key=val_scores.get)
        selection[target_key] = {"selected_model": best, "validation_mae": val_scores}
        print(f"[{target_key}] validation MAE: " +
              ", ".join(f"{k}={v:.4f}" for k, v in val_scores.items()) + f" -> {best}", flush=True)

        # refit on development+validation, one frozen test evaluation
        train = pd.concat([splits["development"], splits["validation"]], ignore_index=True)
        model = model_zoo()[best]
        model.fit(train[feats], train[target])
        for split in ["development", "validation", "test"]:
            part = splits[split]
            pred = model.predict(part[feats])
            if spec["clip_low"] is not None:
                pred = np.maximum(pred, spec["clip_low"])
            evaluate(part, part[target].to_numpy(float), pred, target_key, f"{best} (dev+val refit)",
                     split, summary_rows)
            if split == "test":
                daily_frames.append(daily_metrics(part, part[target].to_numpy(float), pred,
                                                  target_key, f"{best} (dev+val refit)"))
                test_pred = pred.copy()

        # un-learned baselines on the same test rows
        test = splits["test"]
        y_test = test[target].to_numpy(float)
        base_frames = {}
        if spec["baseline_col"]:
            base_pred = test[spec["baseline_col"]].to_numpy(float)
            evaluate(test, y_test, base_pred, target_key, spec["baseline_name"], "test", summary_rows)
            daily_frames.append(daily_metrics(test, y_test, base_pred, target_key, spec["baseline_name"]))
            base_frames[spec["baseline_name"]] = base_pred
        for col, label in EXTRA_BASELINES.get(target_key, {}).items():
            if col not in test.columns:
                continue
            pred = test[col].to_numpy(float)
            if np.isnan(pred).any():
                continue
            evaluate(test, y_test, pred, target_key, label, "test", summary_rows)
            daily_frames.append(daily_metrics(test, y_test, pred, target_key, label))
            base_frames[label] = pred

        # DM test: learned model vs each baseline on daily MAE
        daily_model = [f for f in daily_frames if f is not None]
        ml_daily = pd.DataFrame(
            {"date": test["date"].to_numpy(), "e": np.abs(test_pred - y_test)}).groupby("date").e.mean()
        for label, pred in base_frames.items():
            b_daily = pd.DataFrame(
                {"date": test["date"].to_numpy(), "e": np.abs(pred - y_test)}).groupby("date").e.mean()
            p = dm_p(ml_daily.to_numpy(), b_daily.to_numpy())
            summary_rows.append({
                "target": target_key, "model": f"DM({best} vs {label})", "split": "test",
                "n": int(len(ml_daily)), "mae": p, "rmse": np.nan, "mbe": np.nan, "r2": np.nan,
            })
            print(f"  DM {best} vs {label}: p={p:.4f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary = summary[["target", "model", "split", "n", "mae", "rmse", "mbe", "r2"]]
    summary.to_csv(OUT / "指标_汇总.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_csv(
        OUT / "逐日指标.csv", index=False, encoding="utf-8-sig")
    (OUT / "模型选择.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "purpose": "independent learned day-ahead baselines on the same causal information set",
        "train": "development 2025-02-01..06-30 (150 days), test models refit on development+validation",
        "select": "validation 2025-07-01..09-30 (92 days)",
        "test": "2025-10-01..12-31 (92 days), frozen",
        "models": ["ridge(alpha=10)", "hist_gradient_boosting(max_iter=400,lr=0.05)"],
        "features_are_causal": True,
        "note": "price task uses only past prices + calendar + fixed tariff; PV/net-load tasks also use the operational PV forecast released at 00:00",
    }
    (OUT / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written to", OUT)


if __name__ == "__main__":
    main()
