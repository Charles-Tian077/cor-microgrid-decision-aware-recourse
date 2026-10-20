"""Independent real-data policy/forecast baselines while GPU training queues."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from evaluate_expansion import matched_history_hgb, metrics, run_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ["COR_DATA_DIR"] = str(args.data.resolve())
    import cor_q4_experiment as engine
    obs, q2, rel, ml, dates = engine.read_data()
    args.out.mkdir(parents=True, exist_ok=True)
    hist = matched_history_hgb(obs, ml, engine.FEATURES)
    raw = pd.read_csv(args.data / "10_ML训练表_10min.csv.gz",
                      usecols=["date", "slot_index", "split", "price_yuan_per_kwh"])
    raw = raw.merge(hist, on=["date", "slot_index"], validate="one_to_one")
    summary = raw[raw.split.isin(["validation", "test"])].groupby("split").apply(
        lambda x: pd.Series({"days": x.date.nunique(),
                             "mae": (x.hgb_history_price - x.price_yuan_per_kwh).abs().mean()}),
        include_groups=False).reset_index()
    summary.to_csv(args.out / "cpu_history_hgb_forecast.csv", index=False)
    tables = engine.prep_index(obs, q2, rel,
        hist.rename(columns={"hgb_history_price": "hgb_price"}))
    rows = []
    for split, begin, end in (("validation", "2025-07-01", "2025-09-30"),
                              ("test", "2025-10-01", "2025-12-31")):
        ds = [d for d in dates if begin <= d <= end]
        soc = float(q2.loc[(q2.date == ds[0]) & (q2.slot_index == 1), "soc_start_kwh"].iloc[0])
        for label, alpha, gain, reserve in (
            ("no_update", 0, 0, 1), ("half_gain", 0, 0.5, 1),
            ("full_gain", 0, 1, 1), ("no_reserve", 0, 1, 0),
            ("hgb_history_half_fusion", 0.5, 1, 1)):
            daily = run_policy(ds, label, alpha, gain, reserve, tables, soc, engine)
            daily["chronological_split"] = split
            rows.append(daily)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.out / "cpu_baseline_daily.csv", index=False)
    pd.DataFrame([{"period": period, "policy": label, **metrics(part)}
                  for (period, label), part in out.groupby(["chronological_split", "split"])]).to_csv(
        args.out / "cpu_baseline_summary.csv", index=False)
    print("CPU_BASELINE_COMPLETE", len(out), flush=True)


if __name__ == "__main__":
    main()
