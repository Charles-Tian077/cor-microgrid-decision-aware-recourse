# -*- coding: utf-8 -*-
"""Re-solve the (few) jobs that hit the 45 s HiGHS time limit during the parallel run.

The parallel whole-year run used 6 worker processes; under that load a handful of
LPs hit the 45 s wall-clock limit and were correctly *not* replaced by zeros.
They are re-solved here serially (single process, default numerics), the daily
tables are patched, and the summary tables + markdown report are regenerated with
the original aggregation code.

Run after 脚本/run_full_parallel.py.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "core"))
import run_experiments as R  # noqa: E402
import q4_model as m  # noqa: E402

OUT = HERE.parent / "全日期实验结果"


def main():
    failed = pd.read_csv(OUT / "failed_jobs.csv", parse_dates=["date"])
    if failed.empty:
        print("no failed jobs")
        return
    inputs = R.load_prepared()
    pf = R.causal_prices(inputs, m.CFG)
    by = {d: g.sort_values("slot_index").reset_index(drop=True)
          for d, g in inputs["q2"].groupby("date")}
    risk = pd.read_csv(OUT / "risk_daily.csv", parse_dates=["date"])
    resolution = pd.read_csv(OUT / "resolution_daily.csv", parse_dates=["date"])
    new_risk, new_res = [], []
    for _, job in failed.iterrows():
        date = pd.Timestamp(job["date"])
        i = inputs["dates"].get_loc(date)
        day = by[date]
        ns = m.net_scenarios_from_q2(day)
        ps = m.price_scenarios_for_day(i, pf, m.CFG)
        soc0 = float(day.soc_start_kwh.iloc[0])
        terminal = float(day.soc_end_kwh.iloc[-1])
        meta = {"date": date, "split": R.split_of(date), "month": date.month}
        try:
            if job["experiment"] == "risk":
                rho = float(job["parameter"])
                sol = R.solve_joint(ns, ps, m.scenario_weights(m.QUANTILES), soc0, terminal, m.CFG, rho)
                exe = R.execute(sol["grid_kwh"], i, inputs, day, m.CFG)
                new_risk.append({
                    **meta, "risk_weight": rho,
                    "realized_cost_yuan": exe["total_cost_yuan"],
                    "realized_emergency_kwh": float(exe["emergency"].sum()),
                    "scenario_expected_cost_yuan": sol["expected_cost"],
                    "scenario_cvar90_yuan": sol["cvar90"],
                    "legacy_top2_mean_yuan": float(np.mean(np.sort(sol["scenario_cost"])[-2:])),
                    "planned_grid_kwh": float(sol["grid_kwh"].sum()),
                    "solver_time_sec": sol["solver_time_sec"],
                    "max_eq_residual_kwh": max(sol["eq_residual"], exe["eq_residual"]),
                    "terminal_residual_kwh": max(sol["terminal_residual"], exe["terminal_residual"]),
                    "max_simultaneous_kwh": max(sol["simultaneous_kwh"], exe["simultaneous_kwh"]),
                    "status": "optimal_resolved",
                })
            else:
                count = int(float(job["parameter"]))
                ids = R.SUBSETS[count]
                sol = R.solve_joint(ns[ids], ps[ids], m.scenario_weights(m.QUANTILES[ids]),
                                    soc0, terminal, m.CFG, 0.02)
                exe = R.execute(sol["grid_kwh"], i, inputs, day, m.CFG)
                new_res.append({
                    **meta, "scenario_count": count,
                    "realized_cost_yuan": exe["total_cost_yuan"],
                    "realized_emergency_kwh": float(exe["emergency"].sum()),
                    "solver_time_sec": sol["solver_time_sec"],
                    "scenario_cvar90_yuan": sol["cvar90"],
                    "status": "optimal_resolved",
                })
            print(f"resolved {date.date()} {job['experiment']} {job['parameter']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"STILL FAILED {date.date()} {job['experiment']} {job['parameter']}: {exc}", flush=True)
    if new_risk:
        risk = pd.concat([risk, pd.DataFrame(new_risk)], ignore_index=True)
        risk = risk.drop_duplicates(["date", "risk_weight"], keep="last").sort_values(
            ["date", "risk_weight"]).reset_index(drop=True)
        risk.to_csv(OUT / "risk_daily.csv", index=False, encoding="utf-8-sig")
    if new_res:
        resolution = pd.concat([resolution, pd.DataFrame(new_res)], ignore_index=True)
        resolution = resolution.drop_duplicates(["date", "scenario_count"], keep="last").sort_values(
            ["date", "scenario_count"]).reset_index(drop=True)
        resolution.to_csv(OUT / "resolution_daily.csv", index=False, encoding="utf-8-sig")
    (OUT / "failed_jobs.csv").write_text(
        "date,split,month,experiment,parameter,error\n", encoding="utf-8-sig")
    R.summarize(OUT)
    print("patched; summaries regenerated", flush=True)


if __name__ == "__main__":
    main()
