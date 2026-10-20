# -*- coding: utf-8 -*-
"""Full-coverage (334 real days) Q4 experiment runner, parallel across real dates.

This runner *reuses the unmodified numerical code* of ``run_experiments.py``
(same LP formulation, same parameters, same causal protocol); the only change is
that the 334 real evaluation dates are split across worker processes so that the
whole-year experiment finishes in minutes instead of ~30 min serial.

Outputs go to ``<package>/全日期实验结果`` with exactly the same table schema as
the 66-day pilot package, so the two can be compared row-for-row.

Usage:
    py -3 run_full_parallel.py --workers 6
    py -3 run_full_parallel.py --workers 6 --resume
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "core"))

import run_experiments as R  # noqa: E402
import q4_model as m  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "全日期实验结果"
SHARDS = OUT / "_shards"

_STATE = {}


def _init_worker():
    inputs = R.load_prepared()
    pf = R.causal_prices(inputs, m.CFG)
    net = m.prepare_net_forecasts(inputs, m.CFG)
    by = {
        d: g.sort_values("slot_index").reset_index(drop=True)
        for d, g in inputs["q2"].groupby("date")
    }
    _STATE.update(inputs=inputs, pf=pf, net=net, by=by)


def _run_dates(dates):
    """Same computation as the body of run_experiments.run_pilot, for a date subset."""
    inputs, pf, net, by = _STATE["inputs"], _STATE["pf"], _STATE["net"], _STATE["by"]
    risk_rows, res_rows, rolling_rows, errors = [], [], [], []
    for date in dates:
        date = pd.Timestamp(date)
        i = inputs["dates"].get_loc(date)
        day = by[date]
        ns = m.net_scenarios_from_q2(day)
        ps = m.price_scenarios_for_day(i, pf, m.CFG)
        soc0 = float(day.soc_start_kwh.iloc[0])
        terminal = float(day.soc_end_kwh.iloc[-1])
        meta = {"date": date, "split": R.split_of(date), "month": date.month}
        cached = {}
        for rho in R.RHOS:
            try:
                weights = m.scenario_weights(m.QUANTILES)
                sol = R.solve_joint(ns, ps, weights, soc0, terminal, m.CFG, rho)
                exe = R.execute(sol["grid_kwh"], i, inputs, day, m.CFG)
                row = {
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
                    "status": "optimal",
                }
                risk_rows.append(row)
                cached[rho] = row
            except Exception as exc:  # noqa: BLE001
                errors.append({**meta, "experiment": "risk", "parameter": rho, "error": str(exc)})
        for count, ids in R.SUBSETS.items():
            try:
                if count == 11 and 0.02 in cached:
                    r = cached[0.02]
                    row = {
                        **meta, "scenario_count": 11,
                        "realized_cost_yuan": r["realized_cost_yuan"],
                        "realized_emergency_kwh": r["realized_emergency_kwh"],
                        "solver_time_sec": r["solver_time_sec"],
                        "scenario_cvar90_yuan": r["scenario_cvar90_yuan"],
                        "status": "reused_exact_11_rho02",
                    }
                else:
                    sol = R.solve_joint(
                        ns[ids], ps[ids], m.scenario_weights(m.QUANTILES[ids]),
                        soc0, terminal, m.CFG, 0.02,
                    )
                    exe = R.execute(sol["grid_kwh"], i, inputs, day, m.CFG)
                    row = {
                        **meta, "scenario_count": count,
                        "realized_cost_yuan": exe["total_cost_yuan"],
                        "realized_emergency_kwh": float(exe["emergency"].sum()),
                        "solver_time_sec": sol["solver_time_sec"],
                        "scenario_cvar90_yuan": sol["cvar90"],
                        "status": "optimal",
                    }
                res_rows.append(row)
            except Exception as exc:  # noqa: BLE001
                errors.append({**meta, "experiment": "resolution", "parameter": count, "error": str(exc)})
        for gain in R.GAINS:
            baseline = None
            for reserve in R.RESERVES:
                try:
                    if gain == 0 and baseline is not None:
                        r = dict(baseline)
                    else:
                        r = R.rolling_one(day, i, inputs, net, pf, m.CFG, gain, reserve, terminal)
                        if gain == 0:
                            baseline = dict(r)
                    rolling_rows.append({**meta, "gain": gain, "reserve_quantile": reserve, **r, "status": "optimal"})
                except Exception as exc:  # noqa: BLE001
                    errors.append({**meta, "experiment": "rolling", "parameter": f"{gain},{reserve}", "error": str(exc)})
    return {
        "risk": pd.DataFrame(risk_rows),
        "resolution": pd.DataFrame(res_rows),
        "rolling": pd.DataFrame(rolling_rows),
        "errors": pd.DataFrame(errors, columns=["date", "split", "month", "experiment", "parameter", "error"]),
    }


def _split_evenly(dates, workers):
    bounds = np.linspace(0, len(dates), workers + 1).round().astype(int)
    return [dates[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()

    # 1) Price-forecast grid over all 334 days (cheap, deterministic, serial).
    inputs = R.load_prepared()
    cfg = m.CFG
    if not args.resume or not (OUT / "forecast_daily.csv").exists():
        R.forecast_experiment(inputs, cfg, OUT)
    else:
        print("[resume] reuse forecast_daily.csv", flush=True)

    # 2) Whole-year risk / resolution / control experiments, date-parallel.
    cal = pd.read_csv(ROOT / "数据" / "evaluation_calendar.csv", parse_dates=["date"])
    dates = list(pd.DatetimeIndex(cal["date"]).sort_values())
    assert len(dates) == 334, len(dates)
    chunks = _split_evenly(dates, args.workers)
    print(f"Dispatching {len(dates)} real dates over {len(chunks)} workers", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {}
        for k, chunk in enumerate(chunks):
            pkl = SHARDS / f"shard_{k:02d}.pkl"
            if args.resume and pkl.exists():
                print(f"[resume] shard {k} loaded from {pkl.name}", flush=True)
                parts.append(pickle.loads(pkl.read_bytes()))
                continue
            futures[pool.submit(_run_dates, chunk)] = (k, pkl, chunk)
        done = 0
        for fut in futures:
            k, pkl, chunk = futures[fut]
            res = fut.result()
            pkl.write_bytes(pickle.dumps(res, protocol=4))
            parts.append(res)
            done += 1
            print(
                f"shard {k} done ({len(chunk)} dates, +{done}/{len(futures)}) "
                f"elapsed={time.perf_counter()-tic:.0f}s",
                flush=True,
            )

    risk = pd.concat([p["risk"] for p in parts], ignore_index=True).sort_values(
        ["date", "risk_weight"]).reset_index(drop=True)
    resolution = pd.concat([p["resolution"] for p in parts], ignore_index=True).sort_values(
        ["date", "scenario_count"]).reset_index(drop=True)
    rolling = pd.concat([p["rolling"] for p in parts], ignore_index=True).sort_values(
        ["date", "gain", "reserve_quantile"]).reset_index(drop=True)
    errors = pd.concat([p["errors"] for p in parts], ignore_index=True) if any(
        len(p["errors"]) for p in parts) else pd.DataFrame(
        columns=["date", "split", "month", "experiment", "parameter", "error"])

    for frame, name in [(risk, "risk_daily.csv"), (resolution, "resolution_daily.csv"),
                        (rolling, "rolling_daily.csv"), (errors, "failed_jobs.csv")]:
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    print(f"raw tables written: risk={len(risk)} resolution={len(resolution)} rolling={len(rolling)} failed={len(errors)}", flush=True)

    # 3) Summaries + markdown report, reusing the original aggregation code verbatim.
    R.summarize(OUT)
    meta = {
        "runner": "run_full_parallel.py",
        "workers": args.workers,
        "dates": len(dates),
        "elapsed_sec": round(time.perf_counter() - tic, 1),
        "risk_rows": int(len(risk)),
        "resolution_rows": int(len(resolution)),
        "rolling_rows": int(len(rolling)),
        "failed_rows": int(len(errors)),
        "engine": "scipy.optimize.linprog(method='highs')",
        "numerical_code": "unmodified copy of 扩展数据预实验/脚本/run_experiments.py + core/q4_model.py",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
