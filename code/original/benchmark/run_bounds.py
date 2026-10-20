# -*- coding: utf-8 -*-
"""Value-of-stochastic-programming bounds for the day-ahead layer.

Adds the standard stochastic-programming benchmarks that a reviewer will ask for,
all on the same 334 real days and with the same LP formulation:

  WS   wait-and-see / perfect foresight: deterministic LP given the *realised*
       net load and price, storage can arbitrage with full knowledge;
  RP   recourse problem: the stochastic plan's realised cost (same replay rule as
       the risk-neutral arm of the main experiment);
  EV   expected-value plan: LP on the probability-weighted mean scenario;
  EEV  EV plan evaluated under the 11-scenario model (expected cost);
  VSS  = EEV - RP   (value of stochastic solution, in 元/day);
  EVPI = RP - WS    (expected value of perfect information, in 元/day).

Output: 全日期实验结果/随机规划价值边界.csv (+ _shards for resume)
"""
from __future__ import annotations

import sys
import time
import pickle
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
SH = OUT / "_bounds_shards"
_STATE = {}


def _init():
    inputs = R.load_prepared()
    pf = R.causal_prices(inputs, m.CFG)
    by = {d: g.sort_values("slot_index").reset_index(drop=True)
          for d, g in inputs["q2"].groupby("date")}
    _STATE.update(inputs=inputs, pf=pf, by=by)


def _one_day(date):
    date = pd.Timestamp(date)
    inputs, by = _STATE["inputs"], _STATE["by"]
    i = inputs["dates"].get_loc(date)
    day = by[date]
    net_actual = (inputs["load"][i] - inputs["pv"][i])[None, :]
    price_actual = inputs["price"][i][None, :]
    soc0 = float(day.soc_start_kwh.iloc[0])
    terminal = float(day.soc_end_kwh.iloc[-1])
    cfg = m.CFG

    # perfect foresight (WS)
    ws = R.solve_joint(net_actual, price_actual, np.array([1.0]), soc0, terminal, cfg, 0.0)
    ws_cost = float(ws["expected_cost"])

    # stochastic model inputs
    ns = m.net_scenarios_from_q2(day)
    ps = m.price_scenarios_for_day(i, _STATE["pf"], cfg)
    w = m.scenario_weights(m.QUANTILES)
    rp = R.solve_joint(ns, ps, w, soc0, terminal, cfg, 0.0)
    rp_expected = float(rp["expected_cost"])
    rp_realized = float(R.execute(rp["grid_kwh"], i, inputs, day, cfg)["total_cost_yuan"])

    # expected-value plan (mean scenario), evaluated inside the scenario model
    net_mean = (w[:, None] * ns).sum(axis=0)[None, :]
    price_mean = (w[:, None] * ps).sum(axis=0)[None, :]
    ev = R.solve_joint(net_mean, price_mean, np.array([1.0]), soc0, terminal, cfg, 0.0)
    eev_sol = R.solve_joint(ns, ps, w, soc0, terminal, cfg, 0.0, fixed=ev["grid_kwh"])
    eev = float(eev_sol["expected_cost"])
    ev_realized = float(R.execute(ev["grid_kwh"], i, inputs, day, cfg)["total_cost_yuan"])

    return {
        "date": date, "split": R.split_of(date), "month": date.month,
        "ws_perfect_foresight_cost_yuan": ws_cost,
        "rp_expected_cost_yuan": rp_expected,
        "rp_realized_cost_yuan": rp_realized,
        "ev_plan_expected_cost_yuan": eev,
        "ev_plan_realized_cost_yuan": ev_realized,
        "vss_yuan": eev - rp_expected,
        "evpi_yuan": rp_realized - ws_cost,
        "solver_time_sec": (ws["solver_time_sec"] + rp["solver_time_sec"]
                            + ev["solver_time_sec"] + eev_sol["solver_time_sec"]),
    }


def _run_chunk(chunk):
    return [_one_day(d) for d in chunk]


def main(workers=6):
    OUT.mkdir(exist_ok=True)
    SH.mkdir(exist_ok=True)
    tic = time.perf_counter()
    cal = pd.read_csv(ROOT / "数据" / "evaluation_calendar.csv", parse_dates=["date"])
    dates = list(pd.DatetimeIndex(cal["date"]).sort_values())
    chunks = [list(c) for c in np.array_split(np.array(dates), workers) if len(c)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futs = {}
        for k, chunk in enumerate(chunks):
            pkl = SH / f"bounds_{k:02d}.pkl"
            if pkl.exists():
                rows.extend(pickle.loads(pkl.read_bytes()))
                continue
            futs[pool.submit(_run_chunk, chunk)] = (k, pkl)
        for fut in futs:
            k, pkl = futs[fut]
            part = fut.result()
            pkl.write_bytes(pickle.dumps(part, protocol=4))
            rows.extend(part)
            print(f"bounds shard {k} done, elapsed={time.perf_counter()-tic:.0f}s", flush=True)
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    frame.to_csv(OUT / "随机规划价值边界.csv", index=False, encoding="utf-8-sig")
    print(frame[["ws_perfect_foresight_cost_yuan", "rp_realized_cost_yuan",
                 "ev_plan_realized_cost_yuan", "vss_yuan", "evpi_yuan"]].mean().round(3).to_string())
    print("written", OUT / "随机规划价值边界.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
