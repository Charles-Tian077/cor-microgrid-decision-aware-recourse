"""Causal forecast fusion and physically sequential microgrid replay.

All input tables are immutable. Pilot tunes nothing on the test split; the
full phase selects the fusion weight on validation and evaluates a frozen test.
The 00:00 procurement plan is inherited, so this studies Q4 recourse only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("COR_DATA_DIR", str(HERE.parent / "ML基线" / "数据集" / "csv")))
OUT = HERE / "结果"
sys.path.insert(0, str(HERE))
from q4_core import Config, solve_window

DT = 1.0 / 6.0
RELEASES = (0, 6, 12, 18)
FEATURES = ["month", "weekday", "is_weekend", "is_holiday", "is_makeup_workday",
            "sin_slot", "cos_slot", "sin_doy", "cos_doy", "price_fixed_yuan_per_kwh",
            "price_lag_1d", "price_lag_7d", "price_lag_28d",
            "price_same_slot_mean_7d", "price_same_slot_mean_28d", "price_same_slot_std_28d"]
BASELINE = (0.0, 0.5, 1.0)  # price fusion, rolling gain, reserve multiplier
SETTINGS = (
    BASELINE,
    (0.0, 0.25, 1.0), (0.0, 0.75, 1.0), (0.0, 1.0, 1.0),
    (0.0, 0.5, 0.0), (0.0, 0.5, 0.5), (0.0, 0.5, 1.5),
    (0.5, 0.5, 1.0), (1.0, 0.5, 1.0),
    (0.5, 0.25, 0.5), (0.5, 0.5, 0.5),
)
CFG = replace(Config(), solver_time_limit_sec=60.0)


def read_data():
    obs = pd.read_csv(DATA / "03_观测_10min.csv.gz", usecols=["date", "slot_index", "load_kw", "pv_kw", "price_yuan_per_kwh"])
    q2 = pd.read_csv(DATA / "05_继承输入_Q2.csv.gz", usecols=["date", "slot_index", "q2_planned_grid_kwh", "charge_kwh", "discharge_kwh", "soc_start_kwh", "soc_end_kwh"])
    rel = pd.read_csv(DATA / "06_发布时刻输入.csv.gz", usecols=["date", "release_hour", "slot_index", "net_point_kw", "reserve20_kw", "price_point_yuan_per_kwh", "latest_observation_slot", "history_end_date"])
    ml = pd.read_csv(DATA / "10_ML训练表_10min.csv.gz", usecols=["date", "slot_index", "split", "price_yuan_per_kwh"] + FEATURES)
    for name, frame, key in (("obs", obs, ["date", "slot_index"]), ("q2", q2, ["date", "slot_index"]), ("rel", rel, ["date", "release_hour", "slot_index"]), ("ml", ml, ["date", "slot_index"])):
        if frame.empty or frame.duplicated(key).any():
            raise ValueError(f"{name}: empty/duplicate key")
    if len(obs) != 365 * 144 or len(q2) != 334 * 144 or len(ml) != len(q2) or len(rel) != 334 * 360:
        raise ValueError("Unexpected day/slot/release coverage")
    if rel["latest_observation_slot"].gt(rel["release_hour"] * 6).any() or (pd.to_datetime(rel.history_end_date) >= pd.to_datetime(rel.date)).any():
        raise ValueError("Released-input information leakage")
    if not np.isfinite(obs[["load_kw", "pv_kw", "price_yuan_per_kwh"]].to_numpy()).all():
        raise ValueError("Nonfinite observations")
    if (obs[["load_kw", "pv_kw"]] < 0).any().any() or (obs.price_yuan_per_kwh <= 0).any():
        raise ValueError("Physical bounds violated")
    if not np.isfinite(q2.select_dtypes(include="number").to_numpy()).all() or not np.isfinite(rel.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite Q2/release input")
    dates = sorted(q2.date.unique())
    if len(dates) != 334 or dates[0] != "2025-02-01" or dates[-1] != "2025-12-31":
        raise ValueError("Unexpected evaluation dates")
    return obs, q2, rel, ml, dates


def fit_price(train: pd.DataFrame):
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                 max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
                 random_state=20261018, early_stopping=False)
    x = train[FEATURES].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Missing/nonfinite causal price features")
    model.fit(x, train.price_yuan_per_kwh.to_numpy(float))
    return model


def make_price_predictions(ml: pd.DataFrame):
    dev = ml[ml.split == "development"]
    devval = ml[ml.split.isin(["development", "validation"])]
    mdl_dev = fit_price(dev)
    mdl_test = fit_price(devval)
    out = ml[["date", "slot_index"]].copy()
    x = ml[FEATURES].to_numpy(float)
    val = ml.split.to_numpy() == "test"
    pred = np.empty(len(ml))
    pred[~val] = mdl_dev.predict(x[~val])
    pred[val] = mdl_test.predict(x[val])
    out["hgb_price"] = np.maximum(pred, 0.05)
    out["forecast_error"] = pred - ml.price_yuan_per_kwh.to_numpy(float)
    return out


def causal_recourse(net_kwh: float, q: float, desired_c: float, desired_u: float, soc: float):
    """At the current 10-min slot use only current measured net demand and SOC."""
    cap = CFG.max_power_kw * DT
    c = min(max(float(desired_c), 0.0), cap, (CFG.soc_max_kwh - soc) / CFG.charge_efficiency)
    u = min(max(float(desired_u), 0.0), cap, (soc + CFG.charge_efficiency * c - CFG.soc_min_kwh) * CFG.discharge_efficiency)
    if c > 1e-7 and u > 1e-7:
        both = min(c, u / CFG.charge_efficiency**2)
        c -= both
        u -= both * CFG.charge_efficiency**2
    gap = net_kwh + c - u - q
    if gap > 0:
        red = min(c, gap)
        c -= red
        gap -= red
        extra = min(max(cap-u, 0.0), max((soc + CFG.charge_efficiency*c - CFG.soc_min_kwh)*CFG.discharge_efficiency-u, 0.0), gap)
        u += extra
        gap -= extra
    elif gap < 0:
        red = min(u, -gap)
        u -= red
        gap += red
        extra = min(max(cap-c, 0.0), max((CFG.soc_max_kwh-soc+u/CFG.discharge_efficiency)/CFG.charge_efficiency-c, 0.0), -gap)
        c += extra
        gap += extra
    emergency = max(gap, 0.0)
    spill = max(-gap, 0.0)
    soc_next = soc + CFG.charge_efficiency*c-u/CFG.discharge_efficiency
    if not (CFG.soc_min_kwh-1e-5 <= soc_next <= CFG.soc_max_kwh+1e-5):
        raise AssertionError("SOC bounds")
    if c > 1e-6 and u > 1e-6:
        raise AssertionError("Simultaneous charge/discharge")
    balance = q+u+emergency-c-spill-net_kwh
    if abs(balance) > 1e-5:
        raise AssertionError(f"Balance residual {balance}")
    return soc_next, emergency, spill, c, u, abs(balance)


def one_day(date, alpha, gain, reserve_scale, soc0, tables, strategy):
    obs, q2, rel, mlp = tables
    o = obs.loc[date].sort_index()
    plan = q2.loc[date].sort_index()
    releases = rel.loc[date]
    learned = mlp.loc[date].sort_index().hgb_price.to_numpy(float)
    actual = o.price_yuan_per_kwh.to_numpy(float)
    net = (o.load_kw.to_numpy(float)-o.pv_kw.to_numpy(float))*DT
    q_base = plan.q2_planned_grid_kwh.to_numpy(float)
    if len(o) != 144 or len(plan) != 144 or len(learned) != 144:
        raise ValueError(f"Incomplete day {date}")
    q_final = np.zeros(144); emergency = np.zeros(144); spill = np.zeros(144)
    solved = 0; solve_time = 0.0; max_resid = 0.0
    soc = float(soc0)
    for ridx, release in enumerate(RELEASES):
        start = release*6
        stop = RELEASES[ridx+1]*6 if ridx+1 < len(RELEASES) else 144
        part = releases.loc[release].sort_index()
        if len(part) != 144-start or part.index.min() != start+1 or part.index.max() != 144:
            raise ValueError(f"Release horizon {date} {release}")
        if release == 0:
            q = q_base[start:]; c = plan.charge_kwh.to_numpy(float)[start:]; u = plan.discharge_kwh.to_numpy(float)[start:]
        else:
            baseline_price = part.price_point_yuan_per_kwh.to_numpy(float)
            bias = np.median(actual[max(0,start-6):start]-learned[max(0,start-6):start])
            h = np.arange(1, 145-start)*DT
            learned_release = np.maximum(learned[start:]+bias*np.exp(-h/4.0), 0.05)
            price = np.maximum((1-alpha)*baseline_price+alpha*learned_release, 0.05)
            target = (part.net_point_kw.to_numpy(float)+reserve_scale*part.reserve20_kw.to_numpy(float))*DT
            sol = solve_window(target, price, soc, CFG, baseline_grid_kwh=q_base[start:])
            q = q_base[start:]+gain*(sol["grid_kwh"]-q_base[start:])
            c = (1-gain)*plan.charge_kwh.to_numpy(float)[start:]+gain*sol["charge_kwh"]
            u = (1-gain)*plan.discharge_kwh.to_numpy(float)[start:]+gain*sol["discharge_kwh"]
            solved += 1; solve_time += sol["solver_time_sec"]
            max_resid = max(max_resid, sol["max_eq_residual_kwh"])
        for slot in range(start, stop):
            local = slot-start
            q_final[slot] = q[local]
            soc, emergency[slot], spill[slot], _, _, balance = causal_recourse(net[slot], q[local], c[local], u[local], soc)
            max_resid = max(max_resid, balance)
    up = np.maximum(q_final-q_base, 0.0)
    down = np.maximum(q_base-q_final, 0.0)
    plan_cost = float(actual @ q_base)
    adjustment_cost = float(actual @ (1.5*up+0.5*down))
    emergency_cost = float(actual @ emergency*5.0)
    return dict(date=date, split=strategy, alpha=alpha, gain=gain, reserve_scale=reserve_scale, plan_cost_yuan=plan_cost,
        adjustment_cost_yuan=adjustment_cost, emergency_cost_yuan=emergency_cost,
        total_cost_yuan=plan_cost+adjustment_cost+emergency_cost,
        planned_grid_kwh=float(q_base.sum()), adjusted_grid_kwh=float(q_final.sum()),
        emergency_kwh=float(emergency.sum()), spill_kwh=float(spill.sum()),
        emergency_day=int(emergency.sum()>1e-6), soc_initial_kwh=soc0,
        soc_final_kwh=soc, solver_calls=solved, solver_time_sec=solve_time,
        max_balance_residual_kwh=max_resid)


def prep_index(obs, q2, rel, mlp):
    return (obs.set_index(["date", "slot_index"]), q2.set_index(["date", "slot_index"]),
            rel.set_index(["date", "release_hour", "slot_index"]), mlp.set_index(["date", "slot_index"]))


def run_dates(dates, settings, tables, label, independent_start=False):
    rows=[]
    for alpha,gain,reserve_scale in settings:
        soc = CFG.initial_soc_kwh
        for i,date in enumerate(dates):
            if independent_start:
                soc = CFG.initial_soc_kwh
            row=one_day(date,alpha,gain,reserve_scale,soc,tables,label)
            rows.append(row)
            soc=row["soc_final_kwh"]
            if (i+1)%20==0:
                print(label,alpha,gain,reserve_scale,i+1,len(dates),flush=True)
    return pd.DataFrame(rows)


def summary(daily):
    metrics=[]
    for (label, alpha,gain,reserve_scale),p in daily.groupby(["split","alpha","gain","reserve_scale"]):
        metrics.append(dict(split=label,alpha=alpha,gain=gain,reserve_scale=reserve_scale,days=len(p),total_cost_yuan=p.total_cost_yuan.sum(),
            mean_daily_cost_yuan=p.total_cost_yuan.mean(),emergency_kwh=p.emergency_kwh.sum(),
            emergency_days=p.emergency_day.sum(),adjusted_grid_kwh=p.adjusted_grid_kwh.sum(),
            max_balance_residual_kwh=p.max_balance_residual_kwh.max(),solver_calls=p.solver_calls.sum(),
            solver_time_sec=p.solver_time_sec.sum()))
    return pd.DataFrame(metrics)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("pilot","full","stress"), required=True)
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    obs,q2,rel,ml,dates=read_data()
    pp=make_price_predictions(ml)
    forecast=pp.merge(ml[["date","slot_index","split","price_yuan_per_kwh"]],on=["date","slot_index"],validate="one_to_one")
    fm=forecast.groupby("split").apply(lambda x: pd.Series({"mae_yuan_per_kwh":np.abs(x.forecast_error).mean(),"rmse_yuan_per_kwh":np.sqrt(np.mean(x.forecast_error**2))}),include_groups=False).reset_index()
    fm.to_csv(OUT/"price_forecast_metrics.csv",index=False)
    tables=prep_index(obs,q2,rel,pp)
    if args.phase == "stress":
        verdict=json.loads((OUT/"final_verdict.json").read_text(encoding="utf-8"))
        chosen=tuple(float(x) for x in verdict["selected_alpha_gain_reserve"])
        test_dates=[d for d in dates if "2025-10-01" <= d <= "2025-12-31"]
        rows=[]
        cases=(("unshocked",1.0,1.0,1.0),("load_plus_10pct",1.1,1.0,1.0),
               ("pv_minus_20pct",1.0,0.8,1.0),("price_plus_20pct",1.0,1.0,1.2),
               ("joint_shock",1.1,0.8,1.2))
        for label,lf,pvf,pf in cases:
            alt=obs.copy()
            alt["load_kw"]*=lf;alt["pv_kw"]*=pvf;alt["price_yuan_per_kwh"]*=pf
            tab=prep_index(alt,q2,rel,pp)
            result=run_dates(test_dates,(BASELINE,chosen),tab,label,False)
            result["stress_case"]=label
            rows.append(result)
        stressed=pd.concat(rows,ignore_index=True)
        stressed.to_csv(OUT/"stress_daily.csv",index=False)
        summary(stressed).to_csv(OUT/"stress_summary.csv",index=False)
        print("Stress evaluation completed",flush=True)
        return
    if args.phase == "pilot":
        pilot_dates=[d for d in dates if "2025-07-01" <= d <= "2025-07-14"]
        daily=run_dates(pilot_dates,SETTINGS,tables,"pilot_validation",False)
        daily.to_csv(OUT/"pilot_daily.csv",index=False)
        summ=summary(daily)
        summ.to_csv(OUT/"pilot_summary.csv",index=False)
        mask=(summ.alpha==BASELINE[0])&(summ.gain==BASELINE[1])&(summ.reserve_scale==BASELINE[2])
        base=float(summ.loc[mask,"total_cost_yuan"].iloc[0])
        candidates=summ[(~mask)&(summ.total_cost_yuan<base-1e-4)].sort_values("total_cost_yuan")
        recommendation=[float(candidates.iloc[0][c]) for c in ("alpha","gain","reserve_scale")] if len(candidates) else None
        (OUT/"pilot_decision.json").write_text(json.dumps({"baseline_cost_yuan":base,"improving_setting":recommendation,
             "continue_full":recommendation is not None,"pilot_days":len(pilot_dates),"elapsed_sec":time.perf_counter()-t0},indent=2),encoding="utf-8")
        print(summ.to_string(index=False),flush=True)
        return
    # Chronological held-out validation. 14 pilot days are part of validation,
    # so full selection is still validation-only. Test remains untouched.
    val_dates=[d for d in dates if "2025-07-01" <= d <= "2025-09-30"]
    val=run_dates(val_dates,SETTINGS,tables,"validation",False)
    val.to_csv(OUT/"validation_daily.csv",index=False)
    vs=summary(val); vs.to_csv(OUT/"validation_summary.csv",index=False)
    chosen=vs.sort_values("total_cost_yuan").iloc[0]
    best=tuple(float(chosen[c]) for c in ("alpha","gain","reserve_scale"))
    # Replay the entire 334-day horizon with frozen alpha. The development
    # portion is descriptive/in-sample; validation selects; test is held out.
    full=run_dates(dates,(BASELINE,best) if best!=BASELINE else (BASELINE,),tables,"all_334_days",False)
    splits=pd.Series({d:("development" if d<"2025-07-01" else "validation" if d<"2025-10-01" else "test") for d in dates})
    full["chronological_split"]=full.date.map(splits)
    full.to_csv(OUT/"full_daily.csv",index=False)
    fs=full.groupby("chronological_split",group_keys=False).apply(summary,include_groups=False) if False else None
    ss=[]
    for split,p in full.groupby("chronological_split"):
        x=p.copy(); x["split"]=split;ss.append(summary(x))
    final=pd.concat(ss,ignore_index=True)
    final.to_csv(OUT/"full_summary_by_split.csv",index=False)
    test=full[full.chronological_split=="test"]
    base=test[(test.alpha==BASELINE[0])&(test.gain==BASELINE[1])&(test.reserve_scale==BASELINE[2])].set_index("date").total_cost_yuan
    selected=test[(test.alpha==best[0])&(test.gain==best[1])&(test.reserve_scale==best[2])].set_index("date").total_cost_yuan
    if len(base)!=92 or len(selected)!=92 or not base.index.equals(selected.index):
        raise AssertionError("Test pairing invalid")
    delta=(selected-base).to_numpy(float)
    rng=np.random.default_rng(20261018)
    # 7-day moving-block bootstrap of the *paired* daily saving.
    if best!=BASELINE:
        start=rng.integers(0,len(delta),size=(2000,math.ceil(len(delta)/7)))
        samples=np.concatenate([(start[:,:,None]+np.arange(7))%len(delta)],axis=2).reshape(2000,-1)[:,:len(delta)]
        boot=delta[samples].mean(axis=1)
        ci=np.quantile(boot,[0.025,0.975]).tolist()
        pval=float(wilcoxon(delta,zero_method="wilcox").pvalue)
    else:
        ci=[0.0,0.0];pval=None
    verdict={"selected_alpha_gain_reserve":best,"baseline_alpha_gain_reserve":BASELINE,"test_n_days":len(base),"test_mean_daily_cost_delta_yuan_improved_minus_baseline":float(delta.mean()),
             "test_total_cost_delta_yuan":float(delta.sum()),"test_paired_block_bootstrap_95ci_mean_daily_delta_yuan":ci,
             "test_wilcoxon_p_value_exploratory":pval,"elapsed_sec":time.perf_counter()-t0,
             "note":"Validation-selected alpha; 92-day frozen test. No claim of cross-site generalization."}
    (OUT/"final_verdict.json").write_text(json.dumps(verdict,indent=2),encoding="utf-8")
    print(json.dumps(verdict,indent=2),flush=True)


if __name__=="__main__":
    main()
