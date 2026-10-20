# -*- coding: utf-8 -*-
"""Q4 real-date expansion and chronological retrospective pilot.

Usage: python run_experiments.py --stage all --scale pilot
The normalized data and bundled q4_model.py make pilot/full reruns portable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
import q4_model as m

DATA = ROOT / "数据"
RHOS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
GAINS = [0.0, 0.25, 0.50, 0.75, 1.0]
RESERVES = [0.0, 0.10, 0.20, 0.30, 0.40]
SUBSETS = {3: [0, 5, 10], 5: [0, 2, 5, 8, 10], 7: [0, 1, 3, 5, 7, 9, 10], 11: list(range(11))}
SEED = 20261013


def split_of(date):
    date = pd.Timestamp(date)
    if date < pd.Timestamp("2025-02-01"):
        return "warmup"
    if date < pd.Timestamp("2025-07-01"):
        return "development"
    if date < pd.Timestamp("2025-10-01"):
        return "validation"
    return "test"


def csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def causal_prices(inputs, cfg):
    """Use the known Appendix-1 tariff, never future January prices, on day zero.

    Subsequent same-slot history, weekday weights and shrinkage match the source.
    Raw bands are diagnostic; the final band is recalibrated from past errors.
    """
    actual, dates = inputs["price"], inputs["dates"]
    n, t = actual.shape
    point, residual = np.empty((n, t)), np.empty((n, t))
    for i, day in enumerate(dates):
        ids = np.arange(max(0, i - cfg.residual_window_days), i)
        if not len(ids):
            point[i] = np.maximum(inputs["fixed_price"], .05)
        else:
            weekdays = np.array([dates[j].weekday() for j in ids])
            w = np.exp(-(i - ids) / cfg.price_decay_days)
            w *= np.where(weekdays == day.weekday(), 1.55, 1.0)
            w *= np.where((weekdays >= 5) == (day.weekday() >= 5), 1.18, .88)
            point[i] = np.maximum((w / w.sum()) @ actual[ids], .05)
        residual[i] = actual[i] - point[i]
    raw = {"point": point, "residual": residual, "training_days": np.minimum(np.arange(n), cfg.residual_window_days)}
    return m.shrink_price_forecasts_causally(raw, actual, inputs["fixed_price"])


def weighted_cvar(cost, weights, alpha=.90):
    """Exact discrete CVaR: fractionally include the last tail-probability bin."""
    cost, weights = np.asarray(cost, float), np.asarray(weights, float)
    weights = weights / weights.sum()
    remaining, total = 1.0 - alpha, 0.0
    for i in np.argsort(cost)[::-1]:
        take = min(float(weights[i]), remaining)
        total += take * cost[i]
        remaining -= take
        if remaining <= 1e-14:
            break
    return total / (1.0 - alpha)


def solve_joint(net, prices, weights, soc0, terminal, cfg, rho=0., fixed=None):
    """Source two-stage LP, plus identical terminal-energy anchors for all arms."""
    net, prices = np.asarray(net, float), np.asarray(prices, float)
    weights = np.asarray(weights, float); weights /= weights.sum()
    assert net.shape == prices.shape and net.shape[1] == 144
    assert np.isfinite(net).all() and np.isfinite(prices).all() and (prices > 0).all()
    layout = m.build_joint_layout(len(net), cfg)
    ns, t = net.shape
    terminal_rows = coo_matrix((np.ones(ns), (np.arange(ns),
        [layout.soc.start + s * (t + 1) + t for s in range(ns)])), shape=(ns, layout.n)).tocsr()
    aeq = vstack([layout.a_eq, terminal_rows]).tocsr()
    beq = np.zeros(layout.a_eq.shape[0])
    for s in range(ns):
        beq[s * (t + 1)] = soc0
    beq[ns * (t + 1):] = (net * cfg.dt_hours).reshape(-1)
    beq = np.r_[beq, np.full(ns, terminal)]
    obj = np.zeros(layout.n)
    obj[layout.q] = (1-rho) * (weights @ prices)
    obj[layout.emergency] = ((1-rho) * weights[:, None] * cfg.emergency_multiplier * prices).ravel()
    obj[layout.zeta] = rho
    obj[layout.u] = rho * weights / (1-cfg.cvar_alpha)
    rr, cc, vv = [], [], []
    for s in range(ns):
        for k in range(t):
            rr.extend([s,s]); cc.extend([layout.q.start+k,layout.emergency.start+s*t+k])
            vv.extend([prices[s,k],cfg.emergency_multiplier*prices[s,k]])
        rr.extend([s,s]); cc.extend([layout.zeta,layout.u.start+s]); vv.extend([-1.,-1.])
    aub = coo_matrix((vv,(rr,cc)),shape=(ns,layout.n)).tocsr()
    bounds = list(layout.bounds)
    if fixed is not None:
        for k, val in enumerate(fixed):
            bounds[layout.q.start+k] = (float(val),float(val))
    opts = {"time_limit":cfg.solver_time_limit_sec,"primal_feasibility_tolerance":1e-7,"dual_feasibility_tolerance":1e-7}
    tic = time.perf_counter()
    primary = linprog(obj,A_eq=aeq,b_eq=beq,A_ub=aub,b_ub=np.zeros(ns),bounds=bounds,method="highs",options=opts)
    if not primary.success:
        raise RuntimeError(primary.message)
    tol = max(1e-5,abs(primary.fun)*1e-10)
    secondary_obj = np.zeros(layout.n)
    secondary_obj[layout.charge] = np.repeat(weights,t)
    secondary_obj[layout.discharge] = np.repeat(weights,t)
    secondary_obj[layout.spill] = 1e-6*np.repeat(weights,t)
    secondary = linprog(secondary_obj,A_eq=aeq,b_eq=beq,
        A_ub=vstack([aub,coo_matrix(obj.reshape(1,-1))]).tocsr(),
        b_ub=np.r_[np.zeros(ns),primary.fun+tol],bounds=bounds,method="highs",options=opts)
    if not secondary.success:
        raise RuntimeError(secondary.message)
    x=secondary.x
    charge=x[layout.charge].reshape(ns,t); discharge=x[layout.discharge].reshape(ns,t)
    state=x[layout.soc].reshape(ns,t+1); emg=x[layout.emergency].reshape(ns,t)
    spill=x[layout.spill].reshape(ns,t)
    q=x[layout.q]
    costs=prices @ q + np.sum(cfg.emergency_multiplier*prices*emg,axis=1)
    balance=q[None,:]+discharge+emg-charge-spill-net*cfg.dt_hours
    recursion=state[:,1:]-state[:,:-1]-cfg.charge_efficiency*charge+discharge/cfg.discharge_efficiency
    return {"grid_kwh":q,"soc":state,"charge":charge,"discharge":discharge,"emergency":emg,"spill":spill,
        "scenario_cost":costs,"expected_cost":float(weights@costs),"cvar90":weighted_cvar(costs,weights),
        "solver_time_sec":time.perf_counter()-tic,"eq_residual":float(np.max(np.abs(aeq@x-beq))),
        "balance_residual":float(np.max(np.abs(balance))),"recursion_residual":float(np.max(np.abs(recursion))),
        "terminal_residual":float(np.max(np.abs(state[:,-1]-terminal))),
        "simultaneous_kwh":float(np.minimum(charge,discharge).max()),
        "cost_drift":float(obj@x-primary.fun),"status":"optimal"}


def execute(q, day_index, inputs, day, cfg):
    soc0=float(day.soc_start_kwh.iloc[0]); target=float(day.soc_end_kwh.iloc[-1])
    net=(inputs["load"][day_index]-inputs["pv"][day_index])[None,:]
    price=inputs["price"][day_index][None,:]
    out=solve_joint(net,price,np.array([1.]),soc0,target,cfg,rho=0,fixed=q)
    out["total_cost_yuan"]=float(price[0]@q+cfg.emergency_multiplier*price[0]@out["emergency"][0])
    return out


def prepare():
    cfg=m.Config(root=ROOT.parent,project_root=ROOT.parent.parent)
    # Dataclass path defaults are evaluated at definition time: override all input paths explicitly.
    project=ROOT.parent.parent
    cfg=replace(cfg,data_file=project/"数据预处理/数据结果/model_ready_features.csv",
        pv_forecast_file=project/"数据预处理/数据结果/pv_forecasts_10min_pchip.csv",
        q2_schedule_file=project/"第二问/数据结果/q2_schedule_10min.csv",
        q3_schedule_file=project/"第三问/数据结果/q3_schedule_10min.csv",
        template42=project/"附件/附件5/result4-2.xlsx",template43=project/"附件/附件5/result4-3.xlsx")
    inputs=m.load_inputs(cfg)
    import openpyxl
    wb=openpyxl.load_workbook(project/"附件/附件1.xlsx",read_only=True,data_only=True)
    profile=np.array([r[2] for r in list(wb.active.values)[1:]],float); wb.close()
    assert profile.shape==(144,)
    inputs["load_fc"][0]=profile  # Known attachment profile for the no-history initialization.
    DATA.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DATA/"inputs.npz",dates=inputs["dates"].to_numpy(dtype="datetime64[D]"),
        load=inputs["load"],pv=inputs["pv"],price=inputs["price"],fixed_price=inputs["fixed_price"],
        load_fc=inputs["load_fc"],**{f"pv_{r}":inputs["pv_release"][r] for r in cfg.release_hours})
    cols=["date","slot_index","planned_grid_kwh","soc_start_kwh","soc_end_kwh",*m.NET_QUANTILE_COLUMNS]
    q2=inputs["q2"][cols].copy()
    assert not q2.duplicated(["date","slot_index"]).any()
    csv(q2,DATA/"inherited_q2.csv")
    base=pd.read_csv(cfg.data_file)
    obs=base[["date_label","slot_index","timestamp_start","timestamp_end","load_kw","pv_kw", "price_variable_yuan_per_kwh"]].copy()
    obs.rename(columns={"date_label":"date","price_variable_yuan_per_kwh":"price_yuan_per_kwh"},inplace=True)
    obs["split"]=obs.date.map(split_of)
    assert len(obs)==52560 and not obs.duplicated(["date","slot_index"]).any()
    csv(obs,DATA/"observations_10min.csv")
    dates=inputs["dates"][31:]
    chosen=[]
    for month in sorted(dates.month.unique()):
        dd=dates[dates.month==month]
        chosen.extend(dd[np.round(np.linspace(0,len(dd)-1,6)).astype(int)])
    chosen=set(chosen)
    calendar=pd.DataFrame({"date":dates})
    calendar["split"]=calendar.date.map(split_of)
    calendar["pilot_selected"]=calendar.date.isin(chosen)
    calendar["selection_rule"]="six evenly spaced calendar dates per month; no outcome ranking"
    csv(calendar,DATA/"evaluation_calendar.csv")
    pf=causal_prices(inputs,cfg); net=m.prepare_net_forecasts(inputs,cfg)
    by={d:g.sort_values("slot_index") for d,g in inputs["q2"].groupby("date")}
    net_s=np.stack([m.net_scenarios_from_q2(by[d]) for d in dates])
    price_s=np.stack([m.price_scenarios_for_day(inputs["dates"].get_loc(d),pf,cfg) for d in dates])
    np.savez_compressed(DATA/"scenario_bank.npz",dates=dates.to_numpy(dtype="datetime64[D]"),
        quantiles=m.QUANTILES,weights=m.scenario_weights(m.QUANTILES),net_kw=net_s,price_yuan_per_kwh=price_s)
    pieces=[]
    for d in dates:
        i=inputs["dates"].get_loc(d)
        for release in cfg.release_hours:
            k=np.arange(release*6,144)
            pp,pu=m.corrected_price_forecast(i,release,inputs["price"][i],pf,cfg)
            reserve=m.empirical_reserve(net["residual"][release],i,cfg)
            pieces.append(pd.DataFrame({"date":d,"release_hour":release,"slot_index":k+1,
                "split":split_of(d),"net_point_kw":net["point"][release][i],"reserve20_kw":reserve,
                "price_point_yuan_per_kwh":pp,"price_upper_yuan_per_kwh":pu,
                "latest_observation_slot":release*6,"history_end_date":d-pd.Timedelta(days=1)}))
    released=pd.concat(pieces,ignore_index=True)
    assert len(released)==120240
    assert ((released.slot_index-1)>=released.latest_observation_slot).all()
    csv(released,DATA/"released_forecast_inputs.csv")
    source_rows=[]
    for path in [cfg.data_file,cfg.pv_forecast_file,cfg.q2_schedule_file, project/"附件/附件1.xlsx"]:
        source_rows.append({"source":str(path.relative_to(project)),"bytes":path.stat().st_size,"sha256":digest(path)})
    csv(pd.DataFrame(source_rows),DATA/"source_provenance.csv")
    csv(pd.DataFrame([
        ["observations",len(obs),365,"date,slot_index","2025-01-01","2025-12-31"],
        ["evaluation",48096,334,"date,slot_index","2025-02-01","2025-12-31"],
        ["released_inputs",len(released),334,"date,release_hour,slot_index","2025-02-01","2025-12-31"],
        ["scenario_cells",int(net_s.size),334,"date,quantile,slot_index","2025-02-01","2025-12-31"],
        ["pilot_days",len(chosen),len(chosen),"date","2025-02-01","2025-12-31"]
    ],columns=["dataset","rows_or_cells","unique_days","primary_key","start","end"]),DATA/"dataset_sizes.csv")
    print("Data prepared: 365 observation days, 334 evaluation days, 66 pilot days.",flush=True)


def load_prepared():
    z=np.load(DATA/"inputs.npz",allow_pickle=False)
    inputs={k:z[k] for k in ["load","pv","price","fixed_price","load_fc"]}
    inputs["dates"]=pd.DatetimeIndex(z["dates"])
    inputs["pv_release"]={r:z[f"pv_{r}"] for r in m.CFG.release_hours}
    inputs["q2"]=pd.read_csv(DATA/"inherited_q2.csv",parse_dates=["date"])
    return inputs


def forecast_experiment(inputs,cfg,out):
    rows=[]
    for window in [14,21,35,49,60]:
        for decay in [7.,14.,21.,28.]:
            pf=causal_prices(inputs,replace(cfg,residual_window_days=window,price_decay_days=decay))
            for i in range(31,len(inputs["dates"])):
                actual=inputs["price"][i]; error=actual-pf["point"][i]
                rows.append({"date":inputs["dates"][i],"split":split_of(inputs["dates"][i]),
                    "window_days":window,"decay_days":decay,"mae":np.mean(np.abs(error)),
                    "mse":np.mean(error**2),"coverage80":np.mean((actual>=pf["lower"][i])&(actual<=pf["upper"][i])),
                    "mean_width":np.mean(pf["upper"][i]-pf["lower"][i])})
    frame=pd.DataFrame(rows); csv(frame,out/"forecast_daily.csv")
    agg=frame.groupby(["split","window_days","decay_days"],as_index=False).agg(days=("date","count"),mae=("mae","mean"),mse=("mse","mean"),coverage80=("coverage80","mean"),mean_width=("mean_width","mean"))
    agg["rmse"]=np.sqrt(agg.pop("mse")); csv(agg,out/"forecast_summary.csv")
    val=agg[agg.split=="validation"].sort_values(["mae","window_days","decay_days"])
    best=val.iloc[0]
    chosen=agg[(agg.window_days==best.window_days)&(agg.decay_days==best.decay_days)].copy()
    chosen["role"]="validation_selected"
    formal=agg[(agg.window_days==35)&(agg.decay_days==14)].copy(); formal["role"]="formal_reference"
    csv(pd.concat([chosen,formal],ignore_index=True),out/"forecast_selected.csv")
    print("Forecast: 20 configurations x 334 days complete.",flush=True)


def rolling_one(day,i,inputs,net,pf,cfg,gain,reserve,terminal):
    c=replace(cfg,rolling_adjustment_gain=gain,empirical_reserve_quantile=reserve)
    q0=day.planned_grid_kwh.to_numpy(float); q=q0.copy(); wall=0.; max_window=0.
    if gain>0:
        for j,rh in enumerate(c.release_hours[1:],1):
            a=rh*6; b=c.release_hours[j+1]*6 if j+1<len(c.release_hours) else 144
            target=(net["point"][rh][i]+m.empirical_reserve(net["residual"][rh],i,c))*c.dt_hours
            price,_=m.corrected_price_forecast(i,rh,inputs["price"][i],pf,c)
            sol=m.solve_window(target,price,float(day.soc_start_kwh.iloc[a]),c,
                baseline_grid_kwh=q0[a:],terminal_soc_kwh=terminal)
            q[a:b]=q0[a:b]+gain*(sol["grid_kwh"][:b-a]-q0[a:b])
            wall+=sol["solver_time_sec"]; max_window=max(max_window,sol["max_eq_residual_kwh"])
    exe=execute(q,i,inputs,day,c)
    p=inputs["price"][i]; up=np.maximum(q-q0,0); down=np.maximum(q0-q,0)
    plan_cost=float(p@q0); adjust=float(c.upward_adjustment_multiplier*p@up+c.downward_adjustment_penalty_multiplier*p@down)
    emg_cost=float(c.emergency_multiplier*p@exe["emergency"][0])
    return {"total_cost_yuan":plan_cost+adjust+emg_cost,"emergency_kwh":float(exe["emergency"].sum()),
        "plan_cost_yuan":plan_cost,"adjustment_cost_yuan":adjust,"emergency_cost_yuan":emg_cost,
        "solver_time_sec":wall+exe["solver_time_sec"],"max_window_residual":max_window,
        "balance_residual":exe["balance_residual"],"recursion_residual":exe["recursion_residual"],
        "terminal_residual":exe["terminal_residual"],"simultaneous_kwh":exe["simultaneous_kwh"]}


def run_pilot(scale):
    cfg=m.CFG; inputs=load_prepared(); pf=causal_prices(inputs,cfg); net=m.prepare_net_forecasts(inputs,cfg)
    out=ROOT/("预实验结果" if scale=="pilot" else "全日期实验结果"); out.mkdir(exist_ok=True)
    forecast_experiment(inputs,cfg,out)
    cal=pd.read_csv(DATA/"evaluation_calendar.csv",parse_dates=["date"])
    dates=pd.DatetimeIndex(cal.loc[cal.pilot_selected if scale=="pilot" else np.ones(len(cal),bool),"date"])
    by={d:g.sort_values("slot_index").reset_index(drop=True) for d,g in inputs["q2"].groupby("date")}
    risk_rows=[]; res_rows=[]; rolling_rows=[]; errors=[]
    for di,date in enumerate(dates):
        i=inputs["dates"].get_loc(date); day=by[date]
        ns=m.net_scenarios_from_q2(day); ps=m.price_scenarios_for_day(i,pf,cfg)
        soc0=float(day.soc_start_kwh.iloc[0]); terminal=float(day.soc_end_kwh.iloc[-1])
        meta={"date":date,"split":split_of(date),"month":date.month}
        cached={}
        for rho in RHOS:
            try:
                weights=m.scenario_weights(m.QUANTILES)
                sol=solve_joint(ns,ps,weights,soc0,terminal,cfg,rho)
                exe=execute(sol["grid_kwh"],i,inputs,day,cfg)
                row={**meta,"risk_weight":rho,"realized_cost_yuan":exe["total_cost_yuan"],
                    "realized_emergency_kwh":float(exe["emergency"].sum()),
                    "scenario_expected_cost_yuan":sol["expected_cost"],"scenario_cvar90_yuan":sol["cvar90"],
                    "legacy_top2_mean_yuan":float(np.mean(np.sort(sol["scenario_cost"])[-2:])),
                    "planned_grid_kwh":float(sol["grid_kwh"].sum()),"solver_time_sec":sol["solver_time_sec"],
                    "max_eq_residual_kwh":max(sol["eq_residual"],exe["eq_residual"]),
                    "terminal_residual_kwh":max(sol["terminal_residual"],exe["terminal_residual"]),
                    "max_simultaneous_kwh":max(sol["simultaneous_kwh"],exe["simultaneous_kwh"]),"status":"optimal"}
                risk_rows.append(row); cached[rho]=row
            except Exception as exc:
                errors.append({**meta,"experiment":"risk","parameter":rho,"error":str(exc)})
        for count,ids in SUBSETS.items():
            try:
                if count==11 and .02 in cached:
                    r=cached[.02]
                    row={**meta,"scenario_count":11,"realized_cost_yuan":r["realized_cost_yuan"],
                        "realized_emergency_kwh":r["realized_emergency_kwh"],"solver_time_sec":r["solver_time_sec"],
                        "scenario_cvar90_yuan":r["scenario_cvar90_yuan"],"status":"reused_exact_11_rho02"}
                else:
                    sol=solve_joint(ns[ids],ps[ids],m.scenario_weights(m.QUANTILES[ids]),soc0,terminal,cfg,.02)
                    exe=execute(sol["grid_kwh"],i,inputs,day,cfg)
                    row={**meta,"scenario_count":count,"realized_cost_yuan":exe["total_cost_yuan"],
                        "realized_emergency_kwh":float(exe["emergency"].sum()),"solver_time_sec":sol["solver_time_sec"],
                        "scenario_cvar90_yuan":sol["cvar90"],"status":"optimal"}
                res_rows.append(row)
            except Exception as exc:
                errors.append({**meta,"experiment":"resolution","parameter":count,"error":str(exc)})
        for gain in GAINS:
            baseline=None
            for reserve in RESERVES:
                try:
                    if gain==0 and baseline is not None:
                        r=dict(baseline)
                    else:
                        r=rolling_one(day,i,inputs,net,pf,cfg,gain,reserve,terminal)
                        if gain==0: baseline=dict(r)
                    rolling_rows.append({**meta,"gain":gain,"reserve_quantile":reserve,**r,"status":"optimal"})
                except Exception as exc:
                    errors.append({**meta,"experiment":"rolling","parameter":f"{gain},{reserve}","error":str(exc)})
        # Result tables are real completed calculations, not log/state files.
        if (di+1)%6==0 or di+1==len(dates):
            csv(pd.DataFrame(risk_rows),out/"risk_daily.csv")
            csv(pd.DataFrame(res_rows),out/"resolution_daily.csv")
            csv(pd.DataFrame(rolling_rows),out/"rolling_daily.csv")
            print(f"Executed {di+1}/{len(dates)} real dates; failed jobs={len(errors)}",flush=True)
    csv(pd.DataFrame(errors,columns=["date","split","month","experiment","parameter","error"]),out/"failed_jobs.csv")
    summarize(out)


def bootstrap_mean(values,dates,draws=2000):
    """Circular month-cluster bootstrap for sparse date sampling, not 7-day blocks."""
    values=np.asarray(values,float); dates=pd.DatetimeIndex(dates)
    finite=np.isfinite(values)
    values=values[finite]; dates=dates[finite]   # missing pairs must not propagate NaN
    groups=[np.flatnonzero(dates.month==mo) for mo in sorted(dates.month.unique())]
    rng=np.random.default_rng(SEED); means=[]
    if len(groups)<2: return np.nan,np.nan
    for _ in range(draws):
        ids=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        means.append(values[ids].mean())
    return tuple(np.quantile(means,[.025,.975]))


def summarize(out):
    risk=pd.read_csv(out/"risk_daily.csv",parse_dates=["date"])
    rolling=pd.read_csv(out/"rolling_daily.csv",parse_dates=["date"])
    resolution=pd.read_csv(out/"resolution_daily.csv",parse_dates=["date"])
    for name,df,keys,cost,emg in [
        ("risk",risk,["risk_weight"],"realized_cost_yuan","realized_emergency_kwh"),
        ("rolling",rolling,["gain","reserve_quantile"],"total_cost_yuan","emergency_kwh"),
        ("resolution",resolution,["scenario_count"],"realized_cost_yuan","realized_emergency_kwh")]:
        combined=pd.concat([df,df.assign(split="all")],ignore_index=True)
        agg=combined.groupby(["split",*keys],as_index=False).agg(days=("date","nunique"),
            mean_cost_yuan=(cost,"mean"),total_cost_yuan=(cost,"sum"),mean_emergency_kwh=(emg,"mean"),
            total_emergency_kwh=(emg,"sum"),mean_solver_sec=("solver_time_sec","mean"))
        csv(agg,out/f"{name}_summary.csv")
    base=risk[risk.risk_weight==0].set_index("date")
    paired=[]
    for rho,df in risk.groupby("risk_weight"):
        x=df.set_index("date").join(base[["realized_cost_yuan","realized_emergency_kwh"]],rsuffix="_base",validate="one_to_one")
        delta=x.realized_cost_yuan-x.realized_cost_yuan_base
        lo,hi=bootstrap_mean(delta,x.index)
        paired.append({"risk_weight":rho,"days":len(x),"mean_cost_delta_yuan":delta.mean(),
            "mean_emergency_reduction_kwh":(x.realized_emergency_kwh_base-x.realized_emergency_kwh).mean(),
            "month_cluster_ci_low_yuan":lo,"month_cluster_ci_high_yuan":hi})
    csv(pd.DataFrame(paired),out/"risk_paired.csv")
    # Freeze the validation minimum-cost configuration before reading its test outcome.
    rs=pd.read_csv(out/"rolling_summary.csv")
    val=rs[rs.split=="validation"].sort_values(["mean_cost_yuan","gain","reserve_quantile"])
    best=val.iloc[0]
    selections=[]
    for role,gain,reserve in [("validation_selected",best.gain,best.reserve_quantile),("formal_reference",.5,.2),("no_update",0.,0.)]:
        selections.append(rs[(rs.gain==gain)&(rs.reserve_quantile==reserve)].assign(role=role))
    csv(pd.concat(selections,ignore_index=True),out/"rolling_selected.csv")
    frozen=rolling[(rolling.gain==best.gain)&(rolling.reserve_quantile==best.reserve_quantile)].copy()
    reference=rolling[(rolling.gain==.5)&(rolling.reserve_quantile==.2)].copy()
    x=frozen.merge(reference,on="date",suffixes=("_selected","_formal"),validate="one_to_one")
    csv(x,out/"selected_vs_formal_daily.csv")
    report(out)


def md_table(frame,columns):
    frame=frame[columns]
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for _,r in frame.iterrows():
        lines.append("| "+" | ".join(f"{v:.6g}" if isinstance(v,(float,np.floating)) else str(v) for v in r)+" |")
    return "\n".join(lines)


def report(out):
    risk=pd.read_csv(out/"risk_summary.csv")
    selected=pd.read_csv(out/"rolling_selected.csv")
    fsel=pd.read_csv(out/"forecast_selected.csv")
    rd=pd.read_csv(out/"risk_daily.csv")
    roll=pd.read_csv(out/"rolling_daily.csv")
    failed=pd.read_csv(out/"failed_jobs.csv")
    sizes=pd.read_csv(DATA/"dataset_sizes.csv")
    n=rd.date.nunique()
    is_pilot=n==66
    selected_test=selected[(selected.role=="validation_selected")&(selected.split=="test")].iloc[0]
    formal_test=selected[(selected.role=="formal_reference")&(selected.split=="test")].iloc[0]
    selected_val=selected[(selected.role=="validation_selected")&(selected.split=="validation")].iloc[0]
    formal_val=selected[(selected.role=="formal_reference")&(selected.split=="validation")].iloc[0]
    chosen_price=fsel[(fsel.role=="validation_selected")&(fsel.split=="test")].iloc[0]
    formal_price=fsel[(fsel.role=="formal_reference")&(fsel.split=="test")].iloc[0]
    text=f"""# 第四题扩展数据与预实验结果

## 1. 扩展了什么

本次扩展的是同一微网真实日期的实验覆盖面。原始附件仍是 2025 年单站观测，未添加虚构年份、站点或扰动观测。全年观测 365 天、52,560 个十分钟时段，其中 1 月为暖启动，正式可评估日期为 2 月 1 日至 12 月 31 日的 334 天、48,096 个时段。全部日期已形成可复用数据集和场景库。场景单元是模型输入，不能算成独立观测样本。

{md_table(sizes,['dataset','rows_or_cells','unique_days','primary_key'])}

上一版风险前沿仅使用 11 个事后挑选的月度压力日，控制相图仅使用 4 个指定日。本次预实验按每个月 6 个等距日历日期取样，共 {n} 天，避免仅挑选不利或有利结果。开发期为 2—6 月（150 天；预实验 30 天），验证期为 7—9 月（92 天；预实验 18 天），测试期为 10—12 月（92 天；预实验 18 天）。价格预测实验覆盖全部 334 天。其余本报告总量只表示抽样日期合计，不外推成年总量。

## 2. 口径修正及与旧版的关系

场景概率由分位点中点划分，11 个场景并非等概率。新代码按离散概率精确计算 CVaR，保留旧版最高两个成本平均值作为对照列。旧图上的该近似值不能直接当作本报告的 CVaR。

新实验还对每个场景以及实际日回放统一固定日末电量为继承的第二问日末边界，避免通过消耗期末库存换取费用优势。旧扩展脚本的实际日回放没有固定日末电量。因此新结果不能与旧 11 日结果混合后计算收益；需要同口径重算。

零历史日的价格初始化改用题给附件 1 的电价曲线，负荷初始化使用附件 1 的负荷曲线，随后只使用过去观测。旧函数使用 1 月全月中位数初始化 1 月 1 日价格预测、全年负荷中位数兜底，这些值可通过历史残差影响早期评估，新流程消除了这条路径。

本次控制实验沿用原版的继承状态锚点和完成日储能再优化。购电修正遵守发布时钟，但储能回放使用当日完整实测序列，所以费用是具有事后储能再调度的匹配回放结果，尚不等于完整在线控制效果。全年及指定日期此前已被查看，本次时间划分是回顾性验证，不能宣称为从未见过的独立外部测试。

## 3. 实际运行的实验

- 因果电价预测：20 组窗口—衰减组合 × 334 天，每个日期 144 时段；按验证期 MAE 选参数，再报告测试期表现。不能以全部日期最低误差作为测试精度。
- 风险前沿：{n} 天 × 7 个风险权重，11 个共尾净负荷—电价场景。
- 场景分辨率：{n} 天 × 3/5/7/11 个场景，在风险权重 0.02 下比较；11 场景结果复用相同计算。
- 控制相图：{n} 天 × 5 个执行增益 × 5 个储备分位水平，按验证期平均费用选参数，再冻结到测试期。

求解失败条目数为 {len(failed)}；失败内容如有则逐条保存在 failed_jobs.csv，不用零值代替。风险表完成 {len(rd)} 行，控制表完成 {len(roll)} 行。

## 4. 风险前沿真实结果

下表为全部预实验日期的匹配平均，成本单位为元/日，紧急电量为 kWh/日，耗时为秒/日前 LP（含一级与二级择优）。

{md_table(risk[risk.split=='all'],['risk_weight','days','mean_cost_yuan','mean_emergency_kwh','mean_solver_sec'])}

risk_paired.csv 提供同日期相对风险中性方案的费用变化、紧急量降低及按月份聚类重采样的 95% 描述性区间。因为抽样日并不连续，不能冒充七日连续移动块 Bootstrap。风险权重体现费用与可靠性的偏好，本实验不根据测试结果宣布某个风险权重普适最优。

## 5. 控制参数的冻结测试

下表选参仅依据验证期。formal_reference 为原版执行增益 0.5、储备分位 0.2；no_update 为完全不修改合同。费用为元/日，紧急量为 kWh/日。

{md_table(selected[selected.split.isin(['validation','test'])],['role','split','gain','reserve_quantile','days','mean_cost_yuan','mean_emergency_kwh'])}

测试期仅有 3 个月、18 个抽样日期，不能据此证明全年稳定优越。不同控制参数下的费用与紧急电量必须同时比较；平均费用最低不等于最可靠。

## 6. 因果价格预测的冻结测试

{md_table(fsel[fsel.split.isin(['validation','test'])],['role','split','window_days','decay_days','days','mae','rmse','coverage80'])}

误差单位为元/kWh，coverage80 是经验覆盖率比例。价格最优参数本次只在预测实验中评估，没有偷偷替换控制实验的正式 35 日窗口、14 日衰减。因此不能把价格 MAE 的下降直接写成已经实现的费用下降。

## 7. 物理与可复现检查

风险规划与实际回放最大等式残差为 {rd.max_eq_residual_kwh.max():.3g} kWh，日末边界最大残差为 {rd.terminal_residual_kwh.max():.3g} kWh；连续 LP 的最大同时充放电量为 {max(0.,rd.max_simultaneous_kwh.max()):.3g} kWh。控制回放最大平衡残差为 {roll.balance_residual.max():.3g} kWh，状态递推残差为 {roll.recursion_residual.max():.3g} kWh。若同时充放电超过数值容差，必须升级为含充放电模式二元变量的模型，不能将连续 LP 默认为严格互斥模型。

全部原始附件和原论文保持原样。数据/目录保存真实观测、继承输入、发布信息输入、场景数组、日期划分及源文件哈希；脚本/目录保存全部计算与绘图入口；本报告和具体方案说明各指标的定义、样本口径与后续执行方法。

## 8. 下一步实验的执行条件

本次验证集选出的控制参数为 γ={selected_val.gain:g}、τ={selected_val.reserve_quantile:g}。它在验证期相对原参数费用变化为 {selected_val.mean_cost_yuan-formal_val.mean_cost_yuan:+.2f} 元/日，在测试期变化为 {selected_test.mean_cost_yuan-formal_test.mean_cost_yuan:+.2f} 元/日，测试紧急量变化为 {selected_test.mean_emergency_kwh-formal_test.mean_emergency_kwh:+.2f} kWh/日。若验证改善而测试退化，不能替换正式控制方案，应保持原参数并把参数季节稳定性作为全量实验重点。

验证期选择的价格参数为 {chosen_price.window_days:g} 日窗口、{chosen_price.decay_days:g} 日衰减；测试 MAE 从原参数的 {formal_price.mae:.6f} 降到 {chosen_price.mae:.6f} 元/kWh，变化为 {(chosen_price.mae/formal_price.mae-1)*100:+.2f}%。这可支持价格预测层的进一步比较，但尚未构成使用新价格参数重新求解调度后的经济收益证据。

先检查验证及测试是否同向改善，再使用 --scale full 扩展至全部 334 天。完整计算在独立的全日期实验结果目录生成同结构表格，不覆盖预实验。全量结果仍属于当前站点、当前年度的回顾性研究。要增加真实年份或站点样本，必须取得同站新年度或新站点的同步负荷、光伏、电价、预测发布记录和设备参数；当前文件不包含这些数据。
"""
    if not is_pilot:
        text=text.replace("预实验结果","全日期实验结果").replace("本次预实验按每个月 6 个等距日历日期取样，共 334 天","本次完整实验使用全部可评估日期，共 334 天")
        text=text.replace("预实验 30 天","完整实验 150 天").replace("预实验 18 天","完整实验 92 天")
        text=text.replace("测试期仅有 3 个月、18 个抽样日期","测试期有 3 个月、92 个连续日期")
        text=text.replace("因为抽样日并不连续，不能冒充七日连续移动块 Bootstrap。","该汇总沿用月份聚类重采样口径，未另算连续七日块 Bootstrap。")
    (ROOT/("预实验结果说明.md" if is_pilot else "全日期实验结果说明.md")).write_text(text,encoding="utf-8")


def validate_math():
    # Independent linearized RU definition checks weighted tail arithmetic.
    cost=np.array([1.,5.,10.]); w=np.array([.8,.15,.05]); alpha=.9
    aub=np.column_stack([-np.ones(3),-np.eye(3)])
    lp=linprog(np.r_[1.,w/(1-alpha)],A_ub=aub,b_ub=-cost,bounds=[(None,None)]+[(0,None)]*3,method="highs")
    assert lp.success and abs(weighted_cvar(cost,w,alpha)-lp.fun)<1e-10
    assert abs(weighted_cvar(cost,w,alpha)-7.5)<1e-10
    assert abs(m.scenario_weights(m.QUANTILES).sum()-1)<1e-12


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["all","prepare","pilot","summarize"],default="all")
    ap.add_argument("--scale",choices=["pilot","full"],default="pilot")
    args=ap.parse_args(); validate_math(); tic=time.perf_counter()
    if args.stage in ["all","prepare"]: prepare()
    if args.stage in ["all","pilot"]: run_pilot(args.scale)
    if args.stage=="summarize": summarize(ROOT/("预实验结果" if args.scale=="pilot" else "全日期实验结果"))
    print(f"Completed in {time.perf_counter()-tic:.1f} seconds.",flush=True)
