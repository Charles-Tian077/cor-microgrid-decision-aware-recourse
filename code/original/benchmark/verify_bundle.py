# -*- coding: utf-8 -*-
"""Independent dataset, weighted-risk, causal-prefix and physical checks."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import run_experiments as e


def main(scale):
    root=e.ROOT;data=e.DATA;out=root/("预实验结果" if scale=="pilot" else "全日期实验结果")
    obs=pd.read_csv(data/"observations_10min.csv",parse_dates=["date"])
    cal=pd.read_csv(data/"evaluation_calendar.csv",parse_dates=["date"])
    rel=pd.read_csv(data/"released_forecast_inputs.csv",parse_dates=["date","history_end_date"])
    q2=pd.read_csv(data/"inherited_q2.csv",parse_dates=["date"])
    assert len(obs)==52560 and obs.date.nunique()==365
    assert not obs.duplicated(["date","slot_index"]).any()
    assert obs.groupby("date").slot_index.nunique().eq(144).all()
    assert len(cal)==334 and cal.pilot_selected.sum()==66
    assert not rel.duplicated(["date","release_hour","slot_index"]).any()
    assert (rel.history_end_date<rel.date).all()
    assert ((rel.slot_index-1)>=rel.latest_observation_slot).all()
    joined=obs[obs.split!="warmup"].merge(q2,on=["date","slot_index"],validate="one_to_one",indicator=True)
    assert len(joined)==48096 and joined._merge.eq("both").all()
    inputs=e.load_prepared(); cfg=e.m.CFG
    p=e.causal_prices(inputs,cfg)
    alt=dict(inputs);alt["price"]=inputs["price"].copy();cut=inputs["dates"].get_loc(pd.Timestamp("2025-08-15"))
    alt["price"][cut:]*=3.7
    pp=e.causal_prices(alt,cfg)
    assert np.array_equal(p["point"][:cut+1],pp["point"][:cut+1])
    assert np.array_equal(p["lower"][:cut+1],pp["lower"][:cut+1])
    # A release may use only past slots, not the current/future realization.
    i=inputs["dates"].get_loc(pd.Timestamp("2025-09-01"))
    future=inputs["price"][i].copy();future[36:]*=4.
    a=e.m.corrected_price_forecast(i,6,inputs["price"][i],p,cfg)
    b=e.m.corrected_price_forecast(i,6,future,p,cfg)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    x=e.m.corrected_load_forecast(inputs["load_fc"][i],inputs["load"][i],6,cfg)
    future=inputs["load"][i].copy();future[36:]*=4.
    y=e.m.corrected_load_forecast(inputs["load_fc"][i],future,6,cfg)
    assert np.array_equal(x,y)
    risk=pd.read_csv(out/"risk_daily.csv");roll=pd.read_csv(out/"rolling_daily.csv")
    nd=66 if scale=="pilot" else 334
    assert len(risk)==nd*7 and len(roll)==nd*25
    assert len(pd.read_csv(out/"resolution_daily.csv"))==nd*4
    assert len(pd.read_csv(out/"forecast_daily.csv"))==334*20
    assert len(pd.read_csv(out/"failed_jobs.csv"))==0
    assert risk.max_eq_residual_kwh.max()<1e-6
    assert risk.terminal_residual_kwh.max()<1e-6
    assert risk.max_simultaneous_kwh.max()<1e-6
    assert roll.balance_residual.max()<1e-6 and roll.recursion_residual.max()<1e-6
    assert roll.terminal_residual.max()<1e-6 and roll.simultaneous_kwh.max()<1e-6
    assert np.allclose(roll.total_cost_yuan,roll.plan_cost_yuan+roll.adjustment_cost_yuan+roll.emergency_cost_yuan,atol=1e-7,rtol=0)
    e.validate_math()
    # Check all original sources when the source project is available.
    checked=0
    for _,r in pd.read_csv(data/"source_provenance.csv").iterrows():
        source=root.parent.parent/r.source
        if source.exists():
            assert e.digest(source)==r.sha256
            checked+=1
    print(f"Verified: 365 observation days, 334 evaluation days, {nd} experimental dates.")
    print("Verified: exact weighted CVaR, future-perturbation causality, key joins, physical residuals and settlement.")
    print(f"Unchanged original source hashes checked: {checked}.")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--scale",choices=["pilot","full"],default="pilot")
    main(ap.parse_args().scale)
