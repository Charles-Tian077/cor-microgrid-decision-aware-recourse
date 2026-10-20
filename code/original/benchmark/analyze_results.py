# -*- coding: utf-8 -*-
"""Statistical inference for the Q4 whole-year experiments.

Consumes the *numerical* outputs of run_full_parallel.py (334 real days) and the
original 66-day pilot package, and writes one tidy long table plus detail tables:

    统计检验/统计检验_长表.csv          tidy: analysis, subject, metric, value, ci, p, n
    统计检验/风险前沿_配对.csv
    统计检验/控制网格_配对.csv
    统计检验/分辨率_配对.csv
    统计检验/预测_配对.csv
    统计检验/分季节_控制.csv
    统计检验/年度汇总_关键策略.csv
    统计检验/统计检验说明.md

Inference protocol
------------------
* Whole-year series are *continuous* (334 consecutive real dates), so paired
  differences use a circular 7-day moving-block bootstrap (4,000 draws), which
  preserves short-range serial dependence.
* Split-level / month-level subsets are not continuous samples; there we use the
  month-cluster bootstrap inherited from the pilot package. The two are never
  mixed in one interval.
* Paired tests are two-sided Wilcoxon signed-rank on daily paired differences;
  p-values within an experiment family are Holm-corrected. Nothing is called
  "significant" on the basis of a validation-only comparison: the frozen test
  comparison is reported separately from the selection comparison.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
FULL = BASE / "全日期实验结果"
PILOT = Path(r"D:\Users\数学建模\扩展数据预实验\预实验结果")
STAT = BASE / "统计检验"
SEED = 20261017
DRAWS = 4000
BLOCK = 7


# --------------------------------------------------------------------------- #
# inference helpers
# --------------------------------------------------------------------------- #
def circular_block_bootstrap_ci(x, block=BLOCK, draws=DRAWS, seed=SEED):
    """95% CI for the mean of a *continuous* daily series (circular blocks)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < block + 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    k = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    sampled = x[idx.reshape(draws, -1)[:, :n]]
    means = sampled.mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def month_cluster_ci(values, dates, draws=DRAWS, seed=SEED):
    """95% CI resampling whole calendar months (for non-continuous subsets)."""
    values = np.asarray(values, float)
    dates = pd.DatetimeIndex(dates)
    groups = [np.flatnonzero(dates.month == mo) for mo in sorted(dates.month.unique())]
    if len(groups) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        ids = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        means.append(values[ids].mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_ci(delta, dates, continuous):
    return circular_block_bootstrap_ci(delta) if continuous else month_cluster_ci(delta, dates)


def wilcoxon_p(delta):
    delta = np.asarray(delta, float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 8 or np.allclose(delta, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def dm_test(score_a, score_b, lag=0):
    """Diebold-Mariano on loss differential d = score_a - score_b (negative => A better)."""
    d = np.asarray(score_a, float) - np.asarray(score_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return math.nan, math.nan
    mean_d = d.mean()
    gamma0 = np.mean((d - mean_d) ** 2)
    var = gamma0
    for k in range(1, lag + 1):
        gamma = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    if var <= 0:
        return math.nan, math.nan
    stat = mean_d / math.sqrt(var / n)
    return float(stat), float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))


def r6(x):
    return float(round(float(x), 6))


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_results(folder):
    folder = Path(folder)
    risk = pd.read_csv(folder / "risk_daily.csv", parse_dates=["date"])
    resolution = pd.read_csv(folder / "resolution_daily.csv", parse_dates=["date"])
    rolling = pd.read_csv(folder / "rolling_daily.csv", parse_dates=["date"])
    forecast = pd.read_csv(folder / "forecast_daily.csv", parse_dates=["date"])
    return risk, resolution, rolling, forecast


ROWS = []


def add(analysis, subject, metric, value, ci_low=math.nan, ci_high=math.nan,
        p_value=math.nan, n_days=math.nan, split="all", note=""):
    ROWS.append({
        "analysis": analysis, "subject": subject, "metric": metric,
        "value": r6(value) if isinstance(value, (int, float, np.floating)) else value,
        "ci95_low": r6(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci95_high": r6(ci_high) if np.isfinite(ci_high) else np.nan,
        "p_value": r6(p_value) if np.isfinite(p_value) else np.nan,
        "n_days": int(n_days) if np.isfinite(n_days) else np.nan,
        "split": split, "note": note,
    })


# --------------------------------------------------------------------------- #
# 1) risk frontier
# --------------------------------------------------------------------------- #
def analyse_risk(risk, continuous=True):
    detail = []
    base = risk[risk.risk_weight == 0.0].set_index("date")
    pvals = []
    for rho, grp in risk.groupby("risk_weight"):
        g = grp.set_index("date").join(
            base[["realized_cost_yuan", "realized_emergency_kwh"]],
            rsuffix="_base", validate="one_to_one").dropna()
        d_cost = g.realized_cost_yuan - g.realized_cost_yuan_base
        d_red = g.realized_emergency_kwh_base - g.realized_emergency_kwh
        lo, hi = paired_ci(d_cost, g.index, continuous)
        p = wilcoxon_p(d_cost)
        detail.append({
            "risk_weight": rho, "days": len(g),
            "mean_cost_yuan": float(g.realized_cost_yuan.mean()),
            "mean_emergency_kwh": float(g.realized_emergency_kwh.mean()),
            "mean_cost_delta_yuan": float(d_cost.mean()),
            "ci95_low": lo, "ci95_high": hi, "wilcoxon_p": p,
            "mean_emergency_reduction_kwh": float(d_red.mean()),
            "cost_per_kwh_avoided_yuan": float(d_cost.mean() / d_red.mean()) if d_red.mean() > 1e-9 else math.nan,
        })
        pvals.append(p)
    p_holm = holm(pvals)
    for row, ph in zip(detail, p_holm):
        row["holm_p"] = ph
        add("risk_frontier", f"rho={row['risk_weight']:g}", "mean_realized_cost_yuan", row["mean_cost_yuan"],
            n_days=row["days"], note="matched replay of the day-ahead plan")
        add("risk_frontier", f"rho={row['risk_weight']:g}", "mean_realized_emergency_kwh", row["mean_emergency_kwh"],
            n_days=row["days"])
        add("risk_frontier", f"rho={row['risk_weight']:g}", "paired_cost_delta_vs_rho0_yuan",
            row["mean_cost_delta_yuan"], row["ci95_low"], row["ci95_high"], row["holm_p"], row["days"])
        add("risk_frontier", f"rho={row['risk_weight']:g}", "mean_emergency_reduction_vs_rho0_kwh",
            row["mean_emergency_reduction_kwh"], n_days=row["days"])
        add("risk_frontier", f"rho={row['risk_weight']:g}", "cost_per_kwh_avoided_yuan",
            row["cost_per_kwh_avoided_yuan"], n_days=row["days"],
            note="marginal insurance price implied by the matched replay; descriptive only")
    pd.DataFrame(detail).to_csv(STAT / "风险前沿_配对.csv", index=False, encoding="utf-8-sig")

    # split-level heterogeneity
    for split in ["development", "validation", "test"]:
        sub = risk[risk.split == split]
        b = sub[sub.risk_weight == 0.0].set_index("date")
        for rho, grp in sub.groupby("risk_weight"):
            g = grp.set_index("date").join(b[["realized_cost_yuan"]], rsuffix="_base", validate="one_to_one")
            d = g.realized_cost_yuan - g.realized_cost_yuan_base
            lo, hi = month_cluster_ci(d, g.index)
            add("risk_frontier_by_split", f"rho={rho:g}", "paired_cost_delta_vs_rho0_yuan",
                d.mean(), lo, hi, wilcoxon_p(d), len(d), split)

    # monthly pattern of the reliability gain
    month_rows = []
    base = risk[risk.risk_weight == 0.0].set_index("date")
    for mo, grp in risk.groupby("month"):
        b = base[base.month == mo]
        for rho in sorted(risk.risk_weight.unique()):
            g = grp[grp.risk_weight == rho].set_index("date").join(
                b[["realized_emergency_kwh"]], rsuffix="_base", validate="one_to_one")
            month_rows.append({
                "month": int(mo), "risk_weight": rho, "days": len(g),
                "mean_cost_yuan": float(g.realized_cost_yuan.mean()),
                "mean_emergency_kwh": float(g.realized_emergency_kwh.mean()),
                "emergency_reduction_kwh": float((g.realized_emergency_kwh_base - g.realized_emergency_kwh).mean()),
            })
    pd.DataFrame(month_rows).to_csv(STAT / "风险前沿_分月.csv", index=False, encoding="utf-8-sig")

    # Pareto set
    frontier = pd.DataFrame(detail)[["risk_weight", "mean_cost_yuan", "mean_emergency_kwh"]].copy()
    pareto = []
    for _, row in frontier.iterrows():
        dominated = ((frontier.mean_cost_yuan <= row.mean_cost_yuan) &
                     (frontier.mean_emergency_kwh <= row.mean_emergency_kwh) &
                     ((frontier.mean_cost_yuan < row.mean_cost_yuan) |
                      (frontier.mean_emergency_kwh < row.mean_emergency_kwh))).any()
        pareto.append(not dominated)
    frontier["on_pareto_front"] = pareto
    frontier.to_csv(STAT / "风险前沿_Pareto.csv", index=False, encoding="utf-8-sig")
    return detail, frontier


# --------------------------------------------------------------------------- #
# 2) control grid
# --------------------------------------------------------------------------- #
def analyse_control(rolling, continuous=True):
    formal = rolling[(rolling.gain == 0.5) & (rolling.reserve_quantile == 0.2)].set_index("date")
    grid_rows = []
    pvals = []
    for (gain, reserve), grp in rolling.groupby(["gain", "reserve_quantile"]):
        g = grp.set_index("date").join(formal[["total_cost_yuan", "emergency_kwh"]],
                                       rsuffix="_formal", validate="one_to_one").dropna()
        d = g.total_cost_yuan - g.total_cost_yuan_formal
        d_emg = g.emergency_kwh - g.emergency_kwh_formal
        lo, hi = paired_ci(d, g.index, continuous)
        p = wilcoxon_p(d)
        grid_rows.append({
            "gain": gain, "reserve_quantile": reserve, "days": len(g),
            "mean_cost_yuan": float(g.total_cost_yuan.mean()),
            "mean_emergency_kwh": float(g.emergency_kwh.mean()),
            "mean_cost_delta_vs_formal_yuan": float(d.mean()), "ci95_low": lo, "ci95_high": hi,
            "wilcoxon_p": p, "mean_emergency_delta_vs_formal_kwh": float(d_emg.mean()),
            "cost_per_kwh_avoided_yuan": float(-d.mean() / d_emg.mean()) if abs(d_emg.mean()) > 1e-9 else math.nan,
        })
        pvals.append(p)
    ph = holm(pvals)
    for row, p in zip(grid_rows, ph):
        row["holm_p"] = p
    pd.DataFrame(grid_rows).to_csv(STAT / "控制网格_配对.csv", index=False, encoding="utf-8-sig")
    for row in grid_rows:
        tag = f"gain={row['gain']:g},reserve={row['reserve_quantile']:g}"
        add("control_grid", tag, "mean_total_cost_yuan", row["mean_cost_yuan"], n_days=row["days"])
        add("control_grid", tag, "mean_emergency_kwh", row["mean_emergency_kwh"], n_days=row["days"])
        add("control_grid", tag, "paired_cost_delta_vs_formal_yuan", row["mean_cost_delta_vs_formal_yuan"],
            row["ci95_low"], row["ci95_high"], row["holm_p"], row["days"])
        add("control_grid", tag, "mean_emergency_delta_vs_formal_kwh", row["mean_emergency_delta_vs_formal_kwh"],
            n_days=row["days"])

    # main effects and interaction of the grid
    pivot_cost = rolling.pivot_table(index="gain", columns="reserve_quantile",
                                     values="total_cost_yuan", aggfunc="mean")
    pivot_emg = rolling.pivot_table(index="gain", columns="reserve_quantile",
                                    values="emergency_kwh", aggfunc="mean")
    pd.concat({"mean_cost_yuan": pivot_cost, "mean_emergency_kwh": pivot_emg}, axis=0).to_csv(
        STAT / "控制网格_均值曲面.csv", encoding="utf-8-sig")
    for gain, v in pivot_cost.mean(axis=1).items():
        add("control_main_effect", f"gain={gain:g}", "mean_cost_yuan_marginal_over_reserve", v)
    for reserve, v in pivot_cost.mean(axis=0).items():
        add("control_main_effect", f"reserve={reserve:g}", "mean_cost_yuan_marginal_over_gain", v)

    # frozen comparison on validation / test, plus seasonal strata
    val = rolling[rolling.split == "validation"].groupby(["gain", "reserve_quantile"])["total_cost_yuan"].mean()
    best = val.sort_values().index[0]
    add("control_selection", f"validation_best=gain{best[0]:g},reserve{best[1]:g}", "validation_mean_cost_yuan",
        float(val.loc[best]), n_days=int((rolling.split == "validation").sum() / 25))
    selected = rolling[(rolling.gain == best[0]) & (rolling.reserve_quantile == best[1])].set_index("date")
    season_rows = []
    for split in ["development", "validation", "test"]:
        g = selected[selected.split == split].join(
            formal[formal.split == split][["total_cost_yuan", "emergency_kwh"]],
            rsuffix="_formal", validate="one_to_one").dropna()
        d = g.total_cost_yuan - g.total_cost_yuan_formal
        de = g.emergency_kwh - g.emergency_kwh_formal
        lo, hi = month_cluster_ci(d, g.index)
        season_rows.append({
            "scope": split, "days": len(g),
            "selected_mean_cost_yuan": float(g.total_cost_yuan.mean()),
            "formal_mean_cost_yuan": float(g.total_cost_yuan_formal.mean()),
            "cost_delta_yuan": float(d.mean()), "ci95_low": lo, "ci95_high": hi,
            "wilcoxon_p": wilcoxon_p(d),
            "selected_mean_emergency_kwh": float(g.emergency_kwh.mean()),
            "formal_mean_emergency_kwh": float(g.emergency_kwh_formal.mean()),
            "emergency_delta_kwh": float(de.mean()),
        })
        tag = f"selected(gain={best[0]:g},reserve={best[1]:g}) vs formal(0.5,0.2)"
        add("control_frozen", tag, f"paired_cost_delta_{split}_yuan", d.mean(), lo, hi,
            wilcoxon_p(d), len(g), split)
        add("control_frozen", tag, f"paired_emergency_delta_{split}_kwh", de.mean(), n_days=len(g), split=split)
    for season in ["winter", "spring", "summer", "autumn"]:
        months = {"winter": [12, 1, 2], "spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11]}[season]
        g = selected[selected.month.isin(months)].join(
            formal[formal.month.isin(months)][["total_cost_yuan", "emergency_kwh"]],
            rsuffix="_formal", validate="one_to_one").dropna()
        if g.empty:
            continue
        d = g.total_cost_yuan - g.total_cost_yuan_formal
        de = g.emergency_kwh - g.emergency_kwh_formal
        lo, hi = month_cluster_ci(d, g.index)
        season_rows.append({
            "scope": season, "days": len(g),
            "selected_mean_cost_yuan": float(g.total_cost_yuan.mean()),
            "formal_mean_cost_yuan": float(g.total_cost_yuan_formal.mean()),
            "cost_delta_yuan": float(d.mean()), "ci95_low": lo, "ci95_high": hi,
            "wilcoxon_p": wilcoxon_p(d),
            "selected_mean_emergency_kwh": float(g.emergency_kwh.mean()),
            "formal_mean_emergency_kwh": float(g.emergency_kwh_formal.mean()),
            "emergency_delta_kwh": float(de.mean()),
        })
        add("control_frozen", f"selected vs formal", f"paired_cost_delta_{season}_yuan",
            d.mean(), lo, hi, wilcoxon_p(d), len(g), season)
    pd.DataFrame(season_rows).to_csv(STAT / "分季节_控制.csv", index=False, encoding="utf-8-sig")

    # cost decomposition across the grid (mean per day)
    comp = rolling.groupby(["gain", "reserve_quantile"]).agg(
        plan=("plan_cost_yuan", "mean"), adjustment=("adjustment_cost_yuan", "mean"),
        emergency=("emergency_cost_yuan", "mean"), total=("total_cost_yuan", "mean")).reset_index()
    comp.to_csv(STAT / "控制网格_费用分解.csv", index=False, encoding="utf-8-sig")
    return grid_rows, best


# --------------------------------------------------------------------------- #
# 3) scenario resolution
# --------------------------------------------------------------------------- #
def analyse_resolution(resolution, continuous=True):
    ref = resolution[resolution.scenario_count == 11].set_index("date")
    rows = []
    pvals = []
    for count, grp in resolution.groupby("scenario_count"):
        g = grp.set_index("date").join(ref[["realized_cost_yuan", "solver_time_sec"]],
                                       rsuffix="_ref", validate="one_to_one").dropna()
        d = g.realized_cost_yuan - g.realized_cost_yuan_ref
        lo, hi = paired_ci(d, g.index, continuous)
        p = wilcoxon_p(d)
        rows.append({
            "scenario_count": int(count), "days": len(g),
            "mean_cost_yuan": float(g.realized_cost_yuan.mean()),
            "mean_emergency_kwh": float(g.realized_emergency_kwh.mean()),
            "cost_delta_vs_11_yuan": float(d.mean()), "ci95_low": lo, "ci95_high": hi,
            "wilcoxon_p": p, "mean_solver_sec": float(g.solver_time_sec.mean()),
            "speedup_vs_11": float(ref.solver_time_sec.mean() / g.solver_time_sec.mean()),
        })
        pvals.append(p)
    ph = holm(pvals)
    for row, p in zip(rows, ph):
        row["holm_p"] = p
        tag = f"scenarios={row['scenario_count']}"
        add("scenario_resolution", tag, "mean_cost_yuan", row["mean_cost_yuan"], n_days=row["days"])
        add("scenario_resolution", tag, "mean_emergency_kwh", row["mean_emergency_kwh"], n_days=row["days"])
        add("scenario_resolution", tag, "paired_cost_delta_vs_11_yuan", row["cost_delta_vs_11_yuan"],
            row["ci95_low"], row["ci95_high"], row["holm_p"], row["days"])
        add("scenario_resolution", tag, "mean_solver_sec_11_scenario_plan", row["mean_solver_sec"], n_days=row["days"])
        add("scenario_resolution", tag, "speedup_vs_11_scenarios", row["speedup_vs_11"], n_days=row["days"])
    pd.DataFrame(rows).to_csv(STAT / "分辨率_配对.csv", index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# 4) price forecast
# --------------------------------------------------------------------------- #
def analyse_forecast(forecast):
    formal = forecast[(forecast.window_days == 35) & (forecast.decay_days == 14.0)].set_index("date")
    sel = forecast[(forecast.window_days == 14) & (forecast.decay_days == 28.0)].set_index("date")
    rows = []
    pvals = []
    for (window, decay), grp in forecast.groupby(["window_days", "decay_days"]):
        g = grp.set_index("date").join(formal[["mae", "mse"]], rsuffix="_formal", validate="one_to_one").dropna()
        d_abs = g.mae - g.mae_formal
        lo, hi = month_cluster_ci(d_abs, g.index)
        dm, dm_p = dm_test(g.mae, g.mae_formal, lag=0)
        p = wilcoxon_p(d_abs)
        rows.append({
            "window_days": int(window), "decay_days": float(decay), "days": len(g),
            "mae": float(g.mae.mean()), "rmse": float(np.sqrt(g.mse.mean())),
            "coverage80": float(g.coverage80.mean()), "mean_width": float(g.mean_width.mean()),
            "mae_delta_vs_formal": float(d_abs.mean()), "ci95_low": lo, "ci95_high": hi,
            "wilcoxon_p": p, "dm_stat": dm, "dm_p": dm_p,
        })
        pvals.append(p)
    ph = holm(pvals)
    for row, p in zip(rows, ph):
        row["holm_p"] = p
        tag = f"w{row['window_days']}_d{row['decay_days']:g}"
        for split in ["development", "validation", "test"]:
            x = forecast[(forecast.window_days == row["window_days"]) &
                         (forecast.decay_days == row["decay_days"]) & (forecast.split == split)]
            add("price_forecast", tag, f"mae_{split}", float(x.mae.mean()), n_days=len(x), split=split)
            add("price_forecast", tag, f"coverage80_{split}", float(x.coverage80.mean()), n_days=len(x), split=split)
    pd.DataFrame(rows).to_csv(STAT / "预测_全网格.csv", index=False, encoding="utf-8-sig")

    # frozen selected-vs-formal test comparison
    for label, g in [("selected(w14,d28)", sel), ("formal(w35,d14)", formal)]:
        for split in ["development", "validation", "test"]:
            x = g[g.split == split]
            add("price_forecast_frozen", label, f"mae_{split}", float(x.mae.mean()), n_days=len(x), split=split)
    for split in ["development", "validation", "test"]:
        a = sel[sel.split == split].mae
        b = formal[formal.split == split].mae
        join = pd.concat([a.rename("selected"), b.rename("formal")], axis=1, join="inner")
        dm, dm_p = dm_test(join.selected, join.formal, lag=0)
        lo, hi = month_cluster_ci(join.selected - join.formal, join.index)
        add("price_forecast_frozen", "selected(w14,d28) vs formal(w35,d14)",
            f"mae_delta_{split}", float((join.selected - join.formal).mean()), lo, hi,
            wilcoxon_p(join.selected - join.formal), len(join), split,
            note=f"DM stat={dm:.3f}")
    pd.DataFrame(rows)[["window_days", "decay_days", "mae_delta_vs_formal", "dm_p", "holm_p"]].to_csv(
        STAT / "预测_配对.csv", index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# 5) evaluated-period totals
# --------------------------------------------------------------------------- #
def analyse_totals(risk, rolling):
    rows = []
    for rho in [0.0, 0.02, 0.1]:
        x = risk[risk.risk_weight == rho]
        rows.append({
            "family": "day_ahead_risk", "strategy": f"rho={rho:g}", "days": len(x),
            "total_cost_yuan": float(x.realized_cost_yuan.sum()),
            "mean_daily_cost_yuan": float(x.realized_cost_yuan.mean()),
            "total_emergency_kwh": float(x.realized_emergency_kwh.sum()),
            "mean_daily_emergency_kwh": float(x.realized_emergency_kwh.mean()),
            "days_with_emergency": int((x.realized_emergency_kwh > 1e-6).sum()),
            "mean_solver_sec": float(x.solver_time_sec.mean()),
        })
    for gain, reserve, label in [(0.5, 0.2, "formal_reference"), (0.0, 0.0, "validation_selected"),
                                 (0.0, 0.2, "no_update_matrix"), (1.0, 0.4, "aggressive_full_follow")]:
        x = rolling[(rolling.gain == gain) & (rolling.reserve_quantile == reserve)]
        if x.empty:
            continue
        rows.append({
            "family": "rolling_control", "strategy": f"gain={gain:g},reserve={reserve:g} ({label})",
            "days": len(x), "total_cost_yuan": float(x.total_cost_yuan.sum()),
            "mean_daily_cost_yuan": float(x.total_cost_yuan.mean()),
            "total_emergency_kwh": float(x.emergency_kwh.sum()),
            "mean_daily_emergency_kwh": float(x.emergency_kwh.mean()),
            "days_with_emergency": int((x.emergency_kwh > 1e-6).sum()),
            "mean_solver_sec": float(x.solver_time_sec.mean()),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(STAT / "年度汇总_关键策略.csv", index=False, encoding="utf-8-sig")
    for _, row in frame.iterrows():
        add("evaluated_period_totals", f"{row.family}:{row.strategy}", "total_cost_yuan",
            row.total_cost_yuan, n_days=row.days, note="Feb-Dec 2025, 334 evaluated days (not a calendar year)")
        add("evaluated_period_totals", f"{row.family}:{row.strategy}", "total_emergency_kwh",
            row.total_emergency_kwh, n_days=row.days)


# --------------------------------------------------------------------------- #
# 5b) stochastic-programming value bounds (WS / RP / EV / VSS / EVPI)
# --------------------------------------------------------------------------- #
def analyse_bounds():
    path = FULL / "随机规划价值边界.csv"
    if not path.exists():
        add("stochastic_programming_bounds", "unavailable", "missing_table", 0.0)
        return
    b = pd.read_csv(path, parse_dates=["date"])
    cols = {
        "ws_perfect_foresight_cost_yuan": "wait_and_see_perfect_foresight_cost_yuan",
        "rp_expected_cost_yuan": "recourse_problem_expected_cost_yuan",
        "rp_realized_cost_yuan": "recourse_problem_realized_cost_yuan",
        "ev_plan_expected_cost_yuan": "expected_value_plan_under_scenarios_yuan",
        "ev_plan_realized_cost_yuan": "expected_value_plan_realized_cost_yuan",
        "vss_yuan": "value_of_stochastic_solution_yuan_per_day",
        "evpi_yuan": "expected_value_of_perfect_information_yuan_per_day",
    }
    for col, label in cols.items():
        x = b[col].to_numpy(float)
        lo, hi = circular_block_bootstrap_ci(x)
        add("stochastic_programming_bounds", "all_334_days", f"mean_{label}", x.mean(), lo, hi, n_days=len(b))
        add("stochastic_programming_bounds", "all_334_days", f"total_{label}", x.sum(), n_days=len(b),
            note="Feb-Dec 2025, 334 evaluated days (not a calendar year)")
        for split in ["development", "validation", "test"]:
            part = b[b.split == split]
            add("stochastic_programming_bounds", split, f"mean_{label}", part[col].mean(),
                n_days=len(part), split=split)
    add("stochastic_programming_bounds", "all_334_days", "share_of_days_vss_positive",
        float((b.vss_yuan > 0).mean()), n_days=len(b),
        note="fraction of days on which the stochastic plan beats the expected-value plan")
    add("stochastic_programming_bounds", "all_334_days", "share_of_days_evpi_positive",
        float((b.evpi_yuan > 0).mean()), n_days=len(b))


# --------------------------------------------------------------------------- #
# 6) physical residuals (quality evidence reused by the workbook)
# --------------------------------------------------------------------------- #
def analyse_residuals(risk, rolling):
    checks = [
        ("max_scenario_equality_residual_kwh", float(risk.max_eq_residual_kwh.max())),
        ("max_terminal_soc_residual_kwh", float(risk.terminal_residual_kwh.max())),
        ("max_simultaneous_charge_discharge_kwh", float(max(0.0, risk.max_simultaneous_kwh.max()))),
        ("max_replay_balance_residual_kwh", float(rolling.balance_residual.max())),
        ("max_replay_recursion_residual_kwh", float(rolling.recursion_residual.max())),
        ("max_replay_terminal_residual_kwh", float(rolling.terminal_residual.max())),
        ("max_rolling_window_residual_kwh", float(rolling.max_window_residual.max())),
    ]
    for name, value in checks:
        add("physical_checks", "all_days_and_configs", name, value, n_days=len(risk),
            note="pass if within numerical tolerance; values are ~1e-12 for equalities")
    return checks


def main():
    STAT.mkdir(parents=True, exist_ok=True)
    risk, resolution, rolling, forecast = load_results(FULL)
    analyse_risk(risk, continuous=True)
    analyse_control(rolling, continuous=True)
    analyse_resolution(resolution, continuous=True)
    analyse_forecast(forecast)
    analyse_totals(risk, rolling)
    analyse_bounds()
    checks = analyse_residuals(risk, rolling)

    # pilot vs full headline comparison (descriptive)
    try:
        pr = pd.read_csv(PILOT / "risk_summary.csv")
        pfull = pd.read_csv(FULL / "risk_summary.csv")
        for rho in sorted(pfull.risk_weight.unique()):
            add("pilot_vs_full", f"rho={rho:g}", "mean_cost_yuan_pilot66",
                float(pr[(pr.split == "all") & (pr.risk_weight == rho)].mean_cost_yuan.iloc[0]), n_days=66)
            add("pilot_vs_full", f"rho={rho:g}", "mean_cost_yuan_full334",
                float(pfull[(pfull.split == "all") & (pfull.risk_weight == rho)].mean_cost_yuan.iloc[0]), n_days=334)
    except Exception as exc:  # noqa: BLE001
        add("pilot_vs_full", "unavailable", "error", str(exc)[:160])

    out = pd.DataFrame(ROWS)
    out.to_csv(STAT / "统计检验_长表.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_rows": int(len(out)),
        "analyses": sorted(out.analysis.unique().tolist()),
        "full_days": int(risk.date.nunique()),
        "rolling_configs": int(rolling.groupby(["gain", "reserve_quantile"]).ngroups),
        "risk_weights": sorted(map(float, risk.risk_weight.unique())),
        "seed": SEED, "bootstrap_draws": DRAWS, "block_days": BLOCK,
        "residual_checks": {k: r6(v) for k, v in checks},
    }
    (STAT / "statistics_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["n_rows", "full_days", "rolling_configs"]}, ensure_ascii=False))
    print("statistics written to", STAT)


if __name__ == "__main__":
    main()
