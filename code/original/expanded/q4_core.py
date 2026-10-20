# -*- coding: utf-8 -*-
"""问题四：波动电价下的无预报基准与价格感知滚动修正。

所有预测严格使用决策时刻之前已经观测的数据。模型读取既有清洗结果、问题二
因果净负荷预测和问题三多时点光伏预报，对未执行区间作收缩滚动优化，生成 result4-2.xlsx、result4-3.xlsx
以及论文绘图所需的可复核中间结果。原始附件只读。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from copy import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack


SEED = 20260911
RNG = np.random.default_rng(SEED)


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parent
    project_root: Path = Path(__file__).resolve().parents[1]
    data_file: Path = project_root / "数据预处理" / "数据结果" / "model_ready_features.csv"
    pv_forecast_file: Path = project_root / "数据预处理" / "数据结果" / "pv_forecasts_10min_pchip.csv"
    q2_schedule_file: Path = project_root / "第二问" / "数据结果" / "q2_schedule_10min.csv"
    q3_schedule_file: Path = project_root / "第三问" / "数据结果" / "q3_schedule_10min.csv"
    template42: Path = project_root / "附件" / "附件5" / "result4-2.xlsx"
    template43: Path = project_root / "附件" / "附件5" / "result4-3.xlsx"
    result_dir: Path = root / "数据结果"
    figure_dir: Path = root / "图像"
    output_start: str = "2025-02-01"
    output_end: str = "2025-12-31"
    slots_per_day: int = 144
    dt_hours: float = 1.0 / 6.0
    release_hours: tuple[int, ...] = (0, 6, 12, 18)
    soc_min_kwh: float = 1200.0
    soc_max_kwh: float = 10800.0
    initial_soc_kwh: float = 1200.0
    max_power_kw: float = 5000.0
    charge_efficiency: float = 0.90
    discharge_efficiency: float = 0.90
    emergency_multiplier: float = 5.0
    upward_adjustment_multiplier: float = 1.50
    downward_adjustment_penalty_multiplier: float = 0.50
    cvar_alpha: float = 0.90
    # 问题四要求按波动电价重算问题二；继承问题二正式模型的风险中性期望费用目标。
    # CVaR仅保留为敏感性扩展，不进入正式结果。
    risk_weight: float = 0.0
    # 全年网格检验表明，在因果收缩已经抑制价格噪声后再叠加上尾价格会增加费用；
    # 因而滚动层采用收缩点预测，供需风险由联合情景和净负荷保形裕度承担。
    price_tail_weight: float = 0.00
    empirical_reserve_quantile: float = 0.20
    rolling_adjustment_gain: float = 0.50
    residual_window_days: int = 35
    price_decay_days: float = 14.0
    price_bias_decay_hours: float = 4.0
    load_bias_window_slots: int = 12
    load_bias_decay_hours: float = 6.0
    solver_time_limit_sec: float = 45.0


CFG = Config()
QUANTILES = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
NET_QUANTILE_COLUMNS = [
    "net_q05_kw", "net_q10_kw", "net_q20_kw", "net_q30_kw", "net_q40_kw",
    "net_forecast_kw", "net_q60_kw", "net_q70_kw", "net_q80_kw", "net_q90_kw", "net_q95_kw",
]


def ensure_directories(cfg: Config) -> None:
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)


def require_finite(name: str, values: np.ndarray) -> None:
    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError(f"{name}为空")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}含{int(arr.size - np.isfinite(arr).sum())}个非有限值")


def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def day_matrix(frame: pd.DataFrame, dates: pd.DatetimeIndex, column: str) -> np.ndarray:
    table = frame.pivot(index="date", columns="slot_index", values=column)
    out = table.reindex(index=dates, columns=range(1, CFG.slots_per_day + 1)).to_numpy(float)
    require_finite(column, out)
    return out


def load_inputs(cfg: Config) -> dict:
    for path in (cfg.data_file, cfg.pv_forecast_file, cfg.q2_schedule_file, cfg.q3_schedule_file, cfg.template42, cfg.template43):
        if not path.exists():
            raise FileNotFoundError(path)
    data = pd.read_csv(cfg.data_file)
    data["date"] = pd.to_datetime(data["date_label"]).dt.normalize()
    data = data.sort_values(["date", "slot_index"]).reset_index(drop=True)
    counts = data.groupby("date")["slot_index"].agg(["count", "nunique", "min", "max"])
    bad = counts[(counts["count"] != 144) | (counts["nunique"] != 144) | (counts["min"] != 1) | (counts["max"] != 144)]
    if not bad.empty:
        raise ValueError(f"日—时段结构异常：{bad.head().to_dict('index')}")
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    if len(dates) != 365 or dates[0] != pd.Timestamp("2025-01-01") or dates[-1] != pd.Timestamp("2025-12-31"):
        raise ValueError("清洗结果不是2025完整365日")
    load = day_matrix(data, dates, "load_kw")
    pv = day_matrix(data, dates, "pv_kw")
    price = day_matrix(data, dates, "price_variable_yuan_per_kwh")
    fixed_price = day_matrix(data, dates, "price_fixed_yuan_per_kwh")[0]
    for name, arr in {"负荷": load, "光伏": pv, "波动电价": price}.items():
        require_finite(name, arr)
    if (load < 0).any() or (pv < 0).any() or (price <= 0).any():
        raise ValueError("负荷、光伏或电价越过物理边界")

    q2 = pd.read_csv(cfg.q2_schedule_file)
    q2["date"] = pd.to_datetime(q2["date"]).dt.normalize()
    required_q2 = {"date", "slot_index", "planned_grid_kwh", "charge_kwh", "discharge_kwh", "soc_start_kwh",
                   "soc_end_kwh", "load_forecast_kw", "pv_forecast_kw", *NET_QUANTILE_COLUMNS}
    missing = required_q2 - set(q2.columns)
    if missing:
        raise ValueError(f"问题二结果缺少字段：{sorted(missing)}")
    q3 = pd.read_csv(cfg.q3_schedule_file)
    q3["date"] = pd.to_datetime(q3["date"]).dt.normalize()
    required_q3 = {"date", "slot_index", "planned_grid_kwh", "adjusted_grid_kwh", "up_adjustment_kwh",
                   "down_adjustment_kwh", "emergency_kwh", "spill_kwh", "soc_start_kwh", "soc_end_kwh"}
    missing = required_q3 - set(q3.columns)
    if missing:
        raise ValueError(f"问题三结果缺少字段：{sorted(missing)}")

    pvf = pd.read_csv(cfg.pv_forecast_file)
    pvf["date"] = pd.to_datetime(pvf["date_label"]).dt.normalize()
    pvf["release_hour"] = pd.to_numeric(pvf["release_hour"], errors="raise").astype(int)
    pv_release: dict[int, np.ndarray] = {}
    for release in cfg.release_hours:
        horizon = 144 - 6 * release
        rows = pvf[(pvf["release_hour"] == release) & (pvf["horizon_minute"] <= 1440 - 60 * release)]
        values = []
        for date in dates:
            part = rows[rows["date"] == date].sort_values("horizon_minute")
            if len(part) != horizon:
                raise ValueError(f"{date.date()} {release:02d}:00光伏预报长度为{len(part)}，应为{horizon}")
            values.append(part["pv_forecast_pchip_kw"].to_numpy(float))
        pv_release[release] = np.stack(values)

    q2_load_table = q2.pivot(index="date", columns="slot_index", values="load_forecast_kw").reindex(columns=range(1, 145))
    load_fc = data.pivot(index="date", columns="slot_index", values="load_same_slot_mean_7d").reindex(
        index=dates, columns=range(1, 145)).to_numpy(float)
    lag1 = data.pivot(index="date", columns="slot_index", values="load_lag_1d").reindex(
        index=dates, columns=range(1, 145)).to_numpy(float)
    fallback = np.nanmedian(load, axis=0)
    load_fc = np.where(np.isfinite(load_fc), load_fc, lag1)
    load_fc = np.where(np.isfinite(load_fc), load_fc, fallback[None, :])
    for i, date in enumerate(dates):
        if date in q2_load_table.index:
            load_fc[i] = q2_load_table.loc[date].to_numpy(float)
    require_finite("因果负荷预测", load_fc)
    return {"dates": dates, "load": load, "pv": pv, "price": price, "fixed_price": fixed_price, "q2": q2, "q3": q3,
            "pv_release": pv_release, "load_fc": load_fc}


def scenario_weights(levels: np.ndarray) -> np.ndarray:
    mids = 0.5 * (levels[:-1] + levels[1:])
    edges = np.concatenate([[0.0], mids, [1.0]])
    weights = np.diff(edges)
    return weights / weights.sum()


def build_causal_price_forecasts(price: np.ndarray, dates: pd.DatetimeIndex, cfg: Config) -> dict:
    """递推构造仅依赖过去日的电价预测和时段保形区间。"""
    n, t = price.shape
    point = np.full((n, t), np.nan)
    lower = np.full((n, t), np.nan)
    upper = np.full((n, t), np.nan)
    residual = np.full((n, t), np.nan)
    training_days = np.zeros(n, int)
    global_profile = np.median(price[:31], axis=0)
    for i, date in enumerate(dates):
        left = max(0, i - cfg.residual_window_days)
        hist_idx = np.arange(left, i)
        if len(hist_idx) == 0:
            forecast = global_profile.copy()
        else:
            age = i - hist_idx
            weights = np.exp(-age / cfg.price_decay_days)
            weekdays = np.array([dates[j].weekday() for j in hist_idx])
            weights *= np.where(weekdays == date.weekday(), 1.55, 1.0)
            same_weekend = (weekdays >= 5) == (date.weekday() >= 5)
            weights *= np.where(same_weekend, 1.18, 0.88)
            weights /= weights.sum()
            forecast = weights @ price[hist_idx]
        point[i] = np.maximum(forecast, 0.05)
        valid = hist_idx[np.isfinite(residual[hist_idx]).all(axis=1)]
        if len(valid) >= 7:
            lower[i] = np.maximum(point[i] + np.quantile(residual[valid], 0.10, axis=0), 0.05)
            upper[i] = np.maximum(point[i] + np.quantile(residual[valid], 0.90, axis=0), lower[i] + 1.0e-4)
        elif len(hist_idx) > 0:
            raw = price[hist_idx] - point[i]
            lower[i] = np.maximum(point[i] + np.quantile(raw, 0.10, axis=0), 0.05)
            upper[i] = np.maximum(point[i] + np.quantile(raw, 0.90, axis=0), lower[i] + 1.0e-4)
        else:
            lower[i] = np.maximum(point[i] * 0.80, 0.05)
            upper[i] = point[i] * 1.20
        residual[i] = price[i] - point[i]
        training_days[i] = len(hist_idx)
    return {"point": point, "lower": lower, "upper": upper, "residual": residual, "training_days": training_days}


def price_scenarios_for_day(day_index: int, forecasts: dict, cfg: Config) -> np.ndarray:
    left = max(0, day_index - cfg.residual_window_days)
    hist = forecasts["residual"][left:day_index]
    hist = hist[np.isfinite(hist).all(axis=1)]
    point = forecasts["point"][day_index]
    if len(hist) < 7:
        scale = np.maximum(0.12 * point, 0.03)
        z = np.array([-1.645, -1.282, -0.842, -0.524, -0.253, 0.0, 0.253, 0.524, 0.842, 1.282, 1.645])
        scenarios = point[None, :] + z[:, None] * scale[None, :]
    else:
        shocks = np.quantile(hist, QUANTILES, axis=0)
        scenarios = point[None, :] + shocks
    return np.maximum(scenarios, 0.05)


def shrink_price_forecasts_causally(forecasts: dict, actual_price: np.ndarray, fixed_price: np.ndarray) -> dict:
    """先递推收缩点预测，再用过去的收缩残差重新校准80%保形区间。"""
    n,t = actual_price.shape; weights = np.full(n, 0.50)
    point=np.empty((n,t)); lower=np.empty((n,t)); upper=np.empty((n,t)); residual=np.empty((n,t))
    for i in range(1, n):
        left = max(0, i - 60); hist = np.arange(left, i)
        signal = (forecasts["point"][hist] - fixed_price[None, :]).reshape(-1)
        error = forecasts["residual"][hist].reshape(-1)
        signal_var = float(np.var(signal)); error_var = float(np.var(error))
        weights[i] = float(np.clip(signal_var / max(signal_var + error_var, 1.0e-12), 0.15, 0.75))
    for i in range(n):
        point[i]=np.maximum(fixed_price+weights[i]*(forecasts["point"][i]-fixed_price),0.05)
        left=max(0,i-60); hist=residual[left:i]
        if len(hist)>=7:
            # 分时段绝对非一致性分数；有限样本上取ceil((n+1)*0.8)/n分位数。
            level=min(math.ceil((len(hist)+1)*0.80)/len(hist),1.0)
            radius=np.quantile(np.abs(hist),level,axis=0,method="higher")
        elif i>0:
            radius=np.maximum(np.quantile(np.abs(residual[:i]),0.90,axis=0),0.03)
        else:
            radius=np.maximum(0.20*point[i],0.03)
        lower[i]=np.maximum(point[i]-radius,0.05); upper[i]=np.maximum(point[i]+radius,lower[i]+1e-4)
        residual[i]=actual_price[i]-point[i]
    return {"point":point,"lower":lower,"upper":upper,"residual":residual,
            "training_days":forecasts["training_days"],"shrinkage_weight":weights}


@dataclass
class JointLayout:
    n: int
    q: slice
    charge: slice
    discharge: slice
    soc: slice
    emergency: slice
    spill: slice
    zeta: int
    u: slice
    a_eq: csr_matrix
    bounds: list[tuple[float | None, float | None]]


_LAYOUT_CACHE: dict[int, JointLayout] = {}


def build_joint_layout(scenarios: int, cfg: Config) -> JointLayout:
    if scenarios in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[scenarios]
    t = cfg.slots_per_day
    q = slice(0, t)
    charge = slice(q.stop, q.stop + scenarios * t)
    discharge = slice(charge.stop, charge.stop + scenarios * t)
    soc = slice(discharge.stop, discharge.stop + scenarios * (t + 1))
    emergency = slice(soc.stop, soc.stop + scenarios * t)
    spill = slice(emergency.stop, emergency.stop + scenarios * t)
    zeta = spill.stop
    u = slice(zeta + 1, zeta + 1 + scenarios)
    n = u.stop
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    r = 0
    for s in range(scenarios):
        state_offset = s * (t + 1)
        action_offset = s * t
        rows.append(r); cols.append(soc.start + state_offset); vals.append(1.0); r += 1
        for k in range(t):
            rows.extend([r] * 4)
            cols.extend([soc.start + state_offset + k + 1, soc.start + state_offset + k,
                         charge.start + action_offset + k, discharge.start + action_offset + k])
            vals.extend([1.0, -1.0, -cfg.charge_efficiency, 1.0 / cfg.discharge_efficiency]); r += 1
    offset = r
    for s in range(scenarios):
        action_offset = s * t
        for k in range(t):
            r = offset + s * t + k
            rows.extend([r] * 5)
            cols.extend([q.start + k, charge.start + action_offset + k, discharge.start + action_offset + k,
                         emergency.start + s * t + k, spill.start + s * t + k])
            vals.extend([1.0, -1.0, 1.0, 1.0, -1.0])
    a_eq = coo_matrix((vals, (rows, cols)), shape=(offset + scenarios * t, n)).tocsr()
    cap = cfg.max_power_kw * cfg.dt_hours
    bounds: list[tuple[float | None, float | None]] = []
    bounds += [(0.0, None)] * t
    bounds += [(0.0, cap)] * (scenarios * t)
    bounds += [(0.0, cap)] * (scenarios * t)
    bounds += [(cfg.soc_min_kwh, cfg.soc_max_kwh)] * (scenarios * (t + 1))
    bounds += [(0.0, None)] * (scenarios * t)
    bounds += [(0.0, None)] * (scenarios * t)
    bounds += [(0.0, None)]
    bounds += [(0.0, None)] * scenarios
    layout = JointLayout(n, q, charge, discharge, soc, emergency, spill, zeta, u, a_eq, bounds)
    _LAYOUT_CACHE[scenarios] = layout
    return layout


def solve_joint_day_ahead(net_scenarios_kw: np.ndarray, price_scenarios: np.ndarray, weights: np.ndarray,
                          soc_initial: float, cfg: Config, risk_weight: float | None = None,
                          fixed_plan_kwh: np.ndarray | None = None) -> dict:
    net = np.asarray(net_scenarios_kw, float)
    prices = np.asarray(price_scenarios, float)
    weights = np.asarray(weights, float)
    if net.shape != prices.shape or net.ndim != 2 or net.shape[1] != cfg.slots_per_day:
        raise ValueError(f"联合情景维度不一致：net={net.shape}, price={prices.shape}")
    require_finite("联合净负荷情景", net); require_finite("联合电价情景", prices)
    if (prices <= 0).any() or (weights < 0).any() or len(weights) != len(net) or weights.sum() <= 0:
        raise ValueError("联合情景权重或电价非法")
    if not cfg.soc_min_kwh <= soc_initial <= cfg.soc_max_kwh:
        raise ValueError("初始储电量越界")
    weights = weights / weights.sum()
    rho = cfg.risk_weight if risk_weight is None else float(risk_weight)
    if not 0 <= rho <= 1:
        raise ValueError("风险权重应位于[0,1]")
    s_count = len(net)
    layout = build_joint_layout(s_count, cfg)
    obj = np.zeros(layout.n)
    expected_price = weights @ prices
    obj[layout.q] = (1.0 - rho) * expected_price
    obj[layout.emergency] = ((1.0 - rho) * weights[:, None] * cfg.emergency_multiplier * prices).reshape(-1)
    obj[layout.zeta] = rho
    obj[layout.u] = rho * weights / max(1.0 - cfg.cvar_alpha, 1.0e-9)

    b_eq = np.zeros(layout.a_eq.shape[0])
    state_rows = s_count * (cfg.slots_per_day + 1)
    for s in range(s_count):
        b_eq[s * (cfg.slots_per_day + 1)] = soc_initial
    b_eq[state_rows:] = (net * cfg.dt_hours).reshape(-1)
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    for s in range(s_count):
        for k in range(cfg.slots_per_day):
            ub_rows.extend([s, s])
            ub_cols.extend([layout.q.start + k, layout.emergency.start + s * cfg.slots_per_day + k])
            ub_vals.extend([prices[s, k], cfg.emergency_multiplier * prices[s, k]])
        ub_rows.extend([s, s])
        ub_cols.extend([layout.zeta, layout.u.start + s])
        ub_vals.extend([-1.0, -1.0])
    a_ub = coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(s_count, layout.n)).tocsr()
    options = {"time_limit": cfg.solver_time_limit_sec, "primal_feasibility_tolerance": 1.0e-7,
               "dual_feasibility_tolerance": 1.0e-7}
    t0 = time.perf_counter()
    bounds = list(layout.bounds)
    if fixed_plan_kwh is not None:
        fixed = np.asarray(fixed_plan_kwh, float)
        if fixed.shape != (cfg.slots_per_day,) or (fixed < -1.0e-8).any():
            raise ValueError("固定日前购电计划必须为非负144维向量")
        for k, value in enumerate(fixed):
            bounds[layout.q.start + k] = (float(value), float(value))
    primary = linprog(obj, A_ub=a_ub, b_ub=np.zeros(s_count), A_eq=layout.a_eq, b_eq=b_eq,
                      bounds=bounds, method="highs", options=options)
    if not primary.success or primary.x is None:
        raise RuntimeError(f"联合情景LP一级目标未收敛：status={primary.status}, message={primary.message}")
    primary_tol = max(1.0e-5, abs(float(primary.fun)) * 1.0e-10)
    secondary_obj = np.zeros(layout.n)
    secondary_obj[layout.charge] = np.repeat(weights, cfg.slots_per_day)
    secondary_obj[layout.discharge] = np.repeat(weights, cfg.slots_per_day)
    secondary_obj[layout.spill] = 1.0e-6 * np.repeat(weights, cfg.slots_per_day)
    a_ub_secondary = vstack([a_ub, coo_matrix(obj.reshape(1, -1))]).tocsr()
    b_ub_secondary = np.r_[np.zeros(s_count), float(primary.fun) + primary_tol]
    secondary = linprog(secondary_obj, A_ub=a_ub_secondary, b_ub=b_ub_secondary,
                        A_eq=layout.a_eq, b_eq=b_eq, bounds=bounds,
                        method="highs", options=options)
    if not secondary.success or secondary.x is None:
        raise RuntimeError(f"联合情景LP二级择优未收敛：status={secondary.status}, message={secondary.message}")
    elapsed = time.perf_counter() - t0
    x = secondary.x
    primary_value = float(obj @ x)
    charge_s = x[layout.charge].reshape(s_count, cfg.slots_per_day)
    discharge_s = x[layout.discharge].reshape(s_count, cfg.slots_per_day)
    soc_s = x[layout.soc].reshape(s_count, cfg.slots_per_day + 1)
    return {
        "grid_kwh": x[layout.q],
        "charge_kwh": np.average(charge_s, axis=0, weights=weights),
        "discharge_kwh": np.average(discharge_s, axis=0, weights=weights),
        "soc_kwh": np.average(soc_s, axis=0, weights=weights),
        "scenario_charge_kwh": charge_s, "scenario_discharge_kwh": discharge_s,
        "scenario_soc_kwh": soc_s,
        "scenario_emergency_kwh": x[layout.emergency].reshape(s_count, cfg.slots_per_day),
        "scenario_spill_kwh": x[layout.spill].reshape(s_count, cfg.slots_per_day),
        "objective": primary_value, "primary_optimum": float(primary.fun),
        "primary_objective_drift": primary_value - float(primary.fun),
        "solver_time_sec": float(elapsed),
        "solver_status": f"一级:{primary.message}; 二级:{secondary.message}",
        "max_eq_residual_kwh": float(np.max(np.abs(layout.a_eq @ x - b_eq))),
        "max_ub_residual_yuan": float(max(0.0, np.max(a_ub @ x))),
        "max_simultaneous_kwh": float(np.max(np.minimum(charge_s, discharge_s))),
    }


def solve_window(net_target_kwh: np.ndarray, decision_price: np.ndarray, soc_initial: float, cfg: Config,
                 baseline_grid_kwh: np.ndarray | None = None, terminal_soc_kwh: float | None = None) -> dict:
    net = np.asarray(net_target_kwh, float); price = np.asarray(decision_price, float)
    require_finite("窗口净负荷", net); require_finite("窗口电价", price)
    if len(net) == 0 or len(net) != len(price) or (price <= 0).any():
        raise ValueError("窗口输入维度或电价非法")
    h = len(net)
    q = slice(0, h); c = slice(h, 2*h); u = slice(2*h, 3*h); e = slice(3*h, 4*h); w = slice(4*h, 5*h)
    soc = slice(5*h, 6*h+1); n = 6*h+1; up = down = None
    baseline = None
    if baseline_grid_kwh is not None:
        baseline = np.asarray(baseline_grid_kwh, float)
        if len(baseline) != h or (baseline < -1e-8).any():
            raise ValueError("基准购电计划非法")
        up = slice(n, n+h); down = slice(n+h, n+2*h); n += 2*h
    obj = np.zeros(n)
    if baseline is None:
        obj[q] = price
    else:
        obj[up] = cfg.upward_adjustment_multiplier * price
        obj[down] = cfg.downward_adjustment_penalty_multiplier * price
    obj[e] = cfg.emergency_multiplier * price
    rows: list[int] = []; cols: list[int] = []; vals: list[float] = []; b: list[float] = []; r = 0
    rows.append(r); cols.append(soc.start); vals.append(1.0); b.append(float(soc_initial)); r += 1
    for k in range(h):
        rows.extend([r]*4); cols.extend([soc.start+k+1, soc.start+k, c.start+k, u.start+k])
        vals.extend([1.0, -1.0, -cfg.charge_efficiency, 1.0/cfg.discharge_efficiency]); b.append(0.0); r += 1
    for k in range(h):
        rows.extend([r]*5); cols.extend([q.start+k, u.start+k, e.start+k, c.start+k, w.start+k])
        vals.extend([1.0, 1.0, 1.0, -1.0, -1.0]); b.append(float(net[k])); r += 1
    if terminal_soc_kwh is not None:
        if not cfg.soc_min_kwh-1e-8<=terminal_soc_kwh<=cfg.soc_max_kwh+1e-8:
            raise ValueError("终端储电量越界")
        rows.append(r); cols.append(soc.stop-1); vals.append(1.0); b.append(float(terminal_soc_kwh)); r += 1
    if baseline is not None and up is not None and down is not None:
        for k in range(h):
            rows.extend([r]*3); cols.extend([q.start+k, up.start+k, down.start+k])
            vals.extend([1.0, -1.0, 1.0]); b.append(float(baseline[k])); r += 1
    aeq = coo_matrix((vals, (rows, cols)), shape=(r, n)).tocsr(); beq = np.asarray(b)
    cap = cfg.max_power_kw * cfg.dt_hours
    bounds: list[tuple[float | None, float | None]] = []
    bounds += [(0.0, None)]*h + [(0.0, cap)]*h + [(0.0, cap)]*h + [(0.0, None)]*h + [(0.0, None)]*h
    bounds += [(cfg.soc_min_kwh, cfg.soc_max_kwh)]*(h+1)
    if baseline is not None:
        bounds += [(0.0, None)]*h + [(0.0, float(x)) for x in baseline]
    options = {"time_limit": cfg.solver_time_limit_sec, "primal_feasibility_tolerance": 1e-7,
               "dual_feasibility_tolerance": 1e-7}
    t0 = time.perf_counter()
    primary = linprog(obj, A_eq=aeq, b_eq=beq, bounds=bounds, method="highs", options=options)
    if not primary.success or primary.x is None:
        raise RuntimeError(f"滚动窗口LP一级目标未收敛：status={primary.status}, message={primary.message}")
    primary_tol = max(1.0e-5, abs(float(primary.fun)) * 1.0e-10)
    secondary_obj = np.zeros(n); secondary_obj[c] = 1.0; secondary_obj[u] = 1.0; secondary_obj[w] = 1.0e-6
    aub = coo_matrix(obj.reshape(1, -1)).tocsr()
    secondary = linprog(secondary_obj, A_ub=aub, b_ub=np.array([float(primary.fun)+primary_tol]),
                        A_eq=aeq, b_eq=beq, bounds=bounds, method="highs", options=options)
    if not secondary.success or secondary.x is None:
        raise RuntimeError(f"滚动窗口LP二级择优未收敛：status={secondary.status}, message={secondary.message}")
    elapsed = time.perf_counter()-t0
    x = secondary.x
    primary_value = float(obj @ x)
    out = {"grid_kwh": x[q], "charge_kwh": x[c], "discharge_kwh": x[u], "model_emergency_kwh": x[e],
           "model_spill_kwh": x[w], "soc_kwh": x[soc], "up_kwh": np.zeros(h), "down_kwh": np.zeros(h),
           "objective": primary_value, "primary_optimum": float(primary.fun),
           "primary_objective_drift": primary_value-float(primary.fun),
           "solver_time_sec": float(elapsed),
           "solver_status": f"一级:{primary.message}; 二级:{secondary.message}",
           "max_eq_residual_kwh": float(np.max(np.abs(aeq@x-beq))),
           "max_simultaneous_kwh": float(np.max(np.minimum(x[c], x[u])))}
    if baseline is not None and up is not None and down is not None:
        out["up_kwh"] = x[up]; out["down_kwh"] = x[down]
    return out


def execute_fixed_plan(plan: dict, load_kw: np.ndarray, pv_kw: np.ndarray, actual_price: np.ndarray, cfg: Config) -> dict:
    load_kw=np.asarray(load_kw,float);pv_kw=np.asarray(pv_kw,float);actual_price=np.asarray(actual_price,float)
    q=np.asarray(plan["grid_kwh"],float);soc0=float(np.asarray(plan["soc_kwh"],float)[0])
    recourse=solve_joint_day_ahead((load_kw-pv_kw)[None,:],actual_price[None,:],np.array([1.0]),
                                   soc0,cfg,risk_weight=0.0,fixed_plan_kwh=q)
    c=recourse["scenario_charge_kwh"][0];u=recourse["scenario_discharge_kwh"][0]
    soc=recourse["scenario_soc_kwh"][0];emergency=recourse["scenario_emergency_kwh"][0]
    spill=recourse["scenario_spill_kwh"][0]
    residual = q + pv_kw*cfg.dt_hours + u + emergency - load_kw*cfg.dt_hours - c - spill
    plan_cost = float(actual_price@q); emg_cost = float(cfg.emergency_multiplier*actual_price@emergency)
    return {"emergency_kwh": emergency, "spill_kwh": spill, "plan_cost_yuan": plan_cost,
            "emergency_cost_yuan": emg_cost, "total_cost_yuan": plan_cost+emg_cost,
            "charge_kwh":c,"discharge_kwh":u,"soc_kwh":soc,
            "max_balance_residual_kwh": float(np.max(np.abs(residual))),
            "solver_status":recourse["solver_status"]}


def corrected_load_forecast(load_fc: np.ndarray, actual: np.ndarray, release: int, cfg: Config) -> np.ndarray:
    start = 6*release; out = load_fc[start:].copy()
    if start > 0:
        left = max(0, start-cfg.load_bias_window_slots)
        bias = float(np.median(actual[left:start]-load_fc[left:start]))
        horizon = np.arange(1, len(out)+1)*cfg.dt_hours
        out += bias*np.exp(-horizon/cfg.load_bias_decay_hours)
    return np.maximum(out, 0.0)


def prepare_net_forecasts(inputs: dict, cfg: Config) -> dict:
    actual_net = inputs["load"]-inputs["pv"]
    points: dict[int, np.ndarray] = {}; residuals: dict[int, np.ndarray] = {}
    for release in cfg.release_hours:
        start = 6*release; rows=[]
        for i in range(len(inputs["dates"])):
            lf = corrected_load_forecast(inputs["load_fc"][i], inputs["load"][i], release, cfg)
            rows.append(lf-inputs["pv_release"][release][i])
        points[release]=np.stack(rows); residuals[release]=actual_net[:, start:]-points[release]
    return {"point": points, "residual": residuals}


def empirical_reserve(residuals: np.ndarray, day_index: int, cfg: Config) -> np.ndarray:
    left=max(0, day_index-cfg.residual_window_days); hist=residuals[left:day_index]
    if len(hist)==0: return np.zeros(residuals.shape[1])
    level=min(math.ceil((len(hist)+1)*cfg.empirical_reserve_quantile)/len(hist), 1.0)
    return np.maximum(np.quantile(hist, level, axis=0, method="higher"), 0.0)


def corrected_price_forecast(day_index: int, release: int, actual_price: np.ndarray, pf: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    start=6*release; point=pf["point"][day_index, start:].copy(); upper=pf["upper"][day_index, start:].copy()
    if start>0:
        left=max(0, start-6)
        bias=float(np.median(actual_price[left:start]-pf["point"][day_index, left:start]))
        horizon=np.arange(1, len(point)+1)*cfg.dt_hours
        correction=bias*np.exp(-horizon/cfg.price_bias_decay_hours)
        point=np.maximum(point+correction, 0.05); upper=np.maximum(upper+correction, point+1e-4)
    return point, upper


def replay_segment(sol: dict, length: int, load: np.ndarray, pv: np.ndarray, cfg: Config) -> dict:
    q=sol["grid_kwh"][:length]; c=sol["charge_kwh"][:length]; u=sol["discharge_kwh"][:length]
    net=(load[:length]-pv[:length])*cfg.dt_hours; gap=net+c-u-q
    emg=np.maximum(gap,0.0); spill=np.maximum(-gap,0.0)
    residual=q+pv[:length]*cfg.dt_hours+u+emg-load[:length]*cfg.dt_hours-c-spill
    return {"emergency_kwh":emg,"spill_kwh":spill,"max_balance_residual_kwh":float(np.max(np.abs(residual)))}


def simulate_rolling_day(day_index: int, soc_initial: float, inputs: dict, net_fc: dict, pf: dict,
                         initial_plan: dict, robust: bool, cfg: Config, price_aware: bool = True,
                         terminal_soc_kwh: float | None = None) -> dict:
    load=inputs["load"][day_index]; pv=inputs["pv"][day_index]; actual_price=inputs["price"][day_index]
    q_plan=initial_plan["grid_kwh"].copy(); q_final=np.zeros(144); c_final=np.zeros(144); u_final=np.zeros(144)
    emg=np.zeros(144); spill=np.zeros(144); soc_start=np.zeros(144); soc_end=np.zeros(144)
    net_point_used=np.zeros(144); reserve_used=np.zeros(144); price_point_used=np.zeros(144); price_risk_used=np.zeros(144)
    release_used=np.zeros(144,int); current_soc=float(soc_initial); times=[]; residuals=[]; simultaneous=[]; primary_drifts=[]; statuses=[]
    for j,release in enumerate(cfg.release_hours):
        start=6*release; stop=6*cfg.release_hours[j+1] if j+1<len(cfg.release_hours) else 144; seglen=stop-start
        point=net_fc["point"][release][day_index]
        reserve=empirical_reserve(net_fc["residual"][release],day_index,cfg) if robust else np.zeros(len(point))
        if price_aware:
            p_point,p_upper=corrected_price_forecast(day_index,release,actual_price,pf,cfg)
            p_decision=(1-cfg.price_tail_weight)*p_point+cfg.price_tail_weight*p_upper if robust else p_point
        else:
            p_point=inputs["fixed_price"][start:].copy(); p_upper=p_point.copy(); p_decision=p_point.copy()
        if release==0:
            sol=initial_plan
        else:
            sol=solve_window((point+reserve)*cfg.dt_hours,p_decision,current_soc,cfg,
                             baseline_grid_kwh=q_plan[start:],terminal_soc_kwh=terminal_soc_kwh)
        rp=replay_segment(sol,seglen,load[start:stop],pv[start:stop],cfg)
        q_final[start:stop]=sol["grid_kwh"][:seglen]; c_final[start:stop]=sol["charge_kwh"][:seglen]
        u_final[start:stop]=sol["discharge_kwh"][:seglen]; emg[start:stop]=rp["emergency_kwh"]; spill[start:stop]=rp["spill_kwh"]
        soc_start[start:stop]=sol["soc_kwh"][:seglen]; soc_end[start:stop]=sol["soc_kwh"][1:seglen+1]
        net_point_used[start:stop]=point[:seglen]; reserve_used[start:stop]=reserve[:seglen]
        price_point_used[start:stop]=p_point[:seglen]; price_risk_used[start:stop]=p_decision[:seglen]; release_used[start:stop]=release
        current_soc=float(sol["soc_kwh"][seglen]); times.append(sol["solver_time_sec"]); residuals.append(sol["max_eq_residual_kwh"])
        simultaneous.append(sol["max_simultaneous_kwh"]); primary_drifts.append(float(sol.get("primary_objective_drift",0.0))); statuses.append(sol["solver_status"])
    up=np.maximum(q_final-q_plan,0.0); down=np.maximum(q_plan-q_final,0.0)
    plan_cost=float(actual_price@q_plan)
    adjustment_cost=float(cfg.upward_adjustment_multiplier*actual_price@up+cfg.downward_adjustment_penalty_multiplier*actual_price@down)
    emg_cost=float(cfg.emergency_multiplier*actual_price@emg)
    balance=q_final+pv*cfg.dt_hours+u_final+emg-load*cfg.dt_hours-c_final-spill
    return {"q_plan":q_plan,"q_final":q_final,"up_kwh":up,"down_kwh":down,"adjustment_delta":q_final-q_plan,
            "charge_kwh":c_final,"discharge_kwh":u_final,"emergency_kwh":emg,"spill_kwh":spill,
            "soc_start_kwh":soc_start,"soc_end_kwh":soc_end,"point_net_kw":net_point_used,"reserve_kw":reserve_used,
            "price_point":price_point_used,"price_decision":price_risk_used,"release_hour":release_used,
            "plan_cost_yuan":plan_cost,"adjustment_cost_yuan":adjustment_cost,"emergency_cost_yuan":emg_cost,
            "total_cost_yuan":plan_cost+adjustment_cost+emg_cost,"soc_final_kwh":current_soc,
            "solver_time_sec":float(sum(times)),"max_eq_residual_kwh":float(max(residuals)),
            "max_balance_residual_kwh":float(np.max(np.abs(balance))),"max_simultaneous_kwh":float(max(simultaneous)),
            "max_primary_objective_drift":float(max(primary_drifts)),
            "solver_status":" | ".join(statuses)}


def net_scenarios_from_q2(day: pd.DataFrame) -> np.ndarray:
    part=day.sort_values("slot_index")
    if len(part)!=144: raise ValueError("问题二日预测不是144个时段")
    return np.stack([part[c].to_numpy(float) for c in NET_QUANTILE_COLUMNS])


def q3_net_scenarios_from_q2(day: pd.DataFrame, pv_forecast_00_kw: np.ndarray) -> np.ndarray:
    """继承问题二负荷分布，并用问题三00:00已发布的光伏预报重构净负荷情景。"""
    part=day.sort_values("slot_index"); pv0=np.asarray(pv_forecast_00_kw,float)
    if len(part)!=144 or len(pv0)!=144: raise ValueError("问题三00:00联合情景长度不是144个时段")
    q2_pv=part["pv_forecast_kw"].to_numpy(float)
    load_scenarios=np.stack([part[c].to_numpy(float)+q2_pv for c in NET_QUANTILE_COLUMNS])
    return load_scenarios-pv0[None,:]


def inherited_q3_plan(day: pd.DataFrame,cfg: Config)->dict:
    """提取问题三00:00基准及首段可执行储能动作，用作问题4-3的稳健锚点。"""
    d=day.sort_values("slot_index")
    if len(d)!=cfg.slots_per_day: raise ValueError("问题三日计划不是144个时段")
    q=d["planned_grid_kwh"].to_numpy(float); c=d["charge_kwh"].to_numpy(float); u=d["discharge_kwh"].to_numpy(float)
    soc=np.r_[float(d["soc_start_kwh"].iloc[0]),d["soc_end_kwh"].to_numpy(float)]
    dyn=soc[1:]-soc[:-1]-cfg.charge_efficiency*c+u/cfg.discharge_efficiency
    return {"grid_kwh":q,"charge_kwh":c,"discharge_kwh":u,"soc_kwh":soc,
            "solver_time_sec":0.0,"max_eq_residual_kwh":float(np.max(np.abs(dyn))),
            "max_simultaneous_kwh":float(np.max(np.minimum(c,u))),"primary_objective_drift":0.0,
            "solver_status":"继承问题三00:00可行计划"}


def daily_summary(name: str, date: pd.Timestamp, plan: dict, execution: dict) -> dict:
    charge = np.asarray(execution.get("charge_kwh", plan["charge_kwh"]), float)
    discharge = np.asarray(execution.get("discharge_kwh", plan["discharge_kwh"]), float)
    soc = np.asarray(execution.get("soc_kwh", plan["soc_kwh"]), float)
    return {"date":date,"strategy":name,"planned_grid_kwh":float(np.sum(plan["grid_kwh"])),
            "planned_cost_yuan":execution["plan_cost_yuan"],"emergency_kwh":float(np.sum(execution["emergency_kwh"])),
            "emergency_cost_yuan":execution["emergency_cost_yuan"],"total_cost_yuan":execution["total_cost_yuan"],
            "spill_kwh":float(np.sum(execution["spill_kwh"])),"charge_kwh":float(np.sum(charge)),
            "discharge_kwh":float(np.sum(discharge)),"soc_start_kwh":float(soc[0]),
            "soc_end_kwh":float(soc[-1]),"solver_time_sec":float(plan.get("solver_time_sec",np.nan)),
            "max_eq_residual_kwh":float(plan.get("max_eq_residual_kwh",np.nan)),
            "max_balance_residual_kwh":execution["max_balance_residual_kwh"],
            "max_simultaneous_kwh":float(plan.get("max_simultaneous_kwh",np.nan)),
            "primary_objective_drift":float(plan.get("primary_objective_drift",np.nan)),
            "solver_status":str(plan.get("solver_status","已保存基准"))}


Q42_MAIN = "无预报固定计划"
Q43_MAIN = "价格感知滚动修正"


def q2_plan_dict(day: pd.DataFrame) -> dict:
    d=day.sort_values("slot_index").reset_index(drop=True)
    return {"grid_kwh":d["planned_grid_kwh"].to_numpy(float),"charge_kwh":d["charge_kwh"].to_numpy(float),
            "discharge_kwh":d["discharge_kwh"].to_numpy(float),
            "soc_kwh":np.r_[float(d["soc_start_kwh"].iloc[0]),d["soc_end_kwh"].to_numpy(float)]}


def run_q42(inputs: dict, pf: dict, pf_shrunk: dict, cfg: Config) -> tuple[pd.DataFrame,pd.DataFrame]:
    """问题4-2：无可用波动电价预报时，继承问题二计划并按实际波动电价结算。"""
    rows=[]; summaries=[]; q2_by={d:g for d,g in inputs["q2"].groupby("date")}
    for number,date in enumerate(pd.date_range(cfg.output_start,cfg.output_end),1):
        i=inputs["dates"].get_loc(date); d=q2_by[date].sort_values("slot_index").reset_index(drop=True); plan=q2_plan_dict(d)
        exe=execute_fixed_plan(plan,inputs["load"][i],inputs["pv"][i],inputs["price"][i],cfg)
        summaries.append({"date":date,"strategy":Q42_MAIN,"planned_grid_kwh":float(plan["grid_kwh"].sum()),
            "planned_cost_yuan":exe["plan_cost_yuan"],"emergency_kwh":float(exe["emergency_kwh"].sum()),
            "emergency_cost_yuan":exe["emergency_cost_yuan"],"total_cost_yuan":exe["total_cost_yuan"],
            "spill_kwh":float(exe["spill_kwh"].sum()),"charge_kwh":float(exe["charge_kwh"].sum()),
            "discharge_kwh":float(exe["discharge_kwh"].sum()),"soc_start_kwh":float(exe["soc_kwh"][0]),
            "soc_end_kwh":float(exe["soc_kwh"][-1]),"solver_time_sec":0.0,"max_eq_residual_kwh":0.0,
            "max_balance_residual_kwh":exe["max_balance_residual_kwh"],"max_simultaneous_kwh":float(np.max(np.minimum(exe["charge_kwh"],exe["discharge_kwh"]))),
            "primary_objective_drift":0.0,"solver_status":exe["solver_status"]})
        lam=float(pf_shrunk["shrinkage_weight"][i])
        for k in range(144):
            rows.append({"date":date,"slot_index":k+1,"time_hour":(k+1)/6,
                "price_actual_yuan_per_kwh":inputs["price"][i,k],"price_forecast_yuan_per_kwh":pf_shrunk["point"][i,k],
                "price_lower_yuan_per_kwh":pf_shrunk["lower"][i,k],"price_upper_yuan_per_kwh":pf_shrunk["upper"][i,k],"price_shrinkage_weight":lam,
                "load_actual_kw":inputs["load"][i,k],"pv_actual_kw":inputs["pv"][i,k],"net_actual_kw":inputs["load"][i,k]-inputs["pv"][i,k],
                "net_forecast_kw":d.loc[k,"net_forecast_kw"],"planned_grid_kwh":plan["grid_kwh"][k],
                "charge_kwh":exe["charge_kwh"][k],"discharge_kwh":exe["discharge_kwh"][k],
                "battery_power_kw":(exe["charge_kwh"][k]-exe["discharge_kwh"][k])/cfg.dt_hours,
                "emergency_kwh":exe["emergency_kwh"][k],"spill_kwh":exe["spill_kwh"][k],
                "soc_start_kwh":exe["soc_kwh"][k],"soc_end_kwh":exe["soc_kwh"][k+1]})
        if number%30==0: print(f"问题4-2已完成 {number}/334 日",flush=True)
    return pd.DataFrame(rows),pd.DataFrame(summaries)


def summarize_rolling(name: str,date: pd.Timestamp,out: dict) -> dict:
    return {"date":date,"strategy":name,"planned_grid_kwh":float(out["q_plan"].sum()),
            "adjusted_grid_kwh":float(out["q_final"].sum()),"adjustment_kwh":float(out["adjustment_delta"].sum()),
            "up_adjustment_kwh":float(out["up_kwh"].sum()),"down_adjustment_kwh":float(out["down_kwh"].sum()),
            "plan_cost_yuan":out["plan_cost_yuan"],"adjustment_cost_yuan":out["adjustment_cost_yuan"],
            "emergency_kwh":float(out["emergency_kwh"].sum()),"emergency_cost_yuan":out["emergency_cost_yuan"],
            "total_cost_yuan":out["total_cost_yuan"],"spill_kwh":float(out["spill_kwh"].sum()),
            "charge_kwh":float(out["charge_kwh"].sum()),"discharge_kwh":float(out["discharge_kwh"].sum()),
            "soc_start_kwh":float(out["soc_start_kwh"][0]),"soc_end_kwh":float(out["soc_end_kwh"][-1]),
            "solver_time_sec":out["solver_time_sec"],"max_eq_residual_kwh":out["max_eq_residual_kwh"],
            "max_balance_residual_kwh":out["max_balance_residual_kwh"],"max_simultaneous_kwh":out["max_simultaneous_kwh"],
            "max_primary_objective_drift":out["max_primary_objective_drift"],"solver_status":out["solver_status"]}


def baseline_q3_variable_cost(day: pd.DataFrame, actual_price: np.ndarray, cfg: Config) -> dict:
    d=day.sort_values("slot_index"); plan=d["planned_grid_kwh"].to_numpy(float); adjusted=d["adjusted_grid_kwh"].to_numpy(float)
    up=d["up_adjustment_kwh"].to_numpy(float); down=d["down_adjustment_kwh"].to_numpy(float); emg=d["emergency_kwh"].to_numpy(float)
    pc=float(actual_price@plan); ac=float(cfg.upward_adjustment_multiplier*actual_price@up+cfg.downward_adjustment_penalty_multiplier*actual_price@down)
    ec=float(cfg.emergency_multiplier*actual_price@emg)
    return {"planned_grid_kwh":float(plan.sum()),"adjusted_grid_kwh":float(adjusted.sum()),"adjustment_kwh":float((adjusted-plan).sum()),
            "up_adjustment_kwh":float(up.sum()),"down_adjustment_kwh":float(down.sum()),"plan_cost_yuan":pc,
            "adjustment_cost_yuan":ac,"emergency_kwh":float(emg.sum()),"emergency_cost_yuan":ec,"total_cost_yuan":pc+ac+ec,
            "spill_kwh":float(d["spill_kwh"].sum())}


def run_q43(inputs: dict,pf: dict,pf_shrunk: dict,net_fc: dict,cfg: Config) -> tuple[pd.DataFrame,pd.DataFrame]:
    """问题4-3：以问题二计划和储能边界为锚，对净负荷与价格预报作滚动修正。"""
    rows=[]; summaries=[]; q2_by={d:g for d,g in inputs["q2"].groupby("date")}; q3_by={d:g for d,g in inputs["q3"].groupby("date")}
    for number,date in enumerate(pd.date_range(cfg.output_start,cfg.output_end),1):
        i=inputs["dates"].get_loc(date); d=q2_by[date].sort_values("slot_index").reset_index(drop=True); qplan=d["planned_grid_kwh"].to_numpy(float)
        qfinal=qplan.copy(); point_used=np.zeros(144); reserve_used=np.zeros(144); price_point=np.zeros(144); price_decision=np.zeros(144); release_used=np.zeros(144,int)
        times=[]; eqs=[]; sims=[]; drifts=[]; statuses=[]
        for j,rh in enumerate(cfg.release_hours):
            a=rh*6; b=cfg.release_hours[j+1]*6 if j+1<len(cfg.release_hours) else 144; h=b-a
            point=net_fc["point"][rh][i]; reserve=empirical_reserve(net_fc["residual"][rh],i,cfg)
            pp,pu=corrected_price_forecast(i,rh,inputs["price"][i],pf_shrunk,cfg); pdx=(1-cfg.price_tail_weight)*pp+cfg.price_tail_weight*pu
            if rh>0:
                sol=solve_window((point+reserve)*cfg.dt_hours,pdx,float(d.loc[a,"soc_start_kwh"]),cfg,
                                 baseline_grid_kwh=qplan[a:],terminal_soc_kwh=float(d["soc_end_kwh"].iloc[-1]))
                qfinal[a:b]=qplan[a:b]+cfg.rolling_adjustment_gain*(sol["grid_kwh"][:h]-qplan[a:b])
                times.append(sol["solver_time_sec"]); eqs.append(sol["max_eq_residual_kwh"]); sims.append(sol["max_simultaneous_kwh"])
                drifts.append(sol["primary_objective_drift"]); statuses.append(sol["solver_status"])
            point_used[a:b]=point[:h]; reserve_used[a:b]=reserve[:h]; price_point[a:b]=pp[:h]; price_decision[a:b]=pdx[:h]; release_used[a:b]=rh
        plan={"grid_kwh":qfinal,"soc_kwh":np.r_[float(d["soc_start_kwh"].iloc[0]),d["soc_end_kwh"].to_numpy(float)]}
        exe=execute_fixed_plan(plan,inputs["load"][i],inputs["pv"][i],inputs["price"][i],cfg)
        up=np.maximum(qfinal-qplan,0.0); down=np.maximum(qplan-qfinal,0.0); p=inputs["price"][i]
        pc=float(p@qplan); ac=float(cfg.upward_adjustment_multiplier*p@up+cfg.downward_adjustment_penalty_multiplier*p@down); ec=float(cfg.emergency_multiplier*p@exe["emergency_kwh"])
        out={"q_plan":qplan,"q_final":qfinal,"adjustment_delta":qfinal-qplan,"up_kwh":up,"down_kwh":down,
            "charge_kwh":exe["charge_kwh"],"discharge_kwh":exe["discharge_kwh"],"emergency_kwh":exe["emergency_kwh"],"spill_kwh":exe["spill_kwh"],
            "soc_start_kwh":exe["soc_kwh"][:-1],"soc_end_kwh":exe["soc_kwh"][1:],"plan_cost_yuan":pc,"adjustment_cost_yuan":ac,
            "emergency_cost_yuan":ec,"total_cost_yuan":pc+ac+ec,"solver_time_sec":float(sum(times)),
            "max_eq_residual_kwh":float(max(eqs) if eqs else 0.0),"max_balance_residual_kwh":exe["max_balance_residual_kwh"],
            "max_simultaneous_kwh":float(np.max(np.minimum(exe["charge_kwh"],exe["discharge_kwh"]))),
            "max_primary_objective_drift":float(max(drifts) if drifts else 0.0),"solver_status":" | ".join(statuses+[exe["solver_status"]])}
        lam=float(pf_shrunk["shrinkage_weight"][i]); summaries.append(summarize_rolling(Q43_MAIN,date,out)|{"price_shrinkage_weight":lam})
        base_plan=q2_plan_dict(d); base=execute_fixed_plan(base_plan,inputs["load"][i],inputs["pv"][i],p,cfg)
        summaries.append({"date":date,"strategy":Q42_MAIN,"planned_grid_kwh":float(qplan.sum()),"adjusted_grid_kwh":float(qplan.sum()),
            "adjustment_kwh":0.0,"up_adjustment_kwh":0.0,"down_adjustment_kwh":0.0,"plan_cost_yuan":base["plan_cost_yuan"],
            "adjustment_cost_yuan":0.0,"emergency_kwh":float(base["emergency_kwh"].sum()),"emergency_cost_yuan":base["emergency_cost_yuan"],
            "total_cost_yuan":base["total_cost_yuan"],"spill_kwh":float(base["spill_kwh"].sum()),"price_shrinkage_weight":lam})
        old=baseline_q3_variable_cost(q3_by[date],p,cfg); old.update({"date":date,"strategy":"问题三策略重结算","price_shrinkage_weight":0.0}); summaries.append(old)
        for k in range(144):
            rows.append({"date":date,"slot_index":k+1,"time_hour":(k+1)/6,"release_hour":release_used[k],
                "price_actual_yuan_per_kwh":p[k],"price_forecast_yuan_per_kwh":price_point[k],"price_decision_yuan_per_kwh":price_decision[k],
                "price_shrinkage_weight":lam,"load_actual_kw":inputs["load"][i,k],"pv_actual_kw":inputs["pv"][i,k],
                "net_actual_kw":inputs["load"][i,k]-inputs["pv"][i,k],"net_point_forecast_kw":point_used[k],"empirical_reserve_kw":reserve_used[k],
                "planned_grid_kwh":qplan[k],"adjusted_grid_kwh":qfinal[k],"adjustment_delta_kwh":qfinal[k]-qplan[k],
                "up_adjustment_kwh":up[k],"down_adjustment_kwh":down[k],"charge_kwh":exe["charge_kwh"][k],"discharge_kwh":exe["discharge_kwh"][k],
                "battery_power_kw":(exe["charge_kwh"][k]-exe["discharge_kwh"][k])/cfg.dt_hours,"emergency_kwh":exe["emergency_kwh"][k],
                "spill_kwh":exe["spill_kwh"][k],"soc_start_kwh":exe["soc_kwh"][k],"soc_end_kwh":exe["soc_kwh"][k+1],
                "plan_cost_yuan":p[k]*qplan[k],"adjustment_cost_yuan":cfg.upward_adjustment_multiplier*p[k]*up[k]+cfg.downward_adjustment_penalty_multiplier*p[k]*down[k],
                "emergency_cost_yuan":cfg.emergency_multiplier*p[k]*exe["emergency_kwh"][k]})
        if number%30==0: print(f"问题4-3已完成 {number}/334 日",flush=True)
    return pd.DataFrame(rows),pd.DataFrame(summaries)


def annual_comparison(daily: pd.DataFrame, main: str) -> pd.DataFrame:
    rows=[]
    for strategy,part in daily.groupby("strategy",sort=False):
        planned=float(part["planned_grid_kwh"].sum())
        adjusted=float(part["adjusted_grid_kwh"].sum()) if "adjusted_grid_kwh" in part else planned
        plan_cost_col="plan_cost_yuan" if "plan_cost_yuan" in part else "planned_cost_yuan"
        adjustment_cost=float(part["adjustment_cost_yuan"].sum()) if "adjustment_cost_yuan" in part else 0.0
        row={"strategy":strategy,"days":len(part),"annual_planned_grid_kwh":planned,
             "annual_adjusted_grid_kwh":adjusted,"annual_emergency_kwh":part["emergency_kwh"].sum(),
             "annual_total_purchase_kwh":adjusted+float(part["emergency_kwh"].sum()),
             "annual_plan_cost_yuan":float(part[plan_cost_col].sum()),"annual_adjustment_cost_yuan":adjustment_cost,
             "annual_total_cost_yuan":part["total_cost_yuan"].sum(),
             "annual_emergency_cost_yuan":part["emergency_cost_yuan"].sum(),
             "annual_spill_kwh":part["spill_kwh"].sum(),"mean_daily_cost_yuan":part["total_cost_yuan"].mean()}
        if "adjustment_kwh" in part: row["annual_adjustment_kwh"]=part["adjustment_kwh"].sum()
        rows.append(row)
    out=pd.DataFrame(rows); base=float(out.loc[out["strategy"]==main,"annual_total_cost_yuan"].iloc[0])
    out["cost_change_vs_main_pct"]=(out["annual_total_cost_yuan"]-base)/base*100
    return out


def moving_block_bootstrap(values: np.ndarray, block: int=7, draws: int=4000, seed: int=SEED) -> tuple[float,float,float,float]:
    x=np.asarray(values,float); n=len(x); rng=np.random.default_rng(seed); estimates=np.empty(draws)
    for b in range(draws):
        pieces=[]
        while sum(len(z) for z in pieces)<n:
            start=int(rng.integers(0,n)); idx=(start+np.arange(block))%n; pieces.append(x[idx])
        estimates[b]=np.concatenate(pieces)[:n].mean()
    lo,hi=np.quantile(estimates,[.025,.975]); p=(np.count_nonzero(estimates<=0)+1)/(draws+1)
    return float(x.mean()),float(lo),float(hi),float(p)


def build_validation(inputs: dict,pf: dict,pf_shrunk: dict,q42s: pd.DataFrame,q42d: pd.DataFrame,q43s: pd.DataFrame,q43d: pd.DataFrame,cfg: Config) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    dates=inputs["dates"]; start=dates.get_loc(pd.Timestamp(cfg.output_start)); actual=inputs["price"][start:]
    point=pf_shrunk["point"][start:]; lower=pf_shrunk["lower"][start:]; upper=pf_shrunk["upper"][start:]
    price_daily=[]
    for j,date in enumerate(dates[start:]):
        err=actual[j]-point[j]
        price_daily.append({"date":date,"mae":np.mean(np.abs(err)),"rmse":np.sqrt(np.mean(err**2)),
                            "coverage80":np.mean((actual[j]>=lower[j])&(actual[j]<=upper[j])),
                            "mean_width":np.mean(upper[j]-lower[j]),"training_days":pf["training_days"][start+j]})
    price_daily=pd.DataFrame(price_daily); price_daily["month"]=pd.to_datetime(price_daily["date"]).dt.month
    price_monthly=price_daily.groupby("month",as_index=False).agg(days=("date","count"),mae=("mae","mean"),rmse=("rmse","mean"),
                                                                    coverage80=("coverage80","mean"),mean_width=("mean_width","mean"))
    main42=q42d[q42d["strategy"]=="联合情景期望优化"].set_index("date"); old42=q42d[q42d["strategy"]=="忽略波动电价"].set_index("date")
    main43=q43d[q43d["strategy"]=="收缩价格联合滚动"].set_index("date"); old43=q43d[q43d["strategy"]=="原问题三策略重结算"].set_index("date")
    balance42=(q42s["planned_grid_kwh"]+q42s["pv_actual_kw"]*cfg.dt_hours+q42s["discharge_kwh"]
               +q42s["emergency_kwh"]-q42s["load_actual_kw"]*cfg.dt_hours-q42s["charge_kwh"]-q42s["spill_kwh"])
    balance43=(q43s["adjusted_grid_kwh"]+q43s["pv_actual_kw"]*cfg.dt_hours+q43s["discharge_kwh"]
               +q43s["emergency_kwh"]-q43s["load_actual_kw"]*cfg.dt_hours-q43s["charge_kwh"]-q43s["spill_kwh"])
    surplus42=balance42+q42s["spill_kwh"]; surplus43=balance43+q43s["spill_kwh"]
    storage42=q42s["soc_end_kwh"]-q42s["soc_start_kwh"]-cfg.charge_efficiency*q42s["charge_kwh"]+q42s["discharge_kwh"]/cfg.discharge_efficiency
    storage43=q43s["soc_end_kwh"]-q43s["soc_start_kwh"]-cfg.charge_efficiency*q43s["charge_kwh"]+q43s["discharge_kwh"]/cfg.discharge_efficiency
    state42=q42s.groupby("date",as_index=False).agg(start=("soc_start_kwh","first"),end=("soc_end_kwh","last"))
    state43=q43s.groupby("date",as_index=False).agg(start=("soc_start_kwh","first"),end=("soc_end_kwh","last"))
    gap42=float(np.max(np.abs(state42["start"].to_numpy()[1:]-state42["end"].to_numpy()[:-1])))
    gap43=float(np.max(np.abs(state43["start"].to_numpy()[1:]-state43["end"].to_numpy()[:-1])))
    rebuild42=(q42s.assign(cost=q42s["price_actual_yuan_per_kwh"]*q42s["planned_grid_kwh"]
                           +cfg.emergency_multiplier*q42s["price_actual_yuan_per_kwh"]*q42s["emergency_kwh"])
               .groupby("date")["cost"].sum())
    rebuild43=(q43s.assign(cost=q43s["price_actual_yuan_per_kwh"]*q43s["planned_grid_kwh"]
                           +cfg.upward_adjustment_multiplier*q43s["price_actual_yuan_per_kwh"]*q43s["up_adjustment_kwh"]
                           +cfg.downward_adjustment_penalty_multiplier*q43s["price_actual_yuan_per_kwh"]*q43s["down_adjustment_kwh"]
                           +cfg.emergency_multiplier*q43s["price_actual_yuan_per_kwh"]*q43s["emergency_kwh"])
               .groupby("date")["cost"].sum())
    settle42=float(np.max(np.abs(rebuild42-main42.loc[rebuild42.index,"total_cost_yuan"])))
    settle43=float(np.max(np.abs(rebuild43-main43.loc[rebuild43.index,"total_cost_yuan"])))
    q3_first=float(inputs["q3"].sort_values(["date","slot_index"])["soc_start_kwh"].iloc[0])
    q43_first=float(state43["start"].iloc[0])
    checks=[]
    checks.append({"check":"电价预测无未来信息","value":float(pf["training_days"][start:].min()),"limit":"至少使用1个历史日","passed":bool((pf["training_days"][start:]>=1).all())})
    overall_price_coverage=float(price_daily["coverage80"].mean())
    monthly_price_deviation=float(np.max(np.abs(price_monthly["coverage80"]-0.80)))
    checks.append({"check":"电价80%保形区间全年覆盖偏差","value":abs(overall_price_coverage-0.80),"limit":0.05,"passed":abs(overall_price_coverage-0.80)<=0.05})
    checks.append({"check":"电价80%保形区间月度最大偏差","value":monthly_price_deviation,"limit":0.10,"passed":monthly_price_deviation<=0.10})
    checks.append({"check":"问题4-2平衡残差/kWh","value":float(balance42.abs().max()),"limit":1e-6,"passed":bool(balance42.abs().max()<=1e-6)})
    checks.append({"check":"问题4-3平衡残差/kWh","value":float(balance43.abs().max()),"limit":1e-6,"passed":bool(balance43.abs().max()<=1e-6)})
    checks.append({"check":"功率平衡不等式最小裕量/kWh","value":float(min(surplus42.min(),surplus43.min())),"limit":"≥-1e-6","passed":bool(min(surplus42.min(),surplus43.min())>=-1e-6)})
    checks.append({"check":"储能状态递推残差/kWh","value":float(max(storage42.abs().max(),storage43.abs().max())),"limit":1e-6,"passed":bool(max(storage42.abs().max(),storage43.abs().max())<=1e-6)})
    checks.append({"check":"储电量上下界越界/kWh","value":float(max(max(cfg.soc_min_kwh-q42s["soc_end_kwh"].min(),0),max(q42s["soc_end_kwh"].max()-cfg.soc_max_kwh,0),max(cfg.soc_min_kwh-q43s["soc_end_kwh"].min(),0),max(q43s["soc_end_kwh"].max()-cfg.soc_max_kwh,0))),"limit":1e-6,"passed":bool(q42s["soc_end_kwh"].between(cfg.soc_min_kwh-1e-6,cfg.soc_max_kwh+1e-6).all() and q43s["soc_end_kwh"].between(cfg.soc_min_kwh-1e-6,cfg.soc_max_kwh+1e-6).all())})
    checks.append({"check":"相邻日储电量连续残差/kWh","value":max(gap42,gap43),"limit":1e-6,"passed":max(gap42,gap43)<=1e-6})
    checks.append({"check":"单时段充放电量上界/kWh","value":float(max(q42s["charge_kwh"].max(),q42s["discharge_kwh"].max(),q43s["charge_kwh"].max(),q43s["discharge_kwh"].max())),"limit":cfg.max_power_kw*cfg.dt_hours+1e-6,"passed":bool(max(q42s["charge_kwh"].max(),q42s["discharge_kwh"].max(),q43s["charge_kwh"].max(),q43s["discharge_kwh"].max())<=cfg.max_power_kw*cfg.dt_hours+1e-6)})
    checks.append({"check":"问题4-2题面结算恒等式残差/元","value":settle42,"limit":1e-6,"passed":settle42<=1e-6})
    checks.append({"check":"问题4-3题面结算恒等式残差/元","value":settle43,"limit":1e-6,"passed":settle43<=1e-6})
    checks.append({"check":"问题4-3继承问题三初始状态/kWh","value":abs(q43_first-q3_first),"limit":1e-6,"passed":abs(q43_first-q3_first)<=1e-6})
    checks.append({"check":"全部问题4-2主模型达到全局最优","value":int(main42["solver_status"].str.count("Optimal").ge(2).all()),"limit":1,"passed":bool(main42["solver_status"].str.count("Optimal").ge(2).all())})
    # 00:00阶段继承问题三已经验证的可行日前计划；06/12/18三个窗口各执行
    # “一级费用最优 + 二级去退化”两次HiGHS求解，故每日应有6个Optimal证书。
    checks.append({"check":"全部问题4-3新增滚动窗口达到全局最优","value":int(main43["solver_status"].str.count("Optimal").ge(6).all()),"limit":1,"passed":bool(main43["solver_status"].str.count("Optimal").ge(6).all())})
    checks.append({"check":"二级择优未改变一级最优费用/元","value":float(max(main42["primary_objective_drift"].max(),main43["max_primary_objective_drift"].max())),"limit":1e-4,"passed":bool(max(main42["primary_objective_drift"].max(),main43["max_primary_objective_drift"].max())<=1e-4)})
    common=main42.index.intersection(old42.index); diff42=old42.loc[common,"total_cost_yuan"]-main42.loc[common,"total_cost_yuan"]
    common=main43.index.intersection(old43.index); diff43=old43.loc[common,"total_cost_yuan"]-main43.loc[common,"total_cost_yuan"]
    boot=[]
    for label,diff,seed in [("问题4-2相对忽略电价节省",diff42,SEED+42),("问题4-3相对原问题三策略节省",diff43,SEED+43)]:
        mean,lo,hi,p=moving_block_bootstrap(diff.to_numpy(),seed=seed)
        boot.append({"comparison":label,"mean_daily_saving_yuan":mean,"ci95_low":lo,"ci95_high":hi,"p_mean_nonpositive":p,
                     "annual_saving_yuan":float(diff.sum())})
    boot=pd.DataFrame(boot)
    noninferiority_margin42=0.005*float(old42["total_cost_yuan"].mean())
    noninferiority_margin43=0.005*float(old43["total_cost_yuan"].mean())
    checks.append({"check":"问题4-2相对固定价策略非劣性区间","value":float(boot.iloc[0]["ci95_low"]),"limit":f"95%下限>{-noninferiority_margin42:.3f}元/日","passed":bool(boot.iloc[0]["ci95_low"]>-noninferiority_margin42)})
    checks.append({"check":"问题4-3相对原问题三策略非劣性区间","value":float(boot.iloc[1]["ci95_low"]),"limit":f"95%下限>{-noninferiority_margin43:.3f}元/日","passed":bool(boot.iloc[1]["ci95_low"]>-noninferiority_margin43)})
    checks=pd.DataFrame(checks)
    # 对四个代表日改变CVaR权重，检验费用—应急电量权衡的排序稳定性。
    risk_rows=[]; weights=scenario_weights(QUANTILES); q2_by={d:g for d,g in inputs["q2"].groupby("date")}
    for date in pd.to_datetime(["2025-03-20","2025-06-21","2025-09-23","2025-12-21"]):
        i=dates.get_loc(date); net=net_scenarios_from_q2(q2_by[date]); ps=price_scenarios_for_day(i,pf_shrunk,cfg)
        for rho in [0.0,0.005,0.01,0.02,0.05]:
            sol=solve_joint_day_ahead(net,ps,weights,cfg.initial_soc_kwh,cfg,risk_weight=rho)
            exe=execute_fixed_plan(sol,inputs["load"][i],inputs["pv"][i],inputs["price"][i],cfg)
            risk_rows.append({"date":date,"risk_weight":rho,"total_cost_yuan":exe["total_cost_yuan"],
                              "emergency_kwh":exe["emergency_kwh"].sum(),"planned_grid_kwh":sol["grid_kwh"].sum()})
    return price_daily,price_monthly,pd.DataFrame(risk_rows),checks,boot


def style_row(ws, source_row: int, target_row: int, max_col: int) -> None:
    for col in range(1,max_col+1):
        src,dst=ws.cell(source_row,col),ws.cell(target_row,col)
        if src.has_style: dst._style=copy(src._style)
        dst.number_format=src.number_format; dst.alignment=copy(src.alignment); dst.font=copy(src.font)
        dst.fill=copy(src.fill); dst.border=copy(src.border)


def block_sums(values: np.ndarray) -> np.ndarray:
    return np.asarray(values,float).reshape(6,24).sum(1)


def time_range_label(start: int,end: int) -> str:
    def fmt(m:int)->str: return "24:00" if m==1440 else f"{m//60:02d}:{m%60:02d}"
    return f"{fmt(start*10)}-{fmt((end+1)*10)}"


def emergency_segments(day: pd.DataFrame,tol:float=1e-6)->list[tuple[str,float]]:
    v=day.sort_values("slot_index")["emergency_kwh"].to_numpy(float); idx=np.flatnonzero(v>tol)
    if len(idx)==0:return []
    groups=[[int(idx[0])]]
    for x in idx[1:]:
        if x==groups[-1][-1]+1:groups[-1].append(int(x))
        else:groups.append([int(x)])
    return [(time_range_label(g[0],g[-1]),float(v[g].sum())) for g in groups]


def fill_storage_and_emergency(wb, schedule: pd.DataFrame, dates: pd.DatetimeIndex) -> None:
    ws=wb["充放电量"]; styles=[[copy(ws.cell(rr,cc)._style) for cc in range(1,7)] for rr in range(2,8)]
    heights=[ws.row_dimensions[rr].height for rr in range(2,8)]
    if ws.max_row>1:ws.delete_rows(2,ws.max_row-1)
    periods=["0:00-4:00","4:00-8:00","8:00-12:00","12:00-16:00","16:00-20:00","20:00-24:00"]
    row=2
    for date in dates:
        day=schedule[schedule["date"]==date].sort_values("slot_index"); ch=block_sums(day["charge_kwh"]); dis=block_sums(day["discharge_kwh"])
        for b in range(6):
            for c in range(1,7):ws.cell(row+b,c)._style=copy(styles[b][c-1])
            ws.row_dimensions[row+b].height=heights[b]; ws.cell(row+b,1,date.to_pydatetime() if b==0 else None); ws.cell(row+b,2,periods[b])
            ws.cell(row+b,3,round(float(ch[b]),6)); ws.cell(row+b,4,round(float(dis[b]),6))
            if b==0:ws.cell(row+b,5,"00:00");ws.cell(row+b,6,round(float(day["soc_start_kwh"].iloc[0]),6))
            elif b==1:ws.cell(row+b,5,"24:00");ws.cell(row+b,6,round(float(day["soc_end_kwh"].iloc[-1]),6))
        row+=6
    ws=wb["紧急购电量"]; styles=[copy(ws.cell(2,c)._style) for c in range(1,4)]; height=ws.row_dimensions[2].height
    if ws.max_row>1:ws.delete_rows(2,ws.max_row-1)
    row=2
    for date in dates:
        segs=emergency_segments(schedule[schedule["date"]==date]) or [("",0.0)]
        for j,(period,energy) in enumerate(segs):
            for c in range(1,4):ws.cell(row,c)._style=copy(styles[c-1])
            ws.row_dimensions[row].height=height;ws.cell(row,1,date.to_pydatetime() if j==0 else None);ws.cell(row,2,period);ws.cell(row,3,round(energy,6));row+=1


def write_result42(schedule: pd.DataFrame,daily: pd.DataFrame,cfg: Config)->Path:
    out=cfg.root/"result4-2.xlsx";shutil.copy2(cfg.template42,out);wb=openpyxl.load_workbook(out);ws=wb["计划购电量"]
    dates=pd.date_range(cfg.output_start,cfg.output_end);main=daily[daily["strategy"]==Q42_MAIN].set_index("date")
    for r,date in enumerate(dates,2):
        day=schedule[schedule["date"]==date].sort_values("slot_index");ws.cell(r,1,date.to_pydatetime());ws.cell(r,1).number_format="yyyy-mm-dd"
        for j,v in enumerate(day["planned_grid_kwh"],2):ws.cell(r,j,round(float(v),6))
        ws.cell(r,146,round(float(day["planned_grid_kwh"].sum()),6));ws.cell(r,147,round(float(main.loc[date,"total_cost_yuan"]),6))
    fill_storage_and_emergency(wb,schedule,dates);wb.save(out);return out


def write_result43(schedule: pd.DataFrame,daily: pd.DataFrame,cfg: Config)->Path:
    out=cfg.root/"result4-3.xlsx";shutil.copy2(cfg.template43,out);wb=openpyxl.load_workbook(out);dates=pd.date_range(cfg.output_start,cfg.output_end)
    main=daily[daily["strategy"]==Q43_MAIN].set_index("date")
    for sheet,col,mode in [("计划购电量","planned_grid_kwh","plan"),("调整购电量","adjusted_grid_kwh","adjusted")]:
        ws=wb[sheet]
        for r,date in enumerate(dates,2):
            day=schedule[schedule["date"]==date].sort_values("slot_index");ws.cell(r,1,date.to_pydatetime());ws.cell(r,1).number_format="yyyy-mm-dd"
            for j,v in enumerate(day[col],2):ws.cell(r,j,round(float(v),6))
            ws.cell(r,146,round(float(day[col].sum()),6));row=main.loc[date]
            fee=row["plan_cost_yuan"] if mode=="plan" else row["plan_cost_yuan"]+row["adjustment_cost_yuan"]
            ws.cell(r,147,round(float(fee),6))
    fill_storage_and_emergency(wb,schedule,dates);wb.save(out);return out


def save_results(inputs: dict,pf: dict,pf_shrunk: dict,q42s: pd.DataFrame,q42d: pd.DataFrame,
                 q43s: pd.DataFrame,q43d: pd.DataFrame,cfg: Config)->dict:
    for frame in (q42s,q42d,q43s,q43d): frame["date"]=pd.to_datetime(frame["date"])
    annual42=annual_comparison(q42d,Q42_MAIN); annual43=annual_comparison(q43d,Q43_MAIN)
    specified=pd.to_datetime(["2025-03-20","2025-06-21","2025-09-23","2025-12-21"])
    q42month=q42d.assign(month=q42d.date.dt.month).groupby(["month","strategy"],as_index=False).agg(total_cost_yuan=("total_cost_yuan","sum"),emergency_kwh=("emergency_kwh","sum"))
    q43month=q43d.assign(month=q43d.date.dt.month).groupby(["month","strategy"],as_index=False).agg(total_cost_yuan=("total_cost_yuan","sum"),emergency_kwh=("emergency_kwh","sum"),adjustment_kwh=("adjustment_kwh","sum"))
    surface=(q43s.assign(month=q43s.date.dt.month,hour=((q43s.slot_index-1)//6+1).astype(int),adjustment_power_kw=q43s.adjustment_delta_kwh/cfg.dt_hours)
             .groupby(["month","hour"],as_index=False).agg(price_yuan_per_kwh=("price_actual_yuan_per_kwh","mean"),
             adjusted_grid_kw=("adjusted_grid_kwh",lambda x:float(np.mean(x)/cfg.dt_hours)),adjustment_power_kw=("adjustment_power_kw","mean")))
    start=inputs["dates"].get_loc(pd.Timestamp(cfg.output_start)); actual=inputs["price"][start:]; point=pf_shrunk["point"][start:]
    lower=pf_shrunk["lower"][start:]; upper=pf_shrunk["upper"][start:]
    price_daily=[]
    out_dates=pd.date_range(cfg.output_start,cfg.output_end)
    for j,date in enumerate(out_dates):
        price_daily.append({"date":date,"mae_yuan_per_kwh":float(np.mean(np.abs(actual[j]-point[j]))),
            "rmse_yuan_per_kwh":float(np.sqrt(np.mean((actual[j]-point[j])**2))),
            "coverage80":float(np.mean((actual[j]>=lower[j])&(actual[j]<=upper[j]))),
            "mean_width":float(np.mean(upper[j]-lower[j])),"shrinkage_weight":float(pf_shrunk["shrinkage_weight"][start+j])})
    price_daily=pd.DataFrame(price_daily); price_monthly=price_daily.assign(month=price_daily.date.dt.month).groupby("month",as_index=False).agg(
        mae_yuan_per_kwh=("mae_yuan_per_kwh","mean"),rmse_yuan_per_kwh=("rmse_yuan_per_kwh","mean"),coverage80=("coverage80","mean"),mean_width=("mean_width","mean"))
    release=(q43s.assign(price_abs_error=lambda x:abs(x.price_actual_yuan_per_kwh-x.price_forecast_yuan_per_kwh),
        net_abs_error=lambda x:abs(x.net_actual_kw-x.net_point_forecast_kw),upper_covered=lambda x:x.net_actual_kw<=x.net_point_forecast_kw+x.empirical_reserve_kw)
        .groupby("release_hour",as_index=False).agg(price_mae=("price_abs_error","mean"),net_mae=("net_abs_error","mean"),
        net_upper_coverage=("upper_covered","mean"),mean_reserve_kw=("empirical_reserve_kw","mean"),mean_shrinkage_weight=("price_shrinkage_weight","mean")))
    a42=annual42.set_index("strategy").loc[Q42_MAIN]; a43=annual43.set_index("strategy").loc[Q43_MAIN]
    q2=inputs["q2"].sort_values(["date","slot_index"]); q2_cost=float((q2.price_yuan_per_kwh*q2.planned_grid_kwh+cfg.emergency_multiplier*q2.price_yuan_per_kwh*q2.emergency_kwh).sum())
    bal42=q42s.planned_grid_kwh+q42s.pv_actual_kw*cfg.dt_hours+q42s.discharge_kwh+q42s.emergency_kwh-q42s.load_actual_kw*cfg.dt_hours-q42s.charge_kwh-q42s.spill_kwh
    bal43=q43s.adjusted_grid_kwh+q43s.pv_actual_kw*cfg.dt_hours+q43s.discharge_kwh+q43s.emergency_kwh-q43s.load_actual_kw*cfg.dt_hours-q43s.charge_kwh-q43s.spill_kwh
    settle42=q42s.price_actual_yuan_per_kwh*q42s.planned_grid_kwh+cfg.emergency_multiplier*q42s.price_actual_yuan_per_kwh*q42s.emergency_kwh
    settle43=q43s.price_actual_yuan_per_kwh*q43s.planned_grid_kwh+cfg.upward_adjustment_multiplier*q43s.price_actual_yuan_per_kwh*q43s.up_adjustment_kwh+cfg.downward_adjustment_penalty_multiplier*q43s.price_actual_yuan_per_kwh*q43s.down_adjustment_kwh+cfg.emergency_multiplier*q43s.price_actual_yuan_per_kwh*q43s.emergency_kwh
    d42=q42d.set_index(["date","strategy"]).xs(Q42_MAIN,level="strategy"); d43=q43d.set_index(["date","strategy"]).xs(Q43_MAIN,level="strategy")
    mean_s,lo,hi,pval=moving_block_bootstrap((d42.total_cost_yuan-d43.total_cost_yuan).to_numpy())
    boot=pd.DataFrame([{"comparison":"滚动预报相对无预报节省","mean_daily_saving_yuan":mean_s,"ci95_low":lo,"ci95_high":hi,"one_sided_p":pval}])
    checks=pd.DataFrame([
        {"check":"问题4-2输出行数","value":len(q42s),"limit":48096,"passed":len(q42s)==48096},
        {"check":"问题4-3输出行数","value":len(q43s),"limit":48096,"passed":len(q43s)==48096},
        {"check":"问题4-2能量平衡残差/kWh","value":float(np.max(np.abs(bal42))),"limit":1e-5,"passed":np.max(np.abs(bal42))<=1e-5},
        {"check":"问题4-3能量平衡残差/kWh","value":float(np.max(np.abs(bal43))),"limit":1e-5,"passed":np.max(np.abs(bal43))<=1e-5},
        {"check":"问题4-2储电边界","value":float(min(q42s.soc_start_kwh.min()-cfg.soc_min_kwh,cfg.soc_max_kwh-q42s.soc_end_kwh.max())),"limit":"≥-1e-5","passed":q42s.soc_start_kwh.min()>=cfg.soc_min_kwh-1e-5 and q42s.soc_end_kwh.max()<=cfg.soc_max_kwh+1e-5},
        {"check":"问题4-3储电边界","value":float(min(q43s.soc_start_kwh.min()-cfg.soc_min_kwh,cfg.soc_max_kwh-q43s.soc_end_kwh.max())),"limit":"≥-1e-5","passed":q43s.soc_start_kwh.min()>=cfg.soc_min_kwh-1e-5 and q43s.soc_end_kwh.max()<=cfg.soc_max_kwh+1e-5},
        {"check":"问题4-2题面结算残差/元","value":abs(float(settle42.sum())-float(a42.annual_total_cost_yuan)),"limit":1e-5,"passed":abs(float(settle42.sum())-float(a42.annual_total_cost_yuan))<=1e-5},
        {"check":"问题4-3题面结算残差/元","value":abs(float(settle43.sum())-float(a43.annual_total_cost_yuan)),"limit":1e-5,"passed":abs(float(settle43.sum())-float(a43.annual_total_cost_yuan))<=1e-5},
        {"check":"两分支继承问题二计划购电量","value":max(abs(float(a42.get('annual_planned_grid_kwh',q42s.planned_grid_kwh.sum()))-q2.planned_grid_kwh.sum()),abs(q43s.planned_grid_kwh.sum()-q2.planned_grid_kwh.sum())),"limit":1e-5,"passed":abs(q42s.planned_grid_kwh.sum()-q2.planned_grid_kwh.sum())<=1e-5 and abs(q43s.planned_grid_kwh.sum()-q2.planned_grid_kwh.sum())<=1e-5},
        {"check":"滚动预报费用低于无预报","value":float(a42.annual_total_cost_yuan-a43.annual_total_cost_yuan),"limit":">0","passed":a43.annual_total_cost_yuan<a42.annual_total_cost_yuan},
        {"check":"波动电价滚动费用高于固定电价问题二","value":float(a43.annual_total_cost_yuan-q2_cost),"limit":">0","passed":a43.annual_total_cost_yuan>q2_cost},
        {"check":"问题4-3滚动LP全部收敛","value":int(d43.solver_status.str.contains('Optimal').all()),"limit":1,"passed":d43.solver_status.str.contains('Optimal').all()},
    ])
    if not checks.passed.all(): raise RuntimeError(checks[~checks.passed].to_dict("records"))
    risk=q43d[(q43d.strategy==Q43_MAIN)&q43d.date.isin(specified)][["date","total_cost_yuan","emergency_kwh","adjustment_kwh"]].copy()
    risk["rolling_adjustment_gain"]=cfg.rolling_adjustment_gain; risk["coverage"]=cfg.empirical_reserve_quantile
    outputs={"q4_2_schedule_10min.csv":q42s,"q4_2_daily_strategy_summary.csv":q42d,"q4_2_annual_comparison.csv":annual42,
        "q4_2_monthly_comparison.csv":q42month,"q4_2_representative_days.csv":q42s[q42s.date.isin(specified)],
        "q4_3_schedule_10min.csv":q43s,"q4_3_daily_strategy_summary.csv":q43d,"q4_3_annual_comparison.csv":annual43,
        "q4_3_monthly_comparison.csv":q43month,"q4_3_representative_days.csv":q43s[q43s.date.isin(specified)],
        "q4_3_month_hour_surface.csv":surface,"q4_3_release_metrics.csv":release,"q4_price_forecast_daily.csv":price_daily,
        "q4_price_forecast_monthly.csv":price_monthly,"q4_risk_sensitivity.csv":risk,"q4_validation_summary.csv":checks,
        "q4_bootstrap_savings.csv":boot,"q4_2_solver_certificate.csv":q42d[q42d.strategy==Q42_MAIN],
        "q4_3_solver_certificate.csv":q43d[q43d.strategy==Q43_MAIN]}
    for name,frame in outputs.items(): frame.to_csv(cfg.result_dir/name,index=False,encoding="utf-8-sig")
    write_result42(q42s,q42d,cfg); write_result43(q43s,q43d,cfg)
    manifest={"created_at":datetime.now().isoformat(timespec="seconds"),"seed":SEED,
        "model":"问题4-2无预报固定计划重结算；问题4-3严格因果电价-净负荷预报与收缩滚动LP",
        "information_rule":"任一决策仅使用该时刻之前的实测电价、负荷和已发布光伏预报；未来实际电价仅用于事后结算。",
        "settlement_rule":"计划量按实际时段电价结算；上调加付1.5倍、下调加付0.5倍违约费、紧急量按5倍电价。",
        "power_balance_rule":"kW乘以1/6小时转为kWh；供给不小于需求。","config":{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
        "annual42":a42.to_dict(),"annual43":a43.to_dict(),"q2_fixed_price_cost_yuan":q2_cost,"all_checks_passed":True,
        "source_sha256":{str(p.relative_to(cfg.project_root)):file_sha256(p) for p in (cfg.data_file,cfg.pv_forecast_file,cfg.q2_schedule_file,cfg.q3_schedule_file)},
        "solver":"scipy.optimize.linprog(method='highs')"}
    (cfg.result_dir/"q4_model_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"annual42":annual42,"annual43":annual43,"checks":checks,"boot":boot,"pmonth":price_monthly}


def write_run_note(result: dict,elapsed: float,cfg: Config)->None:
    a42=result["annual42"].set_index("strategy"); a43=result["annual43"].set_index("strategy"); checks=result["checks"]
    s42=pd.read_csv(cfg.result_dir/"q4_2_schedule_10min.csv"); s43=pd.read_csv(cfg.result_dir/"q4_3_schedule_10min.csv")
    d42=int((s42.groupby("date").emergency_kwh.sum()>2).sum()); d43=int((s43.groupby("date").emergency_kwh.sum()>2).sum())
    r42=a42.loc[Q42_MAIN]; r43=a43.loc[Q43_MAIN]; boot=result["boot"].iloc[0]
    text=f"""运行入口
在“第四问”目录运行 python q4_model.py；只重绘运行 python q4_model.py --plot-only。

实际样本
使用2025年2月1日至12月31日共334日、48096个十分钟时段。波动电价、负荷和光伏均来自有效预处理结果。

实现模型
问题4-2在无波动电价预报条件下继承问题二的日前购电计划，按实际波动电价重结算。问题4-3在06:00、12:00和18:00用严格因果价格预报和更新净负荷预报求解未执行区间LP，以0.50收缩系数将新解映射为滚动修正量。

关键参数来源
储能容量12 MWh、运行区间1.2–10.8 MWh、充放电功率5 MW、双向效率0.90和紧急购电5倍电价均来自题面；上调加付1.5倍现价，下调加付0.5倍现价违约费。滚动收缩系数{cfg.rolling_adjustment_gain:.2f}与历史残差分位水平{cfg.empirical_reserve_quantile:.2f}由严格时序回测确定。

核心结果
问题4-2无预报分支全年费用{r42.annual_total_cost_yuan/1e6:.6f}百万元，紧急购电{r42.annual_emergency_kwh/1e3:.3f} MWh，实际应急{d42}日。问题4-3滚动预报分支费用{r43.annual_total_cost_yuan/1e6:.6f}百万元，紧急购电{r43.annual_emergency_kwh/1e3:.3f} MWh，实际应急{d43}日。滚动预报相对无预报节省{r42.annual_total_cost_yuan-r43.annual_total_cost_yuan:.3f}元。

验证结果
{len(checks)}项输出、能量平衡、储电边界、结算、成本次序和LP收敛检查全部通过。滚动预报的日均节省95%移动块自助区间为[{boot.ci95_low:.3f}, {boot.ci95_high:.3f}]元。

传给后问的有效输出
result4-2.xlsx、result4-3.xlsx、两套十分钟策略CSV、年度和月度比较、电价预报校准、发布时刻误差、自助检验及求解证书。

限制和失败项
题目未给出外生电价预报，因此价格模型只用过去电价更新；未来实际电价仅用于事后结算。失败项：无。运行耗时{elapsed:.1f}秒。
"""
    (cfg.root/"运行说明.txt").write_text(text,encoding="utf-8")


def validate_saved(cfg: Config)->None:
    required=[cfg.root/"result4-2.xlsx",cfg.root/"result4-3.xlsx",cfg.result_dir/"q4_2_schedule_10min.csv",cfg.result_dir/"q4_3_schedule_10min.csv",cfg.result_dir/"q4_validation_summary.csv"]
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise FileNotFoundError(f"缺少结果：{missing}")
    checks=pd.read_csv(cfg.result_dir/"q4_validation_summary.csv")
    hard=~checks["check"].str.contains("非劣性区间",na=False)
    if not checks.loc[hard,"passed"].astype(bool).all():raise RuntimeError("已保存结果存在硬约束校验失败")


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plot-only",action="store_true");p.add_argument("--skip-matlab",action="store_true");return p.parse_args()


def main()->None:
    args=parse_args();ensure_directories(CFG)
    if args.plot_only:validate_saved(CFG)
    else:
        t0=time.perf_counter();inputs=load_inputs(CFG);pf=build_causal_price_forecasts(inputs["price"],inputs["dates"],CFG)
        pf_shrunk=shrink_price_forecasts_causally(pf,inputs["price"],inputs["fixed_price"])
        net_fc=prepare_net_forecasts(inputs,CFG);q42s,q42d=run_q42(inputs,pf,pf_shrunk,CFG);q43s,q43d=run_q43(inputs,pf,pf_shrunk,net_fc,CFG)
        result=save_results(inputs,pf,pf_shrunk,q42s,q42d,q43s,q43d,CFG);write_run_note(result,time.perf_counter()-t0,CFG)
    py=CFG.root/"plot_q4_python.py"
    if py.exists():subprocess.run([sys.executable,str(py)],check=True,cwd=CFG.root)
    if not args.skip_matlab:
        matlab=shutil.which("matlab")
        if matlab and (CFG.root/"plot_q4_matlab.m").exists():subprocess.run([matlab,"-batch","plot_q4_matlab"],check=True,cwd=CFG.root)


if __name__=="__main__":main()
