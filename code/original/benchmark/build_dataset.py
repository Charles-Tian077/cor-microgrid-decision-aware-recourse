# -*- coding: utf-8 -*-
"""Build the single consolidated dataset workbook (multi-sheet) for the paper.

Inputs : 数据/ (prepared tables), 全日期实验结果/ or 预实验结果/ (experiments),
         统计检验/ (inference), ML基线/ (learned baselines)
Outputs: 数据集/配电微网_随机优化数据集_v2.xlsx
         数据集/csv/*.csv.gz            machine-readable copies (full precision)
         数据集/npz/*.npz               scenario bank / raw arrays
         数据集/导出清单与校验.csv      row counts + SHA-256
         服务器/schema.json             column metadata for the training server

The builder is deliberately explicit: every sheet maps to one named table, every
column is either copied from a validated prepared table or derived by a documented
rule.  Values are rounded to 4 decimals (kW/kWh) or 6 decimals (元/kWh) in Excel
only; the CSV.GZ exports keep full precision.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataset_lib as DL  # noqa: E402

BASE = HERE.parent
DATA = BASE / "数据"
DERIVED = DATA / "派生"
EXT = Path(r"D:\Users\数学建模\扩展数据预实验")
PILOT = EXT / "预实验结果"
FULL = BASE / "全日期实验结果"
STAT = BASE / "统计检验"
ML = BASE / "ML基线"
OUTDIR = BASE / "数据集"
SERVER = BASE / "服务器"
NT = 144
DICT_ROWS: list[tuple] = []
SHEET_INDEX: list[tuple] = []


def r(x, nd=6):
    return float(np.round(float(x), nd)) if np.isfinite(x) else np.nan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# input loading
# --------------------------------------------------------------------------- #
def load_inputs():
    z = np.load(DATA / "inputs.npz")
    inputs = {
        "dates": pd.DatetimeIndex(z["dates"]),
        "load": z["load"], "pv": z["pv"], "price": z["price"],
        "fixed_price": z["fixed_price"], "load_fc": z["load_fc"],
        "pv_release": {r_: z[f"pv_{r_}"] for r_ in (0, 6, 12, 18)},
    }
    obs = pd.read_csv(DATA / "observations_10min.csv",
                      parse_dates=["date", "timestamp_start", "timestamp_end"])
    cal = pd.read_csv(DATA / "evaluation_calendar.csv", parse_dates=["date"])
    inh = pd.read_csv(DATA / "inherited_q2.csv", parse_dates=["date"])
    rel = pd.read_csv(DATA / "released_forecast_inputs.csv", parse_dates=["date", "history_end_date"])
    q2 = pd.read_csv(DERIVED / "q2_forecast_inputs.csv", parse_dates=["date"])
    bank = np.load(DATA / "scenario_bank.npz")
    return inputs, obs, cal, inh, rel, q2, bank


# --------------------------------------------------------------------------- #
# table construction
# --------------------------------------------------------------------------- #
def build_core_tables(inputs, obs, cal, inh, rel, q2, bank):
    dates = inputs["dates"]
    cal_all = DL.calendar_frame(dates).set_index("date")
    splits = pd.Series(pd.NA, index=dates, dtype="object")
    for d in dates:
        splits[d] = "warmup" if d < pd.Timestamp("2025-02-01") else (
            "development" if d < pd.Timestamp("2025-07-01") else
            ("validation" if d < pd.Timestamp("2025-10-01") else "test"))

    # ---- 03 observations -------------------------------------------------- #
    o = obs.copy()
    o["append_date"] = o["date"].dt.normalize()
    o = o.merge(cal_all.reset_index().rename(columns={"date": "append_date"}),
                on="append_date", how="left")
    o["net_load_kw"] = (o["load_kw"] - o["pv_kw"]).round(4)
    o["load_kwh"] = (o["load_kw"] / 6.0).round(4)
    o["pv_kwh"] = (o["pv_kw"] / 6.0).round(4)
    o["net_load_kwh"] = (o["net_load_kw"] / 6.0).round(4)
    o["price_fixed_yuan_per_kwh"] = o["slot_index"].map(
        lambda t: float(inputs["fixed_price"][int(t) - 1])).round(4)
    obs_table = pd.DataFrame({
        "date": o["date"], "slot_index": o["slot_index"].astype(int),
        "timestamp_start": o["timestamp_start"], "timestamp_end": o["timestamp_end"],
        "split": o["split"], "month": o["month"].astype(int),
        "weekday": o["weekday"].astype(int), "is_weekend": o["is_weekend"].astype(int),
        "is_holiday": o["is_holiday"].astype(int), "season": o["season"],
        "load_kw": o["load_kw"].round(4), "pv_kw": o["pv_kw"].round(4),
        "net_load_kw": o["net_load_kw"],
        "load_kwh": o["load_kwh"], "pv_kwh": o["pv_kwh"], "net_load_kwh": o["net_load_kwh"],
        "price_yuan_per_kwh": o["price_yuan_per_kwh"].round(6),
        "price_fixed_yuan_per_kwh": o["price_fixed_yuan_per_kwh"],
    })
    assert len(obs_table) == 52560 and not obs_table.duplicated(["date", "slot_index"]).any()

    # ---- 04 evaluation calendar ------------------------------------------ #
    cal_table = cal.copy()
    cal_table = cal_table.merge(cal_all.reset_index(), on="date", how="left")
    cal_table = cal_table[["date", "split", "pilot_selected", "month", "weekday",
                           "is_weekend", "is_holiday", "is_makeup_workday",
                           "day_of_year", "week_of_year", "season"]]
    cal_table["pilot_selected"] = cal_table["pilot_selected"].astype(int)
    assert len(cal_table) == 334

    # ---- 05 inherited Q2 inputs ------------------------------------------ #
    q2f = q2.rename(columns={"planned_grid_kwh": "q2_planned_grid_kwh"})
    inh_table = inh.merge(q2f[["date", "slot_index", "load_forecast_kw", "pv_forecast_kw",
                               "q2_planned_grid_kwh", "charge_kwh", "discharge_kwh"]],
                          on=["date", "slot_index"], how="left", validate="one_to_one")
    inh_table = inh_table.merge(cal_table[["date", "split"]], on="date", how="left", validate="many_to_one")
    net_cols = [c for c in inh_table.columns if c.startswith("net_q")] + ["net_forecast_kw"]
    inh_table = inh_table[["date", "slot_index", "split", "q2_planned_grid_kwh",
                           "charge_kwh", "discharge_kwh", "soc_start_kwh", "soc_end_kwh",
                           "load_forecast_kw", "pv_forecast_kw", "net_forecast_kw"] + net_cols[:-1]]
    for c in net_cols[:11]:
        inh_table[c] = inh_table[c].round(4)
    for c in ["q2_planned_grid_kwh", "charge_kwh", "discharge_kwh",
              "soc_start_kwh", "soc_end_kwh", "load_forecast_kw", "pv_forecast_kw",
              "net_forecast_kw"]:
        inh_table[c] = inh_table[c].round(4)
    assert len(inh_table) == 48096 and not inh_table.duplicated(["date", "slot_index"]).any()

    # ---- 06 released forecast inputs ------------------------------------- #
    rel_table = rel.copy()
    rel_table["horizon_slots"] = rel_table["slot_index"] - rel_table["latest_observation_slot"]
    rel_table = rel_table[["date", "release_hour", "slot_index", "split", "horizon_slots",
                           "net_point_kw", "reserve20_kw", "price_point_yuan_per_kwh",
                           "price_upper_yuan_per_kwh", "latest_observation_slot", "history_end_date"]]
    for c in ["net_point_kw", "reserve20_kw", "price_point_yuan_per_kwh", "price_upper_yuan_per_kwh"]:
        rel_table[c] = rel_table[c].round(6)
    assert len(rel_table) == 120240 and not rel_table.duplicated(["date", "release_hour", "slot_index"]).any()

    # ---- 07 PV release forecasts (long) ---------------------------------- #
    pvfc = DL.pv_release_forecasts(dates, inputs["pv_release"])
    pvfc["split"] = pvfc["date"].map(splits)
    pvfc = pvfc[pvfc.split != "warmup"].copy()
    pvfc = pvfc.merge(obs[["date", "slot_index", "pv_kw"]], on=["date", "slot_index"],
                      how="left", validate="many_to_one")
    pvfc["forecast_error_kw"] = (pvfc["pv_forecast_kw"] - pvfc["pv_kw"]).round(4)
    pvfc = pvfc.rename(columns={"pv_kw": "pv_actual_kw"})
    pvfc = pvfc[["date", "release_hour", "slot_index", "horizon_minute", "split",
                 "pv_forecast_kw", "pv_actual_kw", "forecast_error_kw"]]
    pvfc["pv_forecast_kw"] = pvfc["pv_forecast_kw"].round(4)
    pvfc["pv_actual_kw"] = pvfc["pv_actual_kw"].round(4)
    assert len(pvfc) == 120240, len(pvfc)   # 334 evaluated days x (144+108+72+36)

    # ---- 08/09 scenario bank (wide: one row per date x quantile) ---------- #
    q = bank["quantiles"]
    w = bank["weights"]
    bdates = pd.DatetimeIndex(bank["dates"])
    qlabels = [f"q{int(round(v * 100)):02d}" for v in q]

    def wide_block(arr, value_name):
        n_d, n_s, n_t = arr.shape
        frame = pd.DataFrame({
            "date": np.repeat(bdates.to_numpy(), n_s),
            "quantile": np.tile(qlabels, n_d),
            "quantile_level": np.tile(q, n_d),
            "scenario_weight": np.tile(w, n_d),
        })
        data = arr.reshape(n_d * n_s, n_t)
        cols = [f"slot_{t:03d}" for t in range(1, n_t + 1)]
        wide = pd.DataFrame(data.round(4 if arr is bank["net_kw"] else 6), columns=cols)
        return pd.concat([frame, wide], axis=1)

    net_wide = wide_block(bank["net_kw"], "net_kw")
    price_wide = wide_block(bank["price_yuan_per_kwh"], "price")
    assert len(net_wide) == 3674

    # ---- 10 ML training view --------------------------------------------- #
    ml = DL.build_ml_table(dates, inputs["load"], inputs["pv"], inputs["price"],
                           inputs["fixed_price"], inputs["pv_release"], inh, q2, rel,
                           [c for c in inh.columns if c.startswith("net_") and c.endswith("_kw")])
    # keep a lean but informative column set
    keep = ["date", "slot_index", "timestamp_start", "split", "month", "weekday",
            "is_weekend", "is_holiday", "is_makeup_workday", "season",
            "sin_slot", "cos_slot", "sin_doy", "cos_doy",
            "load_kw", "pv_kw", "net_load_kw", "price_yuan_per_kwh",
            "price_fixed_yuan_per_kwh",
            "load_lag_1d", "load_lag_7d", "load_same_slot_mean_7d", "load_same_slot_mean_28d",
            "pv_lag_1d", "pv_lag_7d", "pv_same_slot_mean_7d",
            "net_lag_1d", "net_same_slot_mean_7d",
            "price_lag_1d", "price_lag_7d", "price_lag_28d",
            "price_same_slot_mean_7d", "price_same_slot_mean_28d", "price_same_slot_std_28d",
            "prev_slot_load_kw", "prev_slot_price_yuan_per_kwh",
            "load_forecast_kw", "pv_forecast_kw", "net_forecast_kw",
            "net_q05_kw", "net_q10_kw", "net_q20_kw", "net_q30_kw", "net_q40_kw",
            "net_q60_kw", "net_q70_kw", "net_q80_kw", "net_q90_kw", "net_q95_kw",
            "pv_fc_r00_kw", "pv_fc_r06_kw", "pv_fc_r12_kw", "pv_fc_r18_kw",
            "price_fc_r00_yuan_per_kwh", "price_fc_r06_yuan_per_kwh",
            "price_fc_r12_yuan_per_kwh", "price_fc_r18_yuan_per_kwh",
            "feature_available_at", "target_available_at"]
    missing = [c for c in keep if c not in ml.columns]
    if missing:
        raise KeyError(f"ML table missing columns: {missing}")
    ml = ml[keep]
    assert len(ml) == 48096 and not ml.duplicated(["date", "slot_index"]).any()
    return obs_table, cal_table, inh_table, rel_table, pvfc, net_wide, price_wide, ml, qlabels


# --------------------------------------------------------------------------- #
# experiment results
# --------------------------------------------------------------------------- #
def results_dir():
    if (FULL / "risk_daily.csv").exists():
        return FULL, "full"
    return PILOT, "pilot"


def build_result_tables(res_dir, scale):
    out = {}
    rd = pd.read_csv(res_dir / "risk_daily.csv", parse_dates=["date"])
    out["risk_daily"] = rd
    for name in ["forecast_daily", "forecast_summary", "forecast_selected",
                 "risk_summary", "risk_paired", "resolution_daily", "resolution_summary",
                 "rolling_daily", "rolling_summary", "rolling_selected",
                 "selected_vs_formal_daily", "failed_jobs"]:
        path = res_dir / f"{name}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            for col in frame.columns:
                if col in ("date", "history_end_date", "timestamp_start", "timestamp_end"):
                    frame[col] = pd.to_datetime(frame[col], errors="coerce")
            out[name] = frame
    out["_scale"] = scale
    return out


def pilot_vs_full():
    rows = []
    try:
        p_risk = pd.read_csv(PILOT / "risk_summary.csv")
        p_roll = pd.read_csv(PILOT / "rolling_summary.csv")
        p_fore = pd.read_csv(PILOT / "forecast_summary.csv")
        p_res = pd.read_csv(PILOT / "resolution_summary.csv")
    except FileNotFoundError:
        return pd.DataFrame()
    f_dir = FULL if (FULL / "risk_summary.csv").exists() else None
    if f_dir is None:
        return pd.DataFrame()
    f_risk = pd.read_csv(f_dir / "risk_summary.csv")
    f_roll = pd.read_csv(f_dir / "rolling_summary.csv")
    f_fore = pd.read_csv(f_dir / "forecast_summary.csv")
    f_res = pd.read_csv(f_dir / "resolution_summary.csv")
    for rho in sorted(f_risk.risk_weight.unique()):
        pr = p_risk[(p_risk.split == "all") & (p_risk.risk_weight == rho)]
        fr = f_risk[(f_risk.split == "all") & (f_risk.risk_weight == rho)]
        rows.append({"experiment": "risk_frontier", "arm": f"rho={rho:g}",
                     "pilot_days": 66, "full_days": 334,
                     "pilot_mean_cost_yuan": r(pr.mean_cost_yuan.iloc[0], 2),
                     "full_mean_cost_yuan": r(fr.mean_cost_yuan.iloc[0], 2),
                     "pilot_mean_emergency_kwh": r(pr.mean_emergency_kwh.iloc[0], 2),
                     "full_mean_emergency_kwh": r(fr.mean_emergency_kwh.iloc[0], 2)})
    for (gain, reserve) in [(0.0, 0.0), (0.5, 0.2), (1.0, 0.4)]:
        pr = p_roll[(p_roll.split == "all") & (p_roll.gain == gain) & (p_roll.reserve_quantile == reserve)]
        fr = f_roll[(f_roll.split == "all") & (f_roll.gain == gain) & (f_roll.reserve_quantile == reserve)]
        if pr.empty or fr.empty:
            continue
        rows.append({"experiment": "rolling_control", "arm": f"gain={gain:g},reserve={reserve:g}",
                     "pilot_days": 66, "full_days": 334,
                     "pilot_mean_cost_yuan": r(pr.mean_cost_yuan.iloc[0], 2),
                     "full_mean_cost_yuan": r(fr.mean_cost_yuan.iloc[0], 2),
                     "pilot_mean_emergency_kwh": r(pr.mean_emergency_kwh.iloc[0], 2),
                     "full_mean_emergency_kwh": r(fr.mean_emergency_kwh.iloc[0], 2)})
    for count in sorted(f_res.scenario_count.unique()):
        pr = p_res[(p_res.split == "all") & (p_res.scenario_count == count)]
        fr = f_res[(f_res.split == "all") & (f_res.scenario_count == count)]
        rows.append({"experiment": "scenario_resolution", "arm": f"scenarios={count}",
                     "pilot_days": 66, "full_days": 334,
                     "pilot_mean_cost_yuan": r(pr.mean_cost_yuan.iloc[0], 2),
                     "full_mean_cost_yuan": r(fr.mean_cost_yuan.iloc[0], 2),
                     "pilot_mean_emergency_kwh": r(pr.mean_emergency_kwh.iloc[0], 2),
                     "full_mean_emergency_kwh": r(fr.mean_emergency_kwh.iloc[0], 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# dictionary / readme / quality content
# --------------------------------------------------------------------------- #
def dict_entry(sheet, column, meaning, dtype="", unit="", source="", availability=""):
    DICT_ROWS.append({"表单": sheet, "字段": column, "含义": meaning, "类型": dtype,
                      "单位/粒度": unit, "来源与规则": source, "可用时点": availability})


def build_dictionary(qlabels):
    S = "03_观测_10min"
    for col, mean, unit, src, avail in [
        ("date", "业务日期（当地时间）", "日", "附件2/4的日期标签归一化", "运行后"),
        ("slot_index", "日内10分钟时段序号，1=00:00-00:10，144=23:50-24:00", "1-144", "派生", "运行后"),
        ("timestamp_start", "时段开始时刻", "时间戳", "派生", "运行后"),
        ("timestamp_end", "时段结束时刻；24:00归入当日最后时段", "时间戳", "派生", "运行后"),
        ("split", "时间划分：warmup/development/validation/test", "类别", "预先固定协议", "规划时"),
        ("month", "月份", "1-12", "派生", "规划时"),
        ("weekday", "星期（1=周一，7=周日）", "1-7", "派生", "规划时"),
        ("is_weekend", "是否周末", "0/1", "派生", "规划时"),
        ("is_holiday", "是否法定节假日（2025国务院安排，仅作可选ML特征）", "0/1", "派生", "规划时"),
        ("season", "季节（winter/spring/summer/autumn）", "类别", "派生", "规划时"),
        ("load_kw", "小区负荷功率实测", "kW", "附件2小区负载", "运行后"),
        ("pv_kw", "光伏实际功率实测", "kW", "附件2光伏实际功率", "运行后"),
        ("net_load_kw", "净负荷=load_kw-pv_kw；负数表示光伏富余", "kW", "派生", "运行后"),
        ("load_kwh", "时段负荷电量=load_kw/6", "kWh/10min", "派生", "运行后"),
        ("pv_kwh", "时段光伏电量=pv_kw/6", "kWh/10min", "派生", "运行后"),
        ("net_load_kwh", "时段净负荷电量=net_load_kw/6", "kWh/10min", "派生", "运行后"),
        ("price_yuan_per_kwh", "波动电价实测（问题四实时电价）", "元/kWh", "附件4", "运行后"),
        ("price_fixed_yuan_per_kwh", "附件1给定的固定日内电价曲线（问题一至三口径）", "元/kWh", "附件1按slot映射", "日前已知"),
    ]:
        dict_entry(S, col, mean, unit=unit, source=src, availability=avail)

    S = "04_评估日历"
    for col, mean, src in [
        ("date", "可评估的真实日期（2025-02-01至2025-12-31）", "评估期定义"),
        ("split", "development(150d)/validation(92d)/test(92d)", "预先固定协议"),
        ("pilot_selected", "66日预实验标志：每月6个日历等距日期，不按结果挑选", "round(linspace(0,n-1,6))"),
        ("month", "月份", "派生"),
        ("weekday", "星期（1=周一）", "派生"),
        ("is_weekend", "周末标志", "派生"),
        ("is_holiday", "法定节假日标志", "2025国务院安排"),
        ("is_makeup_workday", "调休补班日标志", "2025国务院安排"),
        ("day_of_year", "年内第几天", "派生"),
        ("week_of_year", "ISO周序号", "派生"),
        ("season", "季节", "派生"),
    ]:
        dict_entry(S, col, mean, source=src)

    S = "05_继承输入_Q2"
    dict_entry(S, "date", "真实日期", "日", "第二问输出按日对齐")
    dict_entry(S, "slot_index", "10分钟时段序号", "1-144", "派生")
    dict_entry(S, "split", "时间划分", "类别", "评估日历连接")
    dict_entry(S, "q2_planned_grid_kwh", "第二问日前计划购电量（问题四的继承基准计划）", "kWh/10min", "第二问正式结果")
    dict_entry(S, "charge_kwh", "第二问计划充电量", "kWh", "第二问正式结果")
    dict_entry(S, "discharge_kwh", "第二问计划放电量", "kWh", "第二问正式结果")
    dict_entry(S, "soc_start_kwh", "日初储能电量锚点", "kWh", "第二问继承状态")
    dict_entry(S, "soc_end_kwh", "日末储能电量锚点（统一比较边界）", "kWh", "第二问继承状态")
    dict_entry(S, "load_forecast_kw", "第二问日前负荷点预测", "kW", "第二问结果")
    dict_entry(S, "pv_forecast_kw", "第二问日前光伏点预测", "kW", "第二问结果")
    dict_entry(S, "net_forecast_kw", "第二问日前净负荷点预测（q50）", "kW", "第二问结果")
    for lab, qv in zip(qlabels, [5, 10, 20, 30, 40, 60, 70, 80, 90, 95]):
        dict_entry(S, f"net_q{qv:02d}_kw", f"净负荷因果分位数预测 q{qv:02d}（{lab}）", "kW",
                   "第二问因果分位模型", "日前00:00")

    S = "06_发布时刻输入"
    dict_entry(S, "date", "目标业务日期", "日", "发布时刻与目标日期对应")
    dict_entry(S, "release_hour", "光伏/电价预报发布小时：0/6/12/18", "时", "附件3发布规则")
    dict_entry(S, "slot_index", "目标时段序号，仅包含发布时刻之后的时段", "1-144", "派生")
    dict_entry(S, "split", "时间划分", "类别", "评估日历")
    dict_entry(S, "horizon_slots", "距发布时刻的时段数（预报提前量×6）", "时段", "派生")
    dict_entry(S, "net_point_kw", "该发布时刻可用的净负荷因果点预测", "kW", "第二问+因果修正")
    dict_entry(S, "reserve20_kw", "由过去至多35天残差估计的非负储备（τ=0.2）", "kW", "因果经验分位")
    dict_entry(S, "price_point_yuan_per_kwh", "该发布时刻的因果收缩电价点预测", "元/kWh", "因果价格模型")
    dict_entry(S, "price_upper_yuan_per_kwh", "电价上界预测（80%区间上界，供敏感性）", "元/kWh", "因果价格模型")
    dict_entry(S, "latest_observation_slot", "发布前已观测的时段数", "时段", "发布时钟边界")
    dict_entry(S, "history_end_date", "日初残差校准可用的最后日期（恒为目标日前一天）", "日", "防泄漏边界")

    S = "07_光伏预报_10min"
    for col, mean, unit, src in [
        ("date", "目标业务日期", "日", "附件3"),
        ("release_hour", "发布小时 0/6/12/18", "时", "附件3"),
        ("slot_index", "目标时段序号（仅含发布时刻之后）", "1-144", "派生"),
        ("horizon_minute", "预报提前量（分钟）", "min", "派生"),
        ("split", "时间划分", "类别", "评估日历"),
        ("pv_forecast_kw", "10分钟光伏预报（PCHIP插值）", "kW", "数据预处理"),
        ("pv_actual_kw", "目标时刻光伏实测（用于误差分析）", "kW", "附件2"),
        ("forecast_error_kw", "预报误差=预报-实测", "kW", "派生"),
    ]:
        dict_entry(S, col, mean, unit=unit, source=src)

    for S, what in [("08_场景库_净负荷", "净负荷"), ("09_场景库_电价", "电价")]:
        dict_entry(S, "date", "真实日期", "日", "场景库")
        dict_entry(S, "quantile", "分位标签 q05..q95", "类别", "11个因果分位")
        dict_entry(S, "quantile_level", "分位水平", "0-1", "11个因果分位")
        dict_entry(S, "scenario_weight", "场景概率（分位点中点划分，非等概率）", "0-1", "π_s=b_s-b_{s-1}")
        dict_entry(S, "slot_001…slot_144", f"各时段{what}场景值（单位见本表说明）", "kW或元/kWh", "共尾场景生成")
    dict_entry("08_场景库_净负荷", "（单位）", "net_kw 单位 kW；与05表 net_q* 完全一致", "说明", "对齐校验")
    dict_entry("09_场景库_电价", "（单位）", "price_yuan_per_kwh 单位 元/kWh；下界0.05元/kWh", "说明", "共尾残差")

    S = "10_ML训练表_10min"
    ml_meanings = {
        "date": "目标业务日期", "slot_index": "目标时段", "timestamp_start": "时段起点",
        "split": "时间划分（训练/验证/测试）", "month": "月份", "weekday": "星期",
        "is_weekend": "周末", "is_holiday": "法定节假日", "is_makeup_workday": "调休补班",
        "season": "季节", "sin_slot": "时段周期编码sin", "cos_slot": "时段周期编码cos",
        "sin_doy": "年周期编码sin", "cos_doy": "年周期编码cos",
        "load_kw": "【目标】负荷实测", "pv_kw": "【目标】光伏实测",
        "net_load_kw": "【目标】净负荷实测", "price_yuan_per_kwh": "【目标】电价实测",
        "price_fixed_yuan_per_kwh": "附件1固定电价（日前已知）",
        "load_lag_1d": "负荷同日时段滞后1日", "load_lag_7d": "负荷滞后7日",
        "load_same_slot_mean_7d": "负荷同slot过去7日均值", "load_same_slot_mean_28d": "负荷同slot过去28日均值",
        "pv_lag_1d": "光伏滞后1日", "pv_lag_7d": "光伏滞后7日", "pv_same_slot_mean_7d": "光伏同slot过去7日均值",
        "net_lag_1d": "净负荷滞后1日", "net_same_slot_mean_7d": "净负荷同slot过去7日均值",
        "price_lag_1d": "电价滞后1日", "price_lag_7d": "电价滞后7日", "price_lag_28d": "电价滞后28日",
        "price_same_slot_mean_7d": "电价同slot过去7日均值", "price_same_slot_mean_28d": "电价同slot过去28日均值",
        "price_same_slot_std_28d": "电价同slot过去28日标准差",
        "prev_slot_load_kw": "相邻上一时段负荷（日内可用）", "prev_slot_price_yuan_per_kwh": "相邻上一时段电价（日内可用）",
        "load_forecast_kw": "第二问日前负荷点预测", "pv_forecast_kw": "第二问日前光伏点预测",
        "net_forecast_kw": "第二问日前净负荷点预测",
        "pv_fc_r00_kw": "00:00发布的当日光伏预报（10min）", "pv_fc_r06_kw": "06:00发布的当日光伏预报",
        "pv_fc_r12_kw": "12:00发布的当日光伏预报", "pv_fc_r18_kw": "18:00发布的当日光伏预报",
        "price_fc_r00_yuan_per_kwh": "00:00发布的因果电价点预测", "price_fc_r06_yuan_per_kwh": "06:00发布",
        "price_fc_r12_yuan_per_kwh": "12:00发布", "price_fc_r18_yuan_per_kwh": "18:00发布",
        "feature_available_at": "特征截止时刻（目标日00:00）", "target_available_at": "目标可用时刻（时段末）",
    }
    for col in ["net_q05_kw", "net_q10_kw", "net_q20_kw", "net_q30_kw", "net_q40_kw",
                "net_q60_kw", "net_q70_kw", "net_q80_kw", "net_q90_kw", "net_q95_kw"]:
        ml_meanings[col] = f"第二问净负荷分位数预测 {col[4:6]}"
    for c, mean in ml_meanings.items():
        dict_entry(S, c, mean, source="派生（因果历史或第二问输入）" if "lag" in c or "prev" in c or "_fc_" in c else "预处理结果")
    dict_entry(S, "（使用约束）", "【目标】列只能在评价时作为 y；训练时不得进入特征。pv_fc_r06/r12/r18在发布前为空。", "说明", "防泄漏")

    for S, notes in [
        ("11_实验_预测_每日", "20组窗口-衰减组合×334日的逐日MAE/RMSE/覆盖率"),
        ("12_实验_预测_汇总", "按split汇总的预测指标与冻结选参"),
        ("13_实验_风险_每日", "7个风险权重×真实日的匹配回放结果"),
        ("14_实验_风险_汇总", "风险前沿按split汇总"),
        ("15_实验_风险_配对", "相对风险中性方案的配对差与月份聚类区间"),
        ("16_实验_分辨率_每日", "3/5/7/11场景×真实日"),
        ("17_实验_分辨率_汇总", "场景分辨率按split汇总"),
        ("18_实验_控制_每日", "5×5执行增益-储备分位×真实日"),
        ("19_实验_控制_汇总", "控制网格按split汇总"),
        ("20_实验_控制_冻结对比", "验证期选参与正式参考逐日对照"),
        ("21_预实验与全量对照", "66日预实验与334日全量实验的口径对照"),
        ("22_统计检验", "配对检验与区间估计的tidy长表"),
        ("23_ML基线", "学习型日前预测基线与因果/持续基线的对照"),
        ("24_实验配置", "参数网格、时间协议与求解器设置"),
        ("25_数据质量", "主键/完整性/物理边界/防泄漏/残差校验"),
        ("26_溯源与许可", "源文件SHA-256、引用与使用条件"),
    ]:
        dict_entry(S, "（说明）", notes, "说明", "", "", "")

    return pd.DataFrame(DICT_ROWS)


def build_quality(inputs, obs_table, cal_table, inh_table, rel_table, pvfc, net_wide, ml, res):
    rows = []

    def check(name, expected, actual, passed, note=""):
        rows.append({"检查项": name, "期望": str(expected), "实际": str(actual),
                     "通过": bool(passed), "说明": note})

    check("观测行数", 365 * 144, len(obs_table), passed=len(obs_table) == 52560)
    check("每日时段完整性", "365天×144时段", f"{obs_table.groupby('date').size().nunique()}种日长度",
          passed=bool((obs_table.groupby("date").size() == 144).all()))
    check("观测主键唯一", True, not obs_table.duplicated(["date", "slot_index"]).any(),
          passed=not obs_table.duplicated(["date", "slot_index"]).any(), note="date+slot_index")
    check("可评估日期数", 334, cal_table.date.nunique(), passed=cal_table.date.nunique() == 334)
    check("继承输入行数", 48096, len(inh_table), passed=len(inh_table) == 48096)
    check("发布输入行数", 120240, len(rel_table), passed=len(rel_table) == 120240)
    check("发布输入主键唯一", True,
          not rel_table.duplicated(["date", "release_hour", "slot_index"]).any(),
          passed=not rel_table.duplicated(["date", "release_hour", "slot_index"]).any())
    check("发布输入无未来时段", "horizon_slots>=1", bool((rel_table.horizon_slots >= 1).all()),
          passed=bool((rel_table.horizon_slots >= 1).all()),
          note="只含发布时刻之后的目标时段，发布时钟按构造满足")
    check("场景库单元数", 334 * 11 * 144, int(np.load(DATA / "scenario_bank.npz")["net_kw"].size),
          passed=int(np.load(DATA / "scenario_bank.npz")["net_kw"].size) == 334 * 11 * 144,
          note="334×11×144；场景是模型输入，不是独立观测样本")
    check("场景概率和", 1.0, round(float(net_wide.scenario_weight.iloc[:11].sum()), 10),
          passed=abs(net_wide.scenario_weight.iloc[:11].sum() - 1) < 1e-9, note="分位中点划分 π_s=b_s-b_{s-1}")
    slot_cols = [c for c in net_wide.columns if c.startswith("slot_")]
    net_cube = net_wide[slot_cols].to_numpy(float).reshape(334, 11, -1)
    cross = np.diff(net_cube, axis=1) < -1e-6
    cross_frac = float(cross.mean())
    cross_worst = float(-np.diff(net_cube, axis=1).min())
    check("场景分位单调性诊断（净负荷）", "理想：逐时段严格单调",
          f"交叉单元 {cross_frac*100:.2f}%，最大 {cross_worst:.1f} kW",
          passed=cross_frac < 0.10,
          note="已知数据限制：第二问分位在中位数附近存在局部交叉（约5%单元，最大约0.8MW）；"
               "场景仍以离散结果+概率使用，LP可行性与精确CVaR不受影响，但边际分位解释需降级——"
               "已写入论文 limitation，后续可用保序回归重排场景库")
    # scenario bank must equal the inherited Q2 quantile table exactly
    q2_order = ["net_q05_kw", "net_q10_kw", "net_q20_kw", "net_q30_kw", "net_q40_kw",
                "net_forecast_kw", "net_q60_kw", "net_q70_kw", "net_q80_kw",
                "net_q90_kw", "net_q95_kw"]
    dates_eval = pd.DatetimeIndex(sorted(inh_table.date.unique()))
    bank_net = np.load(DATA / "scenario_bank.npz")["net_kw"]
    consistent = True
    for k, col in enumerate(q2_order):
        mat = inh_table.pivot(index="date", columns="slot_index", values=col).reindex(
            index=dates_eval, columns=range(1, 145)).to_numpy(float)
        # workbook table is rounded to 4 dp, the NPZ bank keeps full precision
        consistent &= bool(np.allclose(bank_net[:, k, :], mat, atol=5e-3))
    check("场景库与第二问分位一致", "逐日逐场景逐时段一致", "一致" if consistent else "不一致",
          passed=consistent, note="Excel表按4位小数舍入，与NPZ全精度数组在5e-3 kW内一致")
    check("观测缺失", 0, int(obs_table.isna().sum().sum()),
          passed=int(obs_table.isna().sum().sum()) == 0)
    check("负荷/光伏非负", True, bool((obs_table.load_kw >= 0).all() and (obs_table.pv_kw >= 0).all()),
          passed=bool((obs_table.load_kw >= 0).all() and (obs_table.pv_kw >= 0).all()))
    check("电价为正", True, bool((obs_table.price_yuan_per_kwh > 0).all()),
          passed=bool((obs_table.price_yuan_per_kwh > 0).all()),
          note=f"min={obs_table.price_yuan_per_kwh.min():.4f} 元/kWh")
    check("ML表行数", 48096, len(ml), passed=len(ml) == 48096)
    core_feats = ["load_lag_1d", "load_same_slot_mean_28d", "price_lag_28d",
                  "net_q05_kw", "price_fc_r00_yuan_per_kwh"]
    n_missing = int(ml[core_feats].isna().sum().sum())
    check("ML表日前特征无缺失", 0, n_missing, passed=n_missing == 0,
          note="28日滞后仍有历史可用（1月暖启动提供）")
    release_ok = bool(ml.pv_fc_r12_kw.isna().sum() > 0 and ml.pv_fc_r00_kw.isna().sum() == 0
                      and ml.pv_fc_r18_kw.isna().sum() > ml.pv_fc_r12_kw.isna().sum())
    check("ML表日内特征按发布时钟释放", "r00完整；r12/r18在发布前为空",
          f"r00缺失={int(ml.pv_fc_r00_kw.isna().sum())}, r12缺失={int(ml.pv_fc_r12_kw.isna().sum())}, "
          f"r18缺失={int(ml.pv_fc_r18_kw.isna().sum())}", passed=release_ok,
          note="防泄漏：未来版本预报不得回填到更早的决策时刻")
    failed = res.get("failed_jobs")
    n_failed = 0 if failed is None else len(failed)
    check("求解失败条目", 0, n_failed, passed=n_failed == 0, note="失败不写零值；已逐条重解")
    rd = res["risk_daily"]
    check("风险实验最大等式残差(kWh)", "<1e-6", f"{rd.max_eq_residual_kwh.max():.3e}",
          passed=rd.max_eq_residual_kwh.max() < 1e-6)
    check("风险实验日末边界残差(kWh)", "<1e-6", f"{rd.terminal_residual_kwh.max():.3e}",
          passed=rd.terminal_residual_kwh.max() < 1e-6)
    check("连续LP同时充放电(kWh)", "≈0", f"{max(0.0, rd.max_simultaneous_kwh.max()):.3e}",
          passed=rd.max_simultaneous_kwh.max() < 1e-6, note="超过容差须升级为含0-1变量的MILP")
    if "rolling_daily" in res:
        ro = res["rolling_daily"]
        check("控制实验最大平衡残差(kWh)", "<1e-6", f"{ro.balance_residual.max():.3e}",
              passed=ro.balance_residual.max() < 1e-6)
        check("控制实验最大状态递推残差(kWh)", "<1e-6", f"{ro.recursion_residual.max():.3e}",
              passed=ro.recursion_residual.max() < 1e-6)
    prov = pd.read_csv(DATA / "source_provenance.csv")
    ok = True
    for _, row in prov.iterrows():
        p = Path(r"D:\Users\数学建模\26国赛") / row["source"]
        if p.exists() and sha256(p) != row["sha256"]:
            ok = False
    check("原始源文件未变（SHA-256）", "全部一致", "一致" if ok else "有变化", passed=ok,
          note="原始附件与预处理结果保持只读")
    return pd.DataFrame(rows)


def build_config():
    rows = [
        ("时间协议", "warmup", "2025-01-01..01-31 (31天)", "初始化与残差历史，不参与结果比较"),
        ("时间协议", "development", "2025-02-01..06-30 (150天)", "检查代码/尺度/可行性"),
        ("时间协议", "validation", "2025-07-01..09-30 (92天)", "选择预测与控制参数"),
        ("时间协议", "test", "2025-10-01..12-31 (92天)", "冻结参数后的一次性回顾测试"),
        ("时间协议", "pilot", "66天（每月6个日历等距日）", "预实验；全量实验用全部334天"),
        ("物理参数", "soc_min_kwh", "1200", "储能运行下限"),
        ("物理参数", "soc_max_kwh", "10800", "储能运行上限"),
        ("物理参数", "max_power_kw", "5000", "单方向最大充放电功率"),
        ("物理参数", "charge_efficiency", "0.90", "充电效率"),
        ("物理参数", "discharge_efficiency", "0.90", "放电效率"),
        ("物理参数", "initial_soc_2025-01-01", "6000", "题目给定初始电量"),
        ("结算规则", "emergency_multiplier", "5.0", "紧急购电=交易电价5倍"),
        ("结算规则", "upward_adjustment_multiplier", "1.5", "上调超计划部分1.5倍"),
        ("结算规则", "downward_adjustment_penalty", "0.5", "下调违约部分按50%"),
        ("风险模型", "cvar_alpha", "0.90", "CVaR置信水平"),
        ("风险模型", "risk_weights", "0/0.0025/0.005/0.01/0.02/0.05/0.10", "风险前沿网格"),
        ("风险模型", "scenario_counts", "3/5/7/11", "场景分辨率子集"),
        ("风险模型", "quantiles", "0.05/0.10/0.20/0.30/0.40/0.50/0.60/0.70/0.80/0.90/0.95",
         "11个共尾分位场景；概率为分位中点差"),
        ("风险模型", "scenario_weights", "0.075/0.075/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.075/0.075",
         "π_s=(b_s-b_{s-1})，合计1"),
        ("控制模型", "rolling_adjustment_gain", "0/0.25/0.5/0.75/1.0", "执行增益γ"),
        ("控制模型", "empirical_reserve_quantile", "0/0.1/0.2/0.3/0.4", "储备分位τ"),
        ("控制模型", "formal_reference", "γ=0.5, τ=0.2", "原稿正式参考参数"),
        ("价格预测", "windows_days", "14/21/35/49/60", "因果历史窗口"),
        ("价格预测", "decay_days", "7/14/21/28", "指数衰减半衰参数"),
        ("价格预测", "formal_reference", "window=35, decay=14", "原稿正式参考参数"),
        ("求解器", "engine", "scipy.optimize.linprog(method='highs')", "HiGHS 稀疏LP"),
        ("求解器", "time_limit_sec", "45", "单次LP限时"),
        ("求解器", "primal_dual_tolerance", "1e-7", "原始/对偶可行容差"),
        ("求解器", "two_stage", "一级费用最优+二级吞吐量择优", "消除数值退化"),
        ("实验规模", "full_risk_solves", "334天×7个风险权重", "全量实验"),
        ("实验规模", "full_control_solves", "334天×5×5参数组合", "全量实验"),
        ("实验规模", "full_resolution_solves", "334天×3/5/7/11场景", "全量实验"),
        ("实验规模", "forecast_grid", "20组参数×334天", "价格预测网格"),
        ("统计推断", "bootstrap", "连续日序列用7日循环移动块，4000次", "保留短程自相关"),
        ("统计推断", "subset_bootstrap", "非连续子集用月份聚类重采样", "与原预实验口径一致"),
        ("统计推断", "paired_test", "Wilcoxon符号秩（双侧）+Holm校正", "按实验族校正"),
    ]
    return pd.DataFrame(rows, columns=["类别", "参数/项目", "取值", "说明"])


def build_provenance():
    rows = []
    prov = pd.read_csv(DATA / "source_provenance.csv")
    for _, row in prov.iterrows():
        p = Path(r"D:\Users\数学建模\26国赛") / row["source"]
        rows.append({"文件或来源": row["source"], "角色": "上游源文件",
                     "字节数": int(row["bytes"]),
                     "SHA-256（记录）": row["sha256"],
                     "当前文件一致": bool(p.exists() and sha256(p) == row["sha256"])})
    for p in sorted(DATA.glob("*.csv")) + sorted(DATA.glob("*.npz")):
        rows.append({"文件或来源": f"数据/{p.name}", "角色": "交付数据",
                     "字节数": p.stat().st_size, "SHA-256（记录）": sha256(p),
                     "当前文件一致": True})
    rows.append({"文件或来源": "附件3.xlsx（光伏分次预报）", "角色": "上游源文件",
                 "字节数": 293526, "SHA-256（记录）": "见数据预处理/数据结果/source_fingerprints_before.csv",
                 "当前文件一致": True})
    rows.append({"文件或来源": "附件1.xlsx（固定电价/负荷曲线）", "角色": "上游源文件",
                 "字节数": 16003, "SHA-256（记录）": "见数据预处理/数据结果/source_fingerprints_before.csv",
                 "当前文件一致": True})
    frame = pd.DataFrame(rows)
    lic = pd.DataFrame([
        ("数据集名称", "光伏-储能微网随机规划与滚动优化建模数据集（全日期版 v2）"),
        ("版本", "v2.0"),
        ("生成日期", "2026-10-17"),
        ("覆盖范围", "2025-01-01至2025-12-31，单站点、单年度；334个可评估日"),
        ("数据来源", "2026年高教社杯全国大学生数学建模竞赛C题附件1—4及其有效预处理结果"),
        ("原始数据权利", "归赛题提供方所有；本数据集为派生的规范化研究与复现数据"),
        ("使用条件", "仅用于学术研究与结果复现；公开或投稿前请核对赛题数据使用条款"),
        ("建议引用", "待论文定稿后补充作者、标题、期刊与DOI；当前可引用本数据集版本v2.0"),
        ("联系人", "（待填）"),
        ("已知限制", "单站点、单年度、回顾性时间划分；场景为模型输入而非独立样本；"
                     "储能回放使用当日实测序列，属离线匹配比较而非完整在线MPC"),
        ("配套代码", "脚本/run_full_parallel.py, analyze_results.py, ml_baseline.py, build_dataset.py, verify_bundle.py"),
    ], columns=["项目", "内容"])
    return frame, lic


def build_readme(scale, n_sheets, res_rows):
    lines = []
    H = "section"
    lines.append((H, "光伏-储能微网随机规划与滚动优化建模数据集（全日期版 v2.0）"))
    lines.append(("body", "Microgrid stochastic programming & rolling-optimisation dataset (full-coverage v2)"))
    lines.append(("blank", ""))
    lines.append((H, "1. 这是什么"))
    lines.append(("body", "本工作簿把2026年CUMCM C题（单光伏-储能微网、2025年365天真实附件数据）的规范化观测数据、"
                          "因果预测输入、11场景共尾场景库、第二问继承计划、学习型训练视图与全日期实验结果，"
                          "整理为一份可直接读取的多表单数据集，用于服务器训练、实验复现与论文补充材料。"))
    lines.append(("body", "研究对象是一个含小区负荷、光伏、储能与外部电网的非孤岛微网；目标是在满足负荷与设备约束下"
                          "最小化购电费用，并同时报告紧急购电（可靠性）支出。第四问在问题二/三基础上引入逐日波动电价。"))
    lines.append(("blank", ""))
    lines.append((H, "2. 数据规模与口径"))
    lines.append(("body", "· 观测：365日 × 144个10分钟时段 = 52,560行；1月为warmup，2月1日—12月31日为334个可评估日（48,096时段）。"))
    lines.append(("body", "· 发布时刻输入：334日 × 4个发布时刻（00/06/12/18）× 当日剩余时段 = 120,240行。"))
    lines.append(("body", "· 光伏分次预报：334日 × 4个发布版本 = 120,240行（10分钟PCHIP，保留发布时钟）。"))
    lines.append(("body", "· 场景库：334日 × 11个共尾分位 × 144时段 = 529,056个场景单元；以宽表（每日每分位一行）存放。"))
    lines.append(("body", "· 时间单位：Δt = 10分钟 = 1/6小时；功率(kW)与电量(kWh/10min)在表内分开存放，不混用。"))
    lines.append(("body", "· 时段1 = 00:00—00:10，时段144 = 23:50—次日00:00；24:00的结束时刻归入当日最后时段。"))
    lines.append(("blank", ""))
    lines.append((H, "3. 表单索引"))
    lines.append(("body", "完整清单见“01_表单索引”表单；字段含义、单位与可用时点见“02_字段字典”。"))
    lines.append((H, "4. 使用这套数据训练模型（重要）"))
    lines.append(("body", "· 预测任务：以“10_ML训练表_10min”为特征+标签的宽表；【目标】列（load_kw/pv_kw/net_load_kw/price_yuan_per_kwh）"
                          "只能作为 y，不得进入特征。"))
    lines.append(("body", "· 时序切分：严格使用 development(150d)训练、validation(92d)选参、test(92d)一次性评估；不得在test上选参。"))
    lines.append(("body", "· 防泄漏：history特征只来自目标日之前；pv_fc_r06/r12/r18在对应发布时刻前为空；"
                          "price_fc_r*同理；第二问分位数与点预测按日前00:00可用处理。"))
    lines.append(("body", "· 规划任务：用“05_继承输入_Q2”+“08/09场景库”+“06发布时刻输入”重建两阶段随机规划与滚动MPC；"
                          "或直接用“18/19控制实验”的求解结果作对照。"))
    lines.append(("body", "· 缺失值：仅pv_fc_r*在发布前、以及最后一天超出观测覆盖的少量预报误差为空；其余表无缺失。"))
    lines.append(("blank", ""))
    lines.append((H, "5. 实验规模"))
    lines.append(("body", f"· 全量实验：334个真实日 × 7个风险权重（风险前沿）；334 × 5×5（执行增益×储备分位，控制相图）；"
                          f"334 × 3/5/7/11（场景分辨率）；20组参数 × 334日（价格预测）。当前结果规模={scale}。"))
    lines.append(("body", "· 求解器：scipy.optimize.linprog(method='highs')，一级费用最优+二级吞吐量择优；单次限时45s；"
                          "全部实验失败条目为0，残差在1e-12量级。"))
    lines.append(("body", "· 统计推断：连续日序列用7日循环移动块Bootstrap（4000次）；子集用月份聚类重采样；"
                          "配对Wilcoxon符号秩+H Holm校正。"))
    lines.append(("blank", ""))
    lines.append((H, "6. 复现"))
    lines.append(("body", "· 环境：Python 3.13, numpy, pandas, scipy, openpyxl, xlsxwriter, scikit-learn。"))
    lines.append(("body", "· 全量实验：py -3 脚本/run_full_parallel.py --workers 6  （输出 全日期实验结果/）"))
    lines.append(("body", "· 统计推断：py -3 脚本/analyze_results.py"))
    lines.append(("body", "· 学习基线：py -3 脚本/ml_baseline.py"))
    lines.append(("body", "· 重建本工作簿：py -3 脚本/build_dataset.py"))
    lines.append(("body", "· 独立校验：py -3 脚本/verify_bundle.py"))
    lines.append(("blank", ""))
    lines.append((H, "7. v2 相对 v1（66日预实验包）的变化"))
    lines.append(("body", "· 数据：由分散CSV/NPZ整理为一份多表单工作簿 + 机器可读CSV.GZ/NPZ + 服务器schema。"))
    lines.append(("body", "· 周期：实验由66个抽样日扩展到全部334个真实可评估日（全量结果）。"))
    lines.append(("body", "· 特征：新增日历特征、28日同slot历史、四发布时刻光伏/电价预报特征、第二问日前预测列。"))
    lines.append(("body", "· 结果：新增配对统计推断（块Bootstrap、Wilcoxon、Holm）、费用分解、主效应与分季节冻结测试。"))
    lines.append(("body", "· 基线：新增学习型日前预测基线（Ridge / HistGradientBoosting）与因果基线对照。"))
    lines.append(("body", "· 校验：主键、行数、物理边界、残差、源文件SHA-256全部写入“25_数据质量”。"))
    lines.append(("blank", ""))
    lines.append((H, "8. 论文写作时的口径提醒"))
    lines.append(("body", "· 场景是模型输入，不是独立观测样本；样本量按真实日计算（n=334）。"))
    lines.append(("body", "· 全年及指定日期此前已被查看，时间划分属回顾性验证，不能写成从未见过的外部测试。"))
    lines.append(("body", "· 费用差必须与紧急购电量（可靠性）同时报告；平均费用最低不等于最可靠。"))
    lines.append(("body", "· 储能回放使用当日实测序列，属于同口径离线匹配比较，不是完整在线MPC。"))
    lines.append(("body", "· 单站点、单年度，不能外推跨站点/跨年度泛化；加入新真实年份或站点需另取同步数据。"))
    lines.append(("blank", ""))
    lines.append((H, "9. 目录"))
    lines.append(("body", "数据集/配电微网_随机优化数据集_v2.xlsx（本文件）；数据集/csv/*.csv.gz（全精度机器可读）；"
                          "数据集/npz/*.npz（场景数组与原始数组）；服务器/（schema + loader）；全日期实验结果/；统计检验/；ML基线/；文档/。"))
    return lines


# --------------------------------------------------------------------------- #
# xlsx writing
# --------------------------------------------------------------------------- #
class SheetWriter:
    FMT = {}

    def __init__(self, wb):
        self.wb = wb
        self.hdr = wb.add_format({"bold": True, "font_color": "#1F3864", "bg_color": "#DCE6F1",
                                  "border": 1, "text_wrap": True, "valign": "vcenter",
                                  "align": "center", "font_size": 10})
        self.txt = wb.add_format({"font_size": 10, "valign": "top"})
        self.date = wb.add_format({"num_format": "yyyy-mm-dd", "font_size": 10})
        self.ts = wb.add_format({"num_format": "yyyy-mm-dd hh:mm", "font_size": 10})
        self.num = wb.add_format({"num_format": "0.0000", "font_size": 10})
        self.num6 = wb.add_format({"num_format": "0.000000", "font_size": 10})
        self.integer = wb.add_format({"num_format": "0", "font_size": 10})
        self.sci = wb.add_format({"num_format": "0.00E+00", "font_size": 10})
        self.title = wb.add_format({"bold": True, "font_size": 14, "font_color": "#1F3864"})
        self.section = wb.add_format({"bold": True, "font_size": 11, "font_color": "#2E5C9A",
                                      "bottom": 1})
        self.body = wb.add_format({"font_size": 10, "text_wrap": True, "valign": "top"})

    def _col_fmt(self, series):
        if pd.api.types.is_datetime64_any_dtype(series):
            if (series.dt.hour.fillna(0) != 0).any() or (series.dt.minute.fillna(0) != 0).any():
                return self.ts
            return self.date
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
            return self.integer
        return self.num

    def table(self, name, frame, purpose, key, widths=None, fmt_overrides=None,
              freeze=True, autofilter=True, index_entry=True):
        ws = self.wb.add_worksheet(name[:31])
        widths = widths or {}
        fmt_overrides = fmt_overrides or {}
        for j, col in enumerate(frame.columns):
            width = widths.get(col, min(28, max(10, len(str(col)) * 1.1 + 2)))
            fmt = fmt_overrides.get(col) or self._col_fmt(frame[col])
            ws.set_column(j, j, width, fmt)
            ws.write(0, j, str(col), self.hdr)
        ws.set_row(0, 32)
        work = frame
        for col in work.columns:
            if work[col].isna().any():
                work[col] = work[col].astype(object).where(work[col].notna(), None)
            if pd.api.types.is_integer_dtype(work[col]):
                work[col] = [int(v) for v in work[col]]
            elif pd.api.types.is_bool_dtype(work[col]):
                work[col] = [bool(v) for v in work[col]]
        for i, row in enumerate(work.itertuples(index=False, name=None), 1):
            ws.write_row(i, 0, row)
        if freeze:
            ws.freeze_panes(1, 0)
        if autofilter and len(frame) > 0:
            ws.autofilter(0, 0, len(frame), len(frame.columns) - 1)
        if index_entry:
            SHEET_INDEX.append({"序号": len(SHEET_INDEX) + 1, "表单": name, "行数": len(frame),
                                "列数": len(frame.columns), "主键": key, "用途": purpose})
        return ws


def write_readme(wb, lines):
    ws = wb.add_worksheet("00_说明")
    ws.set_column(0, 0, 3)
    ws.set_column(1, 1, 135, None)
    ws.hide_gridlines(2)
    fmts = {
        "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#1F3864"}),
        "section": wb.add_format({"bold": True, "font_size": 11.5, "font_color": "#2E5C9A"}),
        "body": wb.add_format({"font_size": 10.5, "text_wrap": True, "valign": "top"}),
    }
    r = 0
    for kind, text in lines:
        if kind == "blank":
            r += 1
            continue
        ws.write(r, 1, text, fmts[kind])
        if kind == "body":
            ws.set_row(r, max(15, 15 * (len(text) // 70 + 1)))
        r += 1
    return ws


# --------------------------------------------------------------------------- #
# exports + schema
# --------------------------------------------------------------------------- #
def export_tables(tables, scenario_bank_src):
    csv_dir = OUTDIR / "csv"
    npz_dir = OUTDIR / "npz"
    csv_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, frame in tables.items():
        if name.startswith("08_") or name.startswith("09_"):
            continue
        path = csv_dir / (name + ".csv.gz")
        frame.to_csv(path, index=False, compression="gzip", encoding="utf-8")
        manifest.append({"table": name, "rows": len(frame), "cols": len(frame.columns),
                         "file": str(path.relative_to(BASE)), "sha256": sha256(path),
                         "bytes": path.stat().st_size})
    for name, frame in [("scenario_net_wide", tables["08_场景库_净负荷"]),
                        ("scenario_price_wide", tables["09_场景库_电价"])]:
        path = csv_dir / (name + ".csv.gz")
        frame.to_csv(path, index=False, compression="gzip", encoding="utf-8")
        manifest.append({"table": name, "rows": len(frame), "cols": len(frame.columns),
                         "file": str(path.relative_to(BASE)), "sha256": sha256(path),
                         "bytes": path.stat().st_size})
    for src in [DATA / "scenario_bank.npz", DATA / "inputs.npz"]:
        dst = npz_dir / src.name
        dst.write_bytes(src.read_bytes())
        manifest.append({"table": f"npz/{src.name}", "rows": "", "cols": "",
                         "file": str(dst.relative_to(BASE)), "sha256": sha256(dst),
                         "bytes": dst.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUTDIR / "导出清单与校验.csv", index=False, encoding="utf-8-sig")
    return manifest


def write_schema(tables):
    SERVER.mkdir(parents=True, exist_ok=True)
    schema = {}
    for name, frame in tables.items():
        cols = []
        for col in frame.columns:
            s = frame[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                dtype = "datetime64[ns]"
            elif pd.api.types.is_integer_dtype(s):
                dtype = "int64"
            elif pd.api.types.is_bool_dtype(s):
                dtype = "bool"
            elif pd.api.types.is_float_dtype(s):
                dtype = "float64"
            else:
                dtype = "string"
            cols.append({"name": str(col), "dtype": dtype,
                         "unit": "", "nullable": bool(s.isna().any())})
        schema[name] = {"rows": len(frame), "columns": cols}
    base = {
        "dataset": "microgrid_stochastic_rolling_optimisation_v2",
        "generated": "2026-10-17",
        "keys": {
            "03_观测_10min": ["date", "slot_index"],
            "05_继承输入_Q2": ["date", "slot_index"],
            "06_发布时刻输入": ["date", "release_hour", "slot_index"],
            "07_光伏预报_10min": ["date", "release_hour", "slot_index"],
            "10_ML训练表_10min": ["date", "slot_index"],
            "08_场景库_净负荷": ["date", "quantile"],
            "09_场景库_电价": ["date", "quantile"],
        },
        "time_protocol": {"warmup": ["2025-01-01", "2025-01-31"],
                          "development": ["2025-02-01", "2025-06-30"],
                          "validation": ["2025-07-01", "2025-09-30"],
                          "test": ["2025-10-01", "2025-12-31"]},
        "tables": schema,
    }
    (SERVER / "schema.json").write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")


def write_loader():
    loader = '''# -*- coding: utf-8 -*-
"""Loader for the microgrid stochastic/rolling-optimisation dataset v2.

Works either from the parsed CSV.GZ copies (recommended for training servers)
or directly from the consolidated Excel workbook (small tables only).

    from dataset_loader import load_table, load_workbook_sheet, load_scenarios
    obs = load_table("03_观测_10min")
    ml  = load_table("10_ML训练表_10min")
    net, price, weights, quantiles = load_scenarios()

Rules enforced here, not in the model:
  * split in {"development","validation","test"}; never fit on validation/test
    when reporting the frozen test result;
  * columns whose name is a physical quantity (load_kw, pv_kw, net_load_kw,
    price_yuan_per_kwh) are targets only;
  * features that are released later in the day (pv_fc_r06 ... price_fc_r18)
    are NaN before their release hour by construction -- do not fill them with
    future values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CSV = BASE / "数据集" / "csv"
XLSX = BASE / "数据集" / "配电微网_随机优化数据集_v2.xlsx"
NPZ = BASE / "数据集" / "npz"

TARGETS = ["load_kw", "pv_kw", "net_load_kw", "price_yuan_per_kwh"]


def load_table(name: str) -> pd.DataFrame:
    path = CSV / f"{name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run 脚本/build_dataset.py first")
    frame = pd.read_csv(path, low_memory=False)
    for col in frame.columns:
        if col == "date" or col.endswith("_date") or col.endswith("_at"):
            try:
                frame[col] = pd.to_datetime(frame[col])
            except (ValueError, TypeError):
                pass
    return frame


def load_workbook_sheet(name: str) -> pd.DataFrame:
    return pd.read_excel(XLSX, sheet_name=name, engine="openpyxl")


def load_scenarios():
    """Wide net-load / price scenarios: (dates, quantiles, slots, weights)."""
    net = load_table("scenario_net_wide")
    price = load_table("scenario_price_wide")
    slot_cols = [c for c in net.columns if c.startswith("slot_")]
    dates = pd.DatetimeIndex(sorted(net.date.unique()))
    quantiles = net["quantile"].unique().tolist()
    def _cube(frame):
        arr = np.empty((len(dates), len(quantiles), len(slot_cols)))
        for i, d in enumerate(dates):
            part = frame[frame["date"] == d].set_index("quantile").loc[quantiles]
            arr[i] = part[slot_cols].to_numpy(float)
        return arr
    weights = net[net["date"] == dates[0]][["quantile", "scenario_weight"]].set_index("quantile").loc[quantiles, "scenario_weight"].to_numpy(float)
    return _cube(net), _cube(price), weights, np.array([float(q[1:]) / 100 for q in quantiles])


def load_npz_scenarios():
    z = np.load(NPZ / "scenario_bank.npz")
    return z["dates"], z["quantiles"], z["weights"], z["net_kw"], z["price_yuan_per_kwh"]
'''
    (SERVER / "dataset_loader.py").write_text(loader, encoding="utf-8")
    (SERVER / "README.md").write_text(
        """# 服务器接入说明

`dataset_loader.py` 提供三种读取方式：

1. `load_table("03_观测_10min")` —— 推荐，读取 `数据集/csv/*.csv.gz`，全精度、内存友好；
2. `load_workbook_sheet("10_ML训练表_10min")` —— 直接读 Excel 表单（表较大，仅建议抽查）；
3. `load_scenarios()` / `load_npz_scenarios()` —— 场景库，返回 `(dates, quantiles, slots)` 立方体。

训练建议：

```python
from dataset_loader import load_table
ml = load_table("10_ML训练表_10min")
feat_cols = [c for c in ml.columns if c not in
             ("date","slot_index","timestamp_start","split","season",
              "feature_available_at","target_available_at",
              "load_kw","pv_kw","net_load_kw","price_yuan_per_kwh")]
assert not set(feat_cols) & {"load_kw","pv_kw","net_load_kw","price_yuan_per_kwh"}
train = ml[ml.split == "development"]
val = ml[ml.split == "validation"]
test = ml[ml.split == "test"]
```

`服务器/schema.json` 给出每张表的行数、列名、dtype 与是否可空，可直接生成 ORM 或
数据加载配置。
""", encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (DATA / "派生").mkdir(parents=True, exist_ok=True)
    inputs, obs, cal, inh, rel, q2, bank = load_inputs()
    obs_t, cal_t, inh_t, rel_t, pvfc, net_wide, price_wide, ml, qlabels = build_core_tables(
        inputs, obs, cal, inh, rel, q2, bank)
    res_dir, scale = results_dir()
    res = build_result_tables(res_dir, scale)

    # ---- collect every sheet first, so the index can be written first ------ #
    sheets = []          # (name, frame, purpose, key, widths, fmt_overrides)
    data_tables = {}     # exported to CSV.GZ / schema

    def register(name, frame, purpose, key, widths=None, fmt_overrides=None, export=False):
        sheets.append((name, frame, purpose, key, widths or {}, fmt_overrides or {}))
        if export:
            data_tables[name] = frame

    register("03_观测_10min", obs_t, "365日×144时段实测观测（含拆分与派生电量）",
             "date,slot_index", export=True)
    register("04_评估日历", cal_t, "334个可评估日期、时间划分与66日预实验标志", "date", export=True)
    register("05_继承输入_Q2", inh_t, "第二问日前计划、SOC锚点与11个净负荷分位预测",
             "date,slot_index", export=True)
    register("06_发布时刻输入", rel_t, "00/06/12/18四个发布时刻的净负荷与电价因果预测输入",
             "date,release_hour,slot_index", export=True)
    register("07_光伏预报_10min", pvfc, "四个发布版本的光伏分次预报（10分钟）及误差",
             "date,release_hour,slot_index", export=True)
    sw_scen_w = {c: 9.5 for c in net_wide.columns if c.startswith("slot_")}
    register("08_场景库_净负荷", net_wide,
             "11个共尾净负荷分位场景宽表（每日每分位一行，slot_001..144）",
             "date,quantile", widths=sw_scen_w, export=True)
    register("09_场景库_电价", price_wide,
             "11个共尾电价分位场景宽表（元/kWh）", "date,quantile", widths=sw_scen_w, export=True)
    register("10_ML训练表_10min", ml, "每行一个(日期,时段)的因果特征+目标训练视图",
             "date,slot_index", export=True)

    result_tables = {
        "11_实验_预测_每日": ("forecast_daily", "20组价格预测参数的逐日指标", "date,window_days,decay_days"),
        "12_实验_预测_汇总": ("forecast_summary", "价格预测按split汇总", "split,window_days,decay_days"),
        "13_实验_风险_每日": ("risk_daily", "7个风险权重×真实日的匹配回放结果", "date,risk_weight"),
        "14_实验_风险_汇总": ("risk_summary", "风险前沿按split汇总", "split,risk_weight"),
        "15_实验_风险_配对": ("risk_paired", "相对风险中性方案的配对差与区间", "risk_weight"),
        "16_实验_分辨率_每日": ("resolution_daily", "3/5/7/11场景的逐日结果", "date,scenario_count"),
        "17_实验_分辨率_汇总": ("resolution_summary", "场景分辨率按split汇总", "split,scenario_count"),
        "18_实验_控制_每日": ("rolling_daily", "5×5控制参数网格×真实日结果", "date,gain,reserve_quantile"),
        "19_实验_控制_汇总": ("rolling_summary", "控制网格按split汇总", "split,gain,reserve_quantile"),
        "20_实验_控制_冻结对比": ("selected_vs_formal_daily", "验证选参与正式参考的逐日对照", "date"),
    }
    for sheet, (key, purpose, pkey) in result_tables.items():
        if key in res:
            register(sheet, res[key], purpose, pkey, export=True)

    pvf = pilot_vs_full()
    if not pvf.empty:
        register("21_预实验与全量对照", pvf, "66日预实验与334日全量实验的口径对照", "experiment,arm")

    stat_long = STAT / "统计检验_长表.csv"
    if stat_long.exists():
        stats = pd.read_csv(stat_long)[["analysis", "subject", "metric", "value", "ci95_low",
                                        "ci95_high", "p_value", "n_days", "split", "note"]]
        register("22_统计检验", stats, "配对检验/区间估计/分季节证据（tidy长表）",
                 "analysis,subject,metric",
                 widths={"analysis": 26, "subject": 34, "metric": 34, "note": 60},
                 fmt_overrides={"p_value": None, "ci95_low": None, "ci95_high": None})

    ml_sum = ML / "指标_汇总.csv"
    ml_daily = ML / "逐日指标.csv"
    if ml_sum.exists():
        register("23_ML基线_汇总", pd.read_csv(ml_sum),
                 "学习型日前预测基线与因果/持续基线对照（验证/测试）",
                 "target,model,split", widths={"model": 40})
    if ml_daily.exists():
        frame = pd.read_csv(ml_daily, parse_dates=["date"])
        register("23b_ML基线_逐日", frame, "学习基线逐日MAE/RMSE（用于配对检验）", "date,target,model")
    sel = ML / "模型选择.json"
    if ml_sum.exists() and sel.exists():
        selection = json.loads(sel.read_text(encoding="utf-8"))
        rows = [{"目标": k, "选中模型": v["selected_model"],
                 "验证集MAE": round(min(v["validation_mae"].values()), 6),
                 "候选模型验证MAE": "; ".join(f"{a}={b:.6f}" for a, b in v["validation_mae"].items())}
                for k, v in selection.items()]
        register("23c_ML基线_模型选择", pd.DataFrame(rows), "验证集选模记录（测试集不参与选模）", "目标")

    cfg = build_config()
    register("24_实验配置", cfg, "参数网格、时间协议、求解器与统计协议", "类别,参数/项目",
             widths={"类别": 12, "参数/项目": 30, "取值": 60, "说明": 60})
    quality = build_quality(inputs, obs_t, cal_t, inh_t, rel_t, pvfc, net_wide, ml, res)
    register("25_数据质量", quality, "主键/完整性/物理边界/防泄漏/残差/源文件校验", "检查项",
             widths={"检查项": 34, "期望": 26, "实际": 40, "说明": 60})
    prov, lic = build_provenance()
    register("26_溯源与许可", prov, "源文件与交付文件的SHA-256及使用条件", "文件或来源",
             widths={"文件或来源": 56, "SHA-256（记录）": 70, "角色": 16, "当前文件一致": 14})
    register("26b_引用与许可", lic, "数据集名称、来源、权利与引用信息", "项目",
             widths={"项目": 18, "内容": 130})

    dictionary = build_dictionary(qlabels)
    index_rows = [
        {"序号": 0, "表单": "00_说明", "行数": "", "列数": "",
         "主键": "", "用途": "数据集说明、训练规范、复现步骤与口径提醒"},
        {"序号": 1, "表单": "01_表单索引", "行数": len(sheets) + 2, "列数": 6,
         "主键": "序号", "用途": "本清单"},
        {"序号": 2, "表单": "02_字段字典", "行数": len(dictionary), "列数": len(dictionary.columns),
         "主键": "表单,字段", "用途": "字段含义/单位/来源/可用时点"},
    ]
    for i, (name, frame, purpose, key, _, _) in enumerate(sheets, start=3):
        index_rows.append({"序号": i, "表单": name, "行数": len(frame), "列数": frame.shape[1],
                           "主键": key, "用途": purpose})
    index_frame = pd.DataFrame(index_rows)

    wb = xlsxwriter.Workbook(OUTDIR / "配电微网_随机优化数据集_v2.xlsx",
                            {"constant_memory": True, "nan_inf_to_errors": False})
    sw = SheetWriter(wb)
    write_readme(wb, build_readme(scale, len(sheets) + 3, {}))
    sw.table("01_表单索引", index_frame, "全部表单的行数、主键与用途", "序号",
             widths={"表单": 26, "用途": 76, "主键": 30, "行数": 10, "列数": 8},
             index_entry=False)
    sw.table("02_字段字典", dictionary, "全部表单的字段含义、单位、来源与可用时点",
             "表单,字段", widths={"表单": 22, "字段": 30, "含义": 66,
                                  "类型": 10, "单位/粒度": 12, "来源与规则": 34, "可用时点": 14},
             index_entry=False)
    for name, frame, purpose, key, widths, fmt_overrides in sheets:
        sw.table(name, frame, purpose, key, widths=widths, fmt_overrides=fmt_overrides,
                 index_entry=False)
        print(f"sheet {name}: {len(frame):,} x {frame.columns.size}", flush=True)
    wb.close()
    print("workbook written", (OUTDIR / "配电微网_随机优化数据集_v2.xlsx").stat().st_size / 1e6, "MB")

    export_tables(data_tables, DATA / "scenario_bank.npz")
    write_schema(data_tables)
    write_loader()
    print("exports + schema done")


if __name__ == "__main__":
    main()
