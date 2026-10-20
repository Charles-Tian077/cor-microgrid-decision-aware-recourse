# -*- coding: utf-8 -*-
"""Shared dataset construction utilities for the Q4 / microgrid SCI package.

Everything here is *derived from the already-validated prepared tables* (never
from the raw 26国赛 workbooks) so that the dataset stays reproducible from the
delivered package alone:

    数据/observations_10min.csv      365 d x 144 slots observations
    数据/inherited_q2.csv            334 d x 144 slots day-ahead plan + quantiles
    数据/released_forecast_inputs.csv 334 d x 4 releases x slots forecast inputs

The only external file touched is the Q2 schedule table, from which the Q2
day-ahead load and PV point forecasts are copied once into 数据/派生/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# calendar (State Council 2025 arrangement; used as an optional ML feature only,
# never inside the physical/stochastic optimisation model)
# --------------------------------------------------------------------------- #
HOLIDAYS_2025 = set(pd.to_datetime([
    "2025-01-01",                                                     # New Year
    *[f"2025-01-{d:02d}" for d in range(28, 32)],                     # Spring Festival
    *[f"2025-02-{d:02d}" for d in range(1, 5)],
    *[f"2025-04-{d:02d}" for d in range(4, 7)],                       # Qingming
    *[f"2025-05-{d:02d}" for d in range(1, 6)],                       # Labour Day
    "2025-05-31", "2025-06-01", "2025-06-02",                         # Dragon Boat
    *[f"2025-10-{d:02d}" for d in range(1, 9)],                       # National Day + Mid-Autumn
]))

MAKEUP_WORKDAYS = set(pd.to_datetime([
    "2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28", "2025-10-11",
]))


def season_of(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame({"date": dates})
    frame["year"] = dates.year
    frame["month"] = dates.month
    frame["day"] = dates.day
    frame["weekday"] = dates.dayofweek.to_numpy() + 1       # 1 = Monday ... 7 = Sunday
    frame["is_weekend"] = (frame["weekday"] >= 6).astype(int)
    frame["is_holiday"] = np.isin(dates.normalize(), list(HOLIDAYS_2025)).astype(int)
    frame["is_makeup_workday"] = np.isin(dates.normalize(), list(MAKEUP_WORKDAYS)).astype(int)
    frame["day_of_year"] = dates.dayofyear
    frame["week_of_year"] = dates.isocalendar()["week"].to_numpy().astype(int)
    frame["season"] = frame.month.map(season_of)
    return frame


# --------------------------------------------------------------------------- #
# feature builders
# --------------------------------------------------------------------------- #
def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, float).reshape(-1)


def _lag(arr: np.ndarray, day_idx: np.ndarray, k: int) -> np.ndarray:
    return np.stack([arr[d - k] for d in day_idx])


def _same_slot_mean(arr: np.ndarray, day_idx: np.ndarray, window: int) -> np.ndarray:
    out = np.empty((len(day_idx), arr.shape[1]))
    for j, d in enumerate(day_idx):
        out[j] = arr[d - window:d].mean(axis=0)
    return out


def _same_slot_std(arr: np.ndarray, day_idx: np.ndarray, window: int) -> np.ndarray:
    out = np.empty((len(day_idx), arr.shape[1]))
    for j, d in enumerate(day_idx):
        out[j] = arr[d - window:d].std(axis=0)
    return out


def build_ml_table(
    dates_all: pd.DatetimeIndex,
    load: np.ndarray,
    pv: np.ndarray,
    price: np.ndarray,
    fixed_price: np.ndarray,
    pv_release: dict[int, np.ndarray],
    inherited: pd.DataFrame,
    q2_forecast: pd.DataFrame,
    released: pd.DataFrame,
    net_quantile_columns: list[str],
) -> pd.DataFrame:
    """One row per evaluated (date, slot) with causal features and observed targets.

    Availability rules (enforced by construction, not by convention):
      * history features use only days strictly before the target date;
      * the Q2 quantiles / Q2 point forecasts are the day-ahead (00:00) model inputs
        inherited from problem 2 and are therefore available at decision time;
      * ``pv_fc_r*`` and ``price_fc_r*`` are only filled from that release hour onward;
      * everything after ``feature_available_at`` is *not* a feature: the observed
        targets are stored in the ``target_*``-free columns named exactly like the
        physical quantity and must be used as y only.
    """
    dates_all = pd.DatetimeIndex(dates_all)
    eval_dates = dates_all[31:]                     # 2025-02-01 ... 2025-12-31
    day_idx = np.arange(31, len(dates_all))         # positional index inside the 365-day arrays
    nd, nt = len(eval_dates), load.shape[1]
    net = load - pv

    cal = calendar_frame(pd.DatetimeIndex(np.repeat(eval_dates, nt)))
    slot = np.tile(np.arange(1, nt + 1), nd)
    phase = 2.0 * np.pi * (slot - 0.5) / nt
    doy = cal.day_of_year.to_numpy()
    frame = pd.DataFrame({
        "date": cal.date.to_numpy(),
        "slot_index": slot.astype(np.int16),
        "timestamp_start": np.repeat(eval_dates.to_numpy(), nt) + ((slot - 1) * 10).astype("timedelta64[m]"),
        "split": np.repeat(
            np.where(eval_dates < pd.Timestamp("2025-07-01"), "development",
                     np.where(eval_dates < pd.Timestamp("2025-10-01"), "validation", "test")), nt),
        "month": cal.month.to_numpy().astype(np.int8),
        "weekday": cal.weekday.to_numpy().astype(np.int8),
        "is_weekend": cal.is_weekend.to_numpy().astype(np.int8),
        "is_holiday": cal.is_holiday.to_numpy().astype(np.int8),
        "is_makeup_workday": cal.is_makeup_workday.to_numpy().astype(np.int8),
        "season": cal.season.to_numpy(),
        "sin_slot": np.tile(np.sin(phase[:nt]), nd).round(6),
        "cos_slot": np.tile(np.cos(phase[:nt]), nd).round(6),
        "sin_doy": np.sin(2 * np.pi * doy / 365.0).round(6),
        "cos_doy": np.cos(2 * np.pi * doy / 365.0).round(6),
    })

    # ---- observed targets -------------------------------------------------- #
    frame["load_kw"] = _flat(_lag(load, day_idx, 0)).round(4)
    frame["pv_kw"] = _flat(_lag(pv, day_idx, 0)).round(4)
    frame["net_load_kw"] = frame["load_kw"] - frame["pv_kw"]
    frame["price_yuan_per_kwh"] = _flat(_lag(price, day_idx, 0)).round(6)
    frame["price_fixed_yuan_per_kwh"] = np.tile(fixed_price, nd).round(4)

    # ---- calendar/history features ---------------------------------------- #
    for name, arr in [("load", load), ("pv", pv), ("net", net), ("price", price)]:
        frame[f"{name}_lag_1d"] = _flat(_lag(arr, day_idx, 1)).round(4)
        frame[f"{name}_lag_7d"] = _flat(_lag(arr, day_idx, 7)).round(4)
        frame[f"{name}_same_slot_mean_7d"] = _flat(_same_slot_mean(arr, day_idx, 7)).round(4)
        frame[f"{name}_same_slot_mean_28d"] = _flat(_same_slot_mean(arr, day_idx, 28)).round(4)
    frame["price_lag_14d"] = _flat(_lag(price, day_idx, 14)).round(6)
    frame["price_lag_28d"] = _flat(_lag(price, day_idx, 28)).round(6)
    frame["price_same_slot_std_28d"] = _flat(_same_slot_std(price, day_idx, 28)).round(6)

    # intraday "last observed" features (missing at the first slot of the day)
    prev_load = np.stack([np.r_[load[d - 1, -1], load[d, :-1]] for d in day_idx])
    prev_price = np.stack([np.r_[price[d - 1, -1], price[d, :-1]] for d in day_idx])
    prev_load_f = _flat(prev_load)
    prev_price_f = _flat(prev_price)
    frame["prev_slot_load_kw"] = prev_load_f.round(4)
    frame["prev_slot_price_yuan_per_kwh"] = prev_price_f.round(6)

    # ---- day-ahead Q2 model inputs (available at 00:00) -------------------- #
    q2 = q2_forecast.copy()
    q2["date"] = pd.to_datetime(q2["date"])
    for col in ["load_forecast_kw", "pv_forecast_kw"]:
        piv = q2.pivot(index="date", columns="slot_index", values=col).reindex(
            index=eval_dates, columns=range(1, nt + 1)).to_numpy(float)
        frame[col] = _flat(piv).round(4)
    inh = inherited.copy()
    inh["date"] = pd.to_datetime(inh["date"])
    for col in net_quantile_columns:
        piv = inh.pivot(index="date", columns="slot_index", values=col).reindex(
            index=eval_dates, columns=range(1, nt + 1)).to_numpy(float)
        frame[col] = _flat(piv).round(4)

    # ---- PV forecast versions per release hour ---------------------------- #
    for r, arr in sorted(pv_release.items()):
        pad = np.full((len(dates_all), nt), np.nan)
        pad[:, 6 * r:] = arr
        frame[f"pv_fc_r{r:02d}_kw"] = _flat(pad[day_idx]).round(4)

    # ---- price forecast versions per release hour (causal, shrunk) -------- #
    rel = released.copy()
    rel["date"] = pd.to_datetime(rel["date"])
    for r in sorted(rel.release_hour.unique()):
        part = rel[rel.release_hour == r]
        piv = part.pivot(index="date", columns="slot_index",
                         values="price_point_yuan_per_kwh").reindex(
            index=eval_dates, columns=range(1, nt + 1)).to_numpy(float)
        frame[f"price_fc_r{r:02d}_yuan_per_kwh"] = _flat(piv).round(6)

    # ---- availability stamps ---------------------------------------------- #
    frame["feature_available_at"] = np.repeat(eval_dates.to_numpy(), nt)   # 00:00 of the target day
    frame["feature_available_at"] = pd.to_datetime(frame["feature_available_at"])
    frame["target_available_at"] = frame["timestamp_start"] + pd.Timedelta(minutes=10)
    return frame


def pv_release_forecasts(dates_all: pd.DatetimeIndex, pv_release: dict[int, np.ndarray]) -> pd.DataFrame:
    """Long table of the operational PV forecasts actually usable inside the day."""
    rows = []
    for r, arr in sorted(pv_release.items()):
        nd = len(dates_all)
        for j, d in enumerate(dates_all):
            s = 6 * r
            for t in range(s, arr.shape[1] + s):
                rows.append((d, r, t + 1, (t + 1) * 10 - r * 60, arr[j, t - s]))
    frame = pd.DataFrame(rows, columns=["date", "release_hour", "slot_index",
                                        "horizon_minute", "pv_forecast_kw"])
    frame["target_timestamp"] = frame["date"] + pd.to_timedelta((frame.slot_index) * 10, unit="m")
    return frame
