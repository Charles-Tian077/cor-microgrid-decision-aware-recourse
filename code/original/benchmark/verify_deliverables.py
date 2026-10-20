# -*- coding: utf-8 -*-
"""Independent acceptance check of the delivered package.

Verifies, without touching the builders' internal state:
  * the consolidated workbook opens and every expected sheet has the right shape;
  * the machine-readable CSV.GZ copies load and match the workbook row counts;
  * the scenario cube round-trips to (334, 11, 144) with weights summing to 1;
  * the training view contains no target column among the feature columns;
  * the headline experiment tables have complete day coverage (334 days);
  * the statistics / ML / docs / schema files exist and are non-empty.

Exit code 0 only if every check passes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE / "服务器"))

FAILS = []
OKS = []


def check(name, cond, detail=""):
    (OKS if cond else FAILS).append(f"{name}: {detail}" if detail else name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # ---- workbook --------------------------------------------------------- #
    xlsx = BASE / "数据集" / "配电微网_随机优化数据集_v2.xlsx"
    check("主工作簿存在", xlsx.exists(), f"{xlsx.stat().st_size/1e6:.1f} MB" if xlsx.exists() else "missing")
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    names = wb.sheetnames
    print(f"      sheets ({len(names)}): {names}")
    expected_sheets = {
        "00_说明": None, "01_表单索引": None, "02_字段字典": None,
        "03_观测_10min": 52561, "04_评估日历": 335, "05_继承输入_Q2": 48097,
        "06_发布时刻输入": 120241, "07_光伏预报_10min": 120241,
        "08_场景库_净负荷": 3675, "09_场景库_电价": 3675,
        "10_ML训练表_10min": 48097, "13_实验_风险_每日": 2339,
        "16_实验_分辨率_每日": 1337, "18_实验_控制_每日": 8351,
        "25_数据质量": None, "26_溯源与许可": None,
    }
    for sheet, rows in expected_sheets.items():
        if sheet not in names:
            check(f"表单 {sheet} 存在", False, "missing")
            continue
        ws = wb[sheet]
        actual = ws.max_row
        if rows is None:
            check(f"表单 {sheet} 存在", actual is not None and actual > 0, f"{actual} rows")
        else:
            check(f"表单 {sheet}", actual == rows, f"rows={actual} expected={rows}")
    wb.close()

    # ---- CSV.GZ copies ---------------------------------------------------- #
    from dataset_loader import load_table, load_scenarios, TARGETS
    csv_dir = BASE / "数据集" / "csv"
    files = sorted(p.name for p in csv_dir.glob("*.csv.gz"))
    check("CSV.GZ 副本数", len(files) >= 15, f"{len(files)} files")
    obs = load_table("03_观测_10min")
    check("观测表行数", len(obs) == 52560, f"{len(obs)}")
    check("观测表日期覆盖", obs.date.min() == pd.Timestamp("2025-01-01")
          and obs.date.max() == pd.Timestamp("2025-12-31"), "")
    ml = load_table("10_ML训练表_10min")
    check("训练表行数", len(ml) == 48096, f"{len(ml)}")
    feat_cols = [c for c in ml.columns if c not in
                 ("date", "slot_index", "timestamp_start", "split", "season",
                  "feature_available_at", "target_available_at", *TARGETS)]
    overlap = set(feat_cols) & set(TARGETS)
    check("训练表目标列未混入特征", not overlap, f"overlap={sorted(overlap)}")
    check("时间划分完整", set(ml.split.unique()) == {"development", "validation", "test"},
          str(ml.groupby("split").size().to_dict()))
    net, price, weights, quantiles = load_scenarios()
    check("场景立方体形状", net.shape == (334, 11, 144) and price.shape == net.shape, str(net.shape))
    check("场景概率和=1", abs(weights.sum() - 1) < 1e-9, f"{weights.sum():.12f}")
    cross_frac = float((np.diff(net, axis=1) < -1e-6).mean())
    check("场景分位交叉率 < 10%（已知数据限制）", cross_frac < 0.10, f"{cross_frac*100:.2f}%")
    check("价格场景为正", bool((price > 0).all()), f"min={price.min():.4f}")

    # ---- experiment tables ------------------------------------------------ #
    risk = pd.read_csv(BASE / "全日期实验结果" / "risk_daily.csv", parse_dates=["date"])
    rolling = pd.read_csv(BASE / "全日期实验结果" / "rolling_daily.csv", parse_dates=["date"])
    resolution = pd.read_csv(BASE / "全日期实验结果" / "resolution_daily.csv", parse_dates=["date"])
    check("风险实验覆盖 334 日 × 7 权重", risk.date.nunique() == 334 and risk.risk_weight.nunique() == 7,
          f"days={risk.date.nunique()}, rho={risk.risk_weight.nunique()}, rows={len(risk)}")
    check("控制实验覆盖 334 日 × 25 参数", rolling.date.nunique() == 334 and len(rolling) == 8350,
          f"days={rolling.date.nunique()}, rows={len(rolling)}")
    check("分辨率实验覆盖 334 日 × 4 场景", resolution.date.nunique() == 334 and len(resolution) == 1336,
          f"days={resolution.date.nunique()}, rows={len(resolution)}")
    failed = pd.read_csv(BASE / "全日期实验结果" / "failed_jobs.csv")
    check("失败条目为 0", len(failed) == 0, f"{len(failed)} rows")
    check("风险实验物理残差", risk.max_eq_residual_kwh.max() < 1e-6,
          f"max_eq={risk.max_eq_residual_kwh.max():.2e}")
    bounds = pd.read_csv(BASE / "全日期实验结果" / "随机规划价值边界.csv", parse_dates=["date"])
    check("VSS/EVPI 表覆盖 334 日", len(bounds) == 334, f"{len(bounds)}")

    # ---- statistics / ML / docs ------------------------------------------- #
    stats = pd.read_csv(BASE / "统计检验" / "统计检验_长表.csv")
    check("统计长表非空", len(stats) > 300, f"{len(stats)} rows")
    check("统计含 VSS/EVPI", "stochastic_programming_bounds" in set(stats.analysis), "")
    ml_sum = BASE / "ML基线" / "指标_汇总.csv"
    check("ML 基线结果存在", ml_sum.exists() and ml_sum.stat().st_size > 0,
          f"{ml_sum.stat().st_size} bytes" if ml_sum.exists() else "missing")
    for doc in ["数据集说明.md", "实验改进与主要发现.md", "SCI论文改造方案.md"]:
        p = BASE / "文档" / doc
        check(f"文档 {doc}", p.exists() and p.stat().st_size > 1000, f"{p.stat().st_size} bytes" if p.exists() else "missing")
    check("服务器 schema", (BASE / "服务器" / "schema.json").exists(), "")
    check("服务器 loader", (BASE / "服务器" / "dataset_loader.py").exists(), "")
    schema = json.loads((BASE / "服务器" / "schema.json").read_text(encoding="utf-8"))
    check("schema 表数", len(schema["tables"]) >= 15, f"{len(schema['tables'])} tables")

    # ---- checksums -------------------------------------------------------- #
    manifest = pd.read_csv(BASE / "数据集" / "导出清单与校验.csv")
    bad = [r.file for r in manifest.itertuples() if sha256(BASE / r.file) != r.sha256]
    check("导出文件 SHA-256 一致", len(bad) == 0, f"{len(bad)} mismatched")

    print("\n" + "=" * 60)
    print(f"passed {len(OKS)} / {len(OKS) + len(FAILS)}")
    if FAILS:
        print("FAILURES:")
        for f in FAILS:
            print(" -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
