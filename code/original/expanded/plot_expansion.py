"""Figures based only on saved expanded-experiment CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimSun", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10.5, "axes.labelsize": 11,
        "axes.titlesize": 11.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#243746", "text.color": "#243746", "axes.labelcolor": "#243746",
        "xtick.color": "#243746", "ytick.color": "#243746", "savefig.dpi": 500,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    configure()
    r = args.results
    fc = pd.read_csv(r / "forecast_comparison.csv")
    ts = pd.read_csv(r / "policy_test_summary.csv")
    monthly = pd.read_csv(r / "policy_test_monthly.csv")
    names = {"causal_shrinkage": "原因果预测", "hist_gradient_boosting": "梯度提升",
             "hgb_same_28day_history": "同历史梯度提升", "gru_3seed_ensemble": "GRU 集成"}
    colors = ["#557790", "#c77f70", "#71a89a", "#8173a5"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.85), layout="constrained")
    for i, name in enumerate(names):
        subset = fc[fc.forecaster == name].set_index("split")
        axes[0].bar(np.arange(2) + (i - 1.5) * 0.20,
                    subset.loc[["validation", "test"], "mae_yuan_per_kwh"],
                    width=0.19, color=colors[i], label=names[name])
    axes[0].set_xticks([0, 1], ["验证期", "测试期"])
    axes[0].set_ylabel("电价平均绝对误差 / 元每千瓦时")
    axes[0].set_title("(a) 预测误差")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
                   bbox_to_anchor=(.5, -.18))
    axes[0].grid(axis="y", alpha=.14)
    label = {"no_update": "不滚动", "half_gain": "半步修正", "full_gain": "完整修正",
             "no_reserve": "无安全裕度"}
    for p in ts.policy:
        if p not in label:
            label[p] = "新预测融合"
    offsets = {"no_update": (-76, -12), "half_gain": (8, 9),
               "full_gain": (48, -2), "no_reserve": (48, 26)}
    for _, row in ts.iterrows():
        c = "#557790" if row.policy == "half_gain" else "#c77f70" if row.policy == "full_gain" else "#8173a5"
        axes[1].scatter(row.emergency_kwh / 1000, row.total_cost_yuan / 1e6,
                        s=58, color=c, edgecolor="white", linewidth=.7, zorder=3)
        if row.policy in offsets:
            axes[1].annotate(label[row.policy], (row.emergency_kwh / 1000, row.total_cost_yuan / 1e6),
                             xytext=offsets[row.policy], textcoords="offset points", fontsize=8,
                             arrowprops=dict(arrowstyle="-", lw=.6, color="#67737d"))
    axes[1].set_xlabel("测试期紧急购电 / 兆瓦时")
    axes[1].set_ylabel("测试期实际费用 / 百万元")
    axes[1].set_title("(b) 冻结回放的费用与应急量")
    axes[1].grid(alpha=.14)
    fig.savefig(r / "expanded_forecast_decision.png", bbox_inches="tight")
    plt.close(fig)

    months = sorted(monthly.month.unique())
    b = monthly[monthly.split == "half_gain"].set_index("month").loc[months]
    c = monthly[monthly.split == "full_gain"].set_index("month").loc[months]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.25), layout="constrained")
    x = np.arange(len(months))
    axes[0].bar(x, (b.cost_yuan - c.cost_yuan) / 1000, color="#557790", width=.62)
    axes[0].axhline(0, color="#243746", linewidth=.8)
    axes[0].set_title("(a) 完整修正相对半步修正的费用节约")
    axes[0].set_ylabel("节约费用 / 千元")
    axes[1].bar(x, (b.emergency_kwh - c.emergency_kwh) / 1000, color="#c77f70", width=.62)
    axes[1].axhline(0, color="#243746", linewidth=.8)
    axes[1].set_title("(b) 同期紧急购电量减少")
    axes[1].set_ylabel("减少电量 / 兆瓦时")
    for ax in axes:
        ax.set_xticks(x, [s[-2:] + "月" for s in months])
        ax.grid(axis="y", alpha=.14)
    fig.savefig(r / "expanded_monthly_effect.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
