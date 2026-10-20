# -*- coding: utf-8 -*-
"""English publication figures from the same executed pilot tables."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.text import Text
import plot_results as base

OUT=base.ROOT/"论文图像"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],"font.size":12,
    "axes.unicode_minus":True,"svg.fonttype":"none"})
exact={
"(a) 66 日预实验按日历抽取，无结果排序":"(a) Calendar sampling across 66 real dates",
"月内日期（深蓝为实际预实验日）":"Day of month (dark cells mark pilot dates)",
"2025 年月份":"Month of 2025", "可评估真实日期":"Available evaluation dates", "已运行预实验":"Executed pilot dates",
"(b) 全日期数据就绪，计算先覆盖每月 6 日":"(b) Available dates and monthly pilot coverage",
"真实日期数":"Number of real dates",
"(c) 334 日真实电价，保留季节及日内结构":"(c) Observed prices over 334 evaluation days",
"日内时刻（h）":"Hour of day", "评估日期":"Evaluation date", "电价（元/kWh）":"Price (RMB/kWh)",
"原参数 35/14":"Reference 35/14 days", "验证选参 14/28":"Validation selection 14/28 days",
"(d) 预测在 334 日上计算，验证期选参后测试":"(d) Chronological forecast evaluation",
"价格 MAE（元/kWh）":"Price MAE (RMB/kWh)",
"(a) 风险偏好对应不同费用—可靠性位置":"(a) Cost and reliability across time subsets",
"平均紧急购电（kWh/日）":"Mean emergency energy (kWh/day)", "平均费用（千元/日）":"Mean cost (thousand RMB/day)",
"费用增加及月份聚类 95% 区间":"Cost increment and 95% month-cluster interval",
"紧急量降低":"Emergency reduction", "风险权重 ρ":"Risk weight ρ",
"相对 ρ=0 的平均费用增加（千元/日）":"Cost increment over ρ=0 (thousand RMB/day)",
"紧急量降低（kWh/日）":"Emergency reduction (kWh/day)",
"(b) 66 日匹配差异，同时披露收益与代价":"(b) Paired risk cost and emergency reduction",
"按概率精确 CVaR":"Exact probability-weighted CVaR", "旧最高两场景均值":"Mean of two largest scenario costs",
"场景尾部成本（千元/日）":"Scenario tail cost (thousand RMB/day)",
"(c) 非等概率场景必须用概率权重计算风险":"(c) Tail-risk definition under unequal probabilities",
"日前 LP 平均耗时（s）":"Mean day-ahead LP time (s)",
"(d) 场景分辨率、耗时与紧急量共同比较":"(d) Scenario resolution and computational burden",
"执行增益 γ":"Execution gain γ", "储备分位 τ":"Reserve quantile τ",
"费用差（千元/日）":"Cost difference (thousand RMB/day)", "紧急量（kWh/日）":"Emergency energy (kWh/day)",
"控制参数跨时期稳定性：星号为原参数，方框为验证期最低费用参数":"Control stability across periods: star marks reference; square marks validation selection",
}
replace={"开发期":"Development","验证期":"Validation","测试期":"Test","（30日）":" (30 days)","（18日）":" (18 days)",
"：相对原参数费用（千元/日）":": cost difference (thousand RMB/day)","：紧急购电（kWh/日）":": emergency (kWh/day)",
"场景":" scenarios","日":"day"}
for m,label in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],1):exact[f"{m}月"]=label

def save(fig,unused,name):
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        t=artist.get_text()
        if t in exact:t=exact[t]
        else:
            for a,b in replace.items():t=t.replace(a,b)
        artist.set_text(t)
        artist.set_fontfamily("Times New Roman")
    if "03_" in name:
        for ax in fig.axes:
            ax.title.set_fontsize(10)
        fig._suptitle.set_fontsize(12)
    stem={"01_数据扩展与时序验证":"paper_data_forecast","02_风险前沿与场景分辨率":"paper_risk_resolution","03_控制相图与冻结测试":"paper_control_stability"}[name]
    # Freeze automatic tick formatters after translating their generated labels.
    for ax in fig.axes:
        ax.set_xticks(ax.get_xticks(),[exact.get(t.get_text(),replace.get(t.get_text(),t.get_text())) for t in ax.get_xticklabels()])
        ax.set_yticks(ax.get_yticks(),[exact.get(t.get_text(),replace.get(t.get_text(),t.get_text())) for t in ax.get_yticklabels()])
    fig.savefig(OUT/f"{stem}.png",dpi=360,bbox_inches="tight",facecolor="white")
    fig.savefig(OUT/f"{stem}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)

base.save=save
if __name__=="__main__":
    result=base.ROOT/"预实验结果"
    base.coverage(result,OUT);base.risk_dashboard(result,OUT);base.policy_dashboard(result,OUT)
    print(OUT)
