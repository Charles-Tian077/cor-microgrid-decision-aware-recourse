# -*- coding: utf-8 -*-
"""Plot only executed Q4 expansion results. No synthetic metric interpolation."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NAVY,BLUE,CORAL,OLIVE,YELLOW="#466F87","#78B6C8","#E48578","#98AF1E","#F2D98E"
GRID="#E2E9ED"
SPLITS=["development","validation","test"]
NAMES={"development":"开发期","validation":"验证期","test":"测试期"}
COLORS=dict(zip(SPLITS,[NAVY,BLUE,CORAL]))
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["SimSun","Microsoft YaHei"],
    "font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":10,"ytick.labelsize":10,
    "legend.fontsize":10,"axes.unicode_minus":False,"svg.fonttype":"none",
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52616B"})
SEQ=LinearSegmentedColormap.from_list("seq",["#F5F9FA",BLUE,NAVY])
DIV=LinearSegmentedColormap.from_list("div",[NAVY,"#FAFCFD",CORAL])


def save(fig,folder,name):
    folder.mkdir(exist_ok=True)
    fig.savefig(folder/f"{name}.png",dpi=320,bbox_inches="tight",facecolor="white")
    fig.savefig(folder/f"{name}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def coverage(out,figdir):
    cal=pd.read_csv(ROOT/"数据/evaluation_calendar.csv",parse_dates=["date"])
    obs=pd.read_csv(ROOT/"数据/observations_10min.csv",parse_dates=["date"])
    fs=pd.read_csv(out/"forecast_selected.csv")
    fig,ax=plt.subplots(2,2,figsize=(14,8.5),layout="constrained")
    calendar=np.full((11,31),np.nan)
    for _,r in cal.iterrows():calendar[r.date.month-2,r.date.day-1]=1 if r.pilot_selected else 0
    ax[0,0].imshow(calendar,aspect="auto",cmap=LinearSegmentedColormap.from_list("cal",["#EFF3F5",NAVY]),vmin=0,vmax=1)
    ax[0,0].set_xticks([0,5,10,15,20,25,30],[1,6,11,16,21,26,31]);ax[0,0].set_yticks(range(11),range(2,13))
    ax[0,0].set_xlabel("月内日期（深蓝为实际预实验日）");ax[0,0].set_ylabel("2025 年月份")
    ax[0,0].set_title("(a) 66 日预实验按日历抽取，无结果排序",loc="left",weight="bold")
    monthly=cal.groupby(cal.date.dt.month).agg(days=("date","count"),pilot=("pilot_selected","sum"))
    months=monthly.index.to_numpy()
    ax[0,1].bar(months,monthly.days,color="#DCE8EE",label="可评估真实日期")
    ax[0,1].bar(months,monthly.pilot,color=[COLORS[split_of_month(m)] for m in months],label="已运行预实验")
    ax[0,1].set_xticks(months);ax[0,1].set_xlabel("2025 年月份");ax[0,1].set_ylabel("真实日期数")
    ax[0,1].set_title("(b) 全日期数据就绪，计算先覆盖每月 6 日",loc="left",weight="bold")
    ax[0,1].legend(frameon=False);ax[0,1].grid(axis="y",color=GRID)
    mat=obs[obs.date.dt.month>=2].pivot(index="date",columns="slot_index",values="price_yuan_per_kwh")
    im=ax[1,0].imshow(mat,aspect="auto",origin="lower",cmap=SEQ)
    ax[1,0].set_xticks([0,35,71,107,143],[0,6,12,18,24]);ax[1,0].set_xlabel("日内时刻（h）")
    firsts=[i for i,d in enumerate(mat.index) if d.day==1]
    ax[1,0].set_yticks(firsts,[f"{mat.index[i].month}月" for i in firsts]);ax[1,0].set_ylabel("评估日期")
    ax[1,0].set_title("(c) 334 日真实电价，保留季节及日内结构",loc="left",weight="bold")
    fig.colorbar(im,ax=ax[1,0],label="电价（元/kWh）",fraction=.035,pad=.02)
    idx=np.arange(3);w=.33
    for j,(role,label,col) in enumerate([("formal_reference","原参数 35/14",NAVY),("validation_selected","验证选参 14/28",BLUE)]):
        rows=fs[fs.role==role].set_index("split").loc[SPLITS]
        bars=ax[1,1].bar(idx+(j-.5)*w,rows.mae,width=w,color=col,label=label)
        for b,v in zip(bars,rows.mae):ax[1,1].text(b.get_x()+b.get_width()/2,v+.0003,f"{v:.4f}",ha="center",fontsize=9)
    ax[1,1].set_xticks(idx,[NAMES[s] for s in SPLITS]);ax[1,1].set_ylim(0,.115)
    ax[1,1].set_ylabel("价格 MAE（元/kWh）");ax[1,1].set_title("(d) 预测在 334 日上计算，验证期选参后测试",loc="left",weight="bold")
    ax[1,1].legend(frameon=False);ax[1,1].grid(axis="y",color=GRID)
    save(fig,figdir,"01_数据扩展与时序验证")


def split_of_month(month):return "development" if month<7 else ("validation" if month<10 else "test")


def risk_dashboard(out,figdir):
    risk=pd.read_csv(out/"risk_summary.csv")
    raw=pd.read_csv(out/"risk_daily.csv")
    pair=pd.read_csv(out/"risk_paired.csv")
    resolution=pd.read_csv(out/"resolution_summary.csv")
    fig,ax=plt.subplots(2,2,figsize=(14,8.5),layout="constrained")
    for sp in SPLITS:
        r=risk[risk.split==sp].sort_values("risk_weight")
        ax[0,0].plot(r.mean_emergency_kwh,r.mean_cost_yuan/1000,"o-",color=COLORS[sp],label=f"{NAMES[sp]}（{int(r.days.iloc[0])}日）")
        for i in [0,len(r)-1]:
            row=r.iloc[i];ax[0,0].annotate(f"ρ={row.risk_weight:g}",(row.mean_emergency_kwh,row.mean_cost_yuan/1000),xytext=(4,5),textcoords="offset points",fontsize=9)
    ax[0,0].set_xlabel("平均紧急购电（kWh/日）");ax[0,0].set_ylabel("平均费用（千元/日）")
    ax[0,0].set_title("(a) 风险偏好对应不同费用—可靠性位置",loc="left",weight="bold")
    ax[0,0].legend(frameon=False);ax[0,0].grid(color=GRID)
    x=np.arange(len(pair));lo=pair.mean_cost_delta_yuan-pair.month_cluster_ci_low_yuan;hi=pair.month_cluster_ci_high_yuan-pair.mean_cost_delta_yuan
    ax[0,1].errorbar(x,pair.mean_cost_delta_yuan/1000,yerr=np.array([lo,hi])/1000,fmt="o",capsize=4,color=CORAL,label="费用增加及月份聚类 95% 区间")
    twin=ax[0,1].twinx();twin.plot(x,pair.mean_emergency_reduction_kwh,"s--",color=NAVY,label="紧急量降低")
    ax[0,1].set_xticks(x,[f"{v:g}" for v in pair.risk_weight]);ax[0,1].set_xlabel("风险权重 ρ")
    ax[0,1].set_ylabel("相对 ρ=0 的平均费用增加（千元/日）",color=CORAL);twin.set_ylabel("紧急量降低（kWh/日）",color=NAVY)
    ax[0,1].set_title("(b) 66 日匹配差异，同时披露收益与代价",loc="left",weight="bold")
    ax[0,1].grid(axis="y",color=GRID)
    avg=raw.groupby("risk_weight")[["scenario_cvar90_yuan","legacy_top2_mean_yuan"]].mean()
    x=np.arange(len(avg));w=.35
    ax[1,0].bar(x-w/2,avg.scenario_cvar90_yuan/1000,w,color=NAVY,label="按概率精确 CVaR")
    ax[1,0].bar(x+w/2,avg.legacy_top2_mean_yuan/1000,w,color=BLUE,label="旧最高两场景均值")
    ax[1,0].set_xticks(x,[f"{v:g}" for v in avg.index]);ax[1,0].set_xlabel("风险权重 ρ");ax[1,0].set_ylabel("场景尾部成本（千元/日）")
    ax[1,0].set_title("(c) 非等概率场景必须用概率权重计算风险",loc="left",weight="bold")
    ax[1,0].legend(frameon=False);ax[1,0].grid(axis="y",color=GRID)
    r=resolution[resolution.split=="all"].sort_values("scenario_count")
    sc=ax[1,1].scatter(r.mean_solver_sec,r.mean_cost_yuan/1000,s=60+r.mean_emergency_kwh*.9,c=r.scenario_count,cmap=SEQ,edgecolors=NAVY,linewidth=1)
    for _,row in r.iterrows():ax[1,1].annotate(f"{int(row.scenario_count)}场景\n{row.mean_emergency_kwh:.0f} kWh/日",(row.mean_solver_sec,row.mean_cost_yuan/1000),xytext=(6,4),textcoords="offset points",fontsize=10)
    ax[1,1].margins(x=.25,y=.2);ax[1,1].set_xlabel("日前 LP 平均耗时（s）");ax[1,1].set_ylabel("平均费用（千元/日）")
    ax[1,1].set_title("(d) 场景分辨率、耗时与紧急量共同比较",loc="left",weight="bold");ax[1,1].grid(color=GRID)
    save(fig,figdir,"02_风险前沿与场景分辨率")


def policy_dashboard(out,figdir):
    summary=pd.read_csv(out/"rolling_summary.csv")
    sel=pd.read_csv(out/"rolling_selected.csv")
    chosen=sel[(sel.role=="validation_selected")&(sel.split=="test")].iloc[0]
    gains=sorted(summary.gain.unique());res=sorted(summary.reserve_quantile.unique())
    costm=[];emgm=[]
    for sp in SPLITS:
        r=summary[summary.split==sp]
        cm=r.pivot(index="reserve_quantile",columns="gain",values="mean_cost_yuan").loc[res,gains]
        em=r.pivot(index="reserve_quantile",columns="gain",values="mean_emergency_kwh").loc[res,gains]
        costm.append((cm-cm.loc[.2,.5])/1000);emgm.append(em)
    vmax=max(float(np.abs(x.values).max()) for x in costm)
    fig,axs=plt.subplots(2,3,figsize=(15.6,8.3),layout="constrained")
    maxem=max(x.values.max() for x in emgm)
    for j,sp in enumerate(SPLITS):
        for row,mat,cm in [(0,costm[j],DIV),(1,emgm[j],SEQ)]:
            ax=axs[row,j]
            im=ax.imshow(mat,origin="lower",aspect="auto",cmap=cm,vmin=-vmax if row==0 else 0,vmax=vmax if row==0 else maxem)
            ax.set_xticks(range(5),gains);ax.set_yticks(range(5),res);ax.set_xlabel("执行增益 γ");ax.set_ylabel("储备分位 τ")
            title=f"相对原参数费用（千元/日）" if row==0 else "紧急购电（kWh/日）"
            ax.set_title(f"({chr(97+row*3+j)}) {NAMES[sp]}：{title}",loc="left",weight="bold")
            for a in range(5):
                for b in range(5):
                    v=mat.iloc[a,b]
                    dark=abs(v)/vmax>.65 if row==0 else v/maxem>.65
                    ax.text(b,a,f"{v:+.2f}" if row==0 else f"{v:.0f}",ha="center",va="center",fontsize=9,color="white" if dark else "#223642")
            ax.scatter([gains.index(.5)-.22],[res.index(.2)+.22],marker="*",s=100,c=YELLOW,edgecolors="#3D505D",linewidths=.7)
            ax.scatter([gains.index(chosen.gain)],[res.index(chosen.reserve_quantile)],marker="s",s=200,facecolors="none",edgecolors=CORAL,linewidths=2)
            if j==2:fig.colorbar(im,ax=axs[row,:],fraction=.025,pad=.02,label="费用差（千元/日）" if row==0 else "紧急量（kWh/日）")
    fig.suptitle("控制参数跨时期稳定性：星号为原参数，方框为验证期最低费用参数",fontsize=14)
    save(fig,figdir,"03_控制相图与冻结测试")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--scale",choices=["pilot","full"],default="pilot")
    args=ap.parse_args();out=ROOT/("预实验结果" if args.scale=="pilot" else "全日期实验结果")
    figdir=ROOT/("预实验图像" if args.scale=="pilot" else "全日期实验图像")
    coverage(out,figdir);risk_dashboard(out,figdir);policy_dashboard(out,figdir)
    print(f"Figures saved to {figdir}")
