# 微网决策感知滚动优化 · 复现说明

[English README](README.md) · [数据说明](docs/DATA.md) · [模型说明](docs/MODEL.md) · [完整运行协议](docs/REPRODUCIBILITY.md)

本仓库整理当前COR论文使用的数据、建模求解代码、实验结果和复现入口。原始复制文件保留字节内容，并以SHA-256记录来源。所有重跑输出进入 `runs/`，不会覆盖参考结果。仓库现已公开，可用于编辑审阅与研究复现；原始来源标注及适用的数据授权条款继续保留。

## 从这里开始

使用Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py verify
python run.py replay --out runs/replay
python run.py compare --kind replay --input runs/replay --out runs/replay-check
```

1. `verify`：检查数据完整性、主键、因果价格滞后、原代码一致性、费用恒等式、SOC连续性和统计区间。
2. `replay`：实际重新求解92天、4种策略；不是读取参考费用直接输出。
3. `compare`：逐日比较重跑与参考结果，排除硬件相关运行时间，允许规定范围内的浮点误差。

## 其他实验

| 任务 | 命令 |
|---|---|
| 13组验证候选与完整测试 | `python run.py expanded --out runs/expanded` |
| 原控制参数网格与334天连续回放 | `python run.py initial --out runs/initial` |
| 负荷、光伏、电价与联合压力测试 | `python run.py stress --out runs/stress` |
| GRU重新训练 | 安装PyTorch 2.6.0后，`python run.py train --allow-cpu --out runs/training` |
| 重训练预测驱动的完整实验 | `python run.py expanded --predictions runs/training --out runs/expanded-retrained` |

所有命令均从仓库根目录运行，不依赖作者的D盘目录或集群账号。MATLAB仅用于重画论文统计图；训练与求解不依赖MATLAB。

## 对结果的正确理解

- 主测试为2025年10月至12月，共92天；150天开发、92天验证，不能把各策略和场景当成新的独立样本。
- 全幅调整费用4,280,800.67元，半幅调整4,304,804.70元；节省24,004.03元。
- 应急电量分别为11,859.595与19,572.716 kWh，应急天数分别27与35天。
- 334天是2月至12月可评估期，1月用于历史初始化。334天连续回放不等于92天测试的扩大独立样本。
- 当前正式结果使用已完成的CPU诊断GRU预测。原始六份GRU检查点未在本地归档；若包内提供新生成的检查点，会单独记录其重新训练来源，不冒充原权重。
- 上游数据集中还保留早期场景与离线控制实验；当前论文的主要结论来自新的按时序执行回放，二者不能混用。

运行证据位于 `validation/`。具体数据来源、字段、单位、缺失值含义、算法参数、停止条件和数值容差均在 `docs/` 中说明。

## 送审访问

完整复现包 `COR_microgrid_reproducibility.zip` 可作为审稿人可见的补充材料提交，解压后不依赖 GitHub 账户。仓库现已公开，无需 GitHub 账户即可访问；复现压缩包可作为相同归档文件的离线补充材料。

作者依次为 Dawen Tian、Jiayi Li、Xinghai Zhang、Rui Qian、Xiaozhi Du、Yi Gong、Tianhao Geng、Caiwei Yin、Fan Yang。两项基金均由 Xiaozhi Du 获取，贡献声明中记为 Funding acquisition；Fan Yang 提供计算资源。
