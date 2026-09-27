# EPEX 电力现货价格 SDE 建模：季节性剥离、稀疏贝叶斯漂移识别与双边 Tempered Stable ECF 估计

> 硕士毕业论文完整代码。基于 EPEX SPOT DE-LU 小时级日前现货电价（2019-01–2025-09），构建"季节性剥离 → 稀疏贝叶斯漂移函数识别 → 非对称双边 Tempered Stable  Lévy 噪声经验特征函数估计"的三阶段分析流水线，并配套数值模拟、消融实验、稳健性检验与样本外概率覆盖评估。

## 项目简介

电价序列同时具有强季节性、均值回复与极端尖峰，单一连续扩散模型难以刻画。本文将电价建模为带季节性的 Lévy 驱动 Markov 过程：

- **季节性 μ_t**：长波自然样条（年度）+ 周内虚拟变量 + 日内 Fourier 项，仅在 drift 训练区间拟合，避免信息泄漏；
- **漂移函数 b(x)**：可靠性加权的 Student-t 型 spike-and-slab 稀疏贝叶斯回归（Gibbs 采样），候选基为多项式 + 分位结点截断幂基；
- **Lévy 噪声**：非对称双边 tempered stable 测度

$$\nu(dz)=\left[C_+ z^{-1-\alpha}e^{-Mz}\mathbf{1}_{z>0}+C_-|z|^{-1-\alpha}e^{-G|z|}\mathbf{1}_{z<0}\right]dz$$

参数 $\xi=(\mu,C_+,C_-,M,G,\alpha)$ 通过经验特征函数（ECF）最小距离估计：频率网格经 MAD 稳健标度，多初值 LHS + L-BFGS-B + Nelder–Mead 两阶段优化，并与对称约束基线 $(C_+=C_-, M=G)$ 对比。

**数据处理原则**：不填补缺失值、不删除异常值、不做季节调整之外的人为修正；所有区间边界按 Europe/Berlin 墙钟划定。

## 数据

- 来源：[SMARD](https://www.smard.de/)（Bundesnetzagentur），Filter 4169，EPEX SPOT DE-LU Day-Ahead，小时级
- 区间：2019-01-01 00:00 – 2025-09-30 23:00 CET/CEST，约 59,159 条
- 三划分：drift 训练 ≤ 2022-06-30 ｜ TS 估计 ≤ 2024-12-31 ｜ 其后为样本外测试
- 仓库内置审计后的原始 CSV：`01_raw/epex_germany_DELU_hourly_2019_2025Q3_raw.csv`
- 如需重新下载：`python code/download_pipeline.py`（每周 JSON 分片不纳入仓库，运行时自动生成于 `01_raw/weekly_json_chunks/`）

## 快速开始

```bash
pip install -r requirements.txt

# 三阶段主流水线（顺序执行）
python code/run_stage_A.py     # 装载 → 季节项 → X_t = P_t − μ_t → 状态分箱
python code/run_stage_B.py     # spike-and-slab Gibbs 稀疏漂移识别（3 链，burn 10k / post 20k）
python code/run_stage_C.py     # 非对称 TS Lévy 噪声 ECF 最小距离估计

# 数值模拟、消融与稳健性
python code/run_stage_sims.py && python code/sim_merge_tex.py
python code/run_ablation.py
python code/run_robustness.py

# 模型诊断：CDF/分位点与样本外概率覆盖
python code/stage_4_cdf_quantiles.py
python code/stage_5_oos_coverage.py

# 论文插图（基于已落盘的真实实验产出，不捏造数据）
python code/make_thesis_figures.py
```

所有产物写入 `06_stage_outputs/`（已被 .gitignore 排除，可完整重新生成）。

## 项目结构

```
EPEX-Electricity-SDE-Thesis/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_raw/
│   └── epex_germany_DELU_hourly_2019_2025Q3_raw.csv   # SMARD 审计后原始数据
└── code/
    ├── thesis_pipeline_common.py     # 共享模块：时区/装载/3-way split/季节回归/分箱
    ├── download_pipeline.py          # SMARD 下载 + 完整性审计
    ├── run_stage_A.py                # 阶段 A：季节性剥离与状态分箱
    ├── run_stage_B.py                # 阶段 B：加权 spike-and-slab Gibbs
    ├── run_stage_C.py                # 阶段 C：双边 TS 的 ECF 估计
    ├── run_stage_sims.py             # 数值模拟（多 DGP × 样本量）
    ├── sim_merge_tex.py              # 模拟结果合并与 LaTeX 表格
    ├── run_ablation.py               # 消融实验（含 LassoCV 对照）
    ├── run_robustness.py             # 稳健性检验
    ├── compute_phase2_metrics.py     # 第二阶段指标汇总
    ├── stage_4_cdf_quantiles.py      # CDF / 分位数拟合诊断
    ├── stage_4_cdf_quantiles_v2.py
    ├── stage_5_oos_coverage.py       # 样本外概率覆盖检验
    ├── stage_5_oos_coverage_v2.py
    ├── verify_gibbs_enumerative.py   # Gibbs 逐变量后验的枚举解验证
    ├── make_thesis_figures.py        # 论文矢量插图
    └── _sim_tabs.py                  # 模拟表格快速检查脚本
```

## 设计要点

- **无信息泄漏**：候选基结点、可靠性权重、Winsor 阈值等仅用 drift 训练区间确定，再全局应用；
- **可识别性验证**：`verify_gibbs_enumerative.py` 在 2³ 可枚举小模型上比较解析边际后验 PIP 与 Gibbs 采样 PIP，验证变量更新的贝叶斯因子与条件后验实现；
- **估计稳健性**：ECF 频率标度使用 MAD/0.6745；优化采用 24 组拉丁超立方初值与两阶段局部搜索，输出边界命中与多初值收敛诊断；
- **可复现**：所有路径经 `__file__` 相对派生，克隆后无需修改任何代码。

## 相关项目

方法学论文（PC-CF-IARGMM 鲁棒 GMM 与 VIX 实证）见独立仓库：[SDE-GMM-Estimation](https://github.com/yu923zs/SDE-GMM-Estimation)。

## License

MIT
