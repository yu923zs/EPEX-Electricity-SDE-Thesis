# EPEX 电力现货价格 SDE 建模：分段线性漂移与非对称 Tempered Stable 新息的两阶段估计

> 硕士学位论文配套代码。基于 EPEX SPOT DE-LU 小时级日前现货电价（2019-01–2025-09，约 59,159 条），实现"稀疏贝叶斯结点识别 → 经验特征函数最小距离估计"的两阶段流水线，并配套蒙特卡洛模拟、非对称性检验、稳健性检验与 Vasicek 对照。

## 项目简介

电价序列同时具有负值、重尾、左右不对称以及随价格水平变化的均值回复强度，单一高斯扩散模型无法刻画。本项目将去季节化后的电价建模为 Lévy 驱动的 Markov 过程：

**第一阶段 —— 漂移函数 b(x)**
把结点识别转化为固定铰链基（hinge basis）上的稀疏变量选择：候选结点取去季节化价格的 13 个样本分位点，构造多项式 + 截断幂基后，用可靠性加权的 Student-t 型 **spike-and-slab** 先验做贝叶斯变量选择，**Gibbs 采样**求解，并以 **Gelman–Rubin split-R̂**（实测 1.0003）验证收敛。相邻基函数近共线，故在簇（cluster）级别聚合后报告。

**第二阶段 —— Lévy 噪声**
非对称双边 tempered stable 测度

$$\nu(dz)=\left[C_+ z^{-1-\alpha}e^{-Mz}\mathbf{1}_{z>0}+C_-|z|^{-1-\alpha}e^{-G|z|}\mathbf{1}_{z<0}\right]dz$$

六个参数 $\xi=(\mu,C_+,C_-,M,G,\alpha)$ 通过**经验特征函数（ECF）最小距离**估计；权重矩阵取 $W=I$ 作为稳健性权衡。

**数据处理原则**：不填补缺失值、不删除异常值、不做季节调整之外的人为修正；区间边界按 Europe/Berlin 墙钟划定。

> **估计口径说明**：正文采用**全样本**估计（不做留出集划分），因此报告的是样本内结果。早期版本曾实现三段划分与样本外覆盖率检验，相关脚本已归档至 `code/legacy_v1/`。

## 数据

- 来源：[SMARD](https://www.smard.de/)（Bundesnetzagentur），Filter 4169，EPEX SPOT DE-LU Day-Ahead，小时级
- 区间：2019-01-01 00:00 – 2025-09-30 23:00 CET/CEST，约 59,159 条
- 仓库内置审计后的原始 CSV：`01_raw/epex_germany_DELU_hourly_2019_2025Q3_raw.csv`

## 快速开始

```bash
pip install -r requirements.txt

# 实证主流程（子命令：stageA / stageB / stageC / asymtest / robust / all）
python code/pipeline/run_empirical.py stageA     # 数据概况表 + 全样本季节拟合
python code/pipeline/run_empirical.py stageB     # Stage I 稀疏贝叶斯结构识别（Gibbs）
python code/pipeline/run_empirical.py stageC     # Stage II ECF 点估计 + bootstrap 标准误
python code/pipeline/run_empirical.py asymtest   # 参数自助零分布下的非对称性检验
python code/pipeline/run_empirical.py robust     # pre / crisis / post 三段稳健性

# 数值模拟与图表
python code/pipeline/run_stage_sims.py
python code/pipeline/run_vasicek.py
python code/pipeline/make_emp_tables.py
python code/pipeline/make_emp_figures.py
```

所有产物写入 `06_stage_outputs/`（已被 `.gitignore` 排除，可完整重新生成）。

### 路径配置

代码默认把**仓库根目录**当作基准路径（`BASE`），无需修改任何源码即可运行。若要把数据与输出放在别处，设置环境变量：

```bash
# Windows PowerShell
$env:EPEX_BASE = "D:\data\epex"
# Linux / macOS
export EPEX_BASE=/data/epex
```

脚本按 `BASE/01_raw/`（原始数据）、`BASE/05_logs/`（日志）、`BASE/06_stage_outputs/`（产出）定位文件。

## 项目结构

```
EPEX-Electricity-SDE-Thesis/
├── 01_raw/
│   └── epex_germany_DELU_hourly_2019_2025Q3_raw.csv   # SMARD 审计后原始数据
├── figures/
│   └── fig1_price_series.pdf … fig5_alpha_profile.pdf # 论文矢量图
└── code/
    ├── pipeline/                  # 新版主流程（两阶段）
    │   ├── thesis_pipeline_common.py   # 共享模块：装载 / 季节回归 / 分位结点 / 工具
    │   ├── run_empirical.py            # 实证章全流程（stageA/B/C、asymtest、robust）
    │   ├── run_stage_B.py              # Stage I：spike-and-slab Gibbs
    │   ├── run_stage_C.py              # Stage II：非对称 TS 的 ECF 估计
    │   ├── run_asymtest.py             # 参数自助零分布的非对称检验
    │   ├── run_stage_sims.py           # 蒙特卡洛模拟（多 DGP × 样本量）
    │   ├── run_vasicek.py              # Vasicek 2×2 对照
    │   ├── make_emp_tables.py          # 实证表格
    │   ├── make_emp_figures.py         # 实证插图
    │   ├── make_fig15_seg_qq.py        # 分段 QQ 诊断
    │   ├── diag_alpha_full.py          # 稳定性指数 α 的诊断
    │   ├── stage_5_oos_coverage_v2.py  # 样本外概率覆盖（方法增量）
    │   ├── stage_6_price_prediction.py # 价格预测扩展
    │   └── plot_fig7_5_offline.py
    ├── method_lab/                # 方法实验台：变体、基线、快速验证
    │   ├── run_lab.py             # stageI / profile_refine / VARIANTS
    │   ├── run_baselines.py       # 基线方法对照
    │   ├── make_figures.py
    │   └── debug_*.py, dump_*.py  # 调试与模型落盘工具
    └── legacy_v1/                 # 早期三阶段实现（保留供追溯）
        ├── run_stage_A/B/C.py
        ├── stage_4_cdf_quantiles*.py   # CDF / 分位数拟合
        ├── stage_5_oos_coverage*.py    # 样本外区间与 VaR 覆盖率（含 Christoffersen LR 检验）
        ├── run_ablation.py, run_robustness.py
        ├── verify_gibbs_enumerative.py # 2³ 可枚举模型验证 Gibbs 实现
        └── download_pipeline.py        # SMARD 下载 + 完整性审计
```

## 设计要点

- **无信息泄漏**：候选结点、可靠性权重等仅用训练区间确定后全局应用；
- **可识别性验证**：`legacy_v1/verify_gibbs_enumerative.py` 在 2³ 可枚举小模型上比较解析边际后验 PIP 与 Gibbs 采样 PIP，验证变量更新的贝叶斯因子与条件后验实现；
- **估计稳健性**：ECF 频率标度使用 MAD/0.6745；优化采用拉丁超立方多初值 + 两阶段局部搜索，输出边界命中与多初值收敛诊断；
- **可复现**：所有路径经 `__file__` 相对派生或环境变量覆盖，克隆后无需修改源码。

## 主要结果

- 漂移斜率在约 **−25.4 EUR/MWh** 处发生结构变化（集合识别，非点识别）；
- 下行跳跃强度为上行方向的 **1.66 倍**，2023 年后降至 **0.99**；
- Gibbs 三链的 split-R̂ 为 **1.0003**（截距）与 **1.0004**。

## 相关项目

方法学论文（稳健特征函数 GMM 与 VIX 实证）见独立仓库：[SDE-GMM-Estimation](https://github.com/yu923zs/SDE-GMM-Estimation)。

## License

MIT
