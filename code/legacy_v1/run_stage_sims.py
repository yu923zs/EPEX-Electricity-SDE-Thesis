# -*- coding: utf-8 -*-
"""
Stage SIM:  第 6 章仿真实验的主调度
=====================================================

设计约束（严格对齐论文第 5 / 6 章主模型约束 4–9）：
  (4)  工作似然：IG 混合 Student-t 厚尾 + 可靠性权重 w_k（非 NIG）
  (5)  Spike-Slab: 严格点质量 (θ=0 ⇔ β=0)
  (6)  Gibbs 5 闭式条件后验：λ/σ² IG, [γ,β_active] 联合正态 (Chol),
       θ_j via O_j 解析贝叶斯因子, τ² IG
  (7)  Plug-in 残差 r_i = ΔX_i − b̂(X_{t_i})·Δt  — 不用 √Δt 标准化
  (8)  Lévy measure: 非对称 TS (CGMY 型), ν(dz)=C₊ z^{−1−α}e^{−M z} (z>0)
                                                     + C₋ |z|^{−1−α}e^{−G |z|} (z<0)
       h(z)=z·1_{|z|≤1} （ψ 公式使用该截断）
  (9)  ECF obj W=I (主), 并以 ASY 主模型 vs SYM 嵌套基线 (C₊=C₋, M=G) 比较。

DGP (3 个，小时 Δt=1h, 小时级)：
  DGP 1 对称 TS + 1 折点稀疏漂移
    b₀(x)=γ₀+β₁x+β₃(x-c₃)₊   (候选库还含 x², x³ + 6 假 hinge → 稀疏挑战)
    Lₜ sym TS: C₊=C₋=C=1.5, M=G=0.03, α=0.75, μ=0

  DGP 2 非对称 TS + 同漂移（主场景、验证 TS 参数恢复 & Plug-in 质量）
    b₀ = DGP 1
    Lₜ ASY TS: C₊=0.9, C₋=1.8, M=0.02, G=0.06, α=0.60, μ=0.6
         (C₋/C₊≈2, G/M=3, 与真实数据同量级的不对称结构)

  DGP 3 漂移结构变点 + ASY TS （鲁棒场景）
    t<T/2 ：DGP 2 漂移
    t≥T/2：加一个额外折点项 β₆(x−c₆)₊，TS 参数不变
    （验证 Stage I 的季节 + Gibbs 对漂移结构改变的稳健性；
      注意：主方法的季节是 Fourier+dow+hinge 无全局趋势，
      训练段对漂移结构突变具有抵抗力，预期 TPR 相比 DGP2 略降）

评估 (Tab 6.1/6.2)：
  Tab 6.1 — 稀疏恢复：TPR / FDR / MedSE / |Â| 中位数（PIP>0.5）
  Tab 6.2 — TS 参数：Oracle vs Plug-in 的 RMSE × 6 参数 + 95% 覆盖率
                (asy & sym baseline 同时报告)
  Tab 6.3 — Q 模型选择正确率 / Q_asy < Q_sym 比例 / 收敛率 (8/8)
                跨 DGP 的 drift MSE 随 T 的下降

总运算规模（默认）： S=10（种子） × 3 DGP × 3 T（3k/10k/25k）= 90 重复
  · 单重复 DGP3 T=25k 估计 ~2-3 min → 总计预计 90×2.5 min ≈ 225 分钟；
    实际我们为了可复现性将 Stage B burn/post / Stage C 起始点数略降：
      StageB Gibbs: 1 链, burn=2000, post=5000 （真实 3×10k/20k）
      StageC starts: 非对称 8, 对称 6 （真实 24/24）
    → 单重复预计 ~ 30–90 秒 (T=3k) / 1–3 min (T=10k) / 3–8 min (T=25k)
    → S=10 总计 90 job ≈ 3–6 小时。
  · 可选传参：--quick（S=2, 仅 DGP 2, T=3k/10k）用于 5–15 分钟级自检。
  · 若指定 --only-T=3000 --only-DGP=2 --S=1：可用于 ~1 分钟快速冒烟。
"""

import os, sys, time, pickle, argparse, itertools
from typing import Optional
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from thesis_pipeline_common import (
    build_drift_work_samples, save_pickle, ensure_dir,
)
from run_stage_B import (
    build_phi, gibbs_ss, PRIOR as B_PRIOR_DEFAULT,
)
from run_stage_C import (
    run_multi_start, build_ecf_data,
)

BASE = os.path.abspath(os.path.join(HERE, ".."))
SIM_DIR = os.path.join(BASE, "06_stage_outputs", "stage_sim_results")
ensure_dir(SIM_DIR)

# ---------- 候选基（匹配真实 stage B：x¹,x²,x³ + 6 个 hinge = 9，共 p=12） ----------
CANDIDATE_HINGE_QUANTS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85]
POLY_PWRS = [1, 2, 3]  # x, x², x³


# ---------- 季节：简化为 Fourier 5×2h + dow 6，无长波 hinge (和真实一致) ----------
def gen_seasonal(t_cet, nh: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(t_cet)
    tdt = pd.Series(t_cet)
    hod = tdt.dt.hour.to_numpy(dtype=np.float64) + tdt.dt.minute.to_numpy(dtype=np.float64)/60.0
    dow = tdt.dt.dayofweek.to_numpy(dtype=np.int64)
    # Fourier
    om = 2*np.pi/24.0
    F = np.empty((n, 2*nh), dtype=np.float64)
    beta_F = rng.normal(0, [8, 6, 4, 2, 1, 6, 5, 3, 2, 1], size=2*nh)  # 24h
    for k in range(nh):
        F[:, 2*k]   = np.cos((k+1)*om*hod)
        F[:, 2*k+1] = np.sin((k+1)*om*hod)
    # dow
    D = np.zeros((n, 6), dtype=np.float64)
    beta_D = rng.normal(0, [3, 2, 2, 2, 3, 4], size=6)
    for d in range(6):
        D[dow == d, d] = 1.0
    mu = 70.0 + F @ beta_F + D @ beta_D
    # 加一点长期平滑微趋势（保持在 ±15 之内，避免爆炸）
    t_norm = np.linspace(-1, 1, n)
    mu += 6*np.sin(2*np.pi * 0.05 * t_norm * np.sqrt(n/3000))
    return mu


# ---------- DGP 参数 ----------
# 折点采用预固定值（不依赖待生成样本），避免循环定义。
# 候选库显式包含这些固定折点，确保真值在候选空间内。
# DGP1: 仅线性漂移（对称 TS 基线）
# DGP2: 线性 + 2 个折点（非对称 TS，主场景）
# DGP3: DGP2 + 切换后叠加第 3 个折点（非对称 TS，鲁棒场景）

_TRUE_GAMMA = 0.8
_TRUE_BETA_POLY1 = -0.05
_TRUE_HINGE_COEFS = (-0.06, +0.07, -0.04)  # hinge1, hinge2, switch
# 预固定折点（EUR/MWh），基于实证章 Stage B 的高 PIP 折点位置
_TRUE_C_HINGE1 = -14.94
_TRUE_C_HINGE2 = 28.31
_TRUE_C_SWITCH = 5.0

TS_DGP = {
    1: dict(name="DGP1_SymTS",
            xi=(0.0, 1.5, 1.5, 0.03, 0.03, 0.75),
            symmetric=True, switch=False, n_hinges=0),
    2: dict(name="DGP2_AsymTS",
            xi=(0.6, 0.9, 1.8, 0.02, 0.06, 0.60),
            symmetric=False, switch=False, n_hinges=2),
    3: dict(name="DGP3_Switch_AsymTS",
            xi=(0.6, 0.9, 1.8, 0.02, 0.06, 0.60),
            symmetric=False, switch=True, n_hinges=3),
}


# ---------- TS 增量采样：精确截断 Lévy-Poisson 序列 (tempered stable via rejection / compound Poisson 近似)
# 实现：使用 Gaussian quadrature + 复合 Poisson 近似：
#   将 Lévy 强度近似为小跳连续 + 大跳 Poisson（阈值 a=0.05）。
#   对于 α 接近 0.6，小时 Δt=1, M=0.02, C=0.9；
#   小跳用高斯（中心 CLT），大跳用 Poisson 强度 λ = ν(|dz|>a) 采样 + 接受-拒绝 tempered 尾。
def sample_ts_increments(n: int, xi6, rng, a_small: float = 0.05):
    mu, Cp, Cm, M, G, alpha = [float(v) for v in xi6]
    dt = 1.0
    # (1) 小跳 |z|∈(0, a] 的前两阶：构造高斯 N(m_small, v_small)
    # J0+ = C₊ ∫₀ᵃ e^{-M z} z^{−1−α} z  dz = C₊ ∫₀ᵃ z^{-α} e^{-M z} dz
    from scipy import integrate as _ig

    def ig(f, lo, hi):
        return float(_ig.quad(f, lo, hi, limit=800, epsabs=1e-9, epsrel=1e-7)[0])
    # positive tail | small
    m_p_s = Cp * ig(lambda z: np.exp(-M*z) * z ** (-alpha), 1e-12, a_small)
    v_p_s = Cp * ig(lambda z: np.exp(-M*z) * z ** (1 - alpha), 1e-12, a_small)  # E[z²] 其实差一次幂（E[X²]）
    # 更正：小跳 S 的 2 阶 = C₊ ∫ z² e^{-M z} z^{-1-α} dz = C₊ ∫ z^{1-α} e^{-M z} dz —— 上面就是这个！
    # 一阶: ∫ z ν(dz) = C₊ ∫ z·z^{-1-α} e^{-M z} dz = C₊ ∫ z^{-α} e^{-M z} dz   OK
    # negative tail
    m_n_s = -Cm * ig(lambda z: np.exp(-G*z) * z ** (-alpha), 1e-12, a_small)   # z < 0 所以 - |z|
    v_n_s = Cm * ig(lambda z: np.exp(-G*z) * z ** (1 - alpha), 1e-12, a_small)   # z² 项 正
    m_small = m_p_s + m_n_s
    v_small = v_p_s + v_n_s  # = E[S_small^2] (mean 加到小跳均值中)
    # 注意：h(z)=z·1_{|z|≤1} 截断项的漂移已在 ψ 公式中体现；
    #       这里直接 sample increments via Lévy triplet 不需要额外截断漂移
    #       (我们用的积分是原始 ν，和 ECF ψ 公式保持一致：ψ 中漂移是 μ·u + Im∫(e^{iuz}-1-iuh(z)) ν(dz)
    #       —— 我们采样的 L_{t+1}-L_t 应满足 E[e^{iu ΔL}] = exp(ψ(u))。
    #       这里近似 ΔL = drift_correction + small jump Gaussian + large jump CP)。
    #       我们用 ψ 计算均值 E[ΔL] = -ψ'(-i 0)= μ + ∫_{|z|>1} z ν(dz)
    #       （因为 e^{iuz}-1-iuh(z) 的导数在 u=0 给出：i (h(z) + ∫ e^{0} ... 展开太繁琐，
    #       我们直接使用数值方式：在 u=0 处有限差分得到 drift）
    from run_stage_C import ts_phi_dt
    h = 1e-4
    phi_up = ts_phi_dt(np.array([-1j*h]), xi6, dt=1.0)[0]  # φ(-ih)=exp(ψ(-ih))
    # log(φ(-ih))/h ≈ ψ'(-i·0) (右导数) = -E[ΔL]
    # → E[ΔL] = -(1/h)*log φ(-ih)
    mean_exact = float(np.real(-np.log(phi_up)/h))
    # 大跳 Poisson 强度：(a, ∞) 正负尾
    lam_pos = Cp * ig(lambda z: np.exp(-M*z) * z ** (-1 - alpha), a_small, np.inf)
    lam_neg = Cm * ig(lambda z: np.exp(-G*z) * z ** (-1 - alpha), a_small, np.inf)
    N_pos = rng.poisson(lam_pos * dt, size=n)
    N_neg = rng.poisson(lam_neg * dt, size=n)
    big_pos = np.zeros(n, dtype=np.float64)
    big_neg = np.zeros(n, dtype=np.float64)
    # 接受-拒绝采样：f(z)=C·z^{-1-α} e^{-rate z} / 包络 g(z)∝ z^{-1-α}
    for i in range(n):
        for _ in range(N_pos[i]):
            # 采样 f(z) ∝ z^{-1-α} e^{-M z} , z≥a_small, 用 envelope z^{-1-α} 拒绝
            while True:
                # envelope proposal via inverse CDF: z=a_small * U^{-1/α}
                u = rng.uniform(0.0, 1.0 - 1e-16)
                z_prop = a_small * (u ** (-1.0 / max(alpha, 1e-6)))
                # 接受概率 = exp(-M z_prop)
                if rng.random() < np.exp(-M * z_prop):
                    big_pos[i] += z_prop
                    break
        for _ in range(N_neg[i]):
            while True:
                u = rng.uniform(0.0, 1.0 - 1e-16)
                z_prop_abs = a_small * (u ** (-1.0 / max(alpha, 1e-6)))
                if rng.random() < np.exp(-G * z_prop_abs):
                    big_neg[i] -= z_prop_abs
                    break
    # 小跳：高斯，均值 m_small，方差 v_small - m_small**2
    var_small = max(0.0, v_small - m_small*m_small)
    small = rng.normal(m_small, np.sqrt(var_small), size=n)
    # 联合增量 ΔL，并校正到 理论均值 mean_exact
    raw = big_pos + big_neg + small
    correction = mean_exact - float(np.mean(raw))
    out = raw + correction
    return out


# ---------- 主仿真单元 ----------
def simulate_one(seed: int, dgp_id: int, T_levels: int, *,
                 quick: bool = False):
    rng = np.random.default_rng(seed)
    cfg = TS_DGP[dgp_id]
    xi6_true = cfg["xi"]
    T_inc = T_levels - 1

    # (1) 时间序列 (CET, 每小时)
    ts0 = pd.Timestamp("2020-01-01 00:00:00", tz="Europe/Berlin")
    ts_cet = pd.date_range(ts0, periods=T_levels, freq="h", tz="Europe/Berlin")

    # (2) 季节分量
    mu_seas = gen_seasonal(ts_cet, nh=5, seed=seed + 1000)

    # (3) 非季节水平 X:  模拟 dX = b(X)dt + dL, 用 Euler
    #   使用预固定折点 c_h1/c_h2/c_sw（不依赖待生成样本），避免循环定义。
    c_h1 = _TRUE_C_HINGE1
    c_h2 = _TRUE_C_HINGE2
    c_sw = _TRUE_C_SWITCH
    n_hinges = cfg["n_hinges"]
    TRUE_HINGES_THIS_RUN = []
    if n_hinges >= 1:
        TRUE_HINGES_THIS_RUN.append((c_h1, _TRUE_HINGE_COEFS[0]))
    if n_hinges >= 2:
        TRUE_HINGES_THIS_RUN.append((c_h2, _TRUE_HINGE_COEFS[1]))
    if n_hinges >= 3:
        TRUE_HINGES_THIS_RUN.append((c_sw, _TRUE_HINGE_COEFS[2]))

    after_sw = None
    if cfg["switch"]:
        after_sw = np.zeros(T_levels, dtype=bool)
        after_sw[T_levels//2:] = True

    def true_drift(x_arr, after_switch_arr):
        b = _TRUE_GAMMA + _TRUE_BETA_POLY1 * x_arr
        for (c, coef) in TRUE_HINGES_THIS_RUN:
            if c == c_sw:
                # switch hinge 不在此循环中施加（避免重复计数）
                # 它仅在 after_switch 块中按时间激活
                continue
            b = b + coef * np.maximum(x_arr - c, 0.0)
        if after_switch_arr is not None:
            # switch hinge 仅在切换后激活
            b = b + np.where(after_switch_arr,
                             _TRUE_HINGE_COEFS[2] * np.maximum(x_arr - c_sw, 0.0),
                             0.0)
        return b

    X = np.zeros(T_levels, dtype=np.float64)
    X[0] = float(rng.normal(0, 3.0))
    dL = sample_ts_increments(T_inc, xi6_true, rng)  # length T_inc
    for i in range(T_inc):
        sw_here = None if after_sw is None else np.array([after_sw[i]])
        b_here = float(true_drift(np.array([X[i]]), sw_here)[0])
        X[i+1] = X[i] + b_here * 1.0 + dL[i]

    # (4) 观测价格 P = X + season
    P = X + mu_seas

    # --- Stage A' 季节 (等效真实 Stage A fit_seasonal + drift_work 构造)
    # 使用真实等价 fit_seasonal: train on 全量 (因为仿真没 train/est/test 分裂) + 构造 drift 工作
    df = pd.DataFrame({
        "ts_cet": ts_cet,
        "epoch_ms_utc": (ts_cet.astype("int64") // 10**6).astype(np.int64),
        "price_eur_mwh_raw": P.astype(np.float64),
    })
    # 简化季节 OLS：使用真实实现 fit_seasonal(train=df, full=df)  → 必须拆分 drift_train vs rest
    # 在仿真中没有 TS-est 分裂：所有行作为 drift_train
    from thesis_pipeline_common import fit_seasonal
    artA_seas = fit_seasonal(df, df, hod_nharm=5)
    X_deseas = P - artA_seas["mu_all"].to_numpy(dtype=np.float64)
    ts_cet_arr = ts_cet
    t_train = ts_cet_arr.to_numpy() if hasattr(ts_cet_arr, "to_numpy") else np.asarray(ts_cet_arr)
    # Drift work (K=80 以缩短 Stage B; 真实是 K=100)
    dw = build_drift_work_samples(X_deseas, t_train=t_train,
                                  K=80, tau_q=0.95, n_min=20,
                                  winsor_pct=0.95, eps_v=1e-6, dt_hours=1.0)

    x_k = dw["x_k"]; b_hat = dw["b_hat"]; w_k = dw["w_k"]
    print(f"  [diag-dgp] X_level: [{float(np.nanmin(X_deseas)):.1f}, {float(np.nanmax(X_deseas)):.1f}] "
          f"med={float(np.nanmedian(X_deseas)):.2f}  std={float(np.nanstd(X_deseas)):.2f}", flush=True)
    print(f"  [diag-dgp] dL: [{float(np.nanmin(dL)):.2f}, {float(np.nanmax(dL)):.2f}] "
          f"med={float(np.nanmedian(dL)):.2f}  std={float(np.nanstd(dL)):.2f}", flush=True)
    print(f"  [diag-stageA] K_kept={len(x_k)} b_hat: [{float(np.nanmin(b_hat)):.2f}, {float(np.nanmax(b_hat)):.2f}] "
          f"med={float(np.nanmedian(b_hat)):.2f}  sd={float(np.nanstd(b_hat)):.2f}", flush=True)
    print(f"  [diag-stageA] x_k: [{float(np.nanmin(x_k)):.1f}, {float(np.nanmax(x_k)):.1f}]", flush=True)

    # --- Stage B' Gibbs SS (1 chain, light)
    prior = dict(B_PRIOR_DEFAULT)
    prior["n_chains"] = 1
    prior["n_burn"] = 2000 if not quick else 600
    prior["n_post"] = 5000 if not quick else 1500
    prior["seed"] = seed + 2000
    # 加密 hinge 候选分位（50 个）覆盖 X 分布范围；
    # 真值折点为预固定值，由 true_nz_mask 中 5 EUR 邻域匹配确定。
    prior["hinge_c_quantiles"] = np.linspace(0.02, 0.98, 50)
    Phi, colnames, metaPhi = build_phi(x_k, x_k, prior)
    p = Phi.shape[1]

    # 初始化：加权 OLS 的 top-half
    Wsr = np.sqrt(w_k)
    Aw = np.column_stack([Wsr, (Phi * Wsr[:, None])])
    yw = b_hat * Wsr
    beta_ols, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    t_vals = np.zeros(p)
    try:
        res = yw - Aw @ beta_ols
        s2 = float(np.sum(res**2) / max(len(yw)-Aw.shape[1], 1))
        covb = s2 * np.linalg.inv(Aw.T @ Aw + 1e-12 * np.eye(Aw.shape[1]))
        t_vals = np.abs(beta_ols[1:]) / np.sqrt(np.maximum(np.diag(covb)[1:], 1e-300))
    except Exception:
        pass
    init_c = (t_vals >= np.median(t_vals))
    inits = [init_c]
    all_trace = {"gamma": [], "beta": [], "theta": [],
                 "sigma2": [], "tau2": [], "loglik": []}
    all_final = []
    for cid, th0 in enumerate(inits):
        n_burn = prior["n_burn"]; n_post = prior["n_post"]
        ch_seed = prior["seed"] + 1000*cid
        # gibbs_ss 返回 (trace_dict, final_dict)；trace 直接含 "gamma"(n_post,) 等
        trace_ch, final_ch = gibbs_ss(b_hat, Phi, w_k, prior, n_burn, n_post, ch_seed, th0)
        all_trace["gamma"].append(trace_ch["gamma"])   # shape (n_post,)
        all_trace["beta"].append(trace_ch["beta"])     # shape (n_post, p)
        all_trace["theta"].append(trace_ch["theta"])
        all_trace["sigma2"].append(trace_ch["sigma2"])
        all_trace["tau2"].append(trace_ch["tau2"])
        all_trace["loglik"].append(trace_ch["loglik"])
        all_final.append(final_ch)
    all_trace["gamma"] = np.array(all_trace["gamma"])  # (1, n_post)
    all_trace["beta"] = np.array(all_trace["beta"])    # (1, n_post, p)
    all_trace["theta"] = np.array(all_trace["theta"]).astype(np.int8)
    all_trace["sigma2"] = np.array(all_trace["sigma2"])
    all_trace["tau2"] = np.array(all_trace["tau2"])
    all_trace["loglik"] = np.array(all_trace["loglik"])


    # 汇总 PIP / 选择
    # theta 形状 (1, n_post, p)；PIP 对每个 (chain, draw) 求均值 (chain, draw)
    pip = np.mean(all_trace["theta"].astype(bool), axis=(0, 1))
    selected = pip > 0.5
    # 真实非零指标：对照 colnames（poly x^1, x^2, x^3, hinges c）
    def true_nz_mask_for_dgp(cols, meta):
        nz = np.zeros(len(cols), dtype=bool)
        # x^1 非零 (真 β₁ 非零)
        poly1_idx = None
        for i, c in enumerate(cols):
            if c == "x^1" or c.startswith("x^1"):
                nz[i] = True; poly1_idx = i
        # hinges: 此运行真实折点 = TRUE_HINGES_THIS_RUN
        hinges_idx = []; hinges_cval = []
        import re as _re_nb
        _hr = _re_nb.compile(r"\(x-(.*?)\)_\+")  # 内部 literal 作为 float（双负 = 负）
        for i, c in enumerate(cols):
            mm = _hr.search(c)
            if not mm:
                continue
            try:
                cn = float(mm.group(1).strip())
            except Exception:
                continue
            hinges_idx.append(i); hinges_cval.append(cn)
        hinges_idx = np.array(hinges_idx, dtype=np.int64); hinges_cval = np.array(hinges_cval, dtype=np.float64)
        for (cval_true, _coef) in TRUE_HINGES_THIS_RUN:
            if len(hinges_cval) == 0:
                continue
            dists = np.abs(hinges_cval - cval_true)
            nearest_j = int(np.argmin(dists))
            # 若最近候选 < 5 EUR 距离 → 视作真项（candidate 基于 X quantile 构造时应 <1-2 EUR）
            if dists[nearest_j] < 5.0:
                nz[int(hinges_idx[nearest_j])] = True
        # 诊断
        true_cvals = [round(h[0], 3) for h in TRUE_HINGES_THIS_RUN]
        print(f"  [diag-true_nz] hinges_true_cands = {true_cvals}", flush=True)
        print(f"  [diag-true_nz] hinges_cand_col   = {[round(v,3) for v in hinges_cval.tolist()]}", flush=True)
        print(f"  [diag-true_nz] poly1 idx={poly1_idx}  nz_idx={np.where(nz)[0].tolist()}  cols_nz={[cols[i] for i in np.where(nz)[0]]}", flush=True)
        print(f"  [diag-true_nz] PIP at nz = {pip[nz].round(3).tolist()}", flush=True)
        top = np.argsort(-pip)[:6]
        print(f"  [diag-true_nz] Top-6 PIP cols: {[(cols[j], round(float(pip[j]),3)) for j in top]}", flush=True)
        return nz

    true_nz = true_nz_mask_for_dgp(colnames, metaPhi)
    # —— 指标：容忍过完备基（hinge 邻域内任一选中 = 识别成功，邻域内多选不算 FD） ——
    TOL_HINGE = 8.0  # EUR，邻域半径
    hinge_cands_arr = np.array([
        (i, float(c))
        for (i, c) in zip(
            [j for j in range(len(colnames)) if re.search(r"\(x[+-].*\)_\+", colnames[j])],
            [re.search(r"\(x-(.*?)\)_\+", colnames[j]).group(1)
             for j in range(len(colnames)) if re.search(r"\(x[+-].*\)_\+", colnames[j])])
    ], dtype=object) if False else None
    # 重建所有 hinge（col idx → c_val）
    import re as _re2
    _hre2 = _re2.compile(r"\(x-(.*?)\)_\+")
    hinge_map = {}  # idx -> c_true
    for i, cname in enumerate(colnames):
        mm = _hre2.search(cname)
        if mm:
            inner = mm.group(1).strip()
            try:
                hinge_map[i] = float(inner)   # handles "+26" -> ValueError? Nope, format is always (x-{c})_+
            except Exception:
                pass
    # 真 hinge c 集合（TRUE_HINGES_THIS_RUN）+ poly1 真索引
    true_hinge_cvals = np.array([float(h[0]) for h in TRUE_HINGES_THIS_RUN], dtype=np.float64)
    poly1_true_idx = None
    for i, c in enumerate(colnames):
        if c == "x^1" or c.startswith("x^1"):
            poly1_true_idx = i; break
    # --- (A) TPR（真项命中数 / 真项总数）
    # poly1: 是否 selected
    poly_hit = int(bool(selected[poly1_true_idx])) if poly1_true_idx is not None else 0
    # 每个真 hinge：邻域内是否存在选中 hinge
    hinge_hits = 0
    for tc in true_hinge_cvals:
        # 找任一 selected hinge idx 在邻域
        found = False
        for idx_sel in np.where(selected)[0]:
            if idx_sel in hinge_map and abs(hinge_map[idx_sel] - tc) < TOL_HINGE:
                found = True; break
        hinge_hits += int(found)
    tp_new = poly_hit + hinge_hits
    pos_tot_new = (1 if poly1_true_idx is not None else 0) + len(true_hinge_cvals)
    tpr = tp_new / max(pos_tot_new, 1)
    # --- (B) FDR（虚假发现数 / 选中数）
    selected_idx = np.where(selected)[0].astype(np.int64)
    false_discoveries = 0
    for si in selected_idx:
        if si == poly1_true_idx:
            continue  # poly1 真项
        if si in hinge_map:
            c_sel = hinge_map[si]
            # 是否在任意真 hinge 邻域？
            in_nbhd = bool(np.any(np.abs(true_hinge_cvals - c_sel) < TOL_HINGE))
            if not in_nbhd:
                false_discoveries += 1
        else:
            # x^2, x^3 等多项式（真系数为零）
            false_discoveries += 1
    sel_tot_new = int(selected.sum())
    fdr = false_discoveries / max(sel_tot_new, 1)
    # 诊断
    print(f"  [diag-metrics-new] poly_hit={poly_hit}  hinges_hit={hinge_hits}/{len(true_hinge_cvals)}  "
          f"tp={tp_new}/{pos_tot_new} → TPR={tpr:.2f}   FD={false_discoveries}/{sel_tot_new} → FDR={fdr:.2f}", flush=True)
    # 兼容变量名：保持旧变量 (tp/fp/pos_tot/sel_tot) 不再使用，用新 tp_new/false_discoveries
    # --- Oracle vs Plug-in TS residuals (仅 TS 参数比较)
    # oracle r = ΔX - b₀(X) 已知 drift + true drift
    X_inc = np.diff(X_deseas)  # T_inc
    X_level = X_deseas[:-1]
    after_sw_vec = None
    if cfg["switch"]:
        after_sw_vec = after_sw[:-1]
    b0_vec = true_drift(X_level, after_sw_vec)  # 时间依赖真值（DGP3 切换前后不同）
    r_or = X_inc - b0_vec
    # plug-in: b̂_med(X_level)
    # 构造 P x_level → Phi(X_level) → γ + Φ β_med (使用 gibbs 中值)
    gamma_med = float(np.median(all_trace["gamma"]))
    beta_med = np.median(np.reshape(all_trace["beta"], (-1, p)), axis=0)
    Phi_level, _, _ = build_phi(x_k, X_level, prior)  # x_train_for_design=x_k, x_eval=X_level
    b_plugin = gamma_med + Phi_level @ beta_med
    r_pi = X_inc - b_plugin
    # MedSE: b̂_med(X_level) 与时间依赖真实 b_0(X_level, t) 逐点对比
    # 对 DGP1/2，after_sw_vec=None 故 b0_vec 即 b_0(X)；
    # 对 DGP3，b0_vec 随切换时间变化，正确反映时变真值。
    se_seq = (b_plugin - b0_vec) ** 2
    medse = float(np.median(se_seq))
    coverage_prop = float(np.nan)  # 逐点覆盖昂贵（后略）

    # --- Stage C: ECF fits (ASY + SYM baseline), 轻量 starts
    def _fit(rr, n_start_asy=6, n_start_sym=4, seed_=20260900):
        # 返回 (xi, Q, conv_count, n_hits_boundary_sum, n_start) × 2  (asy, sym)
        #   xi for ASY = 6-tuple or None; SYM = 4-tuple or None
        #   conv_count = # starts 收敛 (int);
        #   n_start = 实际 starts
        try:
            dat = build_ecf_data(rr)
        except Exception as e:
            print(f"  [sim-stageC build_ecf fail] {e}", flush=True)
            return (None, float("inf"), 0, 0, n_start_asy), (None, float("inf"), 0, 0, n_start_sym)
        # ASY
        try:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_a = run_multi_start(dat, symmetric=False, n_start=n_start_asy, seed=seed_)
            conv_a = int(sum(1 for r in res_a if r.get("converged")))
            hits_a = int(sum(r.get("bounds_hit_count", 0) for r in res_a))
            best_a = res_a[0]; xi_best_a = tuple(best_a["xi_final"].tolist() if hasattr(best_a["xi_final"],'tolist') else list(best_a["xi_final"]))
            Q_a = float(best_a["Q_final"])
            ok_a = True
        except Exception as e:
            ok_a = False
            print(f"  [sim-stageC ASY fail] {e}", flush=True)
            xi_best_a, Q_a, conv_a, hits_a = None, float("inf"), 0, 0
        # SYM
        try:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    res_s = run_multi_start(dat, symmetric=True, n_start=n_start_sym, seed=seed_+1)
            conv_s = int(sum(1 for r in res_s if r.get("converged")))
            hits_s = int(sum(r.get("bounds_hit_count", 0) for r in res_s))
            best_s = res_s[0]; xi_best_s = tuple(best_s["xi_final"].tolist() if hasattr(best_s["xi_final"],'tolist') else list(best_s["xi_final"]))
            Q_s = float(best_s["Q_final"])
            ok_s = True
        except Exception as e:
            ok_s = False
            print(f"  [sim-stageC SYM fail] {e}", flush=True)
            xi_best_s, Q_s, conv_s, hits_s = None, float("inf"), 0, 0
        return (xi_best_a, Q_a, conv_a, hits_a, n_start_asy), (xi_best_s, Q_s, conv_s, hits_s, n_start_sym)

    def _conv_valid(conv_cnt, n):
        return int(bool(n > 0 and conv_cnt * 2 >= n))  # ≥ 50% starts 收敛计为有效

    # Oracle
    fits_or_a, fits_or_s = _fit(r_or, n_start_asy=6, n_start_sym=4, seed_=seed+5000)
    # Plugin
    fits_pi_a, fits_pi_s = _fit(r_pi, n_start_asy=6, n_start_sym=4, seed_=seed+6000)

    out_row = dict(
        seed=seed, dgp_id=dgp_id, dgp_name=cfg["name"], T_levels=T_levels,
        sym_dgp=cfg["symmetric"], switch=cfg["switch"],
        # Tab 6.1 drift selection (邻域容忍指标)
        tpr=tpr, fdr=fdr, medse=medse,
        size_selected=sel_tot_new, true_positive=tp_new, false_positive=false_discoveries,
        pos_total=pos_tot_new,
        # TS params true
        true_mu=xi6_true[0], true_Cp=xi6_true[1], true_Cm=xi6_true[2],
        true_M=xi6_true[3], true_G=xi6_true[4], true_alpha=xi6_true[5],
    )
    # 依次 pack Oracle ASY → SYM → Plug-in ASY → SYM 参数 (6 维 + Q + conv_valid + n_hits)
    def _pack_asy6(xi):
        return [np.nan]*6 if xi is None else list(xi)
    def _pack_sym6(xi4):
        if xi4 is None: return [np.nan]*6
        mu_s, C_s, M_s, a_s = xi4
        return [mu_s, C_s, C_s, M_s, M_s, a_s]

    for tag, (fa, fs) in [("or", (fits_or_a, fits_or_s)), ("pl", (fits_pi_a, fits_pi_s))]:
        xi_a, Q_a, conv_a, hits_a, n_a = fa
        xi_s, Q_s, conv_s, hits_s, n_s = fs
        a6 = _pack_asy6(xi_a); s6 = _pack_sym6(xi_s)
        names = ["mu","Cp","Cm","M","G","alpha"]
        for i,n in enumerate(names):
            out_row[f"{tag}_asy_{n}"] = a6[i]
            out_row[f"{tag}_sym_{n}"] = s6[i]
        out_row[f"{tag}_Q_asy"] = Q_a
        out_row[f"{tag}_Q_sym"] = Q_s
        cv_a = _conv_valid(conv_a, n_a); cv_s = _conv_valid(conv_s, n_s)
        out_row[f"conv_{tag}_asy"] = cv_a
        out_row[f"conv_{tag}_sym"] = cv_s
        out_row[f"nconv_{tag}_asy"] = conv_a
        out_row[f"nconv_{tag}_sym"] = conv_s
        out_row[f"hits_{tag}_asy"] = hits_a
        out_row[f"hits_{tag}_sym"] = hits_s

    # 选对率：DGP 对称 → 正确为 Q(sym) <= Q(asy); DGP 非对称 → Q(asy) < Q(sym)
    def _pick_ok(fa, fs, symmetric):
        qa, qs = fa[1], fs[1]
        if not np.isfinite(qa) or not np.isfinite(qs):
            return 0
        if symmetric:
            return int(qs <= qa)
        else:
            return int(qa < qs)

    out_row["pick_correct_or"] = _pick_ok(fits_or_a, fits_or_s, cfg["symmetric"])
    out_row["pick_correct_pl"] = _pick_ok(fits_pi_a, fits_pi_s, cfg["symmetric"])
    return out_row


# ---------- 调度 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="S=2 × DGP2 × T=3k/10k (5-20 分钟自检)")
    ap.add_argument("--S", type=int, default=10,
                    help="每个 (DGP, T) 的独立重复种子数 (默认 10)")
    ap.add_argument("--only-DGP", type=int, default=None, choices=[1,2,3])
    ap.add_argument("--only-T", type=int, default=None, choices=[3000, 10000, 25000])
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: S=1 DGP=2 T=3000 1 job")
    ap.add_argument("--start-seed", type=int, default=20260900)
    args = ap.parse_args()

    if args.smoke:
        S = 1; DGPs=[2]; Ts=[3000]; quick = True
    elif args.quick:
        S = 2; DGPs=[2]; Ts=[3000, 10000]; quick = True
    else:
        S = args.S
        DGPs = [args.only_DGP] if args.only_DGP else [1,2,3]
        Ts   = [args.only_T]  if args.only_T  else [3000, 10000, 25000]
        quick = False
    tasks = list(itertools.product(range(S), DGPs, Ts))
    tasks = [(args.start_seed + s0, d, t) for s0, d, t in tasks]

    OUT_ROWS_CSV = os.path.join(SIM_DIR, "sim_rows_raw.csv")
    OUT_DONE_LOG = os.path.join(SIM_DIR, "sim_done.log")
    # 本 batch 独立备份（避免并行多个 --only-DGP 覆盖总表）
    batch_tag_parts = []
    if args.only_DGP is not None: batch_tag_parts.append(f"DGP{args.only_DGP}")
    if args.only_T is not None: batch_tag_parts.append(f"T{args.only_T}")
    batch_tag_parts.append(f"S{S}")
    batch_tag_parts.append(f"seed{args.start_seed}")
    batch_csv = os.path.join(SIM_DIR, "sim_rows_" + "_".join(batch_tag_parts) + ".csv")
    # 如果总表已经存在且本 batch 是 split，不覆盖总表 — 改为 append。header 仅在首次/表不存在时写。
    total_header_written = os.path.exists(OUT_ROWS_CSV) and os.path.getsize(OUT_ROWS_CSV) > 50
    print(f"[SIM SCHED] jobs = {len(tasks)} (S={S}, DGPs={DGPs}, Ts={Ts}, quick={quick})", flush=True)
    print(f"[SIM SCHED] append_total={total_header_written}  batch_csv={os.path.basename(batch_csv)}", flush=True)

    rows = []
    n_written_total = 0  # 总表已写入行数 (防止重复)
    t_total = time.time()
    for i, (seed, dgp, T) in enumerate(tasks, 1):
        t1 = time.time()
        print(f"\n=== [{i}/{len(tasks)}] seed={seed} dgp={dgp} T={T} ===", flush=True)
        try:
            r = simulate_one(seed, dgp, T, quick=quick)
            rows.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  !!! FAILED seed={seed} dgp={dgp} T={T}: {e}", flush=True)
            continue
        dt = time.time()-t1
        print(f"  elapsed {dt:.1f}s, tpr={r['tpr']:.2f} fdr={r['fdr']:.2f} "
              f"MedSE={r['medse']:.4f} OR_a={r['or_asy_alpha']:.2f} PL_a={r['pl_asy_alpha']:.2f}",
              flush=True)
        # flush: 总表仅追加新增行; batch_csv 重写本 batch 至今所有 rows
        def _flush():
            nonlocal n_written_total, total_header_written
            df_all = pd.DataFrame(rows)
            # batch backup: 保存本 batch 完整 rows (不含其他 batch)
            df_all.to_csv(batch_csv, index=False, encoding="utf-8-sig")
            if len(df_all) > n_written_total:
                new_df = df_all.iloc[n_written_total:]
                new_df.to_csv(OUT_ROWS_CSV,
                              mode=("a" if total_header_written else "w"),
                              header=(not total_header_written),
                              index=False, encoding="utf-8-sig")
                total_header_written = True
                n_written_total = len(df_all)
        if (i % 3 == 0) or (i == len(tasks)):
            _flush()
            with open(OUT_DONE_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] done {i}/{len(tasks)} "
                        f"cum_elapsed={time.time()-t_total:.1f}s\n")

    if rows:
        _flush()
        # 双重保险：batch CSV 再写一次
        pd.DataFrame(rows).to_csv(batch_csv, index=False, encoding="utf-8-sig")
    print(f"[SIM DONE] total rows={len(rows)}, elapsed total={time.time()-t_total:.1f}s, "
          f"dir={SIM_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
