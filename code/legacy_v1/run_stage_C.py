# -*- coding: utf-8 -*-
"""
================================================================
 run_stage_C.py  — 阶段 C：非对称双边 tempered stable 噪声建模
                    与 ECF 最小距离估计 (论文 第 5 章)
================================================================
严格遵守 论文模型 8~11：
  (1) 残差 r_i = ΔX_i − b̂(X_{t_i}) Δt   （未标准化，Δt=1h）
  (2) Lévy 测度 ν(dz) = [C₊ z^{-1-α}e^{-Mz} 1_{z>0}
                         + C₋|z|^{-1-α}e^{-G|z|} 1_{z<0}] dz
      0<α<2; C₊,C₋,M,G>0
  (3) 截断 h(z)=z·1_{|z|≤1}
      ψ(u;ξ)=iμ u + ∫(e^{iuz}−1−iuh(z))ν(dz)
  (4) φ_{Δt}(u;ξ)=exp{Δt·ψ(u;ξ)}    Δt=1.0
  (5) ECF: J 频率 u_j=c_j/s_R，s_R = MAD(r)/0.6745 (稳健)
      矩条件：ℜ(cos) - ℜ(φ)、ℑ(sin)-ℑ(φ)   共 2J 维向量 m̂
      Q(ξ)=m̂ᵀ W m̂, W=I（主模型）
  (6) 优化：
        * 非对称 ξ=(μ, C₊, C₋, M, G, α) → log-space C₊,C₋,M,G, α logit
        * 对称基线：C₊=C₋=C, M=G（4 参）
        * 多初值 (LHS N_start=24) → 第一阶段 L-BFGS-B → 第二阶段 Nelder-Mead
        * 收敛判据：目标值相对差 < 1e-7；边界命中诊断
  (7) 对比：(a) 非对称 vs 对称基线 目标值
            (b) plug-in 残差 vs 伪-oracle（= 用训练段 X 做线性 drift 作“伪-oracle”
                真实数据下真 oracle 不可知 → 仅提供 placeholder 接口）

输入:  06_stage_outputs/stage_A_drift_train/stageA_X_full.csv
       06_stage_outputs/stage_A_drift_train/stageA_drift_train_artifacts.pkl
       06_stage_outputs/stage_B_sparse_drift/stageB_run_artifacts.pkl
输出:  06_stage_outputs/stage_C_ts_ecf/
         stageC_residual_stats.csv        r_i 统计（plug-in vs 伪-oracle）
         stageC_ecf_freq_grid.csv        u_j 网格
         stageC_estimates_asymmetric.csv  非对称 TS 参数 + SE(bootstrap placeholder)
         stageC_estimates_symmetric.csv   对称 TS 基线
         stageC_ecf_objective.csv         Q 值 / 命中 / 收敛 starts
         stageC_run_artifacts.pkl
         stageC_report.txt
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import integrate, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from thesis_pipeline_common import (
    DIR_OUT, ensure_dir, save_pickle, load_pickle,
)

A_XCSV = os.path.join(DIR_OUT, "stage_A_drift_train", "stageA_X_full.csv")
A_PKL  = os.path.join(DIR_OUT, "stage_A_drift_train", "stageA_drift_train_artifacts.pkl")
B_PKL  = os.path.join(DIR_OUT, "stage_B_sparse_drift", "stageB_run_artifacts.pkl")

DIR_C = ensure_dir(os.path.join(DIR_OUT, "stage_C_ts_ecf"))


# ================================================================
#  TS 特征指数  —— 矢量 Gauss-Laguerre (N_GL) + 0-奇异段 Gauss-Jacobi
#   精度 N_GL = 96 ~ 10^-5 水平的 ∫_0^∞ e^{-t} f(t) dt；
#   对 α∈(0,2) 所有稳定；单次 Q(u_full) 在 J≈37 下 <50ms。
# ================================================================
from numpy.polynomial import laguerre as _lag, legendre as _leg

_N_GL = 160    # Laguerre 节点（t∈[M,∞) 尾；160 阶 scipy roots 稳定无 NaN）
_N_GJ = 384    # (0,1] Legendre-Gauss 节点：积分有 t^{-1-α} 奇异 + 高频振荡，节点加倍
# scipy.special 的 Laguerre/Legendre 根比 numpy.polynomial 高阶稳定（避免权重 inf/NaN）
from scipy import special as _scsp
_t_gl, _w_gl = _scsp.roots_laguerre(_N_GL)
_s_gj, _w_gj = _scsp.roots_legendre(_N_GJ)
# [-1,1] → (0,1]: t=(s+1)/2, dt=ds/2
_t_01 = 0.5 * (_s_gj + 1.0)
_w_01 = 0.5 * _w_gj
assert np.isfinite(_t_gl).all() and np.isfinite(_w_gl).all() and (_w_gl > 0).all(), "Laguerre roots nonfinite!"
assert np.isfinite(_t_01).all() and np.isfinite(_w_01).all() and (_w_01 > 0).all(), "Legendre roots nonfinite!"


def _psi_full_u_asym(u_arr, mu, Cp, Cm, M, G, alpha):
    """对 u_arr (J,) 返回 ψ array (J, dtype complex128)。
    严格匹配 §5.3 Lévy-Khintchine：h(z)=z·1_{|z|≤1}.
    拆积分段 (0,1], [1,∞) × ± 双尾，对 α ∈ (0,2) 收敛。

    正尾 (0,1]:  变量 t = z ∈ (0,1]
        I₁⁺ = ∫₀¹ (cos(uz)-1) · Cp t^{-1-α} exp(-M t) dt
        J₁⁺ = ∫₀¹ (sin(ut) - u t) · Cp t^{-1-α} exp(-M t) dt   (因为 h(t)=t)
    正尾 [1,∞): 变量 t = z ∈ [1,∞)
        I₂⁺ = ∫₁^∞ (cos(uz)-1) · Cp t^{-1-α} exp(-M t) dt
        J₂⁺ = ∫₁^∞ (sin(ut))    · Cp t^{-1-α} exp(-M t) dt   (h(t)=0)
    负尾完全对称（G 替换 M, Cm 替换 Cp），z→t=|z| 后：
        cos(u·(-t)) - 1 = cos(u t)-1   (same)
        sin(u·(-t)) - u·(-t·1_{t≤1}) = -(sin(ut) - u·t·1_{t≤1})  (sign flip)
    """
    # ---- 提前准备：对 t ∈ (0,1] × [-∞,1] 及 t ∈ [1,∞) × [1,∞) 的公共幂项 ----
    # (0,1] 节点 t_01 (~N_GJ)
    t1 = _t_01                         # ∈ (0,1)
    w1 = _w_01
    ta1 = t1 ** (-1.0 - alpha)
    # ============================================================
    # [1,∞) 尾：标准变量代换 t = M z（正尾） / t = G z（负尾）
    # 使 exp(-M z)·exp(+t) = 1 抵消 GL 高斯权重 e^{-t}
    # 积分下限： z ∈ [1,∞)  ⇔ t ∈ [M,∞)
    # 令 t = s + M, s≥0；dt=ds，e^{-t}=e^{-M}e^{-s}，GL ∫_0^∞ e^{-s} f(s) ds.
    # ============================================================
    s = _t_gl
    ws = _w_gl
    tp = s + M                  # ∈ [M, ∞)
    tm = s + G
    zp = tp / M                 # z = t/M ≥ 1
    zm = tm / G
    # Cp·M^α 缩放因子
    Cp_Ma = Cp * (M ** alpha)
    Cm_Ga = Cm * (G ** alpha)
    # weight = C·x^α · e^{-M/G} · t^{-1-α} · w_s
    wCp2 = Cp_Ma * np.exp(-M) * (tp ** (-1.0 - alpha)) * ws
    wCm2 = Cm_Ga * np.exp(-G) * (tm ** (-1.0 - alpha)) * ws
    # exp(-M z) 在各段 (0,1] 用节点直接求值
    eMt1 = np.exp(-M * t1)
    eGt1 = np.exp(-G * t1)
    # 预计算：(0,1] 段的 weight × 幂 × 衰减
    wCp1 = (Cp * ta1 * eMt1) * w1          # 标量 Cp * 向量 * 向量
    wCm1 = (Cm * ta1 * eGt1) * w1
    # [1,∞) 段权重 wCp2/wCm2 在上方已按变量代换算好，变量名重用
    # u 列 × t/节点 行 → outer
    u = np.asarray(u_arr, dtype=np.float64)
    # (0,1] 实部
    u_t1 = np.outer(u, t1)               # (J, N1)
    c1 = np.cos(u_t1) - 1.0              # (J, N1)
    s1 = np.sin(u_t1) - u_t1             # sin(ut) - u·t (因 h(t)=t)
    I_Rp1 = c1 @ wCp1
    I_Rm1 = c1 @ wCm1
    I_Ip1 = s1 @ wCp1
    # 负尾虚部：sin(u·(-t)) - u·(-t) = -sin(ut) + u·t = -(sin(ut)-u·t) = -s1
    I_Im1 = (-s1) @ wCm1

    # [1,∞) 段
    u_zp2 = np.outer(u, zp)              # (J, N2)
    u_zm2 = np.outer(u, zm)
    c2p = np.cos(u_zp2) - 1.0
    s2p = np.sin(u_zp2)                   # h(z)=0 无矫正
    c2m = np.cos(u_zm2) - 1.0
    s2m = np.sin(u_zm2)
    I_Rp2 = c2p @ wCp2
    I_Rm2 = c2m @ wCm2
    I_Ip2 = s2p @ wCp2
    # 负尾 h=0 → sin(u·(-z)) = -sin(u z)
    I_Im2 = (-s2m) @ wCm2

    ReI = I_Rp1 + I_Rp2 + I_Rm1 + I_Rm2
    ImI = I_Ip1 + I_Ip2 + I_Im1 + I_Im2
    # iμu → 实部 0, 虚部 + μ·u
    return ReI + 1.0j * (mu * u + ImI)


def ts_psi_vectorized(u_arr, xi, dt: float = 1.0):
    """xi = [μ, C₊, C₋, M, G, α] for ASYMMETRIC. 调用时必须合法 (0<α<2 etc)."""
    mu, Cp, Cm, M, G, alpha = xi
    if not (0.0 < alpha < 2.0 and Cp > 0 and Cm > 0 and M > 0 and G > 0):
        return np.full(len(u_arr), complex(1e30, 1e30), dtype=np.complex128)
    return _psi_full_u_asym(np.asarray(u_arr, dtype=np.float64), mu, Cp, Cm, M, G, alpha)


def ts_phi_dt(u_arr, xi, dt: float = 1.0):
    """φ_{Δt}(u) = exp(Δt ψ(u)). 稳定溢出钳位。"""
    psi = ts_psi_vectorized(u_arr, xi, dt=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Re 过大时直接截断（exp 发散）: 把 Re(Δt·ψ) 夹到 [-800, 100]
        v = dt * psi
        re = np.real(v); im = np.imag(v)
        re = np.clip(re, -800.0, 100.0)
        expv = np.exp(re) * (np.cos(im) + 1j * np.sin(im))
    mask_bad = ~np.isfinite(expv)
    if mask_bad.any():
        expv = np.where(mask_bad, complex(1e30, 1e30), expv)
    return expv


# ================================================================
#  ECF 目标函数（W=I 主模型）
# ================================================================
@dataclass
class ECFData:
    r: np.ndarray             # (n_R,) 非标准化残差
    sR: float                 # MAD/0.6745 稳健尺度
    u: np.ndarray             # (J,) 频率 u_j = c_j/sR
    c_grid: np.ndarray        # (J,) 无量纲 c_j
    cosR: np.ndarray          # (J,) E[cos(u_j r)]
    sinR: np.ndarray          # (J,) E[sin(u_j r)]
    nR: int
    dt_hours: float = 1.0


def build_ecf_data(resid: np.ndarray,
                   c_grid: np.ndarray = np.concatenate([
                       np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
                       np.linspace(1.5, 12.0, 22),
                       np.linspace(14.0, 30.0, 9)
                   ])) -> ECFData:
    r = np.asarray(resid, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    nR = len(r)
    if nR < 100:
        raise RuntimeError("[build_ecf_data] too few valid residuals!")
    # sR = MAD / Φ^{-1}(0.75) ≈ MAD / 0.67449
    mad = float(np.median(np.abs(r - np.median(r))))
    sR = mad / 0.6744897501960817
    if sR <= 1e-12:
        sR = 1e-12
    u = c_grid / sR
    # 外积 → (nR, J)  → 余弦/正弦均值
    ur = np.outer(r, u)
    cosR = np.mean(np.cos(ur), axis=0)
    sinR = np.mean(np.sin(ur), axis=0)
    return ECFData(r=r, sR=sR, u=u, c_grid=c_grid,
                   cosR=cosR, sinR=sinR, nR=nR)


def compute_Q(xi: np.ndarray, dat: ECFData, symmetric: bool = False) -> float:
    """目标 Q = m^T I m / (2J) 归一化（方便跨 J 比较）。
    若 symmetric=True：xi 为 [μ, C, M, α] 4 参数，C₊=C₋=C, M=G。
    """
    if symmetric:
        mu, C, M, alpha = xi
        xi_full = np.array([mu, C, C, M, M, alpha])
    else:
        xi_full = np.asarray(xi, dtype=np.float64)
    phi = ts_phi_dt(dat.u, xi_full, dt=dat.dt_hours)
    rR = dat.cosR - phi.real
    rI = dat.sinR - phi.imag
    val = float(np.sum(rR * rR) + np.sum(rI * rI)) / (2.0 * len(dat.u))
    if not np.isfinite(val):
        return 1e30
    return val


# ================================================================
#  参数变换：对数空间 + α logit → 无约束优化，反变换回参数域
# ================================================================
_ALPHA_LB = 1e-3
_ALPHA_UB = 2.0 - 1e-3


def to_unconstrained(xi, symmetric=False):
    if symmetric:
        mu, C, M, alpha = xi
        return np.array([mu,
                         np.log(max(C, 1e-300)),
                         np.log(max(M, 1e-300)),
                         np.log((max(alpha, _ALPHA_LB) - _ALPHA_LB)
                                / (_ALPHA_UB - max(alpha, _ALPHA_LB)))])
    mu, Cp, Cm, M, G, alpha = xi
    return np.array([mu,
                     np.log(max(Cp, 1e-300)),
                     np.log(max(Cm, 1e-300)),
                     np.log(max(M, 1e-300)),
                     np.log(max(G, 1e-300)),
                     np.log((max(alpha, _ALPHA_LB) - _ALPHA_LB)
                            / (_ALPHA_UB - max(alpha, _ALPHA_LB)))])


def to_constrained(uvec, symmetric=False):
    alpha_raw = 1.0 / (1.0 + np.exp(-uvec[-1]))
    alpha = _ALPHA_LB + (_ALPHA_UB - _ALPHA_LB) * alpha_raw
    if symmetric:
        mu, lC, lM = uvec[0], uvec[1], uvec[2]
        return np.array([mu, np.exp(lC), np.exp(lM), alpha])
    mu, lCp, lCm, lM, lG = uvec[0], uvec[1], uvec[2], uvec[3], uvec[4]
    return np.array([mu, np.exp(lCp), np.exp(lCm), np.exp(lM), np.exp(lG), alpha])


# ================================================================
#  多初值 + 两阶段优化
# ================================================================
def starting_points_lhs(n_start: int, symmetric: bool, dat: ECFData,
                        seed: int = 20260902):
    """LHS over plausible parameter ranges (基于 MAD 尺度的经验先验)。

    启发式（不影响理论，仅优化器初始化）:
      μ ∈ [-sR/2, sR/2]
      C ∈ [0.1 * sR^{α0}, 10 * sR^{α0}]   （先扫 α 再按 C 归一）
      M,G ∈ [0.1 / sR, 5 / sR]
      α ∈ [0.1, 1.9]
    """
    rng = np.random.default_rng(seed)
    J = 4 if symmetric else 6
    starts = np.empty((n_start, J))
    for k in range(n_start):
        alpha = 0.1 + rng.random() * 1.8
        sR = dat.sR
        mu = sR * (rng.random() - 0.5)
        sp = sR ** alpha
        if symmetric:
            C  = sp * 10 ** (rng.uniform(-2, 1))
            M  = (1.0 / sR) * 10 ** (rng.uniform(-1, 1))
            starts[k] = [mu, C, M, alpha]
        else:
            Cp = sp * 10 ** (rng.uniform(-2, 1))
            Cm = sp * 10 ** (rng.uniform(-2, 1))
            M  = (1.0 / sR) * 10 ** (rng.uniform(-1, 1))
            G  = (1.0 / sR) * 10 ** (rng.uniform(-1, 1))
            starts[k] = [mu, Cp, Cm, M, G, alpha]
    return starts


def run_multi_start(dat: ECFData, symmetric: bool, n_start: int = 24,
                    seed: int = 20260902,
                    progress_cb=None):
    """progress_cb(idx, total, start_label, q_stage1, q_stage2, done_sofar_bestQ)
       None-safe: 如未传则不调用。"""
    starts = starting_points_lhs(n_start, symmetric, dat, seed)
    results = []
    best_q_sofar = float("inf")
    for idx, xi0 in enumerate(starts):
        # ---- Stage 1: L-BFGS-B on Q(λ-空间) 作大位移 ----
        def Q_of_u(u):
            try:
                xi = to_constrained(u, symmetric)
            except Exception:
                return 1e30
            return compute_Q(xi, dat, symmetric)

        u0 = to_unconstrained(xi0, symmetric)
        try:
            res1 = optimize.minimize(Q_of_u, u0, method="L-BFGS-B",
                                     options={"maxiter": 500, "ftol": 1e-9})
            uA = res1.x
            qA = float(res1.fun)
        except Exception:
            uA, qA = u0, Q_of_u(u0)
        # ---- Stage 2: Nelder-Mead 精炼（在真实参数空间直接跑 + bound）----
        def Q_of_x(xi):
            return compute_Q(xi, dat, symmetric)

        xiA = to_constrained(uA, symmetric)
        bounds = None
        if symmetric:
            bounds = [(None, None), (1e-8, None), (1e-8, None),
                      (_ALPHA_LB, _ALPHA_UB)]
        else:
            bounds = [(None, None), (1e-8, None), (1e-8, None),
                      (1e-8, None), (1e-8, None),
                      (_ALPHA_LB, _ALPHA_UB)]
        try:
            res2 = optimize.minimize(Q_of_x, xiA, method="Nelder-Mead",
                                     bounds=bounds,
                                     options={"maxiter": 2000, "fatol": 1e-9,
                                              "xatol": 1e-9})
            xiF = np.asarray(res2.x, dtype=np.float64)
            qF = float(res2.fun)
            conv = bool(res2.success)
        except Exception:
            xiF, qF, conv = xiA, qA, False
        # 边界命中诊断
        hits = []
        if symmetric:
            if xiF[1] <= 1e-6: hits.append("C_lower")
            if xiF[2] <= 1e-6: hits.append("M_lower")
            if xiF[3] <= _ALPHA_LB + 1e-3: hits.append("alpha_lower")
            if xiF[3] >= _ALPHA_UB - 1e-3: hits.append("alpha_upper")
        else:
            labs = ["Cp_lower", "Cm_lower", "M_lower", "G_lower"]
            for i, name in zip([1, 2, 3, 4], labs):
                if xiF[i] <= 1e-6: hits.append(name)
            if xiF[5] <= _ALPHA_LB + 1e-3: hits.append("alpha_lower")
            if xiF[5] >= _ALPHA_UB - 1e-3: hits.append("alpha_upper")
        results.append({
            "start_idx": idx, "xi0": xi0, "xi_final": xiF,
            "Q_start": compute_Q(xi0, dat, symmetric),
            "Q_final": qF, "converged": conv,
            "bound_hits": hits, "bounds_hit_count": len(hits),
        })
        if qF < best_q_sofar:
            best_q_sofar = qF
        if progress_cb is not None:
            try:
                progress_cb(idx, len(starts), symmetric, qA, qF, best_q_sofar)
            except Exception:
                pass
    # 排序：按 Q_final 升序
    results.sort(key=lambda r: r["Q_final"])
    return results


# ================================================================
#  残差构造: plug-in
# ================================================================
def construct_plugin_residuals() -> tuple[np.ndarray, ECFData | None, dict]:
    """返回 (TS 估计区间残差 r_i, ECFData, meta).

    r_i = ΔX_i − b̂_med(X_{t_i}) Δt,   Δt=1.0 （非标准化）。
    b̂_med = γ_med + Φ(X) β_med  (Stage B 后验中位数漂移 plug-in)。
    """
    if not (os.path.exists(A_XCSV) and os.path.exists(A_PKL) and os.path.exists(B_PKL)):
        raise RuntimeError("Need stage A + B outputs first!")
    dfX = pd.read_csv(A_XCSV, encoding="utf-8-sig")
    artA = load_pickle(A_PKL); artB = load_pickle(B_PKL)
    # TS 估计区间
    df_ts = dfX[dfX["split"] == "ts_est"].reset_index(drop=True)
    X = df_ts["X"].to_numpy(dtype=np.float64)
    DX = np.diff(X)       # ΔX[i] = X[i+1]-X[i], 长度 nR-1
    Xi_level = X[:-1]     # 对应于残差索引 i 的状态 X_{t_i}
    # 用 Stage B 的 b̂_med 估计
    beta_med = artB["beta_med"]
    gamma_med = artB["gamma_med"]
    colnames = artB["colnames"]
    # build design Φ at Xi_level: poly + hinge（结点信息在 artB）
    poly_pwrs = list(artB["basis_meta"]["poly_pwrs"])
    c_nodes = artB["basis_meta"]["c_nodes"]
    # 训练支持范围（阶段 B 箱代表态）
    x_support_min = float(artB["x_k_design_min"])
    x_support_max = float(artB["x_k_design_max"])
    # 裁剪超出训练支持的状态：漂移冻结后不再外推，
    # 超出范围的 X 用边界漂移代替，避免 hinge 基在极端 X 下爆炸。
    Xi_level_clipped = np.clip(Xi_level, x_support_min, x_support_max)
    n_clipped_low = int(np.sum(Xi_level < x_support_min))
    n_clipped_high = int(np.sum(Xi_level > x_support_max))
    n_clipped = n_clipped_low + n_clipped_high
    pct_clipped = 100.0 * n_clipped / max(len(Xi_level), 1)
    def Phi_of(x):
        blocks = []
        for p in poly_pwrs:
            blocks.append(np.asarray(x) ** p)
        for c in c_nodes:
            blocks.append(np.maximum(np.asarray(x, dtype=np.float64) - float(c), 0.0))
        return np.column_stack(blocks) if blocks else np.zeros((len(x), 0))
    Ph = Phi_of(Xi_level_clipped)
    if Ph.shape[1] == 0:
        b_hat_level = gamma_med
    else:
        b_hat_level = gamma_med + Ph @ beta_med
    resid = DX - b_hat_level * 1.0  # Δt = 1.0h (主模型非标准化)

    meta = {"n_ts_rows": len(df_ts), "n_resid": len(resid),
            "n_level_valid": int(np.isfinite(X).sum()),
            "resid_finite": int(np.isfinite(resid).sum()),
            "X_min": float(np.nanmin(X)), "X_max": float(np.nanmax(X)),
            "X_support_min": x_support_min, "X_support_max": x_support_max,
            "n_clipped_low": n_clipped_low, "n_clipped_high": n_clipped_high,
            "pct_clipped": pct_clipped,
            "resid_min": float(np.nanmin(resid)), "resid_max": float(np.nanmax(resid)),
            "resid_median": float(np.nanmedian(resid)),
            "resid_mad": float(np.median(np.abs(resid - np.nanmedian(resid)))),
            "drift_used": "median_posterior (gamma_med, beta_med) from Stage B, X clipped to train support",
            "poly_pwrs": poly_pwrs, "c_nodes": c_nodes.tolist(),
            "artB_path": B_PKL}

    dat = build_ecf_data(resid)
    # 额外构造伪 oracle 残差（= 加权 OLS 线性 drift 作基线）
    # 用于 Stage 报告 “oracle vs plug-in” 占位说明（真实数据无 oracle）
    from numpy.polynomial import polynomial as P_
    # b_OR1(x) = OLS 线性 + x² 在 TS-est 子样本的 drift level 拟合（仅作比较对象）
    # 注意：这不是真 oracle；只是敏感性代理。
    good = np.isfinite(DX) & np.isfinite(b_hat_level)
    Aor = np.column_stack([np.ones(good.sum()), Xi_level[good], Xi_level[good] ** 2])
    dor = DX[good]
    try:
        coef_or, *_ = np.linalg.lstsq(Aor, dor, rcond=None)
        dorhat = Aor @ coef_or
        r_oracle_pseudo = np.empty_like(DX) * np.nan
        r_oracle_pseudo[good] = dor - dorhat
        meta["pseudo_oracle_fit"] = "OLS(1, x, x²) on (Xi_level, DX) — used only for sensitivity comparison"
    except Exception:
        r_oracle_pseudo = np.empty_like(DX) * np.nan
    return resid, dat, meta, r_oracle_pseudo


# ================================================================
#  Bootstrap SE  —  (a) i.i.d. resample  (b) moving-block (MBB)
#  为每个 resample 重新构造 ECFData（基于 r*，重新算 sR* 和 freq 经验 cos/sin）
#  再用较少 n_start 从原最佳 xi 的邻域重新拟合；保存 ξ*，
#  最终 SE = std(ξ* 跨样本)，同时报告两种方案 SE（保守取 max）。
# ================================================================
def bootstrap_ecf_se(dat_orig: ECFData, resid_orig: np.ndarray,
                     xi_hat_asy: np.ndarray | None,
                     xi_hat_sym: np.ndarray | None,
                     B: int = 200,
                     n_start: int = 4,
                     block_len: int = 24,   # MBB 块长=24h (日周期)
                     seed: int = 20260903,
                     L=None) -> dict:
    """Bootstrap SE，分别用 i.i.d. 重采样与移动块（MBB）重采样估计标准误。

    返回 se_asy / se_sym 为两套结果：(se_iid, se_mbb)，不再混合。
    """
    rng = np.random.default_rng(seed)
    nR = len(resid_orig)
    # 每个 resample 的起始点 = 以 xi_hat 为中心加 log-尺度 ±10% 扰动
    def mk_starts(xi_hat, symmetric, n_s):
        starts = np.empty((n_s, len(xi_hat)), dtype=np.float64)
        for k in range(n_s):
            s = xi_hat.copy()
            # μ 扰动高斯
            s[0] = xi_hat[0] + rng.normal(0, max(0.5, abs(xi_hat[0])*0.1))
            # C±/M/G 乘性
            idx_pos = list(range(1, len(xi_hat)-1))
            for j in idx_pos:
                s[j] = max(1e-8, xi_hat[j] * (10 ** rng.uniform(-0.3, 0.3)))
            # α 在约束邻域
            s[-1] = float(np.clip(
                xi_hat[-1] + rng.normal(0, 0.05),
                _ALPHA_LB + 1e-2, _ALPHA_UB - 1e-2))
            starts[k] = s
        # 再加一套 LHS 宽范围兜底
        lhs_wide = starting_points_lhs(n_s, symmetric, dat_orig,
                                       seed=int(seed + abs(hash((k, symmetric, B))) % 2**30))
        return np.vstack([starts, lhs_wide[:max(1, n_s//2)]])

    # 分开两套 draws：iid (偶数 b) 和 mbb (奇数 b)
    draws_asy_iid = []
    draws_asy_mbb = []
    draws_sym_iid = []
    draws_sym_mbb = []
    q_asy_iid = []
    q_asy_mbb = []
    q_sym_iid = []
    q_sym_mbb = []
    # MBB 块数
    n_blocks = int(np.ceil(nR / block_len))
    for b in range(B):
        # --- (a) iid resample for 偶数索引，(b) MBB 为奇数索引  ---
        if b % 2 == 0:
            idx = rng.integers(0, nR, size=nR)
            label = "iid"
        else:
            bidx = rng.integers(0, nR - block_len + 1, size=n_blocks)
            idx = (bidx[:, None] + np.arange(block_len)[None, :]).ravel()[:nR]
            label = "mbb"
        r_star = resid_orig[idx]
        try:
            dat_star = build_ecf_data(r_star)
        except Exception:
            continue

        # ASY
        if xi_hat_asy is not None:
            sA = mk_starts(xi_hat_asy, False, n_start)
            res = []
            for xi0 in sA:
                try:
                    u0 = to_unconstrained(xi0, False)
                    def Q_of_u(u):
                        try:
                            xi = to_constrained(u, False)
                        except Exception:
                            return 1e30
                        return compute_Q(xi, dat_star, False)
                    r1 = optimize.minimize(Q_of_u, u0, method="L-BFGS-B",
                                           options={"maxiter": 300, "ftol": 1e-8})
                    xiA = to_constrained(np.asarray(r1.x, dtype=np.float64), False)
                    def Q_of_x(xi):
                        return compute_Q(xi, dat_star, False)
                    r2 = optimize.minimize(Q_of_x, xiA, method="Nelder-Mead",
                                           bounds=[(None,None),(1e-8,None),(1e-8,None),
                                                   (1e-8,None),(1e-8,None),
                                                   (_ALPHA_LB,_ALPHA_UB)],
                                           options={"maxiter": 1200, "fatol": 1e-8,
                                                    "xatol": 1e-8})
                    res.append((float(r2.fun), np.asarray(r2.x, dtype=np.float64)))
                except Exception:
                    pass
            if res:
                res.sort(key=lambda t: t[0])
                qA, xiA_best = res[0]
                if label == "iid":
                    draws_asy_iid.append(xiA_best); q_asy_iid.append(qA)
                else:
                    draws_asy_mbb.append(xiA_best); q_asy_mbb.append(qA)

        # SYM
        if xi_hat_sym is not None:
            sS = mk_starts(xi_hat_sym, True, n_start)
            res = []
            for xi0 in sS:
                try:
                    u0 = to_unconstrained(xi0, True)
                    def Q_of_u(u):
                        try:
                            xi = to_constrained(u, True)
                        except Exception:
                            return 1e30
                        return compute_Q(xi, dat_star, True)
                    r1 = optimize.minimize(Q_of_u, u0, method="L-BFGS-B",
                                           options={"maxiter": 300, "ftol": 1e-8})
                    xiS = to_constrained(np.asarray(r1.x, dtype=np.float64), True)
                    def Q_of_x(xi):
                        return compute_Q(xi, dat_star, True)
                    r2 = optimize.minimize(Q_of_x, xiS, method="Nelder-Mead",
                                           bounds=[(None,None),(1e-8,None),(1e-8,None),
                                                   (_ALPHA_LB,_ALPHA_UB)],
                                           options={"maxiter": 1200, "fatol": 1e-8,
                                                    "xatol": 1e-8})
                    res.append((float(r2.fun), np.asarray(r2.x, dtype=np.float64)))
                except Exception:
                    pass
            if res:
                res.sort(key=lambda t: t[0])
                qS, xiS_best = res[0]
                if label == "iid":
                    draws_sym_iid.append(xiS_best); q_sym_iid.append(qS)
                else:
                    draws_sym_mbb.append(xiS_best); q_sym_mbb.append(qS)

        if (L is not None) and ((b + 1) % 20 == 0 or b == 0):
            L(f"    [bootstrap {label}] {b+1:>4d}/{B}:  "
              f"asy valid={len(draws_asy_iid)+len(draws_asy_mbb):>4d},  "
              f"sym valid={len(draws_sym_iid)+len(draws_sym_mbb):>4d}")

    # 汇总 SE = sd(有限项样本) —— iid 与 mbb 分开
    def colstd_list(lst):
        if len(lst) < 10:
            return None
        m = np.array(lst, dtype=np.float64)
        return np.std(m, axis=0, ddof=1)

    se_asy_iid = colstd_list(draws_asy_iid)
    se_asy_mbb = colstd_list(draws_asy_mbb)
    se_sym_iid = colstd_list(draws_sym_iid)   # 4 维
    se_sym_mbb = colstd_list(draws_sym_mbb)

    # 取保守 SE = max(iid, mbb) 作为主报告值
    def _safe_max(a, b):
        if a is None and b is None:
            return None
        if a is None: return b
        if b is None: return a
        return np.maximum(a, b)

    se_asy = _safe_max(se_asy_iid, se_asy_mbb)
    se_sym_full = _safe_max(se_sym_iid, se_sym_mbb)

    def _rep_sym(se4):
        if se4 is None:
            return None
        return np.array([se4[0], se4[1], se4[1], se4[2], se4[2], se4[3]])

    out = {
        "B": B, "block_len": block_len, "n_start_per_resample": n_start,
        "se_asy_iid": se_asy_iid,
        "se_asy_mbb": se_asy_mbb,
        "se_sym_iid": se_sym_iid,
        "se_sym_mbb": se_sym_mbb,
        "se_asy": se_asy,                          # max(iid, mbb) — 保守
        "se_sym_full": se_sym_full,
        "se_sym_rep": _rep_sym(se_sym_full),
        "n_valid_asy_iid": len(draws_asy_iid),
        "n_valid_asy_mbb": len(draws_asy_mbb),
        "n_valid_sym_iid": len(draws_sym_iid),
        "n_valid_sym_mbb": len(draws_sym_mbb),
        "q_asy_iid": q_asy_iid,
        "q_asy_mbb": q_asy_mbb,
        "q_sym_iid": q_sym_iid,
        "q_sym_mbb": q_sym_mbb,
        "draws_asy_iid": draws_asy_iid,
        "draws_asy_mbb": draws_asy_mbb,
        "draws_sym_iid": draws_sym_iid,
        "draws_sym_mbb": draws_sym_mbb,
    }
    return out


# ================================================================
#  主入口
# ================================================================
def main():
    t0 = time.time()
    lines = []
    def L(x):
        msg = str(x)
        lines.append(msg)
        # 后台运行时 stdout 可能被管道完全缓冲；强制 flush 避免长时间看不见进度
        print(msg, flush=True)
    L("="*72); L("[Stage C] START  " + time.strftime("%Y-%m-%d %H:%M:%S"))

    # (1) 残差 + ECF 数据
    resid, dat, metaR, r_or_pseudo = construct_plugin_residuals()
    L(f"  TS-est rows (level) = {metaR['n_ts_rows']}, resid (Δ) = {metaR['n_resid']}")
    L(f"  X range (ts-est)   = [{metaR['X_min']:.3f}, {metaR['X_max']:.3f}]")
    L(f"  X train support    = [{metaR['X_support_min']:.3f}, {metaR['X_support_max']:.3f}]")
    L(f"  X clipped (out of support) = {metaR['n_clipped_low']+metaR['n_clipped_high']} "
      f"({metaR['pct_clipped']:.2f}%): low={metaR['n_clipped_low']}, high={metaR['n_clipped_high']}")
    L(f"  resid summary       = median {metaR['resid_median']:.3f}, MAD {metaR['resid_mad']:.3f}")
    L(f"  robust sR = MAD/0.6745 = {dat.sR:.3f}, J = {len(dat.u)} frequencies")
    L(f"  u_grid min/max = {dat.u.min():.4f} / {dat.u.max():.4f}")
    pd.DataFrame({
        "metric": ["n_ts_rows_level", "n_resid_delta", "resid_finite",
                   "X_min", "X_max", "resid_min", "resid_max",
                   "resid_median", "resid_mad", "robust_sR",
                   "n_cos_sin_freq"],
        "plugin_value": [metaR["n_ts_rows"], metaR["n_resid"], metaR["resid_finite"],
                         metaR["X_min"], metaR["X_max"], metaR["resid_min"], metaR["resid_max"],
                         metaR["resid_median"], metaR["resid_mad"], dat.sR, len(dat.u)],
        "pseudo_oracle_value": [metaR["n_ts_rows"], metaR["n_resid"],
                                int(np.isfinite(r_or_pseudo).sum()),
                                metaR["X_min"], metaR["X_max"],
                                float(np.nanmin(r_or_pseudo)),
                                float(np.nanmax(r_or_pseudo)),
                                float(np.nanmedian(r_or_pseudo)),
                                float(np.median(np.abs(r_or_pseudo -
                                             np.nanmedian(r_or_pseudo)))),
                                "— (computed in-original scale)", "— (inherited)"],
    }).to_csv(os.path.join(DIR_C, "stageC_residual_stats.csv"),
              index=False, encoding="utf-8-sig")

    pd.DataFrame({"j": np.arange(len(dat.u)), "c_j": dat.c_grid,
                  "u_j": dat.u, "cos_E": dat.cosR, "sin_E": dat.sinR}).to_csv(
        os.path.join(DIR_C, "stageC_ecf_freq_grid.csv"),
        index=False, encoding="utf-8-sig")

    def cb(idx, total, lbl, q1, q2, best):
        tag = "ASY" if lbl is False else ("SYM" if lbl is True else str(lbl))
        L(f"    [{tag}] start {idx+1:02d}/{total:02d}:  LBFGSB_Q={q1:.6f} NM_Q={q2:.6f}  best_sofar={best:.6f}")

    # (2) 非对称 TS 估计（慢：24 起始 × 积分 × 每起始 2 stage 迭代 → 后台允许几十分钟）
    L("\n  ---- Fitting ASYMMETRIC TS (C₊≠C₋, M≠G) ----")
    res_asy = run_multi_start(dat, symmetric=False, n_start=24, seed=90201, progress_cb=cb)
    bestA = res_asy[0]
    L(f"    Best Q = {bestA['Q_final']:.8f} (started at Q0={bestA['Q_start']:.8f})")
    L(f"    Best ξ = {dict(zip(['μ','C₊','C₋','M','G','α'], bestA['xi_final']))}")
    L(f"    Converged={bestA['converged']}, boundary hits={bestA['bound_hits']}")
    converged_startsA = sum(1 for r in res_asy if r["converged"])
    L(f"    # converged starts / {len(res_asy)} = {converged_startsA}")

    # (3) 对称 TS 基线
    L("\n  ---- Fitting SYMMETRIC TS baseline (C₊=C₋, M=G) ----")
    res_sym = run_multi_start(dat, symmetric=True, n_start=24, seed=90202, progress_cb=cb)
    bestS = res_sym[0]
    L(f"    Best Q = {bestS['Q_final']:.8f} (ΔQ = {bestS['Q_final']-bestA['Q_final']:+.8f})")
    L(f"    Best ξ = {dict(zip(['μ','C','M','α'], bestS['xi_final']))}")
    L(f"    Converged={bestS['converged']}, boundary hits={bestS['bound_hits']}")
    converged_startsS = sum(1 for r in res_sym if r["converged"])
    L(f"    # converged starts / {len(res_sym)} = {converged_startsS}")

    # (4) Bootstrap SE：iid + MBB 混合 B=200
    L("\n  ---- Bootstrap SE (B=200, iid + moving-block L=24h) ----")
    t_bs = time.time()
    bs_out = bootstrap_ecf_se(dat, resid,
                              xi_hat_asy=bestA["xi_final"],
                              xi_hat_sym=bestS["xi_final"],
                              B=200, n_start=4, block_len=24,
                              seed=20260903, L=L)
    L(f"    Bootstrap elapsed = {time.time()-t_bs:.1f}s")
    L(f"    valid draws: asy iid={bs_out['n_valid_asy_iid']}/100, mbb={bs_out['n_valid_asy_mbb']}/100  |  "
      f"sym iid={bs_out['n_valid_sym_iid']}/100, mbb={bs_out['n_valid_sym_mbb']}/100")
    L(f"    SE asy iid (μ,C₊,C₋,M,G,α) = {np.round(bs_out['se_asy_iid'],5).tolist() if bs_out['se_asy_iid'] is not None else 'N/A'}")
    L(f"    SE asy mbb (μ,C₊,C₋,M,G,α) = {np.round(bs_out['se_asy_mbb'],5).tolist() if bs_out['se_asy_mbb'] is not None else 'N/A'}")
    L(f"    SE asy max(iid,mbb)        = {np.round(bs_out['se_asy'],5).tolist() if bs_out['se_asy'] is not None else 'N/A'}")
    L(f"    SE sym iid (μ,C,M,α)       = {np.round(bs_out['se_sym_iid'],5).tolist() if bs_out['se_sym_iid'] is not None else 'N/A'}")
    L(f"    SE sym mbb (μ,C,M,α)       = {np.round(bs_out['se_sym_mbb'],5).tolist() if bs_out['se_sym_mbb'] is not None else 'N/A'}")

    # (5) 报告参数表（含 Bootstrap SE 实填）
    pd.DataFrame({
        "parameter": ["μ", "C₊", "C₋", "M", "G", "α"],
        "estimate_asym": bestA["xi_final"],
        "SE_asym_bootstrap": bs_out["se_asy"],
        "estimate_sym": [bestS["xi_final"][0],
                         bestS["xi_final"][1], bestS["xi_final"][1],
                         bestS["xi_final"][2], bestS["xi_final"][2],
                         bestS["xi_final"][3]],
        "SE_sym_bootstrap": bs_out["se_sym_rep"],
    }).to_csv(os.path.join(DIR_C, "stageC_estimates_compare.csv"),
              index=False, encoding="utf-8-sig")
    # 额外保存完整 bootstrap draws 摘要（供 chap7 dump 读取）
    q_asy_all = np.array(bs_out["q_asy_iid"] + bs_out["q_asy_mbb"], dtype=np.float64)
    q_sym_all = np.array(bs_out["q_sym_iid"] + bs_out["q_sym_mbb"], dtype=np.float64)
    q_asy_all = q_asy_all[np.isfinite(q_asy_all)]
    q_sym_all = q_sym_all[np.isfinite(q_sym_all)]
    n_valid_asy = bs_out["n_valid_asy_iid"] + bs_out["n_valid_asy_mbb"]
    n_valid_sym = bs_out["n_valid_sym_iid"] + bs_out["n_valid_sym_mbb"]
    pd.DataFrame({
        "metric": ["B_total", "block_len", "n_start_per_resample",
                   "n_valid_asy", "n_valid_sym",
                   "n_valid_asy_iid", "n_valid_asy_mbb",
                   "n_valid_sym_iid", "n_valid_sym_mbb",
                   "Q_asy_med", "Q_asy_q025", "Q_asy_q975",
                   "Q_sym_med", "Q_sym_q025", "Q_sym_q975"],
        "value": [bs_out["B"], bs_out["block_len"], bs_out["n_start_per_resample"],
                  n_valid_asy, n_valid_sym,
                  bs_out["n_valid_asy_iid"], bs_out["n_valid_asy_mbb"],
                  bs_out["n_valid_sym_iid"], bs_out["n_valid_sym_mbb"],
                  float(np.median(q_asy_all)) if len(q_asy_all) else np.nan,
                  float(np.quantile(q_asy_all, 0.025)) if len(q_asy_all) >= 10 else np.nan,
                  float(np.quantile(q_asy_all, 0.975)) if len(q_asy_all) >= 10 else np.nan,
                  float(np.median(q_sym_all)) if len(q_sym_all) else np.nan,
                  float(np.quantile(q_sym_all, 0.025)) if len(q_sym_all) >= 10 else np.nan,
                  float(np.quantile(q_sym_all, 0.975)) if len(q_sym_all) >= 10 else np.nan],
    }).to_csv(os.path.join(DIR_C, "stageC_bootstrap_se_summary.csv"),
              index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "model": ["Asymmetric TS", "Symmetric TS baseline"],
        "Q_min": [bestA["Q_final"], bestS["Q_final"]],
        "Q_start_worst": [max(r["Q_start"] for r in res_asy),
                          max(r["Q_start"] for r in res_sym)],
        "n_start": [len(res_asy), len(res_sym)],
        "converged_starts": [converged_startsA, converged_startsS],
        "best_bound_hit?": ["/".join(bestA["bound_hits"]) or "none",
                            "/".join(bestS["bound_hits"]) or "none"],
    }).to_csv(os.path.join(DIR_C, "stageC_ecf_objective.csv"),
              index=False, encoding="utf-8-sig")

    # (6) 保存 artifacts 用于后续报告表
    artC = {
        "resid_plugin": resid,
        "resid_pseudo_oracle": r_or_pseudo,
        "residual_meta": metaR,
        "ecf_grid_c": dat.c_grid, "ecf_grid_u": dat.u,
        "ecf_cos": dat.cosR, "ecf_sin": dat.sinR,
        "best_asymmetric": bestA, "all_starts_asymmetric": res_asy,
        "best_symmetric": bestS,  "all_starts_symmetric": res_sym,
        "bootstrap": bs_out,
    }
    save_pickle(artC, os.path.join(DIR_C, "stageC_run_artifacts.pkl"))

    L(f"\n  [Stage C] DONE in {time.time()-t0:.1f}s. dir = {DIR_C}")
    with open(os.path.join(DIR_C, "stageC_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.exit(main())
