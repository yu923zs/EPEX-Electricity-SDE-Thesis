# -*- coding: utf-8 -*-
"""
================================================================
 run_stage_B.py — 阶段 B：可靠性加权 Student-t 型 spike-and-slab
                  drift 稀疏识别 (论文 第 4 章)
================================================================
输入: 06_stage_outputs/stage_A_drift_train/stageA_drift_train_artifacts.pkl
输出: 06_stage_outputs/stage_B_sparse_drift/
  - stageB_candidate_basis.csv   基元清单（含是否来自训练分位结点）
  - stageB_pip_coef_summary.csv PIP + 后验 med/95% 区间（按 PIP 降序）
  - stageB_posterior_drift_band.csv
       x_grid, b_med, b_q025, b_q975, b_mean
  - stageB_diagnostics.csv       R-hat / ESS / 对数似然
  - stageB_run_artifacts.pkl     给 Stage C 用于计算 plug-in 残差
  - stageB_report.txt            运行日志 + 自检

超参（严格匹配 §4）:
  - 截距 γ 始终保留、不参与变量选择
  - 候选基 φ_j ∈ {x, x², x³, (x-c₁)_+, …, (x-cₘ)_+}
       * 结点 c_j: 训练区间 X 的 0.10…0.90 等分位数 → 9 个
       * 总 p = 3 + 9 = 12 个非截距候选 → 选 PIP 稳定可解释
  - IG(shape,rate) 参数化 p(v|a,b) ∝ v^{-(a+1)} exp(-b/v)
       a_λ=b_λ=2.0（Student-t 边际自由度 2a_λ=4）
       a_σ=b_σ=2.0,  a_τ=b_τ=1.0（弱信息）
  - π₀ = 0.5（先验不预置稀疏方向，与论文 §4.3 一致）
  - σ_γ² = 100
  - τ² slab 先验 IG(a_τ,b_τ)
  - Gibbs: 3 链，burn=10k, post=20k → 60k draws
  - 初始化：3 种 disperse：(a) θ 全 0, (b) θ 全 1, (c) 基于加权 OLS top-50% t 值进
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from thesis_pipeline_common import (
    BASE, DIR_OUT, ensure_dir, save_pickle, load_pickle,
)

ARTIFACTS_A = os.path.join(DIR_OUT, "stage_A_drift_train",
                           "stageA_drift_train_artifacts.pkl")
STAGE_B_DIR = ensure_dir(os.path.join(DIR_OUT, "stage_B_sparse_drift"))

# ==================== 贝叶斯超参 ====================
PRIOR = dict(
    # Candidate basis nodes
    hinge_c_quantiles=np.arange(0.10, 0.91, 0.10),   # 9 个
    include_poly=(1, 2, 3),                           # x, x², x³
    # Priors
    a_lam=2.0, b_lam=2.0,
    a_sig=2.0, b_sig=2.0,
    a_tau=1.0, b_tau=1.0,
    pi0=0.5,
    sigma_gamma2=100.0,
    # MCMC
    n_chains=3,
    n_burn=10_000,
    n_post=20_000,
    thin=1,
    seed=20250902,
)


# ================================================================
#  候选基
# ================================================================
def build_phi(x_train_for_design: np.ndarray,
              x_eval: np.ndarray,
              cfg: dict) -> tuple[np.ndarray, list[str], dict]:
    """在 x_train_for_design（训练 X 的分位）上定结点，返回 Φ on x_eval.

    返回 Φ = 非截距候选（p 列）；截距在 η_A 里单独补。
    """
    qtiles = cfg["hinge_c_quantiles"]
    c_nodes = np.quantile(x_train_for_design, qtiles)
    c_nodes = np.unique(c_nodes)
    poly_pwrs = list(cfg["include_poly"])
    colnames = []
    blocks = []
    for p in poly_pwrs:
        colnames.append(f"x^{p}")
        blocks.append(x_eval ** p)
    for j, c in enumerate(c_nodes):
        colnames.append(f"(x-{c:.3f})_+")
        blocks.append(np.maximum(x_eval - float(c), 0.0))
    Phi = np.column_stack(blocks)
    meta = {"c_nodes": c_nodes, "poly_pwrs": poly_pwrs,
            "n_candidates": Phi.shape[1], "colnames": colnames}
    return Phi, colnames, meta


# ================================================================
#  Gibbs Sampler (严格匹配 模型 4~7)
# ================================================================
def gibbs_ss(y: np.ndarray, Phi: np.ndarray, w: np.ndarray,
             prior: dict, n_burn: int, n_post: int, seed: int,
             init_theta: np.ndarray | None = None,
             likelihood: str = "student_t"):
    """严格点质量 spike-and-slab + IG 混合 Student-t 工作似然。

    y (K,), Φ (K,p), w (K,)  —— 可靠性权重 已 winsor+归一。
    likelihood: "student_t" (默认, IG 混合厚尾) 或 "normal" (lam=1, 高斯似然)。
    返回 dict(samples, trace_loglike, final_state)
    """
    K, p = Phi.shape
    a_lam, b_lam = prior["a_lam"], prior["b_lam"]
    a_sig, b_sig = prior["a_sig"], prior["b_sig"]
    a_tau, b_tau = prior["a_tau"], prior["b_tau"]
    pi0 = prior["pi0"]
    sg2 = prior["sigma_gamma2"]
    is_normal = (likelihood == "normal")

    rng = np.random.default_rng(seed)
    # ---- 初始化 ----
    theta = init_theta.astype(bool) if init_theta is not None \
        else (rng.random(p) < pi0)
    beta = np.zeros(p)
    # 初始活跃集合 → 加权 OLS 作初值
    A_cur = np.column_stack([np.ones(K), Phi[:, theta]])
    Wsr = np.sqrt(w)
    yw = y * Wsr; Aw = A_cur * Wsr[:, None]
    eta0, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    gamma = float(eta0[0])
    beta[theta] = eta0[1:]
    lam = np.ones(K)
    # σ², τ² 用先验均值
    sigma2 = b_sig / max(a_sig - 1, 0.5) if a_sig > 1 else 1.0
    tau2   = b_tau / max(a_tau - 1, 0.5) if a_tau > 1 else 1.0
    # 记录
    total = n_burn + n_post
    thin = prior.get("thin", 1)
    store_every = thin
    n_store = n_post // thin
    trace = {
        "gamma": np.empty(n_store),
        "beta": np.empty((n_store, p)),
        "theta": np.empty((n_store, p), dtype=np.int8),
        "sigma2": np.empty(n_store),
        "tau2": np.empty(n_store),
        "loglik": np.empty(n_store),
    }
    store_idx = 0
    # 预计算：ϕᵢᵀ D⁻¹ ϕⱼ 的对角用于 θ 更新需要 a_j
    sqrt_w = np.sqrt(w)

    def _sample_lam_sig_tau(r, theta_cur, beta_cur):
        nonlocal lam, sigma2, tau2
        if not is_normal:
            # (1) λ_k | · ~ IG(a_lam + 1/2, b_lam + w_k r_k^2 / (2 σ²))
            #     IG(a, b) rate 参数化:  X ~ 1/Gamma(a, scale=1/b)
            resid_term = (w * r * r) / sigma2
            # winsor 极端项：防止 r_k 异常大 / w_k 异常大导致 λ_k gamma 参数爆炸
            term_hi = np.quantile(resid_term, 0.99)
            if np.isfinite(term_hi) and term_hi > 0:
                resid_term = np.clip(resid_term, 0.0, 1e4 + term_hi)
            b_post = b_lam + 0.5 * resid_term
            a_post = np.full(K, a_lam + 0.5, dtype=np.float64)
            # rate  = b_post → gamma shape=a_post, scale=1/b_post, 再 reciprocal → IG
            b_safe = np.clip(b_post, 1e-6, 1e10)
            scale = 1.0 / b_safe
            # 防 nan: gamma 分布如果 scale 0 返回 0 → 1/0 爆炸；这里先 clamp
            g = rng.gamma(shape=a_post, scale=scale, size=K)
            g = np.where(np.isfinite(g) & (g > 0), g, 1e-6)
            lam = 1.0 / g
            lam = np.clip(lam, 1e-5, 1e5)  # IG 先验的 λ_k 不应极端（Student-t 自由度 4 时 ≈ 1 附近）
        # (2) σ² | · ~ IG(a_sig + K/2, b_sig + Σ w_k r_k^2 / (2 λ_k))
        ratio = w * r * r / lam
        # 同样 clamp ratio：除以 lam（近 0 时）的 inf
        ratio = np.where(np.isfinite(ratio), ratio, 0.0)
        if np.any(np.isinf(ratio)) or np.any(np.isnan(ratio)):
            ratio = np.clip(ratio, -1e12, 1e12)
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
        sum_term = 0.5 * float(np.sum(ratio))
        b_sig_safe = max(b_sig, 1e-6)
        a_sig_post_shape = a_sig + 0.5 * K
        scale_sig = 1.0 / max(b_sig_safe + sum_term, 1e-6)
        sigma2 = float(1.0 / max(rng.gamma(shape=a_sig_post_shape, scale=scale_sig), 1e-12))
        sigma2 = float(np.clip(sigma2, 1e-8, 1e6))
        # (3) τ² | · ~ IG(a_tau + m/2, b_tau + ½ Σ_{j∈A} β_j²)
        m = int(theta_cur.sum())
        if m > 0:
            ssq = float(np.sum(beta_cur[theta_cur] ** 2))
        else:
            ssq = 0.0
        tau2 = 1.0 / rng.gamma(shape=a_tau + 0.5 * m,
                               scale=1.0 / (b_tau + 0.5 * ssq))

    def _sample_eta_A(theta_cur):
        nonlocal gamma, beta
        A = theta_cur.sum()
        ZA = np.column_stack([np.ones(K), Phi[:, theta_cur]])
        # D = diag(σ² λ_k / w_k) → D^{-1} = diag(w_k / (σ² λ_k))
        Dinv_diag = w / (sigma2 * lam + 1e-300)
        ZA_w = ZA * Dinv_diag[:, None]
        # Prior V_0A
        V0A_inv_diag = np.concatenate([[1.0 / sg2], np.full(int(A), 1.0 / tau2)])
        V_inv = np.diag(V0A_inv_diag) + ZA.T @ ZA_w
        rhs = ZA.T @ (Dinv_diag * y)
        try:
            V = np.linalg.inv(V_inv)
        except np.linalg.LinAlgError:
            # 岭保护（避免非活跃切换瞬间 A=0 不触发，ZA=[1] 通常可逆）
            V = np.linalg.inv(V_inv + 1e-8 * np.eye(V_inv.shape[0]))
        # 用特征分解做对称 + 半正定化，保证 Cholesky 不炸
        Vs = 0.5 * (V + V.T)
        try:
            L = np.linalg.cholesky(Vs + 1e-10 * np.eye(Vs.shape[0]))
        except np.linalg.LinAlgError:
            # Robust 回退：特征值夹稳到 jitter
            w_eig, v_eig = np.linalg.eigh(Vs)
            jitter = max(1e-8, 1e-8 * float(np.max(np.abs(w_eig))))
            w_safe = np.where(w_eig > jitter, w_eig, jitter)
            V_safe = (v_eig * w_safe) @ v_eig.T
            L = np.linalg.cholesky(V_safe + 1e-12 * np.eye(V_safe.shape[0]))
        muA = Vs @ rhs
        eta_samp = muA + L @ rng.standard_normal(Vs.shape[0])
        # 数值夹稳：保证 η 样本不会出现 1e60 极端（不改变后验主体，仅防级联 inf）
        eta_samp = np.clip(eta_samp, -1e6, 1e6)
        gamma = float(eta_samp[0])
        if A > 0:
            idx = np.where(theta_cur)[0]
            for k, j in enumerate(idx):
                beta[j] = float(eta_samp[1 + k])

    def _sample_theta_and_beta():
        nonlocal theta, beta
        # 对每个变量 j 按顺序 Gibbs：积分 β_j → 贝叶斯因子比 O_j
        # D^{-1} 对角线（每次都基于当前 λ, σ², w）
        Dinv_diag = w / (sigma2 * lam + 1e-300)
        r = y - gamma - Phi @ beta   # r_{-j} 计算基础
        for j in range(p):
            phij = Phi[:, j]
            # 把 β_j 的影响先从 r 中“移除”(即 r = r_{-j})
            r_adj = r + phij * beta[j]
            a_j = float(phij @ (Dinv_diag * phij))
            c_j = float(phij @ (Dinv_diag * r_adj))
            # O_j = π/(1-π) * (1+a_j τ²)^{-1/2} * exp(τ² c_j² / (2(1+a_j τ²)))
            prior_odds = pi0 / max(1.0 - pi0, 1e-12)
            denom = 1.0 + a_j * tau2
            exp_arg = tau2 * (c_j ** 2) / (2.0 * denom)
            # 防溢出：exp_arg clamp [-1e3, 500]
            log_Oj = (np.log(prior_odds) - 0.5 * np.log(max(denom, 1e-300))
                      + np.clip(exp_arg, -1000.0, 500.0))
            log_Oj = float(np.clip(log_Oj, -700.0, 700.0))
            # p_in = 1/(1+e^{-log_Oj})
            logit = log_Oj
            if logit >= 500:
                p_in = 1.0
            elif logit <= -500:
                p_in = 0.0
            else:
                p_in = 1.0 / (1.0 + np.exp(-logit))
            # 极小/大保护
            p_in = min(max(p_in, 1e-12), 1 - 1e-12)
            new_inc = bool(rng.random() < p_in)
            theta[j] = new_inc
            if new_inc:
                # 条件后验采样 β_j：β_j | θ_j=1, rest ~ N(mu, V)
                Vj = 1.0 / (1.0 / tau2 + max(a_j, 0.0))
                Muj = Vj * c_j
                new_b = float(rng.normal(Muj, np.sqrt(max(Vj, 0.0))))
                # 防止极端 β（不影响约束，但保证数值稳定）
                if np.isfinite(new_b):
                    new_b = float(np.clip(new_b, -1e6, 1e6))
                else:
                    new_b = 0.0
                beta[j] = new_b
                r = r_adj - phij * beta[j]
            else:
                beta[j] = 0.0
                r = r_adj

    # --------------- 主循环 ---------------
    for it in range(total):
        # 残差 = y - γ - Φ β
        r_cur = y - gamma - Phi @ beta
        _sample_lam_sig_tau(r_cur, theta, beta)
        _sample_eta_A(theta)
        _sample_theta_and_beta()

        # 只在 post-burn + thin 间隔 保存
        if it >= n_burn:
            rel = it - n_burn
            if rel % thin == 0 and store_idx < n_store:
                trace["gamma"][store_idx] = gamma
                trace["beta"][store_idx] = beta
                trace["theta"][store_idx] = theta.astype(np.int8)
                trace["sigma2"][store_idx] = sigma2
                trace["tau2"][store_idx] = tau2
                r_cur2 = y - gamma - Phi @ beta
                Dinv_diag = w / (sigma2 * lam + 1e-300)
                ll = -0.5 * float(np.sum(np.log(sigma2 * lam / w + 1e-300))) \
                     - 0.5 * float(np.sum(Dinv_diag * r_cur2 * r_cur2))
                trace["loglik"][store_idx] = ll
                store_idx += 1
    for k in trace:
        trace[k] = trace[k][:store_idx]
    final = {"gamma": gamma, "beta": beta, "theta": theta,
             "sigma2": sigma2, "tau2": tau2, "lam": lam}
    return trace, final


# ================================================================
#  主入口
# ================================================================
def main():
    log_lines = []
    def L(x): log_lines.append(str(x)); print(x)
    t0 = time.time()
    L("="*72)
    L("[Stage B] START  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    L(f"  Prior/CFG keys: {list(PRIOR.keys())}")

    if not os.path.exists(ARTIFACTS_A):
        raise FileNotFoundError(f"Need run_stage_A.py first. Missing {ARTIFACTS_A}")
    artA = load_pickle(ARTIFACTS_A)
    dw = artA["drift_work"]
    x_k = dw["x_k"]; b_hat = dw["b_hat"]; w_k = dw["w_k"]; n_k = dw["n_k"]
    K_kept = len(x_k)
    L(f"  Drift work samples loaded: K={K_kept}   sum(w_k)={w_k.sum():.6f}")
    L(f"  x_k range: [{x_k.min():.3f}, {x_k.max():.3f}]  "
      f"b_hat range: [{b_hat.min():.3f}, {b_hat.max():.3f}]")

    # (1) 候选基（结点仅由 x_k（训练箱代表态）决定 → 无泄漏）
    Phi, colnames, metaPhi = build_phi(x_k, x_k, PRIOR)
    p = Phi.shape[1]
    L(f"  Candidate basis: p={p} non-intercept columns")
    L(f"    colnames = {colnames}")
    L(f"    hinge nodes = {np.round(metaPhi['c_nodes'],4).tolist()}")

    # 保存 basis 清单
    pd.DataFrame({"term": ["(intercept γ)"] + colnames,
                  "role": ["always-in (no selection)"] +
                          ["poly" if c.startswith("x^") else "hinge"
                           for c in colnames]}).to_csv(
        os.path.join(STAGE_B_DIR, "stageB_candidate_basis.csv"),
        index=False, encoding="utf-8-sig")

    # (2) 初始化 3 种 dispersed
    #  (a) θ=0, (b) θ=1, (c) 加权 OLS top 半
    rng0 = np.random.default_rng(PRIOR["seed"])
    Wsr = np.sqrt(w_k)
    Aw = np.column_stack([Wsr, (Phi * Wsr[:, None])])
    yw = b_hat * Wsr
    beta_ols, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    t_vals = np.zeros(p)
    res = yw - Aw @ beta_ols
    dof = max(len(yw) - len(beta_ols), 1)
    s2 = float(np.sum(res ** 2) / dof)
    try:
        covb = s2 * np.linalg.inv(Aw.T @ Aw + 1e-12 * np.eye(Aw.shape[1]))
        t_vals = np.abs(beta_ols[1:]) / np.sqrt(np.maximum(np.diag(covb)[1:], 1e-300))
    except Exception:
        pass
    init_c = (t_vals >= np.median(t_vals))

    inits = [np.zeros(p, dtype=bool),
             np.ones(p, dtype=bool),
             init_c]

    # (3) 3 条链并行（此处顺序循环，保持可复现；如需真正多进程可改）
    all_trace = []
    all_final = []
    for cid, th0 in enumerate(inits):
        L(f"  Chain {cid+1}/{PRIOR['n_chains']}: init θ sum = {int(th0.sum())} / {p}")
        tr, fin = gibbs_ss(b_hat, Phi, w_k, PRIOR,
                           n_burn=PRIOR["n_burn"], n_post=PRIOR["n_post"],
                           seed=PRIOR["seed"] + cid * 1_000_003,
                           init_theta=th0)
        all_trace.append(tr); all_final.append(fin)
        # Quick diagnostics for this chain: active size
        ms = tr["theta"].sum(axis=1)
        L(f"    post |A| mean±sd = {ms.mean():.2f} ± {ms.std():.2f}   "
          f"loglik end = {tr['loglik'][-1]:.3f}")

    # (4) 跨链 R-hat & 合并 ESS (简单版：单变量 split-R hat  for 关键量)
    def r_hat_split(x_chain_list):
        """Basic split-R-hat: each chain split into halves → m=2n_chains vars."""
        parts = []
        for xs in x_chain_list:
            n = len(xs)
            parts.append(xs[:n // 2])
            parts.append(xs[n // 2:])
        m = len(parts); n = min(len(p) for p in parts)
        parts = np.array([p[:n] for p in parts])  # (m,n)
        chain_means = parts.mean(axis=1); overall_mean = chain_means.mean()
        B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
        W = np.mean(np.var(parts, axis=1, ddof=1))
        if W + 1e-300 <= 0:
            return np.nan
        var_hat = (1 - 1 / n) * W + B / n
        return float(np.sqrt(max(var_hat, 0.0) / (W + 1e-300)))

    def ess_geyer_init(x: np.ndarray):
        """Monotone early-stop initial convex ESS (Geyer 1992 initial monotone seq)."""
        n = len(x); xc = x - x.mean(); s0 = np.var(x, ddof=0) * n
        if s0 <= 0: return float(n)
        # 自相关 (FFT 实现)
        m = int(2 ** np.ceil(np.log2(2 * n)))
        f = np.fft.fft(xc - xc.mean(), m)
        acf = np.fft.ifft(f * np.conj(f))[:n].real
        acf /= acf[0] if acf[0] != 0 else 1.0
        # Geyer initial monotone: τ = 1 + 2 Σ_{k odd <= stop} ρ_k
        tau = 1.0; prev = np.inf
        for k in range(1, n - 1, 2):
            if k + 1 >= n: break
            pair_sum = acf[k] + acf[k + 1]
            if pair_sum < 0 or pair_sum > prev:
                # 保持单调: 钳制
                pair_sum = max(0.0, min(pair_sum, prev))
            prev = pair_sum
            if pair_sum <= 0:
                break
            tau += 2 * pair_sum
        return float(n / max(tau, 1.0))

    diag_rows = []
    # γ
    diag_rows.append({"param": "gamma",
                      "R_hat": r_hat_split([c["gamma"] for c in all_trace]),
                      "ESS_min": min(ess_geyer_init(c["gamma"]) for c in all_trace)})
    for name, key in [("sigma2", "sigma2"), ("tau2", "tau2"), ("loglik", "loglik")]:
        diag_rows.append({"param": name,
                          "R_hat": r_hat_split([c[key] for c in all_trace]),
                          "ESS_min": min(ess_geyer_init(c[key]) for c in all_trace)})
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(STAGE_B_DIR, "stageB_diagnostics.csv"),
        index=False, encoding="utf-8-sig")
    L("  Diagnostics (R-hat):")
    for r in diag_rows:
        L(f"    {r['param']:8s}  R̂={r['R_hat']:.4f}   ESS_min≥{r['ESS_min']:.0f}")

    # (5) 合并 3 链的所有 post draws → 计算 PIP & 后验系数
    beta_draws = np.concatenate([c["beta"] for c in all_trace], axis=0)  # (S,p)
    theta_draws = np.concatenate([c["theta"] for c in all_trace], axis=0).astype(np.int8)
    gamma_draws = np.concatenate([c["gamma"] for c in all_trace], axis=0)
    S = len(beta_draws)
    pip = theta_draws.mean(axis=0)
    rows = []
    for j, nm in enumerate(colnames):
        inc = theta_draws[:, j] == 1
        if inc.any():
            b_inc = beta_draws[inc, j]
            med = float(np.median(b_inc)); q025 = float(np.quantile(b_inc, 0.025))
            q975 = float(np.quantile(b_inc, 0.975))
            mean_s = float(np.mean(b_inc))
        else:
            med = q025 = q975 = mean_s = 0.0
        rows.append({"term": nm, "PIP": float(pip[j]),
                     "med_post_beta": med, "mean_post_beta": mean_s,
                     "q025_beta": q025, "q975_beta": q975,
                     "n_post_inc": int(inc.sum())})
    df_pip = pd.DataFrame(rows).sort_values("PIP", ascending=False).reset_index(drop=True)
    df_pip.to_csv(os.path.join(STAGE_B_DIR, "stageB_pip_coef_summary.csv"),
                  index=False, encoding="utf-8-sig")
    L("  PIP 排序表（前 10 行）：\n" + df_pip.head(10).to_string())

    # (6) 后验漂移带：在 x_grid（训练箱 x_k）上评估 b(x) = γ + Φβ，
    #     逐 draw 算 取 med/q025/q975/mean
    x_grid = np.linspace(x_k.min(), x_k.max(), 200)
    Phi_grid, _, _ = build_phi(x_k, x_grid, PRIOR)  # 设计仍基于训练 x_k 结点
    # 直接向量化
    b_all = gamma_draws[:, None] + beta_draws @ Phi_grid.T  # (S, 200)
    b_med = np.median(b_all, axis=0)
    b_q025 = np.quantile(b_all, 0.025, axis=0)
    b_q975 = np.quantile(b_all, 0.975, axis=0)
    b_mean = np.mean(b_all, axis=0)
    pd.DataFrame({"x": x_grid, "b_med": b_med, "b_q025": b_q025,
                  "b_q975": b_q975, "b_mean": b_mean}).to_csv(
        os.path.join(STAGE_B_DIR, "stageB_posterior_drift_band.csv"),
        index=False, encoding="utf-8-sig")

    # (7) 为 Stage C 提供“plug-in 漂移函数”的核心：
    #     - b_hat(x) = E_{post}[γ + Φ(x)β] —— 我们用 后验均值 点估计
    #       （论文是 固定第一阶段 drift 后做 plug-in → 任意一致选择均可。
    #        这里保存 Φ_grid/b_mean 已足够；另外保存 后验均值 γ̄ 与 β̄）
    gamma_bar = float(np.mean(gamma_draws))
    beta_bar = np.mean(beta_draws, axis=0)
    # 同时给出“中值漂移”（更稳健）作为第二选择
    beta_med = np.median(beta_draws, axis=0)
    gamma_med = float(np.median(gamma_draws))

    artB = {
        "prior": PRIOR,
        "basis_meta": metaPhi,
        "colnames": colnames,
        "gamma_bar": gamma_bar,
        "beta_bar": beta_bar,
        "gamma_med": gamma_med,
        "beta_med": beta_med,
        "x_k_design_min": float(x_k.min()),
        "x_k_design_max": float(x_k.max()),
        "build_phi_quantiles": PRIOR["hinge_c_quantiles"],
        "build_phi_poly_pwrs": PRIOR["include_poly"],
    }
    save_pickle(artB, os.path.join(STAGE_B_DIR, "stageB_run_artifacts.pkl"))

    # (8) report
    L(f"\n  [Stage B] DONE in {time.time()-t0:.1f}s. dir = {STAGE_B_DIR}")
    with open(os.path.join(STAGE_B_DIR, "stageB_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    # numpy warnings -> 忽略（R-hat 中 NaN 由零方差导致）
    warnings.filterwarnings("ignore")
    sys.exit(main())
