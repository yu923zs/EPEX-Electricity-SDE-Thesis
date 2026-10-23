# -*- coding: utf-8 -*-
"""
Stage 4 v2:  TS CF → CDF via adaptive scipy.integrate.quad
----------------------------------------------------------
- 直接使用数值稳定的 Cosine 反演公式：
    F(x) = 1/2 - (1/π) ∫₀^∞  Im[e^{-i u x} φ(u)] / u  du
  其中 φ(u) = exp(ψ(u)) (dt=1)
- 积分用 quad 自适应到 ∞。
- 对一组分位概率 p ∈ PROBS，用 Brentq 求 x=F^{-1}(p) 的根：
    F(x) - p = 0
- 输出：stage_4_quantiles_compare_v2.csv / stage_4_report.txt
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
from scipy import integrate, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_stage_C import ts_phi_dt

BASE = os.path.abspath(os.path.join(HERE, ".."))
DIR_C = os.path.join(BASE, "06_stage_outputs", "stage_C_ts_ecf")
OUT_TAB = os.path.join(DIR_C, "stage_4_quantiles_compare_v2.csv")
OUT_REP = os.path.join(DIR_C, "stage_4_report_v2.txt")

PROBS = np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50,
                  0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999])


def L(*args): print("[Stage4 v2]", *args, flush=True)


def _ts_phi_single(u, xi6):
    """ts_phi_dt expects an array; wrap for scalar u in quad."""
    uarr = np.atleast_1d(np.asarray(u, dtype=np.float64))
    # use the vectorised function; take first element
    return ts_phi_dt(uarr, xi6, dt=1.0)[0]


def cdf_at_x(x: float, xi, symmetric: bool, limit: int = 500):
    """F(x) 用 scipy.integrate.quad 自适应积分。"""
    if symmetric:
        mu, C, M, alpha = [float(v) for v in xi]
        xi6 = (mu, C, C, M, M, alpha)
    else:
        mu, Cp, Cm, M, G, alpha = [float(v) for v in xi]
        xi6 = (mu, Cp, Cm, M, G, alpha)

    def integrand(u):
        # Im[exp(-i u x) phi(u)] / u
        # = [cos(u x) * Im phi(u) - sin(u x) * Re phi(u)] / u
        # stability: 当 u→0 时 Im[φ] ~ ψ(u) ~ u * μ + 高阶 → /u 有限 ~ μ；quad 自动处理 0 端点
        phi = _ts_phi_single(u, xi6)
        cu = np.cos(u * x); su = np.sin(u * x)
        return (cu * np.imag(phi) - su * np.real(phi)) / u

    eps = 1e-10
    # split into (0, 1.0] & (1.0, inf) for stability if needed
    try:
        v1, e1 = integrate.quad(integrand, 0.0, 1.0, limit=limit,
                                epsabs=1e-8, epsrel=1e-6)
        v2, e2 = integrate.quad(integrand, 1.0, np.inf, limit=max(limit, 800),
                                epsabs=1e-8, epsrel=1e-5)
        val = v1 + v2
    except Exception:
        # fallback: 单次 (eps, 1000)
        val, _ = integrate.quad(integrand, 1e-9, 1000.0, limit=2000,
                                epsabs=1e-7, epsrel=1e-5)
    F = 0.5 - val / np.pi
    # clamp
    if np.isnan(F):
        F = 0.5
    return float(np.clip(F, eps, 1 - eps))


def quantile_at_p(p: float, xi, symmetric: bool,
                  x_low_bracket: float = -500.0,
                  x_high_bracket: float = +1000.0,
                  max_iter: int = 5000,
                  tol: float = 1e-4):
    """Brentq 找 x* s.t. F(x*) = p."""
    # 先验证端点：F(x_low) < p < F(x_high)
    F_lo = cdf_at_x(x_low_bracket, xi, symmetric)
    F_hi = cdf_at_x(x_high_bracket, xi, symmetric)
    k_expand = 0
    while F_lo >= p and k_expand < 10:
        x_low_bracket *= 2.0
        F_lo = cdf_at_x(x_low_bracket, xi, symmetric)
        k_expand += 1
    while F_hi <= p and k_expand < 20:
        x_high_bracket *= 2.0
        F_hi = cdf_at_x(x_high_bracket, xi, symmetric)
        k_expand += 1
    if F_lo > p or F_hi < p:
        # 极端兜底：返回更外推的 bracket
        return np.nan
    try:
        def f_obj(x):
            return cdf_at_x(x, xi, symmetric) - p
        r = optimize.brentq(f_obj, x_low_bracket, x_high_bracket,
                            xtol=tol, maxiter=max_iter)
        return float(r)
    except Exception:
        return np.nan


def main():
    t0 = time.time()
    L("load artifacts...")
    with open(os.path.join(DIR_C, "stageC_run_artifacts.pkl"), "rb") as f:
        artC = pickle.load(f)
    resid = np.asarray(artC["resid_plugin"]).ravel()
    xi_asy = artC["best_asymmetric"]["xi_final"]
    xi_sym = artC["best_symmetric"]["xi_final"]

    q_emp = np.quantile(resid, PROBS, method="linear")
    q_asy = np.full(len(PROBS), np.nan)
    q_sym = np.full(len(PROBS), np.nan)
    F_asy_at_emp = np.full(len(PROBS), np.nan)
    F_sym_at_emp = np.full(len(PROBS), np.nan)
    for i, p in enumerate(PROBS):
        t1 = time.time()
        q_asy[i] = quantile_at_p(p, xi_asy, False)
        q_sym[i] = quantile_at_p(p, xi_sym, True)
        F_asy_at_emp[i] = cdf_at_x(q_emp[i], xi_asy, False)
        F_sym_at_emp[i] = cdf_at_x(q_emp[i], xi_sym, True)
        L(f"  p={p:.4f}  qEmp={q_emp[i]:+.3f}  qAsy={q_asy[i]:+.3f}  qSym={q_sym[i]:+.3f}  "
          f"FAsy(qEmp)={F_asy_at_emp[i]:.5f}  FSym(qEmp)={F_sym_at_emp[i]:.5f}  "
          f"took {time.time()-t1:.1f}s")

    df = pd.DataFrame({
        "prob": PROBS,
        "q_empirical": q_emp,
        "q_model_asy": q_asy,
        "q_model_sym": q_sym,
        "abs_err_asy": np.abs(q_asy - q_emp),
        "abs_err_sym": np.abs(q_sym - q_emp),
        "F_asy_at_qEmp": F_asy_at_emp,
        "F_sym_at_qEmp": F_sym_at_emp,
        "empirical_exceed_pct": (1.0 - PROBS) * 100,
        "asy_exceed_at_qEmp_pct": (1.0 - F_asy_at_emp) * 100,
        "sym_exceed_at_qEmp_pct": (1.0 - F_sym_at_emp) * 100,
        "gap_asy_pct": ((1.0 - F_asy_at_emp) - (1.0 - PROBS)) * 100,
        "gap_sym_pct": ((1.0 - F_sym_at_emp) - (1.0 - PROBS)) * 100,
    })
    df.to_csv(OUT_TAB, index=False, encoding="utf-8-sig")
    L(f"Wrote {OUT_TAB}")

    def _fmt(vals, fmt="%+.2f"):
        return ", ".join(fmt % v for v in vals)

    lines = []
    lines.append(f"Stage 4 v2: adaptive-quant CDF inversion (scipy quad + brentq)")
    lines.append(f"n_ts_resid = {len(resid)}")
    lines.append(f"xi_asy (mu,Cp,Cm,M,G,alpha) = {np.round(xi_asy,4).tolist()}")
    lines.append(f"xi_sym (mu,C,M,alpha) = {np.round(xi_sym,4).tolist()}")
    lines.append("Probs     : " + ", ".join(f"{v:.4f}" for v in PROBS.tolist()))
    lines.append("q_Empirical: " + _fmt(q_emp))
    lines.append("q_model ASY: " + _fmt(q_asy))
    lines.append("q_model SYM: " + _fmt(q_sym))
    # 重点指标：5% 双侧 / 1% 双侧 超越率
    i05 = int(np.where(PROBS == 0.05)[0][0])
    i95 = int(np.where(PROBS == 0.95)[0][0])
    i01 = int(np.where(PROBS == 0.01)[0][0])
    i99 = int(np.where(PROBS == 0.99)[0][0])
    for (lbl, ia, ib) in [("5/95", i05, i95), ("1/99", i01, i99)]:
        lines.append(
            f"{lbl} exceed emp= {100*(1-PROBS[ib]):.2f}/{100*PROBS[ia]:.2f} %  "
            f"ASY model= {100*(1-F_asy_at_emp[ib]):.2f}/{100*F_asy_at_emp[ia]:.2f} %   "
            f"SYM model= {100*(1-F_sym_at_emp[ib]):.2f}/{100*F_sym_at_emp[ia]:.2f} %")
    lines.append(f"Total elapsed = {time.time()-t0:.1f}s")
    with open(OUT_REP, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        L(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
