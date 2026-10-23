# -*- coding: utf-8 -*-
"""
Stage 4:  TS CF → CDF (Cosine inversion) → 经验 vs 模型分位对比
---------------------------------------------------------------
- 使用已拟合的 ASY/SYM TS 参数（stageC_run_artifacts.pkl）
- Cosine 反演公式：
    F(x) = 1/2 - (1/π) ∫₀^∞ Im[ e^{-iux} φ(u) ] / u  du
       = 1/2 - (1/π) ∫₀^∞ [cos(ux) Im φ(u) - sin(ux) Re φ(u)] / u  du   （注 e^{-iux}=cosux - i sinux）
       → Im[ e^{-iux} φ ] = cos(ux) · Im φ(u) - sin(ux) · Re φ(u)
  为了在 x=0 附近更稳定，我们用分布函数改写并做 Gauss-Legendre 梯形；
  对 u > u_max 采用指数尾部截断近似。
- 输出：stage_4_tail_quantiles_compare.csv 到 stage_C_ts_ecf 目录
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 复用 run_stage_C 的 ψ 计算、CF、数据载入
from run_stage_C import (
    ts_phi_dt,
)

BASE = os.path.abspath(os.path.join(HERE, ".."))
DIR_C = os.path.join(BASE, "06_stage_outputs", "stage_C_ts_ecf")
OUT_TAB = os.path.join(DIR_C, "stage_4_quantiles_compare.csv")
OUT_PDFGRID = os.path.join(DIR_C, "stage_4_pdf_cdf_grid.csv")
OUT_REPORT = os.path.join(DIR_C, "stage_4_report.txt")


def L(*args, flush=True):
    print("[Stage4]", *args, flush=flush)


def ts_cdf_x(x_arr, xi, symmetric: bool, u_max=60.0, N_u=2 ** 14):
    """
    一维 Cosine 反演 F(x)。对每个 x 独立积分：
        F(x) = 0.5 - (1/π) ∫_0^{u_max} Im[ e^{-i u x} φ(u) ] / u  du
             + tail_correction (可忽略若 u_max 足够大 & φ 衰减)。
    TS 的 |φ(u)| ~ O(u^{-α}) 衰减（慢于布朗 O(u^{-2})），
    我们取 N_u = 2^14=16384 个均匀 u 点并用梯形。
    xi: 为 6 维（μ,Cp,Cm,M,G,α）或 4 维（μ,C,M,α），symmetric 决定格式。
    """
    u = np.linspace(0.0, u_max, N_u + 1)[1:]  # drop u=0 (avoid 1/u)
    du = u[1] - u[0]
    if symmetric:
        mu, C, M, alpha = float(xi[0]), float(xi[1]), float(xi[2]), float(xi[3])
        xi6 = (mu, C, C, M, M, alpha)
        phi = ts_phi_dt(u, xi6, dt=1.0)
    else:
        mu, Cp, Cm, M, G, alpha = [float(v) for v in xi]
        phi = ts_phi_dt(u, (mu, Cp, Cm, M, G, alpha), dt=1.0)
    # 构造矩阵：x 维度 × u
    # e^{-i u x} = cos(ux) - i sin(ux)
    # Im[ e^{-i u x} phi ] = cos(ux)*Im φ - sin(ux)*Re φ
    x_arr = np.asarray(x_arr, dtype=np.float64).reshape(-1)
    Re_phi = np.real(phi); Im_phi = np.imag(phi)
    ux = np.outer(x_arr, u)  # (n_x, N_u)
    cos_ux = np.cos(ux); sin_ux = np.sin(ux)
    integrand = (cos_ux * Im_phi[None, :] - sin_ux * Re_phi[None, :]) / u[None, :]
    # trapezoid rule: weights du for all interior points, (du/2) at endpoints
    w = np.full(N_u, du)
    w[0] *= 0.5; w[-1] *= 0.5
    intgrl = integrand @ w
    F = 0.5 - (1.0 / np.pi) * intgrl
    # clip
    return np.clip(F, 0.0, 1.0)


def ts_quantile_from_grid(q, x_grid, F_grid):
    """线性插值得到 q 分位。"""
    idx = np.searchsorted(F_grid, q)
    if idx == 0:
        return x_grid[0]
    if idx >= len(F_grid):
        return x_grid[-1]
    x0, x1 = x_grid[idx - 1], x_grid[idx]
    F0, F1 = F_grid[idx - 1], F_grid[idx]
    if F1 - F0 < 1e-18:
        return x1
    return x0 + (q - F0) * (x1 - x0) / (F1 - F0)


def main():
    t0 = time.time()
    L("load artifacts...")
    with open(os.path.join(DIR_C, "stageC_run_artifacts.pkl"), "rb") as f:
        artC = pickle.load(f)
    resid = np.asarray(artC["resid_plugin"]).ravel()
    xi_asy = artC["best_asymmetric"]["xi_final"]
    xi_sym = artC["best_symmetric"]["xi_final"]
    L(f"resid n = {len(resid)}, xi_asy = {xi_asy.round(4).tolist()}")
    L(f"xi_sym (4d) = {xi_sym.round(4).tolist()}")

    # (1) 经验尾部分位表（与 chap7 数字对齐）
    probs = np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50,
                      0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999])
    q_emp = np.quantile(resid, probs, method="linear")
    L("computing ASY CDF on grid...")
    # x_grid: 覆盖 [-600, +800] 足以包围 resid 极值 (-355 / +412)，并延伸到 VaR 0.1% 外侧
    x_grid = np.linspace(-650.0, +850.0, 2 ** 13 + 1)  # 8193 个点，步长 ~0.183 EUR
    F_asy = ts_cdf_x(x_grid, xi_asy, symmetric=False)
    L("computing SYM CDF on grid...")
    F_sym = ts_cdf_x(x_grid, xi_sym, symmetric=True)
    # 端点单调检查（反演截断数值噪声导致的非单调时修复）
    def _fix_monotone(F):
        for i in range(1, len(F)):
            if F[i] < F[i - 1]:
                F[i] = F[i - 1]
        return np.clip(F, 0, 1)
    F_asy = _fix_monotone(F_asy)
    F_sym = _fix_monotone(F_sym)

    q_asy = np.array([ts_quantile_from_grid(p, x_grid, F_asy) for p in probs])
    q_sym = np.array([ts_quantile_from_grid(p, x_grid, F_sym) for p in probs])

    # (2) 超越率（经验 5/95、1/99、0.5/99.5 等用 ASY/SYM CDF 在经验分位处取值 → 模型预测超越率）
    #  经验分位 q_p_emp 代入模型 CDF，报告 CDF(q_p_emp) 与 p 的偏差。
    F_asy_at_qemp = np.array([ts_quantile_from_grid.__wrapped__ if False else None
                              for _ in probs])  # placeholder unused
    # 直接线性插值 F(x=q_emp)
    F_asy_at = np.interp(q_emp, x_grid, F_asy, left=0.0, right=1.0)
    F_sym_at = np.interp(q_emp, x_grid, F_sym, left=0.0, right=1.0)

    # (3) 汇总表（CSV）
    df = pd.DataFrame({
        "prob": probs,
        "q_empirical": q_emp,
        "q_model_asy": q_asy,
        "q_model_sym": q_sym,
        "abs_err_asy": np.abs(q_asy - q_emp),
        "abs_err_sym": np.abs(q_sym - q_emp),
        "rel_err_asy_%": np.where(np.abs(q_emp) > 1e-6,
                                  (q_asy - q_emp) / np.where(np.abs(q_emp) < 1, 1, np.abs(q_emp)) * 100,
                                  np.nan),
        "rel_err_sym_%": np.where(np.abs(q_emp) > 1e-6,
                                  (q_sym - q_emp) / np.where(np.abs(q_emp) < 1, 1, np.abs(q_emp)) * 100,
                                  np.nan),
        "F_asy_at_qEmp": F_asy_at,
        "F_sym_at_qEmp": F_sym_at,
        "empirical_exceed": 1.0 - probs,
        "asy_exceed_at_qEmp": 1.0 - F_asy_at,
        "sym_exceed_at_qEmp": 1.0 - F_sym_at,
    })
    df.to_csv(OUT_TAB, index=False, encoding="utf-8-sig")
    L(f"Wrote {OUT_TAB} (rows={len(df)})")

    # (4) PDF/CDF 网格 CSV 为将来绘图
    pd.DataFrame({
        "x": x_grid,
        "F_asy": F_asy,
        "F_sym": F_sym,
    }).to_csv(OUT_PDFGRID, index=False, encoding="utf-8-sig")

    # (5) 汇总误差指标
    mad_asy = float(np.mean(df["abs_err_asy"]))
    mad_sym = float(np.mean(df["abs_err_sym"]))
    wmad_asy_tail = float(np.mean(np.r_[df["abs_err_asy"].values[:5],
                                        df["abs_err_asy"].values[-5:]]))
    wmad_sym_tail = float(np.mean(np.r_[df["abs_err_sym"].values[:5],
                                        df["abs_err_sym"].values[-5:]]))
    # 5% 双侧超越率偏差绝对值（P(r<q05), P(r>q95)）
    idx05 = int(np.where(probs == 0.05)[0][0])
    idx95 = int(np.where(probs == 0.95)[0][0])
    idx01 = int(np.where(probs == 0.01)[0][0])
    idx99 = int(np.where(probs == 0.99)[0][0])
    def _line(name, arr):
        a = ", ".join(f"{v:.4f}" for v in arr)
        return f"{name}: {a}"
    lines = []
    lines.append(f"Stage 4  CF→CDF cosine inversion, N_u=16384, x_grid {len(x_grid)} pts, "
                 f"x in [{x_grid[0]:+.1f}, {x_grid[-1]:+.1f}] EUR/MWh")
    lines.append(f"MAE (all p):       ASY={mad_asy:.3f}, SYM={mad_sym:.3f} EUR/MWh")
    lines.append(f"MAE (10 tail p):   ASY={wmad_asy_tail:.3f}, SYM={wmad_sym_tail:.3f} EUR/MWh")
    lines.append(_line("Prob levels        ", probs.tolist()))
    lines.append(_line("q_empirical        ", q_emp.tolist()))
    lines.append(_line("q_model_asy        ", q_asy.tolist()))
    lines.append(_line("q_model_sym        ", q_sym.tolist()))
    lines.append(
        f"5%/95% exceed emp=5.00/5.00 %,  "
        f"asy={100*F_asy_at[idx05]:.2f}/{100*(1-F_asy_at[idx95]):.2f} %,  "
        f"sym={100*F_sym_at[idx05]:.2f}/{100*(1-F_sym_at[idx95]):.2f} %")
    lines.append(
        f"1%/99% exceed emp=1.00/1.00 %,  "
        f"asy={100*F_asy_at[idx01]:.2f}/{100*(1-F_asy_at[idx99]):.2f} %,  "
        f"sym={100*F_sym_at[idx01]:.2f}/{100*(1-F_sym_at[idx99]):.2f} %")
    lines.append(f"elapsed = {time.time()-t0:.1f}s")
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        L(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
