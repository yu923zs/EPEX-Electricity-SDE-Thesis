# -*- coding: utf-8 -*-
"""
Stage 5:  测试段样本外区间 / VaR 覆盖率 + Christoffersen 检验
----------------------------------------------------------------
- 测试段: 2025-01-01 00:00 ~ 2025-09-30 23:00 CET,  6,551 level → 6,550 1h Δ
- drift: b̂_med(x) = γ_med + Φ(x)^T β_med，结点扩展范围夹稳至训练态箱代表态的 [x_min, x_max]
- noise: TS ASY / SYM，Δt=1h CF → F_{Z(1)} 用 Stage 4 反演得到的 CDF 网格
- 构造：
    * 90%/95%/99% 一步预测区间:  [q(α/2), q(1-α/2)]
    * VaR 0.5%/99.5%: q(0.005), q(0.995)
- 报告:
    * 无条件覆盖率 hit_rate (%)
    * 尾部 exceed (%) 与名义的偏差
    * Christoffersen (1998) 条件覆盖率（LR_uc + LR_ind + LR_cc）的渐近 χ² p值
- 输出: stage_5_oos_coverage.csv / stage_5_oos_report.txt 到 stage_C_ts_ecf
"""
import os, sys, time, pickle
from typing import Optional, Union
import numpy as np
import pandas as pd
from scipy import stats as _s

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from thesis_pipeline_common import (
    load_smard_df, split_3way, fit_seasonal,
    _hour_in_day_fourier_basis, _dow_dummies, piecewise_linear_hinge_basis,
)
from run_stage_C import ts_phi_dt


def seasonal_predict_from_coefs(coefs, ts_new):
    """
    按 fit_seasonal 返回的参数（coef_lin / lw_hinge_knots_days / t_fit_min_ts /
    hod_nharm / lw_df）对新时间序列重新构造设计矩阵并预测 μ̂。
    """
    t_new = pd.to_datetime(pd.Series(ts_new).reset_index(drop=True))
    t_ref = pd.to_datetime(coefs["t_fit_min_ts"])
    t_days = (t_new - t_ref).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    hod = _hour_in_day_fourier_basis(t_new, nh=coefs.get("hod_nharm", 5))
    dow = _dow_dummies(t_new)
    knots = np.asarray(coefs["lw_hinge_knots_days"], dtype=np.float64).ravel()
    hinge = piecewise_linear_hinge_basis(t_days, knots)
    hinge_cols = [f"lw_hinge_{i}" for i in range(hinge.shape[1])]
    parts = [hod.reset_index(drop=True),
             dow.reset_index(drop=True),
             pd.DataFrame(hinge, columns=hinge_cols)]
    X2 = pd.concat(parts, axis=1)
    names_want = list(coefs["feature_names_lin"])  # intercept + ...
    # 按列名对齐; 但 intercept 将在下方手动 prepend
    cols_feat = names_want[1:]  # exclude 'intercept'
    X2 = X2.reindex(columns=cols_feat, fill_value=0.0).to_numpy(dtype=np.float64)
    A = np.column_stack([np.ones(len(X2)), X2])
    beta = np.asarray(coefs["coef_lin"], dtype=np.float64).ravel()
    return A @ beta

BASE = os.path.abspath(os.path.join(HERE, ".."))
DIR_A = os.path.join(BASE, "06_stage_outputs", "stage_A_drift_train")
DIR_B = os.path.join(BASE, "06_stage_outputs", "stage_B_sparse_drift")
DIR_C = os.path.join(BASE, "06_stage_outputs", "stage_C_ts_ecf")

OUT_COV = os.path.join(DIR_C, "stage_5_oos_coverage.csv")
OUT_VAR = os.path.join(DIR_C, "stage_5_oos_var_hit.csv")
OUT_REP = os.path.join(DIR_C, "stage_5_oos_report.txt")


def L(*args): print("[Stage5]", *args, flush=True)


# ================================================================
#  Christoffersen (1998) 检验
# ================================================================
def christoffersen_lr(hit, alpha_nom):
    """
    hit = 1 当 ΔX < VaR (下尾) 或 ΔX > VaR (上尾)，否则 0。alpha_nom=名义命中率。
    返回 (LR_uc, LR_ind, LR_cc, p_uc, p_ind, p_cc)。
    """
    hit = np.asarray(hit, dtype=int).ravel()
    n = len(hit)
    n_1 = int(hit.sum())
    n_0 = n - n_1
    # 二项 π_ML = n_1 / n
    pi_ml = n_1 / n if n > 0 else 0.0
    # LR_uc:  H0: pi = alpha
    def ll_bin(n0, n1, p):
        eps = 1e-300
        return n0 * np.log(max(1 - p, eps)) + n1 * np.log(max(p, eps))
    lr_uc = 2 * (ll_bin(n_0, n_1, pi_ml) - ll_bin(n_0, n_1, alpha_nom))
    p_uc = 1 - _s.chi2.cdf(max(lr_uc, 0.0), df=1)
    # LR_ind: 1 阶马尔可夫 vs iid（对 0/1 序列）
    # n_ij = #(t+1 = j, t = i)
    if n >= 2:
        h0 = hit[:-1]; h1 = hit[1:]
        n00 = int(np.sum((h0 == 0) & (h1 == 0)))
        n01 = int(np.sum((h0 == 0) & (h1 == 1)))
        n10 = int(np.sum((h0 == 1) & (h1 == 0)))
        n11 = int(np.sum((h0 == 1) & (h1 == 1)))
        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        ll_A = ll_bin(n00, n01, pi01) + ll_bin(n10, n11, pi11)
        ll_0 = ll_bin(n00 + n10, n01 + n11, pi_ml)
        lr_ind = 2 * (ll_A - ll_0)
        p_ind = 1 - _s.chi2.cdf(max(lr_ind, 0.0), df=1)
    else:
        lr_ind = 0.0
        p_ind = 1.0
    lr_cc = lr_uc + lr_ind
    p_cc = 1 - _s.chi2.cdf(max(lr_cc, 0.0), df=2)
    return dict(LR_uc=float(lr_uc), LR_ind=float(lr_ind), LR_cc=float(lr_cc),
                p_uc=float(p_uc), p_ind=float(p_ind), p_cc=float(p_cc))


# ================================================================
#  Drift 重建（复用 Stage B 的 β_med/γ_med/basis_meta）
# ================================================================
def _pwlin_hinge(x, c):
    return np.where(x > c, x - c, 0.0)


def reconstruct_b_med(x_arr, artB):
    colnames = list(artB["colnames"])  # includes x1,x2,x3, hinge names
    beta = np.asarray(artB["beta_med"]).ravel()  # p
    gamma = float(artB["gamma_med"])
    basis = artB["basis_meta"]
    P = np.empty((len(x_arr), len(colnames)), dtype=np.float64)
    poly_pwrs = list(basis["poly_pwrs"])
    c_nodes = list(basis["c_nodes"])
    col_idx = 0
    for pwr in poly_pwrs:
        P[:, col_idx] = np.power(x_arr, pwr)
        col_idx += 1
    for c in c_nodes:
        P[:, col_idx] = _pwlin_hinge(x_arr, c)
        col_idx += 1
    b = gamma + P @ beta
    return b


# ================================================================
#  TS 预测 CDF 反演（与 Stage 4 相同，局部实现避免循环依赖）
# ================================================================
def ts_cdf_x(x_arr, xi, symmetric, u_max=60.0, N_u=2 ** 14):
    u = np.linspace(0.0, u_max, N_u + 1)[1:]
    du = u[1] - u[0]
    if symmetric:
        mu, C, M, alpha = [float(v) for v in xi]
        phi = ts_phi_dt(u, (mu, C, C, M, M, alpha), dt=1.0)
    else:
        mu, Cp, Cm, M, G, alpha = [float(v) for v in xi]
        phi = ts_phi_dt(u, (mu, Cp, Cm, M, G, alpha), dt=1.0)
    x_arr = np.asarray(x_arr, dtype=np.float64).reshape(-1)
    Re_p = np.real(phi); Im_p = np.imag(phi)
    ux = np.outer(x_arr, u)
    cu = np.cos(ux); su = np.sin(ux)
    itd = (cu * Im_p[None, :] - su * Re_p[None, :]) / u[None, :]
    w = np.full(N_u, du); w[0] *= 0.5; w[-1] *= 0.5
    F = 0.5 - (1.0 / np.pi) * (itd @ w)
    F = np.clip(F, 0, 1)
    # 强制单调
    for i in range(1, len(F)):
        if F[i] < F[i-1]: F[i] = F[i-1]
    return np.clip(F, 0, 1)


def q_from_cdf_grid(p, xs, Fs):
    if p <= Fs[0]: return xs[0]
    if p >= Fs[-1]: return xs[-1]
    idx = int(np.searchsorted(Fs, p))
    F0, F1 = Fs[idx-1], Fs[idx]
    x0, x1 = xs[idx-1], xs[idx]
    if F1 - F0 < 1e-18: return x1
    return x0 + (p - F0) * (x1 - x0) / (F1 - F0)


def main():
    t0 = time.time()
    # ---- load
    L("loading splits & artifacts...")
    df_all = load_smard_df()
    splits = split_3way(df_all)
    train_price = splits["drift_train"].reset_index(drop=True)
    test_price = splits["test"].reset_index(drop=True)
    # fit seasonal 仅基于 drift_train 返回 X_all/mu_all（对 full df）及系数
    seas = fit_seasonal(train_price, df_all.reset_index(drop=True))
    with open(os.path.join(DIR_B, "stageB_run_artifacts.pkl"), "rb") as f:
        artB = pickle.load(f)
    with open(os.path.join(DIR_C, "stageC_run_artifacts.pkl"), "rb") as f:
        artC = pickle.load(f)
    xi_asy = artC["best_asymmetric"]["xi_final"]
    xi_sym = artC["best_symmetric"]["xi_final"]
    # 训练箱代表态 x 范围：从 stageA 的 bin_stats 取
    df_bin = pd.read_csv(os.path.join(DIR_A, "stageA_drift_bin_stats.csv"))
    x_min = float(df_bin["x_k"].min())
    x_max = float(df_bin["x_k"].max())
    # 扩展 5% 安全边（用 max 防止 0 负边）
    x_min = x_min - 0.05 * max(abs(x_min), 1.0)
    x_max = x_max + 0.05 * max(abs(x_max), 1.0)
    L(f"clamp x_test within [{x_min:.2f}, {x_max:.2f}] EUR/MWh")

    # ---- 构造测试段 X_test = P - seasonal_predict
    ts_test = test_price["ts_cet"].reset_index(drop=True)
    P_test_arr = test_price["price_eur_mwh_raw"].to_numpy(dtype=np.float64).ravel()
    mu_test = seasonal_predict_from_coefs(seas, ts_test)
    X_test_arr = P_test_arr - mu_test
    L(f"test n_level = {len(X_test_arr)}, X range = [{X_test_arr.min():.2f}, {X_test_arr.max():.2f}]")
    X_level = X_test_arr[:-1]
    DX = np.diff(X_test_arr)
    X_clamped = np.clip(X_level, x_min, x_max)
    b_test = reconstruct_b_med(X_clamped, artB)
    r_true = DX - b_test  # 真实残差 (test)
    n = len(r_true)
    L(f"test n_Δt = {n}, r_true min/max/med = [{r_true.min():.2f}, {r_true.max():.2f}, {np.median(r_true):.2f}]")

    # ---- TS (Δt=1h) 预测 CDF 网格 —— 足够覆盖 [-1000, 1000]
    x_grid = np.linspace(-1000.0, +1000.0, 2**13 + 1)  # 8193
    L("computing ASY CDF grid for OOS...")
    F_asy = ts_cdf_x(x_grid, xi_asy, symmetric=False)
    L("computing SYM CDF grid for OOS...")
    F_sym = ts_cdf_x(x_grid, xi_sym, symmetric=True)

    nominals = {
        "interval": [(0.10, 0.05, 0.95),
                     (0.05, 0.025, 0.975),
                     (0.01, 0.005, 0.995)],  # (alpha, lo, hi)
        "var_lo":   [0.01, 0.025, 0.05, 0.005, 0.001],
        "var_hi":   [0.99, 0.975, 0.95, 0.995, 0.999],
    }

    # 对于每个模型: 预先求对应 nominals 下的阈值 q = x_grid 插值
    def q_of(p_list, F, xg):
        return np.array([q_from_cdf_grid(p, xg, F) for p in p_list])

    # -- 区间覆盖率: interval α / 1-α名义 coverage 指标
    rows_cov = []
    rows_var = []
    for (mname, Fg) in [("ASY_TS", F_asy), ("SYM_TS", F_sym)]:
        for (alpha, plo, phi) in nominals["interval"]:
            q_lo, q_hi = q_of([plo, phi], Fg, x_grid)
            inside = np.mean((r_true >= q_lo) & (r_true <= q_hi))
            miss = 1 - inside
            lo_hit = float(np.mean(r_true < q_lo))
            hi_hit = float(np.mean(r_true > q_hi))
            # 覆盖率 Christoffersen: hit = 1 如果 outside（对称双侧）
            hit_out = ((r_true < q_lo) | (r_true > q_hi)).astype(int)
            ch_out = christoffersen_lr(hit_out, alpha_nom=alpha)
            rows_cov.append({
                "model": mname, "nominal_coverage": 1 - alpha,
                "alpha": alpha,
                "q_lo_EUR_per_MWh": float(q_lo), "q_hi_EUR_per_MWh": float(q_hi),
                "actual_coverage": float(inside),
                "coverage_gap_pct": float((inside - (1-alpha)) * 100),
                "lo_tail_hit_pct": lo_hit * 100,
                "hi_tail_hit_pct": hi_hit * 100,
                **{f"ch_{k}": float(v) for k, v in ch_out.items()},
            })
        for p in nominals["var_lo"]:
            q = q_of([p], Fg, x_grid)[0]
            hit = (r_true < q).astype(int)
            hit_rate = float(np.mean(hit))
            ch = christoffersen_lr(hit, alpha_nom=p)
            rows_var.append({
                "model": mname, "tail": "lower", "nominal_p": p,
                "q_predict_EUR": float(q),
                "actual_hit_rate": hit_rate,
                "gap_pct": (hit_rate - p) * 100,
                **{f"ch_{k}": float(v) for k, v in ch.items()},
            })
        for p in nominals["var_hi"]:
            q = q_of([p], Fg, x_grid)[0]
            hit = (r_true > q).astype(int)
            hit_rate = float(np.mean(hit))
            ch = christoffersen_lr(hit, alpha_nom=1 - p)
            rows_var.append({
                "model": mname, "tail": "upper", "nominal_p": 1 - p,
                "nominal_quantile": p,
                "q_predict_EUR": float(q),
                "actual_hit_rate": hit_rate,
                "gap_pct": (hit_rate - (1 - p)) * 100,
                **{f"ch_{k}": float(v) for k, v in ch.items()},
            })

    df_cov = pd.DataFrame(rows_cov)
    df_var = pd.DataFrame(rows_var)
    df_cov.to_csv(OUT_COV, index=False, encoding="utf-8-sig")
    df_var.to_csv(OUT_VAR, index=False, encoding="utf-8-sig")
    L(f"Wrote {OUT_COV}  ({len(df_cov)} rows)")
    L(f"Wrote {OUT_VAR}  ({len(df_var)} rows)")

    # ---- 报告 TXT
    lines = [f"Stage 5 OOS coverage @ test segment 2025-01-01..09-30 CET, n={n}"]
    lines.append("=" * 78)
    lines.append("INTERVAL COVERAGE:")
    for _, r in df_cov.iterrows():
        lines.append(
            f"  {r['model']:7s}  cover={100*r['nominal_coverage']:3.0f}%  "
            f"actual={100*r['actual_coverage']:5.2f}%  gap={r['coverage_gap_pct']:+.2f}pp  "
            f"lo/hi hit={r['lo_tail_hit_pct']:.2f}/{r['hi_tail_hit_pct']:.2f}%  "
            f"Ch.p_uc={r['ch_p_uc']:.3f}  Ch.p_cc={r['ch_p_cc']:.3f}  "
            f"q=[{r['q_lo_EUR_per_MWh']:+.2f},{r['q_hi_EUR_per_MWh']:+.2f}]"
        )
    lines.append("-" * 78)
    lines.append("VaR TAIL HIT:")
    for _, r in df_var.iterrows():
        np_ = r.get("nominal_p", r.get("nominal_quantile"))
        lines.append(
            f"  {r['model']:7s} {r['tail']:5s}  nom={100*r['nominal_p']:.3f}%  "
            f"actual={100*r['actual_hit_rate']:.3f}%  gap={r['gap_pct']:+.3f}pp  "
            f"q={r['q_predict_EUR']:+.2f}  Ch.p_uc={r['ch_p_uc']:.3f}  Ch.p_cc={r['ch_p_cc']:.3f}"
        )
    lines.append("-" * 78)
    lines.append(f"Elapsed total = {time.time()-t0:.1f}s")
    with open(OUT_REP, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        L(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
