# -*- coding: utf-8 -*-
"""
Stage 5 v2:  OOS 1-step interval / VaR + Christoffersen
--------------------------------------------------------
- 替换均匀 Cosine grid：使用 scipy quad 自适应 CDF（stage4v2 公式）。
- 为区间 90/95/99% (α/2=0.05,0.025,0.005) & VaR 下/上尾 1/2.5/5/0.5/0.1 %
  直接用 brentq 根找分位数阈值。
- 对 6,550 OOS 残差计算覆盖 / 命中 + Christoffersen (LR_uc, LR_ind, LR_cc) p 值
- 输出：stage_5_oos_coverage_v2.csv / stage_5_oos_var_hit_v2.csv / stage_5_oos_report_v2.txt
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats as _s

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from thesis_pipeline_common import (
    load_smard_df, split_3way, fit_seasonal,
    _hour_in_day_fourier_basis, _dow_dummies, piecewise_linear_hinge_basis,
)
from run_stage_C import ts_phi_dt

BASE = os.path.abspath(os.path.join(HERE, ".."))
DIR_A = os.path.join(BASE, "06_stage_outputs", "stage_A_drift_train")
DIR_B = os.path.join(BASE, "06_stage_outputs", "stage_B_sparse_drift")
DIR_C = os.path.join(BASE, "06_stage_outputs", "stage_C_ts_ecf")

OUT_COV = os.path.join(DIR_C, "stage_5_oos_coverage_v2.csv")
OUT_VAR = os.path.join(DIR_C, "stage_5_oos_var_hit_v2.csv")
OUT_REP = os.path.join(DIR_C, "stage_5_oos_report_v2.txt")


def L(*args): print("[Stage5 v2]", *args, flush=True)


# ---------------- CDF via adaptive quad (shared impl with stage4v2) ----------------
def _ts_phi_single(u, xi6):
    uarr = np.atleast_1d(np.asarray(u, dtype=np.float64))
    return ts_phi_dt(uarr, xi6, dt=1.0)[0]


def cdf_at_x(x, xi, symmetric):
    if symmetric:
        mu, C, M, alpha = [float(v) for v in xi]
        xi6 = (mu, C, C, M, M, alpha)
    else:
        mu, Cp, Cm, M, G, alpha = [float(v) for v in xi]
        xi6 = (mu, Cp, Cm, M, G, alpha)

    def integrand(u):
        phi = _ts_phi_single(u, xi6)
        cu = np.cos(u * x); su = np.sin(u * x)
        return (cu * np.imag(phi) - su * np.real(phi)) / u

    try:
        v1, _ = integrate.quad(integrand, 0.0, 1.0, limit=400,
                               epsabs=1e-8, epsrel=1e-6)
        v2, _ = integrate.quad(integrand, 1.0, np.inf, limit=800,
                               epsabs=1e-8, epsrel=1e-5)
        F = 0.5 - (v1 + v2) / np.pi
    except Exception:
        F = 0.5
    if not np.isfinite(F):
        F = 0.5
    eps = 1e-10
    return float(np.clip(F, eps, 1 - eps))


def quantile(p, xi, symmetric, bracket_low=-500.0, bracket_high=1000.0):
    F_lo = cdf_at_x(bracket_low, xi, symmetric)
    F_hi = cdf_at_x(bracket_high, xi, symmetric)
    k = 0
    while F_lo >= p and k < 12:
        bracket_low *= 2.0
        F_lo = cdf_at_x(bracket_low, xi, symmetric); k += 1
    while F_hi <= p and k < 24:
        bracket_high *= 2.0
        F_hi = cdf_at_x(bracket_high, xi, symmetric); k += 1
    if F_lo > p or F_hi < p:
        return np.nan
    try:
        return float(optimize.brentq(lambda x: cdf_at_x(x, xi, symmetric) - p,
                                      bracket_low, bracket_high,
                                      xtol=1e-3, maxiter=2000))
    except Exception:
        return np.nan


# ---------------- Christoffersen ----------------
def christoffersen_lr(hit, alpha_nom):
    hit = np.asarray(hit, dtype=int).ravel()
    n = len(hit); n1 = int(hit.sum()); n0 = n - n1
    pi_ml = n1 / n if n else 0.0
    def ll_bin(n0, n1, p):
        eps = 1e-300
        return n0 * np.log(max(1 - p, eps)) + n1 * np.log(max(p, eps))
    lr_uc = 2 * (ll_bin(n0, n1, pi_ml) - ll_bin(n0, n1, alpha_nom))
    p_uc = 1 - _s.chi2.cdf(max(lr_uc, 0.0), df=1)
    if n >= 2:
        h0 = hit[:-1]; h1 = hit[1:]
        n00 = int(np.sum((h0 == 0) & (h1 == 0)))
        n01 = int(np.sum((h0 == 0) & (h1 == 1)))
        n10 = int(np.sum((h0 == 1) & (h1 == 0)))
        n11 = int(np.sum((h0 == 1) & (h1 == 1)))
        pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
        pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
        ll_A = ll_bin(n00, n01, pi01) + ll_bin(n10, n11, pi11)
        ll_0 = ll_bin(n00 + n10, n01 + n11, pi_ml)
        lr_ind = 2 * (ll_A - ll_0)
        p_ind = 1 - _s.chi2.cdf(max(lr_ind, 0.0), df=1)
    else:
        lr_ind = 0.0; p_ind = 1.0
    lr_cc = lr_uc + lr_ind
    p_cc = 1 - _s.chi2.cdf(max(lr_cc, 0.0), df=2)
    return dict(LR_uc=float(lr_uc), LR_ind=float(lr_ind), LR_cc=float(lr_cc),
                p_uc=float(p_uc), p_ind=float(p_ind), p_cc=float(p_cc))


# ---------------- 其他辅助 ----------------
def seasonal_predict_from_coefs(coefs, ts_new):
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
    cols = list(coefs["feature_names_lin"])[1:]
    X2 = X2.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=np.float64)
    A = np.column_stack([np.ones(len(X2)), X2])
    beta = np.asarray(coefs["coef_lin"], dtype=np.float64).ravel()
    return A @ beta


def _pwlin_hinge(x, c): return np.where(x > c, x - c, 0.0)


def reconstruct_b_med(x_arr, artB):
    beta = np.asarray(artB["beta_med"]).ravel()
    gamma = float(artB["gamma_med"])
    basis = artB["basis_meta"]
    P = np.empty((len(x_arr), len(artB["colnames"])), dtype=np.float64)
    j = 0
    for pwr in basis["poly_pwrs"]:
        P[:, j] = np.power(x_arr, pwr); j += 1
    for c in basis["c_nodes"]:
        P[:, j] = _pwlin_hinge(x_arr, c); j += 1
    return gamma + P @ beta


def main():
    t0 = time.time()
    L("load splits & artifacts...")
    df_all = load_smard_df()
    splits = split_3way(df_all)
    train_price = splits["drift_train"].reset_index(drop=True)
    test_price = splits["test"].reset_index(drop=True)
    seas = fit_seasonal(train_price, df_all.reset_index(drop=True))
    with open(os.path.join(DIR_B, "stageB_run_artifacts.pkl"), "rb") as f:
        artB = pickle.load(f)
    with open(os.path.join(DIR_C, "stageC_run_artifacts.pkl"), "rb") as f:
        artC = pickle.load(f)
    xi_asy = artC["best_asymmetric"]["xi_final"]
    xi_sym = artC["best_symmetric"]["xi_final"]
    df_bin = pd.read_csv(os.path.join(DIR_A, "stageA_drift_bin_stats.csv"))
    x_min = float(df_bin["x_k"].min()); x_max = float(df_bin["x_k"].max())
    x_min = x_min - 0.05 * max(abs(x_min), 1.0)
    x_max = x_max + 0.05 * max(abs(x_max), 1.0)
    L(f"drift clamp within [{x_min:.2f}, {x_max:.2f}]")

    ts_test = test_price["ts_cet"].reset_index(drop=True)
    P_test_arr = test_price["price_eur_mwh_raw"].to_numpy(dtype=np.float64).ravel()
    mu_test = seasonal_predict_from_coefs(seas, ts_test)
    X_test_arr = P_test_arr - mu_test
    X_level = X_test_arr[:-1]; DX = np.diff(X_test_arr)
    X_clamped = np.clip(X_level, x_min, x_max)
    b_test = reconstruct_b_med(X_clamped, artB)
    r_true = DX - b_test
    n = len(r_true)
    L(f"test n_Δt={n}, r_true min/med/max=[{r_true.min():.1f}, {np.median(r_true):.1f}, {r_true.max():.1f}]")

    # nominals
    intervals = [(0.10, 0.05, 0.95),
                 (0.05, 0.025, 0.975),
                 (0.01, 0.005, 0.995)]
    var_lo_ps = [0.01, 0.025, 0.05, 0.005, 0.001]
    var_hi_qtiles = [0.99, 0.975, 0.95, 0.995, 0.999]

    # 计算 12 个阈值 (asy/sym × 6 quantiles) 用 quantile()
    unique_ps = sorted(set(
        [plo for (_, plo, _) in intervals] +
        [phi for (_, _, phi) in intervals] +
        var_lo_ps + var_hi_qtiles
    ))
    L(f"computing {len(unique_ps)} quantiles per model (adaptive quad+brentq)...")
    thresh = {}
    for mname, xi, sym in [("ASY_TS", xi_asy, False), ("SYM_TS", xi_sym, True)]:
        thresh[mname] = {}
        tA = time.time()
        for p in unique_ps:
            qp = quantile(p, xi, sym)
            thresh[mname][p] = qp
            L(f"  {mname}  p={p:.4f} → q={qp:+.2f} EUR/MWh  ({time.time()-tA:.1f}s 子累计)")
        L(f"  {mname} quantiles done in {time.time()-tA:.1f}s")

    rows_cov = []
    rows_var = []
    for mname, xi, sym in [("ASY_TS", xi_asy, False), ("SYM_TS", xi_sym, True)]:
        # intervals
        for alpha, plo, phi in intervals:
            q_lo = thresh[mname][plo]; q_hi = thresh[mname][phi]
            inside = float(np.mean((r_true >= q_lo) & (r_true <= q_hi)))
            lo_hit = float(np.mean(r_true < q_lo)); hi_hit = float(np.mean(r_true > q_hi))
            hit_out = ((r_true < q_lo) | (r_true > q_hi)).astype(int)
            ch = christoffersen_lr(hit_out, alpha_nom=alpha)
            rows_cov.append({
                "model": mname, "nominal_coverage": 1 - alpha, "alpha": alpha,
                "q_lo_EUR": q_lo, "q_hi_EUR": q_hi,
                "actual_coverage": inside,
                "coverage_gap_pp": (inside - (1 - alpha)) * 100,
                "lo_tail_hit_pct": lo_hit * 100, "hi_tail_hit_pct": hi_hit * 100,
                **{f"ch_{k}": float(v) for k, v in ch.items()},
            })
        for p in var_lo_ps:
            q = thresh[mname][p]
            hit = (r_true < q).astype(int)
            hr = float(np.mean(hit))
            ch = christoffersen_lr(hit, alpha_nom=p)
            rows_var.append({
                "model": mname, "tail": "lower", "nominal_p": p,
                "q_predict_EUR": q,
                "actual_hit_rate": hr, "gap_pp": (hr - p) * 100,
                **{f"ch_{k}": float(v) for k, v in ch.items()},
            })
        for qtile in var_hi_qtiles:
            q = thresh[mname][qtile]
            nom_hit = 1 - qtile
            hit = (r_true > q).astype(int)
            hr = float(np.mean(hit))
            ch = christoffersen_lr(hit, alpha_nom=nom_hit)
            rows_var.append({
                "model": mname, "tail": "upper", "nominal_p": nom_hit,
                "nominal_qtile": qtile, "q_predict_EUR": q,
                "actual_hit_rate": hr, "gap_pp": (hr - nom_hit) * 100,
                **{f"ch_{k}": float(v) for k, v in ch.items()},
            })

    df_c = pd.DataFrame(rows_cov); df_v = pd.DataFrame(rows_var)
    df_c.to_csv(OUT_COV, index=False, encoding="utf-8-sig")
    df_v.to_csv(OUT_VAR, index=False, encoding="utf-8-sig")
    L(f"Wrote {OUT_COV} ({len(df_c)} rows)")
    L(f"Wrote {OUT_VAR} ({len(df_v)} rows)")

    lines = [f"Stage 5 v2 OOS @ test 2025-01..09 CET, n={n}",
             "=" * 78,
             "INTERVAL COVERAGE (Christoffersen LR p_uc, p_cc):"]
    for _, r in df_c.iterrows():
        lines.append(
            f"  {r['model']:7s} cover={100*(1-r['alpha']):3.0f}%  "
            f"actual={100*r['actual_coverage']:5.2f}%  gap={r['coverage_gap_pp']:+.2f}pp  "
            f"lo/hi hits={r['lo_tail_hit_pct']:.2f}/{r['hi_tail_hit_pct']:.2f}%  "
            f"q=[{r['q_lo_EUR']:+.1f},{r['q_hi_EUR']:+.1f}]  "
            f"p_uc={r['ch_p_uc']:.3f} p_cc={r['ch_p_cc']:.3f}"
        )
    lines.append("-" * 78)
    lines.append("VaR TAIL HITS (gap_pp = actual − nominal):")
    for _, r in df_v.iterrows():
        np_ = r["nominal_p"]
        lines.append(
            f"  {r['model']:7s} {r['tail']:5s} nom={100*np_:.3f}%  "
            f"actual={100*r['actual_hit_rate']:.3f}%  gap={r['gap_pp']:+.3f}pp  "
            f"q={r['q_predict_EUR']:+.1f}  p_uc={r['ch_p_uc']:.3f} p_cc={r['ch_p_cc']:.3f}"
        )
    lines.append(f"Total elapsed = {time.time()-t0:.1f}s")
    with open(OUT_REP, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        L(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
