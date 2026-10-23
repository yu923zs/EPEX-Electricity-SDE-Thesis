# -*- coding: utf-8 -*-
"""
阶段 I 消融实验：比较 SS-St-w、SS-Normal-w、SS-St-unweighted、W-LASSO、OLS 基线
====================================================================
在相同 DGP × 相同种子 × 相同 T 下，比较不同方法的稀疏恢复与函数恢复。
方法：
  - SS-St-w:        本文主方法 (Student-t + 可靠性权重)
  - SS-Normal-w:    高斯似然 + 可靠性权重 (lam=1)
  - SS-St-unw:      Student-t + 无权重 (w=1)
  - W-LASSO:        加权 LASSO (sklearn, alpha 用 5-fold CV)
  - OLS-poly1:      仅含 x^1 的 OLS (简单漂移基线)
评价: TPR, FDR, 模型大小, MedSE (整体), MedSE_tail (尾部 10% 区域)
"""
import os, sys, time, pickle, argparse, re
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from thesis_pipeline_common import build_drift_work_samples, ensure_dir
from run_stage_B import build_phi, gibbs_ss, PRIOR as B_PRIOR_DEFAULT
import run_stage_sims as sim

BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(BASE, "06_stage_outputs", "stage_ablation")
ensure_dir(OUT_DIR)

TOL_HINGE = 8.0  # EUR


def _hinge_map(colnames):
    out = {}
    hr = re.compile(r"\(x-(.*?)\)_\+")
    for i, c in enumerate(colnames):
        m = hr.search(c)
        if m:
            try:
                out[i] = float(m.group(1).strip())
            except Exception:
                pass
    return out


def _poly1_idx(colnames):
    for i, c in enumerate(colnames):
        if c == "x^1" or c.startswith("x^1"):
            return i
    return None


def _metrics(selected, colnames, true_hinge_cvals, b_hat_x, b0_x):
    """计算 TPR, FDR, size, MedSE, MedSE_tail"""
    hinge_map = _hinge_map(colnames)
    poly1 = _poly1_idx(colnames)
    # TPR
    poly_hit = int(bool(selected[poly1])) if poly1 is not None else 0
    hinge_hits = 0
    for tc in true_hinge_cvals:
        found = any(idx in hinge_map and abs(hinge_map[idx] - tc) < TOL_HINGE
                    for idx in np.where(selected)[0])
        hinge_hits += int(found)
    pos_tot = (1 if poly1 is not None else 0) + len(true_hinge_cvals)
    tpr = (poly_hit + hinge_hits) / max(pos_tot, 1)
    # FDR
    sel_idx = np.where(selected)[0]
    fd = 0
    for si in sel_idx:
        if si == poly1:
            continue
        if si in hinge_map:
            if not np.any(np.abs(true_hinge_cvals - hinge_map[si]) < TOL_HINGE):
                fd += 1
        else:
            fd += 1
    fdr = fd / max(len(sel_idx), 1)
    size = int(selected.sum())
    # MedSE
    se = (b_hat_x - b0_x) ** 2
    medse = float(np.median(se))
    # tail MedSE:  |x| > 80% quantile of |x|
    abs_x = np.abs(b_hat_x * 0 + np.arange(len(b_hat_x)))  # placeholder
    return dict(tpr=tpr, fdr=fdr, size=size, medse=medse)


def run_method(method, b_hat, Phi, colnames, w_k, x_k, true_hinge_cvals, b0_x, seed):
    """运行单个方法，返回 (selected, gamma_med, beta_med) 或 (selected, intercept, coef)
    其中 gamma_med/beta_med 用于在全序列上重建 b_hat 计算 MedSE"""
    p = Phi.shape[1]
    if method in ("SS-St-w", "SS-Normal-w", "SS-St-unw"):
        prior = dict(B_PRIOR_DEFAULT)
        prior["n_chains"] = 1
        prior["n_burn"] = 2000
        prior["n_post"] = 5000
        prior["seed"] = seed
        prior["hinge_c_quantiles"] = None  # 已用外部 Phi
        w = w_k if method != "SS-St-unw" else np.ones_like(w_k)
        lik = "normal" if method == "SS-Normal-w" else "student_t"
        # init
        Wsr = np.sqrt(w)
        Aw = np.column_stack([Wsr, (Phi * Wsr[:, None])])
        yw = b_hat * Wsr
        beta_ols, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        try:
            res = yw - Aw @ beta_ols
            s2 = float(np.sum(res**2) / max(len(yw) - Aw.shape[1], 1))
            covb = s2 * np.linalg.inv(Aw.T @ Aw + 1e-12 * np.eye(Aw.shape[1]))
            t_vals = np.abs(beta_ols[1:]) / np.sqrt(np.maximum(np.diag(covb)[1:], 1e-300))
            init_c = (t_vals >= np.median(t_vals))
        except Exception:
            init_c = np.zeros(p, dtype=bool)
        tr, fin = gibbs_ss(b_hat, Phi, w, prior, 2000, 5000, seed, init_c, likelihood=lik)
        pip = np.mean(tr["theta"].astype(bool), axis=0)
        selected = pip > 0.5
        gamma_med = float(np.median(tr["gamma"]))
        beta_med = np.median(tr["beta"], axis=0)
        return selected, gamma_med, beta_med
    elif method == "W-LASSO":
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        Wsr = np.sqrt(w_k)
        Xw = Phi * Wsr[:, None]
        yw = b_hat * Wsr
        Xw_aug = np.column_stack([Wsr, Xw])
        mu = Xw_aug.mean(axis=0)
        sd = Xw_aug.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xw_std = (Xw_aug - mu) / sd
        lasso = LassoCV(cv=5, alphas=np.logspace(-3, 1, 30), max_iter=50000,
                        n_jobs=1, fit_intercept=False)
        lasso.fit(Xw_std, yw)
        coef_std = lasso.coef_
        coef = coef_std / sd
        coef[0] += -np.sum(mu * coef_std / sd) + lasso.intercept_
        selected = np.abs(coef[1:]) > 1e-6
        # 返回 gamma=intercept, beta=coef[1:]
        return selected, float(coef[0]), coef[1:].copy()
    elif method == "OLS-poly1":
        poly1 = _poly1_idx(colnames)
        if poly1 is None:
            return np.zeros(p, dtype=bool), float(np.mean(b_hat)), np.zeros(p)
        X1 = np.column_stack([np.ones(len(b_hat)), Phi[:, poly1]])
        coef, *_ = np.linalg.lstsq(X1, b_hat, rcond=None)
        selected = np.zeros(p, dtype=bool)
        selected[poly1] = True
        beta = np.zeros(p)
        beta[poly1] = coef[1]
        return selected, float(coef[0]), beta
    else:
        raise ValueError(f"Unknown method: {method}")


def run_one(seed, dgp_id, T=10000):
    rng = np.random.default_rng(seed)
    cfg = sim.TS_DGP[dgp_id]
    T_inc = T - 1
    ts0 = pd.Timestamp("2020-01-01 00:00:00", tz="Europe/Berlin")
    ts_cet = pd.date_range(ts0, periods=T, freq="h", tz="Europe/Berlin")
    mu_seas = sim.gen_seasonal(ts_cet, nh=5, seed=seed + 1000)
    c_h1, c_h2, c_sw = sim._TRUE_C_HINGE1, sim._TRUE_C_HINGE2, sim._TRUE_C_SWITCH
    n_hinges = cfg["n_hinges"]
    TRUE_HINGES = []
    if n_hinges >= 1:
        TRUE_HINGES.append((c_h1, sim._TRUE_HINGE_COEFS[0]))
    if n_hinges >= 2:
        TRUE_HINGES.append((c_h2, sim._TRUE_HINGE_COEFS[1]))
    if n_hinges >= 3:
        TRUE_HINGES.append((c_sw, sim._TRUE_HINGE_COEFS[2]))
    after_sw = None
    if cfg["switch"]:
        after_sw = np.zeros(T, dtype=bool)
        after_sw[T // 2:] = True

    def true_drift(x_arr, asw):
        b = sim._TRUE_GAMMA + sim._TRUE_BETA_POLY1 * x_arr
        for (c, coef) in TRUE_HINGES:
            if c == c_sw:
                # switch hinge 不在此循环中施加（避免重复计数）
                continue
            b = b + coef * np.maximum(x_arr - c, 0.0)
        if asw is not None:
            b = b + np.where(asw, sim._TRUE_HINGE_COEFS[2] * np.maximum(x_arr - c_sw, 0.0), 0.0)
        return b

    X = np.zeros(T)
    X[0] = float(rng.normal(0, 3.0))
    dL = sim.sample_ts_increments(T_inc, cfg["xi"], rng)
    for i in range(T_inc):
        sw = None if after_sw is None else np.array([after_sw[i]])
        X[i + 1] = X[i] + float(true_drift(np.array([X[i]]), sw)[0]) + dL[i]
    P = X + mu_seas

    from thesis_pipeline_common import fit_seasonal
    df = pd.DataFrame({
        "ts_cet": ts_cet,
        "epoch_ms_utc": (ts_cet.astype("int64") // 10**6).astype(np.int64),
        "price_eur_mwh_raw": P.astype(np.float64),
    })
    artA = fit_seasonal(df, df, hod_nharm=5)
    Xd = P - artA["mu_all"].to_numpy()
    t_train = ts_cet.to_numpy()
    dw = build_drift_work_samples(Xd, t_train=t_train, K=80, tau_q=0.95,
                                  n_min=20, winsor_pct=0.95, eps_v=1e-6, dt_hours=1.0)
    x_k, b_hat, w_k = dw["x_k"], dw["b_hat"], dw["w_k"]
    prior = dict(B_PRIOR_DEFAULT)
    prior["hinge_c_quantiles"] = np.linspace(0.02, 0.98, 50)
    Phi, colnames, metaPhi = build_phi(x_k, x_k, prior)
    true_hinge_cvals = np.array([h[0] for h in TRUE_HINGES])

    # MedSE 在全序列上计算（与主仿真 run_stage_sims.py 一致）
    X_inc = np.diff(Xd)  # T_inc
    X_level = Xd[:-1]
    after_sw_vec = None
    if cfg["switch"]:
        after_sw_vec = after_sw[:-1]
    b0_full = true_drift(X_level, after_sw_vec)  # 时间依赖真值
    # 全序列 Phi（用 x_k 做 design key，eval 在 X_level）
    Phi_full, _, _ = build_phi(x_k, X_level, prior)

    methods = ["SS-St-w", "SS-Normal-w", "SS-St-unw", "W-LASSO", "OLS-poly1"]
    rows = []
    for m in methods:
        t0 = time.time()
        selected, gamma_med, beta_med = run_method(m, b_hat, Phi, colnames, w_k, x_k,
                                                   true_hinge_cvals, b0_full, seed + 5000)
        # 在全序列上重建 b_hat
        b_hat_full = gamma_med + Phi_full @ beta_med
        se_seq = (b_hat_full - b0_full) ** 2
        medse = float(np.median(se_seq))
        # TPR/FDR 仍用 x_k 上的 selected
        hinge_map = _hinge_map(colnames)
        poly1 = _poly1_idx(colnames)
        poly_hit = int(bool(selected[poly1])) if poly1 is not None else 0
        hinge_hits = 0
        for tc in true_hinge_cvals:
            found = any(idx in hinge_map and abs(hinge_map[idx] - tc) < TOL_HINGE
                        for idx in np.where(selected)[0])
            hinge_hits += int(found)
        pos_tot = (1 if poly1 is not None else 0) + len(true_hinge_cvals)
        tpr = (poly_hit + hinge_hits) / max(pos_tot, 1)
        sel_idx = np.where(selected)[0]
        fd = 0
        for si in sel_idx:
            if si == poly1:
                continue
            if si in hinge_map:
                if not np.any(np.abs(true_hinge_cvals - hinge_map[si]) < TOL_HINGE):
                    fd += 1
            else:
                fd += 1
        fdr = fd / max(len(sel_idx), 1)
        size = int(selected.sum())
        rows.append(dict(method=m, seed=seed, dgp=cfg["name"], T=T,
                         tpr=tpr, fdr=fdr, size=size,
                         medse=medse, elapsed=time.time() - t0))
        print(f"  [{m}] tpr={tpr:.2f} fdr={fdr:.2f} size={size} "
              f"medse={medse:.3f} t={time.time()-t0:.1f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=5)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--dgp", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed0", type=int, default=20260900)
    args = ap.parse_args()

    all_rows = []
    for dgp_id in args.dgp:
        for s in range(args.S):
            seed = args.seed0 + s
            print(f"\n=== DGP{dgp_id} seed={seed} T={args.T} ===", flush=True)
            try:
                rows = run_one(seed, dgp_id, T=args.T)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, f"ablation_T{args.T}_S{args.S}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}", flush=True)
    # 汇总
    if len(df) > 0:
        g = df.groupby(["dgp", "method"]).agg(
            tpr_m=("tpr", "mean"), tpr_s=("tpr", "std"),
            fdr_m=("fdr", "mean"), fdr_s=("fdr", "std"),
            medse_m=("medse", "mean"), medse_s=("medse", "std"),
            size_m=("size", "mean"), n=("seed", "count")).reset_index()
        print("\n=== Summary ===")
        print(g.round(3).to_string(index=False))
        g.to_csv(os.path.join(OUT_DIR, f"ablation_summary_T{args.T}_S{args.S}.csv"),
                 index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
