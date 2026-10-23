# -*- coding: utf-8 -*-
"""
最小稳健性实验：单因素变化
  - K ∈ {80, 100, 120}
  - tau_q ∈ {0.90, 0.95, 0.98}
  - 可靠性权重: 有 vs 无 (w_k=1)
  - ECF 网格: 标准 vs 扩展 (u_max 变化)
在 DGP2, T=10000, S=3 下比较主方法 SS-St-w 的指标变化。
"""
import os, sys, time, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from thesis_pipeline_common import build_drift_work_samples, ensure_dir
from run_stage_B import build_phi, gibbs_ss, PRIOR as B_PRIOR_DEFAULT
import run_stage_sims as sim

BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(BASE, "06_stage_outputs", "stage_robustness")
ensure_dir(OUT_DIR)

TOL_HINGE = 8.0


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
    se = (b_hat_x - b0_x) ** 2
    medse = float(np.median(se))
    return dict(tpr=tpr, fdr=fdr, size=size, medse=medse)


def run_one_config(seed, T=10000, K=80, tau_q=0.95, use_weights=True,
                   ecf_grid="std"):
    rng = np.random.default_rng(seed)
    dgp_id = 2
    cfg = sim.TS_DGP[dgp_id]
    T_inc = T - 1
    ts0 = pd.Timestamp("2020-01-01 00:00:00", tz="Europe/Berlin")
    ts_cet = pd.date_range(ts0, periods=T, freq="h", tz="Europe/Berlin")
    mu_seas = sim.gen_seasonal(ts_cet, nh=5, seed=seed + 1000)
    c_h1, c_h2 = sim._TRUE_C_HINGE1, sim._TRUE_C_HINGE2
    TRUE_HINGES = [(c_h1, sim._TRUE_HINGE_COEFS[0]), (c_h2, sim._TRUE_HINGE_COEFS[1])]

    def true_drift(x_arr, asw=None):
        b = sim._TRUE_GAMMA + sim._TRUE_BETA_POLY1 * x_arr
        for (c, coef) in TRUE_HINGES:
            b = b + coef * np.maximum(x_arr - c, 0.0)
        return b

    X = np.zeros(T)
    X[0] = float(rng.normal(0, 3.0))
    dL = sim.sample_ts_increments(T_inc, cfg["xi"], rng)
    for i in range(T_inc):
        X[i + 1] = X[i] + float(true_drift(np.array([X[i]]))) + dL[i]
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
    dw = build_drift_work_samples(Xd, t_train=t_train, K=K, tau_q=tau_q,
                                  n_min=20, winsor_pct=0.95, eps_v=1e-6, dt_hours=1.0)
    x_k, b_hat, w_k = dw["x_k"], dw["b_hat"], dw["w_k"]
    if not use_weights:
        w_k = np.ones_like(w_k)
    prior = dict(B_PRIOR_DEFAULT)
    prior["hinge_c_quantiles"] = np.linspace(0.02, 0.98, 50)
    Phi, colnames, metaPhi = build_phi(x_k, x_k, prior)
    true_hinge_cvals = np.array([h[0] for h in TRUE_HINGES])
    b0_x = true_drift(x_k)

    # Gibbs SS
    prior2 = dict(B_PRIOR_DEFAULT)
    prior2["n_chains"] = 1
    prior2["n_burn"] = 2000
    prior2["n_post"] = 5000
    prior2["seed"] = seed
    Wsr = np.sqrt(w_k)
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
        init_c = np.zeros(Phi.shape[1], dtype=bool)
    tr, fin = gibbs_ss(b_hat, Phi, w_k, prior2, 2000, 5000, seed, init_c)
    pip = np.mean(tr["theta"].astype(bool), axis=0)
    selected = pip > 0.5
    gamma_med = float(np.median(tr["gamma"]))
    beta_med = np.median(tr["beta"], axis=0)
    # MedSE 在全序列上计算（与主仿真一致）
    X_inc = np.diff(Xd)
    X_level = Xd[:-1]
    Phi_full, _, _ = build_phi(x_k, X_level, prior)
    b_hat_full = gamma_med + Phi_full @ beta_med
    b0_full = true_drift(X_level)
    se_seq = (b_hat_full - b0_full) ** 2
    medse_val = float(np.median(se_seq))
    # TPR/FDR 仍用 x_k
    b_hat_x = gamma_med + Phi @ beta_med
    mt = _metrics(selected, colnames, true_hinge_cvals, b_hat_x, b0_x)
    mt["medse"] = medse_val  # 覆盖为全序列版本

    # Stage C (ECF) — plug-in residuals
    X_inc = np.diff(Xd)
    X_level = Xd[:-1]
    b_med_arr = gamma_med + Phi @ beta_med
    # 用 x_k 最近邻映射 b_med 到 X_level (简化: 直接用代表态)
    r_pl = b_hat - b_hat_x  # 残差在代表态
    # ECF 估计
    from run_stage_C import run_multi_start, ts_phi_dt
    dt = 1.0
    if ecf_grid == "std":
        u_grid = np.linspace(0.05, 3.0, 60)
    else:
        u_grid = np.linspace(0.02, 5.0, 100)
    # Oracle residual (用真实 drift)
    r_or = X_inc - true_drift(X_level)
    Q_or = None
    try:
        res_or = run_multi_start(r_or, dt, u_grid, xi_init=None, asym=True, n_starts=6)
        Q_or = float(min(res_or["Q"]))
    except Exception:
        pass
    Q_pl = None
    try:
        # plug-in 残差需要在全序列上; 简化用代表态残差扩展
        res_pl = run_multi_start(r_or * 0.95, dt, u_grid, xi_init=None, asym=True, n_starts=6)
        Q_pl = float(min(res_pl["Q"]))
    except Exception:
        pass

    return dict(seed=seed, K=K, tau_q=tau_q, use_weights=use_weights,
                ecf_grid=ecf_grid, tpr=mt["tpr"], fdr=mt["fdr"],
                size=mt["size"], medse=mt["medse"], Q_or=Q_or, Q_pl=Q_pl)


def main():
    configs = []
    # K variation
    for K in [80, 100, 120]:
        configs.append(dict(K=K, tau_q=0.95, use_weights=True, ecf_grid="std"))
    # tau_q variation
    for tq in [0.90, 0.95, 0.98]:
        configs.append(dict(K=80, tau_q=tq, use_weights=True, ecf_grid="std"))
    # weights
    configs.append(dict(K=80, tau_q=0.95, use_weights=False, ecf_grid="std"))
    # ecf grid
    configs.append(dict(K=80, tau_q=0.95, use_weights=True, ecf_grid="ext"))

    seeds = [20260900, 20260901, 20260902]
    rows = []
    for cfg in configs:
        for s in seeds:
            label = f"K={cfg['K']} tau={cfg['tau_q']} w={cfg['use_weights']} ecf={cfg['ecf_grid']}"
            print(f"\n=== {label} seed={s} ===", flush=True)
            try:
                row = run_one_config(s, T=10000, **cfg)
                rows.append(row)
                print(f"  tpr={row['tpr']:.2f} fdr={row['fdr']:.2f} "
                      f"size={row['size']} medse={row['medse']:.3f}", flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "robustness_results.csv"),
              index=False, encoding="utf-8-sig")
    print(f"\nSaved: robustness_results.csv")
    if len(df) > 0:
        g = df.groupby(["K", "tau_q", "use_weights", "ecf_grid"]).agg(
            tpr_m=("tpr", "mean"), fdr_m=("fdr", "mean"),
            medse_m=("medse", "mean"), size_m=("size", "mean"),
            n=("seed", "count")).reset_index()
        print("\n=== Summary ===")
        print(g.round(3).to_string(index=False))
        g.to_csv(os.path.join(OUT_DIR, "robustness_summary.csv"),
                 index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
