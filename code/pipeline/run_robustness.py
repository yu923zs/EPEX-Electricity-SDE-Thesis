# -*- coding: utf-8 -*-
"""
最小稳健性实验：单因素变化
  - K ∈ {80, 100, 120}
  - tau_q ∈ {0.90, 0.95, 0.98}
  - 可靠性权重: 有 vs 无 (w_k=1)
  - ECF 网格: 标准 vs 扩展 (u_max 变化)
在 DGP2, T=10000, S=3 下比较主方法 SS-St-w 的指标变化。

FIX(S7) 2026-10-11（独立版本目录，不覆盖原结果）:
  ① 修复 ECF 调用签名错误（原 run_multi_start(r, dt, u_grid, ...) → TypeError 被吞，
     Q_or/Q_pl 全空）→ 改为 run_multi_start(ECFData, symmetric, n_start, seed)；
  ② 修复 plug-in 残差误用 r_or*0.95 → 改为 X_inc − b_hat_full（后验中位漂移全序列）；
  ③ 本次仅重跑 ecf_grid ∈ {std, ext} 两配置，输出至 06_stage_outputs_fix_20261011。
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
# FIX(S7): 输出重定向到独立版本目录，不覆盖原 06_stage_outputs/stage_robustness
OUT_DIR = os.path.join(BASE, "06_stage_outputs_fix_20261011", "stage_robustness_ecf")
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
    # FIX(S7): ① 残差按全序列构造 r_i = ΔX_i − b(X_{t_i})Δt（与 run_stage_C 同式，Δt=1）；
    #          ② run_multi_start 使用正确签名 (dat: ECFData, symmetric, n_start, seed)，
    #             返回按 Q_final 升序的 list[dict]，最优为 res[0]；
    #          ③ plug-in 残差 = X_inc − b_hat_full（Stage B 后验中位漂移全序列），
    #             不再误用 r_or*0.95。
    # Oracle residual（仿真场景真实 drift 已知）
    r_or = X_inc - true_drift(X_level)
    # Plug-in residual（b_hat_full 在上方 MedSE 计算处已定义：gamma_med + Phi_full @ beta_med）
    r_pl = X_inc - b_hat_full
    # ECF 估计：u 直接以 u-space 网格给定（std: u∈[0.05,3.0] J=60; ext: u∈[0.02,5.0] J=100）
    from run_stage_C import run_multi_start, ECFData
    if ecf_grid == "std":
        u_grid = np.linspace(0.05, 3.0, 60)
    else:
        u_grid = np.linspace(0.02, 5.0, 100)

    def _make_dat(r: np.ndarray) -> ECFData:
        # 与 build_ecf_data 同构，但 u 由外部网格直接给定（u-space 网格稳健性维度）
        rr = np.asarray(r, dtype=np.float64).ravel()
        rr = rr[np.isfinite(rr)]
        mad = float(np.median(np.abs(rr - np.median(rr))))
        sR = max(mad / 0.6744897501960817, 1e-12)
        ur = np.outer(rr, u_grid)
        return ECFData(r=rr, sR=sR, u=u_grid, c_grid=u_grid,
                       cosR=np.mean(np.cos(ur), axis=0),
                       sinR=np.mean(np.sin(ur), axis=0), nR=len(rr))

    Q_or = None; xi_or = None
    try:
        res_or = run_multi_start(_make_dat(r_or), symmetric=False, n_start=6, seed=seed)
        Q_or = float(res_or[0]["Q_final"]); xi_or = res_or[0]["xi_final"]
    except Exception:
        pass
    Q_pl = None; xi_pl = None
    try:
        res_pl = run_multi_start(_make_dat(r_pl), symmetric=False, n_start=6, seed=seed)
        Q_pl = float(res_pl[0]["Q_final"]); xi_pl = res_pl[0]["xi_final"]
    except Exception:
        pass

    return dict(seed=seed, K=K, tau_q=tau_q, use_weights=use_weights,
                ecf_grid=ecf_grid, tpr=mt["tpr"], fdr=mt["fdr"],
                size=mt["size"], medse=mt["medse"], Q_or=Q_or, Q_pl=Q_pl,
                xi_or=str(xi_or), xi_pl=str(xi_pl))


def main():
    # FIX(S7): 本次重跑仅针对 ECF 网格维度（std vs ext，对应论文表 6.5/§6.7）；
    #          其余维度（K / tau_q / 权重）不受 S7 影响，沿用原结果不重跑。
    configs = [
        dict(K=80, tau_q=0.95, use_weights=True, ecf_grid="std"),
        dict(K=80, tau_q=0.95, use_weights=True, ecf_grid="ext"),
    ]

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
