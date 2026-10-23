# -*- coding: utf-8 -*-
"""
================================================================
 run_empirical.py — 实证章(第7章)全流程
   数据: EPEX SPOT Germany DE-LU hourly, 2019-01-01 ~ 2025-09-30 (SMARD.de)
   方法: 正文最终方法(全样本 Euler + 13 铰链 + 正交化 + 滑窗聚合 +
         剖面细化 + |t|>=2 过滤; Stage II 单尺度 ECF, W=I)
================================================================
子命令:
  stageA    数据概况表 + 全样本季节拟合
  stageB    Stage I 稀疏贝叶斯结构识别(全样本)
  stageC    Stage II ECF 点估计 + bootstrap SE(全样本, 串行 B=200)
  asymtest  非对称性检验(参数自助零分布, 进程并行)
  robust    三段稳健性(pre/crisis/post, 各自 A->B->C 点估计)
  all       stageA -> stageB -> stageC(串行后台)
输出: 硕士论文3/06_stage_outputs/empirical_v1/...
纪律: 原始数据只读; 每个中间结果落盘; 缺失不填补, 估计时剔除非有限增量并计数。
修复记录: bootstrap 结果先于绘图步骤落盘; alpha 接近下界时跳过模拟法 QQ
         (采样器在 alpha->0 时数值溢出), 残差诊断改用分段 QQ + ECF 匹配图。
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../code/pipeline
CODE = os.path.abspath(os.path.join(HERE, ".."))                  # .../code
LAB = os.path.join(CODE, "method_lab")
for p in (HERE, LAB):
    if p not in sys.path:
        sys.path.insert(0, p)

from thesis_pipeline_common import (  # noqa: E402
    load_smard_df, fit_seasonal, ensure_dir, save_pickle, load_pickle,
)

BASE3 = r"C:\Users\86183\Desktop\硕士论文3"
OUT_ROOT = ensure_dir(os.path.join(BASE3, "06_stage_outputs", "empirical_v1"))
LOG_PATH = os.path.join(OUT_ROOT, "emp_run.log")

SEED_MAIN = 20250902
N_START = 24
BOOT_B = 200
BOOT_BLOCK = 24
ASY_K = 200          # 非对称检验零分布重复数
ASY_WORKERS = 6
ALPHA_SIM_MIN = 0.05  # 低于此值 TS 增量采样器数值不稳定, 跳过模拟法 QQ


# 正文最终方法 = method_lab VARIANTS 表中经消融确认的最优配置(编号仅在代码仓出现)
def final_variant_cfg():
    import run_lab
    return dict(run_lab.VARIANTS["v13"])


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ================================================================
#  工具: 分段
# ================================================================
SEGMENTS = {
    "seg_pre":    ("2019-01-01 00:00", "2021-12-31 23:00"),
    "seg_crisis": ("2022-01-01 00:00", "2023-12-31 23:00"),
    "seg_post":   ("2024-01-01 00:00", "2025-09-30 23:00"),
}


def subset_cet(df, lo, hi):
    ts = df["ts_cet"]
    lo_t = pd.Timestamp(lo, tz=ts.dt.tz)
    hi_t = pd.Timestamp(hi, tz=ts.dt.tz)
    sub = df.loc[(ts >= lo_t) & (ts <= hi_t)].copy().reset_index(drop=True)
    return sub


# ================================================================
#  Stage A: 数据概况 + 季节拟合
# ================================================================
def run_stageA(df: pd.DataFrame, out_dir: str, tag: str):
    ensure_dir(out_dir)
    art = fit_seasonal(df, df, hod_nharm=5)
    X = art["X_all"].to_numpy(dtype=np.float64)
    P = df["price_eur_mwh_raw"].to_numpy(dtype=np.float64)

    n = len(P)
    n_na = int((~np.isfinite(P)).sum())
    Pf = P[np.isfinite(P)]
    dP = np.diff(Pf)
    sd_dP = float(np.nanstd(dP))
    rows = [
        ("n_hours", n),
        ("n_missing_price", n_na),
        ("n_negative_price", int((Pf < 0).sum())),
        ("pct_negative_price", round(100.0 * (Pf < 0).mean(), 3)),
        ("mean", float(np.mean(Pf))),
        ("sd", float(np.std(Pf))),
        ("min", float(np.min(Pf))),
        ("q005", float(np.quantile(Pf, 0.005))),
        ("q25", float(np.quantile(Pf, 0.25))),
        ("median", float(np.median(Pf))),
        ("q75", float(np.quantile(Pf, 0.75))),
        ("q995", float(np.quantile(Pf, 0.995))),
        ("max", float(np.max(Pf))),
        ("n_spike_abs_dP_gt_3sd", int((np.abs(dP) > 3 * sd_dP).sum())),
        ("seasonal_R2_train", art["R2_train"]),
        ("seasonal_R2_full", art["R2_full"]),
        ("X_min", float(np.nanmin(X))),
        ("X_max", float(np.nanmax(X))),
        ("X_n_nonfinite", int((~np.isfinite(X)).sum())),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(
        os.path.join(out_dir, f"{tag}_data_summary.csv"),
        index=False, encoding="utf-8-sig")

    df_out = df[["ts_cet", "epoch_ms_utc", "price_eur_mwh_raw"]].copy()
    df_out["mu_hat"] = art["mu_all"].to_numpy(dtype=np.float64)
    df_out["X"] = X
    df_out.to_csv(os.path.join(out_dir, f"{tag}_X_full.csv"),
                  index=False, encoding="utf-8-sig")
    save_pickle(art, os.path.join(out_dir, f"{tag}_seasonal_artifacts.pkl"))
    log(f"[A:{tag}] n={n}, neg={int((Pf<0).sum())} "
        f"({100.0*(Pf<0).mean():.2f}%), R2_full={art['R2_full']:.4f}, "
        f"X in [{np.nanmin(X):.1f}, {np.nanmax(X):.1f}]")
    return art


# ================================================================
#  Stage B: Stage I 结构识别(正文最终配置)
# ================================================================
def final_drift_fit(s1, refined_cs, nu=4.0):
    """复现 profile_refine 的最终 IRLS 拟合, 返回系数/SE/t/拟合漂移/残差。"""
    import run_lab
    y = s1["y"]; x = s1["x_level"]
    sel = s1["selected"]; hinge_map = s1["hinge_map"]
    poly_cols = [j for j in np.where(sel)[0] if j not in hinge_map]
    blocks, names = [], []
    if poly_cols:
        blocks.append(s1["Phi_level"][:, poly_cols])
        names += [s1["colnames"][j] for j in poly_cols]
    for c in refined_cs:
        blocks.append(np.maximum(x - c, 0.0).reshape(-1, 1))
        names.append(f"(x-{float(c):.2f})_+")
    Xfin = np.column_stack(blocks) if blocks else np.zeros((len(y), 0))
    if Xfin.shape[1] == 0:
        b = np.full(len(y), float(np.median(y)))
        return dict(names=["intercept"], coef=np.array([float(np.median(y))]),
                    se=np.array([np.nan]), t=np.array([np.nan]),
                    b_level=b, resid=y - b, n_poly=0)
    coef = run_lab.refit_student_t(y, Xfin, n_iter=40, nu=nu)
    r = y - coef[0] - Xfin @ coef[1:]
    s = max(np.median(np.abs(r - np.median(r))) / 0.6744897501960817, 1e-8)
    u = (nu + 1.0) / (nu + (r / s) ** 2)
    A = np.column_stack([np.ones(len(y)), Xfin])
    AtWA = A.T @ (A * u[:, None])
    try:
        covb = np.linalg.inv(AtWA + 1e-10 * np.eye(AtWA.shape[0])) * (s ** 2)
        se = np.sqrt(np.maximum(np.diag(covb), 1e-300))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.nan)
    t_stat = np.abs(coef) / se
    return dict(names=["intercept"] + names, coef=coef, se=se, t=t_stat,
                b_level=coef[0] + Xfin @ coef[1:], resid=r,
                n_poly=len(poly_cols))


def run_stageB(X: np.ndarray, out_dir: str, tag: str, seed: int = SEED_MAIN):
    import run_lab
    ensure_dir(out_dir)
    vc = final_variant_cfg()
    # 缺失处理: 不填补; 剔除非有限水平点(增量随之重算)并计数
    n_bad = int((~np.isfinite(X)).sum())
    if n_bad > 0:
        X = X[np.isfinite(X)]
        log(f"[B:{tag}] 剔除非有限水平点 {n_bad} 个(缺失不填补)")
    t0 = time.time()
    s1 = run_lab.stageI(X, vc, seed)
    log(f"[B:{tag}] Gibbs 完成, 用时 {time.time()-t0:.0f}s")

    refined_cs, b_level = run_lab.profile_refine(
        s1, margin=vc.get("refine_margin", 0.0),
        t_thr=vc.get("t_thr", 0.0), sel_ic=vc.get("sel_ic", None))
    fit = final_drift_fit(s1, refined_cs)

    pd.DataFrame({
        "candidate": s1["colnames"],
        "PIP": s1["pip"],
        "beta_post_med": s1["beta_med"],
        "selected_gibbs": s1["selected"].astype(int),
    }).to_csv(os.path.join(out_dir, f"{tag}_pip_coef.csv"),
              index=False, encoding="utf-8-sig")

    cl_rows = []
    for cl in (s1.get("clusters_sel") or []):
        cl_rows.append(dict(c_rep=cl["c_rep"], c_lo=cl["c_lo"],
                            c_hi=cl["c_hi"], cpip=cl["cpip"]))
    cl_df = pd.DataFrame(cl_rows)
    if len(cl_df) > 0:
        cl_df["c_refined"] = np.nan
        cl_df["delta_hat"] = np.nan
        cl_df["abs_t"] = np.nan
        cl_df["dropped_by_t_filter"] = 0
    # 细化折点按"窗口归属"回填到对应簇，而非按行位置：
    # profile_refine 的 t 门限会丢弃部分簇的折点，若按位置填充，
    # 幸存折点会整体左移、写进别的簇的行（簇表自相矛盾）。
    margin = float(vc.get("refine_margin", 0.0))
    used = set()
    for i_ref, c in enumerate(refined_cs):
        j = fit["n_poly"] + 1 + i_ref   # coef 中位置(含截距)
        match = None
        for i_cl, cl in enumerate(cl_rows):
            if i_cl in used:
                continue
            if (cl["c_lo"] - margin - 1e-9) <= float(c) <= (cl["c_hi"] + margin + 1e-9):
                match = i_cl
                break
        if match is None:
            free = [i for i in range(len(cl_rows)) if i not in used]
            if not free:
                continue
            match = min(free, key=lambda i: abs(cl_rows[i]["c_rep"] - float(c)))
            log(f"[B:{tag}] 警告: 细化折点 {float(c):.2f} 不在任何未分配簇窗口内,"
                f" 就近归入 c_rep={cl_rows[match]['c_rep']:.2f} 的簇")
        used.add(match)
        cl_df.loc[match, "c_refined"] = float(c)
        cl_df.loc[match, "delta_hat"] = float(fit["coef"][j])
        cl_df.loc[match, "abs_t"] = float(fit["t"][j])
    for i_cl, cl in enumerate(cl_rows):
        if i_cl not in used:
            cl_df.loc[i_cl, "dropped_by_t_filter"] = 1
            log(f"[B:{tag}] 簇 c_rep={cl['c_rep']:.2f} (cpip={cl['cpip']:.3f}) "
                f"的细化折点未通过 |t| >= {vc.get('t_thr', 0.0)} 门限, 已丢弃")
    cl_df.to_csv(os.path.join(out_dir, f"{tag}_clusters.csv"),
                 index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "term": fit["names"], "coef": fit["coef"],
        "se": fit["se"], "abs_t": fit["t"],
    }).to_csv(os.path.join(out_dir, f"{tag}_final_drift.csv"),
              index=False, encoding="utf-8-sig")

    save_pickle(dict(s1={k: v for k, v in s1.items() if k != "theta"},
                     refined_cs=np.asarray(refined_cs, float),
                     b_level=fit["b_level"], resid=fit["resid"],
                     n_dropped_increments=n_bad),
                os.path.join(out_dir, f"{tag}_stageB_artifacts.pkl"))
    log(f"[B:{tag}] 细化折点 = {np.round(refined_cs, 2).tolist()}, "
        f"簇数 = {len(s1.get('clusters_sel') or [])}")
    return s1, refined_cs, fit


# ================================================================
#  Stage C: ECF 点估计 + bootstrap SE
# ================================================================
def ecf_point_fits(resid: np.ndarray, seed: int = 90201):
    from run_stage_C import build_ecf_data, run_multi_start
    r = resid[np.isfinite(resid)]
    dat = build_ecf_data(r)
    res_asy = run_multi_start(dat, symmetric=False, n_start=N_START, seed=seed)
    res_sym = run_multi_start(dat, symmetric=True, n_start=N_START, seed=seed + 1)
    return dat, res_asy, res_sym


def run_stageC(resid: np.ndarray, out_dir: str, tag: str,
               do_bootstrap: bool = True):
    from run_stage_C import bootstrap_ecf_se, ts_phi_dt
    ensure_dir(out_dir)
    r = resid[np.isfinite(resid)]
    t0 = time.time()
    dat, res_asy, res_sym = ecf_point_fits(r)
    bestA, bestS = res_asy[0], res_sym[0]
    log(f"[C:{tag}] 点估计完成 {time.time()-t0:.0f}s, "
        f"Q_asy={bestA['Q_final']:.6f}, Q_sym={bestS['Q_final']:.6f}, "
        f"dQ={bestS['Q_final']-bestA['Q_final']:.6f}")

    bs_out = None
    if do_bootstrap:
        t1 = time.time()
        bs_out = bootstrap_ecf_se(
            dat, r, xi_hat_asy=bestA["xi_final"], xi_hat_sym=bestS["xi_final"],
            B=BOOT_B, n_start=4, block_len=BOOT_BLOCK, seed=20260903,
            L=lambda m: log(f"[C:{tag}] {m}"))
        log(f"[C:{tag}] bootstrap 完成 {time.time()-t1:.0f}s")

    # 关键结果先落盘(含 bootstrap), 再做绘图类附产物
    save_pickle(dict(best_asym=bestA, best_sym=bestS,
                     all_starts_asym=res_asy, all_starts_sym=res_sym,
                     bootstrap=bs_out, sR=dat.sR, n_resid=len(r)),
                os.path.join(out_dir, f"{tag}_stageC_artifacts.pkl"))

    se_a = bs_out["se_asy"] if bs_out else np.full(6, np.nan)
    se_s = bs_out["se_sym_rep"] if bs_out else np.full(6, np.nan)
    pd.DataFrame({
        "parameter": ["mu", "C_plus", "C_minus", "M", "G", "alpha"],
        "est_asym": bestA["xi_final"], "se_asym": se_a,
        "est_sym": [bestS["xi_final"][0], bestS["xi_final"][1],
                    bestS["xi_final"][1], bestS["xi_final"][2],
                    bestS["xi_final"][2], bestS["xi_final"][3]],
        "se_sym": se_s,
    }).to_csv(os.path.join(out_dir, f"{tag}_estimates.csv"),
              index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "model": ["asym", "sym"],
        "Q_min": [bestA["Q_final"], bestS["Q_final"]],
        "n_start": [len(res_asy), len(res_sym)],
        "converged": [sum(1 for z in res_asy if z["converged"]),
                      sum(1 for z in res_sym if z["converged"])],
        "bound_hits": ["/".join(bestA["bound_hits"]) or "none",
                       "/".join(bestS["bound_hits"]) or "none"],
    }).to_csv(os.path.join(out_dir, f"{tag}_objective.csv"),
              index=False, encoding="utf-8-sig")

    # ECF 匹配网格(图7.3 数据源)
    phi_a = ts_phi_dt(dat.u, bestA["xi_final"])
    phi_s = ts_phi_dt(dat.u, np.array([bestS["xi_final"][0], bestS["xi_final"][1],
                                       bestS["xi_final"][1], bestS["xi_final"][2],
                                       bestS["xi_final"][2], bestS["xi_final"][3]]))
    pd.DataFrame({
        "u_j": dat.u, "c_j": dat.c_grid,
        "cos_emp": dat.cosR, "sin_emp": dat.sinR,
        "cos_asym": phi_a.real, "sin_asym": phi_a.imag,
        "cos_sym": phi_s.real, "sin_sym": phi_s.imag,
    }).to_csv(os.path.join(out_dir, f"{tag}_ecf_fit_grid.csv"),
              index=False, encoding="utf-8-sig")

    # 残差 QQ(模拟法): alpha 过低时采样器数值溢出, 跳过并记录
    alpha_hat = float(bestA["xi_final"][5])
    if alpha_hat >= ALPHA_SIM_MIN:
        from run_stage_sims import sample_ts_increments
        rng = np.random.default_rng(20261015)
        sim_inc = sample_ts_increments(min(len(r), 200000),
                                       tuple(bestA["xi_final"]), rng)
        sim_inc = sim_inc[np.isfinite(sim_inc)]
        qs = np.linspace(0.001, 0.999, 999)
        pd.DataFrame({
            "q": qs,
            "resid_quantile": np.quantile(r, qs),
            "model_quantile": np.quantile(sim_inc, qs),
        }).to_csv(os.path.join(out_dir, f"{tag}_resid_qq.csv"),
                  index=False, encoding="utf-8-sig")
        log(f"[C:{tag}] QQ(模拟法) 完成, alpha={alpha_hat:.3f}")
    else:
        log(f"[C:{tag}] alpha={alpha_hat:.4f} < {ALPHA_SIM_MIN}, "
            f"采样器数值不稳定, 跳过模拟法 QQ(残差诊断见分段 QQ 与 ECF 匹配图)")

    def acf(v, nlag=48):
        v = v - np.mean(v)
        d = np.sum(v ** 2)
        return np.array([1.0] + [float(np.sum(v[k:] * v[:-k]) / d)
                                 for k in range(1, nlag + 1)])
    pd.DataFrame({"lag": np.arange(49),
                  "acf_resid": acf(r), "acf_resid_sq": acf(r ** 2)}
                 ).to_csv(os.path.join(out_dir, f"{tag}_resid_acf.csv"),
                          index=False, encoding="utf-8-sig")
    return dat, bestA, bestS


# ================================================================
#  非对称性检验: 参数自助零分布(并行)
# ================================================================
def _null_rep(args):
    k, n, xi_sym6, xi_asy6, seed = args
    import sys as _s
    for p in (HERE, LAB):
        if p not in _s.path:
            _s.path.insert(0, p)
    import numpy as _np
    from scipy import optimize as _opt
    from run_stage_sims import sample_ts_increments
    from run_stage_C import (build_ecf_data, compute_Q, to_unconstrained,
                             to_constrained, starting_points_lhs,
                             _ALPHA_LB, _ALPHA_UB)
    rng = _np.random.default_rng(seed + k)
    r_star = sample_ts_increments(n, tuple(xi_sym6), rng)
    r_star = r_star[_np.isfinite(r_star)]
    dat = build_ecf_data(r_star)

    def quick_fit(xi_warm, symmetric):
        starts = []
        for _ in range(3):
            s = _np.asarray(xi_warm, float).copy()
            s[0] = xi_warm[0] + rng.normal(0, max(0.5, abs(xi_warm[0]) * 0.1))
            for j in range(1, len(s) - 1):
                s[j] = max(1e-8, xi_warm[j] * (10 ** rng.uniform(-0.3, 0.3)))
            s[-1] = float(_np.clip(xi_warm[-1] + rng.normal(0, 0.05),
                                   _ALPHA_LB + 1e-2, _ALPHA_UB - 1e-2))
            starts.append(s)
        starts += list(starting_points_lhs(3, symmetric, dat,
                                           seed=int(seed + k * 7)))
        best_q, best_xi = _np.inf, None
        for xi0 in starts:
            try:
                u0 = to_unconstrained(xi0, symmetric)
                r1 = _opt.minimize(
                    lambda u: compute_Q(to_constrained(u, symmetric), dat, symmetric),
                    u0, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-8})
                xiA = to_constrained(_np.asarray(r1.x, float), symmetric)
                if symmetric:
                    bnd = [(None, None), (1e-8, None), (1e-8, None),
                           (_ALPHA_LB, _ALPHA_UB)]
                else:
                    bnd = [(None, None), (1e-8, None), (1e-8, None),
                           (1e-8, None), (1e-8, None), (_ALPHA_LB, _ALPHA_UB)]
                r2 = _opt.minimize(
                    lambda xi: compute_Q(xi, dat, symmetric),
                    xiA, method="Nelder-Mead", bounds=bnd,
                    options={"maxiter": 1200, "fatol": 1e-8, "xatol": 1e-8})
                if float(r2.fun) < best_q:
                    best_q, best_xi = float(r2.fun), _np.asarray(r2.x, float)
            except Exception:
                continue
        return best_q, best_xi

    q_s, xi_s = quick_fit(xi_sym6, True)
    q_a, xi_a = quick_fit(xi_asy6, False)
    return (k, q_s, q_a, q_s - q_a)


def run_asymtest(out_dir: str):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    full_dir = os.path.join(OUT_ROOT, "full")
    art = load_pickle(os.path.join(full_dir, "full_stageC_artifacts.pkl"))
    artB = load_pickle(os.path.join(full_dir, "full_stageB_artifacts.pkl"))
    resid = artB["resid"]
    r = resid[np.isfinite(resid)]
    n = len(r)
    xi_a = np.asarray(art["best_asym"]["xi_final"], float)
    xi_s4 = np.asarray(art["best_sym"]["xi_final"], float)
    xi_s6 = np.array([xi_s4[0], xi_s4[1], xi_s4[1], xi_s4[2], xi_s4[2], xi_s4[3]])
    dq_obs = float(art["best_sym"]["Q_final"] - art["best_asym"]["Q_final"])
    log(f"[ASY] dQ_obs = {dq_obs:.6f}, n = {n}, K = {ASY_K}")

    null_csv = os.path.join(out_dir, "asymtest_null.csv")
    done = set()
    if os.path.exists(null_csv):
        prev = pd.read_csv(null_csv)
        done = set(prev["k"].tolist())
    tasks = [(k, n, xi_s6, xi_a, 770000) for k in range(ASY_K) if k not in done]
    log(f"[ASY] 剩余 {len(tasks)}/{ASY_K} 个零分布重复")

    t0 = time.time()
    with open(null_csv, "a", encoding="utf-8", newline="") as f:
        if not done:
            f.write("k,q_sym,q_asym,dq\n")
        with ProcessPoolExecutor(max_workers=ASY_WORKERS) as ex:
            futs = {ex.submit(_null_rep, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futs)):
                k, q_s, q_a, dq = fut.result()
                f.write(f"{k},{q_s:.10f},{q_a:.10f},{dq:.10f}\n")
                f.flush(); os.fsync(f.fileno())
                if (i + 1) % 20 == 0:
                    log(f"[ASY] {i+1}/{len(tasks)} 完成, "
                        f"用时 {(time.time()-t0)/60:.1f}min")

    res = pd.read_csv(null_csv)["dq"].to_numpy(dtype=float)
    res = res[np.isfinite(res)]
    p_val = (1.0 + float(np.sum(res >= dq_obs))) / (len(res) + 1.0)
    pd.DataFrame({
        "metric": ["dQ_obs", "K_valid", "null_dQ_mean", "null_dQ_sd",
                   "null_dQ_q95", "p_value"],
        "value": [dq_obs, len(res), float(np.mean(res)), float(np.std(res)),
                  float(np.quantile(res, 0.95)), p_val],
    }).to_csv(os.path.join(out_dir, "asymtest_summary.csv"),
              index=False, encoding="utf-8-sig")
    log(f"[ASY] p = {p_val:.4f} (K_valid={len(res)})")


# ================================================================
#  三段稳健性
# ================================================================
def run_robust(df: pd.DataFrame):
    rows = []
    for seg, (lo, hi) in SEGMENTS.items():
        out_dir = ensure_dir(os.path.join(OUT_ROOT, seg))
        sub = subset_cet(df, lo, hi)
        log(f"[R:{seg}] {lo} ~ {hi}, n = {len(sub)}")
        art = run_stageA(sub, out_dir, seg)
        X = art["X_all"].to_numpy(dtype=np.float64)
        s1, refined_cs, fit = run_stageB(X, out_dir, seg,
                                         seed=SEED_MAIN + 100)
        dat, bestA, bestS = run_stageC(fit["resid"], out_dir, seg,
                                       do_bootstrap=False)
        rows.append(dict(
            segment=seg, n_hours=len(sub),
            knots="; ".join(f"{c:.2f}" for c in refined_cs),
            n_knots=len(refined_cs),
            mu=bestA["xi_final"][0], C_plus=bestA["xi_final"][1],
            C_minus=bestA["xi_final"][2], M=bestA["xi_final"][3],
            G=bestA["xi_final"][4], alpha=bestA["xi_final"][5],
            Q_asym=bestA["Q_final"], Q_sym=bestS["Q_final"],
        ))
        pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT, "empR_segments.csv"),
                                  index=False, encoding="utf-8-sig")
    log("[R] 三段稳健性完成")


# ================================================================
#  主入口
# ================================================================
def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stageA", "stageB", "stageC",
                                    "asymtest", "robust", "all"])
    args = ap.parse_args()
    t_start = time.time()
    log(f"=== run_empirical {args.cmd} START ===")

    full_dir = ensure_dir(os.path.join(OUT_ROOT, "full"))

    if args.cmd in ("stageA", "all"):
        df = load_smard_df()
        run_stageA(df, full_dir, "full")

    if args.cmd in ("stageB", "all"):
        dfX = pd.read_csv(os.path.join(full_dir, "full_X_full.csv"),
                          encoding="utf-8-sig")
        X = dfX["X"].to_numpy(dtype=np.float64)
        run_stageB(X, full_dir, "full")

    if args.cmd in ("stageC", "all"):
        artB = load_pickle(os.path.join(full_dir, "full_stageB_artifacts.pkl"))
        run_stageC(artB["resid"], full_dir, "full", do_bootstrap=True)

    if args.cmd == "asymtest":
        run_asymtest(full_dir)

    if args.cmd == "robust":
        df = load_smard_df()
        run_robust(df)

    log(f"=== run_empirical {args.cmd} DONE "
        f"({(time.time()-t_start)/60:.1f} min) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
