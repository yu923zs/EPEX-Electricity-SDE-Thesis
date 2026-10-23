# -*- coding: utf-8 -*-
"""
================================================================
 method_lab/run_lab.py — 方法改进迭代实验台（小规模、固定种子、全真实运行）
================================================================
目的：在固定数据缓存上对比 Stage I（稀疏漂移识别）与 Stage II（TS-ECF
      参数估计）的逐项改进，所有指标均来自真实运行，不允许任何编造。

DGP（小时 Δt=1h，与论文第 6 章同源，复用 run_stage_sims 的采样器与常数）：
  A: b(x)=0.8-0.05x                      + 对称 TS (C=1.5,M=G=0.03,α=0.75,μ=0)
  B: b(x)=0.8-0.05x-0.06(x+14.94)_+       + 非对称 TS (μ=0.6,C+=0.9,C-=1.8,
                                                M=0.02,G=0.06,α=0.60)
  C: b(x)=B + 0.07(x-28.31)_+             + 同 B 的非对称 TS

变体（一次只改一个组件，收益可归因）：
  v0: 基线 = 分箱(K=80)+50 hinge+逐点 PIP>0.5（复现论文 Stage I 现状）
  v1: 分箱 + 13 hinge（剪候选库，降共线）
  v2: 全样本 Euler Student-t 回归（去掉分箱伪观测）+13 hinge
  v3: v2 + 邻域聚合选择（hinge 簇 PIP = P(簇内任一 θ=1)，抗 PIP 稀释）
  v4: v3 + 选中集 Student-t IRLS 去偏重估（降低 plug-in 漂移收缩偏差）
  v5: v4 + 多尺度 ECF（1h + 24h 重叠聚合残差联合匹配，W=I）
指标：TPR/FDR/F1/|Ŝ|/折点 Wasserstein/MedSE；Stage II：ASY 6 参 bias/RMSE。
"""
import os, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "代码_fix_20261011"))
for p in (HERE, CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

from thesis_pipeline_common import fit_seasonal, build_drift_work_samples, ensure_dir
from run_stage_sims import sample_ts_increments, gen_seasonal
from run_stage_B import build_phi, gibbs_ss, PRIOR as B_PRIOR_DEFAULT
from run_stage_C import (build_ecf_data, compute_Q, run_multi_start,
                         to_unconstrained, to_constrained, starting_points_lhs,
                         _ALPHA_LB, _ALPHA_UB)
from scipy import optimize
from scipy.stats import wasserstein_distance

BASE = os.path.abspath(os.path.join(HERE, ".."))
CACHE_DIR = ensure_dir(os.path.join(HERE, "data_cache"))
RES_DIR = ensure_dir(os.path.join(HERE, "results"))

# ---------------- DGP 定义 ----------------
GAMMA0, BETA1 = 0.8, -0.05
DGP_LAB = {
    "A": dict(xi=(0.0, 1.5, 1.5, 0.03, 0.03, 0.75),
              hinges=[], name="A_linear_symTS"),
    "B": dict(xi=(0.6, 0.9, 1.8, 0.02, 0.06, 0.60),
              hinges=[(-14.94, -0.06)], name="B_1knot_asymTS"),
    "C": dict(xi=(0.6, 0.9, 1.8, 0.02, 0.06, 0.60),
              hinges=[(-14.94, -0.06), (28.31, 0.07)], name="C_2knot_asymTS"),
}
TRUE_POLY1 = True  # x^1 为真项（β1≠0），x^2,x^3 为假项


def true_drift_arr(x, dgp):
    b = GAMMA0 + BETA1 * x
    for (c, coef) in DGP_LAB[dgp]["hinges"]:
        b = b + coef * np.maximum(x - c, 0.0)
    return b


# ---------------- 数据生成（带磁盘缓存，保证各变体配对比较） ----------------
def gen_data(seed, dgp, T):
    f = os.path.join(CACHE_DIR, f"data_{dgp}_T{T}_seed{seed}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["X_deseas"], z["X_true"]
    rng = np.random.default_rng(seed)
    cfg = DGP_LAB[dgp]
    ts0 = pd.Timestamp("2020-01-01 00:00:00", tz="Europe/Berlin")
    ts_cet = pd.date_range(ts0, periods=T, freq="h", tz="Europe/Berlin")
    mu_seas = gen_seasonal(ts_cet, nh=5, seed=seed + 1000)
    X = np.zeros(T)
    X[0] = float(rng.normal(0, 3.0))
    dL = sample_ts_increments(T - 1, cfg["xi"], rng)
    for i in range(T - 1):
        X[i + 1] = X[i] + float(true_drift_arr(np.array([X[i]]), dgp)[0]) + dL[i]
    P = X + mu_seas
    df = pd.DataFrame({
        "ts_cet": ts_cet,
        "epoch_ms_utc": (ts_cet.astype("int64") // 10**6).astype(np.int64),
        "price_eur_mwh_raw": P.astype(np.float64)})
    artA = fit_seasonal(df, df, hod_nharm=5)
    X_deseas = P - artA["mu_all"].to_numpy(dtype=np.float64)
    np.savez_compressed(f, X_deseas=X_deseas, X_true=X)
    return X_deseas, X


# ---------------- Stage I ----------------
def run_gibbs(y, Phi, w, seed, n_burn, n_post, init_theta):
    prior = dict(B_PRIOR_DEFAULT)
    prior["seed"] = int(seed)
    trace, final = gibbs_ss(y, Phi, w, prior, n_burn, n_post, int(seed),
                            init_theta.astype(bool))
    return trace


def stageI(X_deseas, vc, seed):
    """返回 dict: pip, theta(2d), beta_med, gamma_med, colnames, c_nodes,
    hinge_idx, poly1_idx, Phi_level, y, w, selected(逐点 PIP>0.5 或簇规则)"""
    y = np.diff(X_deseas)          # Δt=1
    x_level = X_deseas[:-1]
    n_h = vc["n_hinge"]
    prior = dict(B_PRIOR_DEFAULT)
    qlo, qhi = vc.get("hinge_q_range", (0.02, 0.98))
    prior["hinge_c_quantiles"] = np.linspace(qlo, qhi, n_h)

    if vc["stageI"] == "binned":
        dw = build_drift_work_samples(X_deseas, t_train=None, K=vc.get("K", 80),
                                      tau_q=0.95, n_min=20, winsor_pct=0.95,
                                      eps_v=1e-6, dt_hours=1.0)
        x_d, y_d, w_d = dw["x_k"], dw["b_hat"], dw["w_k"]
    else:  # full
        x_d, y_d, w_d = x_level, y, np.ones(len(y))

    Phi, colnames, meta = build_phi(x_d, x_d, prior)
    p = Phi.shape[1]
    # 评估用全样本水平设计
    Phi_level, _, _ = build_phi(x_d, x_level, prior)
    # hinge 对 {1,x,x²,x³} 正交化（保持线性张成不变，降低与多项式共线）
    import re as _re0
    _hr0 = _re0.compile(r"\(x-(.*?)\)_\+")
    hinge_cols = [j for j, cn in enumerate(colnames) if _hr0.search(cn)]
    if vc.get("orth_hinge") and len(hinge_cols) > 0:
        Zd = np.column_stack([np.ones(len(x_d)), x_d, x_d ** 2, x_d ** 3])
        Zl = np.column_stack([np.ones(len(x_level)), x_level,
                              x_level ** 2, x_level ** 3])
        for j in hinge_cols:
            a_j, *_ = np.linalg.lstsq(Zd, Phi[:, j], rcond=None)
            Phi[:, j] = Phi[:, j] - Zd @ a_j
            Phi_level[:, j] = Phi_level[:, j] - Zl @ a_j
    # 初始化：加权 OLS top-half t
    Wsr = np.sqrt(w_d)
    Aw = np.column_stack([Wsr, Phi * Wsr[:, None]])
    yw = y_d * Wsr
    b_ols, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    try:
        res = yw - Aw @ b_ols
        s2 = float(np.sum(res**2) / max(len(yw) - Aw.shape[1], 1))
        covb = s2 * np.linalg.inv(Aw.T @ Aw + 1e-12 * np.eye(Aw.shape[1]))
        t_vals = np.abs(b_ols[1:]) / np.sqrt(np.maximum(np.diag(covb)[1:], 1e-300))
    except Exception:
        t_vals = np.zeros(p)
    init = (t_vals >= np.median(t_vals))

    tr = run_gibbs(y_d, Phi, w_d, seed + 2000, vc["n_burn"], vc["n_post"], init)
    theta = tr["theta"].astype(bool)          # (S,p)
    beta_d = tr["beta"]                       # (S,p)
    pip = theta.mean(axis=0)
    beta_med = np.median(beta_d, axis=0)
    gamma_med = float(np.median(tr["gamma"]))

    import re as _re
    _hr = _re.compile(r"\(x-(.*?)\)_\+")
    hinge_map, poly1_idx = {}, None
    for i, cn in enumerate(colnames):
        if cn.startswith("x^1"):
            poly1_idx = i
        mm = _hr.search(cn)
        if mm:
            try:
                hinge_map[i] = float(mm.group(1).strip())
            except Exception:
                pass

    # ---- 选择规则 ----
    if vc.get("sel_mode") == "window" and len(hinge_map) > 0:
        # 滑窗证据聚合 + 非极大值抑制：
        #   每个候选 c_j 的窗 W_j = {|c-c_j|≤tol}，wcpip_j = P(W_j 内任一 θ=1)；
        #   按 wcpip 降序，阈值 + NMS（半径 nms）选取窗中心；
        #   簇 span = 窗口本身（供剖面细化）。
        tol_w = vc.get("win_tol", 8.0)
        thr = vc.get("cpip_thr", 0.4)
        nms = vc.get("nms_radius", 12.0)
        msm = vc.get("min_side_mass", 0.0)
        items = sorted(hinge_map.items(), key=lambda kv: kv[1])
        js = [j for (j, _) in items]
        cs_arr = np.array([c for (_, c) in items], dtype=float)
        wscore = {}
        for (j, c) in items:
            W = [jj for (jj, cc) in items if abs(cc - c) <= tol_w]
            wscore[j] = float(theta[:, W].any(axis=1).mean())
        order = sorted(items, key=lambda kv: -wscore[kv[0]])
        sel_centers = []  # (c_anchor, j_anchor)
        for (j, c) in order:
            if wscore[j] <= thr:
                continue
            if any(abs(c - a) <= nms for (a, _) in sel_centers):
                continue
            if msm > 0.0 and min(float(np.mean(x_level < c)),
                                 float(np.mean(x_level > c))) < msm:
                continue
            sel_centers.append((float(c), j))
        selected = np.zeros(p, dtype=bool)
        for j in range(p):
            if j not in hinge_map and pip[j] > 0.5:
                selected[j] = True
        for (_, j) in sel_centers:
            selected[j] = True
        clusters_sel = [dict(c_rep=c, c_lo=c - tol_w, c_hi=c + tol_w,
                             cpip=wscore[j]) for (c, j) in sel_centers]
        rep_cs = np.array([c for (c, _) in sel_centers], dtype=float)
    elif vc.get("cluster_bw") is not None and len(hinge_map) > 0:
        # 邻域聚合：按折点位置排序，间隔 ≤ bw 归簇；簇 PIP = P(簇内任一 θ=1)
        bw = vc["cluster_bw"]
        hsorted = sorted(hinge_map.items(), key=lambda kv: kv[1])
        clusters, cur = [], [hsorted[0]]
        for it in hsorted[1:]:
            if it[1] - cur[-1][1] <= bw:
                cur.append(it)
            else:
                clusters.append(cur); cur = [it]
        clusters.append(cur)
        sel_hinge_idx, cluster_info, clusters_sel = [], [], []
        cpip_thr = vc.get("cpip_thr", 0.5)
        msm = vc.get("min_side_mass", 0.0)
        for cl in clusters:
            cols = [j for (j, _) in cl]
            any_inc = theta[:, cols].any(axis=1)
            cpip = float(any_inc.mean())
            if cpip > cpip_thr:
                # 代表折点：以簇内平均 |β| 为权的 c 加权平均
                wabs = np.abs(beta_d[:, cols]).mean(axis=0)
                if wabs.sum() <= 1e-12:
                    c_rep = float(np.mean([c for (_, c) in cl]))
                    j_rep = cols[0]
                else:
                    c_rep = float(np.sum(wabs * np.array([c for (_, c) in cl]))
                                  / wabs.sum())
                    j_rep = cols[int(np.argmax(wabs))]
                # 双侧样本量守门：折点两侧样本占比均需 ≥ min_side_mass，
                # 否则该折点的"斜率变化"在识别上不可信（尾部稀疏区）
                if msm > 0.0:
                    frac_below = float(np.mean(x_level < c_rep))
                    frac_above = float(np.mean(x_level > c_rep))
                    if min(frac_below, frac_above) < msm:
                        continue
                sel_hinge_idx.append(j_rep)
                cluster_info.append((c_rep, j_rep))
                clusters_sel.append(dict(c_rep=c_rep,
                                         c_lo=float(min(c for (_, c) in cl)),
                                         c_hi=float(max(c for (_, c) in cl)),
                                         cpip=cpip))
        selected = np.zeros(p, dtype=bool)
        # 非 hinge 列（多项式）仍按逐点 PIP>0.5
        for j in range(p):
            if j not in hinge_map and pip[j] > 0.5:
                selected[j] = True
        for j in sel_hinge_idx:
            selected[j] = True
        rep_cs = np.array([c for (c, _) in cluster_info], dtype=float)
    else:
        selected = pip > 0.5
        rep_cs = np.array([hinge_map[j] for j in np.where(selected)[0]
                           if j in hinge_map], dtype=float)
        clusters_sel = []

    return dict(pip=pip, selected=selected, colnames=colnames,
                clusters_sel=clusters_sel, theta=theta,
                hinge_map=hinge_map, poly1_idx=poly1_idx,
                beta_med=beta_med, gamma_med=gamma_med,
                Phi_level=Phi_level, x_level=x_level, y=y,
                rep_cs=rep_cs, x_d=x_d, y_d=y_d, w_d=w_d,
                prior=prior)


def refit_student_t(y, X_mat, n_iter=30, nu=4.0):
    """Student-t IRLS 去偏重估（含截距列）。返回系数向量。"""
    A = np.column_stack([np.ones(len(y)), X_mat])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    for _ in range(n_iter):
        r = y - A @ coef
        s = np.median(np.abs(r - np.median(r))) / 0.6744897501960817
        s = max(s, 1e-8)
        u = (nu + 1.0) / (nu + (r / s) ** 2)
        wu = np.sqrt(u)
        coef_new, *_ = np.linalg.lstsq(A * wu[:, None], y * wu, rcond=None)
        if np.max(np.abs(coef_new - coef)) < 1e-8 * (1 + np.max(np.abs(coef))):
            coef = coef_new; break
        coef = coef_new
    return coef  # [gamma, beta_S...]


def _student_t_neg2ll(y, Xmat, coef, nu=4.0):
    """固定 ν 的 Student-t 负二倍对数似然（尺度 s 由残差 MAD 稳健估计）。"""
    r = y - coef[0] - (Xmat @ coef[1:] if Xmat.shape[1] else 0.0)
    s = max(np.median(np.abs(r - np.median(r))) / 0.6744897501960817, 1e-8)
    n = len(y)
    ll = n * np.log(s) + ((nu + 1.0) / 2.0) * np.sum(np.log1p((r / s) ** 2 / nu))
    return 2.0 * ll


def profile_refine(s1, margin=3.0, step=0.5, nu=4.0, t_thr=0.0, sel_ic=None):
    """对每个选中簇做折点剖面细化（固定其他项，逐簇网格搜索）。
    全模型 = 截距 + 选中多项式 + 每簇一个细化折点；Student-t IRLS 重估。
    sel_ic="bic": 在按 |t| 排序的嵌套子模型上做 BIC 规模选择（自适应惩罚，
    替代固定 t 阈值）；sel_ic=None 时沿用 t_thr 门限。
    返回 (refined_cs, b_level)。"""
    y = s1["y"]; x = s1["x_level"]
    sel = s1["selected"]; hinge_map = s1["hinge_map"]
    poly_cols = [j for j in np.where(sel)[0] if j not in hinge_map]
    Xpoly = s1["Phi_level"][:, poly_cols] if poly_cols else np.zeros((len(y), 0))
    refined = []
    for cl in (s1.get("clusters_sel") or []):
        grid = np.arange(cl["c_lo"] - margin, cl["c_hi"] + margin + 1e-9, step)
        best_c, best_crit = None, np.inf
        for c in grid:
            h_c = np.maximum(x - c, 0.0).reshape(-1, 1)
            Xc = np.column_stack([Xpoly, h_c]) if Xpoly.shape[1] else h_c
            coef = refit_student_t(y, Xc, n_iter=8, nu=nu)
            r = y - coef[0] - Xc @ coef[1:]
            s = max(np.median(np.abs(r - np.median(r))) / 0.6744897501960817, 1e-8)
            crit = float(np.mean(np.log1p((r / s) ** 2 / nu)))
            if crit < best_crit:
                best_crit, best_c = crit, float(c)
        refined.append(best_c)
    refined = np.array(refined, float) if refined else np.array([], float)
    blocks = [Xpoly] if Xpoly.shape[1] else []
    for c in refined:
        blocks.append(np.maximum(x - c, 0.0))
    Xfin = np.column_stack(blocks) if blocks else np.zeros((len(y), 0))
    if Xfin.shape[1] == 0:
        b_level = np.full(len(y), float(np.median(y)))
        return refined, b_level
    coef = refit_student_t(y, Xfin, n_iter=40, nu=nu)
    n_poly = Xfin.shape[1] - len(refined)
    if len(refined) > 0:
        # ---- 折点 |t| 统计量（用于排序 / t 门限）----
        r = y - coef[0] - Xfin @ coef[1:]
        s = max(np.median(np.abs(r - np.median(r))) / 0.6744897501960817, 1e-8)
        u = (nu + 1.0) / (nu + (r / s) ** 2)   # IRLS 终权重
        A = np.column_stack([np.ones(len(y)), Xfin])
        AtWA = A.T @ (A * u[:, None])
        t_stats = None
        try:
            covb = np.linalg.inv(AtWA + 1e-10 * np.eye(AtWA.shape[0])) * (s ** 2)
            se = np.sqrt(np.maximum(np.diag(covb), 1e-300))
            t_stats = np.abs(coef[1 + n_poly:]) / se[1 + n_poly:]
        except np.linalg.LinAlgError:
            pass
        # ---- (A) BIC 规模选择：按 |t| 降序的嵌套子模型，取 BIC 最小 ----
        if sel_ic == "bic" and t_stats is not None:
            order = np.argsort(-t_stats)
            n = len(y)
            best_ic, best_m = np.inf, 0
            for m in range(0, len(refined) + 1):
                cs_m = refined[order[:m]]
                blocks_m = [Xpoly] if Xpoly.shape[1] else []
                for c in cs_m:
                    blocks_m.append(np.maximum(x - c, 0.0))
                Xm = np.column_stack(blocks_m) if blocks_m else np.zeros((len(y), 0))
                if Xm.shape[1] == 0:
                    cm = np.array([float(np.median(y))])
                    ic = _student_t_neg2ll(y, np.zeros((n, 0)), cm, nu) + 1.0 * np.log(n)
                else:
                    cm = refit_student_t(y, Xm, n_iter=40, nu=nu)
                    ic = _student_t_neg2ll(y, Xm, cm, nu) + (1 + Xm.shape[1]) * np.log(n)
                if ic < best_ic:
                    best_ic, best_m = ic, m
            keep_idx = order[:best_m]
            if best_m < len(refined):
                refined = refined[np.sort(keep_idx)]
                blocks = [Xpoly] if Xpoly.shape[1] else []
                for c in refined:
                    blocks.append(np.maximum(x - c, 0.0))
                Xfin = np.column_stack(blocks) if blocks else np.zeros((len(y), 0))
                if Xfin.shape[1] == 0:
                    b_level = np.full(len(y), float(np.median(y)))
                    return refined, b_level
                coef = refit_student_t(y, Xfin, n_iter=40, nu=nu)
        # ---- (B) 固定 t 门限过滤 ----
        elif t_thr > 0.0 and t_stats is not None:
            keep = t_stats >= t_thr
            if not np.all(keep):
                refined = refined[keep]
                blocks = [Xpoly] if Xpoly.shape[1] else []
                for c in refined:
                    blocks.append(np.maximum(x - c, 0.0))
                Xfin = np.column_stack(blocks) if blocks else np.zeros((len(y), 0))
                if Xfin.shape[1] == 0:
                    b_level = np.full(len(y), float(np.median(y)))
                    return refined, b_level
                coef = refit_student_t(y, Xfin, n_iter=40, nu=nu)
    b_level = coef[0] + Xfin @ coef[1:]
    return refined, b_level


# ---------------- Stage II ----------------
def ecf_fit_asy(r, n_start, seed, multiscale=False):
    """返回 (xi6, Q). multiscale: 联合 1h + 24h(重叠 stride=6h) 目标，W=I。"""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    dat1 = build_ecf_data(r)
    if not multiscale:
        res = run_multi_start(dat1, symmetric=False, n_start=n_start, seed=seed)
        best = res[0]
        return np.asarray(best["xi_final"], float), float(best["Q_final"])
    # 24h 重叠聚合
    h, stride = 24, 6
    n = len(r)
    idx0 = np.arange(0, n - h, stride)
    r24 = np.array([r[i:i + h].sum() for i in idx0])
    dat24 = build_ecf_data(r24)

    def Qmulti(xi):
        return compute_Q(xi, dat1, False) + compute_Q(xi, dat24, False)

    starts = starting_points_lhs(n_start, False, dat1, seed)
    best_xi, best_q = None, np.inf
    for xi0 in starts:
        try:
            u0 = to_unconstrained(xi0, False)
            r1 = optimize.minimize(
                lambda u: Qmulti(to_constrained(u, False)), u0,
                method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-9})
            xiA = to_constrained(np.asarray(r1.x, float), False)
            r2 = optimize.minimize(
                Qmulti, xiA, method="Nelder-Mead",
                bounds=[(None, None), (1e-8, None), (1e-8, None),
                        (1e-8, None), (1e-8, None), (_ALPHA_LB, _ALPHA_UB)],
                options={"maxiter": 1500, "fatol": 1e-9, "xatol": 1e-9})
            if float(r2.fun) < best_q:
                best_q = float(r2.fun); best_xi = np.asarray(r2.x, float)
        except Exception:
            continue
    if best_xi is None:
        return np.full(6, np.nan), np.inf
    return best_xi, best_q


# ---------------- 指标 ----------------
TOL = 8.0
WASS_PENALTY = 100.0


def selection_metrics(s1, dgp, refined_cs=None):
    hinges_true = DGP_LAB[dgp]["hinges"]
    true_cs = np.array([c for (c, _) in hinges_true], float)
    sel = s1["selected"]; hinge_map = s1["hinge_map"]
    poly1_idx = s1["poly1_idx"]
    # TPR: poly1 + 每个真折点邻域内是否有选中 hinge
    poly_hit = int(bool(sel[poly1_idx])) if poly1_idx is not None else 0
    if refined_cs is not None:
        sel_hinge_cs = np.asarray(refined_cs, float)
        n_false_poly = int(sum(1 for j in np.where(sel)[0]
                               if j not in hinge_map and j != poly1_idx))
        nsel = n_false_poly + int(bool(sel[poly1_idx])) + len(sel_hinge_cs)
        fd = n_false_poly
        for c in sel_hinge_cs:
            if len(true_cs) == 0 or not np.any(np.abs(true_cs - c) < TOL):
                fd += 1
    else:
        sel_hinge_cs = np.array([hinge_map[j] for j in np.where(sel)[0]
                                 if j in hinge_map], float)
        nsel = int(sel.sum())
        fd = 0
        for j in np.where(sel)[0]:
            if j == poly1_idx:
                continue
            if j in hinge_map:
                if len(true_cs) == 0 or not np.any(
                        np.abs(true_cs - hinge_map[j]) < TOL):
                    fd += 1
            else:
                fd += 1
    hits = 0
    for tc in true_cs:
        if len(sel_hinge_cs) and np.any(np.abs(sel_hinge_cs - tc) < TOL):
            hits += 1
    tp = poly_hit + hits
    pos_tot = (1 if poly1_idx is not None else 0) + len(true_cs)
    tpr = tp / max(pos_tot, 1)
    fdr = fd / max(nsel, 1)
    prec = 1.0 - fdr
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
    # 折点 Wasserstein（选中 hinge 位置 vs 真折点，等权）
    if len(true_cs) == 0 and len(sel_hinge_cs) == 0:
        wass = 0.0
    elif len(true_cs) == 0 or len(sel_hinge_cs) == 0:
        wass = WASS_PENALTY
    else:
        wass = float(wasserstein_distance(true_cs, sel_hinge_cs))
    return dict(tpr=tpr, fdr=fdr, f1=f1, nsel=nsel, wass=wass)


def medse_metric(b_hat_level, x_level, dgp):
    b0 = true_drift_arr(x_level, dgp)
    return float(np.median((b_hat_level - b0) ** 2))


# ---------------- 单重复 ----------------
def run_one(task):
    (seed, dgp, T, vname, vc, stage2, n_start) = task
    t0 = time.time()
    X_deseas, X_true = gen_data(seed, dgp, T)
    s1 = stageI(X_deseas, vc, seed)
    row = dict(variant=vname, seed=seed, dgp=dgp, T=T)
    if vc.get("profile_refine"):
        refined_cs, b_level = profile_refine(
            s1, margin=vc.get("refine_margin", 3.0),
            t_thr=vc.get("t_thr", 0.0),
            sel_ic=vc.get("sel_ic", None))
        row.update(selection_metrics(s1, dgp, refined_cs=refined_cs))
    else:
        row.update(selection_metrics(s1, dgp))
        # plug-in 漂移
        if vc.get("refit") and row["nsel"] > 0:
            cols = np.where(s1["selected"])[0]
            Phi_sel = build_phi_at(s1, cols)   # (T-1, |S|)，全样本水平
            coef = refit_student_t(s1["y"], Phi_sel)
            b_level = coef[0] + Phi_sel @ coef[1:]
        else:
            b_level = s1["gamma_med"] + s1["Phi_level"] @ s1["beta_med"]
    row["medse"] = medse_metric(b_level, s1["x_level"], dgp)

    if stage2 != "none":
        r_pi = s1["y"] - b_level
        # oracle（变体无关，缓存）
        of = os.path.join(RES_DIR, f"oracle_{dgp}_T{T}_seed{seed}"
                                   f"_ms{int(stage2 == 'multi')}_ns{n_start}.json")
        if os.path.exists(of):
            with open(of) as fh:
                xi_or = np.array(json.load(fh)["xi"])
        else:
            r_or = s1["y"] - true_drift_arr(s1["x_level"], dgp)
            xi_or, _ = ecf_fit_asy(r_or, n_start, seed + 5000,
                                   multiscale=(stage2 == "multi"))
            with open(of, "w") as fh:
                json.dump({"xi": np.asarray(xi_or, float).tolist()}, fh)
        xi_pi, Q_pi = ecf_fit_asy(r_pi, n_start, seed + 6000,
                                  multiscale=(stage2 == "multi"))
        names = ["mu", "Cp", "Cm", "M", "G", "alpha"]
        for i, nm in enumerate(names):
            row[f"or_{nm}"] = float(xi_or[i]); row[f"pl_{nm}"] = float(xi_pi[i])
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def build_phi_at(s1, cols):
    # 在 x_level 上构造选中列的设计（列定义来自 s1["Phi_level"] 同序）
    return s1["Phi_level"][:, cols]


# ---------------- 变体表 ----------------
VARIANTS = {
    # 实验室内所有变体统一 MCMC 预算 burn=1500/post=3000，保证组件效应可归因
    "v0": dict(stageI="binned", K=80, n_hinge=50, cluster_bw=None, refit=False,
               n_burn=1500, n_post=3000),
    "v1": dict(stageI="binned", K=80, n_hinge=13, cluster_bw=None, refit=False,
               n_burn=1500, n_post=3000),
    "v2": dict(stageI="full", n_hinge=13, cluster_bw=None, refit=False,
               n_burn=1500, n_post=3000),
    "v3": dict(stageI="full", n_hinge=13, cluster_bw=5.0, refit=False,
               n_burn=1500, n_post=3000),
    "v4": dict(stageI="full", n_hinge=13, cluster_bw=5.0, refit=True,
               n_burn=1500, n_post=3000),
    "v5": dict(stageI="full", n_hinge=13, cluster_bw=5.0, refit=True,
               n_burn=1500, n_post=3000),  # stage2=multi 时体现差异
    "v6": dict(stageI="full", n_hinge=13, cluster_bw=None, refit=False,
               orth_hinge=True, n_burn=1500, n_post=3000),
    "v7": dict(stageI="full", n_hinge=13, cluster_bw=9.0, refit=False,
               orth_hinge=True, n_burn=1500, n_post=3000),
    "v8": dict(stageI="full", n_hinge=13, cluster_bw=9.0, refit=True,
               orth_hinge=True, n_burn=1500, n_post=3000),
    "v9": dict(stageI="full", n_hinge=13, cluster_bw=9.0, refit=False,
               orth_hinge=True, profile_refine=True,
               n_burn=1500, n_post=3000),
    "v10": dict(stageI="full", n_hinge=13, cluster_bw=9.0, refit=False,
                orth_hinge=True, profile_refine=True,
                hinge_q_range=(0.06, 0.94),
                n_burn=1500, n_post=3000),
    "v11": dict(stageI="full", n_hinge=13, cluster_bw=9.0, refit=False,
                orth_hinge=True, profile_refine=True,
                cpip_thr=0.35, min_side_mass=0.05,
                n_burn=1500, n_post=3000),
    "v12": dict(stageI="full", n_hinge=13, sel_mode="window", refit=False,
                orth_hinge=True, profile_refine=True, refine_margin=0.0,
                win_tol=8.0, cpip_thr=0.4, nms_radius=12.0,
                min_side_mass=0.05,
                n_burn=1500, n_post=3000),
    # v13 = v12 + 稳健 t 显著性过滤：细化折点的斜率变化量 δ̂ 的 IRLS t 统计量
    # 尾部稀疏区假折点 se(δ̂)∝1/√(上方样本量) 很大 → 被过滤；真实折点 δ=0.06~0.07 显著
    "v13": dict(stageI="full", n_hinge=13, sel_mode="window", refit=False,
                orth_hinge=True, profile_refine=True, refine_margin=0.0,
                win_tol=8.0, cpip_thr=0.4, nms_radius=12.0,
                min_side_mass=0.05, t_thr=2.0,
                n_burn=1500, n_post=3000),
    # v14 = v13 + 放宽候选准入 cpip 0.4→0.3（debug_cpip 显示真实簇 cpip 低至 0.35），
    # 假阳性由 t_thr=2.0 显著性过滤 + min_side_mass 吸收：宽进严出
    "v14": dict(stageI="full", n_hinge=13, sel_mode="window", refit=False,
                orth_hinge=True, profile_refine=True, refine_margin=0.0,
                win_tol=8.0, cpip_thr=0.3, nms_radius=12.0,
                min_side_mass=0.05, t_thr=2.0,
                n_burn=1500, n_post=3000),
    # v15 = v13 + MCMC 预算加倍（burn 3000 / post 6000）：
    # 诊断显示真实簇 cpip 在 0.35~0.75 徘徊，链长不足致后验未集中是 TPR 瓶颈嫌疑
    "v15": dict(stageI="full", n_hinge=13, sel_mode="window", refit=False,
                orth_hinge=True, profile_refine=True, refine_margin=0.0,
                win_tol=8.0, cpip_thr=0.4, nms_radius=12.0,
                min_side_mass=0.05, t_thr=2.0,
                n_burn=3000, n_post=6000),
    # v16 = v13 + BIC 规模选择替代固定 t 门限：按 |t| 嵌套子模型取 BIC 最小，
    # 惩罚随样本量自适应，针对稀疏尾部小幅假折点（FDR 瓶颈）
    "v16": dict(stageI="full", n_hinge=13, sel_mode="window", refit=False,
                orth_hinge=True, profile_refine=True, refine_margin=0.0,
                win_tol=8.0, cpip_thr=0.4, nms_radius=12.0,
                min_side_mass=0.05, t_thr=0.0, sel_ic="bic",
                n_burn=1500, n_post=3000),
}


def summarize(df):
    out = []
    for dgp in df["dgp"].unique():
        sub = df[df["dgp"] == dgp]
        r = dict(dgp=dgp, S=len(sub),
                 TPR=f"{sub.tpr.mean():.3f}±{sub.tpr.std():.3f}",
                 FDR=f"{sub.fdr.mean():.3f}±{sub.fdr.std():.3f}",
                 F1=f"{sub.f1.mean():.3f}±{sub.f1.std():.3f}",
                 nSel=f"{sub.nsel.mean():.1f}",
                 Wass=f"{sub.wass.mean():.1f}±{sub.wass.std():.1f}",
                 MedSE=f"{sub.medse.mean():.3f}±{sub.medse.std():.3f}")
        if "pl_Cm" in sub:
            tr = DGP_LAB[dgp]["xi"]
            for nm, tv in zip(["mu", "Cp", "Cm", "M", "G", "alpha"], tr):
                b = sub[f"pl_{nm}"] - tv
                r[f"b_{nm}"] = f"{b.mean():+.3f}"
                r[f"r_{nm}"] = f"{np.sqrt((b**2).mean()):.3f}"
        out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v0", choices=list(VARIANTS))
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--dgps", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--stage2", default="none",
                    choices=["none", "light", "multi"])
    ap.add_argument("--n-start", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=91300000)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    vc = VARIANTS[args.variant]
    tasks = [(args.seed0 + s, dgp, args.T, args.variant, vc,
              ("multi" if args.stage2 == "multi" else
               ("1h" if args.stage2 == "light" else "none")),
              args.n_start)
             for s in range(args.S) for dgp in args.dgps]
    print(f"[LAB] variant={args.variant} tasks={len(tasks)} "
          f"stage2={args.stage2} cfg={vc}", flush=True)
    t0 = time.time()
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(run_one, tasks):
                rows.append(r)
                print(f"  done {r['dgp']} seed={r['seed']} "
                      f"tpr={r['tpr']:.2f} fdr={r['fdr']:.2f} "
                      f"medse={r['medse']:.3f} ({r['elapsed_s']}s)", flush=True)
    else:
        rows = [run_one(t) for t in tasks]
        for r in rows:
            print(f"  done {r['dgp']} seed={r['seed']} "
                  f"tpr={r['tpr']:.2f} fdr={r['fdr']:.2f} "
                  f"medse={r['medse']:.3f} ({r['elapsed_s']}s)", flush=True)
    df = pd.DataFrame(rows)
    fcsv = os.path.join(RES_DIR, f"lab_rows_{args.variant}_T{args.T}"
                                   f"_S{args.S}_{args.stage2}.csv")
    df.to_csv(fcsv, index=False, encoding="utf-8-sig")
    summ = summarize(df)
    print(f"\n[SUMMARY {args.variant}] elapsed={time.time()-t0:.0f}s -> {fcsv}")
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(RES_DIR, f"lab_summary_{args.variant}_T{args.T}"
                                      f"_S{args.S}_{args.stage2}.csv"),
                index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    raise SystemExit(main())
