# -*- coding: utf-8 -*-
"""
run_baselines.py — 外部基准方法（非 v0 谱系），与 v13 同 DGP/同种子/同指标
Stage I 基准: ols1 / ols3 / lasso_cv / stepwise_bic
  （朴素 SS-Normal = run_lab.py 的 v2 配置，单独用 run_lab 跑）
Stage II 基准: cmom(累积量1-6矩法) / cgmm(累积量1-8过识别GMM) / mlefft(FFT密度MLE)
  在 oracle 残差与 v13 插件残差上分别运行，T=25000, S=12
全部真实运行，输出至 method_lab/results/baselines_*.csv
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
from scipy import optimize, special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_lab as R
from run_stage_C import ts_phi_dt, _ALPHA_LB, _ALPHA_UB

RES = R.RES_DIR
CACHE = R.CACHE_DIR
PNAMES = ["mu", "Cp", "Cm", "M", "G", "alpha"]


# ============ 公共：数据与设计 ============
def load_xy(dgp, T, seed):
    z = np.load(os.path.join(CACHE, f"data_{dgp}_T{T}_seed{seed}.npz"))
    Xd = z["X_deseas"]
    return Xd[:-1], np.diff(Xd)


def design13(x):
    prior = dict(R.B_PRIOR_DEFAULT)
    prior["hinge_c_quantiles"] = np.linspace(0.02, 0.98, 13)
    Phi, colnames, meta = R.build_phi(x, x, prior)
    import re as _re
    _hr = _re.compile(r"\(x-(.*?)\)_\+")
    hinge_map, poly1_idx = {}, None
    for i, cn in enumerate(colnames):
        if cn.startswith("x^1"):
            poly1_idx = i
        mm = _hr.search(cn)
        if mm:
            hinge_map[i] = float(mm.group(1).strip())
    return Phi, colnames, hinge_map, poly1_idx


def eval_sel(selected, b_level, x, dgp, hinge_map, poly1_idx):
    s1 = dict(selected=selected, hinge_map=hinge_map, poly1_idx=poly1_idx)
    m = R.selection_metrics(s1, dgp)
    m["medse"] = R.medse_metric(b_level, x, dgp)
    return m


# ============ Stage I 基准 ============
def base_ols(x, y, dgp, deg):
    A = np.column_stack([np.ones(len(x))] + [x ** p for p in range(1, deg + 1)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    b = A @ coef
    Phi, colnames, hinge_map, poly1_idx = design13(x)
    sel = np.zeros(Phi.shape[1], bool)
    sel[poly1_idx] = True
    if deg >= 2:
        for i, cn in enumerate(colnames):
            if cn in ("x^2", "x^3") and deg >= int(cn[-1]):
                sel[i] = True
    return eval_sel(sel, b, x, dgp, hinge_map, poly1_idx)


def base_lasso(x, y, dgp, seed):
    from sklearn.linear_model import LassoCV
    Phi, colnames, hinge_map, poly1_idx = design13(x)
    m = LassoCV(cv=10, random_state=seed, n_alphas=60, max_iter=20000)
    m.fit(Phi, y)
    sel = np.abs(m.coef_) > 1e-8
    b = m.intercept_ + Phi @ m.coef_
    return eval_sel(sel, b, x, dgp, hinge_map, poly1_idx)


def _bic(y, X):
    n = len(y)
    A = np.column_stack([np.ones(n), X]) if X.shape[1] else np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rss = float(np.sum((y - A @ coef) ** 2))
    k = A.shape[1]
    return n * np.log(max(rss / n, 1e-300)) + k * np.log(n), coef


def base_stepwise(x, y, dgp):
    Phi, colnames, hinge_map, poly1_idx = design13(x)
    p = Phi.shape[1]
    cur, rest = [], list(range(p))
    bic_cur, _ = _bic(y, Phi[:, cur])
    improved = True
    while improved:
        improved = False
        best = (bic_cur, None, None)
        for j in rest:                      # forward
            b, _ = _bic(y, Phi[:, cur + [j]])
            if b < best[0] - 1e-10:
                best = (b, j, "add")
        for j in list(cur):                 # backward
            cc = [k for k in cur if k != j]
            b, _ = _bic(y, Phi[:, cc])
            if b < best[0] - 1e-10:
                best = (b, j, "drop")
        if best[2] == "add":
            cur.append(best[1]); rest.remove(best[1]); bic_cur = best[0]; improved = True
        elif best[2] == "drop":
            cur.remove(best[1]); rest.append(best[1]); bic_cur = best[0]; improved = True
    _, coef = _bic(y, Phi[:, cur])
    A = np.column_stack([np.ones(len(y)), Phi[:, cur]])
    b = A @ coef
    sel = np.zeros(p, bool); sel[cur] = True
    return eval_sel(sel, b, x, dgp, hinge_map, poly1_idx)


# ============ Stage II：理论累积量（非对称 CTS） ============
def _Gamma_upper(s, x):
    """上不完全伽马 Γ(s,x)=∫_x^∞ t^{s-1}e^{-t}dt。
    s>0 直接用 gammaincc；s∈(-1,0)（即 α∈(1,2)）用递推
    Γ(s,x) = (Γ(s+1,x) − x^s e^{-x})/s。"""
    x = float(max(x, 1e-300))
    if s > 0.0:
        return float(special.gammaincc(s, x) * special.gamma(s))
    return float((special.gammaincc(s + 1.0, x) * special.gamma(s + 1.0)
                  - x ** s * np.exp(-x)) / s)


def ts_cumulants(xi, nmax=8):
    mu, Cp, Cm, M, G, a = xi
    # κ₁ 必须与 run_stage_C 的 h(z)=z·1_{|z|≤1} 截断一致：
    #   κ₁ = μ + ∫_{|z|>1} z ν(dz)
    #      = μ + C₊ M^{α-1} Γ(1-α, M) − C₋ G^{α-1} Γ(1-α, G)
    # （不能用 h≡0 表示的 Γ(1-α)[C₊M^{α-1}−C₋G^{α-1}]，二者相差 ∫_{|z|≤1} zν(dz)）
    k1 = (mu + Cp * M ** (a - 1) * _Gamma_upper(1.0 - a, M)
              - Cm * G ** (a - 1) * _Gamma_upper(1.0 - a, G))
    ks = [k1]
    for n in range(2, nmax + 1):  # n≥2 阶累积量与截断表示无关
        ks.append(special.gamma(n - a) *
                  (Cp * M ** (a - n) + (-1) ** n * Cm * G ** (a - n)))
    return np.array(ks)


def sample_cumulants(r, nmax=8):
    r = r[np.isfinite(r)]
    m1 = r.mean(); d = r - m1
    mc = [0.0] + [float(np.mean(d ** k)) for k in range(1, nmax + 1)]
    k = [0.0] * (nmax + 1)
    k[1] = m1; k[2] = mc[2]; k[3] = mc[3]
    k[4] = mc[4] - 3 * mc[2] ** 2
    k[5] = mc[5] - 10 * mc[3] * mc[2]
    k[6] = mc[6] - 15 * mc[4] * mc[2] - 10 * mc[3] ** 2 + 30 * mc[2] ** 3
    if nmax >= 8:
        k[7] = mc[7] - 21 * mc[5] * mc[2] - 35 * mc[4] * mc[3] + 210 * mc[3] * mc[2] ** 2
        k[8] = (mc[8] - 28 * mc[6] * mc[2] - 56 * mc[5] * mc[3] - 35 * mc[4] ** 2
                + 420 * mc[4] * mc[2] ** 2 + 560 * mc[3] ** 2 * mc[2] - 630 * mc[2] ** 4)
    return np.array(k[1:])


# ---- 参数变换（z 尺度上估计，映回原尺度） ----
def to_u(xi):
    mu, Cp, Cm, M, G, a = xi
    t = (a - _ALPHA_LB) / (_ALPHA_UB - _ALPHA_LB)
    t = min(max(t, 1e-9), 1 - 1e-9)
    return np.array([mu, np.log(max(Cp, 1e-12)), np.log(max(Cm, 1e-12)),
                     np.log(max(M, 1e-12)), np.log(max(G, 1e-12)),
                     np.log(t / (1 - t))])


def to_xi(u):
    u = np.asarray(u, float)
    a = _ALPHA_LB + (_ALPHA_UB - _ALPHA_LB) / (1 + np.exp(-np.clip(u[5], -30, 30)))
    e = np.exp(np.clip(u[1:5], -30, 30))
    return np.array([u[0], e[0], e[1], e[2], e[3], a])


def lhs_starts(n, seed):
    rng = np.random.default_rng(seed)
    U = np.column_stack([
        rng.uniform(-1.0, 1.0, n),
        rng.uniform(np.log(0.1), np.log(5.0), n),
        rng.uniform(np.log(0.1), np.log(5.0), n),
        rng.uniform(np.log(0.005), np.log(0.5), n),
        rng.uniform(np.log(0.005), np.log(0.5), n),
        rng.uniform(-1.5, 1.5, n)])
    for i in range(U.shape[1]):             # LHS 分层（对 6 个参数列）
        perm = rng.permutation(n)
        U[:, i] = U[perm, i]
    return U


def scale_back(xi_z, sR):
    """z 尺度参数映回 r 尺度。C/M/G/α 为精确 TS 尺度映射；
    μ 必须修正 h(z)=z·1_{|z|≤1} 截断阈值的量纲漂移（阈值"1"随尺度改变）：
      sR>1:  μ_r = sR·μ_z − ∫_{1<|z|≤sR} z ν_r(dz)
      sR<1:  μ_r = sR·μ_z + ∫_{sR<|z|≤1} z ν_r(dz)
    其中 ∫_a^b C z^{-α}e^{-Mz}dz = C M^{α-1}[Γ(1-α,Ma)−Γ(1-α,Mb)]。
    （主方法 ECF 全程在 r 尺度估计、无此映射，不受影响。）"""
    mu, Cp, Cm, M, G, a = xi_z
    Cr_p, Cr_m = Cp * sR ** a, Cm * sR ** a
    Mr, Gr = M / sR, G / sR
    if abs(sR - 1.0) > 1e-12:
        lo, hi = (Mr, Mr * sR) if sR > 1.0 else (Mr * sR, Mr)
        sgn = -1.0 if sR > 1.0 else 1.0
        corr_p = Cr_p * Mr ** (a - 1) * (_Gamma_upper(1 - a, lo) - _Gamma_upper(1 - a, hi))
        lo, hi = (Gr, Gr * sR) if sR > 1.0 else (Gr * sR, Gr)
        corr_m = Cr_m * Gr ** (a - 1) * (_Gamma_upper(1 - a, lo) - _Gamma_upper(1 - a, hi))
        mu_r = sR * mu + sgn * (corr_p - corr_m)
    else:
        mu_r = mu
    return np.array([mu_r, Cr_p, Cr_m, Mr, Gr, a])


def _u_bounds():
    """u 空间显式边界（z 尺度）：防止退化拟合溢出到不可读数值区；
    撞界本身即重尾下矩法失效的可报告签名。"""
    t_lo = (0.05 - _ALPHA_LB) / (_ALPHA_UB - _ALPHA_LB)
    t_hi = (1.95 - _ALPHA_LB) / (_ALPHA_UB - _ALPHA_LB)
    return [(-50.0, 50.0),
            (np.log(1e-4), np.log(1e3)), (np.log(1e-4), np.log(1e3)),
            (np.log(1e-3), np.log(50.0)), (np.log(1e-3), np.log(50.0)),
            (np.log(t_lo / (1 - t_lo)), np.log(t_hi / (1 - t_hi)))]


def fit_cumulant(r, n_eq, seed, n_start=24):
    """n_eq=6: 恰好识别矩法; n_eq=8: 过识别 C-GMM. 在 z=r/sR 尺度上求解。"""
    r = r[np.isfinite(r)]
    sR = max(np.median(np.abs(r - np.median(r))) / 0.6744897501960817, 1e-8)
    z = r / sR
    kz = sample_cumulants(z, n_eq)
    # 标准化累积量匹配：g_n = (κ̂_n − κ_n)/κ̂_2^{n/2}（无量纲经典形式）。
    # 避免 (i) 高阶 κ 量级爆炸淹没低阶条件；(ii) κ̂₁≈0 时相对误差出现病态放大。
    s2 = max(float(kz[1]), 1e-12)
    wts = s2 ** (0.5 * np.arange(1, n_eq + 1))
    bnds = _u_bounds()
    _lo = np.array([b[0] for b in bnds]); _hi = np.array([b[1] for b in bnds])

    def obj(u):
        u = np.asarray(u, float)
        viol = np.maximum(_lo - u, 0.0) + np.maximum(u - _hi, 0.0)
        if np.any(viol > 0):   # 光滑墙罚：引导 NM 返回箱体，避免 1e12 悬崖困住单纯形
            return 1e8 + 1e6 * float(np.sum(viol ** 2))
        xi = to_xi(u)
        if not (xi[3] > 0 and xi[4] > 0 and 0 < xi[5] < 2):
            return 1e12
        try:
            th = ts_cumulants(xi, n_eq)
            g = (kz - th) / wts
            v = float(np.sum(g ** 2))
            return v if np.isfinite(v) else 1e12
        except Exception:
            return 1e12

    best, best_f = None, np.inf
    for u0 in lhs_starts(n_start, seed):
        try:
            r1 = optimize.minimize(obj, u0, method="L-BFGS-B", bounds=bnds,
                                   options={"maxiter": 500, "ftol": 1e-12})
            u1, f1 = np.asarray(r1.x, float), float(r1.fun)
            # NM 在 u 空间继续打磨（ξ 空间无约束会产生非法参数）
            r2 = optimize.minimize(obj, u1, method="Nelder-Mead",
                                   options={"maxiter": 2000, "fatol": 1e-12, "xatol": 1e-12})
            if float(r2.fun) < f1:
                u1, f1 = np.asarray(r2.x, float), float(r2.fun)
            if f1 < best_f:
                best_f, best = f1, u1
        except Exception:
            continue
    if best is None:
        return np.full(6, np.nan), np.inf
    return scale_back(to_xi(best), sR), best_f


def fit_mle_fft(r, seed, n_start=4, N=2 ** 12):
    """FFT 密度反演 MLE（z 尺度）。均匀网格 + 梯形权重。
    N=2^12：du=π/x_max 下 u_max≈80–100，|φ(u_max)|≤e^{-30}，分辨率充足；
    更大 N 只会线性放大求积成本（J×544 节点）。"""
    r = r[np.isfinite(r)]
    sR = max(np.median(np.abs(r - np.median(r))) / 0.6744897501960817, 1e-8)
    z = r / sR
    x_max = float(np.max(np.abs(np.quantile(z, [1e-6, 1 - 1e-6])))) * 1.5 + 1.0
    du = np.pi / x_max
    dx = 2.0 * x_max / N
    u = np.arange(N) * du
    w = np.ones(N); w[0] = w[-1] = 0.5
    xs = np.arange(-N // 2, N // 2) * dx

    def logdens(xi):
        phi = ts_phi_dt(u, xi)
        if not np.all(np.isfinite(phi)):
            return None
        Fp = np.fft.fft(phi * w)
        Fm = np.fft.fft(np.conj(phi) * w)
        f = np.empty(N)
        f[N // 2:] = np.real(Fp[:N // 2])      # x >= 0
        f[:N // 2] = np.real(Fm[1:N // 2 + 1])[::-1]  # x < 0
        f *= du / np.pi
        f = np.maximum(f, 1e-300)
        return xs, f

    bnds = _u_bounds()
    _lo = np.array([b[0] for b in bnds]); _hi = np.array([b[1] for b in bnds])

    def nll(u):
        u = np.asarray(u, float)
        viol = np.maximum(_lo - u, 0.0) + np.maximum(u - _hi, 0.0)
        if np.any(viol > 0):   # 光滑墙罚
            return 1e8 + 1e6 * float(np.sum(viol ** 2))
        xi = to_xi(u)
        if not (0 < xi[5] < 2):
            return 1e12
        out = logdens(xi)
        if out is None:
            return 1e12
        xx, ff = out
        zi = np.clip(z, xx[0] + dx, xx[-1] - dx)
        idx = np.clip(((zi - xx[0]) / dx).astype(int), 0, N - 2)
        t = (zi - xx[idx]) / dx
        lv = np.log(ff[idx]) * (1 - t) + np.log(ff[idx + 1]) * t
        v = -float(np.sum(lv))
        return v if np.isfinite(v) else 1e12

    best, best_f = None, np.inf
    for u0 in lhs_starts(n_start, seed):
        try:
            r1 = optimize.minimize(nll, u0, method="L-BFGS-B", bounds=bnds,
                                   options={"maxiter": 150, "ftol": 1e-10})
            u1, f1 = np.asarray(r1.x, float), float(r1.fun)
            r2 = optimize.minimize(nll, u1, method="Nelder-Mead",
                                   options={"maxiter": 500, "fatol": 1e-10, "xatol": 1e-10})
            if float(r2.fun) < f1:
                u1, f1 = np.asarray(r2.x, float), float(r2.fun)
            if f1 < best_f:
                best_f, best = f1, u1
        except Exception:
            continue
    if best is None:
        return np.full(6, np.nan), np.inf
    return scale_back(to_xi(best), sR), best_f


# ============ 任务体 ============
def stage1_one(task):
    method, dgp, T, seed = task
    x, y = load_xy(dgp, T, seed)
    if method == "ols1":
        m = base_ols(x, y, dgp, 1)
    elif method == "ols3":
        m = base_ols(x, y, dgp, 3)
    elif method == "lasso_cv":
        m = base_lasso(x, y, dgp, seed)
    elif method == "stepwise_bic":
        m = base_stepwise(x, y, dgp)
    else:
        raise ValueError(method)
    return dict(method=method, seed=seed, dgp=dgp, T=T, **m)


def stage2_one(task):
    method, src, dgp, seed = task
    t0 = time.time()
    try:
        return _stage2_one_impl(task)
    except Exception as e:
        row = dict(method=method, src=src, seed=seed, dgp=dgp, obj=np.inf,
                   error=str(e)[:200], elapsed_s=round(time.time() - t0, 1))
        for nm in PNAMES:
            row[nm] = np.nan
        return row


def _stage2_one_impl(task):
    method, src, dgp, seed = task
    T = 25000
    t0 = time.time()
    x, y = load_xy(dgp, T, seed)
    if src == "oracle":
        r = y - R.true_drift_arr(x, dgp)
    else:
        with open(os.path.join(RES, "v13_T25000_S12_identified_models.json"),
                  encoding="utf-8") as fh:
            recs = json.load(fh)
        rec = [z for z in recs if z["dgp"] == dgp and z["seed"] == seed][0]
        b = np.full_like(x, rec["intercept"], dtype=float)
        for t in rec["poly_terms"]:
            b = b + t["coef"] * x ** int(t["term"].split("^")[1])
        for k in rec["knots"]:
            b = b + k["delta"] * np.maximum(x - k["c"], 0.0)
        r = y - b
    if method == "cmom":
        xi, q = fit_cumulant(r, 6, seed + 7100)
    elif method == "cgmm":
        xi, q = fit_cumulant(r, 8, seed + 7200)
    elif method == "mlefft":
        xi, q = fit_mle_fft(r, seed + 7300)
    else:
        raise ValueError(method)
    row = dict(method=method, src=src, seed=seed, dgp=dgp, obj=q,
               elapsed_s=round(time.time() - t0, 1))
    for i, nm in enumerate(PNAMES):
        row[nm] = float(xi[i])
    return row


def agg_stage2(df):
    out = []
    for (method, src, dgp), sub in df.groupby(["method", "src", "dgp"]):
        tv = R.DGP_LAB[dgp]["xi"]
        r = dict(method=method, src=src, dgp=dgp, n=len(sub))
        for i, nm in enumerate(PNAMES):
            b = sub[nm] - tv[i]
            r[f"b_{nm}"] = f"{b.mean():+.3f}"
            r[f"r_{nm}"] = f"{np.sqrt((b ** 2).mean()):.3f}"
        out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["stage1", "stage2", "all"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.part in ("stage1", "all"):
        tasks = [(m, dgp, T, seed)
                 for m in ("ols1", "ols3", "lasso_cv", "stepwise_bic")
                 for (T, S) in ((10000, 8), (25000, 12))
                 for seed in [91300000 + s for s in range(S)]
                 for dgp in "ABC"]
        print(f"[BASE-I] {len(tasks)} tasks", flush=True)
        rows = [stage1_one(t) for t in tasks]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RES, "baselines_stage1.csv"),
                  index=False, encoding="utf-8-sig")
        agg = df.groupby(["method", "T", "dgp"]).agg(
            TPR=("tpr", "mean"), FDR=("fdr", "mean"), F1=("f1", "mean"),
            Wass=("wass", "mean"), MedSE=("medse", "mean")).reset_index()
        agg.to_csv(os.path.join(RES, "baselines_stage1_summary.csv"),
                   index=False, encoding="utf-8-sig")
        print(agg.to_string(index=False), flush=True)

    if args.part in ("stage2", "all"):
        # mlefft 最慢，排最前利于负载均衡
        tasks = [(m, src, dgp, seed)
                 for m in ("mlefft", "cgmm", "cmom")
                 for src in ("oracle", "plugin")
                 for dgp in "ABC"
                 for seed in [91300000 + s for s in range(12)]]
        # ---- 断点续跑：partial CSV 逐任务落盘，崩溃后可从断点继续 ----
        pf = os.path.join(RES, "baselines_stage2_partial.csv")
        fields = ["method", "src", "dgp", "seed", "obj", "error"] + PNAMES
        done_rows = []
        if os.path.exists(pf) and os.path.getsize(pf) > 0:
            _pd = pd.read_csv(pf)
            done_rows = _pd.to_dict("records")
        done_keys = {(r["method"], r["src"], r["dgp"], int(r["seed"]))
                     for r in done_rows}
        todo = [t for t in tasks if t not in done_keys]
        print(f"[BASE-II] {len(tasks)} tasks, {len(done_keys)} done, "
              f"{len(todo)} todo", flush=True)

        import csv as _csv
        fh = open(pf, "a", newline="", encoding="utf-8-sig")
        if os.path.getsize(pf) == 0:
            _csv.DictWriter(fh, fieldnames=fields).writeheader()
            fh.flush()

        def _persist(row):
            norm = {k: row.get(k, "") for k in fields}
            _csv.DictWriter(fh, fieldnames=fields).writerow(norm)
            fh.flush()
            os.fsync(fh.fileno())
            # 计时落盘（独立 CSV，不依赖 partial 表结构）：
            # 论文引用的"MLE-FFT 单次约 8.5 分钟 / ECF 秒级"等计时由此可溯源
            tf = os.path.join(RES, "baselines_stage2_timing.csv")
            new_f = not os.path.exists(tf) or os.path.getsize(tf) == 0
            with open(tf, "a", newline="", encoding="utf-8-sig") as tfh:
                w = _csv.DictWriter(
                    tfh, fieldnames=["ts", "method", "src", "dgp", "seed",
                                     "elapsed_s"])
                if new_f:
                    w.writeheader()
                w.writerow(dict(ts=pd.Timestamp.now().isoformat(),
                                method=row.get("method", ""),
                                src=row.get("src", ""),
                                dgp=row.get("dgp", ""),
                                seed=row.get("seed", ""),
                                elapsed_s=row.get("elapsed_s", "")))
            return row

        rows_new = []
        if args.workers > 1 and todo:
            from concurrent.futures import (ProcessPoolExecutor,
                                            as_completed)
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(stage2_one, t): t for t in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    rows_new.append(_persist(fut.result()))
                    if i % 12 == 0 or i == len(todo):
                        print(f"[BASE-II] progress {i}/{len(todo)}",
                              flush=True)
        else:
            for t in todo:
                rows_new.append(_persist(stage2_one(t)))
        fh.close()
        df = pd.DataFrame(done_rows + rows_new)
        df.to_csv(os.path.join(RES, "baselines_stage2.csv"),
                  index=False, encoding="utf-8-sig")
        agg = agg_stage2(df)
        agg.to_csv(os.path.join(RES, "baselines_stage2_summary.csv"),
                   index=False, encoding="utf-8-sig")
        print(agg.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
