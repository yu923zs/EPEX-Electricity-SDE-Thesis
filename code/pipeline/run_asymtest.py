# -*- coding: utf-8 -*-
"""
================================================================
 run_asymtest.py — 实证章非对称性检验(参数自助零分布, 进程并行)
   H0: 对称 TS (C+=C-, M=G) 为真
   统计量: dQ = Q_sym - Q_asy (全样本 ECF 最小距离目标差)
   零分布: 在对称拟合参数下模拟 K=200 组同长残差, 每组重估两模型得 dQ*
   p 值: (1 + #{dQ* >= dQ_obs}) / (K_valid + 1)
输入: empirical_v1/full/full_stageB_artifacts.pkl (残差)
      empirical_v1/full/full_stageC_artifacts.pkl (两模型最优解)
输出: empirical_v1/full/asymtest_null.csv (逐重复, 断点续跑)
      empirical_v1/full/asymtest_summary.csv
================================================================
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

_REPO = os.environ.get("EPEX_BASE") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, "..", "method_lab"))
for p in (HERE, LAB):
    if p not in sys.path:
        sys.path.insert(0, p)

from thesis_pipeline_common import load_pickle  # noqa: E402

OUT_DIR = os.path.join(_REPO, "06_stage_outputs", "empirical_v1", "full")
ASY_K = 200
ASY_WORKERS = 6
LOG_PATH = r"C:\Users\86183\Desktop\硕士论文3\06_stage_outputs\empirical_v1\emp_run.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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
        return best_q

    q_s = quick_fit(xi_sym6, True)
    q_a = quick_fit(xi_asy6, False)
    return (k, q_s, q_a, q_s - q_a)


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    art = load_pickle(os.path.join(OUT_DIR, "full_stageC_artifacts.pkl"))
    artB = load_pickle(os.path.join(OUT_DIR, "full_stageB_artifacts.pkl"))
    r = artB["resid"]
    r = r[np.isfinite(r)]
    n = len(r)
    xi_a = np.asarray(art["best_asym"]["xi_final"], float)
    xi_s4 = np.asarray(art["best_sym"]["xi_final"], float)
    xi_s6 = np.array([xi_s4[0], xi_s4[1], xi_s4[1], xi_s4[2], xi_s4[2], xi_s4[3]])
    dq_obs = float(art["best_sym"]["Q_final"] - art["best_asym"]["Q_final"])
    log(f"[ASY] dQ_obs = {dq_obs:.6f}, n = {n}, K = {ASY_K}")

    null_csv = os.path.join(OUT_DIR, "asymtest_null.csv")
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
    }).to_csv(os.path.join(OUT_DIR, "asymtest_summary.csv"),
              index=False, encoding="utf-8-sig")
    log(f"[ASY] p = {p_val:.4f} (K_valid={len(res)}), 总用时 "
        f"{(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
