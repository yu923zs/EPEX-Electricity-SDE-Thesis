# -*- coding: utf-8 -*-
"""diag_alpha_full.py — 诊断全样本非对称 ECF 拟合 alpha 撞下界问题
1) 重建残差与 ECF 数据, 打印 sR 与 u 网格
2) 重跑 24 初值非对称拟合, 统计各初值终点(撞界比例, Q 分散)
3) 固定其余参数, 在 alpha 网格上做剖面 Q
4) 固定 alpha 于若干内点值, 优化其余 5 参, 比较 Q
"""
import os, sys, time
import numpy as np
import pandas as pd

_REPO = os.environ.get("EPEX_BASE") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, "..", "method_lab"))
for p in (HERE, LAB):
    if p not in sys.path:
        sys.path.insert(0, p)

from thesis_pipeline_common import load_pickle
from run_stage_C import (build_ecf_data, compute_Q, run_multi_start,
                         to_unconstrained, to_constrained, _ALPHA_LB, _ALPHA_UB)
from scipy import optimize

OUT = os.path.join(_REPO, "06_stage_outputs", "empirical_v1", "full")
artB = load_pickle(os.path.join(OUT, "full_stageB_artifacts.pkl"))
r = artB["resid"]; r = r[np.isfinite(r)]
dat = build_ecf_data(r)
print(f"n={len(r)}, sR={dat.sR:.3f}, u range=[{dat.u.min():.4f}, {dat.u.max():.2f}]")

t0 = time.time()
res = run_multi_start(dat, symmetric=False, n_start=24, seed=90201)
print(f"multi-start done in {time.time()-t0:.0f}s")
rows = []
for z in res:
    xi = z["xi_final"]
    rows.append(dict(Q=z["Q_final"], conv=z["converged"],
                     mu=xi[0], Cp=xi[1], Cm=xi[2], M=xi[3], G=xi[4], alpha=xi[5],
                     hits="/".join(z["bound_hits"])))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
n_bound = int((df["alpha"] < 0.01).sum())
print(f"\n撞下界(alpha<0.01)的初值数: {n_bound}/24")

best = res[0]["xi_final"]
print("\n--- 剖面 Q(alpha), 其余参数固定在最优 ---")
for a in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]:
    xi = best.copy(); xi[5] = a
    print(f"alpha={a:6.3f}  Q={compute_Q(xi, dat, False):.8f}")

print("\n--- 固定 alpha, 优化其余 5 参 ---")
for a_fix in [0.05, 0.1, 0.2, 0.3, 0.45]:
    def Q5(v):
        return compute_Q([v[0], v[1], v[2], v[3], v[4], a_fix], dat, False)
    x0 = best.copy(); x0[5] = a_fix
    rr = optimize.minimize(Q5, x0[:5], method="Nelder-Mead",
                           options={"maxiter": 1500, "fatol": 1e-9, "xatol": 1e-9})
    print(f"alpha={a_fix:5.2f} -> Q={rr.fun:.8f} (best free-alpha Q={res[0]['Q_final']:.8f})")
