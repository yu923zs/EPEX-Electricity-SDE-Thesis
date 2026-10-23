# -*- coding: utf-8 -*-
"""诊断：查看指定 DGP/seed 下 Stage I 各候选列 PIP 与后验中位系数。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from run_lab import gen_data, stageI, VARIANTS, DGP_LAB

seed, dgp, vname = 91300000, "B", "v2"
if len(sys.argv) > 1:
    dgp = sys.argv[1]
if len(sys.argv) > 2:
    vname = sys.argv[2]
X_deseas, X_true = gen_data(seed, dgp, 10000)
s1 = stageI(X_deseas, VARIANTS[vname], seed)
pip = s1["pip"]
bm = s1["beta_med"]
true_knots = [c for (c, _) in DGP_LAB[dgp]["hinges"]]
for i, cn in enumerate(s1["colnames"]):
    is_true = cn.startswith("x^1")
    if i in s1["hinge_map"]:
        is_true = any(abs(s1["hinge_map"][i] - tc) < 8 for tc in true_knots)
    mark = " <== TRUE" if is_true else ""
    print(f"{i:2d} {cn:>18s}  PIP={pip[i]:.3f}  beta_med={bm[i]:+.4f}{mark}")
print("gamma_med =", round(s1["gamma_med"], 4))
print("X pct[1,5,50,95,99]:",
      np.round(np.percentile(X_deseas, [1, 5, 50, 95, 99]), 1).tolist())
print("true knots:", true_knots)
