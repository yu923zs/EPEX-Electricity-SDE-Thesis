# -*- coding: utf-8 -*-
"""诊断 v11：触发簇 + 细化位置 + 是否命中真折点（8 EUR 容差）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from run_lab import gen_data, stageI, VARIANTS, DGP_LAB, profile_refine

dgp = sys.argv[1] if len(sys.argv) > 1 else "B"
vname = sys.argv[2] if len(sys.argv) > 2 else "v11"
for seed in range(91300000, 91300008):
    X_deseas, _ = gen_data(seed, dgp, 10000)
    s1 = stageI(X_deseas, VARIANTS[vname], seed)
    refined, _ = profile_refine(s1)
    true_cs = np.array([c for (c, _) in DGP_LAB[dgp]["hinges"]])
    info = []
    for cl in s1["clusters_sel"]:
        info.append(f"[{cl['c_lo']:.0f},{cl['c_hi']:.0f}]cp{cl['cpip']:.2f}")
    hits = [f"{r:.1f}{'OK' if len(true_cs) and np.min(np.abs(true_cs - r)) < 8 else 'X'}"
            for r in refined]
    print(f"seed={seed-91300000} true={np.round(true_cs,1).tolist()} "
          f"clusters={' '.join(info)} refined={' '.join(hits)}")
