# -*- coding: utf-8 -*-
"""诊断：v9 下各簇精确 cpip = P(簇内任一 theta=1)，定位真折点簇未触发原因。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from run_lab import gen_data, stageI, VARIANTS, DGP_LAB

dgp = sys.argv[1] if len(sys.argv) > 1 else "B"
vc = VARIANTS["v9"]
for seed in range(91300000, 91300008):
    X_deseas, _ = gen_data(seed, dgp, 10000)
    s1 = stageI(X_deseas, vc, seed)
    true_cs = np.array([c for (c, _) in DGP_LAB[dgp]["hinges"]])
    hm = s1["hinge_map"]; theta = s1["theta"]
    hs = sorted(hm.items(), key=lambda kv: kv[1])
    bw = vc["cluster_bw"]
    clusters, cur = [], [hs[0]]
    for it in hs[1:]:
        if it[1] - cur[-1][1] <= bw:
            cur.append(it)
        else:
            clusters.append(cur); cur = [it]
    clusters.append(cur)
    parts = [f"seed={seed - 91300000}"]
    for cl in clusters:
        cols = [j for (j, _) in cl]
        cs = [c for (_, c) in cl]
        cpip = float(theta[:, cols].any(axis=1).mean())
        near = len(true_cs) and min(abs(c - true_cs).min() for c in cs) < 8
        tag = "*T*" if near else ""
        parts.append(f"[{min(cs):.0f},{max(cs):.0f}]{cpip:.2f}{tag}")
    print(" ".join(parts))
