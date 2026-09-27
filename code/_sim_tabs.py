import os
import pandas as pd, numpy as np
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(_BASE, "06_stage_outputs", "stage_sim_results", "sim_rows_raw.csv")
df = pd.read_csv(p)
print("N =", len(df))
print("cols sample =", df.columns.tolist()[:10], "...", len(df.columns), "total")

def g(x):
    arr = np.asarray(x, dtype=np.float64)
    m = float(np.nanmean(arr))
    n_ok = int(np.sum(~np.isnan(arr)))
    s = float(np.nanstd(arr, ddof=1)) if n_ok > 1 else 0.0
    return m, s

print("\n=== Tab 6.1 Sparse Recovery (mean +/- sd, S=10) ===")
for (dgp, T), grp in df.groupby(["dgp_name","T_levels"], sort=False):
    tpr_m, tpr_s = g(grp["tpr"]); fdr_m, fdr_s = g(grp["fdr"])
    me_m, me_s = g(grp["medse"])
    sz_m, sz_s = g(grp["size_selected"])
    print(f"{dgp:<28} T={T:>5d}  TPR={tpr_m:.3f}+/-{tpr_s:.3f}  FDR={fdr_m:.3f}+/-{fdr_s:.3f}  MedSE={me_m:.3f}+/-{me_s:.3f}  |S|={sz_m:.2f}+/-{sz_s:.2f}")

print("\n=== Tab 6.2 Plug-in TS bias (mean +/- sd) ===")
print("          bias_alpha (abs), bias_mu (abs), Cp/Cm/M/G (rel, true-denom)")
for (dgp, T), grp in df.groupby(["dgp_name","T_levels"], sort=False):
    ab_m, ab_s = g(grp["pl_a_xi6_5"] - grp["true_alpha"])
    mb_m, mb_s = g(grp["pl_a_xi6_0"] - grp["true_mu"])
    cp_m, cp_s = g((grp["pl_a_xi6_1"] - grp["true_Cp"]) / np.maximum(np.abs(grp["true_Cp"]), 1e-6))
    cm_m, cm_s = g((grp["pl_a_xi6_2"] - grp["true_Cm"]) / np.maximum(np.abs(grp["true_Cm"]), 1e-6))
    mm_m, mm_s = g((grp["pl_a_xi6_3"] - grp["true_M"]) / np.maximum(np.abs(grp["true_M"]), 1e-6))
    gg_m, gg_s = g((grp["pl_a_xi6_4"] - grp["true_G"]) / np.maximum(np.abs(grp["true_G"]), 1e-6))
    print(f"{dgp:<28} T={T:>5d}  a={ab_m:+.3f}+/-{ab_s:.3f}  m={mb_m:+.3f}+/-{mb_s:.3f}  Cp={cp_m:+.3f}+/-{cp_s:.3f}  Cm={cm_m:+.3f}+/-{cm_s:.3f}  M={mm_m:+.3f}+/-{mm_s:.3f}  G={gg_m:+.3f}+/-{gg_s:.3f}")

print("\n=== Tab 6.3 Plug-in Q ASY vs SYM ===")
for (dgp, T), grp in df.groupby(["dgp_name","T_levels"], sort=False):
    qa = grp["pl_a_Q"]; qs = grp["pl_s_Q"]
    ok = np.asarray(qa.notna() & qs.notna() & (qs > 1e-12))
    if ok.sum() == 0:
        print(f"{dgp:<28} T={T:>5d}  no valid"); continue
    qa = np.asarray(qa[ok], dtype=np.float64); qs = np.asarray(qs[ok], dtype=np.float64)
    r = np.clip(qa / qs, 1e-4, 1e4)
    frac = float(np.mean(qa <= qs))
    q05, qm, q95 = np.quantile(r, [0.05, 0.5, 0.95])
    print(f"{dgp:<28} T={T:>5d}  P(Qa<=Qs)={frac:.2f}  Qa/Qs med={qm:.3f}  90%CI=[{q05:.3f}, {q95:.3f}]  n={int(ok.sum())}")

print("\n=== Oracle - Plug-in gap on Q_asy (*1e4, mean +/- sd) ===")
for (dgp, T), grp in df.groupby(["dgp_name","T_levels"], sort=False):
    gap = (grp["pl_a_Q"] - grp["or_a_Q"]) * 1e4
    m, s = g(gap)
    print(f"{dgp:<28} T={T:>5d}  gap = {m:+.3f} +/- {s:.3f}  (x1e-4)")
