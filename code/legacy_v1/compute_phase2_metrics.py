# -*- coding: utf-8 -*-
"""
阶段 II 补充指标：
 1. Oracle vs Plug-in 配对差异：对每个参数计算 (θ̂_oracle - θ̂_plug-in) 的均值/SD
 2. 对称 DGP1：log(Cm/Cp) 和 log(G/M) 是否接近 0
 3. 非对称 DGP2/3：log(Cm/Cp)≈log(2), log(G/M)≈log(3)
直接从 sim_full_results.csv 计算，无需重跑仿真。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIM_DIR = os.path.join(BASE, "06_stage_outputs", "stage_sim_results")
df = pd.read_csv(os.path.join(SIM_DIR, "sim_full_results.csv"))

PARAMS = ["mu", "Cp", "Cm", "M", "G", "alpha"]

# --- 1. 配对差异 (oracle - plug-in) ---
print("=" * 70)
print("Phase II: Oracle vs Plug-in 配对差异 (per DGP)")
print("=" * 70)
paired_rows = []
for dgp_id in [1, 2, 3]:
    sub = df[df.dgp_id == dgp_id]
    if len(sub) == 0:
        continue
    for p in PARAMS:
        diff = sub[f"or_asy_{p}"].values - sub[f"pl_asy_{p}"].values
        paired_rows.append(dict(
            dgp_id=dgp_id, dgp_name=sub.dgp_name.iloc[0], param=p,
            diff_mean=float(np.mean(diff)), diff_sd=float(np.std(diff, ddof=1)),
            diff_med=float(np.median(diff)), n=len(diff)
        ))
        print(f"  DGP{dgp_id} {p:>6s}: Δ(or-pl) mean={np.mean(diff):+.4f}  "
              f"sd={np.std(diff, ddof=1):.4f}  med={np.median(diff):+.4f}  n={len(diff)}")

paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(os.path.join(SIM_DIR, "phase2_paired_diff.csv"),
                 index=False, encoding="utf-8-sig")
print(f"\nSaved: phase2_paired_diff.csv")

# --- 2. 比率检验 ---
print("\n" + "=" * 70)
print("Phase II: 比率检验 log(Cm/Cp) 和 log(G/M)")
print("=" * 70)
ratio_rows = []
for dgp_id in [1, 2, 3]:
    sub = df[df.dgp_id == dgp_id]
    if len(sub) == 0:
        continue
    sym = sub.sym_dgp.iloc[0]
    true_cm_cp = sub.true_Cm.iloc[0] / sub.true_Cp.iloc[0]
    true_g_m = sub.true_G.iloc[0] / sub.true_M.iloc[0]
    for resid_type, prefix in [("oracle", "or"), ("plugin", "pl")]:
        cm = sub[f"{prefix}_asy_Cm"].values
        cp = sub[f"{prefix}_asy_Cp"].values
        g = sub[f"{prefix}_asy_G"].values
        m = sub[f"{prefix}_asy_M"].values
        log_cm_cp = np.log(np.maximum(cm / np.maximum(cp, 1e-12), 1e-12))
        log_g_m = np.log(np.maximum(g / np.maximum(m, 1e-12), 1e-12))
        true_log_cm_cp = np.log(true_cm_cp)
        true_log_g_m = np.log(true_g_m)
        ratio_rows.append(dict(
            dgp_id=dgp_id, resid=resid_type,
            log_CmCp_mean=float(np.mean(log_cm_cp)),
            log_CmCp_sd=float(np.std(log_cm_cp, ddof=1)),
            log_CmCp_true=true_log_cm_cp,
            bias_log_CmCp=float(np.mean(log_cm_cp) - true_log_cm_cp),
            log_GM_mean=float(np.mean(log_g_m)),
            log_GM_sd=float(np.std(log_g_m, ddof=1)),
            log_GM_true=true_log_g_m,
            bias_log_GM=float(np.mean(log_g_m) - true_log_g_m),
            n=len(log_cm_cp)
        ))
        tag = "对称" if sym else "非对称"
        print(f"  DGP{dgp_id} [{tag}] {resid_type:>6s}: "
              f"log(Cm/Cp)={np.mean(log_cm_cp):+.4f}±{np.std(log_cm_cp,ddof=1):.4f} "
              f"(true={true_log_cm_cp:+.4f}, bias={np.mean(log_cm_cp)-true_log_cm_cp:+.4f})  "
              f"log(G/M)={np.mean(log_g_m):+.4f}±{np.std(log_g_m,ddof=1):.4f} "
              f"(true={true_log_g_m:+.4f}, bias={np.mean(log_g_m)-true_log_g_m:+.4f})")

ratio_df = pd.DataFrame(ratio_rows)
ratio_df.to_csv(os.path.join(SIM_DIR, "phase2_ratio_tests.csv"),
                index=False, encoding="utf-8-sig")
print(f"\nSaved: phase2_ratio_tests.csv")

# --- 3. 汇总表：每个 DGP 的参数 Bias/SD/RMSE (oracle & plug-in) ---
print("\n" + "=" * 70)
print("Phase II: 参数 Bias/SD/RMSE 汇总 (asymmetric model)")
print("=" * 70)
param_rows = []
for dgp_id in [1, 2, 3]:
    sub = df[df.dgp_id == dgp_id]
    if len(sub) == 0:
        continue
    for p in PARAMS:
        true_val = sub[f"true_{p}"].iloc[0]
        for resid_type, prefix in [("oracle", "or"), ("plugin", "pl")]:
            est = sub[f"{prefix}_asy_{p}"].values
            bias = float(np.mean(est) - true_val)
            sd = float(np.std(est, ddof=1))
            rmse = float(np.sqrt(np.mean((est - true_val) ** 2)))
            param_rows.append(dict(
                dgp_id=dgp_id, param=p, resid=resid_type,
                bias=bias, sd=sd, rmse=rmse, true=true_val, n=len(est)
            ))
            print(f"  DGP{dgp_id} {p:>6s} {resid_type:>6s}: "
                  f"bias={bias:+.4f}  sd={sd:.4f}  rmse={rmse:.4f}  (true={true_val})")

param_df = pd.DataFrame(param_rows)
param_df.to_csv(os.path.join(SIM_DIR, "phase2_param_bias_rmse.csv"),
                index=False, encoding="utf-8-sig")
print(f"\nSaved: phase2_param_bias_rmse.csv")
