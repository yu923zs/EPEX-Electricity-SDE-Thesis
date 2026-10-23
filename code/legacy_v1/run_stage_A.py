# -*- coding: utf-8 -*-
"""
================================================================
 run_stage_A.py  — 阶段 A：数据装载 → 季节项 → 状态分箱 → drift 工作样本
================================================================
运行目录: 05_logs/；产物写 06_stage_outputs/stage_A_drift_train/
  - stageA_X_full.parquet            全样本三区间的 (ts_cet, P_t, mu_hat, X_t, split_label)
  - stageA_drift_bin_stats.csv       保留箱 (k, n_k, x_k, b̂_k, v̂_k, w_raw, w_k)
  - stageA_drift_train_artifacts.pkl 后续 B 阶段直接使用
  - stageA_report.txt                自检报告（行数/箱数/R²/覆盖率等）

默认超参（与章节 §3/§7 默认一致）:
  K=100, tau_q=0.95, n_min=30, winsor_pct=0.95,
  hod_nharm=5, knots_per_year=6, dt_hours=1.0
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from thesis_pipeline_common import (
    BASE, DIR_OUT, RAW_CSV, SPLIT_BOUNDS,
    load_smard_df, split_3way, fit_seasonal, build_drift_work_samples,
    ensure_dir, save_pickle,
)

# ---------- 超参（主规范默认值，稳健性附录 C 会覆盖） ----------
CFG = dict(
    K=100, tau_q=0.95, n_min=30, winsor_pct=0.95, eps_v=1e-6,
    hod_nharm=5, knots_per_year=6.0, dt_hours=1.0,
)
STAGE_A_DIR = ensure_dir(os.path.join(DIR_OUT, "stage_A_drift_train"))


def main():
    t0 = time.time()
    log_lines = []
    def L(x):
        log_lines.append(str(x))
        print(x)

    L("=" * 72)
    L("[Stage A] START  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    L(f"  Raw CSV     = {RAW_CSV}")
    L(f"  Output dir  = {STAGE_A_DIR}")
    L(f"  CFG         = {CFG}")
    L(f"  3-way splits= {SPLIT_BOUNDS}")

    # (1) 加载 + 三段划分
    df = load_smard_df()
    L(f"  Loaded rows after strict epoch filter: N = {len(df)}")
    splits = split_3way(df)
    for k, v in splits.items():
        L(f"    split[{k}] rows = {len(v):>6d}    CET range: {v['ts_cet'].min()} ~ {v['ts_cet'].max()}")
    N3 = sum(len(v) for v in splits.values())
    if N3 != len(df):
        # 注意：三段是不相交且不等于全集（SPLIT1/SPLIT2 间 2022H2-2024... 间为训练后到 TS 前？—— 不，在我们的划分里三段是严格不交的并集）
        L(f"  [WARN] Σ 3-way rows = {N3} ≠ total N = {len(df)}. "
          f"Diff = {len(df) - N3}. Expected = 0.")
    else:
        L("  [OK] Σ 3-way rows = total N = " + str(N3))

    # (2) 季节项拟合（只用 drift-train）
    train_sub = splits["drift_train"].reset_index(drop=True)
    # 注意：fit_seasonal(full_df=df) 需要 df 保持原始顺序的 index 对齐
    df_idx = df.reset_index(drop=True)
    train_reindexed = splits["drift_train"].reset_index(drop=True)
    # 对齐：用 train_reindexed 的 ts_cet 在 df_idx 中的位置
    fit_mask = df_idx["epoch_ms_utc"].isin(train_reindexed["epoch_ms_utc"])
    train_for_fit = df_idx.loc[fit_mask].reset_index(drop=True)
    seas = fit_seasonal(train_for_fit, df_idx,
                        hod_nharm=CFG["hod_nharm"],
                        knots_per_year=CFG["knots_per_year"])
    L(f"  Seasonal R2 (train-only) = {seas['R2_train']:.6f}")
    L(f"  Seasonal R2 (full, pseudo) = {seas['R2_full']:.6f}")
    L(f"  μ_t design features      = {len(seas['feature_names_lin'])} (incl intercept)")
    L(f"  Long-wave hinge knots_n  = {len(seas['lw_hinge_knots_days'])}  "
      f"(lw_df={seas.get('lw_df','na')}, hod_nharm={seas['hod_nharm']})")

    # (3) 把 X_t 列与 split label 挂回 df_idx → 保存完整 X_all parquet
    def label_for_row(epoch_ms, bounds):
        for name, (lo, hi) in bounds.items():
            from thesis_pipeline_common import cet_to_ms
            if cet_to_ms(lo) <= epoch_ms <= cet_to_ms(hi):
                return name
        return "between_splits"  # 理论上不应出现（三段无间隔划分）
    df_idx["mu_hat"] = seas["mu_all"].values
    df_idx["X"] = seas["X_all"].values
    df_idx["split"] = df_idx["epoch_ms_utc"].apply(lambda e: label_for_row(int(e), SPLIT_BOUNDS))
    L("  Split label counts:\n" +
      df_idx["split"].value_counts().to_string())

    X_path_csv = os.path.join(STAGE_A_DIR, "stageA_X_full.csv")
    # 写出前将 tz-aware 转为字符串，避免部分 parquet 引擎不支持 tz-aware
    out_df = df_idx.copy()
    out_df["ts_cet_str"] = out_df["ts_cet"].dt.strftime("%Y-%m-%d %H:%M:%S %z")
    cols_out = ["epoch_ms_utc", "ts_cet_str", "cet_date", "cet_is_dst_flag",
                "price_eur_mwh_raw", "mu_hat", "X", "split"]
    out_df[cols_out].to_csv(X_path_csv, index=False, encoding="utf-8-sig")
    L(f"  Saved X_full CSV → {X_path_csv}")

    # Optional parquet fallback (if pyarrow later installed)
    try:
        X_path_par = os.path.join(STAGE_A_DIR, "stageA_X_full.parquet")
        out_df[cols_out].to_parquet(X_path_par, engine="pyarrow")
        L(f"  Saved X_full parquet (optional) → {X_path_par}")
    except Exception:
        X_path_par = None
        L(f"  [INFO] pyarrow/fastparquet unavailable → skipped parquet; used CSV only.")

    # (4) 构建 drift 工作样本（仅 drift-train 区间的 X）
    X_train_all = df_idx.loc[df_idx["split"] == "drift_train", "X"].to_numpy(dtype=np.float64)
    L(f"  X_train for binning: N_X = {len(X_train_all)}   (finite = {np.isfinite(X_train_all).sum()})")
    dw = build_drift_work_samples(
        X_train=X_train_all,
        K=CFG["K"], tau_q=CFG["tau_q"], n_min=CFG["n_min"],
        winsor_pct=CFG["winsor_pct"], eps_v=CFG["eps_v"],
        dt_hours=CFG["dt_hours"],
    )
    L(f"  Bins: input K={dw['K_input']}  eff K={dw['K_effective']}  kept K={dw['K_kept']}")
    L(f"  Trunc quantile |ΔX|_τ = {dw['q_trunc']:.6f}   (τ={dw['tau_q']})")
    L(f"  Increments: total={dw['N_increments']}  kept_in_bin_after_cut={dw['N_kept_increments']}  "
      f"discarded_tail={dw['N_discards_tail']}")
    L(f"  n_k kept: min={dw['n_k'].min()}  median={np.median(dw['n_k'])}  max={dw['n_k'].max()}")
    L(f"  b̂_k     : min={dw['b_hat'].min():.6f}  median={np.median(dw['b_hat']):.6f}  max={dw['b_hat'].max():.6f}")
    L(f"  w_k sum  : {dw['w_k'].sum():.6f}  (must equal K_kept = {dw['K_kept']})")
    L(f"  w_k range: min={dw['w_k'].min():.4f}  max={dw['w_k'].max():.4f}")

    # (5) 保存 bin 统计 CSV
    bin_csv = pd.DataFrame({
        "k_1based": dw["keep_bin_label_1based"],
        "k_0based": dw["keep_bin_index_0based"],
        "a_k_left":   dw["bins"][dw["keep_bin_index_0based"]],
        "a_k_right":  dw["bins"][dw["keep_bin_index_0based"] + 1],
        "x_k": dw["x_k"],
        "b_hat": dw["b_hat"],
        "n_k": dw["n_k"],
        "v_hat": dw["v_hat"],
        "w_raw": dw["w_raw"],
        "w_k": dw["w_k"],
    })
    csv_path = os.path.join(STAGE_A_DIR, "stageA_drift_bin_stats.csv")
    bin_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    L(f"  Saved bin stats CSV → {csv_path}")

    # (6) 打包 artifacts.pkl（阶段 B 直接读）
    art = {
        "cfg": CFG,
        "split_rows": {k: len(v) for k, v in splits.items()},
        "seasonal": {k: (v if not isinstance(v, (pd.Series, pd.DataFrame))
                         else v.to_numpy(dtype=np.float64) if isinstance(v, pd.Series)
                         else v)
                     for k, v in seas.items() if k not in ("mu_all", "X_all")},
        "X_full_csv_path": X_path_csv,
        "X_full_parquet_path": X_path_par,
        "bin_stats_csv_path": csv_path,
        "bin_stats": bin_csv.to_dict(orient="list"),
        "drift_work": dw,
        "timing_sec": round(time.time() - t0, 3),
    }
    pkl_path = os.path.join(STAGE_A_DIR, "stageA_drift_train_artifacts.pkl")
    save_pickle(art, pkl_path)
    L(f"  Saved artifacts PKL → {pkl_path}")

    # (7) 写 stageA_report.txt
    rep_path = os.path.join(STAGE_A_DIR, "stageA_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    L(f"\n  [Stage A] DONE in {art['timing_sec']:.2f}s.  Report → {rep_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
