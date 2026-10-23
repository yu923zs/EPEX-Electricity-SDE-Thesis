# -*- coding: utf-8 -*-
"""
================================================================
 run_vasicek.py — 实证章 Vasicek 零模型与 2x2 贡献分解
   零模型: 线性漂移 + 高斯增量 (经典 Vasicek/OU 基准)
     ΔX_i = a + b·X_i + ε_i,  ε ~ N(m_e, s_e^2)
   2x2 分解(同一批残差内可比):
     (漂移: Vasicek/本文) x (噪声: Gaussian/TS-ECF)
   诊断: 残差 sd/偏度/超额峰度/3sd 外次数/极值分位
输入: empirical_v1/full/full_X_full.csv (去季节序列)
      empirical_v1/full/full_stageB_artifacts.pkl (本文残差)
      empirical_v1/full/full_stageC_artifacts.pkl (本文 TS 最优 Q)
输出: empirical_v1/full/vasicek_*.csv
================================================================
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, "..", "method_lab"))
for p in (HERE, LAB):
    if p not in sys.path:
        sys.path.insert(0, p)

from thesis_pipeline_common import load_pickle  # noqa: E402

FULL = r"C:\Users\86183\Desktop\硕士论文3\06_stage_outputs\empirical_v1\full"
SEED = 90211


def gauss_Q(m, s, dat):
    """高斯特征函数在与 dat 相同频点上的 ECF 目标 Q."""
    u = dat.u
    damp = np.exp(-0.5 * (s * u) ** 2)
    cos_m = damp * np.cos(m * u)
    sin_m = damp * np.sin(m * u)
    return float(np.sum((dat.cosR - cos_m) ** 2)
                 + np.sum((dat.sinR - sin_m) ** 2)) / (2.0 * len(u))


def resid_stats(r):
    r = r[np.isfinite(r)]
    sd = float(np.std(r))
    z = (r - np.mean(r)) / sd
    return dict(
        sd=sd,
        skew=float(np.mean(z ** 3)),
        ex_kurt=float(np.mean(z ** 4) - 3.0),
        n_beyond_3sd=int((np.abs(z) > 3).sum()),
        min=float(np.min(r)),
        q005=float(np.quantile(r, 0.005)),
        q995=float(np.quantile(r, 0.995)),
        max=float(np.max(r)),
    )


def main():
    from run_stage_C import build_ecf_data, run_multi_start
    t0 = time.time()

    dfX = pd.read_csv(os.path.join(FULL, "full_X_full.csv"),
                      encoding="utf-8-sig")
    X = dfX["X"].to_numpy(dtype=np.float64)
    X = X[np.isfinite(X)]
    y = np.diff(X)
    x = X[:-1]
    n = len(y)

    # ---------- Vasicek: OLS 线性漂移 ----------
    A = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a_hat, b_hat = float(coef[0]), float(coef[1])
    e = y - A @ coef
    s2 = float(np.sum(e ** 2) / (n - 2))
    XtX_inv = np.linalg.inv(A.T @ A)
    se_ab = np.sqrt(np.diag(XtX_inv) * s2)
    t_ab = np.abs(coef) / se_ab

    # ---------- 本文漂移(全样本最终拟合) ----------
    artB = load_pickle(os.path.join(FULL, "full_stageB_artifacts.pkl"))
    r_ours = artB["resid"]
    artC = load_pickle(os.path.join(FULL, "full_stageC_artifacts.pkl"))
    Q_ours_TS = float(artC["best_asym"]["Q_final"])

    # ---------- 2x2: ECF Q ----------
    dat_e = build_ecf_data(e)                    # Vasicek 残差的 ECF 数据
    dat_r = build_ecf_data(r_ours)               # 本文残差的 ECF 数据
    Q_vas_gauss = gauss_Q(float(np.mean(e)), float(np.std(e)), dat_e)
    Q_ours_gauss = gauss_Q(float(np.mean(r_ours)),
                           float(np.std(r_ours)), dat_r)
    # Vasicek 残差上的 TS-ECF 拟合(24 初值)
    res_ts_e = run_multi_start(dat_e, symmetric=False, n_start=24, seed=SEED)
    Q_vas_TS = float(res_ts_e[0]["Q_final"])
    xi_ts_e = res_ts_e[0]["xi_final"]
    hits_e = "/".join(res_ts_e[0]["bound_hits"]) or "none"

    pd.DataFrame({
        "drift": ["Vasicek(线性)", "Vasicek(线性)", "本文(分段)", "本文(分段)"],
        "noise": ["Gaussian", "TS-ECF", "Gaussian", "TS-ECF"],
        "Q": [Q_vas_gauss, Q_vas_TS, Q_ours_gauss, Q_ours_TS],
        "rms_pct": [100 * np.sqrt(Q_vas_gauss), 100 * np.sqrt(Q_vas_TS),
                    100 * np.sqrt(Q_ours_gauss), 100 * np.sqrt(Q_ours_TS)],
    }).to_csv(os.path.join(FULL, "vasicek_2x2_Q.csv"),
              index=False, encoding="utf-8-sig")

    # ---------- 残差诊断对比 ----------
    st_e = resid_stats(e)
    st_r = resid_stats(r_ours)
    pd.DataFrame({
        "metric": list(st_e.keys()),
        "vasicek_resid": list(st_e.values()),
        "ours_resid": list(st_r.values()),
    }).to_csv(os.path.join(FULL, "vasicek_resid_diag.csv"),
              index=False, encoding="utf-8-sig")

    # ---------- 汇总 ----------
    pd.DataFrame({
        "metric": ["n", "a_hat", "b_hat", "se_a", "se_b", "abs_t_a",
                   "abs_t_b", "resid_sd_vasicek",
                   "ts_on_vasicek_mu", "ts_on_vasicek_Cp", "ts_on_vasicek_Cm",
                   "ts_on_vasicek_M", "ts_on_vasicek_G", "ts_on_vasicek_alpha",
                   "ts_on_vasicek_bound_hits"],
        "value": [n, a_hat, b_hat, float(se_ab[0]), float(se_ab[1]),
                  float(t_ab[0]), float(t_ab[1]), st_e["sd"],
                  xi_ts_e[0], xi_ts_e[1], xi_ts_e[2],
                  xi_ts_e[3], xi_ts_e[4], xi_ts_e[5], hits_e],
    }).to_csv(os.path.join(FULL, "vasicek_null_summary.csv"),
              index=False, encoding="utf-8-sig")

    # ---------- 漂移曲线对比(绘图数据源) ----------
    grid = np.linspace(np.quantile(x, 0.001), np.quantile(x, 0.999), 400)
    fd = pd.read_csv(os.path.join(FULL, "full_final_drift.csv"),
                     encoding="utf-8-sig")
    co = dict(zip(fd["term"], fd["coef"]))
    b_ours = (co["intercept"] + co["x^1"] * grid
              + co["(x--25.37)_+"] * np.maximum(grid + 25.37, 0.0)
              + co["(x-2.31)_+"] * np.maximum(grid - 2.31, 0.0))
    pd.DataFrame({
        "x": grid,
        "b_vasicek": a_hat + b_hat * grid,
        "b_ours": b_ours,
    }).to_csv(os.path.join(FULL, "vasicek_drift_curve.csv"),
              index=False, encoding="utf-8-sig")

    print(f"[Vasicek] a={a_hat:.4f} (|t|={t_ab[0]:.1f}), "
          f"b={b_hat:.5f} (|t|={t_ab[1]:.1f}), resid_sd={st_e['sd']:.3f}")
    print(f"[2x2] Q: Vas+G={Q_vas_gauss:.6e}, Vas+TS={Q_vas_TS:.6e}, "
          f"Ours+G={Q_ours_gauss:.6e}, Ours+TS={Q_ours_TS:.6e}")
    print(f"[TS-on-Vasicek] xi={np.round(xi_ts_e, 4).tolist()}, hits={hits_e}")
    print(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
