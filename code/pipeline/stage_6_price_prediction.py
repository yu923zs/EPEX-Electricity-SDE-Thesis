# -*- coding: utf-8 -*-
"""
================================================================
 stage_6_price_prediction.py — §7.7 原始电价 1-step ahead 预测对比
================================================================
范式: Filtering (1-step ahead conditional)
  P̂_{t+1|t} = μ̂_{t+1} + X_t + b̂(X_t)·Δt  (点预测中位)
  区间 = P̂_med + TS_α/2,  TS_{1-α/2}        (区间预测)

5 个方法:
  M1: Persistence            ŷ_{t+1} = y_t
  M2: SeasonalNaive          ŷ_{t+1} = y_{t-168}  (上周同时段)
  M3: WeeklyHourDOWMean      训练段按 (hour, dow) 聚合 → 外推
  M4: SARIMA(1,0,1)(1,0,1,24) statsmodels
  M5: ThesisPipeline         μ̂ + X_t + b̂(X_t)·Δt + TS 分位数

评估:
  L1 点预测: MAE, RMSE, MAPE
  L2 区间:   Pinball@1/5/50/95/99%, CRPS
  L3 显著性: Diebold-Mariano 矩阵

输出:
  stage_6_point_metrics.csv
  stage_6_pinball_crps.csv
  stage_6_dm_pvalues.csv
  stage_6_report.txt
  figures/fig7_5_price_oos_compare.pdf  (4 季度子图)
================================================================
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
from scipy import stats as _s
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from thesis_pipeline_common import load_smard_df, split_3way, fit_seasonal
from stage_5_oos_coverage_v2 import (
    seasonal_predict_from_coefs, reconstruct_b_med,
    cdf_at_x, quantile,
)

BASE = os.path.abspath(os.path.join(HERE, ".."))
DIR_A = os.path.join(BASE, "06_stage_outputs", "stage_A_drift_train")
DIR_B = os.path.join(BASE, "06_stage_outputs", "stage_B_sparse_drift")
DIR_C = os.path.join(BASE, "06_stage_outputs", "stage_C_ts_ecf")
DIR_FIG = os.path.join(BASE, "06_stage_outputs_fix_20261011")
DIR_OUT = os.path.join(BASE, "06_stage_outputs_fix_20261011", "stage_6_price_pred")
os.makedirs(DIR_OUT, exist_ok=True)

OUT_POINT = os.path.join(DIR_OUT, "stage_6_point_metrics.csv")
OUT_PINB  = os.path.join(DIR_OUT, "stage_6_pinball_crps.csv")
OUT_DM    = os.path.join(DIR_OUT, "stage_6_dm_pvalues.csv")
OUT_REP   = os.path.join(DIR_OUT, "stage_6_report.txt")
OUT_FIG   = os.path.join(DIR_FIG, "fig7_5_price_oos_compare.pdf")


def L(*args): print("[Stage6]", *args, flush=True)


# ============================================================
#  L1 / L2 评估指标
# ============================================================
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

def pinball_loss(y_true, q_lo, q_hi, alpha_lo):
    """alpha_lo: 下尾分位数 (如 0.05)
       y_true: 真实值; q_lo: α 分位数预测; q_hi: 1-α 分位数预测
       返回双侧 pinball: 平均(下尾 pinball + 上尾 pinball)/2
    """
    pl_lo = np.maximum((alpha_lo - 1) * (y_true - q_lo), alpha_lo * (y_true - q_lo))
    pl_hi = np.maximum((1 - alpha_lo) * (y_true - q_hi), alpha_lo * (q_hi - y_true))
    return float(np.mean(0.5 * (pl_lo + pl_hi)))

def crps_general(y_true, q_vals, p_vals):
    """CRPS 的分位数近似（Matheson--Winkler 量化，恰当评分）。

       FIX(S3) 2026-10-11:
       原实现权重和仅 0.505 且使用 Σ w|q-y|（缺 pinball 方向性项），
       既非恰当评分也非 CRPS 的合法近似。此处改为标准形式
           CRPS ≈ Σ_j w_j · 2 ρ_{τ_j}(q_j, y),
           ρ_τ(q,y) = (1{y<q} − τ)(q − y),
       梯形权重（端点质量延伸至 0 与 1，端点区间内分位视为常数）：
           w_1   = (τ_1 + τ_2)/2,
           w_j   = (τ_{j+1} − τ_{j−1})/2  (内部),
           w_J   = 1 − (τ_{J−1} + τ_J)/2,
       权重和恒为 1。退化情形 q_j≡p 时 CRPS = |y−p|（= MAE），
       与点预测 CRPS 的理论性质一致。

       q_vals: 多个分位数预测 (n_samples × n_q)，列按 p_vals 递增
       p_vals: 对应的累积概率 (n_q,)，须 0<τ_1<...<τ_J<1
    """
    q = np.asarray(q_vals, dtype=np.float64)
    tau = np.asarray(p_vals, dtype=np.float64).reshape(-1)
    J = len(tau)
    if J < 2:
        raise ValueError("crps_general 需要至少 2 个分位数")
    if np.any(np.diff(tau) <= 0):
        raise ValueError("p_vals 必须严格递增")
    w = np.empty(J, dtype=np.float64)
    w[0] = 0.5 * (tau[0] + tau[1])
    w[-1] = 1.0 - 0.5 * (tau[-2] + tau[-1])
    for j in range(1, J - 1):
        w[j] = 0.5 * (tau[j + 1] - tau[j - 1])
    y = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    indicator = (y < q).astype(np.float64)
    pinball = (indicator - tau[None, :]) * (q - y)   # ρ_τ ≥ 0
    return float(np.mean(np.sum(2.0 * w[None, :] * pinball, axis=1)))


# ============================================================
#  Diebold-Mariano 检验
# ============================================================
def dm_test(e1, e2, loss_fn=np.abs, h=1):
    """Diebold-Mariano 检验：H0: E[L1]=E[L2]
       e1, e2: 两个方法的预测误差
       返回 (DM 统计量, p 值)
    """
    l1 = loss_fn(e1); l2 = loss_fn(e2)
    d = l1 - l2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var < 1e-12:
        return 0.0, 1.0
    # 长方差 Newey-West (h 步前视)
    n = len(d)
    if h > 1 and n > 2 * h:
        # 简化 NW：以 h 阶滞后自协方差
        d_centered = d - d_mean
        gamma_h = np.sum(d_centered[h:] * d_centered[:-h]) / n
        d_var_nw = d_var + 2 * gamma_h
        if d_var_nw < 1e-12:
            d_var_nw = d_var
    else:
        d_var_nw = d_var
    dm_stat = d_mean / np.sqrt(d_var_nw / n)
    p_val = 2 * (1 - _s.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_val)


# ============================================================
#  基准模型 M1-M3 (M4 SARIMA 单独函数)
# ============================================================
def baseline_persistence(P_test):
    """M1: ŷ_{t+1} = y_t"""
    return P_test[:-1].copy()

def baseline_seasonal_naive(P_test, lag=168):
    """M2: ŷ_{t+1} = y_{t-168} (上周同时段)"""
    n = len(P_test) - 1
    pred = np.full(n, np.nan)
    if lag < n:
        pred[lag:] = P_test[:-lag-1][:n-lag] if False else P_test[:n-lag]
    # 实际: pred[i] = P_test[i - 168 + 1] for i>=167, since pred[i] is for P_test[i+1]
    # 重新写清楚
    pred = np.full(n, np.nan)
    for i in range(n):
        idx_lag = i + 1 - lag  # index in P_test for "上周同时段" of P_test[i+1]
        if idx_lag >= 0:
            pred[i] = P_test[idx_lag]
    # 用 persistence 填前 168 个 NaN
    nan_mask = np.isnan(pred)
    pred[nan_mask] = P_test[:-1][nan_mask]
    return pred

def baseline_weekly_hour_dow_mean(train_price, test_price):
    """M3: 训练段按 (hour, dow) 聚合 → 外推到测试段"""
    tr = train_price.copy()
    tr["hour"] = pd.to_datetime(tr["ts_cet"]).dt.hour
    tr["dow"]  = pd.to_datetime(tr["ts_cet"]).dt.dayofweek
    grp = tr.groupby(["hour", "dow"])["price_eur_mwh_raw"].mean()
    te = test_price.copy()
    te["hour"] = pd.to_datetime(te["ts_cet"]).dt.hour
    te["dow"]  = pd.to_datetime(te["ts_cet"]).dt.dayofweek
    pred = te.apply(lambda r: grp.get((r["hour"], r["dow"]), np.nan), axis=1).to_numpy()
    # 用全局均值填 NaN（理论上不应有）
    if np.isnan(pred).any():
        pred = np.where(np.isnan(pred), tr["price_eur_mwh_raw"].mean(), pred)
    return pred[1:]  # FIX(S2): P_{t+1} 应使用 t+1 时刻的 (hour,dow) 组均值


def baseline_sarima(train_price, test_price, max_iter=100):
    """M4: SARIMA(1,0,1)(1,0,1,24) via statsmodels
       若 statsmodels 不可用 → 退化为 SAR(1)×季节(24) 简化版本（条件最小二乘估计）
       单次拟合训练段末尾 4000 点，1-step ahead rolling forecast
    """
    train_y = train_price["price_eur_mwh_raw"].to_numpy(dtype=np.float64)[-4000:]
    test_y  = test_price["price_eur_mwh_raw"].to_numpy(dtype=np.float64)

    # 尝试 statsmodels
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        L("[M4 SARIMA] fitting on last 4000 train points ...")
        try:
            m = SARIMAX(train_y,
                        order=(1, 0, 1),
                        seasonal_order=(1, 0, 1, 24),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False, maxiter=max_iter)
            L("[M4 SARIMA] 1-step rolling forecast ...")
            hist = list(train_y)
            preds = []
            for i in range(len(test_y) - 1):
                if i > 0 and i % 500 == 0:
                    L(f"  SARIMA at i={i}/{len(test_y)-1}")
                try:
                    f = m.apply(hist).forecast(steps=1)[0]
                except Exception:
                    f = hist[-1]
                preds.append(f)
                hist.append(test_y[i])
            return np.array(preds)
        except Exception as e:
            L(f"[WARN] SARIMA fit failed: {e}, fallback to simplified SAR(1)×seasonal(24)")
    except ImportError:
        L("[WARN] statsmodels not available, using simplified SAR(1)×seasonal(24) baseline")

    # 简化版本：AR(1) + 季节项（前 24 lag）条件最小二乘
    # 模型：y_t = φ * y_{t-1} + ψ * y_{t-24} + c
    L("[M4 Simplified AR(1)×seasonal(24)] fitting on last 4000 train points ...")
    Y = train_y[24:]
    X = np.column_stack([
        np.ones(len(Y)),
        train_y[:-24],   # AR(1) lagged 1 step? No, here we use lag=1 from y_{t-1}
    ])
    # 实际：用 lag 1 + lag 24 两个 AR 项
    Y_t = train_y[24:]
    X_lag1 = train_y[23:-1]   # y_{t-1}
    X_lag24 = train_y[:-24]    # y_{t-24}
    X = np.column_stack([np.ones(len(Y_t)), X_lag1, X_lag24])
    beta, *_ = np.linalg.lstsq(X, Y_t, rcond=None)
    c, phi1, phi24 = beta
    L(f"  simplified AR coef: c={c:.3f}, φ1={phi1:.3f}, φ24={phi24:.3f}")
    L("[M4 Simplified] 1-step rolling forecast ...")
    hist = list(train_y)
    preds = []
    for i in range(len(test_y) - 1):
        if i > 0 and i % 500 == 0:
            L(f"  Simplified AR at i={i}/{len(test_y)-1}")
        if len(hist) >= 24:
            f = c + phi1 * hist[-1] + phi24 * hist[-24]
        else:
            f = hist[-1]
        preds.append(f)
        hist.append(test_y[i])
    return np.array(preds)


# ============================================================
#  Thesis Pipeline M5
# ============================================================
def thesis_pipeline_predict(artB, artC, seas, train_price, test_price, x_min, x_max):
    """M5: μ̂ + X_t + b̂(X_t)·Δt (点预测中位) + TS 分位数 (区间)
       返回 dict: med (n,), q_lo_5 (n,), q_hi_95 (n,), q_lo_1 (n,), q_hi_99 (n,)
    """
    ts_test = test_price["ts_cet"].reset_index(drop=True)
    P_test = test_price["price_eur_mwh_raw"].to_numpy(dtype=np.float64).ravel()
    mu_test = seasonal_predict_from_coefs(seas, ts_test)
    X_test = P_test - mu_test
    # 1-step ahead: 用 X_t 已知, 预测 ΔX_{t+1|t} = b̂(X_t)·Δt (TS 中位数为 mu_drift=0)
    X_level = X_test[:-1]
    X_clamped = np.clip(X_level, x_min, x_max)
    b_med = reconstruct_b_med(X_clamped, artB)
    # 点预测中位: P̂_{t+1|t} = μ̂_{t+1} + X_t + b̂(X_t)
    # 注意 mu_test[1:] 对应 t+1 时刻
    P_med = mu_test[1:] + X_level + b_med

    # TS 分位数（增量层面，所以直接加在 P_med 上）
    out = {"med": P_med}
    # 取 ASY 模型（主模型）
    xi_asy = artC["best_asymmetric"]["xi_final"]
    L("[M5 Thesis] computing TS quantiles (ASY) ...")
    tA = time.time()
    quantiles_needed = [0.005, 0.025, 0.5, 0.975, 0.995]
    q_cache = {}
    for p in quantiles_needed:
        q = quantile(p, xi_asy, symmetric=False)
        q_cache[p] = q
        L(f"  ASY TS p={p:.4f} → q={q:+.2f}  ({time.time()-tA:.1f}s)")
    out["q_lo_1"]  = P_med + q_cache[0.005]
    out["q_lo_5"]   = P_med + q_cache[0.025]
    out["q_med"]    = P_med + q_cache[0.5]
    out["q_hi_95"]  = P_med + q_cache[0.975]
    out["q_hi_99"]  = P_med + q_cache[0.995]
    return out


# ============================================================
#  主流程
# ============================================================
def main():
    t0 = time.time()
    L("loading data & artifacts ...")
    df_all = load_smard_df()
    splits = split_3way(df_all)
    train_price = splits["drift_train"].reset_index(drop=True)
    test_price  = splits["test"].reset_index(drop=True)
    seas = fit_seasonal(train_price, df_all.reset_index(drop=True))
    with open(os.path.join(DIR_B, "stageB_run_artifacts.pkl"), "rb") as f:
        artB = pickle.load(f)
    with open(os.path.join(DIR_C, "stageC_run_artifacts.pkl"), "rb") as f:
        artC = pickle.load(f)
    df_bin = pd.read_csv(os.path.join(DIR_A, "stageA_drift_bin_stats.csv"))
    x_min = float(df_bin["x_k"].min()) - 0.05 * max(abs(df_bin["x_k"].min()), 1.0)
    x_max = float(df_bin["x_k"].max()) + 0.05 * max(abs(df_bin["x_k"].max()), 1.0)
    L(f"drift clamp [{x_min:.2f}, {x_max:.2f}]")

    P_test = test_price["price_eur_mwh_raw"].to_numpy(dtype=np.float64).ravel()
    P_true_next = P_test[1:]  # 真实 P_{t+1}
    n = len(P_true_next)
    L(f"test n={n}, span={test_price['ts_cet'].iloc[0]} → {test_price['ts_cet'].iloc[-1]}")

    # ===== 5 方法预测 =====
    L("\n=== M1 Persistence ===")
    tA = time.time()
    pred_M1 = baseline_persistence(P_test)
    L(f"M1 done ({time.time()-tA:.1f}s)")

    L("\n=== M2 SeasonalNaive(168) ===")
    tA = time.time()
    pred_M2 = baseline_seasonal_naive(P_test, lag=168)
    L(f"M2 done ({time.time()-tA:.1f}s)")

    L("\n=== M3 WeeklyHourDOWMean ===")
    tA = time.time()
    pred_M3 = baseline_weekly_hour_dow_mean(train_price, test_price)
    L(f"M3 done ({time.time()-tA:.1f}s)")

    L("\n=== M4 SARIMA(1,0,1)(1,0,1,24) ===")
    tA = time.time()
    pred_M4 = baseline_sarima(train_price, test_price)
    if pred_M4 is None:
        L("[WARN] M4 failed, fallback to M1 persistence")
        pred_M4 = pred_M1.copy()
    L(f"M4 done ({time.time()-tA:.1f}s)")

    L("\n=== M5 ThesisPipeline (ASY) ===")
    tA = time.time()
    out_M5 = thesis_pipeline_predict(artB, artC, seas, train_price, test_price,
                                     x_min, x_max)
    pred_M5 = out_M5["med"]
    L(f"M5 done ({time.time()-tA:.1f}s)")

    methods = {
        "M1_Persistence":   {"pred": pred_M1, "has_interval": False},
        "M2_SeasonalNaive": {"pred": pred_M2, "has_interval": False},
        "M3_WeeklyHrDOW":   {"pred": pred_M3, "has_interval": False},
        "M4_SARIMA":        {"pred": pred_M4, "has_interval": False},
        "M5_ThesisASY":     {"pred": pred_M5, "has_interval": True,
                              "q_lo_5": out_M5["q_lo_5"], "q_hi_95": out_M5["q_hi_95"],
                              "q_lo_1": out_M5["q_lo_1"], "q_hi_99": out_M5["q_hi_99"]},
    }

    # ===== L1: 点预测精度 =====
    L("\n=== L1 Point metrics ===")
    rows_pt = []
    for name, m in methods.items():
        e = P_true_next - m["pred"]
        rows_pt.append({
            "method": name,
            "MAE":  mae(P_true_next, m["pred"]),
            "RMSE": rmse(P_true_next, m["pred"]),
            "MAPE": mape(P_true_next, m["pred"]),
        })
    df_pt = pd.DataFrame(rows_pt)
    df_pt.to_csv(OUT_POINT, index=False, encoding="utf-8-sig")
    L(df_pt.to_string(index=False))

    # ===== L2: Pinball + CRPS =====
    L("\n=== L2 Pinball + CRPS ===")
    # 仅 M5 有分位数预测；其他方法用 persistence 风格的"经验分位数"= 点预测 ± 经验残差分位数
    # 简化处理: 对没有区间的方法，用训练残差经验分位数 (Fallback)
    train_P = train_price["price_eur_mwh_raw"].to_numpy()
    train_resid = train_P[1:] - train_P[:-1]
    emp_q_lo_1  = np.quantile(train_resid, 0.005)
    emp_q_lo_5  = np.quantile(train_resid, 0.025)
    emp_q_hi_95 = np.quantile(train_resid, 0.975)
    emp_q_hi_99 = np.quantile(train_resid, 0.995)

    pinball_alphas = [0.01, 0.05, 0.50]  # 双侧 α/2
    rows_pb = []
    for name, m in methods.items():
        if m["has_interval"]:
            q_lo_1, q_lo_5 = m["q_lo_1"], m["q_lo_5"]
            q_hi_99, q_hi_95 = m["q_hi_99"], m["q_hi_95"]
            q_med = out_M5["q_med"]
        else:
            # Fallback: 点预测 ± 训练残差经验分位数
            p = m["pred"]
            q_lo_1  = p + emp_q_lo_1
            q_lo_5  = p + emp_q_lo_5
            q_hi_95 = p + emp_q_hi_95
            q_hi_99 = p + emp_q_hi_99
            q_med   = p
        # Pinball 双侧
        pb_99  = pinball_loss(P_true_next, q_lo_1,  q_hi_99, alpha_lo=0.005)
        pb_95  = pinball_loss(P_true_next, q_lo_5,  q_hi_95, alpha_lo=0.025)
        pb_50  = pinball_loss(P_true_next, q_med,   q_med,  alpha_lo=0.5)  # 退化为 |y - med|
        # CRPS（基于多个分位数）
        q_stack = np.column_stack([q_lo_1, q_lo_5, q_med, q_hi_95, q_hi_99])
        p_vals  = np.array([0.005, 0.025, 0.5, 0.975, 0.995])
        crps = crps_general(P_true_next, q_stack, p_vals)
        rows_pb.append({
            "method": name,
            "Pinball_99": pb_99,
            "Pinball_95": pb_95,
            "Pinball_50": pb_50,
            "CRPS": crps,
        })
    df_pb = pd.DataFrame(rows_pb)
    df_pb.to_csv(OUT_PINB, index=False, encoding="utf-8-sig")
    L(df_pb.to_string(index=False))

    # ===== L3: Diebold-Mariano =====
    L("\n=== L3 DM test ===")
    method_names = list(methods.keys())
    n_m = len(method_names)
    dm_mat = np.full((n_m, n_m), np.nan)
    for i in range(n_m):
        ei = P_true_next - methods[method_names[i]]["pred"]
        for j in range(n_m):
            if i == j:
                dm_mat[i, j] = 1.0
                continue
            ej = P_true_next - methods[method_names[j]]["pred"]
            _, p = dm_test(ei, ej, loss_fn=np.abs, h=1)
            dm_mat[i, j] = p
    df_dm = pd.DataFrame(dm_mat, index=method_names, columns=method_names)
    df_dm.index.name = "method_i \\ method_j"
    df_dm.to_csv(OUT_DM, encoding="utf-8-sig")
    L(df_dm.round(3).to_string())

    # ===== 图 7.5: 4 季度子图 =====
    L("\n=== Generating Fig 7.5 (4 quarterly subplots) ===")
    ts_test = pd.to_datetime(test_price["ts_cet"].reset_index(drop=True))
    if not HAS_MPL:
        L("[WARN] matplotlib not available, skip figure generation")
        # 把预测数组保存为 npz 供后续单独画图
        np.savez(os.path.join(DIR_OUT, "stage_6_predictions.npz"),
                 ts_next=pd.to_datetime(test_price["ts_cet"].reset_index(drop=True)).iloc[1:].values,
                 P_true_next=P_true_next,
                 M1=pred_M1, M2=pred_M2, M3=pred_M3, M4=pred_M4, M5=pred_M5,
                 q_lo_5=out_M5["q_lo_5"], q_hi_95=out_M5["q_hi_95"])
        L(f"saved {os.path.join(DIR_OUT, 'stage_6_predictions.npz')} for offline plotting")
    else:
        quarter_edges = pd.date_range(ts_test.iloc[0], ts_test.iloc[-1], periods=5)
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False)
        palette = {"M1_Persistence": "#999999",
                   "M2_SeasonalNaive": "#66c2a5",
                   "M3_WeeklyHrDOW": "#fc8d62",
                   "M4_SARIMA": "#8da0cb",
                   "M5_ThesisASY": "#e41a1c"}
        labels = {"M1_Persistence": "M1 Persistence",
                  "M2_SeasonalNaive": "M2 SeasonalNaive",
                  "M3_WeeklyHrDOW": "M3 WeeklyHourDOWMean",
                  "M4_SARIMA": "M4 SARIMA",
                  "M5_ThesisASY": "M5 Thesis (ASY)"}
        ts_next = ts_test.iloc[1:].reset_index(drop=True)
        for k in range(4):
            ax = axes[k]
            mask = (ts_next >= quarter_edges[k]) & (ts_next < quarter_edges[k+1])
            idx = np.where(mask.to_numpy())[0]
            ax.plot(ts_next.iloc[idx], P_true_next[idx], color="black",
                    linewidth=0.8, label="真实 P$_{t+1}$")
            for name, m in methods.items():
                ax.plot(ts_next.iloc[idx], m["pred"][idx], color=palette[name],
                        linewidth=0.7, alpha=0.85, label=labels[name])
            ax.set_title(f"Q{k+1}: {quarter_edges[k].strftime('%Y-%m-%d')} → "
                         f"{quarter_edges[k+1].strftime('%Y-%m-%d')}",
                         fontsize=9)
            ax.set_ylabel("EUR/MWh", fontsize=8)
            ax.grid(alpha=0.3, lw=0.4)
            if k == 0:
                ax.legend(loc="upper right", fontsize=6.5, ncol=3)
        axes[-1].set_xlabel("时间 (CET)", fontsize=8)
        fig.suptitle("图 7.5 测试段（2025-01..09）1h-ahead 原始电价 P$_{t+1}$ 预测对比",
                     y=0.995, fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
        L(f"saved {OUT_FIG}")

    # FIX(S3): 无论是否可用 matplotlib，均保存预测数组供离线绘图脚本使用
    np.savez(os.path.join(DIR_OUT, "stage_6_predictions.npz"),
             ts_next=pd.to_datetime(test_price["ts_cet"].reset_index(drop=True)).iloc[1:].values,
             P_true_next=P_true_next,
             M1=pred_M1, M2=pred_M2, M3=pred_M3, M4=pred_M4, M5=pred_M5,
             q_lo_5=out_M5["q_lo_5"], q_hi_95=out_M5["q_hi_95"])
    L(f"saved {os.path.join(DIR_OUT, 'stage_6_predictions.npz')}")

    # ===== 报告 =====
    L("\n=== writing report ===")
    lines = [f"Stage 6 OOS Price Prediction (1-step ahead, filtering)",
             f"test span: {ts_test.iloc[0]} → {ts_test.iloc[-1]}, n={n}",
             "=" * 78,
             "\n[L1] Point metrics (lower=better):",
             df_pt.to_string(index=False),
             "",
             "\n[L2] Pinball + CRPS (lower=better):",
             df_pb.to_string(index=False),
             "",
             "\n[L3] DM test p-values (i vs j, H0: equal):",
             df_dm.round(4).to_string(),
             "",
             f"\nTotal elapsed = {time.time()-t0:.1f}s"]
    with open(OUT_REP, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines:
        L(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
