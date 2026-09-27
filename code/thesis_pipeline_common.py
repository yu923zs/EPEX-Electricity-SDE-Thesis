# -*- coding: utf-8 -*-
"""
================================================================
 thesis_pipeline_common.py  — 共享辅助函数模块（A / B / C 共用）
   - 严格复用 05_logs/correction_pass.py 的 Europe/Berlin tz & epoch 过滤。
   - 加载 01_raw/epex_germany_DELU_hourly_2019_2025Q3_raw.csv 并
     按 CET 墙钟区间 3-way split（drift-train / TS-est / test）。
   - 季节项 μ_t 回归（长波自然样条 + 周内 dummies + 日内 Fourier 对），
     只在 drift 训练区间拟合 → 全局预测 → X_t = P_t − μ_t。
   - 状态分箱、截断集 N_k、b̂_k / x_k / v̂_k / w_k 计算（匹配论文 §3）。
================================================================
位置: code/
作者遵循: - 不填补不删除缺失值；
          - 候选基 / 权重 / 阈值仅用 drift 训练区间确定，无信息泄漏；
          - 所有区间边界基于 Europe/Berlin 墙钟而非 naive 本地时区。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------- 路径常量（跨阶段保持一致） ----------
# BASE 指向仓库根目录（本文件位于 code/ 下，向上一级）
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RAW = os.path.join(BASE, "01_raw")
DIR_LOG = os.path.join(BASE, "05_logs")
DIR_OUT = os.path.join(BASE, "06_stage_outputs")
RAW_CSV = os.path.join(DIR_RAW, "epex_germany_DELU_hourly_2019_2025Q3_raw.csv")

START_CET = datetime(2019, 1, 1, 0, 0, 0)
END_CET = datetime(2025, 9, 30, 23, 0, 0)
SPLIT1_CET = datetime(2022, 6, 30, 23, 0, 0)   # train up & incl
SPLIT2_CET = datetime(2024, 12, 31, 23, 0, 0)  # TS up & incl
# 之后为 test

# ---------- 时区与 epoch 转换（严格继承 correction_pass） ----------
try:
    from zoneinfo import ZoneInfo
    TZ_BER = ZoneInfo("Europe/Berlin")
except Exception:
    import pytz
    TZ_BER = pytz.timezone("Europe/Berlin")


def cet_to_ms(dt: datetime) -> int:
    """CET wall-clock (naive datetime) → UTC epoch ms (strict same impl as correction_pass)."""
    aware = dt.replace(tzinfo=TZ_BER)
    return int(aware.timestamp() * 1000)


def ms_to_cet(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(TZ_BER)


# ---------- 数据加载 ----------
def load_smard_df() -> pd.DataFrame:
    """Load audited raw CSV, parse timestamps to CET-aware index, return sorted DF.

    列说明（保持原始 CSV 未修改，仅追加解析列）：
      - epoch_ms_utc, price_eur_mwh_raw (P_t)
      - ts_cet : CET-aware pd.DatetimeIndex-tz
      - 索引为递增的 epoch_ms_utc
    缺失值/异常值：不填补不删除，仅保留 pd.NA。
    """
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"[FATAL] Raw CSV missing: {RAW_CSV}. Run correction_pass.py first.")
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    if len(df) < 50000:
        raise ValueError(f"[FATAL] Raw CSV too short: rows={len(df)}. Expected ~59,159.")

    # --- 强一致性校验：按 (START_CET, END_CET) 重新按 epoch 过滤（与 correction_pass 完全一致） ---
    start_ms = cet_to_ms(START_CET)
    end_ms = cet_to_ms(END_CET)
    mask = (df["epoch_ms_utc"] >= start_ms) & (df["epoch_ms_utc"] <= end_ms)
    df = df.loc[mask].copy()
    df = df.sort_values("epoch_ms_utc").reset_index(drop=True)

    df["ts_cet"] = df["epoch_ms_utc"].apply(
        lambda ms: pd.Timestamp(ms_to_cet(int(ms)))
    )
    # 强制 tz-aware 列（有些 pandas 版本会把 apply 结果降级为 object）
    df["ts_cet"] = pd.to_datetime(df["ts_cet"], utc=False).dt.tz_convert(TZ_BER) \
        if str(df["ts_cet"].dtype).startswith("datetime64") else pd.to_datetime(df["ts_cet"])
    return df


# ---------- 三段区间划分 ----------
SPLIT_BOUNDS = {
    "drift_train": (START_CET, SPLIT1_CET),
    "ts_est":      (datetime(2022, 7, 1, 0, 0, 0), SPLIT2_CET),
    "test":        (datetime(2025, 1, 1, 0, 0, 0), END_CET),
}


def split_3way(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict of 3 named subsets (views on original, no copy if possible).

    每段包含 包含起点 / 包含终点 的闭合 CET 墙钟区间（逐小时）。
    """
    out = {}
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        lo_ms, hi_ms = cet_to_ms(lo), cet_to_ms(hi)
        sub = df.loc[(df["epoch_ms_utc"] >= lo_ms) & (df["epoch_ms_utc"] <= hi_ms)].copy()
        sub = sub.sort_values("epoch_ms_utc").reset_index(drop=True)
        out[name] = sub
    return out


# ---------- 季节项 μ_t（仅在 drift 训练区间拟合，然后预测全样本） ----------
def _hour_in_day_fourier_basis(t: pd.Series, nh: int = 5) -> pd.DataFrame:
    """Hour-of-day Fourier pairs (cos/sin kω, k=1..nh). 24h period."""
    hod = t.dt.hour + t.dt.minute / 60.0
    omega = 2 * np.pi / 24.0
    cols = {}
    for k in range(1, nh + 1):
        cols[f"hod_cos{k}"] = np.cos(k * omega * hod).astype(np.float32)
        cols[f"hod_sin{k}"] = np.sin(k * omega * hod).astype(np.float32)
    return pd.DataFrame(cols, index=t.index)


def _dow_dummies(t: pd.Series) -> pd.DataFrame:
    """Mon..Sat 6 dummies (Sunday dropped as intercept baseline)."""
    dow = t.dt.dayofweek  # 0=Mon..6=Sun
    cols = {}
    for d in range(6):
        cols[f"dow_{d}"] = (dow == d).astype(np.float32)
    return pd.DataFrame(cols, index=t.index)


def _natural_cubic_spline_basis(x: np.ndarray, knots: np.ndarray,
                                 deriv: int = 0) -> np.ndarray:
    """数值稳定的自然三次样条基（NCS），边界外线性延拓。

    实现思路（经典 / Green & Silverman 等价形式）：
      令 K = len(knots)-2，knots 顺序 (ξ_0,ξ_1,…,ξ_{K+1})。
      对每个内部结点 ξ_j（1 ≤ j ≤ K），定义截断幂基：
         d_j(x) = (x - ξ_j)_+^3 - (x - ξ_{K+1})_+^3 * (ξ_{K+1}-ξ_j)/(ξ_{K+1}-ξ_0)
                + (x - ξ_0)_+^3    * (ξ_{K+1}-ξ_j)/(ξ_{K+1}-ξ_0)  [修正：应为 -...]
      —— 我们改用更标准、更紧凑的 **Green-Silverman 递推型基 d_j**，确保在
         x ≤ ξ_0 和 x ≥ ξ_{K+1} 时 自动 线性（二阶导为 0 的 NCS 天然边界条件）。
    输出基矩阵为 (n, K)，不含截距项。
    """
    knots = np.asarray(knots, dtype=np.float64).ravel()
    knots = np.unique(knots)
    if len(knots) < 4:
        # 结点太少，退回 线性 + 二次 作为平滑基（同样 线性外推 → 线性）
        return np.stack([x, np.where(
            (x >= knots[0]) & (x <= knots[-1]), x ** 2,
            # 外推段强制“线性延拓”：二次项值钳制在边界上，差为常数
            (x - knots[0]) * 2 * knots[0] + knots[0] ** 2
                if False else np.where(x < knots[0], knots[0] ** 2, knots[-1] ** 2)
        )], axis=1)

    ξ0, ξKp1 = knots[0], knots[-1]
    inner = knots[1:-1]
    K = len(inner)  # 内部结点数 = 基维度

    # 辅助：截断正部 (·)_+^3
    def tp3(t, c):
        u = t - c
        return np.where(u > 0.0, u ** 3, 0.0)

    # d_j(x) 基的 分子（Green-Silverman NCS 基）
    # d_j(x) = ((x-ξ_j)_+^3 - (x-ξ_{K+1})_+^3 * (ξ_{K+1}-ξ_j)/(ξ_{K+1}-ξ_0)
    #          -((x-ξ_0)_+^3 - (x-ξ_{K+1})_+^3) * (ξ_j - ξ_0)/(ξ_{K+1}-ξ_0))
    #        / (ξ_{K+1} - ξ_0)
    span = ξKp1 - ξ0
    if span <= 0:
        raise ValueError("_natural_cubic_spline_basis: span ≤ 0")
    out = np.empty((len(x), K), dtype=np.float64)
    for j, ξj in enumerate(inner):
        a = tp3(x, ξj)
        b = tp3(x, ξKp1) * (ξKp1 - ξj) / span
        c0 = tp3(x, ξ0) - tp3(x, ξKp1)
        c = c0 * (ξj - ξ0) / span
        out[:, j] = (a - b - c) / span
    return out


def _longwave_natural_spline(df_ts_cet: pd.Series, knots_per_year: float = 6.0,
                             fit_t: pd.Series | None = None) -> tuple[pd.DataFrame, object]:
    """Natural cubic spline basis on (t - t0)/days.

    结点仅由 fit_t（drift 训练区间）时间跨度确定。
    边界外线性延拓 → 对 TS 估计/测试外推段 不会爆炸。
    """
    if fit_t is None:
        fit_t = df_ts_cet

    t_days_all = (df_ts_cet - fit_t.min()).dt.total_seconds().values / 86400.0
    t_days_fit = (fit_t - fit_t.min()).dt.total_seconds().values / 86400.0

    span_days = float(np.nanmax(t_days_fit) - np.nanmin(t_days_fit))
    # 内部结点数：至少 3 个，避免 df 太低（每 6 个/年 ≈ 每 2 个月一个结点）
    n_knots_inner = max(3, int(round(span_days / 365.25 * knots_per_year)))
    boundary = (np.nanmin(t_days_fit), np.nanmax(t_days_fit))
    inner = np.linspace(boundary[0], boundary[1], n_knots_inner + 2)[1:-1]
    knots_all = np.concatenate([[boundary[0]], inner, [boundary[1]]])

    B_all = _natural_cubic_spline_basis(t_days_all, knots_all)
    colnames = [f"lw_ncs_{i}" for i in range(B_all.shape[1])]
    B_all_df = pd.DataFrame(B_all, columns=colnames, index=df_ts_cet.index)
    meta = {"method": "natural_cubic_spline_numpy",
            "inner_knots": inner,
            "boundary": boundary,
            "knots_all": knots_all,
            "fit_t_min_ts": fit_t.min()}
    return B_all_df, meta


# ================= Piecewise-Linear Hinge basis (natural boundary) =================
def piecewise_linear_hinge_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """piecewise-linear HINGE 基：
        basis_j(x) = max(x - knots[j], 0)
    线性本身 + K 个 hinge 联合给出 piecewise-linear 拟合，
    外推时 → 自动线性延拓（无 x³ 爆炸）。
    返回形状 (len(x), len(knots)).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64).ravel()
    B = np.maximum(x[:, None] - knots[None, :], 0.0)
    return B


def fit_seasonal(train_sub: pd.DataFrame, full_df: pd.DataFrame,
                 hod_nharm: int = 5, knots_per_year: float = 6.0,
                 ncs_knots_quant_n: int = 40,
                 lw_df: int = 12):
    """Fit μ_t by OLS on drift-train only.

    设计（**严格外推安全 / 无 x^n 爆炸**）:
      μ̂(t) = β_0 + β_trend·t_days
            + Σ_{k=1..nh}(a_k cos(kω_h) + b_k sin(kω_h))   # hod Fourier
            + Σ_{dow=0..5} δ_dow · 1_{dow=d}                 # dow dummies
            + Σ_{j=1..lw_df} h_j · max(t_days − τ_j, 0)      # 长波 piecewise-linear
                                                                  （τ_j = train-only quantile knots）

    因为全部基对 t_days 最多是线性 / 截断线性，
    外插在 t_days < min(train) 或 t_days > max(train) 时必然是 t 的仿射函数，
    不会出现 R²_full 爆炸负值。
    """
    # --- Design matrices on ALL rows ---
    t_all = full_df["ts_cet"]
    t_fit = train_sub["ts_cet"]
    t_days_all = (t_all - t_fit.min()).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    t_days_fit = (t_fit - t_fit.min()).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0

    hod_all = _hour_in_day_fourier_basis(t_all, nh=hod_nharm)
    dow_all = _dow_dummies(t_all)

    # 长波 hinge 结点：lw_df 个（训练区间 quantile-only）
    qq = np.linspace(0.0, 1.0, lw_df + 2)[1:-1]
    knots_lw = np.quantile(t_days_fit, qq)
    knots_lw = np.unique(knots_lw)
    # 关键修正：长波 hinge 的时间变量在训练区间最大值处截断（平坦外推）。
    # 若不截断，训练区间外所有 (t_days - c_j)_+ 同时线性增长，
    # 等效于隐含线性趋势，会导致 2022-2024 能源危机期 μ̂ 严重高估（R²_full 炸为负值）。
    # 截断后，训练区间外长波分量 = 常数（训练末点值），
    # 真实 regime 水平位移由 X = P - μ̂ 承载，再交由状态相关漂移 b(x) 处理。
    t_days_fit_max = float(np.nanmax(t_days_fit))
    t_days_lw = np.minimum(t_days_all, t_days_fit_max)
    hinge_all = piecewise_linear_hinge_basis(t_days_lw, knots_lw)
    hinge_cols = [f"lw_hinge_{i}" for i in range(hinge_all.shape[1])]
    hinge_df = pd.DataFrame(hinge_all, columns=hinge_cols, index=full_df.index)

    # 合并设计矩阵（不含全局 linear trend t_days；仅用 hinges 捕捉训练内非线性长波。
    # 外推时 hinges 全为 0 或 常数 → μ̂ 外推 = const + Fourier + dow_dummies
    #   → 「周期 + 常数」，电价 regime-shift 下最保守；避免正斜率把 2023-2025 高估。）
    des_all_parts = [hod_all, dow_all, hinge_df]
    des_all = pd.concat(des_all_parts, axis=1)
    feat_names = list(des_all.columns)
    A_all = np.column_stack([np.ones(len(des_all)),
                             des_all.to_numpy(dtype=np.float64)])
    names_all = ["intercept"] + feat_names

    y_all = full_df["price_eur_mwh_raw"].to_numpy(dtype=np.float64)

    # 训练子行（按 epoch_ms_utc 对齐，稳健 regardless of reset_index）
    train_ms = set(train_sub["epoch_ms_utc"].tolist())
    train_mask = full_df["epoch_ms_utc"].isin(train_ms).to_numpy(dtype=bool)

    A_train = A_all[train_mask]
    y_train = y_all[train_mask]
    g = np.isfinite(y_train) & np.all(np.isfinite(A_train), axis=1)
    if g.mean() < 0.95:
        raise RuntimeError(f"[seasonal] bad train rows in OLS: {g.mean():.3%}")
    coef, *rest = np.linalg.lstsq(A_train[g], y_train[g], rcond=None)
    mu_all_np = A_all @ coef
    mu_all = pd.Series(mu_all_np, index=full_df.index, name="mu_hat")
    X_all = pd.Series(y_all - mu_all_np, index=full_df.index, name="X")

    # --- 训练 R² ---
    y_tr = y_train[g]
    mu_tr = mu_all_np[train_mask][g]
    sse_tr = np.nansum((y_tr - mu_tr) ** 2)
    sst_tr = np.nansum((y_tr - np.nanmean(y_tr)) ** 2)
    R2_train = 1.0 - sse_tr / max(sst_tr, 1e-12)

    # --- 全段 pseudo R²（诊断用） ---
    finite_all = np.isfinite(y_all)
    sst_all = np.nansum((y_all - np.nanmean(y_all[finite_all])) ** 2)
    sse_all = np.nansum((y_all[finite_all] - mu_all_np[finite_all]) ** 2)
    R2_full = 1.0 - sse_all / max(sst_all, 1e-12)

    return {
        "mu_all": mu_all,
        "X_all": X_all,
        "R2_train": R2_train,
        "R2_full": R2_full,
        "feature_names_lin": names_all,
        "coef_lin": coef,
        "lw_hinge_knots_days": knots_lw,
        "hod_nharm": hod_nharm,
        "lw_df": lw_df,
        "t_fit_min_ts": t_fit.min(),
        "t_days_fit_max": t_days_fit_max,
        "lw_extrapolation": "flat (capped at train max t_days)",
    }


# ---------- 阶段 A 的核心：状态分箱 + drift 工作样本 + 可靠性权重 ----------
def build_drift_work_samples(X_train: np.ndarray, t_train: np.ndarray | None = None,
                             K: int = 100, tau_q: float = 0.95, n_min: int = 30,
                             winsor_pct: float = 0.95, eps_v: float = 1e-6,
                             dt_hours: float = 1.0):
    """状态分箱 → 截断 → b̂_k / x_k / v̂_k / w_k.

    输入:
      X_train : 训练区间的状态残差向量（长度 N_T，可含 NaN）
      t_train : 可选，时间索引（目前未用，保留）
      K       : 等频分箱数（使用 X_train 的分位作为边界，仅训练区间！）
      tau_q   : 截断分位（双侧，使用 |ΔX| 的分位 q_τ）
      n_min   : 每个箱保留最小局部样本量
      winsor_pct : 原始权重 w^raw 的 winsorize 上限分位（双侧？——仅上限）
      eps_v   : 防止局部方差为 0 的极小常数
      dt_hours: Δt 的单位小时数（论文 Δt=1，这里用 1.0）

    返回:
      bins  : (K+1,) 边界 a_1..a_{K+1}
      keep_k: (K_kept,) 原始箱序号（1-based k）
      x_k   : (K_kept,) 箱代表状态
      b_hat : (K_kept,) drift 工作样本
      n_k   : (K_kept,) 截断后局部样本量
      v_hat : (K_kept,) 局部样本方差（含 eps_v 保护已提前应用？——不含，供诊断）
      w_raw : (K_kept,) 原始精度权重 = n_k dt² / (v̂+ε_v)
      w_k   : (K_kept,) winsorize + 归一化后的可靠性权重（sum = K_kept）
      q_trunc: float, 分位阈值 q_τ
      N_discards_tail : 因尾部截断丢弃的增量数
    """
    X = np.asarray(X_train, dtype=np.float64).ravel()
    finite = np.isfinite(X)
    if finite.mean() < 0.95:
        raise RuntimeError(f"[drift_work_samples] too many NaN X rows: {finite.mean():.3%}")

    # 1) 相邻增量 ΔX（保持 Δt=1h，相邻 epoch 差就是 dt_hours）
    X_level = X[finite]
    DX = np.diff(X_level)
    # 配对：用 X[i] ∈ B_k 且 |ΔX_i| ≤ q_τ 计入 N_k
    X_for_bin = X_level[:-1]  # 与 DX 长度相同

    # 2) 状态箱边界：基于 X_level（包含最后一个值，让分箱更广一些）的等频
    bins = np.quantile(X_level, np.linspace(0.0, 1.0, K + 1))
    bins = np.unique(bins)
    # 如果 X 中重值太多使 unique 后箱数 < K，重分配边界为等距 min→max 并 warning
    if len(bins) < max(10, K // 2):
        # 回退：分位数 → 等距
        x0, x1 = float(np.nanmin(X_level)), float(np.nanmax(X_level))
        bins = np.linspace(x0, x1, K + 1)
    K_eff = len(bins) - 1

    # 3) 截断阈值：基于 |ΔX| 的 tau_q 分位
    q_trunc = float(np.quantile(np.abs(DX), tau_q))
    abs_dx_ok = np.abs(DX) <= q_trunc
    N_discards_tail = int((~abs_dx_ok).sum())

    # 4) 对每个箱计算统计量
    # digitize: bins[i-1] <= X_for_bin < bins[i]（最右箱含末端）
    bin_id = np.digitize(X_for_bin, bins[1:-1], right=False)  # 0..K_eff-1

    keep_idx = []
    stats = {"n": [], "xbar": [], "bhat": [], "vhat": []}
    for k in range(K_eff):
        in_bin = bin_id == k
        # 再叠加 截断条件
        idx = np.where(in_bin & abs_dx_ok)[0]
        n_k = len(idx)
        if n_k < n_min:
            continue
        dXk = DX[idx]
        Xk = X_for_bin[idx]
        dxbar = float(np.mean(dXk))
        xbar  = float(np.mean(Xk))
        vhat  = float(np.var(dXk, ddof=1)) if n_k >= 2 else 0.0
        bhat  = dxbar / dt_hours
        keep_idx.append(k)
        stats["n"].append(n_k)
        stats["xbar"].append(xbar)
        stats["bhat"].append(bhat)
        stats["vhat"].append(vhat)

    if len(keep_idx) == 0:
        raise RuntimeError("[drift_work_samples] NO bin survived n_min filter!")

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    n_k   = np.asarray(stats["n"], dtype=np.int64)
    x_k   = np.asarray(stats["xbar"], dtype=np.float64)
    b_hat = np.asarray(stats["bhat"], dtype=np.float64)
    v_hat = np.asarray(stats["vhat"], dtype=np.float64)

    # 5) 原始精度权重 w^raw = n_k Δt^2 / (v̂_k + ε_v)
    w_raw = (n_k * (dt_hours ** 2)) / (v_hat + eps_v)

    # 6) winsorize 仅上限（防止极端大权重主导）
    if 0.0 < winsor_pct < 1.0:
        w_cap = float(np.quantile(w_raw, winsor_pct))
        w_raw_clip = np.minimum(w_raw, w_cap)
    else:
        w_raw_clip = w_raw.copy()

    # 7) 归一化 sum w_k = K_kept
    denom = float(np.sum(w_raw_clip))
    if denom <= 0:
        raise RuntimeError("[drift_work_samples] Sum(raw_weights clipped) ≤ 0!")
    K_kept = len(keep_idx)
    w_k = K_kept * w_raw_clip / denom

    return {
        "bins": bins,
        "K_input": K,
        "K_effective": K_eff,
        "K_kept": K_kept,
        "keep_bin_index_0based": keep_idx,
        "keep_bin_label_1based": keep_idx + 1,
        "x_k": x_k,
        "b_hat": b_hat,
        "n_k": n_k,
        "v_hat": v_hat,
        "w_raw": w_raw,
        "w_k": w_k,
        "q_trunc": q_trunc,
        "tau_q": tau_q,
        "n_min": n_min,
        "winsor_pct": winsor_pct,
        "eps_v": eps_v,
        "dt_hours": dt_hours,
        "N_discards_tail": N_discards_tail,
        "N_level_finite": int(finite.sum()),
        "N_increments": len(DX),
        "N_kept_increments": int(abs_dx_ok.sum()),
    }


# ---------- 通用 I/O ----------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def save_pickle(obj, path: str):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
