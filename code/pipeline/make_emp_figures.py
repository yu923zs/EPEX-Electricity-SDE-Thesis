# -*- coding: utf-8 -*-
"""
================================================================
 make_emp_figures.py — 实证章(第7章)五张图自动生成
 数据源: 06_stage_outputs/empirical_v1/ 下 CSV (只读)
 输出:  figures/fig10..fig14 (pdf+png)
================================================================
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib

_REPO = os.environ.get("EPEX_BASE") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE3 = _REPO
EMP = os.path.join(BASE3, "06_stage_outputs", "empirical_v1")
FULL = os.path.join(EMP, "full")
FIG = os.path.join(BASE3, "figures")

K1, K2 = -25.37, 2.31


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def main():
    os.makedirs(FIG, exist_ok=True)
    dfX = pd.read_csv(os.path.join(FULL, "full_X_full.csv"),
                      encoding="utf-8-sig")
    ts = pd.to_datetime(dfX["ts_cet"], utc=True).dt.tz_convert("Europe/Berlin")
    P = dfX["price_eur_mwh_raw"].to_numpy(dtype=np.float64)
    X = dfX["X"].to_numpy(dtype=np.float64)

    # ---------- 图10 价格时序 ----------
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(ts, P, lw=0.25, color="0.25")
    ax.axhline(0, color="tab:red", lw=0.8, alpha=0.7)
    ax.axvspan(pd.Timestamp("2022-01-01", tz="Europe/Berlin"),
               pd.Timestamp("2023-12-31 23:00", tz="Europe/Berlin"),
               color="tab:orange", alpha=0.15)
    ax.text(pd.Timestamp("2022-06-01", tz="Europe/Berlin"), 800,
            "能源危机区间", fontsize=9, color="tab:orange")
    n_neg = int((P < 0).sum())
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"EPEX 德国 DE-LU 日前电价（2019-01 至 2025-09，共 {len(P):,} 小时，"
                 f"负价 {n_neg:,} 小时）", fontsize=10)
    save(fig, "fig10_price_series")

    # ---------- 图11 漂移曲线对比 ----------
    dc = pd.read_csv(os.path.join(FULL, "vasicek_drift_curve.csv"),
                     encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax2 = ax.twinx()
    ax2.hist(X[np.isfinite(X)], bins=160, color="0.85", alpha=0.9)
    ax2.set_yticks([])
    ax2.set_ylabel("状态分布（直方图）", fontsize=9, color="0.5")
    ax.plot(dc["x"], dc["b_ours"], lw=2.0, color="tab:blue",
            label="本文分段漂移")
    ax.plot(dc["x"], dc["b_vasicek"], lw=1.4, ls="--", color="tab:gray",
            label="Vasicek 线性漂移")
    for c in (K1, K2):
        ax.axvline(c, color="tab:red", lw=0.9, ls=":")
    ax.annotate(f"折点 {K1}", xy=(K1, 1.9), xytext=(-180, -13.5),
                fontsize=9, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax.annotate(f"折点 {K2}", xy=(K2, 1.5), xytext=(70, -12.0),
                fontsize=9, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax.text(-330, 2.6, "斜率 +0.003", fontsize=9, color="tab:blue")
    ax.annotate("斜率 -0.075", xy=(-12, 1.0), xytext=(35, 6.0),
                fontsize=9, color="tab:blue",
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.7))
    ax.text(170, -8.6, "斜率 -0.029", fontsize=9, color="tab:blue")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("去季节电价状态 x（EUR/MWh）")
    ax.set_ylabel("漂移 b(x)（EUR/MWh/小时）")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("全样本漂移结构识别：分段均值回归与线性基准", fontsize=10)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    save(fig, "fig11_drift_vs_vasicek")

    # ---------- 图12 ECF 匹配 ----------
    eg = pd.read_csv(os.path.join(FULL, "full_ecf_fit_grid.csv"),
                     encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, emp, a_col, s_col, ttl in [
            (axes[0], "cos_emp", "cos_asym", "cos_sym", "实部"),
            (axes[1], "sin_emp", "sin_asym", "sin_sym", "虚部")]:
        ax.plot(eg["u_j"], eg[emp], "o", ms=3.5, color="0.2", label="经验特征函数")
        ax.plot(eg["u_j"], eg[a_col], "-", lw=1.6, color="tab:blue",
                label="非对称 TS")
        ax.plot(eg["u_j"], eg[s_col], "--", lw=1.4, color="tab:orange",
                label="对称受限参照")
        ax.set_xlabel("频率 $u_j$")
        ax.set_title(f"特征函数{ttl}匹配", fontsize=10)
        if emp == "cos_emp":
            ax.set_ylabel("数值")
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("全样本 ECF 拟合（37 个频点）", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig12_ecf_match")

    # ---------- 图13 非对称检验零分布 ----------
    nl = pd.read_csv(os.path.join(FULL, "asymtest_null.csv"),
                     encoding="utf-8-sig")
    dq_obs = 3.49350140e-04
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(nl["dq"], bins=30, color="0.6", edgecolor="white")
    ax.axvline(dq_obs, color="tab:red", lw=1.6)
    ax.text(dq_obs, ax.get_ylim()[1] * 0.85,
            f"  观测 $\\Delta Q$ = {dq_obs:.2e}\n  $p$ = 0.0050",
            color="tab:red", fontsize=9)
    ax.set_xlabel("零假设下的 $\\Delta Q = Q_{sym}-Q_{asy}$")
    ax.set_ylabel("频数")
    ax.set_title("非对称性检验：参数自助零分布（200 次重复）", fontsize=10)
    save(fig, "fig13_asymtest_null")

    # ---------- 图14 残差 ACF ----------
    ac = pd.read_csv(os.path.join(FULL, "full_resid_acf.csv"),
                     encoding="utf-8-sig")
    n = 59158
    band = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(ac["lag"], ac["acf_resid"], "o-", ms=3, lw=1.0,
            color="tab:blue", label="残差 ACF")
    ax.plot(ac["lag"], ac["acf_resid_sq"], "s-", ms=3, lw=1.0,
            color="tab:orange", label="残差平方 ACF")
    ax.axhspan(-band, band, color="0.9", alpha=0.8)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("滞后阶数（小时）")
    ax.set_ylabel("自相关")
    ax.legend(fontsize=9)
    ax.set_title("全样本残差自相关诊断（灰带为 ±1.96/√n）", fontsize=10)
    save(fig, "fig14_resid_acf")

    print("[figures] all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
