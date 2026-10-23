# -*- coding: utf-8 -*-
r"""
make_thesis_figures.py — 论文插图生成（全部基于已落盘的真实实验产出，不捏造任何数据）
输出: figures/fig*.pdf (PDF 矢量图, TrueType 嵌入)
用 base anaconda3 python 运行 (numpy 1.26.4 / pandas 2.2.2 / scipy 1.13.1 / mpl 3.9.2):
    python code/make_thesis_figures.py
"""
import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "figure.dpi": 110,
})

C_TRAIN, C_TSEST, C_TEST = "#4C72B0", "#DD8452", "#55A868"
C_ASY, C_SYM = "#C44E52", "#8172B3"

P_A = os.path.join(ROOT, "06_stage_outputs", "stage_A_drift_train")
P_B = os.path.join(ROOT, "06_stage_outputs", "stage_B_sparse_drift")
P_C = os.path.join(ROOT, "06_stage_outputs", "stage_C_ts_ecf")
P_S = os.path.join(ROOT, "06_stage_outputs", "stage_sim_results")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_stage_C import ts_phi_dt  # noqa: E402  与 Stage4/5 完全同源的 CF 实现

FAILED = []


def done(name):
    print("[FIG] %s ok" % name)


def fail(name, e):
    FAILED.append(name)
    print("[FIG] %s FAILED: %r" % (name, e))


# ---------------------------------------------------------------- 数据加载
print("loading data ...")
df_x = pd.read_csv(os.path.join(P_A, "stageA_X_full.csv"))
df_x["ts"] = pd.to_datetime(df_x["epoch_ms_utc"], unit="ms", utc=True).dt.tz_convert("CET")
df_x = df_x.sort_values("ts").reset_index(drop=True)
print("splits:", df_x["split"].value_counts().to_dict())

df_bin = pd.read_csv(os.path.join(P_A, "stageA_drift_bin_stats.csv"))
df_band = pd.read_csv(os.path.join(P_B, "stageB_posterior_drift_band.csv")).sort_values("x")
df_pip = pd.read_csv(os.path.join(P_B, "stageB_pip_coef_summary.csv"))
df_ecf = pd.read_csv(os.path.join(P_C, "stageC_ecf_freq_grid.csv"))
df_q4 = pd.read_csv(os.path.join(P_C, "stage_4_quantiles_compare_v2.csv"))
df_cov = pd.read_csv(os.path.join(P_C, "stage_5_oos_coverage_v2.csv"))
df_est = pd.read_csv(os.path.join(P_C, "stageC_estimates_compare.csv")).set_index("parameter")
df_sim = pd.read_csv(os.path.join(P_S, "sim_rows_raw.csv"))

xi_asy = tuple(float(df_est.loc[k, "estimate_asym"]) for k in ["μ", "C₊", "C₋", "M", "G", "α"])
xi_sym = (float(df_est.loc["μ", "estimate_sym"]), float(df_est.loc["C₊", "estimate_sym"]),
          float(df_est.loc["C₋", "estimate_sym"]), float(df_est.loc["M", "estimate_sym"]),
          float(df_est.loc["G", "estimate_sym"]), float(df_est.loc["α", "estimate_sym"]))
print("xi_asy =", np.round(xi_asy, 4).tolist())
print("xi_sym =", np.round(xi_sym, 4).tolist())


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    if os.environ.get("FIG_PNG"):  # 质检用 PNG 副本
        fig.savefig(os.path.join(OUT, name.replace(".pdf", ".png")), bbox_inches="tight", dpi=140)
    plt.close(fig)
    print("      -> %s (%d bytes)" % (name, os.path.getsize(path)))


# ================================================== Fig 1: 价格序列 + 三段切分 + X_t
try:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.3), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    ax.plot(df_x["ts"], df_x["price_eur_mwh_raw"], lw=0.30, color="#333333", rasterized=True)
    t0, t1 = df_x["ts"].iloc[0], df_x["ts"].iloc[-1]
    b1 = pd.Timestamp("2022-07-01 00:00:00+02:00")
    b2 = pd.Timestamp("2025-01-01 00:00:00+01:00")
    ax.axvspan(t0, b1, color=C_TRAIN, alpha=0.10)
    ax.axvspan(b1, b2, color=C_TSEST, alpha=0.10)
    ax.axvspan(b2, t1, color=C_TEST, alpha=0.14)
    for b in (b1, b2):
        ax.axvline(b, color="k", lw=0.7, ls="--")
    ymax = df_x["price_eur_mwh_raw"].max()
    ax.text(pd.Timestamp("2020-10-01", tz="CET"), ymax * 0.97, "漂移训练段\n2019-01–2022-06", ha="center", va="top", fontsize=7.5, color=C_TRAIN)
    ax.text(pd.Timestamp("2023-09-15", tz="CET"), ymax * 0.97, "TS 估计段\n2022-07–2024-12", ha="center", va="top", fontsize=7.5, color="#B35E27")
    ax.text(pd.Timestamp("2025-04-20", tz="CET"), ymax * 0.97, "OOS 测试段\n2025-01–09", ha="center", va="top", fontsize=7, color="#2F7A46")
    ax.set_ylabel("日前电价 (EUR/MWh)")
    ax.set_title("EPEX SPOT DE-LU 小时日前价格与去季节化残差水平（2019-01–2025-09, n=59,159）", fontsize=9.5)
    ax = axes[1]
    ax.plot(df_x["ts"], df_x["X"], lw=0.25, color="#555555", rasterized=True)
    ax.axhline(0, color="k", lw=0.6)
    for b in (b1, b2):
        ax.axvline(b, color="k", lw=0.7, ls="--")
    ax.set_ylabel(r"$X_t = P_t - \hat\mu_t$ (EUR/MWh)")
    ax.set_xlabel("时间 (CET)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save(fig, "fig3_1_price_X_split.pdf")
    done("fig3_1")
except Exception as e:
    fail("fig3_1", e)

# ================================================== Fig 2: 季节拟合 zoom (能源危机窗口)
try:
    m = (df_x["ts"] >= pd.Timestamp("2021-12-01", tz="CET")) & \
        (df_x["ts"] <= pd.Timestamp("2022-10-31 23:00", tz="CET"))
    d = df_x[m]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.plot(d["ts"], d["price_eur_mwh_raw"], lw=0.35, color="#888888", label="实际价格 $P_t$", rasterized=True)
    ax.plot(d["ts"], d["mu_hat"], lw=0.9, color=C_TRAIN, label=r"季节基线 $\hat\mu_t$（无全局趋势）")
    tb = pd.Timestamp("2022-06-30 23:00:00+02:00")
    ax.axvline(tb, color="k", ls="--", lw=0.9)
    ax.annotate("漂移训练段终点 2022-06-30", xy=(tb, d["price_eur_mwh_raw"].max() * 0.99),
                xytext=(pd.Timestamp("2022-02-25", tz="CET"), d["price_eur_mwh_raw"].max() * 0.90),
                fontsize=7.5, va="top", arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("时间 (CET)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title(r"能源危机窗口下的季节基线拟合 $\hat\mu_t$（Fourier 5 + 星期效应 + 12 训练分位铰链）", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig3_2_seasonal_zoom.pdf")
    done("fig3_2")
except Exception as e:
    fail("fig3_2", e)

# ================================================== Fig 3: 仿真稀疏恢复指标 vs T
try:
    g = df_sim.groupby(["dgp_id", "T_levels"]).agg(
        tpr_m=("tpr", "mean"), tpr_s=("tpr", "std"),
        fdr_m=("fdr", "mean"), fdr_s=("fdr", "std"),
        mse_m=("medse", "mean"), mse_s=("medse", "std")).reset_index()
    ts = sorted(g["T_levels"].unique())
    names = {1: "DGP1 对称 TS", 2: "DGP2 非对称 TS", 3: "DGP3 非对称+切换"}
    colors = {1: C_SYM, 2: C_ASY, 3: C_TRAIN}
    markers = {1: "o", 2: "s", 3: "^"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    panels = [("tpr", r"TPR (%)"), ("fdr", r"FDR (%)"), ("mse", r"MedSE")]
    for ax, (pre, ylab) in zip(axes, panels):
        for dgp in (1, 2, 3):
            s = g[g["dgp_id"] == dgp].sort_values("T_levels")
            m = s[pre + "_m"] * (100 if pre in ("tpr", "fdr") else 1)
            sd = s[pre + "_s"] * (100 if pre in ("tpr", "fdr") else 1)
            fill = sd.fillna(0)
            ax.plot(s["T_levels"], m, marker=markers[dgp], ms=4, lw=1.1,
                    color=colors[dgp], label=names[dgp])
            ax.fill_between(s["T_levels"], m - fill, m + fill, color=colors[dgp], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(ts)
        ax.set_xticklabels(["3k", "10k", "25k"])
        ax.set_xlabel(r"样本量 $T$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].legend(loc="upper left", fontsize=6.8)
    fig.suptitle(r"仿真：SS-St-w 稀疏漂移恢复随样本量 $T$ 的收敛（$S=10$ 次重复，阴影为 $\pm 1$ sd）", y=1.04, fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig6_1_sim_recovery.pdf")
    done("fig6_1")
except Exception as e:
    fail("fig6_1", e)

# ================================================== Fig 4: 仿真 Q 比 boxplot
try:
    df_sim = df_sim.copy()
    df_sim["logQ"] = np.log10(df_sim["pl_Q_asy"] / df_sim["pl_Q_sym"])
    ts = [3000, 10000, 25000]
    pos, data, cols, labs = [], [], [], []
    p = 1
    for dgp in (1, 2, 3):
        for T in ts:
            s = df_sim[(df_sim["dgp_id"] == dgp) & (df_sim["T_levels"] == T)]["logQ"].values
            data.append(s)
            pos.append(p)
            cols.append({1: C_SYM, 2: C_ASY, 3: C_TRAIN}[dgp])
            labs.append("%dk" % (T // 1000))
            p += 1
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    bp = ax.boxplot(data, positions=pos, widths=0.62, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", ms=3, mfc="#666666", mec="#666666"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.8)
    for el in ("whiskers", "caps", "medians"):
        for line in bp[el]:
            line.set(color="#333333", lw=0.8)
    ax.axhline(0, color="k", lw=0.9, ls="--")
    ax.text(9.45, 0.03, r"$\hat Q^{\rm ASY}=\hat Q^{\rm SYM}$", fontsize=7, ha="right")
    ax.set_xticks([2, 5, 8])
    ax.set_xticklabels(["DGP1 对称 TS", "DGP2 非对称 TS", "DGP3 非对称+漂移切换"])
    ax.margins(x=0.02)
    ax.set_xlabel("每组箱线从左到右对应 $T$ = 3k, 10k, 25k")
    ax.set_ylabel(r"$\log_{10}\,(\hat Q^{\rm ASY}/\hat Q^{\rm SYM})$")
    win = (df_sim["pl_Q_asy"] <= df_sim["pl_Q_sym"]).mean() * 100
    ax.set_title(r"仿真：ECF 拟合优度 $\hat Q$ 非对称 vs 对称基线（$S{=}10{\times}3{\times}3{=}90$ 次；"
                 r"$\mathbb{P}(\hat Q^{\rm ASY}\leq\hat Q^{\rm SYM})=%.1f\%%$）" % win, fontsize=9)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout()
    save(fig, "fig6_2_sim_Qratio.pdf")
    done("fig6_2")
except Exception as e:
    fail("fig6_2", e)

# ================================================== Fig 5: 漂移箱估计 + 后验带 + PIP
try:
    fig = plt.figure(figsize=(7.0, 4.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.15], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    wmax = df_bin["w_k"].max()
    ax.scatter(df_bin["x_k"], df_bin["b_hat"], s=6 + 60 * df_bin["w_k"] / wmax,
               color="#9bb7d4", edgecolor="#4C72B0", linewidths=0.4, alpha=0.85,
               label=r"分箱漂移估计 $\hat b_k$（点径 $\propto$ 可靠性权 $w_k$）", zorder=3)
    ax.fill_between(df_band["x"], df_band["b_q025"], df_band["b_q975"],
                    color="#DD8452", alpha=0.22, lw=0, label=r"后验 95\% 带")
    ax.plot(df_band["x"], df_band["b_med"], color="#B35E27", lw=1.3,
            label=r"后验中位漂移 $\hat b_{\rm med}(x)$")
    ax.axhline(0, color="k", lw=0.6)
    for xv, pip, lab in [(-14.944, 0.9863, r"$(x+14.94)_+$"),
                         (28.306, 0.9674, r"$(x-28.31)_+$")]:
        ax.axvline(xv, color="#C44E52", ls="--", lw=0.9)
    ax.annotate(r"$(x+14.94)_+$" + "\nPIP=98.6%", xy=(-14.944, 1.86), xytext=(6, 1.80),
                fontsize=7, color="#C44E52", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#C44E52"),
                bbox=dict(fc="white", ec="#C44E52", lw=0.5, alpha=0.9, boxstyle="round,pad=0.25"))
    ax.annotate(r"$(x-28.31)_+$" + "\nPIP=96.7%", xy=(28.306, 1.42), xytext=(40, 1.36),
                fontsize=7, color="#C44E52", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#C44E52"),
                bbox=dict(fc="white", ec="#C44E52", lw=0.5, alpha=0.9, boxstyle="round,pad=0.25"))
    ax.set_ylabel(r"漂移 $\hat b(x)$ (EUR/MWh/h)")
    ax.legend(loc="upper left", fontsize=7.2)
    ax.set_title("Gibbs 稀疏漂移后验：分箱估计（大小=可靠性权）、后验中位与 95% 带、被选中的两个铰链项", fontsize=9.5)

    ax2 = fig.add_subplot(gs[1])
    terms = df_pip["term"].tolist()
    pips = df_pip["PIP"].tolist()
    is_sel = [v >= 0.5 for v in pips]
    cols = ["#C44E52" if s else "#BBBBBB" for s in is_sel]
    ax2.bar(range(len(terms)), pips, color=cols, width=0.65)
    ax2.axhline(0.5, color="k", lw=0.7, ls=":")
    ax2.set_xticks(range(len(terms)))
    ax2.set_xticklabels(terms, rotation=38, ha="right", fontsize=6.3)
    ax2.set_ylabel("PIP")
    ax2.set_ylim(0, 1.08)
    ax2.text(len(terms) - 0.4, 0.53, "阈值 0.5", fontsize=6.5, ha="right")
    ax2.set_xlabel("候选基函数（12 项：3 阶多项式 + 9 个训练分位铰链）")
    fig.tight_layout()
    save(fig, "fig7_1_drift_sparse.pdf")
    done("fig7_1")
except Exception as e:
    fail("fig7_1", e)

# ================================================== Fig 6: ECF 拟合 (|phi| + Im phi)
try:
    u_emp = df_ecf["u_j"].values
    re_emp = df_ecf["cos_E"].values
    im_emp = df_ecf["sin_E"].values
    u_smooth = np.unique(np.concatenate(
        [np.logspace(-2.05, np.log10(u_emp.max()) + 0.02, 320), u_emp]))
    phi_asy = ts_phi_dt(u_smooth, xi_asy)
    phi_sym = ts_phi_dt(u_smooth, xi_sym)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)
    ax = axes[0]
    ax.plot(u_smooth, np.abs(phi_asy), color=C_ASY, lw=1.3, label=r"非对称 TS 拟合 $\hat\varphi$")
    ax.plot(u_smooth, np.abs(phi_sym), color=C_SYM, lw=1.3, ls="--", label=r"对称 TS 基线")
    ax.scatter(u_emp, np.hypot(re_emp, im_emp), s=18, facecolor="white",
               edgecolor="#333333", linewidths=0.8, zorder=5, label=r"经验 CF（$J{=}37$ 频点）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$|\varphi(u)|$")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.set_title(r"ECF 最小距离拟合（$W=I$）：特征函数模与虚部（虚部反映非对称性）", fontsize=9.5)
    ax = axes[1]
    ax.plot(u_smooth, np.imag(phi_asy), color=C_ASY, lw=1.3, label=r"非对称 TS")
    ax.plot(u_smooth, np.imag(phi_sym), color=C_SYM, lw=1.3, ls="--", label=r"对称基线（虚部仅来自 $\mu$）")
    ax.scatter(u_emp, im_emp, s=18, facecolor="white", edgecolor="#333333", linewidths=0.8, zorder=5)
    ax.set_xscale("log")
    ax.set_ylabel(r"$\mathrm{Im}\,\varphi(u)$")
    ax.set_xlabel(r"频率 $u$（$u=c\cdot s_R^{-1}$, $s_R=9.22$ EUR/MWh）")
    ax.legend(fontsize=7.2)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    fig.tight_layout()
    save(fig, "fig7_2_ecf_fit.pdf")
    done("fig7_2")
except Exception as e:
    fail("fig7_2", e)

# ================================================== Fig 7: QQ 图 (模型 vs 经验分位)
try:
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    lim = [-145, 150]
    ax.plot(lim, lim, "k--", lw=0.9, label="45° 完美拟合线")
    ax.scatter(df_q4["q_empirical"], df_q4["q_model_asy"], s=26, marker="o",
               facecolor="white", edgecolor=C_ASY, linewidths=1.0, label="非对称 TS", zorder=4)
    ax.scatter(df_q4["q_empirical"], df_q4["q_model_sym"], s=26, marker="^",
               facecolor="white", edgecolor=C_SYM, linewidths=1.0, label="对称基线", zorder=4)
    for _, row in df_q4.iterrows():
        pr = row["prob"]
        if pr in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999):
            lab = ("%.1f%%" % (pr * 100)) if pr >= 0.01 else ("%.1f%%" % (pr * 100))
            ax.annotate(lab, xy=(row["q_empirical"], row["q_model_asy"]),
                        xytext=(3, -8), textcoords="offset points", fontsize=6, color=C_ASY)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"经验分位 $q^{\rm emp}(p)$ (EUR/MWh)")
    ax.set_ylabel(r"模型分位 $\hat q^{\rm model}(p)$ (EUR/MWh)")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="upper left", fontsize=7.5)
    mae_a = df_q4["abs_err_asy"].mean()
    mae_s = df_q4["abs_err_sym"].mean()
    ax.set_title(r"TS 残差分位 QQ 图（15 个分位点；MAE：非对称 %.1f vs 对称 %.1f EUR）" % (mae_a, mae_s), fontsize=8.8)
    fig.tight_layout()
    save(fig, "fig7_3_qq_quantiles.pdf")
    done("fig7_3")
except Exception as e:
    fail("fig7_3", e)

# ================================================== Fig 8: OOS 测试段 r_t + 95% 区间 + 违约点
try:
    xk = df_bin["x_k"].values
    margin = 0.05 * max(np.max(np.abs(xk)), 1.0)
    xmin, xmax = xk.min() - margin, xk.max() + margin
    d = df_x[df_x["split"] == "test"].sort_values("ts").reset_index(drop=True)
    X = d["X"].values
    dX = np.diff(X)
    b = np.interp(np.clip(X[1:], xmin, xmax), df_band["x"].values, df_band["b_med"].values)
    r = dX - b
    ts_r = d["ts"].values[1:]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
    for ax, model, col in [(axes[0], "ASY_TS", C_ASY), (axes[1], "SYM_TS", C_SYM)]:
        row = df_cov[(df_cov["model"] == model) & (np.isclose(df_cov["alpha"], 0.05))].iloc[0]
        qlo, qhi = row["q_lo_EUR"], row["q_hi_EUR"]
        viol = (r < qlo) | (r > qhi)
        ax.axhspan(qlo, qhi, color=col, alpha=0.12, lw=0)
        ax.axhline(qlo, color=col, lw=0.9, ls="--")
        ax.axhline(qhi, color=col, lw=0.9, ls="--")
        ax.plot(ts_r, r, lw=0.22, color="#777777", rasterized=True)
        if viol.any():
            ax.scatter(ts_r[viol], r[viol], s=5, color=col, zorder=5)
        cov = 100 * (1 - viol.mean())
        name = "非对称 TS" if model == "ASY_TS" else "对称基线"
        ax.set_ylabel(r"$r_t$ (EUR/MWh)")
        ax.set_title("OOS 测试段增量残差 $r_t=\\Delta X_t-\\hat b_{\\rm med}(X_t)$ 与 %s 95%% 预测区间 "
                     "（实际覆盖 %.1f%%，红点=越界）" % (name, cov), fontsize=8.6, loc="left")
        print("      %s: q=[%.2f, %.2f], recomputed coverage=%.2f%% (CSV: %.2f%%)"
              % (model, qlo, qhi, cov, 100 * row["actual_coverage"]))
    axes[1].set_xlabel("时间 (CET, 2025-01-01 – 2025-09-30)")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    fig.tight_layout()
    save(fig, "fig7_4_oos_intervals.pdf")
    done("fig7_4")
except Exception as e:
    fail("fig7_4", e)

print("=" * 50)
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
print("ALL FIGURES OK ->", OUT)
