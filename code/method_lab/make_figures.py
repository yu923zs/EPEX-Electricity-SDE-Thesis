# -*- coding: utf-8 -*-
"""
make_figures.py — v13 定稿实验全套图（图1–图7）
全部数据来自 method_lab/results 与 data_cache 的真实运行存档，无任何编造。
输出: <out>/fig1..fig7 (.png + .pdf)
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib

_REPO = os.environ.get("EPEX_BASE") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_lab as R                       # 自带 CODE 路径与 DGP_LAB/true_drift_arr
from run_stage_C import ts_phi_dt

RES = R.RES_DIR
CACHE = R.CACHE_DIR
SEEDS25 = [91300000 + s for s in range(12)]
SEEDS10 = [91300000 + s for s in range(8)]
PARAMS = [("mu", r"$\mu$"), ("Cp", r"$C_{+}$"), ("Cm", r"$C_{-}$"),
          ("M", "M"), ("G", "G"), ("alpha", r"$\alpha$")]
DGP_C = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_json(name):
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


def model_b(rec, x):
    b = np.full_like(x, rec["intercept"], dtype=float)
    for t in rec["poly_terms"]:
        p = int(t["term"].split("^")[1])
        b = b + t["coef"] * x ** p
    for k in rec["knots"]:
        b = b + k["delta"] * np.maximum(x - k["c"], 0.0)
    return b


def xgrid(dgp, T, seeds):
    xs = []
    for sd in seeds:
        f = os.path.join(CACHE, f"data_{dgp}_T{T}_seed{sd}.npz")
        xs.append(np.load(f)["X_deseas"])
    xa = np.concatenate(xs)
    lo, hi = np.percentile(xa, [0.5, 99.5])
    return np.linspace(lo, hi, 400)


def parse_pm(s):
    return float(str(s).split("±")[0])


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=200,
                bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}", flush=True)


# ---------------- 图1: 折点恢复点图 (v13, T=25k) ----------------
def fig1(out):
    recs = load_json("v13_T25000_S12_identified_models.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, dgp in zip(axes, "ABC"):
        true_cs = [c for (c, _) in R.DGP_LAB[dgp]["hinges"]]
        sub = [r for r in recs if r["dgp"] == dgp]
        n_hit = n_false = 0
        for i, r in enumerate(sorted(sub, key=lambda z: z["seed"])):
            for k in r["knots"]:
                hit = any(abs(k["c"] - tc) < 8.0 for tc in true_cs)
                n_hit += hit; n_false += (not hit)
                ax.scatter(k["c"], i, s=min(abs(k["t"]), 15) * 22 + 25,
                           marker="o" if hit else "X",
                           color="#2ca02c" if hit else "#d62728",
                           edgecolor="k", linewidth=0.4, zorder=3)
        for tc in true_cs:
            ax.axvline(tc, color="k", ls="--", lw=1.4)
            ax.axvspan(tc - 8, tc + 8, color="gray", alpha=0.12)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"s{r['seed'] % 100:02d}"
                            for r in sorted(sub, key=lambda z: z["seed"])],
                           fontsize=7)
        ax.set_xlabel("价格 x (EUR/MWh)")
        ax.set_title(f"DGP {dgp}：命中 {n_hit} / 假折点 {n_false}"
                     + ("（真模型无折点）" if dgp == "A" else ""))
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_ylabel("种子（每行一次重复）")
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[
        Line2D([], [], marker="o", color="#2ca02c", ls="", label="命中（±8 EUR）"),
        Line2D([], [], marker="X", color="#d62728", ls="", label="假折点"),
        Line2D([], [], color="k", ls="--", label="真折点")],
        loc="lower right", fontsize=9)
    fig.suptitle("本文方法折点恢复（T=25,000，12 种子；点大小 ∝ |t| 显著性）")
    save(fig, out, "fig1_knot_recovery")


# ---------------- 图2: 漂移函数重构 (v13, T=25k) ----------------
def fig2(out):
    recs = load_json("v13_T25000_S12_identified_models.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, dgp in zip(axes, "ABC"):
        xg = xgrid(dgp, 25000, SEEDS25)
        b0 = R.true_drift_arr(xg, dgp)
        for r in [z for z in recs if z["dgp"] == dgp]:
            ax.plot(xg, model_b(r, xg), color=DGP_C[dgp], alpha=0.45, lw=1.1)
        ax.plot(xg, b0, "k-", lw=2.6, label=r"真漂移 $b_0(x)$")
        for (c, _) in R.DGP_LAB[dgp]["hinges"]:
            ax.axvline(c, color="k", ls=":", lw=1)
        ax.set_xlabel("价格 x (EUR/MWh)"); ax.set_ylabel("漂移 b(x)")
        ax.set_title(f"DGP {dgp}")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("漂移函数重构：12 条估计曲线与真曲线（T=25,000）")
    save(fig, out, "fig2_drift_reconstruction")


# ---------------- 图3: Stage II 六参数 oracle vs 插件 (T=25k) ----------------
def fig3(out):
    df = pd.read_csv(os.path.join(RES, "lab_rows_v13_T25000_S12_light.csv"))
    xi_true = R.DGP_LAB["B"]["xi"]          # B/C 同真值
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    for k, (nm, lab) in enumerate(PARAMS):
        ax = axes[k // 3, k % 3]
        data, pos, cols = [], [], []
        for j, dgp in enumerate(("B", "C")):
            sub = df[df["dgp"] == dgp]
            data += [sub[f"or_{nm}"].values, sub[f"pl_{nm}"].values]
            pos += [j * 3 + 1, j * 3 + 2]
            cols += ["#999999", DGP_C[dgp]]
        bp = ax.boxplot(data, positions=pos, widths=0.62, patch_artist=True,
                        showfliers=True, flierprops=dict(markersize=3))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.axhline(xi_true[k], color="r", ls="--", lw=1.4, label="真值")
        ax.set_xticks(pos)
        ax.set_xticklabels(["B-or", "B-插件", "C-or", "C-插件"], fontsize=8)
        ax.set_title(lab); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Stage II 六参数估计：oracle 与插件口径分布（12 种子，T=25,000）")
    save(fig, out, "fig3_stage2_params")


# ---------------- 图4: 收敛趋势 T=10k vs 25k ----------------
def fig4(out):
    s10 = pd.read_csv(os.path.join(RES, "lab_summary_v13_T10000_S8_light.csv"))
    s25 = pd.read_csv(os.path.join(RES, "lab_summary_v13_T25000_S12_light.csv"))
    mets = ["TPR", "FDR", "F1", "Wass", "MedSE"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for ax, mt in zip(axes, mets):
        vals, labels, cols = [], [], []
        for dgp in ("B", "C"):
            for s, tag in ((s10, "T=10k"), (s25, "T=25k")):
                v = parse_pm(s.loc[s["dgp"] == dgp, mt].iloc[0])
                vals.append(v); labels.append(f"{dgp}@{tag}")
                cols.append(DGP_C[dgp] if tag == "T=25k" else "#bbbbbb")
        bars = ax.bar(range(4), vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8,
                                                  rotation=20)
        ax.set_title(mt); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("收敛趋势：T=10,000 → T=25,000（浅色=T=10k）")
    save(fig, out, "fig4_convergence")


# ---------------- 图5: v0 vs v13 漂移重构对照 (T=10k) ----------------
def fig5(out):
    z0 = np.load(os.path.join(RES, "v0_T10000_S8_curves.npz"))
    z1 = np.load(os.path.join(RES, "v13_T10000_S8_curves.npz"))
    s_v0 = pd.read_csv(os.path.join(RES, "lab_summary_v0_T10000_S8_none.csv"))
    s_v1 = pd.read_csv(os.path.join(RES, "lab_summary_v13_T10000_S8_none.csv"))
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex="col")
    for i, dgp in enumerate("ABC"):
        xg = xgrid(dgp, 10000, SEEDS10)
        b0 = R.true_drift_arr(xg, dgp)
        for j, (z, s, tag) in enumerate(((z0, s_v0, "内部锚点配置"),
                                         (z1, s_v1, "本文方法"))):
            ax = axes[i, j]
            for sd in SEEDS10:
                k = f"{dgp}_{sd}"
                ax.plot(z[f"x_{k}"], z[f"b_{k}"], color=DGP_C[dgp],
                        alpha=0.4, lw=0.9)
            ax.plot(xg, b0, "k-", lw=2.4)
            mse = parse_pm(s.loc[s["dgp"] == dgp, "MedSE"].iloc[0])
            f1 = parse_pm(s.loc[s["dgp"] == dgp, "F1"].iloc[0])
            ax.set_title(f"{tag} | DGP {dgp} | MedSE={mse:.3f} F1={f1:.3f}",
                         fontsize=10)
            ax.grid(alpha=0.3)
            if i == 2:
                ax.set_xlabel("价格 x (EUR/MWh)")
            ax.set_ylabel("b(x)")
    fig.suptitle("内部锚点配置（左）与本文方法（右）：8 种子漂移重构（T=10,000）")
    save(fig, out, "fig5_v0_vs_v13")


# ---------------- 图6: ECF 拟合优度（代表种子, T=25k） ----------------
def fig6(out):
    recs = load_json("v13_T25000_S12_identified_models.json")
    df = pd.read_csv(os.path.join(RES, "lab_rows_v13_T25000_S12_light.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    cfine = np.linspace(0.05, 12.0, 220)
    for j, dgp in enumerate("ABC"):
        sub = [r for r in recs if r["dgp"] == dgp]
        med = sorted(sub, key=lambda z: z["medse"])[len(sub) // 2]
        seed = med["seed"]
        z = np.load(os.path.join(CACHE, f"data_{dgp}_T25000_seed{seed}.npz"))
        Xd = z["X_deseas"]
        xlv, y = Xd[:-1], np.diff(Xd)
        r = y - model_b(med, xlv)
        mad = np.median(np.abs(r - np.median(r)))
        sR = max(mad / 0.6744897501960817, 1e-12)
        u = cfine / sR
        ur = np.outer(r, u)
        emp_re, emp_im = np.cos(ur).mean(0), np.sin(ur).mean(0)
        row = df[(df["dgp"] == dgp) & (df["seed"] == seed)].iloc[0]
        xi_pl = np.array([row[f"pl_{nm}"] for nm, _ in PARAMS])
        xi_or = np.array([row[f"or_{nm}"] for nm, _ in PARAMS])
        xi_tv = np.array(R.DGP_LAB[dgp]["xi"], float)
        ph_pl = ts_phi_dt(u, xi_pl); ph_or = ts_phi_dt(u, xi_or)
        ph_tv = ts_phi_dt(u, xi_tv)
        for i, (part, get) in enumerate(((r"实部 Re $\phi(u)$", np.real),
                                         (r"虚部 Im $\phi(u)$", np.imag))):
            ax = axes[i, j]
            ax.plot(cfine, get(emp_re + 1j * emp_im) if False else
                    (emp_re if i == 0 else emp_im), "k.", ms=2.5,
                    label="经验 CF（插件残差）")
            ax.plot(cfine, get(ph_tv), "r-", lw=1.6,
                    label=r"真值 $\phi(u;\xi_0)$")
            ax.plot(cfine, get(ph_pl), color=DGP_C[dgp], ls="--", lw=1.4,
                    label=r"插件 $\phi(u;\hat{\xi}_{plugin})$")
            ax.plot(cfine, get(ph_or), color="#888888", ls=":", lw=1.4,
                    label=r"oracle $\phi(u;\hat{\xi}_{oracle})$")
            if i == 1:
                ax.set_xlabel("无量纲频率 c = u·sR")
            ax.set_ylabel(part)
            if i == 0:
                ax.set_title(f"DGP {dgp}（代表种子 {seed}）", fontsize=10)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("ECF 拟合优度：经验特征函数与拟合特征函数（T=25,000）")
    save(fig, out, "fig6_ecf_fit")


# ---------------- 图7: v13 方法流程图 ----------------
def fig7(out):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 6.5)

    def box(x, y, w, h, text, fc):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.08",
                                    fc=fc, ec="k", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.2)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="k"))

    ax.text(7, 6.15, "两阶段 plug-in 估计流程",
            ha="center", fontsize=13, weight="bold")
    ax.add_patch(plt.Rectangle((0.15, 3.1), 13.7, 2.5, fc="#eef3fb",
                 ec="none"))
    ax.add_patch(plt.Rectangle((0.15, 0.25), 13.7, 2.2, fc="#f2fbf0",
                 ec="none"))
    ax.text(0.35, 5.32, "Stage I：漂移结构稀疏识别", fontsize=11,
            weight="bold", color="#1f4e79")
    ax.text(0.35, 2.18, "Stage II：TS 参数 ECF 估计", fontsize=11,
            weight="bold", color="#2c6e2c")

    y1, h1 = 3.6, 1.15
    box(0.35, y1, 1.75, h1, "原始价格\nP_t（小时）", "white")
    box(2.45, y1, 1.75, h1, "去季节\nFourier 日内5谐波\n+ 周内模式", "white")
    box(4.55, y1, 2.0, h1, "Euler 差分回归\n全样本 T−1\n候选 {x,x²,x³}\n+13 铰链(正交化)", "white")
    box(6.9, y1, 1.85, h1, "尖峰-平板\nGibbs 采样\nburn1500/post3000", "white")
    box(9.1, y1, 2.2, h1, "滑窗聚合 win8\ncpip≥0.4 + NMS12\n两侧质量≥5%", "white")
    box(11.65, y1, 2.05, h1, "剖面细化\n+ 稳健 t≥2 过滤\n" r"→ $\hat{b}(x)$ plugin", "#dcebf7")
    for x1, x2 in ((2.1, 2.45), (4.2, 4.55), (6.55, 6.9), (8.75, 9.1),
                   (11.3, 11.65)):
        arrow(x1, y1 + h1 / 2, x2, y1 + h1 / 2)

    y2, h2 = 0.7, 1.05
    box(2.0, y2, 2.9, h2, "残差 " r"$r_t=\Delta X^{deseas}-\hat{b}(x)$" "\n（非标准化 Lévy 增量）", "white")
    box(5.6, y2, 4.3, h2, "单尺度 1h ECF 最小距离\nW = I；c 网格标准\nLHS 多初值 + L-BFGS-B + Nelder–Mead", "white")
    box(10.6, y2, 2.9, h2, r"$\hat{\xi}=(\mu, C_{+}, C_{-}, M, G, \alpha)$" "\n非对称 Tempered Stable", "#dff0dc")
    arrow(12.7, y1, 12.7, 2.6)
    arrow(12.7, 2.6, 3.45, 2.6)
    arrow(3.45, 2.6, 3.45, y2 + h2)
    arrow(4.9, y2 + h2 / 2, 5.6, y2 + h2 / 2)
    arrow(9.9, y2 + h2 / 2, 10.6, y2 + h2 / 2)
    save(fig, out, "fig7_pipeline")


# ---------------- 图8: Stage I 外部基准对比 (F1 + MedSE) ----------------
def fig8(out):
    df = pd.read_csv(os.path.join(RES, "table_stage1_baselines.csv"))
    methods = ["v13", "ssnormal", "ols1", "ols3", "lasso_cv", "stepwise"]
    mlab = {"v13": "本文方法", "ssnormal": "SS-Normal", "ols1": "OLS-p1",
            "ols3": "OLS-p3", "lasso_cv": "LASSO", "stepwise": "逐步BIC"}
    mcol = {"v13": "#d62728", "ssnormal": "#ff9896", "ols1": "#1f77b4",
            "ols3": "#aec7e8", "lasso_cv": "#2ca02c", "stepwise": "#9467bd"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for j, dgp in enumerate("ABC"):
        for i, met in enumerate(("f1", "medse")):
            ax = axes[i, j]
            xpos, xtick = [], []
            for k, m in enumerate(methods):
                for t_i, T in enumerate((10000, 25000)):
                    r = df[(df["method"] == m) & (df["T"] == T)
                           & (df["dgp"] == dgp)]
                    v = float(r[f"{met}_mean"].iloc[0])
                    x = k * 2.2 + t_i * 0.9
                    ax.bar(x, v, width=0.82,
                           color=mcol[m],
                           alpha=0.45 if T == 10000 else 1.0,
                           hatch="//" if T == 10000 else "",
                           edgecolor="white" if T == 10000 else "none",
                           linewidth=0.0)
                    ax.text(x, v, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=90)
                xpos.append(k * 2.2 + 0.45)
                xtick.append(mlab[m])
            ax.set_xticks(xpos)
            ax.set_xticklabels(xtick, fontsize=8, rotation=18)
            if met == "f1":
                ax.set_ylim(0, 1.14)
            ax.set_title(f"DGP {dgp} — " +
                         ("F1（越高越好）" if met == "f1"
                          else "MedSE（越低越好）"), fontsize=10)
            ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(fc=mcol[m], label=mlab[m]) for m in methods]
    handles += [Patch(fc="gray", alpha=1.0, label="T=25,000"),
                Patch(fc="gray", alpha=0.45, hatch="//", label="T=10,000")]
    axes[0, 2].legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle("Stage I 漂移结构识别：本文方法与外部基准（各种子均值）")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, out, "fig8_stage1_baselines")


# ---------------- 图9: Stage II 外部基准 RMSE 对比热图 ----------------
def fig9(out):
    df = pd.read_csv(os.path.join(RES, "table_stage2_baselines.csv"))
    methods = ["v13", "cmom", "cgmm", "mlefft"]
    mlab = {"v13": "本文 ECF", "cmom": "C-Mom", "cgmm": "C-GMM",
            "mlefft": "MLE-FFT"}
    plabs = [r"$\mu$", r"$C_{+}$", r"$C_{-}$", "$M$", "$G$", r"$\alpha$"]
    pnames = [p for p, _ in PARAMS]
    import matplotlib.colors as mcolors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True,
                             constrained_layout=True)
    allv = df[[f"r_{p}" for p in pnames]].values
    norm = mcolors.LogNorm(vmin=max(allv[allv > 0].min(), 1e-3),
                           vmax=allv.max())
    for ax, src, ttl in zip(axes, ("oracle", "plugin"),
                            ("oracle 残差", "插件残差")):
        sub = df[df["src"] == src]
        mat = np.full((12, 6), np.nan)
        ytick = []
        for k, m in enumerate(methods):
            for j, dgp in enumerate("ABC"):
                r = sub[(sub["method"] == m) & (sub["dgp"] == dgp)].iloc[0]
                mat[k * 3 + j] = [r[f"r_{p}"] for p in pnames]
                ytick.append(f"{mlab[m]}·{dgp}")
        im = ax.imshow(mat, aspect="auto", norm=norm, cmap="viridis_r")
        for i in range(12):
            for j in range(6):
                v = mat[i, j]
                txt = f"{v:.2f}" if v < 10 else f"{v:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if v > np.sqrt(norm.vmax * norm.vmin)
                        else "black")
        for k in range(1, 4):
            ax.axhline(k * 3 - 0.5, color="white", lw=2.2)
        ax.set_xticks(range(6)); ax.set_xticklabels(plabs, fontsize=10)
        ax.set_yticks(range(12)); ax.set_yticklabels(ytick, fontsize=8)
        ax.set_title(f"{ttl}（颜色=log RMSE，越浅越优）", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.85, label="RMSE（log 刻度）")
    fig.suptitle("Stage II 六参数 RMSE：本文 ECF 与外部基准"
                 "（T=25,000，12 种子）", fontsize=13)
    save(fig, out, "fig9_stage2_baselines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_REPO, "figures"))
    ap.add_argument("--only", default="", help="逗号分隔的图号, 如 8 或 1,3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    allfigs = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6,
               7: fig7, 8: fig8, 9: fig9}
    todo = [int(t) for t in args.only.split(",") if t.strip()] \
        if args.only else sorted(allfigs)
    for n in todo:
        allfigs[n](args.out)
    print("[FIG] done %s -> %s" % (todo, args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
