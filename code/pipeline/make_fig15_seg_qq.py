# -*- coding: utf-8 -*-
"""
make_fig15_seg_qq.py — 附录 C 图：三段分段的模拟法残差 QQ 图
数据源: 06_stage_outputs/empirical_v1/seg_{pre,crisis,post}/seg_*_resid_qq.csv
输出: figures/fig15_seg_qq.pdf / .png
"""
import os
import numpy as np
import pandas as pd
import matplotlib

_REPO = os.environ.get("EPEX_BASE") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = _REPO
EMP = os.path.join(BASE, "06_stage_outputs", "empirical_v1")
FIG = os.path.join(BASE, "figures")

SEGS = [
    ("seg_pre", "2019 至 2021"),
    ("seg_crisis", "2022 至 2023"),
    ("seg_post", "2024 至 2025.9"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=False, sharey=False)
for ax, (seg, title) in zip(axes, SEGS):
    qq = pd.read_csv(os.path.join(EMP, seg, f"{seg}_resid_qq.csv"))
    est = pd.read_csv(os.path.join(EMP, seg, f"{seg}_estimates.csv"))
    alpha = float(est.loc[est["parameter"] == "alpha", "est_asym"].iloc[0])
    mq, rq = qq["model_quantile"].to_numpy(), qq["resid_quantile"].to_numpy()
    ax.scatter(mq, rq, s=6, alpha=0.5, color="#1f4e79", edgecolors="none")
    lo = min(np.nanmin(mq), np.nanmin(rq))
    hi = max(np.nanmax(mq), np.nanmax(rq))
    ax.plot([lo, hi], [lo, hi], color="#c00000", lw=1.2, ls="--")
    ax.set_title(f"{title}（$\\hat\\alpha = {alpha:.3f}$）", fontsize=11)
    ax.set_xlabel("模型分位数（EUR/MWh）")
axes[0].set_ylabel("残差分位数（EUR/MWh）")
fig.suptitle("分段模拟法残差 QQ 图（非对称 TS 模型，0.1% 至 99.9% 分位）", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "fig15_seg_qq.pdf"))
fig.savefig(os.path.join(FIG, "fig15_seg_qq.png"), dpi=150)
print("saved fig15_seg_qq")
for seg, _ in SEGS:
    qq = pd.read_csv(os.path.join(EMP, seg, f"{seg}_resid_qq.csv"))
    mid = qq[(qq.q >= 0.05) & (qq.q <= 0.95)]
    dev = (mid["resid_quantile"] - mid["model_quantile"]).abs()
    print(seg, "中部最大绝对偏差 %.1f" % dev.max(),
          "| 左尾 q=0.001: 数据 %.1f 模型 %.1f" % (qq.resid_quantile.iloc[0], qq.model_quantile.iloc[0]),
          "| 右尾 q=0.999: 数据 %.1f 模型 %.1f" % (qq.resid_quantile.iloc[-1], qq.model_quantile.iloc[-1]))
