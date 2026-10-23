# -*- coding: utf-8 -*-
"""
Fig 7.5 离线绘图脚本：OOS 原始电价预测对比（4 季度子图）
从 stage_6_predictions.npz 读取预测数组，生成 2×2 季度子图 PDF。

用法（需 matplotlib）：
    python plot_fig7_5_offline.py
输出：
    fig7_5_oos_price_pred.pdf  （保存至 06_stage_outputs/stage_6_price_pred/）
"""
import numpy as np
import os

# ---------- 加载数据 ----------
NPZ_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "06_stage_outputs_fix_20261011", "stage_6_price_pred", "stage_6_predictions.npz"
)
OUT_DIR = os.path.dirname(NPZ_PATH)

d = np.load(NPZ_PATH, allow_pickle=True)
ts = d["ts_next"]                    # (6550,) datetime64
P_true = d["P_true_next"]            # (6550,) float64
preds = {
    "M1 Persistence":    d["M1"],
    "M2 SeasonalNaive":  d["M2"],
    "M3 WeeklyHrDOW":    d["M3"],
    "M4 Simplified SAR": d["M4"],
    "M5 ThesisASY":      d["M5"],
}
q_lo = d["q_lo_5"]                   # M5 的 5% 分位
q_hi = d["q_hi_95"]                  # M5 的 95% 分位

n = len(ts)
quarters = [
    ("Q1 (Jan--Mar)",   0,      n // 4),
    ("Q2 (Apr--Jun)",   n//4,   n // 2),
    ("Q3 (Jul--Sep·1)", n//2,   3 * n // 4),
    ("Q3 (Jul--Sep·2)", 3*n//4, n),
]

# ---------- 绘图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

COLORS = {
    "M1 Persistence":   "#7f7f7f",   # 灰
    "M2 SeasonalNaive": "#1f77b4",   # 蓝
    "M3 WeeklyHrDOW":   "#ff7f0e",   # 橙
    "M4 Simplified SAR": "#2ca02c",   # 绿
    "M5 ThesisASY":     "#d62728",   # 红
}

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
axes = axes.ravel()

for idx, (label, i0, i1) in enumerate(quarters):
    ax = axes[idx]
    ts_q = ts[i0:i1]
    P_q  = P_true[i0:i1]

    # M5 的 95% 区间
    ax.fill_between(ts_q, q_lo[i0:i1], q_hi[i0:i1],
                    color=COLORS["M5 ThesisASY"], alpha=0.15, label="M5 90% interval")

    # 真实电价
    ax.plot(ts_q, P_q, color="black", linewidth=0.8, label="$P_{t+1}$ (true)", zorder=5)

    # 各方法预测
    for name, arr in preds.items():
        lw = 1.2 if name.startswith("M5") else 0.6
        ax.plot(ts_q, arr[i0:i1], color=COLORS[name], linewidth=lw,
                alpha=0.85 if name.startswith("M5") else 0.6, label=name)

    ax.set_title(f"2025 {label}", fontsize=11)
    ax.set_ylabel("EUR/MWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    if idx == 0:
        ax.legend(fontsize=7, ncol=3, loc="upper right")

fig.suptitle("Fig 7.5  OOS 1-step ahead $P_{t+1}$ prediction (2025-01--09, $n=6\\,550$)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = os.path.join(OUT_DIR, "fig7_5_oos_price_pred.pdf")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"[plot] saved {out_path}")
plt.close(fig)
