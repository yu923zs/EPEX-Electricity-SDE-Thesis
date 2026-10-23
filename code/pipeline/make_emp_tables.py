# -*- coding: utf-8 -*-
"""
================================================================
 make_emp_tables.py — 实证章(第7章)六张 LaTeX 表格自动生成
 数据源: 06_stage_outputs/empirical_v1/ 下全部 CSV/pkl (只读)
 输出:  results/emp_tables.tex (booktabs 片段, 由 07_empirical.tex 引用)
 另算并落盘嵌套F检验/同尺度稳健准则/质量加权斜率(供表7.5 与正文)
================================================================
"""
import os
import sys
import numpy as np
import pandas as pd

_REPO = os.environ.get("EPEX_BASE") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from thesis_pipeline_common import load_pickle  # noqa: E402

BASE3 = _REPO
EMP = os.path.join(BASE3, "06_stage_outputs", "empirical_v1")
FULL = os.path.join(EMP, "full")
OUT = os.path.join(BASE3, "results", "emp_tables.tex")

K1, K2 = -25.37, 2.31   # 全样本细化折点(full_clusters.csv)


def fm(x, d=2):
    return f"{x:,.{d}f}"


def sci(x, d=2):
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


def main():
    L = []
    w = L.append

    # ---------------- 表7.1 数据概况 ----------------
    ds = pd.read_csv(os.path.join(FULL, "full_data_summary.csv"),
                     encoding="utf-8-sig")
    v = dict(zip(ds["metric"], ds["value"]))
    w("% ===== 表7.1 数据概况 =====")
    w("\\begin{table}[htbp]")
    w("\\centering")
    w("\\caption{EPEX 德国 DE-LU 日前电价描述统计（2019-01-01 至 2025-09-30，小时级）}")
    w("\\label{tab:71-data}")
    w("\\begin{tabular}{lrlr}")
    w("\\toprule")
    w("指标 & 数值 & 指标 & 数值 \\\\")
    w("\\midrule")
    w(f"样本量（小时） & {int(v['n_hours']):,} & 中位数（EUR/MWh） & {fm(v['median'])} \\\\")
    w(f"缺失观测 & {int(v['n_missing_price'])} & 0.5\\% 分位 & {fm(v['q005'])} \\\\")
    w(f"负价小时数 & {int(v['n_negative_price']):,} & 99.5\\% 分位 & {fm(v['q995'])} \\\\")
    w(f"负价占比（\\%） & {fm(v['pct_negative_price'],2)} & 最小值 & {fm(v['min'])} \\\\")
    w(f"均值（EUR/MWh） & {fm(v['mean'])} & 最大值 & {fm(v['max'])} \\\\")
    w(f"标准差（EUR/MWh） & {fm(v['sd'])} & $3\\sigma$ 跳变次数 & {int(v['n_spike_abs_dP_gt_3sd']):,} \\\\")
    w(f"季节拟合 $R^2$ & {fm(v['seasonal_R2_full'],3)} & 去季节状态范围 & "
      f"$[{fm(v['X_min'],0)},\\ {fm(v['X_max'],0)}]$ \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\end{table}\n")

    # ---------------- 表7.2 Stage I ----------------
    fd = pd.read_csv(os.path.join(FULL, "full_final_drift.csv"),
                     encoding="utf-8-sig")
    cl = pd.read_csv(os.path.join(FULL, "full_clusters.csv"),
                     encoding="utf-8-sig")
    pip = pd.read_csv(os.path.join(FULL, "full_pip_coef.csv"),
                      encoding="utf-8-sig")
    pip_x1 = float(pip.loc[pip["candidate"] == "x^1", "PIP"].iloc[0])
    cpip = dict(zip(np.round(cl["c_refined"], 2), cl["cpip"]))
    name_map = {"intercept": "截距 $\\beta_0$", "x^1": "$x$",
                "(x--25.37)_+": "$(x+25.37)_+$", "(x-2.31)_+": f"$(x-{K2})_+$"}
    w("% ===== 表7.2 Stage I 漂移结构识别 =====")
    w("\\begin{table}[htbp]")
    w("\\centering")
    w("\\caption{Stage I 全样本漂移结构识别结果（59,158 个增量，剖面细化后 Student-t 稳健重估）}")
    w("\\label{tab:72-stage1}")
    w("\\begin{tabular}{lrrrl}")
    w("\\toprule")
    w("项 & 系数 & 标准误 & $|t|$ & 结构证据 \\\\")
    w("\\midrule")
    for _, row in fd.iterrows():
        nm = row["term"]
        if nm == "intercept":
            ev = "始终保留"
        elif nm == "x^1":
            ev = f"PIP $= {pip_x1:.2f}$"
        else:
            import re
            key = round(float(re.search(r"\(x-(.+?)\)_\+", nm).group(1)), 2)
            ev = f"簇 cpip $= {cpip.get(key, float('nan')):.3f}$"
        w(f"{name_map[nm]} & {fm(row['coef'],4)} & {fm(row['se'],4)} & "
          f"{fm(row['abs_t'],1)} & {ev} \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\begin{flushleft}\\footnotesize 注：候选折点 $-133.45$ 单点 PIP 为 0.98，"
      "因其左侧样本占比仅约 2\\%，低于两侧质量规则 5\\% 的下限，按方法设计被过滤。\\end{flushleft}")
    w("\\end{table}\n")

    # ---------------- 表7.3 Stage II ----------------
    es = pd.read_csv(os.path.join(FULL, "full_estimates.csv"),
                     encoding="utf-8-sig")
    obj = pd.read_csv(os.path.join(FULL, "full_objective.csv"),
                      encoding="utf-8-sig")
    Q_asy = float(obj.loc[obj["model"] == "asym", "Q_min"].iloc[0])
    Q_sym = float(obj.loc[obj["model"] == "sym", "Q_min"].iloc[0])
    pn = {"mu": "$\\mu$", "C_plus": "$C_+$", "C_minus": "$C_-$",
          "M": "$M$", "G": "$G$", "alpha": "$\\alpha$"}
    w("% ===== 表7.3 Stage II 噪声参数 =====")
    w("\\begin{table}[htbp]")
    w("\\centering")
    w("\\caption{Stage II 非对称 TS 参数估计与对称受限参照（块自助标准误，$B=200$，块长 24）}")
    w("\\label{tab:73-stage2}")
    w("\\begin{tabular}{lcccc}")
    w("\\toprule")
    w("参数 & 非对称估计 & 标准误 & 对称受限参照 & 标准误 \\\\")
    w("\\midrule")
    for _, row in es.iterrows():
        star = "$^{\\dagger}$" if row["parameter"] == "alpha" else ""
        w(f"{pn[row['parameter']]}{star} & {fm(row['est_asym'],4)} & "
          f"{fm(row['se_asym'],4)} & {fm(row['est_sym'],4)} & {fm(row['se_sym'],4)} \\\\")
    w("\\midrule")
    w(f"$Q$（ECF 目标） & {sci(Q_asy)} &  & {sci(Q_sym)} &  \\\\")
    w(f"RMS 偏差（\\%） & {100*np.sqrt(Q_asy):.2f} &  & {100*np.sqrt(Q_sym):.2f} &  \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\begin{flushleft}\\footnotesize 注：$^{\\dagger}\\alpha$ 的非对称估计撞在参数空间下界 "
      "$10^{-3}$，目标函数在 $\\alpha\\in(0,0.05)$ 上变化不超过 0.5\\%，"
      "属弱识别下的边界最优，其标准误仅供参考，详见正文诊断。\\end{flushleft}")
    w("\\end{table}\n")

    # ---------------- 表7.4 非对称检验 ----------------
    ay = pd.read_csv(os.path.join(FULL, "asymtest_summary.csv"),
                     encoding="utf-8-sig")
    a = dict(zip(ay["metric"], ay["value"]))
    w("% ===== 表7.4 非对称性检验 =====")
    w("\\begin{table}[htbp]")
    w("\\centering")
    w("\\caption{噪声非对称性检验（参数自助零分布）}")
    w("\\label{tab:74-asym}")
    w("\\begin{tabular}{lr}")
    w("\\toprule")
    w("指标 & 数值 \\\\")
    w("\\midrule")
    w(f"观测统计量 $\\Delta Q = Q_{{\\mathrm{{sym}}}}-Q_{{\\mathrm{{asy}}}}$ & {sci(a['dQ_obs'])} \\\\")
    w(f"零分布重复数（有效） & {int(a['K_valid'])} \\\\")
    w(f"零分布均值 & {sci(a['null_dQ_mean'])} \\\\")
    w(f"零分布标准差 & {sci(a['null_dQ_sd'])} \\\\")
    w(f"零分布 95\\% 分位 & {sci(a['null_dQ_q95'])} \\\\")
    w(f"$p$ 值 & {a['p_value']:.4f} \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\begin{flushleft}\\footnotesize 注：零分布 200 个重复中无一超过观测值，"
      "$p=1/201$ 为该重复数下的最小可达值。\\end{flushleft}")
    w("\\end{table}\n")

    # ---------------- 表7.5 Vasicek 基准 ----------------
    dfX = pd.read_csv(os.path.join(FULL, "full_X_full.csv"), encoding="utf-8-sig")
    X = dfX["X"].to_numpy(dtype=np.float64)
    X = X[np.isfinite(X)]
    y = np.diff(X)
    x = X[:-1]
    n = len(y)
    A = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ coef
    s2 = float(np.sum(e ** 2) / (n - 2))
    se_ab = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * s2)
    t_ab = np.abs(coef) / se_ab
    H = np.column_stack([A, np.maximum(x - K1, 0), np.maximum(x - K2, 0)])
    c2, *_ = np.linalg.lstsq(H, y, rcond=None)
    e2 = y - H @ c2
    rss1, rss2 = float((e ** 2).sum()), float((e2 ** 2).sum())
    F_stat = ((rss1 - rss2) / 2) / (rss2 / (n - 4))
    artB = load_pickle(os.path.join(FULL, "full_stageB_artifacts.pkl"))
    r_ours = artB["resid"]
    s_common = np.median(np.abs(r_ours - np.median(r_ours))) / 0.6744897501960817
    crit_e = float(np.mean(np.log1p((e / s_common) ** 2 / 4.0)))
    crit_r = float(np.mean(np.log1p((r_ours / s_common) ** 2 / 4.0)))
    m1 = x < K1
    m3 = x > K2
    m2 = ~(m1 | m3)
    co = dict(zip(fd["term"], fd["coef"]))
    sl_low = co["x^1"]
    sl_mid = co["x^1"] + co["(x--25.37)_+"]
    sl_high = sl_mid + co["(x-2.31)_+"]
    w_avg = m1.mean() * sl_low + m2.mean() * sl_mid + m3.mean() * sl_high
    q22 = pd.read_csv(os.path.join(FULL, "vasicek_2x2_Q.csv"),
                      encoding="utf-8-sig")
    Q = {(d, nn): qq for d, nn, qq in
         zip(q22["drift"], q22["noise"], q22["Q"])}

    w("% ===== 表7.5 Vasicek 基准比较 =====")
    w("\\begin{table}[htbp]")
    w("\\centering")
    w("\\caption{与经典 Vasicek 基准的比较}")
    w("\\label{tab:75-vasicek}")
    w("\\begin{tabular}{lcc}")
    w("\\toprule")
    w("\\multicolumn{3}{l}{\\textbf{A. Vasicek 基准估计}（$\\Delta X_i=a+bX_i+\\varepsilon_i$，"
      "$\\varepsilon$ 高斯）}\\\\")
    w("\\midrule")
    w(" & 估计值 & $|t|$ \\\\")
    w(f"截距 $a$ & {fm(coef[0],4)} & {fm(t_ab[0],1)} \\\\")
    w(f"斜率 $b$ & {fm(coef[1],5)} & {fm(t_ab[1],1)} \\\\")
    w(f"残差标准差 & {fm(np.std(e),2)} & \\\\")
    w("\\midrule")
    w("\\multicolumn{3}{l}{\\textbf{B. 噪声刻画的贡献}（同一残差集内可比，ECF 目标 $Q$）}\\\\")
    w("\\midrule")
    w(" & 高斯噪声 & TS 噪声 \\\\")
    w(f"Vasicek 残差 & {sci(Q[('Vasicek(线性)','Gaussian')])} "
      f"（RMS {100*np.sqrt(Q[('Vasicek(线性)','Gaussian')]):.1f}\\%） & "
      f"{sci(Q[('Vasicek(线性)','TS-ECF')])} "
      f"（RMS {100*np.sqrt(Q[('Vasicek(线性)','TS-ECF')]):.1f}\\%） \\\\")
    w(f"本文残差 & {sci(Q[('本文(分段)','Gaussian')])} "
      f"（RMS {100*np.sqrt(Q[('本文(分段)','Gaussian')]):.1f}\\%） & "
      f"{sci(Q[('本文(分段)','TS-ECF')])} "
      f"（RMS {100*np.sqrt(Q[('本文(分段)','TS-ECF')]):.1f}\\%） \\\\")
    w("\\midrule")
    w("\\multicolumn{3}{l}{\\textbf{C. 漂移结构的有效对比}（同一数据、同一准则）}\\\\")
    w("\\midrule")
    w(f"嵌套 $F$ 检验（两铰链项联合） & \\multicolumn{{2}}{{c}}"
      f"{{$F(2,{n-4:,}) = {F_stat:.1f}$，$p<0.001$}} \\\\")
    w(f"稳健准则（同尺度，越低越好） & 线性 {crit_e:.4f} & 本文 {crit_r:.4f} \\\\")
    w(f"斜率交叉核对 & 线性 $b={coef[1]:.4f}$ & 分段加权 ${w_avg:.4f}$ \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\begin{flushleft}\\footnotesize 注：B 中两行基于不同残差集，"
      "跨行比较不构成模型优劣的证据，有效口径为行内高斯与 TS 的对比。"
      "分段加权为三段斜率按样本质量加权。\\end{flushleft}")
    w("\\end{table}\n")

    # ---------------- 表7.6 分段稳健性 ----------------
    sg = pd.read_csv(os.path.join(EMP, "empR_segments.csv"),
                     encoding="utf-8-sig")
    seg_name = {"seg_pre": "危机前（2019 至 2021）",
                "seg_crisis": "危机期（2022 至 2023）",
                "seg_post": "后危机（2024 至 2025.9）"}
    w("% ===== 表7.6 分段稳健性 =====")
    w("\\begin{table}[htbp]")
    w("\\centering")
    w("\\caption{分段稳健性：三个子样本独立重估}")
    w("\\label{tab:76-segments}")
    w("\\begin{tabular}{llrrrrrrr}")
    w("\\toprule")
    w("子样本 & 折点 & $\\mu$ & $C_+$ & $C_-$ & $M$ & $G$ & $\\alpha$ \\\\")
    w("\\midrule")
    for _, row in sg.iterrows():
        knots = row["knots"] if isinstance(row["knots"], str) and row["knots"] else "无显著折点"
        w(f"{seg_name[row['segment']]} & {knots} & {fm(row['mu'],2)} & "
          f"{fm(row['C_plus'],2)} & {fm(row['C_minus'],2)} & {fm(row['M'],3)} & "
          f"{fm(row['G'],3)} & {fm(row['alpha'],3)} \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\end{table}\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[tables] written: {OUT}")
    print(f"[check] F={F_stat:.1f}, crit {crit_e:.4f}->{crit_r:.4f}, "
          f"wavg={w_avg:.4f} vs b={coef[1]:.4f}")
    print(f"[check] slopes: low={sl_low:.4f}, mid={sl_mid:.4f}, high={sl_high:.4f}, "
          f"mass=({m1.mean():.3f},{m2.mean():.3f},{m3.mean():.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
