# -*- coding: utf-8 -*-
"""
dump_v13_models.py — 复现 v13 (T=25000, S=12) Stage I，转储每个种子识别出的精确模型
种子与数据缓存与正式实验完全一致（gen_data 磁盘缓存 + Gibbs seed+2000），
因此识别结果与 lab_rows_v13_T25000_S12_light.csv 中记录的指标逐位对应；
脚本末尾自动与记录值核对 tpr/fdr/f1/wass/medse 并报告最大偏差。
输出: results/v13_T25000_S12_identified_models.json
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_lab as R

VC = R.VARIANTS["v13"]


def dump_one(task):
    seed, dgp, T = task
    X_deseas, _ = R.gen_data(seed, dgp, T)
    s1 = R.stageI(X_deseas, VC, seed)
    refined_cs, b_level = R.profile_refine(
        s1, margin=VC.get("refine_margin", 3.0),
        t_thr=VC.get("t_thr", 0.0), sel_ic=VC.get("sel_ic", None))
    m = R.selection_metrics(s1, dgp, refined_cs=refined_cs)
    medse = R.medse_metric(b_level, s1["x_level"], dgp)

    # 重建最终 IRLS 系数（与 profile_refine 尾部同一确定性过程）
    sel = s1["selected"]; hinge_map = s1["hinge_map"]
    colnames = s1["colnames"]
    poly_cols = [j for j in np.where(sel)[0] if j not in hinge_map]
    y = s1["y"]; x = s1["x_level"]
    Xpoly = s1["Phi_level"][:, poly_cols] if poly_cols else np.zeros((len(y), 0))
    blocks = [Xpoly] if Xpoly.shape[1] else []
    for c in refined_cs:
        blocks.append(np.maximum(x - c, 0.0))
    Xfin = np.column_stack(blocks) if blocks else np.zeros((len(y), 0))
    terms, knots = [], []
    if Xfin.shape[1] > 0:
        coef = R.refit_student_t(y, Xfin, n_iter=40, nu=4.0)
        a0 = float(coef[0]); cvec = coef[1:]
        r = y - a0 - Xfin @ cvec
        s = max(np.median(np.abs(r - np.median(r))) / 0.6744897501960817, 1e-8)
        u = 5.0 / (4.0 + (r / s) ** 2)
        A = np.column_stack([np.ones(len(y)), Xfin])
        covb = np.linalg.inv(A.T @ (A * u[:, None]) + 1e-10 * np.eye(A.shape[1])) * (s ** 2)
        se = np.sqrt(np.maximum(np.diag(covb), 1e-300))
        t_all = np.abs(coef) / se
        n_poly = Xfin.shape[1] - len(refined_cs)
        for k, j in enumerate(poly_cols):
            terms.append(dict(term=colnames[j], coef=float(cvec[k]),
                              t=float(t_all[1 + k])))
        for i, c in enumerate(refined_cs):
            knots.append(dict(c=float(c), delta=float(cvec[n_poly + i]),
                              t=float(t_all[1 + n_poly + i])))
    else:
        a0 = float(np.median(y))
    return dict(seed=int(seed), dgp=dgp, intercept=a0,
                poly_terms=terms, knots=knots,
                nsel=int(m["nsel"]), tpr=float(m["tpr"]), fdr=float(m["fdr"]),
                f1=float(m["f1"]), wass=float(m["wass"]), medse=float(medse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--T", type=int, default=25000)
    ap.add_argument("--seed0", type=int, default=91300000)
    args = ap.parse_args()

    tasks = [(args.seed0 + s, dgp, args.T)
             for s in range(args.S) for dgp in ("A", "B", "C")]
    print(f"[DUMP] v13 identified models: {len(tasks)} tasks", flush=True)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            out = list(ex.map(dump_one, tasks))
    else:
        out = [dump_one(t) for t in tasks]

    fres = os.path.join(R.RES_DIR, f"v13_T{args.T}_S{args.S}_identified_models.json")
    with open(fres, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    # ---- 与正式记录核对 ----
    rec = pd.read_csv(os.path.join(
        R.RES_DIR, f"lab_rows_v13_T{args.T}_S{args.S}_light.csv"))
    rep = pd.DataFrame([{k: r[k] for k in
                         ("seed", "dgp", "tpr", "fdr", "f1", "wass", "medse")}
                        for r in out])
    mg = rep.merge(rec, on=("seed", "dgp"), suffixes=("_new", "_rec"))
    diffs = {k: float(np.max(np.abs(mg[f"{k}_new"] - mg[f"{k}_rec"])))
             for k in ("tpr", "fdr", "f1", "wass", "medse")}
    print(f"[VERIFY] max |dump - recorded| = {diffs}", flush=True)
    print(f"[DUMP] saved -> {fres}", flush=True)
    for r in out:
        ks = ", ".join(f"c={k['c']:.2f}(d={k['delta']:+.4f},t={k['t']:.1f})"
                       for k in r["knots"]) or "-"
        ps = ", ".join(f"{t['term']}={t['coef']:+.4f}" for t in r["poly_terms"]) or "-"
        print(f"  {r['dgp']} seed={r['seed']}: a0={r['intercept']:+.4f} | "
              f"{ps} | knots: {ks} | tpr={r['tpr']:.2f} fdr={r['fdr']:.2f} "
              f"medse={r['medse']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
