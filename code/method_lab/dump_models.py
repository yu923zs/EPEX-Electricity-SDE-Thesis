# -*- coding: utf-8 -*-
"""
dump_models.py — 通用版：复现任意变体 Stage I，转储每种子识别模型 + 漂移曲线
与正式实验同种子、同数据缓存；末尾自动核对 lab_rows 记录指标。
输出:
  results/{variant}_T{T}_S{S}_identified_models.json   (模型结构)
  results/{variant}_T{T}_S{S}_curves.npz               (x, b_hat, b0 曲线)
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_lab as R


def dump_one(task):
    seed, dgp, T, vname = task
    vc = R.VARIANTS[vname]
    X_deseas, _ = R.gen_data(seed, dgp, T)
    s1 = R.stageI(X_deseas, vc, seed)
    sel = s1["selected"]; hinge_map = s1["hinge_map"]
    colnames = s1["colnames"]
    y = s1["y"]; x = s1["x_level"]
    terms, knots = [], []

    if vc.get("profile_refine"):
        refined_cs, b_level = R.profile_refine(
            s1, margin=vc.get("refine_margin", 3.0),
            t_thr=vc.get("t_thr", 0.0), sel_ic=vc.get("sel_ic", None))
        m = R.selection_metrics(s1, dgp, refined_cs=refined_cs)
        poly_cols = [j for j in np.where(sel)[0] if j not in hinge_map]
        Xpoly = s1["Phi_level"][:, poly_cols] if poly_cols else np.zeros((len(y), 0))
        blocks = [Xpoly] if Xpoly.shape[1] else []
        for c in refined_cs:
            blocks.append(np.maximum(x - c, 0.0))
        Xfin = np.column_stack(blocks) if blocks else np.zeros((len(y), 0))
        if Xfin.shape[1] > 0:
            coef = R.refit_student_t(y, Xfin, n_iter=40, nu=4.0)
            a0 = float(coef[0]); cvec = coef[1:]
            r = y - a0 - Xfin @ cvec
            s = max(np.median(np.abs(r - np.median(r))) / 0.6744897501960817, 1e-8)
            u = 5.0 / (4.0 + (r / s) ** 2)
            A = np.column_stack([np.ones(len(y)), Xfin])
            covb = np.linalg.inv(A.T @ (A * u[:, None])
                                 + 1e-10 * np.eye(A.shape[1])) * (s ** 2)
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
    else:
        m = R.selection_metrics(s1, dgp)
        if vc.get("refit") and m["nsel"] > 0:
            cols = np.where(sel)[0]
            Phi_sel = s1["Phi_level"][:, cols]
            coef = R.refit_student_t(y, Phi_sel)
            b_level = coef[0] + Phi_sel @ coef[1:]
            a0 = float(coef[0]); cm = coef[1:]
        else:
            b_level = s1["gamma_med"] + s1["Phi_level"] @ s1["beta_med"]
            a0 = float(s1["gamma_med"]); cm = s1["beta_med"]
        for j in np.where(sel)[0]:
            if j in hinge_map:
                knots.append(dict(c=float(hinge_map[j]), delta=float(cm[j]),
                                  t=float("nan")))
            else:
                terms.append(dict(term=colnames[j], coef=float(cm[j]),
                                  t=float("nan")))

    medse = R.medse_metric(b_level, x, dgp)
    b0 = R.true_drift_arr(x, dgp)
    return (dict(seed=int(seed), dgp=dgp, variant=vname, intercept=a0,
                 poly_terms=terms, knots=knots,
                 nsel=int(m["nsel"]), tpr=float(m["tpr"]), fdr=float(m["fdr"]),
                 f1=float(m["f1"]), wass=float(m["wass"]), medse=float(medse)),
            x, b_level, b0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v13")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--seed0", type=int, default=91300000)
    args = ap.parse_args()

    tasks = [(args.seed0 + s, dgp, args.T, args.variant)
             for s in range(args.S) for dgp in ("A", "B", "C")]
    print(f"[DUMP] {args.variant} T={args.T} S={args.S}: {len(tasks)} tasks",
          flush=True)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(dump_one, tasks))
    else:
        results = [dump_one(t) for t in tasks]

    out = [r[0] for r in results]
    curves = {}
    for (meta, x, b, b0) in results:
        k = f"{meta['dgp']}_{meta['seed']}"
        curves[f"x_{k}"], curves[f"b_{k}"], curves[f"b0_{k}"] = x, b, b0

    fjson = os.path.join(R.RES_DIR,
                         f"{args.variant}_T{args.T}_S{args.S}_identified_models.json")
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    fnpz = os.path.join(R.RES_DIR,
                        f"{args.variant}_T{args.T}_S{args.S}_curves.npz")
    np.savez_compressed(fnpz, **curves)

    rec = pd.read_csv(os.path.join(
        R.RES_DIR, f"lab_rows_{args.variant}_T{args.T}_S{args.S}_none.csv"))
    rep = pd.DataFrame([{k: r[k] for k in
                         ("seed", "dgp", "tpr", "fdr", "f1", "wass", "medse")}
                        for r in out])
    mg = rep.merge(rec, on=("seed", "dgp"), suffixes=("_new", "_rec"))
    diffs = {k: float(np.max(np.abs(mg[f"{k}_new"] - mg[f"{k}_rec"])))
             for k in ("tpr", "fdr", "f1", "wass", "medse")}
    print(f"[VERIFY] max |dump - recorded| = {diffs}", flush=True)
    print(f"[DUMP] saved -> {fjson} , {fnpz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
