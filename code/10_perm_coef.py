"""P4 + P5: headline permutation test and LR coefficient stability.

P5 (fast): refit the headline models across outer LOPO folds, capturing the
standardised-space LogisticRegression coefficients after the StandardScaler.
Summarise per-feature median coefficient, sign consistency across folds, and
median |coef|. Output: results/coef_stability.csv

P4 (slow, background): full nested-LOPO label-permutation test for the
headline model (EEG_F1_PSI, suction, post_bps_ge7, n=98). 500 permutations;
empirical p = (1 + #{AUC_perm >= AUC_obs}) / 501.
Output: results/perm_headline.json

Usage:
  python code/10_perm_coef.py coef   # P5 only
  python code/10_perm_coef.py perm   # P4 only
  python code/10_perm_coef.py both   # default
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

C = import_module("00_common")
X2 = import_module("02_run_experiments")

N_PERM = 500

COEF_MODELS = [
    ("EEG_F1_PSI", ["EEG_F1", "pre_sed_mean"]),
    ("EEG_F2", "EEG_F2"),
]


def nested_loso_coef(X: pd.DataFrame, y: np.ndarray, ids: np.ndarray):
    """nested_loso that also records standardised-space coefficients."""
    n = len(y)
    oof = np.full(n, np.nan)
    coefs = []
    chosen = []
    feat_names = list(X.columns)
    for pid in np.unique(ids):
        te = ids == pid
        tr = ~te
        Xtr, ytr = X.values[tr], y[tr]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=X2.RNG)
        best_c, best_auc = None, -np.inf
        for cv in X2.C_GRID:
            scores = []
            for itr, ite in skf.split(Xtr, ytr):
                if len(np.unique(ytr[itr])) < 2:
                    continue
                pipe = X2.fit_proba(Xtr[itr], ytr[itr], cv)
                p = pipe.predict_proba(Xtr[ite])[:, 1]
                scores.append((p, ytr[ite]))
            if not scores:
                continue
            allp = np.concatenate([s[0] for s in scores])
            ally = np.concatenate([s[1] for s in scores])
            auc = X2.rank_auc(allp, ally)
            if auc > best_auc + 1e-12:
                best_auc, best_c = auc, cv
        if best_c is None:
            best_c = 1.0
        chosen.append(best_c)
        pipe = X2.fit_proba(Xtr, ytr, best_c)
        oof[te] = pipe.predict_proba(X.values[te])[:, 1]
        lr = pipe.named_steps["logisticregression"]
        coefs.append(np.asarray(lr.coef_).ravel())
    return oof, np.asarray(coefs), chosen, feat_names


def p5_coef_stability() -> None:
    t0 = time.time()
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    fdf = fdf[fdf["task"] == "EndotrachealSuction"]
    rows = []
    for label in ("post_bps_ge7", "post_bps_gt3"):
        d = fdf.dropna(subset=[label])
        for mname, spec in COEF_MODELS:
            cols = X2.model_columns(spec, d.columns)
            dd = d.dropna(subset=cols)
            y = dd[label].astype(int).values
            if y.sum() < 5 or (len(y) - y.sum()) < 5:
                continue
            oof, coefs, chosen, names = nested_loso_coef(
                dd[cols], y, dd["participant_id"].values)
            auc, lo, hi = X2.boot_ci(oof, y, dd["participant_id"].values)
            for j, fn in enumerate(names):
                v = coefs[:, j]
                rows.append(dict(
                    label=label, model=mname, feature=fn,
                    n_folds=len(v), median_coef=float(np.median(v)),
                    mean_coef=float(np.mean(v)),
                    sign_consistency=float(np.mean(np.sign(v) ==
                                                  np.sign(np.median(v)))),
                    median_abs=float(np.median(np.abs(v))),
                    min_coef=float(v.min()), max_coef=float(v.max())))
            print(f"[coef] {label} {mname}: n={len(y)} AUC={auc:.3f} "
                  f"folds={len(coefs)} C~{np.median(chosen):.4g}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "coef_stability.csv", index=False)
    print(f"wrote {C.RESULTS / 'coef_stability.csv'} in {time.time()-t0:.1f}s",
          flush=True)


def p4_permutation() -> None:
    t0 = time.time()
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    d = fdf[fdf["task"] == "EndotrachealSuction"]
    cols = X2.model_columns(["EEG_F1", "pre_sed_mean"], d.columns)
    dd = d.dropna(subset=cols + ["post_bps_ge7"])
    X = dd[cols]
    ids = dd["participant_id"].values
    y = dd["post_bps_ge7"].astype(int).values

    oof, _ = X2.nested_loso(X, y, ids)
    auc_obs = X2.rank_auc(oof, y)
    print(f"[perm] observed AUC={auc_obs:.4f} n={len(y)} pos={y.sum()} "
          f"(feature: EEG_F1_PSI)", flush=True)

    rng = np.random.default_rng(X2.RNG)
    ge = 0
    aucs = []
    for i in range(N_PERM):
        yp = rng.permutation(y)
        if yp.sum() < 5 or (len(yp) - yp.sum()) < 5:
            continue
        oof_p, _ = X2.nested_loso(X, yp, ids)
        a = X2.rank_auc(oof_p, yp)
        aucs.append(a)
        if a >= auc_obs - 1e-12:
            ge += 1
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"[perm] {i+1}/{N_PERM} elapsed={elapsed:.0f}s "
                  f"ge={ge} running_p={(1+ge)/(i+2):.4f}", flush=True)

    n_done = len(aucs)
    p_emp = (1 + ge) / (1 + n_done)
    res = dict(model="EEG_F1_PSI", cohort="EndotrachealSuction",
               label="post_bps_ge7", n=int(len(y)), pos=int(y.sum()),
               n_perm=n_done, auc_obs=float(auc_obs),
               auc_perm_mean=float(np.mean(aucs)),
               auc_perm_p95=float(np.percentile(aucs, 95)),
               p_empirical=float(p_emp),
               runtime_s=float(time.time() - t0))
    with open(C.RESULTS / "perm_headline.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[perm] done: AUC_obs={auc_obs:.4f} "
          f"perm_mean={res['auc_perm_mean']:.4f} "
          f"p={p_emp:.4f} in {res['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("coef", "both"):
        p5_coef_stability()
    if mode in ("perm", "both"):
        p4_permutation()
