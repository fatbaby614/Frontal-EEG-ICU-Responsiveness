"""Main experiments: nested-CV patient-level models on ds008115.

Systems  : EEG (feature sets F1-F4) | PSI | NOL | combinations | +PreBPS adjustment
Labels   : Post_BPS>=7 (primary), Post_BPS>3 (sensitivity)
Cohorts  : EndotrachealSuction (primary), Positioning (secondary)
Validation: outer leave-one-subject-out, inner stratified 5-fold for C selection
Statistics: pooled AUROC + patient-level bootstrap 95% CI; paired bootstrap
            for delta-AUC contrasts

Outputs
  results/main_results.csv
  results/preds_long.csv
  results/delta_auc.csv
  results/univariate.csv
  results/selected_C.csv
  results/decision_log.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

C = import_module("00_common")

RNG = 42
N_BOOT = 2000
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

FEATURE_SETS = {
    "EEG_F1": lambda c: [f"F1_norm_{b}" for b in C.BANDS_BJA],
    "EEG_F2": lambda c: ([f"F2_norm_{b}" for b in C.BANDS_CANON]
                         + [f"F2_logabs_{b}" for b in C.BANDS_CANON]
                         + ["F2_spen", "F2_sef95"]),
    "EEG_F3": lambda c: [f"F3_dyn_{b}_{s}" for b in C.BANDS_CANON
                         for s in ("mean", "sd", "slope")],
    "EEG_F4": lambda c: ([f"F3_dyn_{b}_{s}" for b in C.BANDS_CANON
                          for s in ("mean", "sd", "slope")]
                         + [f"F4_riem_{b}_{k}" for b in C.BANDS_RIEM
                            for k in range(10)]),
}
PRIMARY_EEG = "EEG_F2"

MODELS = [
    # (model_name, feature spec: list of columns or system tokens)
    ("EEG_F1", "EEG_F1"),
    ("EEG_F2", "EEG_F2"),
    ("EEG_F3", "EEG_F3"),
    ("EEG_F4", "EEG_F4"),
    ("PSI", ["pre_sed_mean"]),
    ("NOL", ["pre_nol_mean"]),
    ("PSI_NOL", ["pre_sed_mean", "pre_nol_mean"]),
    ("EEG_F1_PSI", ["EEG_F1", "pre_sed_mean"]),
    ("EEG_F2_PSI", ["EEG_F2", "pre_sed_mean"]),
    ("EEG_F2_PSI_NOL", ["EEG_F2", "pre_sed_mean", "pre_nol_mean"]),
    ("PSI_PreBPS", ["pre_sed_mean", "pre_bps"]),
    ("EEG_F2_PreBPS", ["EEG_F2", "pre_bps"]),
]

FAMILIES = {  # analysis sets restricted to complete cases for fair contrasts
    "A_ge7": dict(label="post_bps_ge7", models=["EEG_F2", "PSI", "EEG_F2_PSI"],
                  require=["pre_sed_mean"]),
    "A_gt3": dict(label="post_bps_gt3", models=["EEG_F2", "PSI", "EEG_F2_PSI"],
                  require=["pre_sed_mean"]),
    "B_ge7": dict(label="post_bps_ge7",
                  models=["PSI_NOL", "EEG_F2_PSI", "EEG_F2_PSI_NOL"],
                  require=["pre_sed_mean", "pre_nol_mean"]),
    "B_gt3": dict(label="post_bps_gt3",
                  models=["PSI_NOL", "EEG_F2_PSI", "EEG_F2_PSI_NOL"],
                  require=["pre_sed_mean", "pre_nol_mean"]),
}
PRIMARY_CONTRASTS = [  # (delta name, model a, model b, family)
    ("d_EEGvsPSI_A", "EEG_F2_PSI", "PSI", "A"),
    ("d_EEGvsPSI_NOL_B", "EEG_F2_PSI_NOL", "PSI_NOL", "B"),
]


def model_columns(spec, cols: list[str]) -> list[str]:
    spec = [spec] if isinstance(spec, str) else spec
    out = []
    for s in spec:
        out += FEATURE_SETS[s](cols) if s in FEATURE_SETS else [s]
    return out


def make_lr(C_value: float) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2", C=C_value, solver="liblinear", max_iter=500, random_state=RNG)


def fit_proba(Xtr, ytr, C_value) -> make_pipeline:
    pipe = make_pipeline(StandardScaler(), make_lr(C_value))
    pipe.fit(Xtr, ytr)
    return pipe


def nested_loso(X: pd.DataFrame, y: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, list]:
    """Return pooled out-of-fold probabilities and selected C per fold."""
    n = len(y)
    oof = np.full(n, np.nan)
    chosen = []
    uniq = np.unique(ids)
    for pid in uniq:
        te = ids == pid
        tr = ~te
        Xtr, ytr = X.values[tr], y[tr]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
        best_c, best_auc = None, -np.inf
        for cv in C_GRID:
            scores = []
            for itr, ite in skf.split(Xtr, ytr):
                if len(np.unique(ytr[itr])) < 2:
                    continue
                pipe = fit_proba(Xtr[itr], ytr[itr], cv)
                p = pipe.predict_proba(Xtr[ite])[:, 1]
                scores.append((p, ytr[ite]))
            if not scores:
                continue
            allp = np.concatenate([s[0] for s in scores])
            ally = np.concatenate([s[1] for s in scores])
            auc = rank_auc(allp, ally)
            if auc > best_auc + 1e-12:
                best_auc, best_c = auc, cv
        if best_c is None:
            best_c = 1.0
        chosen.append(best_c)
        pipe = fit_proba(Xtr, ytr, best_c)
        oof[te] = pipe.predict_proba(X.values[te])[:, 1]
    return oof, chosen


def rank_auc(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return np.nan
    n1, n0 = int(y.sum()), int((y == 0).sum())
    r = pd.Series(p).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(p, y, ids, n_boot=N_BOOT, seed=RNG) -> tuple[float, float, float]:
    """Patient-level bootstrap CI for pooled AUROC."""
    auc0 = rank_auc(p, y)
    df = pd.DataFrame({"p": p, "y": y, "id": ids})
    rng = np.random.default_rng(seed)
    aucs = []
    uniq = df["id"].unique()
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(df["id"].values == u)[0] for u in pick])
        a = rank_auc(df["p"].values[idx], df["y"].values[idx])
        if not np.isnan(a):
            aucs.append(a)
    aucs = np.sort(np.array(aucs))
    return float(auc0), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def boot_delta(p1, p2, y, ids, n_boot=N_BOOT, seed=RNG):
    """Paired patient bootstrap for AUC(p1) - AUC(p2)."""
    df = pd.DataFrame({"p1": p1, "p2": p2, "y": y, "id": ids})
    d0 = rank_auc(df.p1, df.y) - rank_auc(df.p2, df.y)
    rng = np.random.default_rng(seed)
    ds = []
    uniq = df["id"].unique()
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(df["id"].values == u)[0] for u in pick])
        d = rank_auc(df.p1.values[idx], df.y.values[idx]) - \
            rank_auc(df.p2.values[idx], df.y.values[idx])
        if not np.isnan(d):
            ds.append(d)
    ds = np.sort(np.array(ds))
    lo, hi = np.percentile(ds, 2.5), np.percentile(ds, 97.5)
    pval = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return float(d0), float(lo), float(hi), float(min(1.0, pval))


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    fdf = fdf[fdf["task"].isin(["EndotrachealSuction", "Positioning"])]

    all_rows, preds = [], []
    selected_c = []

    for task in ("EndotrachealSuction", "Positioning"):
        for label in ("post_bps_ge7", "post_bps_gt3"):
            d = fdf[fdf["task"] == task].copy()
            d = d.dropna(subset=[label])
            for mname, spec in MODELS:
                cols = model_columns(spec, d.columns)
                dd = d.dropna(subset=cols)  # complete cases for this model
                X, y = dd[cols], dd[label].astype(int).values
                if y.sum() < 5 or (len(y) - y.sum()) < 5:
                    print(f"[skip] {task}/{label}/{mname}: too few positives")
                    continue
                oof, ch = nested_loso(X, y, dd["participant_id"].values)
                auc, lo, hi = boot_ci(oof, y, dd["participant_id"].values)
                all_rows.append(dict(cohort=task, label=label, model=mname,
                                     n=len(y), n_pos=int(y.sum()),
                                     auroc=auc, ci_lo=lo, ci_hi=hi))
                for pid, yy, pp in zip(dd["participant_id"], y, oof):
                    preds.append(dict(cohort=task, label=label, model=mname,
                                      participant_id=pid, y=int(yy), p=float(pp)))
                selected_c += [dict(cohort=task, label=label, model=mname, C=c)
                               for c in ch]
                print(f"{task} {label} {mname}: n={len(y)} pos={y.sum()} "
                      f"AUC={auc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)

    res = pd.DataFrame(all_rows)
    res.to_csv(C.RESULTS / "main_results.csv", index=False)
    pd.DataFrame(preds).to_csv(C.RESULTS / "preds_long.csv", index=False)
    pd.DataFrame(selected_c).to_csv(C.RESULTS / "selected_C.csv", index=False)

    # ---- univariate feature table (suction, primary label)
    d = fdf[fdf["task"] == "EndotrachealSuction"]
    uni = []
    for label in ("post_bps_ge7", "post_bps_gt3"):
        for col in ([f"F1_norm_{b}" for b in C.BANDS_BJA]
                    + [f"F2_norm_{b}" for b in C.BANDS_CANON]
                    + [f"F2_logabs_{b}" for b in C.BANDS_CANON]
                    + ["F2_spen", "F2_sef95", "pre_sed_mean", "pre_nol_mean",
                       "pre_bps", "pre_hf", "pre_map"]):
            dd = d[[col, label]].dropna()
            y = dd[label].astype(int).values
            x = dd[col].values
            if y.sum() < 5 or len(y) - y.sum() < 5:
                continue
            auc = rank_auc(x, y)
            u, p = mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
            uni.append(dict(label=label, feature=col, n=len(y),
                            auc_pos_high=auc, p_mw=float(p)))
    pd.DataFrame(uni).to_csv(C.RESULTS / "univariate.csv", index=False)

    # ---- pre-specified delta-AUC contrasts (per cohort/label family)
    pl = pd.DataFrame(preds)
    delta_rows = []
    for fam, meta in FAMILIES.items():
        task = "EndotrachealSuction"
        dd = fdf[fdf["task"] == task].dropna(subset=meta["require"])
        keep_ids = set(dd["participant_id"])
        sub = pl[(pl.cohort == task) & (pl.label == meta["label"])]
        base = {m: g.set_index("participant_id") for m, g in sub.groupby("model")}
        for mname in meta["models"]:
            if mname not in base:
                continue
            b = base[mname]
            b = b[b.index.isin(keep_ids)]
        common = None
        for mname in meta["models"]:
            if mname not in base:
                break
            idx = set(base[mname].index) & keep_ids
            common = idx if common is None else (common & idx)
        if not common:
            continue
        common = sorted(common)
        for cname, ma, mb, _fam in PRIMARY_CONTRASTS:
            if _fam != fam[0] or meta["label"] != ("post_bps_ge7" if fam.endswith("ge7") else "post_bps_gt3"):
                continue
            if ma not in base or mb not in base:
                continue
            A, B = base[ma].loc[common], base[mb].loc[common]
            y = A["y"].values
            d0, lo, hi, pv = boot_delta(A["p"].values, B["p"].values, y,
                                        np.array(common))
            delta_rows.append(dict(contrast=cname, label=meta["label"], n=len(common),
                                   delta_auc=d0, ci_lo=lo, ci_hi=hi, p_boot=pv))
            print(f"DELTA {cname} {meta['label']}: n={len(common)} "
                  f"dAUC={d0:.3f} [{lo:.3f},{hi:.3f}] p={pv:.4f}", flush=True)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(C.RESULTS / "delta_auc.csv", index=False)

    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
