"""Revision analyses (R1-R3), requested by author on 2026-11-23.

R1  Magnitude coding: within the suction BPS>=7 group (BPS 7-12), Spearman
    correlation (FDR-corrected, F1+F2 features) and single-predictor ordinal
    logistic regression (pre-specified BJA bands) of PRE-window EEG features
    against the continuous post-procedural BPS score.
R2  Selection bias: baseline comparison of the included suction cohort
    (n=102) against (a) QC-excluded suction recordings (n=33) and
    (b) all released recordings excluded from the primary cohort (n=78),
    on every baseline variable present in the public dataset.
    NOTE: SAPS II / APACHE II / sedative drug doses / ventilation duration
    are NOT part of ds008115 participants.tsv; only available variables used.
R3  Permutation test for the positioning->suction cross-task direction:
    permute TRAINING labels (positioning cohort) 1000x, refit, score the
    fixed suction test set; empirical one-sided p for observed AUROC
    (EEG_F2 and PSI, C fixed at the value used in 03_cross_task.py).

Outputs: results/revision_e1_magnitude.csv / revision_e1_ordinal.csv
         results/revision_e2_selection.csv
         results/revision_e3_permutation.csv / revision_e3_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

C = import_module("00_common")
X2 = import_module("02_run_experiments")

SEED = 42


# --------------------------------------------------------------------------- #
# R1: magnitude coding within BPS>=7
# --------------------------------------------------------------------------- #
def r1_magnitude(fdf: pd.DataFrame) -> None:
    d = fdf[(fdf["task"] == "EndotrachealSuction") & (fdf["post_bps"] >= 7)]
    feats = ([f"F1_norm_{b}" for b in C.BANDS_BJA]
             + [f"F2_norm_{b}" for b in C.BANDS_CANON]
             + [f"F2_logabs_{b}" for b in C.BANDS_CANON]
             + ["F2_spen", "F2_sef95"])
    rows = []
    for col in feats:
        dd = d[[col, "post_bps"]].dropna()
        rho, p = spearmanr(dd[col], dd["post_bps"])
        rows.append(dict(feature=col, n=len(dd), spearman_rho=float(rho), p=float(p)))
    res = pd.DataFrame(rows)
    res["p_fdr_bh"] = false_discovery_control(res["p"].values, method="bh")
    res.to_csv(C.RESULTS / "revision_e1_magnitude.csv", index=False)
    print(f"[R1] n={len(d)} with post_bps>={7} (values {sorted(d['post_bps'].dropna().unique())})")
    for _, r in res.iterrows():
        star = " *" if r.p_fdr_bh < 0.05 else ""
        print(f"  {r.feature:24s} rho={r.spearman_rho:+.3f}  p={r.p:.4f}  q={r.p_fdr_bh:.4f}{star}")

    # Ordinal logistic regression, pre-specified BJA bands only
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    orows = []
    for col in [f"F1_norm_{b}" for b in C.BANDS_BJA]:
        dd = d[[col, "post_bps"]].dropna()
        y = pd.Categorical(dd["post_bps"], categories=sorted(dd["post_bps"].unique()))
        ex = pd.DataFrame({col: dd[col].values}, index=dd.index)
        try:
            mod = OrderedModel(y, ex, distr="logit")
            out = mod.fit(method="bfgs", disp=False)
            b = float(out.params[col])
            p = float(out.pvalues[col])
            orows.append(dict(feature=col, n=len(dd), coef=b, se=float(out.bse[col]), p=p,
                              converged=True))
            print(f"  [ordinal] {col}: beta={b:+.3f} p={p:.4f}")
        except Exception as e:  # record failure, do not fabricate
            orows.append(dict(feature=col, n=len(dd), coef=np.nan, se=np.nan, p=np.nan,
                              converged=False, note=str(e)))
            print(f"  [ordinal] {col}: FAILED ({e})")
    pd.DataFrame(orows).to_csv(C.RESULTS / "revision_e1_ordinal.csv", index=False)


# --------------------------------------------------------------------------- #
# R2: selection bias
# --------------------------------------------------------------------------- #
CONT_VARS = ["Pre_HF", "Pre_SAP", "Pre_MAP", "Pre_NRS", "Pre_BehavioralPainScale",
             "Pre_SED_Mean", "Pre_NOL_Mean"]
CAT_VARS = ["AgeGroup", "intracranialLesion", "spinalLesion"]


def _smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def _compare_block(inc: pd.DataFrame, exc: pd.DataFrame, tag: str, rows: list) -> None:
    for v in CONT_VARS:
        a, b = inc[v], exc[v]
        a, b = a.dropna(), b.dropna()
        if len(a) < 3 or len(b) < 3:
            rows.append(dict(comparison=tag, variable=v, type="continuous",
                             n_included=len(a), n_excluded=len(b),
                             med_included=np.nan, iqr_included=np.nan,
                             med_excluded=np.nan, iqr_excluded=np.nan,
                             p=np.nan, smd=np.nan, note="insufficient data"))
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(comparison=tag, variable=v, type="continuous",
                         n_included=len(a), n_excluded=len(b),
                         med_included=float(a.median()), iqr_included=float(a.quantile(.75) - a.quantile(.25)),
                         med_excluded=float(b.median()), iqr_excluded=float(b.quantile(.75) - b.quantile(.25)),
                         p=float(p), smd=_smd(inc[v], exc[v]), note=""))
    for v in CAT_VARS:
        lv = sorted(set(inc[v].dropna().unique()) | set(exc[v].dropna().unique()), key=str)
        ti = inc[v].value_counts().reindex(lv, fill_value=0)
        te_ = exc[v].value_counts().reindex(lv, fill_value=0)
        tab = np.vstack([ti.values, te_.values])
        try:
            chi2, p, dof, exp = chi2_contingency(tab)
            note = "" if exp.min() >= 5 else "small expected counts"
            if len(lv) == 2:
                from scipy.stats import fisher_exact
                _, p = fisher_exact(tab)
                note += ";fisher_exact"
        except Exception as e:
            chi2, p, dof, note = np.nan, np.nan, np.nan, str(e)
        rows.append(dict(comparison=tag, variable=v, type="categorical",
                         n_included=int(inc[v].notna().sum()), n_excluded=int(exc[v].notna().sum()),
                         med_included=np.nan, iqr_included=np.nan,
                         med_excluded=np.nan, iqr_excluded=np.nan,
                         p=float(p), smd=np.nan,
                         note=f"chi2={chi2:.2f},dof={dof};{note}"
                              f";included={inc[v].value_counts(dropna=False).to_dict()};"
                              f"excluded={exc[v].value_counts(dropna=False).to_dict()}"))


def r2_selection() -> None:
    p = C.load_participants()
    inc = p[cohort_suction_pass(p)]
    exc_suction = p[(p["InterventionTask"] == "EndotrachealSuction")
                    & ~cohort_suction_pass(p)]
    exc_all = p[~cohort_suction_pass(p)]  # 78 = 33 suction-fail + 43 positioning + 2 other
    assert len(inc) == 102 and len(exc_suction) == 33 and len(exc_all) == 78, \
        f"counts: inc={len(inc)} excS={len(exc_suction)} excAll={len(exc_all)}"
    rows = []
    _compare_block(inc, exc_suction, "suction_qc_fail_vs_included", rows)
    _compare_block(inc, exc_all, "all_excluded_vs_included", rows)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "revision_e2_selection.csv", index=False)
    for tag, g in df.groupby("comparison"):
        print(f"[R2] {tag}")
        for _, r in g.iterrows():
            if r.type == "continuous":
                print(f"  {r.variable:24s} med={r.med_included:.1f}[IQR{r.iqr_included:.1f}] "
                      f"vs {r.med_excluded:.1f}[IQR{r.iqr_excluded:.1f}]  p={r.p:.3f} SMD={r.smd:+.2f}")
            else:
                print(f"  {r.variable:24s} p={r.p:.3f}  {r.note[:90]}")


def cohort_suction_pass(p: pd.DataFrame) -> pd.Series:
    keep = (p["InterventionTask"] == "EndotrachealSuction") \
        & (~p["EEGDATA_Review"].isin(C.BAD_QUALITY)) & (p["EEGDATA_Review"] != "")
    return keep


# --------------------------------------------------------------------------- #
# R3: permutation test, positioning -> suction
# --------------------------------------------------------------------------- #
def r3_permutation(fdf: pd.DataFrame, n_perm: int = 1000) -> None:
    dtr = fdf[fdf["task"] == "Positioning"]
    dte = fdf[fdf["task"] == "EndotrachealSuction"]
    label = "post_bps_ge7"
    specs = {"EEG_F2": X2.model_columns("EEG_F2", fdf.columns), "PSI": ["pre_sed_mean"]}
    fixed_c = {"EEG_F2": 1.0, "PSI": 1.0}  # C_used in 03_cross_task.py
    rows, summary = [], {}
    for mname, cols in specs.items():
        tr = dtr.dropna(subset=cols + [label])
        te = dte.dropna(subset=cols + [label])
        Xtr, ytr = tr[cols].values, tr[label].astype(int).values
        Xte, yte = te[cols].values, te[label].astype(int).values
        pipe = X2.fit_proba(Xtr, ytr, fixed_c[mname])
        auc_obs = X2.rank_auc(pipe.predict_proba(Xte)[:, 1], yte)
        rng = np.random.default_rng(SEED)
        null = np.empty(n_perm)
        for i in range(n_perm):
            yperm = rng.permutation(ytr)
            pm = X2.fit_proba(Xtr, yperm, fixed_c[mname])
            null[i] = X2.rank_auc(pm.predict_proba(Xte)[:, 1], yte)
        p_ge = (1 + int((null >= auc_obs).sum())) / (n_perm + 1)
        p_le = (1 + int((null <= auc_obs).sum())) / (n_perm + 1)
        rows += [dict(model=mname, auc_observed=auc_obs, perm_auc=null[i],
                      perm_index=i) for i in range(n_perm)]
        summary[mname] = dict(auc_observed=float(auc_obs), n_perm=n_perm,
                              null_mean=float(null.mean()), null_sd=float(null.std()),
                              null_q95=float(np.percentile(null, 95)),
                              null_q05=float(np.percentile(null, 5)),
                              null_max=float(null.max()), null_min=float(null.min()),
                              p_upper=p_ge, p_lower=p_le,
                              n_train=len(ytr), n_pos_train=int(ytr.sum()),
                              n_test=len(yte), n_pos_test=int(yte.sum()),
                              C_fixed=fixed_c[mname], seed=SEED)
        s = summary[mname]
        print(f"[R3] {mname}: obs={auc_obs:.3f} null={s['null_mean']:.3f}"
              f"+-{s['null_sd']:.3f} q05={s['null_q05']:.3f} q95={s['null_q95']:.3f} "
              f"p_upper={p_ge:.4f} p_lower={p_le:.4f}")
    pd.DataFrame(rows).to_csv(C.RESULTS / "revision_e3_permutation.csv", index=False)
    (C.RESULTS / "revision_e3_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    t0 = time.time()
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    r1_magnitude(fdf)
    r2_selection()
    r3_permutation(fdf, n_perm=1000)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
