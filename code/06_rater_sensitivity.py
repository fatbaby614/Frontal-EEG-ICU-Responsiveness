"""Rater (Investigator) sensitivity analysis for the suction cohort.

Rationale: the released participants.tsv carries an `Investigator` column
(A/B, the person assessing and recording BPS). If rater identity predicts the
label, the EEG models must be checked against the confound of rater
allocation.

Outputs:
- results/rater_sensitivity.csv   (single-variable + LOPO model results)
- stdout summary (MWU / Fisher tests)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

C = import_module("00_common")
X2 = import_module("02_run_experiments")

PARTICIPANTS = C.DATA / "participants.tsv"


def main() -> None:
    t0 = time.time()
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    part = pd.read_csv(PARTICIPANTS, sep="\t")
    fdf = fdf.merge(
        part[["participant_id", "Investigator"]], on="participant_id", how="left"
    )

    rows: list[dict] = []
    for label in ("post_bps_ge7", "post_bps_gt3"):
        d = fdf[fdf["task"] == "EndotrachealSuction"].dropna(
            subset=[label, "Investigator"]
        )
        d = d.assign(rater=(d["Investigator"] == "A").astype(float))
        y = d[label].astype(int).values
        n_a = int((d["Investigator"] == "A").sum())
        n_b = int((d["Investigator"] == "B").sum())

        # --- descriptive: label distribution by rater
        a_bps = d.loc[d["Investigator"] == "A", "post_bps"].dropna()
        b_bps = d.loc[d["Investigator"] == "B", "post_bps"].dropna()
        _, p_mwu = mannwhitneyu(a_bps, b_bps)
        ct = [
            [int(((d["Investigator"] == "A") & (d[label] == 1)).sum()),
             int(((d["Investigator"] == "A") & (d[label] == 0)).sum())],
            [int(((d["Investigator"] == "B") & (d[label] == 1)).sum()),
             int(((d["Investigator"] == "B") & (d[label] == 0)).sum())],
        ]
        _, p_fisher = fisher_exact(ct)
        rate_a = ct[0][0] / max(1, ct[0][0] + ct[0][1])
        rate_b = ct[1][0] / max(1, ct[1][0] + ct[1][1])
        print(f"[{label}] rater A/B n={n_a}/{n_b}; positive rate A={rate_a:.3f} "
              f"B={rate_b:.3f}; Fisher p={p_fisher:.4g}; "
              f"BPS MWU p={p_mwu:.4g}", flush=True)

        # --- single-variable rater AUROC
        auc_r = X2.rank_auc(d["rater"].values, y)
        rows.append(dict(label=label, model="RATER", n=len(y),
                         pos=int(y.sum()), auroc=auc_r,
                         ci_lo=np.nan, ci_hi=np.nan, delta_vs_f2=np.nan,
                         delta_lo=np.nan, delta_hi=np.nan, delta_p=np.nan))
        print(f"[{label}] single rater AUROC={auc_r:.3f}", flush=True)

        # --- LOPO: EEG_F2 vs EEG_F2 + rater
        results: dict[str, np.ndarray] = {}
        for mname, extra in (("EEG_F2", False), ("EEG_F2_RATER", True)):
            cols = X2.model_columns("EEG_F2", fdf.columns)
            sub = d.dropna(subset=cols + [label])
            Xs, ys = sub[cols], sub[label].astype(int).values
            Xs = Xs.copy()
            if extra:
                Xs["rater_A"] = (sub["Investigator"] == "A").astype(float).values
            oof, chosen = X2.nested_loso(Xs, ys, sub["participant_id"].values)
            auc, lo, hi = X2.boot_ci(oof, ys, sub["participant_id"].values)
            results[mname] = oof
            rows.append(dict(label=label, model=mname, n=len(ys),
                             pos=int(ys.sum()), auroc=auc, ci_lo=lo, ci_hi=hi,
                             delta_vs_f2=np.nan, delta_lo=np.nan,
                             delta_hi=np.nan, delta_p=np.nan))
            print(f"[{label}] {mname}: n={len(ys)} pos={ys.sum()} "
                  f"AUC={auc:.3f} [{lo:.3f},{hi:.3f}] "
                  f"C~{np.median(chosen):.4g}", flush=True)

        # paired delta (EEG_F2_RATER vs EEG_F2) on the same complete-case set
        cols = X2.model_columns("EEG_F2", fdf.columns)
        sub = d.dropna(subset=cols + [label])
        ys = sub[label].astype(int).values
        d0, lo0, hi0, pv = X2.boot_delta(
            results["EEG_F2_RATER"], results["EEG_F2"], ys,
            sub["participant_id"].values,
        )
        rows.append(dict(label=label, model="DELTA(EEG_F2_RATER-EEG_F2)",
                         n=len(ys), pos=int(ys.sum()), auroc=d0,
                         ci_lo=lo0, ci_hi=hi0, delta_vs_f2=d0,
                         delta_lo=lo0, delta_hi=hi0, delta_p=pv))
        print(f"[{label}] delta AUC(+rater)={d0:+.3f} [{lo0:+.3f},{hi0:+.3f}] "
              f"p={pv:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "rater_sensitivity.csv", index=False)
    print(f"wrote {C.RESULTS / 'rater_sensitivity.csv'} "
          f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
