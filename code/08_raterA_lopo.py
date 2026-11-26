"""P3: rater-A-only LOPO sensitivity analysis (suction cohort).

Rationale: rater identity (Investigator A/B) may shape the BPS labels.
Here every model is re-run restricted to the single-rater (A) subset so
that label definitions are internally consistent, asking whether the
headline EEG effect survives within one rater's ratings.

Models  : EEG_F2, PSI, EEG_F1_PSI (EEG_F1 + pre-sedation PSI)
Labels  : post_bps_ge7, post_bps_gt3 (guarded, >=5 per class)
Metrics : LOPO AUROC + patient bootstrap CI; AUPRC / Brier / calibration
          slope (reused from 07_metrics_p2); paired delta AUROC.

Output: results/raterA_lopo.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

C = import_module("00_common")
X2 = import_module("02_run_experiments")
P2 = import_module("07_metrics_p2")

PARTICIPANTS = C.DATA / "participants.tsv"

MODELS = [
    ("EEG_F2", "EEG_F2"),
    ("PSI", ["pre_sed_mean"]),
    ("EEG_F1_PSI", ["EEG_F1", "pre_sed_mean"]),
]


def main() -> None:
    t0 = time.time()
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    part = pd.read_csv(PARTICIPANTS, sep="\t")
    fdf = fdf.merge(
        part[["participant_id", "Investigator"]], on="participant_id", how="left"
    )

    rows: list[dict] = []
    for label in ("post_bps_ge7", "post_bps_gt3"):
        d = fdf[(fdf["task"] == "EndotrachealSuction")
                & (fdf["Investigator"] == "A")].dropna(subset=[label])

        # models that share the same complete-case set for fair pairing
        specs = dict(MODELS)
        common_cols: list[str] = []
        for _, spec in specs.items():
            common_cols += X2.model_columns(spec, fdf.columns)
        common_cols = sorted(set(common_cols))
        sub = d.dropna(subset=common_cols)
        y_c = sub[label].astype(int).values
        if y_c.sum() < 5 or (len(y_c) - y_c.sum()) < 5:
            print(f"[skip] raterA/{label}: too few positives in common set",
                  flush=True)
            continue

        oofs: dict[str, np.ndarray] = {}
        for mname, spec in MODELS:
            cols = X2.model_columns(spec, fdf.columns)
            dd = d.dropna(subset=cols)
            y = dd[label].astype(int).values
            if y.sum() < 5 or (len(y) - y.sum()) < 5:
                print(f"[skip] raterA/{label}/{mname}: class guard", flush=True)
                continue
            oof, chosen = X2.nested_loso(dd[cols], y, dd["participant_id"].values)
            auc, lo, hi = X2.boot_ci(oof, y, dd["participant_id"].values)
            row = dict(label=label, model=mname, n=len(y), pos=int(y.sum()),
                       auroc=auc, ci_lo=lo, ci_hi=hi,
                       delta=np.nan, delta_lo=np.nan, delta_hi=np.nan,
                       delta_p=np.nan)
            row.update(P2.metrics_row(oof, y, dd["participant_id"].values))
            rows.append(row)
            oofs[mname] = oof
            print(f"raterA {label} {mname}: n={len(y)} pos={y.sum()} "
                  f"AUC={auc:.3f} [{lo:.3f},{hi:.3f}] "
                  f"AP={row['auprc']:.3f} Brier={row['brier']:.3f} "
                  f"slope={row['calib_slope']:.2f} C~{np.median(chosen):.4g}",
                  flush=True)

        # paired delta EEG_F1_PSI - PSI on the common complete-case set
        if all(m in oofs for m in ("EEG_F1_PSI", "PSI")):
            # refit both models on the common set so pairing is exact
            res: dict[str, np.ndarray] = {}
            for mname in ("EEG_F1_PSI", "PSI"):
                cols = X2.model_columns(specs[mname], fdf.columns)
                oof, _ = X2.nested_loso(sub[cols], y_c,
                                        sub["participant_id"].values)
                res[mname] = oof
            d0, lo0, hi0, pv = X2.boot_delta(
                res["EEG_F1_PSI"], res["PSI"], y_c, sub["participant_id"].values)
            rows.append(dict(label=label,
                             model="DELTA(EEG_F1_PSI-PSI)",
                             n=len(y_c), pos=int(y_c.sum()),
                             auroc=d0, ci_lo=lo0, ci_hi=hi0,
                             delta=d0, delta_lo=lo0, delta_hi=hi0, delta_p=pv,
                             auprc=np.nan, auprc_lo=np.nan, auprc_hi=np.nan,
                             brier=np.nan, brier_lo=np.nan, brier_hi=np.nan,
                             calib_slope=np.nan))
            print(f"raterA {label} delta(EEG_F1_PSI-PSI)={d0:+.3f} "
                  f"[{lo0:+.3f},{hi0:+.3f}] p={pv:.3f} (n={len(y_c)})",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "raterA_lopo.csv", index=False)
    print(f"wrote {C.RESULTS / 'raterA_lopo.csv'} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
