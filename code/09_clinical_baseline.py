"""P1: clinical baseline comparison (suction cohort).

Reviewers will ask whether pre-procedural EEG adds anything over what a
clinician already has at the bedside. Three clinical-only references are
evaluated and compared against the EEG models on matched complete-case
sets:

  CLIN_PREBPS = baseline BPS alone
  CLIN_VITALS = pre-systolic / MAP / heart rate
  CLIN_ALL    = all four clinical variables

Part A (full set, n up to 102): the three clinical-only models.
Part B (common complete-case set, requires pre_sed_mean + EEG_F1 + clinical):
  CLIN_ALL, EEG_F2, PSI, EEG_F1_PSI, EEG_F1_PSI+CLIN, EEG_F2+CLIN,
  with paired delta AUROC:
    EEG_F1_PSI - CLIN_ALL
    EEG_F1_PSI+CLIN - EEG_F1_PSI
    EEG_F2 - CLIN_ALL

Labels: post_bps_ge7, post_bps_gt3 (class guard >=5 per class).
Metrics: LOPO AUROC + bootstrap CI, AUPRC/Brier/calibration (07_metrics_p2).

Output: results/clinical_baseline.csv
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

CLIN_PREBPS = ["pre_bps"]
CLIN_VITALS = ["pre_sap", "pre_map", "pre_hf"]
CLIN_ALL = CLIN_PREBPS + CLIN_VITALS

CLIN_MODELS = [
    ("CLIN_PREBPS", CLIN_PREBPS),
    ("CLIN_VITALS", CLIN_VITALS),
    ("CLIN_ALL", CLIN_ALL),
]

FULL_MODELS = CLIN_MODELS

COMMON_MODELS = [
    ("CLIN_PREBPS", CLIN_PREBPS),
    ("CLIN_VITALS", CLIN_VITALS),
    ("CLIN_ALL", CLIN_ALL),
    ("EEG_F2", "EEG_F2"),
    ("PSI", ["pre_sed_mean"]),
    ("EEG_F1_PSI", ["EEG_F1", "pre_sed_mean"]),
    ("EEG_F1_PSI_CLIN", ["EEG_F1", "pre_sed_mean"] + CLIN_ALL),
    ("EEG_F2_CLIN", ["EEG_F2"] + CLIN_ALL),
]

DELTAS = [  # (name, model a, model b)
    ("DELTA(EEG_F1_PSI-CLIN_ALL)", "EEG_F1_PSI", "CLIN_ALL"),
    ("DELTA(EEG_F1_PSI-CLIN_VITALS)", "EEG_F1_PSI", "CLIN_VITALS"),
    ("DELTA(EEG_F1_PSI_CLIN-EEG_F1_PSI)", "EEG_F1_PSI_CLIN", "EEG_F1_PSI"),
    ("DELTA(EEG_F2-CLIN_ALL)", "EEG_F2", "CLIN_ALL"),
]


def run_models(d: pd.DataFrame, models, label: str, rows: list[dict],
               store: dict[str, np.ndarray], setname: str) -> None:
    for mname, spec in models:
        cols = X2.model_columns(spec, d.columns)
        dd = d.dropna(subset=cols)
        y = dd[label].astype(int).values
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            print(f"[skip] {mname}/{label}: class guard", flush=True)
            continue
        oof, chosen = X2.nested_loso(dd[cols], y, dd["participant_id"].values)
        auc, lo, hi = X2.boot_ci(oof, y, dd["participant_id"].values)
        row = dict(set=setname, label=label, model=mname, n=len(y),
                   pos=int(y.sum()), auroc=auc, ci_lo=lo, ci_hi=hi,
                   delta=np.nan, delta_lo=np.nan, delta_hi=np.nan,
                   delta_p=np.nan)
        row.update(P2.metrics_row(oof, y, dd["participant_id"].values))
        rows.append(row)
        store[mname] = (oof, dd)
        print(f"{label} {mname}: n={len(y)} pos={y.sum()} "
              f"AUC={auc:.3f} [{lo:.3f},{hi:.3f}] AP={row['auprc']:.3f} "
              f"Brier={row['brier']:.3f} slope={row['calib_slope']:.2f} "
              f"C~{np.median(chosen):.4g}", flush=True)


def main() -> None:
    t0 = time.time()
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    fdf = fdf[fdf["task"] == "EndotrachealSuction"]

    rows: list[dict] = []
    for label in ("post_bps_ge7", "post_bps_gt3"):
        d = fdf.dropna(subset=[label])

        # ---- Part A: clinical-only models on the full set
        store: dict[str, np.ndarray] = {}
        run_models(d, FULL_MODELS, label, rows, store, "full")

        # ---- Part B: common complete-case set
        common_cols: list[str] = list(CLIN_ALL)
        for _, spec in COMMON_MODELS:
            common_cols += X2.model_columns(spec, fdf.columns)
        common_cols = sorted(set(common_cols))
        sub = d.dropna(subset=common_cols)
        y_c = sub[label].astype(int).values
        if y_c.sum() < 5 or (len(y_c) - y_c.sum()) < 5:
            print(f"[skip] common/{label}: class guard", flush=True)
            continue

        cstore: dict[str, np.ndarray] = {}
        run_models(sub, COMMON_MODELS, label, rows, cstore, "common")

        for dname, ma, mb in DELTAS:
            if ma not in cstore or mb not in cstore:
                continue
            # refit both on identical rows (sub) so pairing is exact
            res = {}
            for m in (ma, mb):
                spec = dict(COMMON_MODELS)[m]
                cols = X2.model_columns(spec, sub.columns)
                oof, _ = X2.nested_loso(sub[cols], y_c,
                                        sub["participant_id"].values)
                res[m] = oof
            d0, lo0, hi0, pv = X2.boot_delta(
                res[ma], res[mb], y_c, sub["participant_id"].values)
            rows.append(dict(set="common", label=label, model=dname, n=len(y_c),
                             pos=int(y_c.sum()), auroc=d0,
                             ci_lo=lo0, ci_hi=hi0,
                             delta=d0, delta_lo=lo0, delta_hi=hi0, delta_p=pv,
                             auprc=np.nan, auprc_lo=np.nan, auprc_hi=np.nan,
                             brier=np.nan, brier_lo=np.nan, brier_hi=np.nan,
                             calib_slope=np.nan))
            print(f"{label} {dname}={d0:+.3f} [{lo0:+.3f},{hi0:+.3f}] "
                  f"p={pv:.3f} (n={len(y_c)})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "clinical_baseline.csv", index=False)
    print(f"wrote {C.RESULTS / 'clinical_baseline.csv'} "
          f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
