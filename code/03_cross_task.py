"""Cross-task generalization (pre-specified C3): train suction -> test positioning.

Models trained on the full suction cohort (C selected by inner LOSO),
evaluated on the positioning cohort. Reverse direction exploratory.

Output: results/cross_task.csv + results/cross_task_preds.csv
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
X2 = import_module("02_run_experiments")  # reuse feature/model utils

MODELS_CT = [
    ("EEG_F2", "EEG_F2"),
    ("PSI", ["pre_sed_mean"]),
    ("NOL", ["pre_nol_mean"]),
    ("EEG_F2_PSI", ["EEG_F2", "pre_sed_mean"]),
    ("PSI_NOL", ["pre_sed_mean", "pre_nol_mean"]),
    ("EEG_F2_PSI_NOL", ["EEG_F2", "pre_sed_mean", "pre_nol_mean"]),
]


def main() -> None:
    t0 = time.time()
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    rows, preds = [], []
    for train_task, test_task in (("EndotrachealSuction", "Positioning"),
                                  ("Positioning", "EndotrachealSuction")):
        for label in ("post_bps_ge7", "post_bps_gt3"):
            dtr = fdf[fdf["task"] == train_task]
            dte = fdf[fdf["task"] == test_task]
            for mname, spec in MODELS_CT:
                cols = X2.model_columns(spec, fdf.columns)
                tr = dtr.dropna(subset=cols + [label])
                te = dte.dropna(subset=cols + [label])
                Xtr, ytr = tr[cols], tr[label].astype(int).values
                Xte, yte = te[cols], te[label].astype(int).values
                if ytr.sum() < 5 or yte.sum() < 5:
                    print(f"[skip] {train_task}->{test_task} {label} {mname} "
                          f"(pos train={ytr.sum()}, test={yte.sum()})")
                    continue
                _, ch = X2.nested_loso(Xtr, ytr, tr["participant_id"].values)
                c_med = float(np.median(ch))
                pipe = X2.fit_proba(Xtr, ytr, c_med)
                p = pipe.predict_proba(Xte)[:, 1]
                auc, lo, hi = X2.boot_ci(p, yte, te["participant_id"].values,
                                         n_boot=2000)
                rows.append(dict(train=train_task, test=test_task, label=label,
                                 model=mname, n_train=len(ytr),
                                 n_pos_train=int(ytr.sum()), n_test=len(yte),
                                 n_pos_test=int(yte.sum()), C_used=c_med,
                                 auroc=auc, ci_lo=lo, ci_hi=hi))
                for pid, yy, pp in zip(te["participant_id"], yte, p):
                    preds.append(dict(train=train_task, test=test_task,
                                      label=label, model=mname,
                                      participant_id=pid, y=int(yy), p=float(pp)))
                print(f"{train_task[:4]}->{test_task[:4]} {label} {mname}: "
                      f"n_test={len(yte)} pos={yte.sum()} "
                      f"AUC={auc:.3f} [{lo:.3f},{hi:.3f}] C={c_med}", flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS / "cross_task.csv", index=False)
    pd.DataFrame(preds).to_csv(C.RESULTS / "cross_task_preds.csv", index=False)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
