"""
Forward-direction (suction -> positioning) label permutation test.

Mirror of r3_permutation in 05_revision_analyses.py, which covers only the
reverse direction (positioning -> suction). Same protocol: labels of the
training set are permuted n_perm times, the model is refitted with the
regularisation constant fixed at the value used for the observed estimate
(C_used from 03_cross_task.py = 1.0 for EEG_F2 and PSI), and the fixed test
set is scored, giving empirical one-sided P values.

Outputs: results/revision_e3_permutation_forward.csv / .json
Run from anywhere (paths resolved via 00_common.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

C = import_module("00_common")
X2 = import_module("02_run_experiments")

SEED = 42
N_PERM = 1000


def forward_permutation(fdf: pd.DataFrame, n_perm: int = N_PERM) -> None:
    dtr = fdf[fdf["task"] == "EndotrachealSuction"]
    dte = fdf[fdf["task"] == "Positioning"]
    label = "post_bps_ge7"
    specs = {"EEG_F2": X2.model_columns("EEG_F2", fdf.columns), "PSI": ["pre_sed_mean"]}
    fixed_c = {"EEG_F2": 1.0, "PSI": 1.0}  # C_used in 03_cross_task.py (both directions)
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
        rows += [dict(model=mname, auc_observed=float(auc_obs), perm_auc=float(null[i]))
                 for i in range(n_perm)]
        summary[mname] = dict(
            auc_observed=float(auc_obs), C_used=fixed_c[mname], n_perm=n_perm,
            null_mean=float(null.mean()), null_sd=float(null.std()),
            null_q95=float(np.percentile(null, 95)), null_q05=float(np.percentile(null, 5)),
            null_max=float(null.max()), null_min=float(null.min()),
            p_upper=p_ge, p_lower=p_le,
            n_train=len(ytr), n_pos_train=int(ytr.sum()),
            n_test=len(yte), n_pos_test=int(yte.sum()),
        )
        s = summary[mname]
        print(f"[forward] {mname}: auc_obs={auc_obs:.3f} "
              f"null mean={s['null_mean']:.3f} sd={s['null_sd']:.3f} "
              f"q95={s['null_q95']:.3f} max={s['null_max']:.3f} "
              f"p_upper={p_ge:.4f} p_lower={p_le:.4f}")
    pd.DataFrame(rows).to_csv(C.RESULTS / "revision_e3_permutation_forward.csv", index=False)
    with open(C.RESULTS / "revision_e3_permutation_forward.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {C.RESULTS / 'revision_e3_permutation_forward.csv'} / .json")


if __name__ == "__main__":
    fdf = pd.read_csv(C.RESULTS / "features.csv")
    forward_permutation(fdf)
