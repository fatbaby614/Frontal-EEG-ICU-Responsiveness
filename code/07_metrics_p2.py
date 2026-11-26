"""P2: discrimination beyond AUROC + calibration, from existing OOF predictions.

Reads results/preds_long.csv (no retraining). For every (cohort, label, model):
- AUPRC (average precision) with patient-level bootstrap 95% CI
- Brier score with patient-level bootstrap 95% CI
- calibration slope (logistic calibration: y ~ logit(p))

Output: results/discrimination_calibration.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

C = import_module("00_common")

N_BOOT = 2000
RNG = 42


def calib_slope(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    logit = np.log(pc / (1 - pc))
    try:
        res = sm.Logit(y.astype(float), sm.add_constant(logit)).fit(disp=0)
        return float(res.params[1])
    except Exception:
        return float("nan")


def boot_metric(p, y, ids, fn, n_boot=N_BOOT, seed=RNG):
    """Patient-level bootstrap percentile CI for metric fn(p, y)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    ids = np.asarray(ids)
    rng = np.random.default_rng(seed)
    uniq = np.unique(ids)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(ids == u)[0] for u in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        v = fn(p[idx], yy)
        if not np.isnan(v):
            vals.append(v)
    vals = np.sort(np.asarray(vals))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics_row(p, y, ids, prefix="") -> dict:
    ap = float(average_precision_score(y, p))
    br = float(brier_score_loss(y, p))
    ap_lo, ap_hi = boot_metric(
        p, y, ids, lambda a, b: float(average_precision_score(b, a)))
    b_lo, b_hi = boot_metric(
        p, y, ids, lambda a, b: float(brier_score_loss(b, a)))
    return dict(auprc=ap, auprc_lo=ap_lo, auprc_hi=ap_hi,
                brier=br, brier_lo=b_lo, brier_hi=b_hi,
                calib_slope=calib_slope(p, y))


def main() -> None:
    t0 = time.time()
    pl = pd.read_csv(C.RESULTS / "preds_long.csv")
    rows = []
    for (cohort, label, model), g in pl.groupby(["cohort", "label", "model"]):
        p, y = g["p"].values, g["y"].values
        ids = g["participant_id"].values
        r = dict(cohort=cohort, label=label, model=model, n=len(y),
                 pos=int(y.sum()))
        r.update(metrics_row(p, y, ids))
        rows.append(r)
        print(f"{cohort} {label} {model}: AP={r['auprc']:.3f} "
              f"[{r['auprc_lo']:.3f},{r['auprc_hi']:.3f}] "
              f"Brier={r['brier']:.3f} slope={r['calib_slope']:.2f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "discrimination_calibration.csv", index=False)
    print(f"wrote {C.RESULTS / 'discrimination_calibration.csv'} "
          f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
