"""Feature extraction + W1 replication (BJA 2023 directionality) for ds008115.

Outputs
  results/features.csv          one row per subject, all feature sets
  results/features_cache/*.npz  per-subject band-power series (for figures)
  results/w1_replication.json   BJA-2023 directionality replication stats
  results/cohort_flow.json      cohort selection counts
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

C = import_module("00_common")

# --------------------------------------------------------------------------- #
# Riemannian tangent-space features (minimal, eigh-based)
# --------------------------------------------------------------------------- #
def tangent_features(covs: np.ndarray, sqrt2: bool = True) -> np.ndarray:
    """Affine-invariant tangent vectors at the Riemannian mean.

    covs: (n, d, d). Returns (n, d*(d+1)/2) with off-diagonals scaled by sqrt(2)
    (pyriemann convention).
    """
    from pyriemann.utils.mean import mean_covariance
    Cref = mean_covariance(covs, metric="riemann")
    # whitening
    lam, V = np.linalg.eigh(Cref)
    Wm = (V * (1.0 / np.sqrt(lam))) @ V.T          # Cref^{-1/2}
    n, d, _ = covs.shape
    k = d * (d + 1) // 2
    T = np.empty((n, k))
    iu = np.triu_indices(d)
    for i in range(n):
        S = Wm @ covs[i] @ Wm.T
        s_lam, U = np.linalg.eigh(S)
        L = (U * np.log(np.clip(s_lam, 1e-12, None))) @ U.T
        t = L[iu]
        if sqrt2:
            off = ~np.eye(d, dtype=bool)[iu]
            t[off] *= np.sqrt(2.0)
        T[i] = t
    return T


def band_covariance(data: np.ndarray, sfreq: float, starts: list[int],
                    band: tuple[float, float]) -> np.ndarray:
    """Per-segment (4,4) shrinkage covariances of band-filtered signal."""
    from mne.filter import filter_data
    from pyriemann.estimation import Covariances
    seg = int(round(C.SEG_LEN_S * sfreq))
    xf = filter_data(data, sfreq, band[0], band[1], verbose=False)
    X = np.stack([xf[:, s:s + seg] for s in starts])       # (n, 4, seg)
    return Covariances(estimator="lwf").transform(X)


# --------------------------------------------------------------------------- #
# Per-subject feature computation
# --------------------------------------------------------------------------- #
def compute_subject(row) -> dict | None:
    sub = row["participant_id"]
    task = C.task_of(row)
    try:
        data, sfreq = C.load_eeg(sub, task)
        onset = C.read_onset(sub, task)
    except Exception as e:
        print(f"  [FAIL] {sub}: {e}")
        return None
    ch_best, ch_src = C.best_channel(sub, task, row)
    ci = C.CHANNELS.index(ch_best)
    if abs(sfreq - C.SFREQ) > 0.01:
        print(f"  [WARN] {sub}: sfreq {sfreq}")

    out = {"participant_id": sub, "task": task, "best_channel": ch_best,
           "best_channel_source": ch_src, "onset_s": onset, "sfreq": sfreq}

    psd_pre, freqs = None, None
    series = {}
    for win_name, win in (("pre", C.PRE_WIN), ("post", C.POST_WIN)):
        starts = C.segment_indices(data.shape[1], sfreq, win[0], win[1], onset)
        out[f"n_seg_{win_name}"] = len(starts)
        if len(starts) < 20:
            print(f"  [WARN] {sub}: only {len(starts)} segments in {win_name}")
        psd, freqs = C.segment_psd(data, sfreq, starts)
        tot = psd.sum(axis=-1, keepdims=True)          # total power 0.1-41 Hz
        norm = psd / tot                               # normalized PSD per segment/ch
        cb = psd[:, ci, :]                             # best channel
        cn = norm[:, ci, :]
        if win_name == "pre":
            psd_pre = psd
            for bname, bnd in C.BANDS_CANON.items():
                series[f"norm_{bname}"] = C.band_mean(freqs, cn, bnd)
                series[f"logabs_{bname}"] = np.log10(
                    np.clip(C.band_mean(freqs, cb, bnd), 1e-30, None))
            for bname, bnd in C.BANDS_BJA.items():
                series[f"normbj_{bname}"] = C.band_mean(freqs, cn, bnd)
            series["spen"] = C.spectral_entropy(freqs, cb)
            series["sef95"] = C.sef(freqs, cb, 0.95)
        else:
            for bname, bnd in C.BANDS_BJA.items():
                series[f"postnorm_{bname}"] = C.band_mean(freqs, cn, bnd)
                out[f"F2_postnorm_{bname}"] = float(
                    C.band_mean(freqs, cn, bnd).mean())

    # --- F1: BJA replication features (normalized band power, pre, best ch)
    for bname, bnd in C.BANDS_BJA.items():
        out[f"F1_norm_{bname}"] = float(np.mean(series[f"normbj_{bname}"]))

    # --- F2: extended bandpower (best channel, pre window)
    for bname in C.BANDS_CANON:
        out[f"F2_norm_{bname}"] = float(np.mean(series[f"norm_{bname}"]))
        out[f"F2_logabs_{bname}"] = float(np.mean(series[f"logabs_{bname}"]))
    out["F2_spen"] = float(np.mean(series["spen"]))
    out["F2_sef95"] = float(np.nanmedian(series["sef95"]))

    # --- F3: time-frequency dynamics (normalized band power series, best ch)
    t = np.arange(len(series["norm_" + next(iter(C.BANDS_CANON))]))
    for bname in C.BANDS_CANON:
        s = series[f"norm_{bname}"]
        out[f"F3_dyn_{bname}_mean"] = float(np.mean(s))
        out[f"F3_dyn_{bname}_sd"] = float(np.std(s, ddof=1))
        slope = np.polyfit(t, s, 1)[0] if len(t) > 2 else np.nan
        out[f"F3_dyn_{bname}_slope"] = float(slope)

    # --- F4: F3 + Riemannian spatial covariance features
    starts_pre = C.segment_indices(data.shape[1], sfreq, *C.PRE_WIN, onset)
    for bname, bnd in C.BANDS_RIEM.items():
        try:
            covs = band_covariance(data, sfreq, starts_pre, bnd)
            T = tangent_features(covs)
            mu = T.mean(axis=0)
            for k in range(T.shape[1]):
                out[f"F4_riem_{bname}_{k}"] = float(mu[k])
        except Exception as e:
            print(f"  [WARN] {sub}: riemann {bname} failed: {e}")
            for k in range(10):
                out[f"F4_riem_{bname}_{k}"] = np.nan

    # cache series for figures
    np.savez_compressed(
        C.CACHE / f"{sub}.npz",
        freqs=freqs, onset=onset, sfreq=sfreq,
        **{k: np.asarray(v, dtype=np.float64) for k, v in series.items()},
    )
    return out


# --------------------------------------------------------------------------- #
# W1 replication statistics
# --------------------------------------------------------------------------- #
def w1_replication(df: pd.DataFrame) -> dict:
    """Mann-Whitney comparison of BJA band features, strong responders vs not.

    Mirrors Bublitz et al. 2023: suction cohort, BPS>=7 defines strong
    responsiveness; normalized band power in pre/post windows.
    """
    sub = df[df["task"] == "EndotrachealSuction"].copy()
    res = {}
    for label_name, col in (("BPS>=7", "post_bps_ge7"), ("BPS>3", "post_bps_gt3")):
        mask_pos = sub[col].astype(bool)
        pos = sub[mask_pos]
        neg = sub[~mask_pos]
        r = {"n_pos": int(len(pos)), "n_neg": int(len(neg))}
        feats_test = (["F1_norm_slowdelta", "F1_norm_fastdelta", "F1_norm_alpha",
                       "F2_postnorm_slowdelta", "F2_postnorm_fastdelta",
                       "F2_postnorm_alpha"])
        for feat in feats_test:
            if feat not in sub.columns:
                continue
            a, b = pos[feat].dropna(), neg[feat].dropna()
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            auc = u / (len(a) * len(b))  # rank AUROC (P(X_pos > X_neg))
            med_a, med_b = float(np.median(a)), float(np.median(b))
            r[feat] = {
                "median_pos": med_a, "q1q3_pos": [float(np.percentile(a, 25)),
                                                  float(np.percentile(a, 75))],
                "median_neg": med_b, "q1q3_neg": [float(np.percentile(b, 25)),
                                                  float(np.percentile(b, 75))],
                "pct_diff_median": 100 * (med_a - med_b) / abs(med_b) if med_b != 0 else None,
                "auc_rank": float(auc), "p_mw": float(p),
            }
        res[label_name] = r
    return res


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    parts = C.load_participants()

    rows = []
    flow = {}
    for task in ("EndotrachealSuction", "Positioning", "Other"):
        sel = parts[parts["InterventionTask"] == task]
        kept = parts[C.cohort_mask(parts, task)]
        flow[task] = {"released": int(len(sel)),
                      "quality_pass": int(len(kept))}
    print("Cohort flow:", json.dumps(flow))

    out_csv = C.RESULTS / "features.csv"
    if out_csv.exists():
        print("features.csv exists -> skip extraction, run W1 only")
        fdf = pd.read_csv(out_csv)
    else:
        cohort = parts[C.cohort_mask(parts, "EndotrachealSuction")
                       | C.cohort_mask(parts, "Positioning")]
        print(f"Processing {len(cohort)} subjects ...")
        for i, (_, row) in enumerate(cohort.iterrows(), 1):
            sub = row["participant_id"]
            print(f"[{i}/{len(cohort)}] {sub} ({C.task_of(row)})", flush=True)
            feats = compute_subject(row)
            if feats is None:
                continue
            feats.update({
                "post_bps": row["Post_BehavioralPainScale"],
                "pre_bps": row["Pre_BehavioralPainScale"],
                "post_bps_ge7": int(row["Post_BehavioralPainScale"] >= 7),
                "post_bps_gt3": int(row["Post_BehavioralPainScale"] > 3),
                "pre_sed_mean": row["Pre_SED_Mean"],
                "pre_nol_mean": row["Pre_NOL_Mean"],
                "pre_hf": row["Pre_HF"], "pre_map": row["Pre_MAP"], "pre_sap": row["Pre_SAP"],
                "agegroup": row["AgeGroup"],
                "intracranialLesion": row["intracranialLesion"],
                "spinalLesion": row["spinalLesion"],
            })
            rows.append(feats)

        fdf = pd.DataFrame(rows)
        fdf.to_csv(out_csv, index=False)
        print(f"features.csv: {fdf.shape[0]} rows x {fdf.shape[1]} cols")

    w1 = w1_replication(fdf)
    (C.RESULTS / "w1_replication.json").write_text(
        json.dumps(w1, indent=2, ensure_ascii=False))
    print(json.dumps(w1, indent=2)[:1200])

    (C.RESULTS / "cohort_flow.json").write_text(json.dumps(flow, indent=2))
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
