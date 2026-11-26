"""Shared utilities for the ds008115 pre-intervention EEG prediction analysis.

Analysis decisions are recorded in results/decision_log.md.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parents[1]  # project root (parent of code/)
DATA = ROOT / "data" / "ds008115"
RESULTS = ROOT / "results"
CACHE = RESULTS / "features_cache"
FIGURES = ROOT / "paper" / "JournalofClinicalMonitoringAndComputing" / "figures"

for _d in (DATA, RESULTS, CACHE, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SFREQ = 178.1538462
CHANNELS = ["Fp1", "Fp2", "F7", "F8"]
ONSET_DEFAULT = 240.0  # s; intervention onset in the recording (events.tsv)

# Windowing aligned with Bublitz et al. 2023 (BJA): 2 min pre + 4 min post onset
PRE_WIN = (-120.0, 0.0)
POST_WIN = (0.0, 240.0)

# Multi-taper PSD parameters aligned with BJA 2023
SEG_LEN_S = 3.0
SEG_STEP_S = 1.0
FMIN, FMAX = 0.1, 41.0
TIME_BANDWIDTH = 12.0  # NW = 6 -> 11 good Slepian tapers (floor(tbw - 1) = 11)

# Frequency bands
BANDS_BJA = {  # bands analysed in Bublitz 2023
    "slowdelta": (0.1, 1.5),
    "fastdelta": (2.5, 5.0),
    "alpha": (8.0, 12.0),
}
BANDS_CANON = {  # extended canonical set (contiguous)
    "b0.1-1.5": (0.1, 1.5),
    "b1.5-4": (1.5, 4.0),
    "b4-8": (4.0, 8.0),
    "b8-12": (8.0, 12.0),
    "b12-20": (12.0, 20.0),
    "b20-30": (20.0, 30.0),
    "b30-41": (30.0, 41.0),
}
BANDS_RIEM = {  # wider bands for spatial covariance structure
    "r0.1-4": (0.1, 4.0),
    "r4-8": (4.0, 8.0),
    "r8-16": (8.0, 16.0),
    "r16-30": (16.0, 30.0),
    "r30-41": (30.0, 41.0),
}

BAD_QUALITY = {"Artifacts", "burstSup", "no", "ECG"}  # blank (unreviewed) also excluded


# --------------------------------------------------------------------------- #
# Cohort
# --------------------------------------------------------------------------- #
def load_participants() -> pd.DataFrame:
    df = pd.read_csv(DATA / "participants.tsv", sep="\t")
    df["EEGDATA_Review"] = df["EEGDATA_Review"].fillna("")
    return df


def cohort_mask(df: pd.DataFrame, task: str) -> pd.Series:
    """Prespecified inclusion: task cohort + usable EEG quality."""
    keep = (df["InterventionTask"] == task) & (~df["EEGDATA_Review"].isin(BAD_QUALITY))
    keep &= df["EEGDATA_Review"] != ""  # exclude the single unreviewed recording
    return keep


def get_cohort(task: str) -> pd.DataFrame:
    df = load_participants()
    return df[cohort_mask(df, task)].reset_index(drop=True)


def subject_paths(sub: str, task: str) -> dict:
    base = DATA / sub / "eeg"
    stem = f"{sub}_task-{task}"
    return {
        "edf": base / f"{stem}_eeg.edf",
        "events": base / f"{stem}_events.tsv",
        "channels": base / f"{stem}_channels.tsv",
    }


def task_of(row) -> str:
    return str(row["InterventionTask"])


# --------------------------------------------------------------------------- #
# Channel selection
# --------------------------------------------------------------------------- #
def best_channel(sub: str, task: str, participants_row=None) -> tuple[str, str]:
    """Return (channel_name, source). BIDS channels.tsv 'best' flag preferred;
    participants.tsv integer Channel column as fallback/cross-check."""
    p = subject_paths(sub, task)
    try:
        ch = pd.read_csv(p["channels"], sep="\t")
        sel = ch.loc[ch["quality_selection"] == "best", "name"].tolist()
        if len(sel) == 1:
            return sel[0], "channels.tsv"
    except Exception:
        pass
    if participants_row is not None and not pd.isna(participants_row.get("Channel")):
        idx = int(participants_row["Channel"]) - 1
        return CHANNELS[idx], "participants.tsv"
    return "Fp2", "default"


# --------------------------------------------------------------------------- #
# Signal handling
# --------------------------------------------------------------------------- #
def load_eeg(sub: str, task: str) -> tuple[np.ndarray, float]:
    """Return (data [4, N] in V, sfreq)."""
    p = subject_paths(sub, task)
    raw = mne.io.read_raw_edf(p["edf"], preload=True)
    data = raw.get_data()  # (n_ch, n_times) in volts
    return data, float(raw.info["sfreq"])


def read_onset(sub: str, task: str) -> float:
    p = subject_paths(sub, task)
    ev = pd.read_csv(p["events"], sep="\t")
    return float(ev["onset"].iloc[0])


def segment_indices(n_times: int, sfreq: float, t0: float, t1: float,
                    onset: float) -> list[int]:
    """Start-sample indices of 3-s segments stepped by 1 s inside [t0, t1]+onset."""
    seg = int(round(SEG_LEN_S * sfreq))
    step = int(round(SEG_STEP_S * sfreq))
    s0 = int(round((onset + t0) * sfreq))
    s1 = int(round((onset + t1) * sfreq))
    idx = list(range(s0, s1 - seg + 1, step))
    return [i for i in idx if i >= 0 and i + seg <= n_times]


def segment_psd(data: np.ndarray, sfreq: float, starts: list[int]) -> np.ndarray:
    """Multi-taper PSD for all segments and channels.

    data: (4, N) volts. Returns (n_seg, n_ch, n_freq) power in V^2/Hz.
    Each segment is mean-detrended before the PSD.
    """
    from mne.time_frequency import psd_array_multitaper
    seg = int(round(SEG_LEN_S * sfreq))
    X = np.stack([data[:, s:s + seg] - data[:, s:s + seg].mean(axis=1, keepdims=True)
                  for s in starts])  # (n_seg, 4, seg)
    n_seg, _, _ = X.shape
    Xf = X.reshape(n_seg * 4, seg)
    # bandwidth = 2*NW/T with NW=6, T=3 s  ->  4.0 Hz, 11 good Slepian tapers
    psds, freqs = psd_array_multitaper(
        Xf, sfreq, fmin=FMIN, fmax=FMAX, normalization="length",
        bandwidth=2 * 6.0 / SEG_LEN_S, verbose=False,
    )
    psds = psds.reshape(n_seg, 4, len(freqs))
    return psds.astype(np.float64), freqs


def band_mean(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Mean power inside a band; psd shape (..., n_freq)."""
    m = (freqs >= band[0]) & (freqs <= band[1])
    return psd[..., m].mean(axis=-1)


def spectral_entropy(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Normalized spectral entropy per spectrum (dims before n_freq)."""
    p = psd / psd.sum(axis=-1, keepdims=True)
    p = np.clip(p, 1e-12, None)
    H = -(p * np.log(p)).sum(axis=-1)
    return H / np.log(len(freqs))


def sef(freqs: np.ndarray, psd: np.ndarray, percentile: float = 0.95) -> np.ndarray:
    """Spectral edge frequency: freq below which `percentile` of power lies."""
    csum = np.cumsum(psd, axis=-1)
    tot = csum[..., -1:]
    tot[tot == 0] = np.nan
    frac = csum / tot
    out = np.full(psd.shape[:-1], np.nan)
    for i in range(psd.shape[-1]):
        hit = (frac[..., i] >= percentile) & np.isnan(out)
        out[hit] = freqs[i]
    return out
