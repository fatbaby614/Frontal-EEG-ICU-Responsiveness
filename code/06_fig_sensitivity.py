# -*- coding: utf-8 -*-
"""06_fig_sensitivity.py — Fig. 6: cross-setting spectral sensitivity comparison.

Inputs (produced by 05_spectral_sensitivity.py):
  results/spectral_sensitivity_directionality.csv
  results/spectral_sensitivity_models.csv

Outputs:
  fig6_sensitivity.{pdf,png,svg}

Panel (a): pre-procedural band-power difference (strong responders vs others)
           for the three replication bands across the four spectral settings.
Panel (b): nested-LOPO AUROC [95% CI] for the strong-response outcome of
           EEG F1 / F2 / F2+PSI across the same settings.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
common = import_module("00_common")

FIG = common.FIGURES
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

SETTINGS = ["T3_NW6", "T10_NW4", "T20_NW4", "T60_NW6"]
SETTING_LABELS = ["3 s\n4 Hz", "10 s\n0.8 Hz", "20 s\n0.4 Hz", "60 s\n0.2 Hz"]

BANDS = [
    ("F1_norm_slowdelta", "0.1--1.5 Hz", "#B64342"),
    ("F1_norm_fastdelta", "2.5--5 Hz", "#0F4D92"),
    ("F1_norm_alpha", "8--12 Hz", "#5F5F5F"),
]

MODELS = [
    ("EEG_F1", "EEG F1 (3-band)", "#0F4D92", "o"),
    ("EEG_F2", "EEG F2 (canonical)", "#3775BA", "s"),
    ("EEG_F2_PSI", "EEG F2 + PSI", "#D9908B", "^"),
]


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def main():
    res = Path(__file__).resolve().parents[1] / "results"
    dire = pd.read_csv(res / "spectral_sensitivity_directionality.csv")
    mod = pd.read_csv(res / "spectral_sensitivity_models.csv")

    dire = dire[dire["label"] == "post_bps_ge7"]
    mod = mod[mod["label"] == "post_bps_ge7"]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ---- Panel (a): band-power directionality across settings -------------
    x = np.arange(len(SETTINGS))
    offset = {"F1_norm_slowdelta": -0.22, "F1_norm_fastdelta": 0.0,
              "F1_norm_alpha": 0.22}
    for feat, lab, col in BANDS:
        sub = dire.set_index(["setting", "feature"]).loc[
            [(s, feat) for s in SETTINGS]].reset_index()
        y = sub["pct_diff_median"].values
        xx = x + offset[feat]
        axa.plot(xx, y, "-", color=col, lw=0.9, alpha=0.7, zorder=1)
        axa.plot(xx, y, "o", color=col, ms=4.5, label=lab, zorder=2)
        for xi, yi, pi in zip(xx, y, sub["p_mw"].values):
            va = "bottom" if yi >= 0 else "top"
            off = 3 if yi >= 0 else -3
            axa.annotate(f"P={fmt_p(pi)}", (xi, yi), textcoords="offset points",
                         xytext=(0, off), ha="center", va=va,
                         fontsize=5.5, color=col)

    axa.axhline(0, color="black", lw=0.8)
    axa.set_xticks(x)
    axa.set_xticklabels(SETTING_LABELS)
    axa.set_xlabel("Multitaper window / bandwidth")
    axa.set_ylabel("Difference in median power (%)")
    axa.set_title("(a) Replication-band directionality (BPS $\\geq$7)", loc="left")
    axa.legend(frameon=False, loc="upper left", fontsize=6)
    axa.margins(y=0.28)

    # ---- Panel (b): model AUROC across settings ---------------------------
    for mname, mlab, col, mk in MODELS:
        sub = mod[mod["model"] == mname].set_index("setting").loc[SETTINGS]
        y = sub["auroc"].values
        err_lo = y - sub["ci_lo"].values
        err_hi = sub["ci_hi"].values - y
        axb.errorbar(x, y, yerr=[err_lo, err_hi], fmt=mk + "-", color=col,
                     ms=4.5, lw=0.9, capsize=2.5, elinewidth=0.8,
                     label=mlab, zorder=2)

    axb.axhline(0.5, color="black", lw=0.8, ls="--")
    axb.text(len(SETTINGS) - 0.48, 0.505, "chance", fontsize=5.5,
             ha="right", va="bottom")
    axb.set_xticks(x)
    axb.set_xticklabels(SETTING_LABELS)
    axb.set_xlabel("Multitaper window / bandwidth")
    axb.set_ylabel("AUROC (95% CI)")
    axb.set_ylim(0.35, 0.95)
    axb.set_title("(b) Strong-response AUROC, nested LOPO", loc="left")
    axb.legend(frameon=False, loc="lower left", fontsize=6)

    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"fig6_sensitivity.{ext}",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"written -> {FIG / 'fig6_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
