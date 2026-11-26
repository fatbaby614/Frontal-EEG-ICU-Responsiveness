# -*- coding: utf-8 -*-
"""04_figures.py — generate all manuscript figures into
paper/JournalofClinicalMonitoringAndComputing/figures/.

Inputs (all produced by 01-03):
  results/features.csv, results/w1_replication.json, results/cohort_flow.json
  results/main_results.csv, results/preds_long.csv, results/delta_auc.csv
  results/discrimination_calibration.csv, results/cross_task.csv

Outputs (vector PDF + SVG, 600-dpi PNG preview):
  fig2_bandreplication.{pdf,png,svg}
  fig3_eval_triad.{pdf,png,svg}
  fig4_auc_grid.{pdf,png,svg}
  fig5_delta_crosstask.{pdf,png,svg}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Patch, PathPatch
import matplotlib.path as mpath
from sklearn.metrics import roc_curve, precision_recall_curve

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

# Hue families follow the Fig. 3 caption semantics:
# blue = EEG-only, red = EEG + clinical indices, purple = NOL-based,
# grey = PSI-based, green = pre-BPS corrected. Shades separate models
# within a family.
COLOR = {
    "EEG_F1": "#0F4D92", "EEG_F2": "#3775BA", "EEG_F3": "#6C9BCB", "EEG_F4": "#A9C6E4",
    "EEG_F1_PSI": "#B64342", "EEG_F2_PSI": "#D9908B", "EEG_F2_PSI_NOL": "#8C3332",
    "PSI": "#5F5F5F", "NOL": "#6A4C93", "PSI_NOL": "#9A4D8E",
    "PSI_PreBPS": "#57A870", "EEG_F2_PreBPS": "#2E7D52",
}
LABEL_NICE = {
    "EEG_F1": "EEG BJA-3band", "EEG_F2": "EEG canonical (16)",
    "EEG_F3": "EEG spectral stats (21)", "EEG_F4": "EEG spectral+Riemann (71)",
    "PSI": "PSI", "NOL": "NOL", "PSI_NOL": "PSI+NOL",
    "EEG_F1_PSI": "EEG BJA + PSI", "EEG_F2_PSI": "EEG canonical + PSI",
    "EEG_F2_PSI_NOL": "EEG canonical + PSI + NOL",
    "PSI_PreBPS": "PSI + pre-BPS", "EEG_F2_PreBPS": "EEG canonical + pre-BPS",
}
# Grouped top-to-bottom order used by Fig. 3 (y = 11 - index).
MODEL_ORDER = ["EEG_F1", "EEG_F2", "EEG_F3", "EEG_F4",
               "EEG_F1_PSI", "EEG_F2_PSI", "EEG_F2_PSI_NOL",
               "NOL", "PSI_NOL", "PSI",
               "PSI_PreBPS", "EEG_F2_PreBPS"]
# Family background bands: (start index, end index exclusive, colour)
FAMILY_BANDS = [
    (0, 4, "#0F4D92"),    # EEG-only
    (4, 7, "#B64342"),    # EEG + clinical indices
    (7, 9, "#6A4C93"),    # NOL-based
    (9, 10, "#5F5F5F"),   # PSI-based
    (10, 12, "#2E7D52"),  # pre-BPS corrected
]
FAMILY_LABELS = ["EEG-only", "EEG + indices", "NOL-based", "PSI-based", "Pre-BPS corrected"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def fig_bandreplication():
    """Directional univariate replication: pre-window normalised power boxplots.

    The former cohort-flow panel was removed (workflow covered by fig1_workflow;
    cohort counts are stated in the Methods text), leaving a single-row figure.
    """
    w1 = json.loads((common.RESULTS / "w1_replication.json").read_text())

    feats = pd.read_csv(common.RESULTS / "features.csv")
    feats = feats[feats.task == "EndotrachealSuction"]
    y = feats["post_bps_ge7"].to_numpy().astype(bool)
    pre_keys = ["F1_norm_fastdelta", "F1_norm_slowdelta", "F1_norm_alpha"]
    names = ["2.5–5 Hz\n(fast delta)", "0.1–1.5 Hz\n(slow delta)", "8–12 Hz\n(alpha)"]
    pvals = [w1["BPS>=7"]["F1_norm_fastdelta"]["p_mw"],
             w1["BPS>=7"]["F1_norm_slowdelta"]["p_mw"],
             w1["BPS>=7"]["F1_norm_alpha"]["p_mw"]]
    aucs = [w1["BPS>=7"][k]["auc_rank"] for k in pre_keys]

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.7))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.90, wspace=0.42)

    for j, key in enumerate(pre_keys):
        axb = axes[j]
        pos = feats[key][y == 1]
        neg = feats[key][y == 0]
        bp = axb.boxplot([neg, pos], widths=0.55, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", lw=1.0),
                         whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                         boxprops=dict(lw=0.8))
        for patch, (fc, ec) in zip(bp["boxes"],
                                   [("#C9D9EC", "#3775BA"), ("#E9A6A1", "#B64342")]):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
        # headroom so the AUC annotation clears the whisker/cap tops
        y0, y1 = axb.get_ylim()
        axb.set_ylim(y0, y0 + (y1 - y0) * 1.35)
        axb.set_xticks([1, 2])
        axb.set_xticklabels([f"BPS<7\n(n={len(neg)})", f"BPS$\\geq$7\n(n={len(pos)})"])
        axb.set_title(names[j], fontsize=8)
        star = "***" if pvals[j] < 1e-3 else ("**" if pvals[j] < 1e-2 else
               ("*" if pvals[j] < 0.05 else "ns"))
        axb.text(0.97, 0.96, f"{star}\nAUC={aucs[j]:.2f}", transform=axb.transAxes,
                 ha="right", va="top", fontsize=7)
        if j == 0:
            axb.set_ylabel("Normalised power\n(pre-procedure)")
    save(fig, "fig2_bandreplication")


def fig2():
    CO = "EndotrachealSuction"; LB = "post_bps_ge7"
    res = pd.read_csv(common.RESULTS / "main_results.csv")
    cal = pd.read_csv(common.RESULTS / "discrimination_calibration.csv")
    sub = res[(res.cohort == CO) & (res.label == LB)]
    sub = sub.set_index("model")

    show = ["EEG_F1", "EEG_F2", "PSI", "NOL", "PSI_NOL", "EEG_F1_PSI", "EEG_F2_PSI"]
    cal_models = ["EEG_F1_PSI", "EEG_F1", "PSI", "PSI_NOL"]
    thick = {"EEG_F1_PSI": 1.8, "EEG_F2": 1.4}
    default_lw = 1.0

    preds = pd.read_csv(common.RESULTS / "preds_long.csv")
    preds = preds[(preds.cohort == CO) & (preds.label == LB)]

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 3.15))
    ax_roc, ax_pr, ax_cal = axes

    # (a) ROC
    for m in show:
        d = preds[preds.model == m]
        fpr, tpr, _ = roc_curve(d.y, d.p)
        a = sub.loc[m, "auroc"]; lo = sub.loc[m, "ci_lo"]; hi = sub.loc[m, "ci_hi"]
        ax_roc.plot(fpr, tpr, color=COLOR[m], lw=thick.get(m, default_lw),
                    label=f"{LABEL_NICE[m]}  AUROC={a:.3f} [{lo:.2f}-{hi:.2f}]")
    ax_roc.plot([0, 1], [0, 1], ls=":", color="0.5", lw=0.8)
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_title("(a) ROC")

    # (b) precision-recall
    ref = preds[preds.model == "EEG_F1"]
    prev = float(ref.y.mean())
    for m in show:
        d = preds[preds.model == m]
        precision, recall, _ = precision_recall_curve(d.y, d.p)
        c = cal[(cal.cohort == CO) & (cal.label == LB) & (cal.model == m)]
        lab = LABEL_NICE[m]
        if not c.empty:
            lab = (f"{lab}  AUPRC={c.auprc.iloc[0]:.3f} "
                   f"[{c.auprc_lo.iloc[0]:.2f}-{c.auprc_hi.iloc[0]:.2f}]")
        ax_pr.plot(recall, precision, color=COLOR[m],
                   lw=thick.get(m, default_lw), label=lab)
    ax_pr.plot([0, 1], [prev, prev], ls=":", color="0.5", lw=0.8)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_title("(b) Precision–recall")

    # (c) calibration, equal-count quintile bins of predicted probability
    for m in cal_models:
        d = preds[preds.model == m]
        order = np.argsort(d.p.to_numpy())
        ys = d.y.to_numpy()[order]
        ps = d.p.to_numpy()[order]
        groups = np.array_split(np.arange(len(order)), 5)
        mp, ob = [], []
        for g in groups:
            if len(g) == 0:
                continue
            mp.append(float(np.mean(ps[g])))
            ob.append(float(np.mean(ys[g])))
        ax_cal.plot(mp, ob, "o-", color=COLOR[m],
                    ms=3.0, lw=thick.get(m, 1.2), mec=COLOR[m], mew=0.6)
    ax_cal.plot([0, 1], [0, 1], ls="--", color="0.5", lw=0.8)
    ax_cal.text(0.97, 0.05, "Perfect calibration", fontsize=6.5, color="0.35",
                ha="right", va="bottom")
    ax_cal.set_xlabel("Predicted probability")
    ax_cal.set_ylabel("Observed frequency")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_title("(c) Calibration")

    handles = [Line2D([0], [0], color=COLOR[m], lw=1.6, label=LABEL_NICE[m])
               for m in show]
    handles.append(Line2D([0], [0], ls=":", color="0.5", lw=0.8,
                          label=f"Full-cohort prevalence ({prev:.2f})"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, fontsize=6.5, frameon=False, columnspacing=1.2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.32, wspace=0.36)
    save(fig, "fig3_eval_triad")


def fig3():
    res = pd.read_csv(common.RESULTS / "main_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.6), sharex=True, sharey=True)
    panels = [("EndotrachealSuction", "post_bps_ge7", "Suctioning, BPS$\\geq$7"),
              ("EndotrachealSuction", "post_bps_gt3", "Suctioning, BPS>3"),
              ("Positioning", "post_bps_ge7", "Repositioning, BPS$\\geq$7"),
              ("Positioning", "post_bps_gt3", "Repositioning, BPS>3")]

    for k, (ax, (co, lb, title)) in enumerate(zip(axes.ravel(), panels)):
        sub = res[(res.cohort == co) & (res.label == lb)].set_index("model")
        # family background bands (rows y = 11 - i)
        for a, b, col in FAMILY_BANDS:
            ax.axhspan(11 - (b - 1) - 0.5, 11 - a + 0.5, color=col, alpha=0.07,
                       lw=0, zorder=0)
        for i, m in enumerate(MODEL_ORDER):
            if m not in sub.index:
                continue
            y = 11 - i
            v, lo, hi = (sub.loc[m, "auroc"], sub.loc[m, "ci_lo"], sub.loc[m, "ci_hi"])
            kw = dict(fmt="o", ms=3.5, color=COLOR[m], ecolor="0.35",
                      elinewidth=0.9, capsize=2, mec="none")
            if m == "EEG_F1_PSI":
                kw.update(ms=5.0, mec="white", mew=0.7)
            ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], **kw)
        ax.axvline(0.5, ls=":", color="0.6", lw=0.8)
        n = int(sub["n"].max()); npos = int(sub["n_pos"].max())
        ax.set_title(f"{title}\nn={n} episodes, {npos} positive", fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 11.5)
        ax.set_yticks(list(range(12)))
        if k % 2 == 0:
            # ticks ascend from y=0, points sit at y=11-i → reverse the order
            ax.set_yticklabels([LABEL_NICE[m] for m in reversed(MODEL_ORDER)],
                               fontsize=6.5)
        ax.tick_params(axis="y", labelleft=(k % 2 == 0))
        if k >= 2:
            ax.set_xlabel("AUROC (95% CI, patient-level bootstrap)")
    handles = [Patch(facecolor=col, alpha=0.25, label=lab)
               for (_, _, col), lab in zip(FAMILY_BANDS, FAMILY_LABELS)]
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.945),
               ncol=5, fontsize=6.5, frameon=False, columnspacing=1.4)
    save(fig, "fig4_auc_grid")


def _striped_key(legend, orig_handle, xdescent, ydescent, width, height,
                 fontsize):
    """Legend key with hand-drawn white stripes.

    matplotlib's hatch= writes PDF tiling patterns; PyMuPDF's expansion of
    those patterns yields phantom stroke paths that trip the collision audit,
    so both bars and legend keys draw stripes explicitly.
    """
    x0, y0 = -xdescent, -ydescent
    verts = [(x0, y0), (x0 + width, y0), (x0 + width, y0 + height),
             (x0, y0 + height), (x0, y0)]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 4
    c = y0 - (x0 + width)  # y = x + c hits the key box within this c range
    while c <= (y0 + height) - x0 + 1e-9:
        ya, yb = max(c + x0, y0), min(c + x0 + width, y0 + height)
        if ya < yb - 1e-9:
            verts += [(ya - c, ya), (yb - c, yb)]
            codes += [mpath.Path.MOVETO, mpath.Path.LINETO]
        c += 2.5
    return PathPatch(mpath.Path(verts, codes))


def fig4():
    da = pd.read_csv(common.RESULTS / "delta_auc.csv")
    ct = pd.read_csv(common.RESULTS / "cross_task.csv")

    fig = plt.figure(figsize=(7.08, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.46,
                          left=0.11, right=0.97, top=0.86, bottom=0.38)

    # (a) paired delta-AUROC contrasts
    axa = fig.add_subplot(gs[0])
    rows = [("d_EEGvsPSI_A", "post_bps_ge7", "EEG $-$ PSI\nBPS $\\geq$7"),
            ("d_EEGvsPSI_NOL_B", "post_bps_ge7", "EEG $-$ PSI+NOL\nBPS $\\geq$7"),
            ("d_EEGvsPSI_A", "post_bps_gt3", "EEG $-$ PSI\nBPS>3"),
            ("d_EEGvsPSI_NOL_B", "post_bps_gt3", "EEG $-$ PSI+NOL\nBPS>3")]
    labels, ys = [], []
    for k, (key, lb, lab) in enumerate(rows):
        r = da[(da["contrast"] == key) & (da["label"] == lb)].iloc[0]
        y = 3 - k
        col = "#0F4D92" if lb == "post_bps_ge7" else "#B64342"
        filled = r["p_boot"] < 0.05
        axa.errorbar([r["delta_auc"]], [y],
                     xerr=[[r["delta_auc"] - r["ci_lo"]],
                           [r["ci_hi"] - r["delta_auc"]]],
                     fmt="o", ms=5, color=col, ecolor="0.35", elinewidth=0.9,
                     capsize=2, mec=col, mew=0.9,
                     mfc=col if filled else "white")
        pv = r["p_boot"]
        ptxt = "P<0.001" if pv < 0.001 else f"P={pv:.3f}"
        axa.text(1.02, y, ptxt, transform=axa.get_yaxis_transform(),
                 fontsize=6.5, va="center", ha="left", color="0.15")
        labels.append(lab); ys.append(y)
    axa.axvline(0, color="0.3", lw=0.8)
    axa.text(1.02, 3.55, "P (bootstrap)", transform=axa.get_yaxis_transform(),
             fontsize=6.5, va="center", ha="left", fontweight="bold")
    axa.set_yticks(ys)
    axa.set_yticklabels(labels, fontsize=6.5)
    axa.set_ylim(-0.6, 3.75)
    axa.set_xlim(-0.68, 0.34)
    axa.set_xlabel("$\\Delta$AUROC (EEG $-$ comparator)")
    axa.set_title("(a) Paired contrasts (suctioning)", fontsize=8)
    sig = Line2D([0], [0], marker="o", ls="none", mfc="#4D4D4D", mec="#4D4D4D",
                 ms=4.5, label="P < 0.05")
    ns = Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#4D4D4D",
                ms=4.5, label="Not significant")
    axa.legend(handles=[sig, ns], loc="upper left", fontsize=6.5, frameon=False,
               handletextpad=0.4, borderaxespad=0.25)

    # (b) cross-procedure transfer
    axb = fig.add_subplot(gs[1])
    models = ["EEG_F2", "EEG_F2_PSI", "EEG_F2_PSI_NOL", "NOL", "PSI_NOL", "PSI"]
    combos = [("EndotrachealSuction", "post_bps_ge7"),
              ("Positioning", "post_bps_ge7"),
              ("EndotrachealSuction", "post_bps_gt3"),
              ("Positioning", "post_bps_gt3")]
    w = 0.19
    stripe = []
    for m_i, m in enumerate(models):
        for k, (tr, lb) in enumerate(combos):
            r = ct[(ct.train == tr) & (ct.label == lb) & (ct.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            col = "#0F4D92" if lb == "post_bps_ge7" else "#B64342"
            x = m_i + (k - 1.5) * w
            axb.bar(x, r["auroc"], w, yerr=[[r["auroc"] - r["ci_lo"]],
                                            [r["ci_hi"] - r["auroc"]]],
                    color=col, edgecolor="white", lw=0.5,
                    error_kw=dict(elinewidth=0.8, capsize=1.5, ecolor="0.3"),
                    zorder=3)
            if tr == "Positioning":
                stripe.append((x - w / 2, w, r["auroc"]))
    axb.axhline(0.5, ls=":", color="0.5", lw=0.8, zorder=2)
    axb.set_xticks(range(len(models)))
    short = {"EEG_F2": "EEG F2", "EEG_F2_PSI": "EEG F2\n+PSI",
             "EEG_F2_PSI_NOL": "EEG F2\n+PSI+NOL", "NOL": "NOL",
             "PSI_NOL": "PSI+NOL", "PSI": "PSI"}
    axb.set_xticklabels([short[m] for m in models], fontsize=6.5)
    axb.set_ylim(0, 1.06)
    axb.set_xlim(-0.55, len(models) - 0.45)

    # manual white stripes on "trained on repositioning" bars (slope chosen
    # so they render at 45 degrees); see _striped_key for why hatch= is avoided
    fig.canvas.draw()
    bb = axb.bbox
    x_lo, x_hi = axb.get_xlim()
    y_lo, y_hi = axb.get_ylim()
    slope = (y_hi - y_lo) * bb.width / ((x_hi - x_lo) * bb.height)
    dx_step = 2.5 * fig.dpi / 72.0 * (x_hi - x_lo) / bb.width  # ~2.5 pt
    for bx0, bw, bh in stripe:
        c = -slope * (bx0 + bw)  # y = slope*x + c crosses bottom-right corner
        c_end = bh - slope * bx0  # ... up to top-left corner
        while c <= c_end + 1e-9:
            x1, y1 = bx0, slope * bx0 + c
            x2, y2 = bx0 + bw, slope * (bx0 + bw) + c
            if y1 < y_lo:
                y1, x1 = y_lo, (y_lo - c) / slope
            if y2 > bh:
                y2, x2 = bh, (bh - c) / slope
            if x1 < x2 - 1e-12:
                axb.add_line(Line2D([x1, x2], [y1, y2], color="white", lw=0.5,
                                    solid_capstyle="butt", zorder=4))
            c += slope * dx_step
    axb.set_ylabel("Test AUROC")
    axb.set_title("(b) Cross-procedure transfer", fontsize=8)
    h_ge7 = Patch(facecolor="#0F4D92", label="BPS $\\geq$7")
    h_gt3 = Patch(facecolor="#B64342", label="BPS>3")
    h_sol = Patch(facecolor="#9E9E9E", edgecolor="white", lw=0.5,
                  label="Trained on suctioning")
    h_hat = PathPatch(mpath.Path.unit_rectangle(), facecolor="#9E9E9E",
                      edgecolor="white", lw=0.5,
                      label="Trained on repositioning")
    axb.legend(handles=[h_ge7, h_gt3, h_sol, h_hat], loc="upper center",
               bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=6.5, frameon=False,
               columnspacing=1.2, handletextpad=0.6,
               handler_map={PathPatch: HandlerPatch(patch_func=_striped_key)})
    save(fig, "fig5_delta_crosstask")


if __name__ == "__main__":
    fig_bandreplication()
    fig2()
    fig3()
    fig4()
    print("all figures done")
