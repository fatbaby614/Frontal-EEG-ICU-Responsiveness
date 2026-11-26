# Frontal-EEG-ICU-Responsiveness

Analysis code for the study:

> **Pre-procedural frontal EEG for predicting strong behavioural responsiveness to noxious care procedures in mechanically ventilated ICU patients: a patient-level validation study**

The study is a retrospective secondary analysis of the open dataset [ds008115](https://openneuro.org/datasets/ds008115) (four-channel frontal EEG, PSI and NOL monitor values during endotracheal suctioning and patient repositioning). It tests whether pre-procedural frontal EEG features predict behavioural pain responsiveness (Behavioural Pain Scale) with patient-level generalisation, and whether they add value beyond the commercial indices PSI and NOL.

## Repository layout

```
code/   analysis scripts (this repository)
data/   ds008115 download from OpenNeuro (not included; create locally)
results/  all intermediate and final result tables (created by the scripts)
paper/    figure output directory created by the figure scripts (PDF/SVG/PNG)
```

## Requirements

Python 3 with the following packages:

- `numpy`, `pandas`, `scipy`
- `scikit-learn`, `statsmodels`
- `mne`, `matplotlib`

## Data

Download dataset **ds008115** from OpenNeuro and place it at:

```
data/ds008115/
```

All other paths are resolved relative to the project root by `00_common.py`; results are written to `results/`.

## Scripts

Run scripts from the project root in this order:

| Script | Purpose |
|---|---|
| `00_common.py` | Shared configuration and utilities (imported by all scripts) |
| `01_extract_features.py` | Feature extraction and BJA-2023 directionality replication → `features.csv`, `w1_replication.json` |
| `02_run_experiments.py` | Main nested-LOPO experiments (EEG F1–F4, PSI, NOL, combinations) → `main_results.csv`, `preds_long.csv`, `delta_auc.csv` |
| `03_cross_task.py` | Cross-procedure transfer (train suctioning → test repositioning) → `cross_task.csv` |
| `07_metrics_p2.py` | AUPRC, Brier score and calibration slope from existing out-of-fold predictions → `discrimination_calibration.csv` |
| `04_figures.py` | Manuscript Figures 2–5 (PDF/SVG/PNG) |
| `05_revision_analyses.py` | Revision analyses: magnitude coding, selection-bias comparison, reverse-direction permutation test |
| `05_spectral_sensitivity.py` | Spectral-estimation sensitivity (multitaper window/bandwidth settings) |
| `06_fig_sensitivity.py` | Manuscript Figure 6 (spectral sensitivity) |
| `06_rater_sensitivity.py` | Rater (Investigator A/B) confound sensitivity → `rater_sensitivity.csv` |
| `07_forward_permutation.py` | Forward-direction (suctioning → repositioning) label permutation test |
| `08_raterA_lopo.py` | Rater-A-only LOPO sensitivity → `raterA_lopo.csv` |
| `09_clinical_baseline.py` | Clinical-only baseline comparison (baseline BPS, vital signs) → `clinical_baseline.csv` |
| `10_perm_coef.py` | Headline label-permutation test and logistic-regression coefficient stability (`coef` / `perm` / `both`) |

Example:

```bash
python code/01_extract_features.py
python code/02_run_experiments.py
python code/03_cross_task.py
python code/07_metrics_p2.py
python code/04_figures.py
```

## Data availability

The dataset is publicly available on OpenNeuro: accession **ds008115**.

## Citation

If you use this code, please cite the associated manuscript and the source dataset (OpenNeuro ds008115).
