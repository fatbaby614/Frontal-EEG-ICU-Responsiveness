# -*- coding: utf-8 -*-
"""频谱估计敏感性分析（JCMC 修改意见 3.1，最高优先级）。

背景：原管线用 3 秒分段 + multitaper 带宽 4 Hz（NW=6），无法可靠估计
0.1--1.5 Hz 频段（0.1 Hz 周期为 10 秒；4 Hz 平滑带宽大于该频段宽度）。
本脚本在更长分段窗口、与窗口匹配的带宽下重算 F1/F2 特征，检验：
  1) BJA 方向性复现（0.1--1.5 / 2.5--5 / 8--12 Hz，强反应者 vs 其他）是否稳定；
  2) 主要模型（F1、F2、F2+PSI，BPS>=7 与 BPS>3，吸痰队列）AUROC 是否稳定；
  3) 跨操作迁移（F2，吸痰->体位摆放）是否稳定；
  4) 0.1--1.5 Hz 功率在 120 秒窗内的漂移/非平稳性。

基线设置（3 秒 / NW=6 / 带宽 4 Hz）直接复用 features.csv 与
main_results.csv，不重复计算。

输出（results/）
  spectral_sensitivity_features.csv      各设置的 episode 级特征
  spectral_sensitivity_directionality.csv 方向性复现对比表
  spectral_sensitivity_models.csv        模型 AUROC 对比表
  spectral_sensitivity_drift.csv         低频段漂移/非平稳性
  spectral_sensitivity_summary.md        汇总（供稿件改写参考）
"""
from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
C = import_module("00_common")
E = import_module("02_run_experiments")

# --------------------------------------------------------------------------- #
# 设置网格：基线复用旧结果，其余为敏感性设置
#   bandwidth = 2 * NW / T（mne psd_array_multitaper 约定）
#   分段步长统一 1 s，与基线一致
# --------------------------------------------------------------------------- #
SETTINGS = [
    # name, 段长 T (s), time-bandwidth NW, 带宽 (Hz), 使用基线旧结果?
    ("T3_NW6", 3.0, 6.0, 4.0, True),
    ("T10_NW4", 10.0, 4.0, 0.8, False),
    ("T20_NW4", 20.0, 4.0, 0.4, False),
    ("T60_NW6", 60.0, 6.0, 0.2, False),
]
STEP_S = 1.0
FMIN, FMAX = 0.1, 41.0
MODEL_LABELS = ("post_bps_ge7", "post_bps_gt3")
SENS_MODELS = ["EEG_F1", "EEG_F2", "EEG_F2_PSI"]
# 02 中的模型名 -> 特征集/列 spec（model_columns 只认特征集名）
MODEL_SPECS = dict(E.MODELS)


# --------------------------------------------------------------------------- #
# 本地分段/PSD（参数化窗口与带宽，不改动 00_common 以免影响既有管线）
# --------------------------------------------------------------------------- #
def segment_indices(n_times: int, sfreq: float, t0: float, t1: float,
                    onset: float, seg_s: float, step_s: float = STEP_S) -> list[int]:
    seg = int(round(seg_s * sfreq))
    step = int(round(step_s * sfreq))
    s0 = int(round((onset + t0) * sfreq))
    s1 = int(round((onset + t1) * sfreq))
    idx = list(range(s0, s1 - seg + 1, step))
    return [i for i in idx if i >= 0 and i + seg <= n_times]


def segment_psd_param(data: np.ndarray, sfreq: float, starts: list[int],
                      seg_s: float, nw: float) -> tuple[np.ndarray, np.ndarray]:
    """按指定段长与 NW 计算多锥谱；返回 (n_seg, 4, n_freq), freqs。"""
    from mne.time_frequency import psd_array_multitaper
    seg = int(round(seg_s * sfreq))
    X = np.stack([data[:, s:s + seg] - data[:, s:s + seg].mean(axis=1, keepdims=True)
                  for s in starts])
    n_seg = X.shape[0]
    Xf = X.reshape(n_seg * 4, seg)
    bandwidth = 2.0 * nw / seg_s
    psds, freqs = psd_array_multitaper(
        Xf, sfreq, fmin=FMIN, fmax=FMAX, normalization="length",
        bandwidth=bandwidth, verbose=False)
    return psds.reshape(n_seg, 4, len(freqs)).astype(np.float64), freqs


# --------------------------------------------------------------------------- #
# 单被试特征（列名与 01_extract_features.py 保持一致，便于复用 02 的特征选择）
# --------------------------------------------------------------------------- #
def compute_subject(row, seg_s: float, nw: float) -> dict | None:
    sub, task = row["participant_id"], C.task_of(row)
    try:
        data, sfreq = C.load_eeg(sub, task)
        onset = C.read_onset(sub, task)
    except Exception as e:
        print(f"  [FAIL] {sub}: {e}")
        return None
    ch_best, ch_src = C.best_channel(sub, task, row)
    ci = C.CHANNELS.index(ch_best)
    starts = segment_indices(data.shape[1], sfreq, *C.PRE_WIN, onset, seg_s)
    if not starts:
        print(f"  [WARN] {sub}: 无可用 {seg_s}s 段")
        return None
    psd, freqs = segment_psd_param(data, sfreq, starts, seg_s, nw)
    tot = psd.sum(axis=-1, keepdims=True)          # 每段每通道 0.1--41 Hz 总功率
    norm = psd / tot
    cb, cn = psd[:, ci, :], norm[:, ci, :]          # 最佳质量通道

    out = {"participant_id": sub, "task": task, "best_channel": ch_best,
           "best_channel_source": ch_src, "n_seg": len(starts)}
    # F1：BJA 归一化频段，跨段均值
    for bname, bnd in C.BANDS_BJA.items():
        m = (freqs >= bnd[0]) & (freqs <= bnd[1])
        out[f"F1_norm_{bname}"] = float(cn[:, m].mean())
    # F2：7 归一化 + 7 log10 绝对 + spen + sef95（跨段 mean / sef95 为 median）
    for bname, bnd in C.BANDS_CANON.items():
        m = (freqs >= bnd[0]) & (freqs <= bnd[1])
        out[f"F2_norm_{bname}"] = float(cn[:, m].mean())
        out[f"F2_logabs_{bname}"] = float(np.log10(np.clip(cb[:, m].mean(), 1e-30, None)))
    out["F2_spen"] = float(C.spectral_entropy(freqs, cn).mean())
    sef95 = C.sef(freqs, cb, 0.95)
    out["F2_sef95"] = float(np.nanmedian(sef95))
    # 漂移诊断：低频归一化功率的段序列（前半 vs 后半 + 斜率）
    m_slow = (freqs >= 0.1) & (freqs <= 1.5)
    slow = cn[:, m_slow].mean(axis=1)
    half = len(slow) // 2
    out["_slow_first"] = float(slow[:half].mean())
    out["_slow_last"] = float(slow[half:].mean())
    t = np.arange(len(slow))
    out["_slow_slope"] = float(np.polyfit(t, slow, 1)[0]) if len(slow) > 2 else np.nan
    out["_slow_relrange"] = float((np.nanmax(slow) - np.nanmin(slow)) /
                                  (np.nanmean(slow) + 1e-30))
    # 附带元数据（与基线一致）
    out.update({
        "post_bps": row["Post_BehavioralPainScale"],
        "post_bps_ge7": int(row["Post_BehavioralPainScale"] >= 7),
        "post_bps_gt3": int(row["Post_BehavioralPainScale"] > 3),
        "pre_sed_mean": row["Pre_SED_Mean"],
        "pre_nol_mean": row["Pre_NOL_Mean"],
        "pre_bps": row["Pre_BehavioralPainScale"],
    })
    return out


def extract_setting(name: str, seg_s: float, nw: float) -> pd.DataFrame:
    parts = C.load_participants()
    cohort = parts[C.cohort_mask(parts, "EndotrachealSuction")
                   | C.cohort_mask(parts, "Positioning")]
    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(cohort.iterrows(), 1):
        r = compute_subject(row, seg_s, nw)
        if r is not None:
            r["setting"] = name
            rows.append(r)
        if i % 40 == 0:
            print(f"    [{name}] {i}/{len(cohort)} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    print(f"    [{name}] 完成 {len(df)} 例，用时 {time.time()-t0:.0f}s")
    return df


# --------------------------------------------------------------------------- #
# 分析 1：方向性复现
# --------------------------------------------------------------------------- #
def directionality(fdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, g in fdf.groupby("setting", sort=False):
        d = g[g["task"] == "EndotrachealSuction"]
        for label in MODEL_LABELS:
            for col in [f"F1_norm_{b}" for b in C.BANDS_BJA]:
                dd = d[[col, label]].dropna()
                y = dd[label].astype(int).values
                x = dd[col].values
                if y.sum() < 5 or (len(y) - y.sum()) < 5:
                    continue
                pos, neg = x[y == 1], x[y == 0]
                u, p = mannwhitneyu(pos, neg, alternative="two-sided")
                med_p, med_n = float(np.median(pos)), float(np.median(neg))
                pct = 100 * (med_p - med_n) / abs(med_n) if med_n != 0 else np.nan
                rows.append(dict(
                    setting=setting, label=label, feature=col, n=len(y),
                    n_pos=int(y.sum()),
                    median_pos=med_p, median_neg=med_n, pct_diff_median=pct,
                    auc_rank=float(u / (len(pos) * len(neg))), p_mw=float(p)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 分析 2：模型 AUROC（嵌套 LOPO，与 02 完全同一流程）
# --------------------------------------------------------------------------- #
def run_models(fdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, g in fdf.groupby("setting", sort=False):
        d = g[g["task"] == "EndotrachealSuction"].copy()
        for label in MODEL_LABELS:
            dd0 = d.dropna(subset=[label])
            for mname in SENS_MODELS:
                cols = E.model_columns(MODEL_SPECS[mname], dd0.columns)
                dd = dd0.dropna(subset=cols)
                X, y = dd[cols], dd[label].astype(int).values
                if y.sum() < 5 or (len(y) - y.sum()) < 5:
                    continue
                oof, ch = E.nested_loso(X, y, dd["participant_id"].values)
                auc, lo, hi = E.boot_ci(oof, y, dd["participant_id"].values)
                rows.append(dict(setting=setting, label=label, model=mname,
                                 n=len(y), n_pos=int(y.sum()),
                                 auroc=auc, ci_lo=lo, ci_hi=hi,
                                 median_C=float(np.median(ch))))
                print(f"    [{setting}] {label} {mname}: "
                      f"AUC={auc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 分析 3：跨操作迁移 F2（吸痰 ge7 -> 体位摆放 ge7），与 02 的 cross-task 同法
# --------------------------------------------------------------------------- #
def run_transfer(fdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, g in fdf.groupby("setting", sort=False):
        train = g[g["task"] == "EndotrachealSuction"].dropna(subset=["post_bps_ge7"])
        test = g[g["task"] == "Positioning"].dropna(subset=["post_bps_ge7"])
        for mname in ("EEG_F2", "EEG_F2_PSI"):
            cols = E.model_columns(MODEL_SPECS[mname], train.columns)
            tr = train.dropna(subset=cols)
            te = test.dropna(subset=cols)
            Xtr, ytr = tr[cols], tr["post_bps_ge7"].astype(int).values
            Xte, yte = te[cols], te["post_bps_ge7"].astype(int).values
            if min(ytr.sum(), len(ytr) - ytr.sum(),
                   yte.sum(), len(yte) - yte.sum()) < 5:
                continue
            oof, ch = E.nested_loso(Xtr, ytr, tr["participant_id"].values)
            pipe = E.fit_proba(Xtr, ytr, float(np.median(ch)))
            pte = pipe.predict_proba(Xte.values)[:, 1]
            auc, lo, hi = E.boot_ci(pte, yte, te["participant_id"].values)
            rows.append(dict(setting=setting, model=mname,
                             train_n=len(ytr), train_pos=int(ytr.sum()),
                             test_n=len(yte), test_pos=int(yte.sum()),
                             auroc=auc, ci_lo=lo, ci_hi=hi))
            print(f"    [{setting}] transfer {mname}: AUC={auc:.3f} "
                  f"[{lo:.3f},{hi:.3f}]", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 分析 4：0.1--1.5 Hz 漂移/非平稳性（长窗设置）
# --------------------------------------------------------------------------- #
def drift_stats(fdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, g in fdf.groupby("setting", sort=False):
        if setting == "T3_NW6":
            continue  # 基线未缓存段序列，漂移只在长窗下评估
        d = g[g["task"] == "EndotrachealSuction"]
        first, last, slope = d["_slow_first"], d["_slow_last"], d["_slow_slope"]
        diff = (last - first).to_numpy()
        valid = np.isfinite(diff) & (np.abs(first.to_numpy()) > 0)
        try:
            w = wilcoxon(diff[valid], alternative="two-sided")
            p_w = float(w.pvalue)
        except ValueError:
            p_w = np.nan
        med_slope = float(np.nanmedian(slope))
        # 斜率是否随被试水平系统偏移（Spearman：水平高者漂移方向是否不同）
        sp_r, sp_p = spearmanr(first.to_numpy(), slope.to_numpy())
        rows.append(dict(
            setting=setting, n=int(valid.sum()),
            first_half_median=float(np.nanmedian(first)),
            second_half_median=float(np.nanmedian(last)),
            median_diff=float(np.median(diff[valid])),
            wilcoxon_p=p_w, median_slope_per_seg=med_slope,
            slope_vs_level_rho=float(sp_r), slope_vs_level_p=float(sp_p),
            median_rel_range=float(np.nanmedian(d["_slow_relrange"]))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    # 1) 提取各长窗设置特征；基线读取已有 features.csv
    frames = []
    base = pd.read_csv(C.RESULTS / "features.csv")
    base = base[base["task"].isin(["EndotrachealSuction", "Positioning"])].copy()
    base["setting"] = "T3_NW6"
    base["n_seg"] = base.get("n_seg_pre", np.nan)
    frames.append(base)
    for name, seg_s, nw, _bw, _is_base in SETTINGS:
        if _is_base:
            continue
        df = extract_setting(name, seg_s, nw)
        df.to_csv(C.RESULTS / f"spectral_sensitivity_features_{name}.csv",
                  index=False)
        frames.append(df)
    fdf = pd.concat(frames, ignore_index=True, sort=False)
    keep = (["participant_id", "task", "setting", "n_seg",
             "post_bps_ge7", "post_bps_gt3",
             "pre_sed_mean", "pre_nol_mean", "pre_bps"]
            + [f"F1_norm_{b}" for b in C.BANDS_BJA]
            + [f"F2_norm_{b}" for b in C.BANDS_CANON]
            + [f"F2_logabs_{b}" for b in C.BANDS_CANON]
            + ["F2_spen", "F2_sef95"]
            + ["_slow_first", "_slow_last", "_slow_slope", "_slow_relrange"])
    fdf[[c for c in keep if c in fdf.columns]].to_csv(
        C.RESULTS / "spectral_sensitivity_features.csv", index=False)

    # 2) 方向性
    dir_df = directionality(fdf)
    dir_df.to_csv(C.RESULTS / "spectral_sensitivity_directionality.csv",
                  index=False)

    # 3) 模型
    mod_df = run_models(fdf)
    mod_df.to_csv(C.RESULTS / "spectral_sensitivity_models.csv", index=False)

    # 4) 跨操作迁移
    tr_df = run_transfer(fdf)
    tr_df.to_csv(C.RESULTS / "spectral_sensitivity_transfer.csv", index=False)

    # 5) 低频漂移
    dr_df = drift_stats(fdf)
    dr_df.to_csv(C.RESULTS / "spectral_sensitivity_drift.csv", index=False)

    # 6) 汇总
    write_summary(dir_df, mod_df, tr_df, dr_df)
    print(f"done in {time.time()-t0:.1f}s")


def write_summary(dir_df, mod_df, tr_df, dr_df) -> None:
    lines = ["# 频谱敏感性分析汇总（3s/4Hz 基线 vs 长窗匹配带宽）", ""]
    lines.append("## 方向性复现（吸痰，F1 三频段，Mann-Whitney）")
    lines.append(dir_df.to_string(index=False))
    lines.append("")
    lines.append("## 模型 AUROC（吸痰，嵌套 LOPO）")
    lines.append(mod_df.to_string(index=False))
    lines.append("")
    lines.append("## 跨操作迁移（F2，吸痰 ge7 -> 体位摆放 ge7）")
    lines.append(tr_df.to_string(index=False) if len(tr_df) else "（无）")
    lines.append("")
    lines.append("## 0.1--1.5 Hz 漂移/非平稳性（长窗）")
    lines.append(dr_df.to_string(index=False) if len(dr_df) else "（无）")
    out = C.RESULTS / "spectral_sensitivity_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
