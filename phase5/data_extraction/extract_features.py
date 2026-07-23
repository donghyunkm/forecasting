#!/usr/bin/env python3
"""
Feature Extraction Pipeline for MIMIC-III

Extracts feature vectors containing:
  - 7 correlation pairs (Pearson r computed across sub-windows within a configurable window)
  - 4 vital signs (mean BP, pulse, SpO2, respiratory rate) from numerics records

Usage:
  # Single-job (all patients):
  python3 extract_features.py --config ../config/pipeline_config.yaml

  # Parallel (SLURM array):
  python3 extract_features.py --config ../config/pipeline_config.yaml --job-idx $SLURM_ARRAY_TASK_ID --num-jobs 50

Output per job:
  output/part_XXX/features.npy       (N, 11) — 7 correlations + 4 vital signs
  output/part_XXX/patient_ids.npy    (N,)
  output/part_XXX/seg_names.npy      (N,)
  output/part_XXX/window_times.npy   (N,) — seconds since segment start
  output/part_XXX/metadata.json      summary statistics
"""

import argparse
import csv
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import wfdb
import yaml
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.stats import pearsonr


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_params(cfg: dict) -> dict:
    """Derive numeric parameters from config."""
    fs = cfg["waveform_signals"]["sampling_rate"]
    win_min = cfg["window"]["duration_min"]
    stride_min = cfg["window"]["stride_min"]
    sub_dur = cfg["subwindow"]["duration_sec"]
    sub_stride = cfg["subwindow"]["stride_sec"]

    win_samples = int(win_min * 60 * fs)
    stride_samples = int(stride_min * 60 * fs)
    sub_win_samples = int(sub_dur * fs)
    sub_stride_samples = int(sub_stride * fs)
    n_subwindows = (win_samples - sub_win_samples) // sub_stride_samples + 1

    # Build feature name → index mapping
    feature_names = cfg["feature_names"]
    fi = {f: i for i, f in enumerate(feature_names)}

    # Resolve correlation pair indices
    pairs = cfg["correlation_pairs"]
    pair_indices = [(fi[a], fi[b]) for a, b in pairs]

    return {
        "fs": fs,
        "win_samples": win_samples,
        "stride_samples": stride_samples,
        "sub_win_samples": sub_win_samples,
        "sub_stride_samples": sub_stride_samples,
        "n_subwindows": n_subwindows,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "fi": fi,
        "pair_indices": pair_indices,
        "pair_names": pairs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def interp_nans(col: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs; endpoints filled with nearest valid."""
    nans = np.isnan(col)
    if not nans.any():
        return col
    valid = ~nans
    if not valid.any():
        col[:] = 0.0
        return col
    idx = np.arange(len(col))
    col[nans] = np.interp(idx[nans], idx[valid], col[valid])
    return col


def clip_or_nan(value: float, lo: float, hi) -> float:
    """Return NaN if value is outside [lo, hi]."""
    if np.isnan(value):
        return np.nan
    if lo is not None and value < lo:
        return np.nan
    if hi is not None and value > hi:
        return np.nan
    return value


def dominant_peaks(signal: np.ndarray, distance: int):
    """Return peaks using whichever polarity yields more detections."""
    pos, _ = find_peaks(signal, distance=distance)
    neg, _ = find_peaks(-signal, distance=distance)
    if len(pos) >= len(neg):
        return pos, 1
    return neg, -1


# Physiological bounds (values outside → NaN)
BOUNDS = {
    "HR": (20, 300),
    "RR": (3, 60),
    "SBP": (30, 300),
    "DBP": (10, 200),
    "PP": (5, 200),
    "MAP": (20, 200),
    "ABP_area": (0, 50),
    "PLETH_ACDC": (0, 50),
    "PLETH_amp": (0, 10),
    "ECG_Ramp": (0, 5),
    "HRV_RMSSD": (0, 500),
    "HR_range": (0, 100),
    "ShockIdx": (0, 5),
    "PPV": (0, 100),
    "PVI": (0, 100),
    "PTT": (50, 250),
    "dPdt_max": (10, 5000),
    "ABP_tau": (0.05, 10),
    "RESP_amp": (0, None),
}

# Peak detection distances
HR_MIN_DIST = None   # Set in main based on fs
RR_MIN_DIST = None
ABP_MIN_DIST = None


# ══════════════════════════════════════════════════════════════════════════════
# Sub-window Feature Computation (19 features per sub-window)
# ══════════════════════════════════════════════════════════════════════════════

def compute_subwindow_features(ii_w, pleth_w, resp_w, abp_w, fs, ecg_sign,
                               nan_frac_limit=0.10):
    """
    Compute 19 physiological features from a single sub-window of waveform data.
    Matches the reference implementation in build_windows_m3_120s.py.

    Parameters
    ----------
    ii_w : ndarray — ECG lead II segment
    pleth_w : ndarray — plethysmography segment
    resp_w : ndarray — respiratory impedance segment
    abp_w : ndarray — arterial blood pressure segment
    fs : int — sampling rate
    ecg_sign : int — polarity of ECG (+1 or -1)
    nan_frac_limit : float — max NaN fraction allowed

    Returns
    -------
    ndarray of shape (19,) — feature values (NaN where invalid)
    """
    win_sec = len(ii_w) / fs
    hr_dist = int(0.4 * fs)
    rr_dist = int(1.5 * fs)
    abp_dist = int(0.4 * fs)

    out = np.full(19, np.nan, dtype=np.float32)

    # Check NaN fractions
    if np.isnan(ii_w).mean() > nan_frac_limit:
        return out
    if np.isnan(pleth_w).mean() > nan_frac_limit:
        return out
    if np.isnan(resp_w).mean() > nan_frac_limit:
        return out
    if np.isnan(abp_w).mean() > nan_frac_limit:
        return out

    # Interpolate NaNs
    ii_w = interp_nans(ii_w.copy())
    pleth_w = interp_nans(pleth_w.copy())
    resp_w = interp_nans(resp_w.copy())
    abp_w = interp_nans(abp_w.copy())

    # Peak detection
    peaks_r, _ = find_peaks(ecg_sign * ii_w, distance=hr_dist)
    peaks_s, _ = find_peaks(abp_w, distance=abp_dist)
    troughs_d, _ = find_peaks(-abp_w, distance=abp_dist)
    peaks_p, _ = find_peaks(pleth_w, distance=abp_dist)
    troughs_p, _ = find_peaks(-pleth_w, distance=abp_dist)
    peaks_resp, resp_sign = dominant_peaks(resp_w, rr_dist)
    troughs_resp, _ = find_peaks(-resp_sign * resp_w, distance=rr_dist)

    # ── Feature 0: HR (bpm) ──
    hr = clip_or_nan(len(peaks_r) * (60.0 / win_sec), *BOUNDS["HR"])
    out[0] = hr

    # ── Feature 1: RR (breaths/min) ──
    out[1] = clip_or_nan(len(peaks_resp) * (60.0 / win_sec), *BOUNDS["RR"])

    # ── Features 2-4: SBP, DBP, PP ──
    if len(peaks_s) > 0 and len(troughs_d) > 0:
        sbp = clip_or_nan(float(np.median(abp_w[peaks_s])), *BOUNDS["SBP"])
        dbp = clip_or_nan(float(np.median(abp_w[troughs_d])), *BOUNDS["DBP"])
        if not (np.isnan(sbp) or np.isnan(dbp)):
            pp = clip_or_nan(sbp - dbp, *BOUNDS["PP"])
            out[2] = sbp
            out[3] = dbp
            out[4] = pp

    # ── Feature 5: MAP ──
    out[5] = clip_or_nan(float(np.mean(abp_w)), *BOUNDS["MAP"])

    # ── Feature 6: ABP_area (trough-to-trough beat area) ──
    if len(troughs_d) >= 2:
        areas = []
        for i in range(len(troughs_d) - 1):
            beat = abp_w[troughs_d[i]:troughs_d[i + 1]]
            area = float(np.sum(np.maximum(beat - abp_w[troughs_d[i]], 0.0)) / fs)
            if area > 0:
                areas.append(area)
        if areas:
            out[6] = clip_or_nan(float(np.median(areas)), *BOUNDS["ABP_area"])

    # ── Feature 7: PLETH_ACDC ──
    if len(peaks_p) > 0 and len(troughs_p) > 0:
        ac = float(np.median(pleth_w[peaks_p]) - np.median(pleth_w[troughs_p]))
        dc = float(np.mean(pleth_w))
        if dc > 1e-6 and ac >= 0:
            out[7] = clip_or_nan(ac / dc, *BOUNDS["PLETH_ACDC"])

    # ── Feature 8: PLETH_amp ──
    if len(peaks_p) > 0 and len(troughs_p) > 0:
        amp = float(np.median(pleth_w[peaks_p]) - np.median(pleth_w[troughs_p]))
        out[8] = clip_or_nan(max(0.0, amp), *BOUNDS["PLETH_amp"])

    # ── Feature 9: ECG_Ramp ──
    if len(peaks_r) > 0:
        baseline = float(np.mean(ii_w))
        if ecg_sign == 1:
            r_amp = float(np.median(ii_w[peaks_r])) - baseline
        else:
            r_amp = baseline - float(np.median(ii_w[peaks_r]))
        out[9] = clip_or_nan(max(0.0, r_amp), *BOUNDS["ECG_Ramp"])

    # ── Feature 10: HRV_RMSSD ──
    if len(peaks_r) >= 3:
        rr_ms = np.diff(peaks_r).astype(float) / fs * 1000
        rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        out[10] = clip_or_nan(rmssd, *BOUNDS["HRV_RMSSD"])

    # ── Feature 11: HR_range (ectopic-removed) ──
    if len(peaks_r) >= 3:
        rr_s = np.diff(peaks_r).astype(float) / fs
        med_rr = np.median(rr_s)
        rr_clean = rr_s[(rr_s > 0.5 * med_rr) & (rr_s < 2.0 * med_rr)]
        if len(rr_clean) >= 2:
            hr_inst = 60.0 / rr_clean
            out[11] = clip_or_nan(float(np.max(hr_inst) - np.min(hr_inst)),
                                  *BOUNDS["HR_range"])

    # ── Feature 12: ShockIdx ──
    if not np.isnan(out[0]) and not np.isnan(out[2]) and out[2] > 0:
        out[12] = clip_or_nan(out[0] / out[2], *BOUNDS["ShockIdx"])

    # ── Feature 13: PPV (respiratory-cycle-based) ──
    per_beat_pp = []
    if len(peaks_s) >= 4 and len(troughs_d) >= 4:
        for s_idx in peaks_s:
            prec = troughs_d[troughs_d < s_idx]
            if len(prec) == 0:
                continue
            pp_b = float(abp_w[s_idx] - abp_w[prec[-1]])
            if pp_b > 0:
                per_beat_pp.append((int(s_idx), pp_b))

    if len(per_beat_pp) >= 4 and len(peaks_resp) >= 3:
        bp_idx = np.array([x[0] for x in per_beat_pp])
        bp_pp = np.array([x[1] for x in per_beat_pp])
        ppv_cycles = []
        for ci in range(len(peaks_resp) - 1):
            mask = (bp_idx >= peaks_resp[ci]) & (bp_idx < peaks_resp[ci + 1])
            cyc = bp_pp[mask]
            if len(cyc) >= 2:
                pp_mid = (float(np.max(cyc)) + float(np.min(cyc))) / 2
                if pp_mid > 0:
                    ppv_cycles.append(
                        (float(np.max(cyc)) - float(np.min(cyc))) / pp_mid * 100)
        if len(ppv_cycles) >= 2:
            out[13] = clip_or_nan(float(np.mean(ppv_cycles)), *BOUNDS["PPV"])

    # ── Feature 14: PVI (respiratory-cycle-based) ──
    per_beat_pa = []
    if len(peaks_p) >= 4 and len(troughs_p) >= 4:
        for p_idx in peaks_p:
            prec = troughs_p[troughs_p < p_idx]
            if len(prec) == 0:
                continue
            amp_b = float(pleth_w[p_idx] - pleth_w[prec[-1]])
            if amp_b > 0:
                per_beat_pa.append((int(p_idx), amp_b))

    if len(per_beat_pa) >= 4 and len(peaks_resp) >= 3:
        pa_idx = np.array([x[0] for x in per_beat_pa])
        pa_amp = np.array([x[1] for x in per_beat_pa])
        pvi_cycles = []
        for ci in range(len(peaks_resp) - 1):
            mask = (pa_idx >= peaks_resp[ci]) & (pa_idx < peaks_resp[ci + 1])
            cyc = pa_amp[mask]
            if len(cyc) >= 2:
                max_amp = float(np.max(cyc))
                if max_amp > 0:
                    pvi_cycles.append(
                        (max_amp - float(np.min(cyc))) / max_amp * 100)
        if len(pvi_cycles) >= 2:
            out[14] = clip_or_nan(float(np.mean(pvi_cycles)), *BOUNDS["PVI"])

    # ── Feature 15: PTT (R-peak → ABP diastolic foot, 4-32 samples) ──
    if len(peaks_r) > 0 and len(troughs_d) > 0:
        ptt_list = []
        for r_idx in peaks_r:
            cands = troughs_d[(troughs_d > r_idx + 4) & (troughs_d < r_idx + 32)]
            if len(cands) > 0:
                ptt_list.append(float(cands[0] - r_idx) / fs * 1000)
        if ptt_list:
            out[15] = clip_or_nan(float(np.median(ptt_list)), *BOUNDS["PTT"])

    # ── Feature 16: dPdt_max (per-beat upstroke, smoothed) ──
    if len(peaks_s) > 0 and len(troughs_d) > 0:
        abp_sm = uniform_filter1d(abp_w.astype(np.float64), size=3)
        grad_abp = np.diff(abp_sm) * fs
        dpdt_list = []
        for s_idx in peaks_s:
            prec = troughs_d[troughs_d < s_idx]
            if len(prec) == 0:
                continue
            t_idx = int(prec[-1])
            if s_idx > t_idx + 1:
                seg_grad = grad_abp[t_idx:s_idx]
                if len(seg_grad) > 0:
                    dpdt_list.append(float(np.max(seg_grad)))
        if dpdt_list:
            out[16] = clip_or_nan(float(np.median(dpdt_list)), *BOUNDS["dPdt_max"])

    # ── Feature 17: ABP_tau (dicrotic notch → diastolic log-linear fit) ──
    if len(peaks_s) > 0 and len(troughs_d) > 0:
        tau_list = []
        for s_idx in peaks_s:
            foll = troughs_d[troughs_d > s_idx]
            if len(foll) == 0:
                continue
            t_end = int(foll[0])
            if t_end - s_idx < 8:
                continue
            # Find dicrotic notch region
            notch_lo = s_idx + 5
            notch_hi = min(s_idx + max(8, (t_end - s_idx) // 3), t_end - 3)
            fit_start = notch_lo
            if notch_hi > notch_lo + 2:
                notch_cands, _ = find_peaks(-abp_w[notch_lo:notch_hi], distance=2)
                if len(notch_cands) > 0:
                    fit_start = notch_lo + int(notch_cands[0])
            seg = abp_w[fit_start:t_end + 1].astype(np.float64)
            if len(seg) < 5:
                continue
            y = np.maximum(seg, 1.0)
            t_a = np.arange(len(seg)) / fs
            try:
                slope, _ = np.polyfit(t_a, np.log(y), 1)
                if slope < 0:
                    tau_list.append(-1.0 / slope)
            except Exception:
                pass
        if tau_list:
            out[17] = clip_or_nan(float(np.median(tau_list)), *BOUNDS["ABP_tau"])

    # ── Feature 18: RESP_amp ──
    if len(peaks_resp) > 0 and len(troughs_resp) > 0:
        resp_peaks_vals = resp_sign * resp_w[peaks_resp]
        resp_troughs_vals = resp_sign * resp_w[troughs_resp]
        resp_amp = float(np.median(resp_peaks_vals) - np.median(resp_troughs_vals))
        out[18] = clip_or_nan(max(0.0, resp_amp), *BOUNDS["RESP_amp"])

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Feature Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_window_correlations(seg_signals, anchor_start, params, cfg):
    """
    Compute 7 correlation features for one window.

    Slides sub-windows across the extraction window, computes 19 features per
    sub-window, then computes Pearson r for each of the 7 configured pairs
    across the sub-window time series.

    Returns
    -------
    ndarray of shape (7,) — correlation values (NaN where insufficient data)
    """
    fs = params["fs"]
    win_samples = params["win_samples"]
    sub_win_samples = params["sub_win_samples"]
    sub_stride_samples = params["sub_stride_samples"]
    n_subwindows = params["n_subwindows"]
    pair_indices = params["pair_indices"]
    nan_frac = cfg["quality"]["max_nan_fraction"]
    min_corr = cfg["quality"]["min_corr_samples"]

    # Extract window signals
    anchor_end = anchor_start + win_samples
    ii_win = seg_signals["II"][anchor_start:anchor_end]
    pleth_win = seg_signals["PLETH"][anchor_start:anchor_end]
    resp_win = seg_signals["RESP"][anchor_start:anchor_end]
    abp_win = seg_signals["ABP"][anchor_start:anchor_end]

    # Determine ECG polarity from whole window
    ii_clean = np.where(np.isnan(ii_win), np.nanmedian(ii_win), ii_win).copy()
    _, ecg_sign = dominant_peaks(interp_nans(ii_clean), int(0.4 * fs))

    # Compute feature trajectories: (19, n_subwindows)
    feat_traj = np.full((19, n_subwindows), np.nan, dtype=np.float32)

    for k in range(n_subwindows):
        s = k * sub_stride_samples
        e = s + sub_win_samples

        feat_traj[:, k] = compute_subwindow_features(
            ii_win[s:e], pleth_win[s:e], resp_win[s:e], abp_win[s:e],
            fs, ecg_sign, nan_frac
        )

    # Compute correlations for each pair
    n_pairs = len(pair_indices)
    correlations = np.full(n_pairs, np.nan, dtype=np.float32)

    for p_idx, (fi_a, fi_b) in enumerate(pair_indices):
        feat_a = feat_traj[fi_a, :]
        feat_b = feat_traj[fi_b, :]
        valid = np.isfinite(feat_a) & np.isfinite(feat_b)
        if np.sum(valid) >= min_corr:
            try:
                r, _ = pearsonr(feat_a[valid], feat_b[valid])
                correlations[p_idx] = r
            except Exception:
                pass

    return correlations


# ══════════════════════════════════════════════════════════════════════════════
# Vital Signs Extraction from Numerics Records
# ══════════════════════════════════════════════════════════════════════════════

def load_numerics_record(stay_dir: Path, stay_id: str):
    """
    Attempt to load the numerics record for a stay.
    Returns (record, sig_dict) or (None, None) if unavailable.

    sig_dict maps signal name → column index in p_signal.
    """
    numerics_id = stay_id + "n"
    numerics_path = stay_dir / numerics_id
    if not numerics_path.with_suffix(".hea").exists():
        return None, None
    try:
        rec = wfdb.rdrecord(str(numerics_path))
        sig_dict = {name: i for i, name in enumerate(rec.sig_name)}
        return rec, sig_dict
    except Exception:
        return None, None


def extract_vital_signs_window(numerics_rec, sig_dict, window_start_sec,
                                window_dur_sec, vital_cfg, agg_method="median"):
    """
    Extract vital signs from a numerics record for a time window.

    Parameters
    ----------
    numerics_rec : wfdb Record with p_signal
    sig_dict : dict mapping signal name → column index
    window_start_sec : float — window start relative to record start (seconds)
    window_dur_sec : float — window duration (seconds)
    vital_cfg : list of dicts with 'name' and 'fallback' keys
    agg_method : 'median' or 'mean'

    Returns
    -------
    ndarray of shape (4,) — [mean_BP, pulse, SpO2, resp_rate]
    """
    n_vitals = len(vital_cfg)
    vitals = np.full(n_vitals, np.nan, dtype=np.float32)

    if numerics_rec is None:
        return vitals

    fs_num = numerics_rec.fs  # typically 1/60 Hz (one sample per minute)
    if fs_num <= 0:
        return vitals

    # Convert time window to sample indices
    samp_start = int(window_start_sec * fs_num)
    samp_end = int((window_start_sec + window_dur_sec) * fs_num)
    samp_start = max(0, samp_start)
    samp_end = min(numerics_rec.sig_len, samp_end)

    if samp_end <= samp_start:
        return vitals

    agg_fn = np.nanmedian if agg_method == "median" else np.nanmean

    for v_idx, v_info in enumerate(vital_cfg):
        sig_name = v_info["name"]
        fallback = v_info.get("fallback")

        col_idx = _find_signal(sig_dict, sig_name)
        if col_idx is None and fallback:
            col_idx = _find_signal(sig_dict, fallback)
        if col_idx is None:
            continue

        values = numerics_rec.p_signal[samp_start:samp_end, col_idx]
        valid = values[~np.isnan(values)]
        if len(valid) > 0:
            vitals[v_idx] = agg_fn(valid)

    return vitals


# Signal name variations in MIMIC-III numerics records
_SIGNAL_ALIASES = {
    "ABPMean": ["ABPMean", "ABP Mean", "ABP MEAN"],
    "NBPMean": ["NBPMean", "NBP Mean", "NBP MEAN"],
    "PULSE":   ["PULSE", "Pulse"],
    "HR":      ["HR"],
    "SpO2":    ["SpO2", "%SpO2"],
    "RESP":    ["RESP", "Resp"],
}


def _find_signal(sig_dict: dict, name: str):
    """Look up a signal name with common alias variations."""
    # Direct match first
    if name in sig_dict:
        return sig_dict[name]
    # Check aliases
    aliases = _SIGNAL_ALIASES.get(name, [])
    for alias in aliases:
        if alias in sig_dict:
            return sig_dict[alias]
    # Case-insensitive fallback
    name_lower = name.lower().replace(" ", "")
    for key, idx in sig_dict.items():
        if key.lower().replace(" ", "") == name_lower:
            return idx
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Data Scanning (Pass 1: find qualifying segments)
# ══════════════════════════════════════════════════════════════════════════════

def iter_records(waves_dir: Path):
    """Yield (patient_id, stay_dir, stay_id) from RECORDS-waveforms."""
    records_file = waves_dir / "RECORDS-waveforms"
    with open(records_file) as f:
        for line in f:
            rec = line.strip()
            if not rec:
                continue
            parts = rec.split("/")
            patient_id = parts[1]
            stay_dir = waves_dir / parts[0] / parts[1]
            stay_id = parts[2]
            yield patient_id, stay_dir, stay_id


def get_qualifying_segments(stay_dir: Path, stay_id: str, required_signals: set,
                            min_seg_samples: int):
    """
    Return [(seg_name, sig_len), ...] for segments with all required signals
    and duration >= min_seg_samples.
    """
    try:
        master = wfdb.rdheader(str(stay_dir / stay_id))
    except Exception:
        return []
    result = []
    for seg, length in zip(master.seg_name, master.seg_len):
        if seg == "~" or length < min_seg_samples or "layout" in seg:
            continue
        try:
            hdr = wfdb.rdheader(str(stay_dir / seg))
            if required_signals.issubset(set(hdr.sig_name)):
                result.append((seg, length))
        except Exception:
            continue
    return result


def scan_patients(cfg: dict, params: dict) -> dict:
    """
    Scan RECORDS-waveforms and return dict:
      {patient_id: [(stay_dir, seg_name, sig_len), ...]}
    """
    waves_dir = Path(cfg["paths"]["waveforms_dir"])
    required = set(cfg["waveform_signals"]["required"])
    min_seg_samples = params["win_samples"]  # Must be at least one window long

    print("Pass 1: Scanning segment headers...")
    patient_segs = {}
    n_stays = 0
    for patient_id, stay_dir, stay_id in iter_records(waves_dir):
        n_stays += 1
        if n_stays % 2000 == 0:
            print(f"  [{n_stays:6d} stays] {len(patient_segs)} patients with qualifying segments",
                  flush=True)
        for seg_name, sig_len in get_qualifying_segments(stay_dir, stay_id, required,
                                                          min_seg_samples):
            patient_segs.setdefault(patient_id, []).append((stay_dir, seg_name, sig_len))

    n_segs = sum(len(v) for v in patient_segs.values())
    print(f"  Done: {n_stays} stays → {n_segs} qualifying segments "
          f"across {len(patient_segs)} patients")
    return patient_segs


# ══════════════════════════════════════════════════════════════════════════════
# Window Extraction (Pass 2)
# ══════════════════════════════════════════════════════════════════════════════

def process_patient(patient_id: str, segments: list, cfg: dict, params: dict,
                    rng: random.Random) -> list:
    """
    Extract feature vectors for one patient across all their qualifying segments.

    Returns list of dicts:
      {"features": ndarray(11,), "patient_id": str, "seg_name": str, "window_sec": float}
    """
    fs = params["fs"]
    win_samples = params["win_samples"]
    stride_samples = params["stride_samples"]
    max_windows = cfg["processing"]["max_windows_per_patient"]
    required_sigs = cfg["waveform_signals"]["required"]
    vital_cfg = cfg["vital_signs"]["signals"]
    agg_method = cfg["vital_signs"]["aggregation"]
    win_dur_sec = cfg["window"]["duration_min"] * 60

    results = []

    # Cache loaded segments and numerics per stay_dir
    loaded_numerics = {}

    for stay_dir, seg_name, sig_len in segments:
        # Load waveform segment
        try:
            rec = wfdb.rdrecord(str(stay_dir / seg_name))
        except Exception:
            continue

        # Build signal dict (name → array)
        seg_signals = {}
        for sname in required_sigs:
            try:
                idx = rec.sig_name.index(sname)
                seg_signals[sname] = rec.p_signal[:, idx].astype(np.float64)
            except (ValueError, IndexError):
                break
        if len(seg_signals) < len(required_sigs):
            continue

        # Load numerics for vital signs (best effort)
        # Numerics record corresponds to the stay (master record), not the segment.
        # The stay_id is found by reading RECORDS-waveforms: the path gives us the
        # master record name (e.g., p000020-2183-04-28-17-47), numerics = same + "n"
        stay_dir_key = str(stay_dir)
        if stay_dir_key not in loaded_numerics:
            # Find the master record (stay) for this patient directory
            # by looking for the numerics header file
            numerics_rec, sig_dict = None, None
            try:
                # List .hea files ending with 'n.hea' in stay_dir
                n_heas = list(stay_dir.glob("*n.hea"))
                if n_heas:
                    # Use the first numerics record found
                    numerics_id = n_heas[0].stem  # e.g., "p000020-2183-04-28-17-47n"
                    nrec = wfdb.rdrecord(str(stay_dir / numerics_id))
                    numerics_rec = nrec
                    sig_dict = {name: i for i, name in enumerate(nrec.sig_name)}
            except Exception:
                pass
            loaded_numerics[stay_dir_key] = (numerics_rec, sig_dict)

        numerics_rec, sig_dict = loaded_numerics[stay_dir_key]

        # Determine time offset of this segment within the stay
        # (segments are sub-records; we need to find where this segment
        # starts relative to the numerics record start)
        # For simplicity, use the segment's sample position directly:
        # the waveform segment starts at some offset from the master record.
        # Since numerics and waveforms share the same time base, we compute
        # the segment's offset by reading the master header.
        seg_offset_sec = 0.0
        try:
            # Find master record for this stay
            master_heas = [f for f in stay_dir.glob("*.hea")
                           if not f.stem.endswith("n") and "-" in f.stem]
            if master_heas:
                master_id = master_heas[0].stem
                master = wfdb.rdheader(str(stay_dir / master_id))
                cumulative = 0
                for seg, length in zip(master.seg_name, master.seg_len):
                    if seg == seg_name:
                        seg_offset_sec = cumulative / fs
                        break
                    cumulative += length
        except Exception:
            pass

        # Enumerate anchor positions
        margin = win_samples // 2
        anchors = list(range(margin, sig_len - margin, stride_samples))

        if max_windows and len(anchors) > max_windows:
            anchors = sorted(rng.sample(anchors, max_windows))

        for anchor in anchors:
            anchor_start = anchor - win_samples // 2

            # Compute 7 correlations from waveform features
            correlations = compute_window_correlations(
                seg_signals, anchor_start, params, cfg
            )

            # Extract 4 vital signs from numerics
            # Window start time relative to the numerics record start
            window_start_sec = seg_offset_sec + (anchor_start / fs)
            vitals = extract_vital_signs_window(
                numerics_rec, sig_dict if sig_dict else {},
                window_start_sec, win_dur_sec, vital_cfg, agg_method
            )

            # Combine: [7 correlations, 4 vital signs] = 11-dim vector
            feature_vec = np.concatenate([correlations, vitals])
            results.append({
                "features": feature_vec,
                "patient_id": patient_id,
                "seg_name": seg_name,
                "window_sec": window_start_sec,
            })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MIMIC-III Feature Extraction Pipeline")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--job-idx", type=int, default=None,
                        help="Job index for SLURM array (0-based)")
    parser.add_argument("--num-jobs", type=int, default=None,
                        help="Total number of parallel jobs")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    params = setup_params(cfg)

    print("=" * 80)
    print("MIMIC-III FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"  Window duration:  {cfg['window']['duration_min']} min")
    print(f"  Window stride:    {cfg['window']['stride_min']} min")
    print(f"  Sub-windows:      {params['n_subwindows']} × {cfg['subwindow']['duration_sec']}s")
    print(f"  Correlation pairs: {len(params['pair_indices'])}")
    print(f"  Vital signs:      {len(cfg['vital_signs']['signals'])}")
    print(f"  Output dim:       {len(params['pair_indices']) + len(cfg['vital_signs']['signals'])}")
    print()

    # Scan for qualifying patients/segments
    patient_segs = scan_patients(cfg, params)
    patient_ids_sorted = sorted(patient_segs.keys())

    # Slice for parallel processing
    if args.job_idx is not None and args.num_jobs is not None:
        job_idx = args.job_idx
        num_jobs = args.num_jobs
        # Distribute patients across jobs
        patients_slice = [p for i, p in enumerate(patient_ids_sorted)
                          if i % num_jobs == job_idx]
        print(f"\n  Job {job_idx}/{num_jobs}: processing {len(patients_slice)} patients")
    else:
        patients_slice = patient_ids_sorted
        job_idx = 0
        print(f"\n  Single job: processing {len(patients_slice)} patients")

    # Process patients
    rng = random.Random(cfg["processing"]["random_seed"])
    all_results = []

    for p_idx, patient_id in enumerate(patients_slice):
        if p_idx % 10 == 0:
            print(f"  [{p_idx:5d}/{len(patients_slice)}] Patient {patient_id} "
                  f"({len(all_results)} windows so far)", flush=True)

        segs = patient_segs[patient_id]
        patient_results = process_patient(patient_id, segs, cfg, params, rng)
        all_results.extend(patient_results)

    # Save outputs
    output_dir = Path(cfg["paths"]["output_dir"])
    if args.job_idx is not None:
        part_dir = output_dir / f"part_{job_idx:03d}"
    else:
        part_dir = output_dir
    part_dir.mkdir(parents=True, exist_ok=True)

    n_windows = len(all_results)
    print(f"\n  Extracted {n_windows} windows total")

    if n_windows == 0:
        print("  WARNING: No windows extracted. Check data paths and requirements.")
        return

    # Stack features
    features = np.array([r["features"] for r in all_results], dtype=np.float32)
    patient_ids_arr = np.array([r["patient_id"] for r in all_results])
    seg_names_arr = np.array([r["seg_name"] for r in all_results])
    window_times = np.array([r["window_sec"] for r in all_results], dtype=np.float64)

    np.save(part_dir / "features.npy", features)
    np.save(part_dir / "patient_ids.npy", patient_ids_arr)
    np.save(part_dir / "seg_names.npy", seg_names_arr)
    np.save(part_dir / "window_times.npy", window_times)

    # Save metadata
    n_valid_corr = np.sum(np.isfinite(features[:, :7]), axis=0)
    n_valid_vitals = np.sum(np.isfinite(features[:, 7:]), axis=0)

    pair_names = [f"{a} × {b}" for a, b in cfg["correlation_pairs"]]
    vital_names = [v["name"] for v in cfg["vital_signs"]["signals"]]

    metadata = {
        "n_windows": n_windows,
        "n_patients": len(set(patient_ids_arr)),
        "feature_dim": features.shape[1],
        "feature_names": pair_names + vital_names,
        "correlation_pairs": pair_names,
        "vital_signs": vital_names,
        "window_duration_min": cfg["window"]["duration_min"],
        "valid_counts": {
            "correlations": {name: int(c) for name, c in zip(pair_names, n_valid_corr)},
            "vital_signs": {name: int(c) for name, c in zip(vital_names, n_valid_vitals)},
        },
        "nan_fraction": float(np.isnan(features).sum() / features.size),
    }

    with open(part_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to: {part_dir}")
    print(f"    features.npy:     {features.shape}")
    print(f"    patient_ids.npy:  {patient_ids_arr.shape}")
    print(f"    seg_names.npy:    {seg_names_arr.shape}")
    print(f"    window_times.npy: {window_times.shape}")
    print(f"    NaN fraction:     {metadata['nan_fraction']:.3f}")
    print(f"\n  Feature columns:")
    for i, name in enumerate(pair_names + vital_names):
        valid_pct = 100 * np.sum(np.isfinite(features[:, i])) / n_windows
        print(f"    [{i:2d}] {name:30s} — {valid_pct:.1f}% valid")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
