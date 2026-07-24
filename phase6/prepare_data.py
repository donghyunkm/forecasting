#!/usr/bin/env python3
"""
Phase 6 Data Preparation: Correlation-only forecasting.

Data source: /gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/
  - corr_features_focused.npy  (564596, 7) — 7 pairwise correlations per window
  - patient_ids.npy            (564596,) — patient identifiers
  - seg_names.npy              (564596,) — segment identifiers
  - window_times.npy           (564596,) — timestamps (seconds)
  - block_start_times.npy      (564596,) — block start times (for continuity grouping)

Resolution: 2.5 minutes (150 seconds) between windows.
Continuous segments: 4092 series, avg 138 windows (~345 min), max 145 windows.

Input: 7 correlations (48 steps = 2h) + 1 time position = 8 channels
Output: 7 correlations (12 steps = 30min)
"""

import os
import json
import numpy as np
import torch
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_SOURCE = "/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/"
OUTPUT_DIR = "/gpfs/home/dk5565/forecasting/phase6/phase6_data/processed/"

WINDOW_TOTAL = 60       # 48 history + 12 forecast
WINDOW_HISTORY = 48     # 2h at 2.5-min resolution
WINDOW_FORECAST = 12    # 30min at 2.5-min resolution
STRIDE = 12             # 30-min stride
EXPECTED_DT = 150       # 2.5 minutes in seconds

NUM_CORR = 7            # 7 focused correlation pairs

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

CORRELATION_NAMES = [
    'PLETH_ACDC×PLETH_amp',
    'ABP_area×ABP_tau',
    'ABP_area×ShockIdx',
    'PLETH_amp×ShockIdx',
    'PLETH_ACDC×ShockIdx',
    'ShockIdx×ABP_tau',
    'PLETH_ACDC×ABP_tau',
]


def load_data():
    """Load arrays from data_m3_120s_prediction."""
    print("Loading data from data_m3_120s_prediction...")
    corr_features = np.load(os.path.join(DATA_SOURCE, "corr_features_focused.npy"))  # (N, 7)
    patient_ids = np.load(os.path.join(DATA_SOURCE, "patient_ids.npy"), allow_pickle=True)  # (N,)
    seg_names = np.load(os.path.join(DATA_SOURCE, "seg_names.npy"), allow_pickle=True)  # (N,)
    window_times = np.load(os.path.join(DATA_SOURCE, "window_times.npy"))  # (N,)
    block_start_times = np.load(os.path.join(DATA_SOURCE, "block_start_times.npy"))  # (N,)

    print(f"  Total windows: {len(corr_features):,}")
    print(f"  Feature shape: {corr_features.shape}")
    print(f"  Unique patients: {len(np.unique(patient_ids)):,}")
    print(f"  Unique segments: {len(np.unique(seg_names)):,}")

    return corr_features, patient_ids, seg_names, window_times, block_start_times


def group_into_continuous_segments(corr_features, patient_ids, seg_names, window_times, block_start_times):
    """Group windows into continuous time series using (seg_name, block_start_time).

    Each unique (seg_name, block_start_time) pair defines a contiguous run of windows.
    We verify monotonic time ordering and uniform 150s spacing within each group.
    """
    print("\nGrouping into continuous segments...")

    # Build groups by (seg_name, block_start_time) — these define contiguous series
    groups = defaultdict(list)
    for i in range(len(corr_features)):
        key = (str(seg_names[i]), float(block_start_times[i]))
        groups[key].append(i)

    # Sort each group by window_time and verify continuity
    continuous_segments = []
    skipped = 0

    for key, indices in groups.items():
        indices = np.array(indices)
        times = window_times[indices]
        order = np.argsort(times)
        indices = indices[order]
        times = times[order]

        # Verify uniform spacing
        if len(indices) > 1:
            dts = np.diff(times)
            if not np.all(np.abs(dts - EXPECTED_DT) < 1.0):
                # Split at gaps (shouldn't happen with block_start grouping, but be safe)
                split_points = np.where(np.abs(dts - EXPECTED_DT) >= 1.0)[0] + 1
                sub_segments = np.split(indices, split_points)
                for sub in sub_segments:
                    if len(sub) >= WINDOW_TOTAL:
                        pid = str(patient_ids[sub[0]])
                        continuous_segments.append((pid, sub))
                    else:
                        skipped += 1
                continue

        pid = str(patient_ids[indices[0]])
        if len(indices) >= WINDOW_TOTAL:
            continuous_segments.append((pid, indices))
        else:
            skipped += 1

    print(f"  Total continuous segments: {len(continuous_segments) + skipped:,}")
    print(f"  Segments ≥ {WINDOW_TOTAL} windows: {len(continuous_segments):,}")
    print(f"  Segments too short (skipped): {skipped:,}")

    # Length statistics
    lengths = [len(indices) for _, indices in continuous_segments]
    if lengths:
        print(f"  Length stats — min: {min(lengths)}, max: {max(lengths)}, "
              f"mean: {np.mean(lengths):.1f}, median: {np.median(lengths):.0f}")

    return continuous_segments


def create_sliding_windows(continuous_segments, corr_features):
    """Create sliding windows from continuous segments.

    Each window: (WINDOW_TOTAL, 7) correlations.
    No NaN filtering needed — corr_features_focused has 0% NaN.
    """
    print("\nCreating sliding windows...")
    windows = []
    patient_ids_out = []

    total_windows = 0

    for pid, indices in continuous_segments:
        seg_corr = corr_features[indices]  # (L, 7)
        n_windows = (len(indices) - WINDOW_TOTAL) // STRIDE + 1

        for i in range(n_windows):
            start = i * STRIDE
            window = seg_corr[start:start + WINDOW_TOTAL]  # (60, 7)

            # Double-check no NaN (should never happen)
            if np.any(np.isnan(window)):
                continue

            windows.append(window)
            patient_ids_out.append(pid)
            total_windows += 1

    print(f"  Valid windows: {total_windows:,}")

    return np.array(windows, dtype=np.float32), np.array(patient_ids_out)


def split_patients(patient_ids_out, seed=SEED):
    """Split unique patients into train/val/test sets."""
    print("\nSplitting patients 70/15/15...")
    unique_patients = np.unique(patient_ids_out)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])

    print(f"  Train patients: {len(train_patients)}")
    print(f"  Val patients: {len(val_patients)}")
    print(f"  Test patients: {len(test_patients)}")

    return train_patients, val_patients, test_patients


def compute_norm_stats(windows, patient_ids_out, train_patients):
    """Compute normalization statistics from training patients in Fisher z-space.

    Pipeline: clip correlations to ±0.9999 → arctanh (Fisher z-transform) → z-score normalize.
    """
    print("\nComputing normalization statistics from training set (Fisher z-space)...")
    train_mask = np.array([pid in train_patients for pid in patient_ids_out])
    train_data = windows[train_mask]  # (N_train, 60, 7) — raw correlations

    # Apply Fisher z-transform to training data
    train_z = np.arctanh(np.clip(train_data, -0.9999, 0.9999))

    # Compute mean and std in z-space
    flat = train_z.reshape(-1, NUM_CORR)
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)

    # Prevent division by zero
    stds[stds < 1e-8] = 1.0

    print(f"  Fisher z-space means: {means}")
    print(f"  Fisher z-space stds: {stds}")

    return means, stds


def normalize(windows, means, stds):
    """Apply Fisher z-transform then z-score normalize."""
    # Step 1: clip and apply arctanh (Fisher z-transform)
    z = np.arctanh(np.clip(windows, -0.9999, 0.9999))
    # Step 2: z-score normalize in Fisher z-space
    return (z - means[np.newaxis, np.newaxis, :]) / stds[np.newaxis, np.newaxis, :]


def build_tensors(windows_norm, patient_ids_out, split_patients_set, split_name):
    """Build tensors for a given split.

    Input: 48 steps × 8 (7 correlations + 1 time position)
    Output (target): 12 steps × 7 (7 correlations)
    """
    mask = np.array([pid in split_patients_set for pid in patient_ids_out])
    split_windows = windows_norm[mask]  # (N, 60, 7)
    N = split_windows.shape[0]

    if N == 0:
        print(f"  WARNING: {split_name} has 0 windows!")
        return None

    # Historical: first 48 steps, 7 correlations + 1 time position = 8 channels
    historical_corr = split_windows[:, :WINDOW_HISTORY, :]  # (N, 48, 7)

    # Time position for history: linear 0 → 0.75
    time_hist = np.linspace(0.0, 0.75, WINDOW_HISTORY, dtype=np.float32)
    time_hist = np.tile(time_hist, (N, 1))[:, :, np.newaxis]  # (N, 48, 1)

    historical_ts_numeric = np.concatenate([historical_corr, time_hist], axis=2)  # (N, 48, 8)

    # Future: time position only (known into the future)
    time_future = np.linspace(0.76, 1.0, WINDOW_FORECAST, dtype=np.float32)
    time_future = np.tile(time_future, (N, 1))[:, :, np.newaxis]  # (N, 12, 1)
    future_ts_numeric = time_future

    # Target: 7 correlations for future 12 steps (already normalized)
    target = split_windows[:, WINDOW_HISTORY:, :]  # (N, 12, 7)

    # Target mask: all ones since we filtered out NaN windows
    target_mask = np.ones((N, WINDOW_FORECAST, NUM_CORR), dtype=np.float32)

    # Static features: placeholder (structural requirement for TFT)
    static_feats_numeric = np.zeros((N, 1), dtype=np.float32)

    data_dict = {
        'static_feats_numeric': torch.from_numpy(static_feats_numeric),
        'historical_ts_numeric': torch.from_numpy(historical_ts_numeric.astype(np.float32)),
        'future_ts_numeric': torch.from_numpy(future_ts_numeric.astype(np.float32)),
        'target': torch.from_numpy(target.astype(np.float32)),
        'target_mask': torch.from_numpy(target_mask),
    }

    print(f"  {split_name}: {N:,} windows")
    print(f"    historical_ts_numeric: {data_dict['historical_ts_numeric'].shape}")
    print(f"    future_ts_numeric: {data_dict['future_ts_numeric'].shape}")
    print(f"    target: {data_dict['target'].shape}")

    return data_dict


def main():
    print("=" * 70)
    print("Phase 6 Data Preparation — Correlation-Only Forecasting")
    print("  Source: data_m3_120s_prediction (2.5-min resolution)")
    print("  Input:  48 steps (2h) × 7 correlations + time position")
    print("  Output: 12 steps (30min) × 7 correlations")
    print("=" * 70)

    # Load data
    corr_features, patient_ids, seg_names, window_times, block_start_times = load_data()

    # Group into continuous segments
    continuous_segments = group_into_continuous_segments(
        corr_features, patient_ids, seg_names, window_times, block_start_times)

    # Create sliding windows
    windows, patient_ids_out = create_sliding_windows(continuous_segments, corr_features)

    if len(windows) == 0:
        print("ERROR: No valid windows found! Check data.")
        return

    print(f"\n  Window shape: {windows.shape}  (expected: (N, {WINDOW_TOTAL}, 7))")

    # Split patients
    train_patients, val_patients, test_patients = split_patients(patient_ids_out)

    # Compute normalization stats from training set
    means, stds = compute_norm_stats(windows, patient_ids_out, train_patients)

    # Normalize all windows
    windows_norm = normalize(windows, means, stds)

    # Build tensors for each split
    print("\nBuilding tensors...")
    train_data = build_tensors(windows_norm, patient_ids_out, train_patients, "train")
    val_data = build_tensors(windows_norm, patient_ids_out, val_patients, "val")
    test_data = build_tensors(windows_norm, patient_ids_out, test_patients, "test")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nSaving to {OUTPUT_DIR}...")
    if train_data:
        torch.save(train_data, os.path.join(OUTPUT_DIR, "train_data.pt"))
    if val_data:
        torch.save(val_data, os.path.join(OUTPUT_DIR, "val_data.pt"))
    if test_data:
        torch.save(test_data, os.path.join(OUTPUT_DIR, "test_data.pt"))

    # Save normalization parameters
    norm_params = {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'feature_names': CORRELATION_NAMES,
        'num_features': NUM_CORR,
        'transform': 'fisher_z',
        'clip_range': [-0.9999, 0.9999],
        'description': 'clip to ±0.9999 → arctanh (Fisher z) → z-score normalize with these means/stds',
    }
    with open(os.path.join(OUTPUT_DIR, "norm_params.json"), 'w') as f:
        json.dump(norm_params, f, indent=2)

    # Save split info
    split_info = {
        'seed': SEED,
        'train_patients': sorted(list(train_patients)),
        'val_patients': sorted(list(val_patients)),
        'test_patients': sorted(list(test_patients)),
        'n_train_patients': len(train_patients),
        'n_val_patients': len(val_patients),
        'n_test_patients': len(test_patients),
        'n_train_windows': train_data['target'].shape[0] if train_data else 0,
        'n_val_windows': val_data['target'].shape[0] if val_data else 0,
        'n_test_windows': test_data['target'].shape[0] if test_data else 0,
        'window_total': WINDOW_TOTAL,
        'window_history': WINDOW_HISTORY,
        'window_forecast': WINDOW_FORECAST,
        'stride': STRIDE,
        'expected_dt_sec': EXPECTED_DT,
        'resolution_min': 2.5,
        'data_source': DATA_SOURCE,
        'nan_filter': 'none (corr_features_focused has 0% NaN)',
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"  Train windows: {split_info['n_train_windows']:,}")
    print(f"  Val windows:   {split_info['n_val_windows']:,}")
    print(f"  Test windows:  {split_info['n_test_windows']:,}")
    print(f"  Total:         {split_info['n_train_windows'] + split_info['n_val_windows'] + split_info['n_test_windows']:,}")
    print(f"  Features:      {NUM_CORR} correlations (input & output)")
    print(f"  Resolution:    2.5 min (150s)")
    print(f"  History:       {WINDOW_HISTORY} steps = {WINDOW_HISTORY * 2.5:.0f} min")
    print(f"  Forecast:      {WINDOW_FORECAST} steps = {WINDOW_FORECAST * 2.5:.0f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
