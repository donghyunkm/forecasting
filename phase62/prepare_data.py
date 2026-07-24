#!/usr/bin/env python3
"""
Phase 6.2 Data Preparation: Forecasting cluster labels from correlation history.

Same input as Phase 6 (7 correlations + 1 time position), but the target is now
cluster labels (7 classes, integers 0-6) instead of continuous correlation values.

Data source: /gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/
  - corr_features_focused.npy  (564596, 7)  — 7 pairwise correlations per window
  - cluster_labels.npy         (564596,)    — cluster assignments (0-6)
  - patient_ids.npy            (564596,)    — patient identifiers
  - seg_names.npy              (564596,)    — segment identifiers
  - window_times.npy           (564596,)    — timestamps (seconds)
  - block_start_times.npy      (564596,)    — block start times (for continuity grouping)

Resolution: 2.5 minutes (150 seconds) between windows.

Input: 7 correlations (48 steps = 2h) + 1 time position = 8 channels
Output: cluster labels (12 steps = 30min), 7 classes
"""

import os
import json
import numpy as np
import torch
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_SOURCE = "/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/"
OUTPUT_DIR = "/gpfs/home/dk5565/forecasting/phase62/phase62_data/processed/"

WINDOW_TOTAL = 60       # 48 history + 12 forecast
WINDOW_HISTORY = 48     # 2h at 2.5-min resolution
WINDOW_FORECAST = 12    # 30min at 2.5-min resolution
STRIDE = 12             # 30-min stride
EXPECTED_DT = 150       # 2.5 minutes in seconds

NUM_CORR = 7            # 7 focused correlation pairs
NUM_CLASSES = 7         # 7 cluster labels (0-6)

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
    cluster_labels = np.load(os.path.join(DATA_SOURCE, "cluster_labels.npy"))  # (N,)
    patient_ids = np.load(os.path.join(DATA_SOURCE, "patient_ids.npy"), allow_pickle=True)  # (N,)
    seg_names = np.load(os.path.join(DATA_SOURCE, "seg_names.npy"), allow_pickle=True)  # (N,)
    window_times = np.load(os.path.join(DATA_SOURCE, "window_times.npy"))  # (N,)
    block_start_times = np.load(os.path.join(DATA_SOURCE, "block_start_times.npy"))  # (N,)

    print(f"  Total windows: {len(corr_features):,}")
    print(f"  Correlation shape: {corr_features.shape}")
    print(f"  Cluster labels shape: {cluster_labels.shape}")
    print(f"  Unique clusters: {np.unique(cluster_labels)}")
    print(f"  Unique patients: {len(np.unique(patient_ids)):,}")

    # Class distribution
    print(f"\n  Cluster distribution:")
    vals, counts = np.unique(cluster_labels, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"    Cluster {v}: {c:,} ({100*c/len(cluster_labels):.1f}%)")

    return corr_features, cluster_labels, patient_ids, seg_names, window_times, block_start_times


def group_into_continuous_segments(corr_features, patient_ids, seg_names, window_times, block_start_times):
    """Group windows into continuous time series using (seg_name, block_start_time)."""
    print("\nGrouping into continuous segments...")

    groups = defaultdict(list)
    for i in range(len(corr_features)):
        key = (str(seg_names[i]), float(block_start_times[i]))
        groups[key].append(i)

    continuous_segments = []
    skipped = 0

    for key, indices in groups.items():
        indices = np.array(indices)
        times = window_times[indices]
        order = np.argsort(times)
        indices = indices[order]
        times = times[order]

        if len(indices) > 1:
            dts = np.diff(times)
            if not np.all(np.abs(dts - EXPECTED_DT) < 1.0):
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

    lengths = [len(indices) for _, indices in continuous_segments]
    if lengths:
        print(f"  Length stats — min: {min(lengths)}, max: {max(lengths)}, "
              f"mean: {np.mean(lengths):.1f}, median: {np.median(lengths):.0f}")

    return continuous_segments


def create_sliding_windows(continuous_segments, corr_features, cluster_labels):
    """Create sliding windows from continuous segments.

    Each window: input = (WINDOW_TOTAL, 7) correlations, target = (WINDOW_FORECAST,) cluster labels.
    """
    print("\nCreating sliding windows...")
    windows_corr = []
    windows_labels = []
    patient_ids_out = []

    total_windows = 0

    for pid, indices in continuous_segments:
        seg_corr = corr_features[indices]       # (L, 7)
        seg_labels = cluster_labels[indices]    # (L,)
        n_windows = (len(indices) - WINDOW_TOTAL) // STRIDE + 1

        for i in range(n_windows):
            start = i * STRIDE
            win_corr = seg_corr[start:start + WINDOW_TOTAL]         # (60, 7)
            win_labels = seg_labels[start:start + WINDOW_TOTAL]     # (60,)

            # No NaN in corr_features_focused or cluster_labels
            windows_corr.append(win_corr)
            windows_labels.append(win_labels)
            patient_ids_out.append(pid)
            total_windows += 1

    print(f"  Valid windows: {total_windows:,}")

    return (np.array(windows_corr, dtype=np.float32),
            np.array(windows_labels, dtype=np.int64),
            np.array(patient_ids_out))


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


def compute_norm_stats(windows_corr, patient_ids_out, train_patients):
    """Compute normalization statistics from training patients in Fisher z-space."""
    print("\nComputing normalization statistics from training set (Fisher z-space)...")
    train_mask = np.array([pid in train_patients for pid in patient_ids_out])
    train_data = windows_corr[train_mask]  # (N_train, 60, 7)

    train_z = np.arctanh(np.clip(train_data, -0.9999, 0.9999))

    flat = train_z.reshape(-1, NUM_CORR)
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)
    stds[stds < 1e-8] = 1.0

    print(f"  Fisher z-space means: {means}")
    print(f"  Fisher z-space stds: {stds}")

    return means, stds


def compute_class_weights(windows_labels, patient_ids_out, train_patients):
    """Compute class weights from training set for imbalanced classification."""
    print("\nComputing class weights from training set...")
    train_mask = np.array([pid in train_patients for pid in patient_ids_out])
    train_labels = windows_labels[train_mask]  # (N_train, 60)

    # Use only the forecast portion for class weight computation
    forecast_labels = train_labels[:, WINDOW_HISTORY:]  # (N_train, 12)
    flat_labels = forecast_labels.flatten()

    vals, counts = np.unique(flat_labels, return_counts=True)
    total = counts.sum()
    # Inverse frequency weighting: weight = total / (num_classes * count)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for v, c in zip(vals, counts):
        weights[v] = total / (NUM_CLASSES * c)

    print(f"  Training forecast label distribution:")
    for v, c in zip(vals, counts):
        print(f"    Cluster {v}: {c:,} ({100*c/total:.1f}%) → weight={weights[v]:.3f}")

    return weights


def normalize_correlations(windows_corr, means, stds):
    """Apply Fisher z-transform then z-score normalize."""
    z = np.arctanh(np.clip(windows_corr, -0.9999, 0.9999))
    return (z - means[np.newaxis, np.newaxis, :]) / stds[np.newaxis, np.newaxis, :]


def build_tensors(corr_norm, windows_labels, patient_ids_out, split_patients_set, split_name):
    """Build tensors for a given split.

    Input: 48 steps × 8 (7 correlations + 1 time position)
    Target: 12 steps of cluster labels (integers 0-6)
    """
    mask = np.array([pid in split_patients_set for pid in patient_ids_out])
    split_corr = corr_norm[mask]          # (N, 60, 7)
    split_labels = windows_labels[mask]   # (N, 60)
    N = split_corr.shape[0]

    if N == 0:
        print(f"  WARNING: {split_name} has 0 windows!")
        return None

    # Historical input: first 48 steps, 7 correlations + 1 time position = 8 channels
    historical_corr = split_corr[:, :WINDOW_HISTORY, :]  # (N, 48, 7)

    # Time position for history: linear 0 → 0.75
    time_hist = np.linspace(0.0, 0.75, WINDOW_HISTORY, dtype=np.float32)
    time_hist = np.tile(time_hist, (N, 1))[:, :, np.newaxis]  # (N, 48, 1)

    historical_ts_numeric = np.concatenate([historical_corr, time_hist], axis=2)  # (N, 48, 8)

    # Future: time position only (known into the future)
    time_future = np.linspace(0.76, 1.0, WINDOW_FORECAST, dtype=np.float32)
    time_future = np.tile(time_future, (N, 1))[:, :, np.newaxis]  # (N, 12, 1)
    future_ts_numeric = time_future

    # Target: cluster labels for the 12 forecast steps (integers 0-6)
    target = split_labels[:, WINDOW_HISTORY:]  # (N, 12)

    # Static features: placeholder
    static_feats_numeric = np.zeros((N, 1), dtype=np.float32)

    data_dict = {
        'static_feats_numeric': torch.from_numpy(static_feats_numeric),
        'historical_ts_numeric': torch.from_numpy(historical_ts_numeric.astype(np.float32)),
        'future_ts_numeric': torch.from_numpy(future_ts_numeric.astype(np.float32)),
        'target': torch.from_numpy(target),  # (N, 12) int64
    }

    print(f"  {split_name}: {N:,} windows")
    print(f"    historical_ts_numeric: {data_dict['historical_ts_numeric'].shape}")
    print(f"    future_ts_numeric: {data_dict['future_ts_numeric'].shape}")
    print(f"    target: {data_dict['target'].shape} (dtype={data_dict['target'].dtype})")

    # Report class distribution in this split's targets
    flat_target = target.flatten()
    vals, counts = np.unique(flat_target, return_counts=True)
    total = counts.sum()
    dist_str = ", ".join([f"{v}:{100*c/total:.1f}%" for v, c in zip(vals, counts)])
    print(f"    target distribution: {dist_str}")

    return data_dict


def main():
    print("=" * 70)
    print("Phase 6.2 Data Preparation — Cluster Label Forecasting")
    print("  Source: data_m3_120s_prediction (2.5-min resolution)")
    print("  Input:  48 steps (2h) × 7 correlations + time position")
    print("  Output: 12 steps (30min) × 1 cluster label (7 classes)")
    print("=" * 70)

    # Load data
    corr_features, cluster_labels, patient_ids, seg_names, window_times, block_start_times = load_data()

    # Group into continuous segments
    continuous_segments = group_into_continuous_segments(
        corr_features, patient_ids, seg_names, window_times, block_start_times)

    # Create sliding windows
    windows_corr, windows_labels, patient_ids_out = create_sliding_windows(
        continuous_segments, corr_features, cluster_labels)

    if len(windows_corr) == 0:
        print("ERROR: No valid windows found! Check data.")
        return

    print(f"\n  Correlation window shape: {windows_corr.shape}")
    print(f"  Labels window shape: {windows_labels.shape}")

    # Split patients
    train_patients, val_patients, test_patients = split_patients(patient_ids_out)

    # Compute normalization stats from training set
    means, stds = compute_norm_stats(windows_corr, patient_ids_out, train_patients)

    # Compute class weights for loss function
    class_weights = compute_class_weights(windows_labels, patient_ids_out, train_patients)

    # Normalize correlations
    corr_norm = normalize_correlations(windows_corr, means, stds)

    # Build tensors for each split
    print("\nBuilding tensors...")
    train_data = build_tensors(corr_norm, windows_labels, patient_ids_out, train_patients, "train")
    val_data = build_tensors(corr_norm, windows_labels, patient_ids_out, val_patients, "val")
    test_data = build_tensors(corr_norm, windows_labels, patient_ids_out, test_patients, "test")

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
        'num_corr': NUM_CORR,
        'num_classes': NUM_CLASSES,
        'class_weights': class_weights.tolist(),
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
        'task': 'classification',
        'num_classes': NUM_CLASSES,
        'target_description': 'cluster labels (0-6) for each of 12 forecast steps',
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"  Train windows: {split_info['n_train_windows']:,}")
    print(f"  Val windows:   {split_info['n_val_windows']:,}")
    print(f"  Test windows:  {split_info['n_test_windows']:,}")
    print(f"  Total:         {split_info['n_train_windows'] + split_info['n_val_windows'] + split_info['n_test_windows']:,}")
    print(f"  Task:          Classification (7 classes)")
    print(f"  Input:         {NUM_CORR} correlations + 1 time = 8 channels")
    print(f"  Output:        {WINDOW_FORECAST} steps × {NUM_CLASSES} classes")
    print(f"  Resolution:    2.5 min (150s)")
    print(f"  History:       {WINDOW_HISTORY} steps = {WINDOW_HISTORY * 2.5:.0f} min")
    print(f"  Forecast:      {WINDOW_FORECAST} steps = {WINDOW_FORECAST * 2.5:.0f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
