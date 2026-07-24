#!/usr/bin/env python3
"""
Phase 6.2 v2 Data Preparation: Cluster label forecasting with label history input.

Same as Phase 6.2 but adds past cluster labels as an input feature.
Input: 7 correlations + 1 time position + 1 cluster label history (normalized) = 9 channels
Output: 12 steps × 7 classes (cluster label classification)

The cluster label history is normalized to [0, 1] by dividing by (num_classes - 1) = 6.
"""

import os
import json
import numpy as np
import torch
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_SOURCE = "/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/"
OUTPUT_DIR = "/gpfs/home/dk5565/forecasting/phase62/phase62_data/processed_v2/"

WINDOW_TOTAL = 60       # 48 history + 12 forecast
WINDOW_HISTORY = 48     # 2h at 2.5-min resolution
WINDOW_FORECAST = 12    # 30min at 2.5-min resolution
STRIDE = 12             # 30-min stride
EXPECTED_DT = 150       # 2.5 minutes in seconds

NUM_CORR = 7            # 7 focused correlation pairs
NUM_CLASSES = 7         # 7 cluster labels (0-6)
NUM_INPUT_FEATURES = 9  # 7 corr + 1 time + 1 label history

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
    patient_ids = np.load(os.path.join(DATA_SOURCE, "patient_ids.npy"), allow_pickle=True)
    seg_names = np.load(os.path.join(DATA_SOURCE, "seg_names.npy"), allow_pickle=True)
    window_times = np.load(os.path.join(DATA_SOURCE, "window_times.npy"))
    block_start_times = np.load(os.path.join(DATA_SOURCE, "block_start_times.npy"))

    print(f"  Total windows: {len(corr_features):,}")
    print(f"  Correlation shape: {corr_features.shape}")
    print(f"  Cluster labels: {np.unique(cluster_labels)} ({NUM_CLASSES} classes)")
    print(f"  Unique patients: {len(np.unique(patient_ids)):,}")

    return corr_features, cluster_labels, patient_ids, seg_names, window_times, block_start_times


def group_into_continuous_segments(corr_features, patient_ids, seg_names, window_times, block_start_times):
    """Group windows into continuous time series."""
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

    print(f"  Segments ≥ {WINDOW_TOTAL} windows: {len(continuous_segments):,}")
    print(f"  Segments too short (skipped): {skipped:,}")

    return continuous_segments


def create_sliding_windows(continuous_segments, corr_features, cluster_labels):
    """Create sliding windows with both correlations and labels."""
    print("\nCreating sliding windows...")
    windows_corr = []
    windows_labels = []
    patient_ids_out = []

    for pid, indices in continuous_segments:
        seg_corr = corr_features[indices]
        seg_labels = cluster_labels[indices]
        n_windows = (len(indices) - WINDOW_TOTAL) // STRIDE + 1

        for i in range(n_windows):
            start = i * STRIDE
            windows_corr.append(seg_corr[start:start + WINDOW_TOTAL])
            windows_labels.append(seg_labels[start:start + WINDOW_TOTAL])
            patient_ids_out.append(pid)

    print(f"  Valid windows: {len(windows_corr):,}")

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

    print(f"  Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    return train_patients, val_patients, test_patients


def compute_norm_stats(windows_corr, patient_ids_out, train_patients):
    """Compute Fisher z normalization stats from training set."""
    print("\nComputing normalization statistics...")
    train_mask = np.array([pid in train_patients for pid in patient_ids_out])
    train_data = windows_corr[train_mask]

    train_z = np.arctanh(np.clip(train_data, -0.9999, 0.9999))
    flat = train_z.reshape(-1, NUM_CORR)
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)
    stds[stds < 1e-8] = 1.0

    print(f"  Fisher z means: {means}")
    print(f"  Fisher z stds: {stds}")
    return means, stds


def compute_class_weights(windows_labels, patient_ids_out, train_patients):
    """Compute class weights from training set."""
    print("\nComputing class weights...")
    train_mask = np.array([pid in train_patients for pid in patient_ids_out])
    train_labels = windows_labels[train_mask]
    forecast_labels = train_labels[:, WINDOW_HISTORY:].flatten()

    vals, counts = np.unique(forecast_labels, return_counts=True)
    total = counts.sum()
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for v, c in zip(vals, counts):
        weights[v] = total / (NUM_CLASSES * c)
        print(f"    Cluster {v}: {c:,} ({100*c/total:.1f}%) → weight={weights[v]:.3f}")

    return weights


def normalize_correlations(windows_corr, means, stds):
    """Apply Fisher z-transform then z-score normalize."""
    z = np.arctanh(np.clip(windows_corr, -0.9999, 0.9999))
    return (z - means[np.newaxis, np.newaxis, :]) / stds[np.newaxis, np.newaxis, :]


def build_tensors(corr_norm, windows_labels, patient_ids_out, split_patients_set, split_name):
    """Build tensors for a given split.

    Input: 48 steps × 9 (7 correlations + 1 time + 1 cluster label history)
    Target: 12 steps of cluster labels (integers 0-6)
    """
    mask = np.array([pid in split_patients_set for pid in patient_ids_out])
    split_corr = corr_norm[mask]
    split_labels = windows_labels[mask]
    N = split_corr.shape[0]

    if N == 0:
        print(f"  WARNING: {split_name} has 0 windows!")
        return None

    # Historical correlations: first 48 steps
    historical_corr = split_corr[:, :WINDOW_HISTORY, :]  # (N, 48, 7)

    # Time position for history: linear 0 → 0.75
    time_hist = np.linspace(0.0, 0.75, WINDOW_HISTORY, dtype=np.float32)
    time_hist = np.tile(time_hist, (N, 1))[:, :, np.newaxis]  # (N, 48, 1)

    # Cluster label history: normalize to [0, 1] by dividing by (NUM_CLASSES - 1)
    label_hist = split_labels[:, :WINDOW_HISTORY].astype(np.float32) / (NUM_CLASSES - 1)
    label_hist = label_hist[:, :, np.newaxis]  # (N, 48, 1)

    # Concatenate: 7 corr + 1 time + 1 label = 9
    historical_ts_numeric = np.concatenate([historical_corr, time_hist, label_hist], axis=2)  # (N, 48, 9)

    # Future: time position only
    time_future = np.linspace(0.76, 1.0, WINDOW_FORECAST, dtype=np.float32)
    time_future = np.tile(time_future, (N, 1))[:, :, np.newaxis]  # (N, 12, 1)

    # Target: cluster labels for future steps
    target = split_labels[:, WINDOW_HISTORY:]  # (N, 12)

    # Static features: placeholder
    static_feats_numeric = np.zeros((N, 1), dtype=np.float32)

    data_dict = {
        'static_feats_numeric': torch.from_numpy(static_feats_numeric),
        'historical_ts_numeric': torch.from_numpy(historical_ts_numeric),
        'future_ts_numeric': torch.from_numpy(time_future),
        'target': torch.from_numpy(target),  # (N, 12) int64
    }

    print(f"  {split_name}: {N:,} windows")
    print(f"    historical_ts_numeric: {data_dict['historical_ts_numeric'].shape}")
    print(f"    target: {data_dict['target'].shape}")

    return data_dict


def main():
    print("=" * 70)
    print("Phase 6.2 v2 Data Preparation — Cluster Forecasting + Label History")
    print("  Input:  48 steps × 9 (7 corr + 1 time + 1 cluster label history)")
    print("  Output: 12 steps × 7 classes")
    print("=" * 70)

    corr_features, cluster_labels, patient_ids, seg_names, window_times, block_start_times = load_data()

    continuous_segments = group_into_continuous_segments(
        corr_features, patient_ids, seg_names, window_times, block_start_times)

    windows_corr, windows_labels, patient_ids_out = create_sliding_windows(
        continuous_segments, corr_features, cluster_labels)

    train_patients, val_patients, test_patients = split_patients(patient_ids_out)
    means, stds = compute_norm_stats(windows_corr, patient_ids_out, train_patients)
    class_weights = compute_class_weights(windows_labels, patient_ids_out, train_patients)
    corr_norm = normalize_correlations(windows_corr, means, stds)

    print("\nBuilding tensors...")
    train_data = build_tensors(corr_norm, windows_labels, patient_ids_out, train_patients, "train")
    val_data = build_tensors(corr_norm, windows_labels, patient_ids_out, val_patients, "val")
    test_data = build_tensors(corr_norm, windows_labels, patient_ids_out, test_patients, "test")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving to {OUTPUT_DIR}...")

    torch.save(train_data, os.path.join(OUTPUT_DIR, "train_data.pt"))
    torch.save(val_data, os.path.join(OUTPUT_DIR, "val_data.pt"))
    torch.save(test_data, os.path.join(OUTPUT_DIR, "test_data.pt"))

    norm_params = {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'feature_names': CORRELATION_NAMES,
        'num_corr': NUM_CORR,
        'num_classes': NUM_CLASSES,
        'num_input_features': NUM_INPUT_FEATURES,
        'class_weights': class_weights.tolist(),
        'label_normalization': f'divided by {NUM_CLASSES - 1} to map [0,6] → [0,1]',
        'transform': 'fisher_z',
        'clip_range': [-0.9999, 0.9999],
    }
    with open(os.path.join(OUTPUT_DIR, "norm_params.json"), 'w') as f:
        json.dump(norm_params, f, indent=2)

    split_info = {
        'seed': SEED,
        'train_patients': sorted(list(train_patients)),
        'val_patients': sorted(list(val_patients)),
        'test_patients': sorted(list(test_patients)),
        'n_train_windows': train_data['target'].shape[0],
        'n_val_windows': val_data['target'].shape[0],
        'n_test_windows': test_data['target'].shape[0],
        'window_history': WINDOW_HISTORY,
        'window_forecast': WINDOW_FORECAST,
        'stride': STRIDE,
        'input_channels': '7 corr + 1 time + 1 label_history = 9',
        'label_history': 'cluster labels normalized to [0,1]',
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"  Train: {split_info['n_train_windows']:,}")
    print(f"  Val:   {split_info['n_val_windows']:,}")
    print(f"  Test:  {split_info['n_test_windows']:,}")
    print(f"  Input: 48 × 9 (7 corr + 1 time + 1 label history)")
    print(f"  Output: 12 steps × 7 classes")
    print("=" * 70)


if __name__ == "__main__":
    main()
