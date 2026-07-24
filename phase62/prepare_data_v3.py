#!/usr/bin/env python3
"""
Phase 6.2 v3 Data Preparation: Cluster label forecasting with label history + X_stats features.

Extends v2 by adding summarized X_stats (19 mean + 19 std = 38 physiological features).
Input: 7 correlations + 38 physio stats + 1 time position + 1 cluster label history = 47 channels
Output: 12 steps × 7 classes (cluster label classification)
"""

import os
import json
import numpy as np
import torch
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_SOURCE = "/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/"
OUTPUT_DIR = "/gpfs/home/dk5565/forecasting/phase62/phase62_data/processed_v3/"

WINDOW_TOTAL = 60       # 48 history + 12 forecast
WINDOW_HISTORY = 48     # 2h at 2.5-min resolution
WINDOW_FORECAST = 12    # 30min at 2.5-min resolution
STRIDE = 12             # 30-min stride
EXPECTED_DT = 150       # 2.5 minutes in seconds

NUM_CORR = 7            # 7 focused correlation pairs
NUM_PHYSIO = 19         # 19 physiological features from X_stats
NUM_PHYSIO_STATS = 38   # 19 mean + 19 std
NUM_CLASSES = 7         # 7 cluster labels (0-6)
NUM_INPUT_FEATURES = 47 # 7 corr + 38 physio + 1 time + 1 label history

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

PHYSIO_FEATURE_NAMES = [
    "HR", "RR", "SBP", "DBP", "PP",
    "MAP", "ABP_area", "PLETH_ACDC", "PLETH_amp", "ECG_Ramp",
    "HRV_RMSSD", "HR_range", "ShockIdx", "PPV", "PVI",
    "PTT", "dPdt_max", "ABP_tau", "RESP_amp",
]

# All input feature names: 7 corr + 19 mean + 19 std + 1 time + 1 label
INPUT_FEATURE_NAMES = (
    CORRELATION_NAMES +
    [f"{name}_mean" for name in PHYSIO_FEATURE_NAMES] +
    [f"{name}_std" for name in PHYSIO_FEATURE_NAMES] +
    ['time_position', 'cluster_label_history']
)


def load_data():
    """Load arrays from data_m3_120s_prediction."""
    print("Loading data from data_m3_120s_prediction...")
    corr_features = np.load(os.path.join(DATA_SOURCE, "corr_features_focused.npy"))  # (N, 7)
    x_stats = np.load(os.path.join(DATA_SOURCE, "X_stats.npy"), mmap_mode='r')       # (N, 19, 109)
    cluster_labels = np.load(os.path.join(DATA_SOURCE, "cluster_labels.npy"))         # (N,)
    patient_ids = np.load(os.path.join(DATA_SOURCE, "patient_ids.npy"), allow_pickle=True)
    seg_names = np.load(os.path.join(DATA_SOURCE, "seg_names.npy"), allow_pickle=True)
    window_times = np.load(os.path.join(DATA_SOURCE, "window_times.npy"))
    block_start_times = np.load(os.path.join(DATA_SOURCE, "block_start_times.npy"))

    print(f"  Total windows: {len(corr_features):,}")
    print(f"  Correlation shape: {corr_features.shape}")
    print(f"  X_stats shape: {x_stats.shape}")
    print(f"  Cluster labels: {np.unique(cluster_labels)} ({NUM_CLASSES} classes)")
    print(f"  Unique patients: {len(np.unique(patient_ids)):,}")

    return corr_features, x_stats, cluster_labels, patient_ids, seg_names, window_times, block_start_times


def summarize_x_stats(x_stats):
    """Summarize X_stats (N, 19, 109) into (N, 38) = mean + std over 109 sub-windows.

    NaN handling: nanmean/nanstd — if all 109 sub-windows are NaN for a feature,
    the result is NaN (will be imputed to 0 after normalization).
    """
    print("\nSummarizing X_stats: (N, 19, 109) → (N, 38) [mean + std]...")
    N = x_stats.shape[0]

    CHUNK = 50000
    means_all = np.zeros((N, NUM_PHYSIO), dtype=np.float32)
    stds_all = np.zeros((N, NUM_PHYSIO), dtype=np.float32)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunk = np.array(x_stats[start:end])  # Load chunk into memory
        with np.errstate(all='ignore'):
            means_all[start:end] = np.nanmean(chunk, axis=2)  # (chunk, 19)
            stds_all[start:end] = np.nanstd(chunk, axis=2)    # (chunk, 19)
        if (start // CHUNK) % 5 == 0:
            print(f"    Processed {end:,} / {N:,} windows...")

    # Combine mean + std
    physio_stats = np.concatenate([means_all, stds_all], axis=1)  # (N, 38)

    # Report NaN stats
    nan_count = np.isnan(physio_stats).sum()
    nan_pct = 100 * nan_count / physio_stats.size
    print(f"  Physio stats shape: {physio_stats.shape}")
    print(f"  NaN in physio stats: {nan_count:,} / {physio_stats.size:,} ({nan_pct:.2f}%)")

    for i in range(NUM_PHYSIO):
        mean_nan = np.isnan(physio_stats[:, i]).sum()
        std_nan = np.isnan(physio_stats[:, i + NUM_PHYSIO]).sum()
        if mean_nan > 0 or std_nan > 0:
            print(f"    {PHYSIO_FEATURE_NAMES[i]}: mean_nan={mean_nan:,} ({100*mean_nan/N:.1f}%), "
                  f"std_nan={std_nan:,} ({100*std_nan/N:.1f}%)")

    return physio_stats


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


def create_sliding_windows(continuous_segments, corr_features, physio_stats, cluster_labels):
    """Create sliding windows with correlations, physio stats, and labels."""
    print("\nCreating sliding windows...")
    windows_corr = []
    windows_physio = []
    windows_labels = []
    patient_ids_out = []

    for pid, indices in continuous_segments:
        seg_corr = corr_features[indices]
        seg_physio = physio_stats[indices]
        seg_labels = cluster_labels[indices]
        n_windows = (len(indices) - WINDOW_TOTAL) // STRIDE + 1

        for i in range(n_windows):
            start = i * STRIDE
            windows_corr.append(seg_corr[start:start + WINDOW_TOTAL])
            windows_physio.append(seg_physio[start:start + WINDOW_TOTAL])
            windows_labels.append(seg_labels[start:start + WINDOW_TOTAL])
            patient_ids_out.append(pid)

    print(f"  Valid windows: {len(windows_corr):,}")

    return (np.array(windows_corr, dtype=np.float32),
            np.array(windows_physio, dtype=np.float32),
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


def compute_norm_stats(windows_corr, windows_physio, patient_ids_out, train_patients):
    """Compute normalization stats from training set.

    Correlations: Fisher z-transform then z-score.
    Physio stats: z-score normalization (mean/std from train set).
    """
    print("\nComputing normalization statistics...")
    train_mask = np.array([pid in train_patients for pid in patient_ids_out])

    # Correlation normalization (Fisher z)
    train_corr = windows_corr[train_mask]
    train_z = np.arctanh(np.clip(train_corr, -0.9999, 0.9999))
    flat_corr = train_z.reshape(-1, NUM_CORR)
    corr_means = np.nanmean(flat_corr, axis=0)
    corr_stds = np.nanstd(flat_corr, axis=0)
    corr_stds[corr_stds < 1e-8] = 1.0

    # Physio normalization (z-score)
    train_physio = windows_physio[train_mask]
    flat_physio = train_physio.reshape(-1, NUM_PHYSIO_STATS)
    physio_means = np.nanmean(flat_physio, axis=0)
    physio_stds = np.nanstd(flat_physio, axis=0)
    physio_stds[physio_stds < 1e-8] = 1.0

    print(f"  Corr Fisher z means: {corr_means}")
    print(f"  Corr Fisher z stds: {corr_stds}")
    print(f"  Physio means range: [{physio_means.min():.3f}, {physio_means.max():.3f}]")
    print(f"  Physio stds range: [{physio_stds.min():.3f}, {physio_stds.max():.3f}]")

    return corr_means, corr_stds, physio_means, physio_stds


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


def normalize_and_build_input(windows_corr, windows_physio, windows_labels,
                               corr_means, corr_stds, physio_means, physio_stds):
    """Normalize correlations and physio features, build full input tensor."""
    N = windows_corr.shape[0]

    # Normalize correlations: Fisher z then z-score
    corr_z = np.arctanh(np.clip(windows_corr, -0.9999, 0.9999))
    corr_norm = (corr_z - corr_means[np.newaxis, np.newaxis, :]) / corr_stds[np.newaxis, np.newaxis, :]

    # Normalize physio: z-score
    physio_norm = (windows_physio - physio_means[np.newaxis, np.newaxis, :]) / physio_stds[np.newaxis, np.newaxis, :]
    # Impute NaN → 0 (post-normalization, 0 = population mean)
    nan_count = np.isnan(physio_norm).sum()
    if nan_count > 0:
        print(f"  Imputing {nan_count:,} NaN values in physio features to 0")
        physio_norm = np.nan_to_num(physio_norm, nan=0.0)

    return corr_norm, physio_norm


def build_tensors(corr_norm, physio_norm, windows_labels, patient_ids_out, split_patients_set, split_name):
    """Build tensors for a given split.

    Input: 48 steps × 47 (7 corr + 38 physio + 1 time + 1 label history)
    Target: 12 steps of cluster labels (integers 0-6)
    """
    mask = np.array([pid in split_patients_set for pid in patient_ids_out])
    split_corr = corr_norm[mask]
    split_physio = physio_norm[mask]
    split_labels = windows_labels[mask]
    N = split_corr.shape[0]

    if N == 0:
        print(f"  WARNING: {split_name} has 0 windows!")
        return None

    # Historical correlations: first 48 steps (N, 48, 7)
    hist_corr = split_corr[:, :WINDOW_HISTORY, :]

    # Historical physio stats: first 48 steps (N, 48, 38)
    hist_physio = split_physio[:, :WINDOW_HISTORY, :]

    # Time position for history: linear 0 → 0.75
    time_hist = np.linspace(0.0, 0.75, WINDOW_HISTORY, dtype=np.float32)
    time_hist = np.tile(time_hist, (N, 1))[:, :, np.newaxis]  # (N, 48, 1)

    # Cluster label history: normalize to [0, 1] by dividing by (NUM_CLASSES - 1)
    label_hist = split_labels[:, :WINDOW_HISTORY].astype(np.float32) / (NUM_CLASSES - 1)
    label_hist = label_hist[:, :, np.newaxis]  # (N, 48, 1)

    # Concatenate: 7 corr + 38 physio + 1 time + 1 label = 47
    historical_ts_numeric = np.concatenate(
        [hist_corr, hist_physio, time_hist, label_hist], axis=2)  # (N, 48, 47)

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
    print("Phase 6.2 v3 Data Preparation — Cluster Forecasting + Label History + X_stats")
    print(f"  Input:  48 steps × {NUM_INPUT_FEATURES} (7 corr + 38 physio + 1 time + 1 label)")
    print("  Output: 12 steps × 7 classes")
    print("=" * 70)

    corr_features, x_stats, cluster_labels, patient_ids, seg_names, window_times, block_start_times = load_data()

    # Summarize X_stats into mean + std per window
    physio_stats = summarize_x_stats(x_stats)

    continuous_segments = group_into_continuous_segments(
        corr_features, patient_ids, seg_names, window_times, block_start_times)

    windows_corr, windows_physio, windows_labels, patient_ids_out = create_sliding_windows(
        continuous_segments, corr_features, physio_stats, cluster_labels)

    train_patients, val_patients, test_patients = split_patients(patient_ids_out)
    corr_means, corr_stds, physio_means, physio_stds = compute_norm_stats(
        windows_corr, windows_physio, patient_ids_out, train_patients)
    class_weights = compute_class_weights(windows_labels, patient_ids_out, train_patients)

    print("\nNormalizing features...")
    corr_norm, physio_norm = normalize_and_build_input(
        windows_corr, windows_physio, windows_labels,
        corr_means, corr_stds, physio_means, physio_stds)

    print("\nBuilding tensors...")
    train_data = build_tensors(corr_norm, physio_norm, windows_labels, patient_ids_out, train_patients, "train")
    val_data = build_tensors(corr_norm, physio_norm, windows_labels, patient_ids_out, val_patients, "val")
    test_data = build_tensors(corr_norm, physio_norm, windows_labels, patient_ids_out, test_patients, "test")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving to {OUTPUT_DIR}...")

    torch.save(train_data, os.path.join(OUTPUT_DIR, "train_data.pt"))
    torch.save(val_data, os.path.join(OUTPUT_DIR, "val_data.pt"))
    torch.save(test_data, os.path.join(OUTPUT_DIR, "test_data.pt"))

    norm_params = {
        'corr_means': corr_means.tolist(),
        'corr_stds': corr_stds.tolist(),
        'physio_means': physio_means.tolist(),
        'physio_stds': physio_stds.tolist(),
        'feature_names': INPUT_FEATURE_NAMES,
        'correlation_names': CORRELATION_NAMES,
        'physio_feature_names': PHYSIO_FEATURE_NAMES,
        'num_corr': NUM_CORR,
        'num_physio_stats': NUM_PHYSIO_STATS,
        'num_classes': NUM_CLASSES,
        'num_input_features': NUM_INPUT_FEATURES,
        'class_weights': class_weights.tolist(),
        'label_normalization': f'divided by {NUM_CLASSES - 1} to map [0,6] → [0,1]',
        'corr_transform': 'fisher_z then z-score',
        'physio_transform': 'z-score (NaN imputed to 0)',
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
        'input_channels': '7 corr + 38 physio (19 mean + 19 std) + 1 time + 1 label_history = 47',
        'label_history': 'cluster labels normalized to [0,1]',
        'physio_summary': 'mean + std over 109 sub-windows per X_stats feature',
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"  Train: {split_info['n_train_windows']:,}")
    print(f"  Val:   {split_info['n_val_windows']:,}")
    print(f"  Test:  {split_info['n_test_windows']:,}")
    print(f"  Input: 48 × {NUM_INPUT_FEATURES} (7 corr + 38 physio + 1 time + 1 label)")
    print(f"  Output: 12 steps × 7 classes")
    print("=" * 70)


if __name__ == "__main__":
    main()
