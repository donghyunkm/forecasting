#!/usr/bin/env python3
"""
Phase 5 Ablation Data Preparation: vitals-only input (no correlation features).

Same pipeline as Phase 5 prepare_data.py but drops channels 0-6 (correlations),
keeping only channels 7-10 (vitals). This produces tensors with:
  - historical_ts_numeric: (N, 72, 5)  -- 4 vitals + 1 time position
  - target: (N, 24, 4) -- same as full model

This allows direct comparison with the full Phase 5 model to measure the
contribution of waveform correlation features.
"""

import os
import json
import numpy as np
import torch
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_SOURCE = "/gpfs/home/dk5565/forecasting/phase5/data_extraction/output/merged/"
OUTPUT_DIR = "/gpfs/home/dk5565/forecasting/phase5/ablation/data/processed/"

WINDOW_TOTAL = 96       # 72 history + 24 forecast
WINDOW_HISTORY = 72     # 6h at 5-min stride
WINDOW_FORECAST = 24    # 2h at 5-min stride
STRIDE = 12            # 1h stride
EXPECTED_DT = 300      # 5 minutes in seconds

NUM_FEATURES_FULL = 11  # 7 correlations + 4 vitals (original)
NUM_VITALS = 4          # vitals-only for ablation
VITAL_INDICES = [7, 8, 9, 10]  # ABPMean, PULSE, SpO2, RESP in original 11-dim

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

VITAL_NAMES = ['ABPMean', 'PULSE', 'SpO2', 'RESP']


def load_data():
    """Load merged arrays from mimicEran output."""
    print("Loading merged data...")
    features = np.load(os.path.join(DATA_SOURCE, "features.npy"))       # (N, 11)
    patient_ids = np.load(os.path.join(DATA_SOURCE, "patient_ids.npy"))  # (N,)
    seg_names = np.load(os.path.join(DATA_SOURCE, "seg_names.npy"))      # (N,)
    window_times = np.load(os.path.join(DATA_SOURCE, "window_times.npy"))  # (N,)

    print(f"  Total windows: {len(features):,}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Unique patients: {len(np.unique(patient_ids)):,}")

    return features, patient_ids, seg_names, window_times


def group_and_sort(features, patient_ids, seg_names, window_times):
    """Group windows by (patient_id, seg_name), sort by time within each group."""
    print("\nGrouping by (patient_id, seg_name)...")
    groups = defaultdict(list)

    for i in range(len(features)):
        key = (str(patient_ids[i]), str(seg_names[i]))
        groups[key].append(i)

    print(f"  Total groups (patient, segment): {len(groups):,}")

    # Sort each group by window_time
    for key in groups:
        indices = groups[key]
        times = window_times[indices]
        order = np.argsort(times)
        groups[key] = [indices[o] for o in order]

    return groups


def split_at_gaps(indices, window_times, expected_dt=EXPECTED_DT):
    """Split a sorted index list into continuous sub-segments at temporal gaps."""
    if len(indices) < 2:
        return [indices]

    segments = []
    current_segment = [indices[0]]

    for i in range(1, len(indices)):
        dt = window_times[indices[i]] - window_times[indices[i - 1]]
        if abs(dt - expected_dt) < 1:  # Allow 1s tolerance
            current_segment.append(indices[i])
        else:
            segments.append(current_segment)
            current_segment = [indices[i]]

    segments.append(current_segment)
    return segments


def create_sliding_windows(continuous_segments, features, window_times):
    """Create sliding windows from continuous segments (vitals-only)."""
    print("\nCreating sliding windows (vitals-only)...")
    windows = []
    patient_ids_out = []

    total_segments = 0
    total_windows = 0

    for (pid, seg), indices_list in continuous_segments.items():
        for indices in indices_list:
            if len(indices) < WINDOW_TOTAL:
                continue
            total_segments += 1

            # Extract only vitals (indices 7-10 from the 11-dim features)
            seg_features = features[indices]  # (L, 11)

            # Sliding window
            for start in range(0, len(indices) - WINDOW_TOTAL + 1, STRIDE):
                window_full = seg_features[start:start + WINDOW_TOTAL]  # (96, 11)

                # Check for NaN across ALL 11 features (same filter as full model
                # to keep exact same set of windows for fair comparison)
                if np.any(np.isnan(window_full)):
                    continue

                # Extract only vital signs
                window_vitals = window_full[:, VITAL_INDICES]  # (96, 4)

                windows.append(window_vitals)
                patient_ids_out.append(pid)
                total_windows += 1

    print(f"  Continuous segments ≥ {WINDOW_TOTAL} steps: {total_segments:,}")
    print(f"  Valid windows (no NaN): {total_windows:,}")

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
    """Compute Z-score normalization statistics from training patients (vitals only)."""
    print("\nComputing normalization statistics from training set...")
    train_mask = np.array([pid in train_patients for pid in patient_ids_out])
    train_data = windows[train_mask]  # (N_train, 96, 4)

    # Compute mean and std across all time steps and windows for each vital
    flat = train_data.reshape(-1, NUM_VITALS)  # (N_train * 96, 4)
    means = np.nanmean(flat, axis=0)  # (4,)
    stds = np.nanstd(flat, axis=0)    # (4,)

    # Prevent division by zero
    stds[stds < 1e-8] = 1.0

    print(f"  Means: {means}")
    print(f"  Stds: {stds}")

    return means, stds


def normalize(windows, means, stds):
    """Z-score normalize vitals."""
    return (windows - means[np.newaxis, np.newaxis, :]) / stds[np.newaxis, np.newaxis, :]


def build_tensors(windows_norm, patient_ids_out, split_patients_set, split_name):
    """Build TFT-ready tensors for a given split (vitals-only: 5 input features)."""
    mask = np.array([pid in split_patients_set for pid in patient_ids_out])
    split_windows = windows_norm[mask]  # (N, 96, 4)
    N = split_windows.shape[0]

    if N == 0:
        print(f"  WARNING: {split_name} has 0 windows!")
        return None

    # Historical: first 72 steps, 4 vitals + 1 time position = 5 channels
    historical_vitals = split_windows[:, :WINDOW_HISTORY, :]  # (N, 72, 4)

    # Time position for history
    time_hist = np.linspace(0.0, 0.75, WINDOW_HISTORY, dtype=np.float32)
    time_hist = np.tile(time_hist, (N, 1))[:, :, np.newaxis]  # (N, 72, 1)

    historical_ts_numeric = np.concatenate([historical_vitals, time_hist], axis=2)  # (N, 72, 5)

    # Future: time position only
    time_future = np.linspace(0.76, 1.0, WINDOW_FORECAST, dtype=np.float32)
    time_future = np.tile(time_future, (N, 1))[:, :, np.newaxis]  # (N, 24, 1)
    future_ts_numeric = time_future

    # Target: 4 vitals for future 24 steps (normalized)
    target = split_windows[:, WINDOW_HISTORY:, :]  # (N, 24, 4)

    # Target mask: all ones since we filtered out NaN windows
    target_mask = np.ones((N, WINDOW_FORECAST, NUM_VITALS), dtype=np.float32)

    # Static features: placeholder
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
    print(f"    target: {data_dict['target'].shape}")

    return data_dict


def main():
    print("=" * 70)
    print("Phase 5 Ablation — Data Preparation (Vitals-Only)")
    print("=" * 70)
    print("  Dropping correlation features [0-6], keeping vitals [7-10]")
    print(f"  Historical input: (72, 5) = 4 vitals + 1 time")
    print(f"  Target output: (24, 4) = 4 vitals")
    print()

    # Load data
    features, patient_ids, seg_names, window_times = load_data()

    # Group and sort
    groups = group_and_sort(features, patient_ids, seg_names, window_times)

    # Split at temporal gaps
    print("\nVerifying temporal continuity and splitting at gaps...")
    continuous_segments = {}
    total_gaps = 0
    total_sub_segments = 0

    for key, indices in groups.items():
        sub_segs = split_at_gaps(indices, window_times)
        continuous_segments[key] = sub_segs
        total_sub_segments += len(sub_segs)
        if len(sub_segs) > 1:
            total_gaps += len(sub_segs) - 1

    print(f"  Temporal gaps found: {total_gaps:,}")
    print(f"  Continuous sub-segments: {total_sub_segments:,}")

    # Create sliding windows (vitals-only, but same NaN filter on all 11 features)
    windows, patient_ids_out = create_sliding_windows(
        continuous_segments, features, window_times)

    if len(windows) == 0:
        print("ERROR: No valid windows found!")
        return

    # Split patients (same seed as full model → same split)
    train_patients, val_patients, test_patients = split_patients(patient_ids_out)

    # Compute normalization stats (vitals-only, from training set)
    means, stds = compute_norm_stats(windows, patient_ids_out, train_patients)

    # Normalize
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

    # Save normalization parameters (vitals-only)
    norm_params = {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'feature_names': VITAL_NAMES,
        'vital_indices': [0, 1, 2, 3],  # In the 4-dim space, vitals are at [0-3]
        'vital_names': VITAL_NAMES,
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
        'ablation': 'vitals_only',
        'input_features': 5,
        'description': 'Vitals-only ablation: 4 vitals + 1 time (no correlation features)',
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"  Train windows: {split_info['n_train_windows']:,}")
    print(f"  Val windows:   {split_info['n_val_windows']:,}")
    print(f"  Test windows:  {split_info['n_test_windows']:,}")
    print(f"  Total:         {split_info['n_train_windows'] + split_info['n_val_windows'] + split_info['n_test_windows']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
