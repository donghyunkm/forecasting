#!/usr/bin/env python3
"""
Phase 6.1 Data Preparation: Correlation forecasting with physiological features.

Extends Phase 6 by adding summarized X_stats features to the model input.
X_stats.npy contains 19 physiological features computed in 109 sliding sub-windows
(10s stride, 120s sub-window) per 2.5-min data point. We summarize the 109 sub-windows
into mean and std per feature = 38 additional input channels.

Data source: /gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/
  - corr_features_focused.npy  (564596, 7)  — 7 pairwise correlations per window
  - X_stats.npy                (564596, 19, 109) — 19 physiological features × 109 sub-windows
  - patient_ids.npy            (564596,) — patient identifiers
  - seg_names.npy              (564596,) — segment identifiers
  - window_times.npy           (564596,) — timestamps (seconds)
  - block_start_times.npy      (564596,) — block start times (for continuity grouping)

Resolution: 2.5 minutes (150 seconds) between windows.

Input: 7 correlations + 38 physio stats (19 mean + 19 std) + 1 time position = 46 channels
Output: 7 correlations (12 steps = 30min)
"""

import os
import json
import numpy as np
import torch
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_SOURCE = "/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/"
OUTPUT_DIR = "/gpfs/home/dk5565/forecasting/phase61/phase61_data/processed/"

WINDOW_TOTAL = 60       # 48 history + 12 forecast
WINDOW_HISTORY = 48     # 2h at 2.5-min resolution
WINDOW_FORECAST = 12    # 30min at 2.5-min resolution
STRIDE = 12             # 30-min stride
EXPECTED_DT = 150       # 2.5 minutes in seconds

NUM_CORR = 7            # 7 focused correlation pairs
NUM_PHYSIO = 19         # 19 physiological features from X_stats
NUM_PHYSIO_STATS = 38   # 19 mean + 19 std

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

# All input feature names: 7 corr + 19 mean + 19 std + 1 time
INPUT_FEATURE_NAMES = (
    CORRELATION_NAMES +
    [f"{name}_mean" for name in PHYSIO_FEATURE_NAMES] +
    [f"{name}_std" for name in PHYSIO_FEATURE_NAMES] +
    ['time_position']
)


def load_data():
    """Load arrays from data_m3_120s_prediction."""
    print("Loading data from data_m3_120s_prediction...")
    corr_features = np.load(os.path.join(DATA_SOURCE, "corr_features_focused.npy"))  # (N, 7)
    x_stats = np.load(os.path.join(DATA_SOURCE, "X_stats.npy"), mmap_mode='r')  # (N, 19, 109)
    patient_ids = np.load(os.path.join(DATA_SOURCE, "patient_ids.npy"), allow_pickle=True)  # (N,)
    seg_names = np.load(os.path.join(DATA_SOURCE, "seg_names.npy"), allow_pickle=True)  # (N,)
    window_times = np.load(os.path.join(DATA_SOURCE, "window_times.npy"))  # (N,)
    block_start_times = np.load(os.path.join(DATA_SOURCE, "block_start_times.npy"))  # (N,)

    print(f"  Total windows: {len(corr_features):,}")
    print(f"  Correlation shape: {corr_features.shape}")
    print(f"  X_stats shape: {x_stats.shape}")
    print(f"  Unique patients: {len(np.unique(patient_ids)):,}")
    print(f"  Unique segments: {len(np.unique(seg_names)):,}")

    return corr_features, x_stats, patient_ids, seg_names, window_times, block_start_times


def summarize_x_stats(x_stats):
    """Summarize X_stats (N, 19, 109) into (N, 38) = mean + std over 109 sub-windows.

    NaN handling: nanmean/nanstd — if all 109 sub-windows are NaN for a feature,
    the result is NaN (will be imputed later after normalization).
    """
    print("\nSummarizing X_stats: (N, 19, 109) → (N, 38) [mean + std]...")
    N = x_stats.shape[0]

    # Process in chunks to manage memory (X_stats is mmap'd)
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

    # Per-feature NaN report
    for i in range(NUM_PHYSIO):
        mean_nan = np.isnan(physio_stats[:, i]).sum()
        std_nan = np.isnan(physio_stats[:, i + NUM_PHYSIO]).sum()
        if mean_nan > 0 or std_nan > 0:
            print(f"    {PHYSIO_FEATURE_NAMES[i]}: mean_nan={mean_nan:,} ({100*mean_nan/N:.1f}%), "
                  f"std_nan={std_nan:,} ({100*std_nan/N:.1f}%)")

    return physio_stats


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
                # Split at gaps
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


def create_sliding_windows(continuous_segments, corr_features, physio_stats):
    """Create sliding windows from continuous segments.

    Each window: (WINDOW_TOTAL, 7+38) = (60, 45) — correlations + physio stats.
    Windows with >50% NaN in physio features are excluded.
    """
    print("\nCreating sliding windows...")
    windows_corr = []
    windows_physio = []
    patient_ids_out = []

    total_windows = 0
    nan_rejected = 0

    for pid, indices in continuous_segments:
        seg_corr = corr_features[indices]      # (L, 7)
        seg_physio = physio_stats[indices]      # (L, 38)
        n_windows = (len(indices) - WINDOW_TOTAL) // STRIDE + 1

        for i in range(n_windows):
            start = i * STRIDE
            win_corr = seg_corr[start:start + WINDOW_TOTAL]      # (60, 7)
            win_physio = seg_physio[start:start + WINDOW_TOTAL]   # (60, 38)

            # Check NaN rate in physio features for this window
            nan_rate = np.isnan(win_physio).sum() / win_physio.size
            if nan_rate > 0.5:
                nan_rejected += 1
                continue

            windows_corr.append(win_corr)
            windows_physio.append(win_physio)
            patient_ids_out.append(pid)
            total_windows += 1

    print(f"  Valid windows: {total_windows:,}")
    print(f"  Rejected (>50% NaN in physio): {nan_rejected:,}")

    return (np.array(windows_corr, dtype=np.float32),
            np.array(windows_physio, dtype=np.float32),
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


def compute_norm_stats(windows_corr, windows_physio, patient_ids_out, train_patients):
    """Compute normalization statistics from training patients.

    - Correlations: clip to ±0.9999 → arctanh (Fisher z) → z-score
    - Physio stats: z-score normalize (ignoring NaN)
    """
    print("\nComputing normalization statistics from training set...")
    train_mask = np.array([pid in train_patients for pid in patient_ids_out])

    # --- Correlation normalization (Fisher z-space) ---
    train_corr = windows_corr[train_mask]  # (N_train, 60, 7)
    train_z = np.arctanh(np.clip(train_corr, -0.9999, 0.9999))
    flat_z = train_z.reshape(-1, NUM_CORR)
    corr_means = np.nanmean(flat_z, axis=0)
    corr_stds = np.nanstd(flat_z, axis=0)
    corr_stds[corr_stds < 1e-8] = 1.0

    print(f"  Correlation Fisher z-space means: {corr_means}")
    print(f"  Correlation Fisher z-space stds: {corr_stds}")

    # --- Physio stats normalization (z-score in raw space) ---
    train_physio = windows_physio[train_mask]  # (N_train, 60, 38)
    flat_physio = train_physio.reshape(-1, NUM_PHYSIO_STATS)
    physio_means = np.nanmean(flat_physio, axis=0)
    physio_stds = np.nanstd(flat_physio, axis=0)
    physio_stds[physio_stds < 1e-8] = 1.0

    print(f"  Physio means (first 5): {physio_means[:5]}")
    print(f"  Physio stds (first 5): {physio_stds[:5]}")

    return corr_means, corr_stds, physio_means, physio_stds


def normalize_data(windows_corr, windows_physio, corr_means, corr_stds, physio_means, physio_stds):
    """Apply normalization to both correlation and physio data.

    - Correlations: clip → Fisher z → z-score
    - Physio: z-score normalize, then fill NaN with 0 (= population mean in z-space)
    """
    print("\nNormalizing data...")

    # Correlations: Fisher z + z-score
    corr_z = np.arctanh(np.clip(windows_corr, -0.9999, 0.9999))
    corr_norm = (corr_z - corr_means[np.newaxis, np.newaxis, :]) / corr_stds[np.newaxis, np.newaxis, :]

    # Physio: z-score + NaN imputation
    physio_norm = (windows_physio - physio_means[np.newaxis, np.newaxis, :]) / physio_stds[np.newaxis, np.newaxis, :]

    # Impute NaN with 0 (population mean in z-score space)
    nan_before = np.isnan(physio_norm).sum()
    physio_norm = np.nan_to_num(physio_norm, nan=0.0)
    print(f"  Imputed {nan_before:,} NaN values with 0 in physio features")

    return corr_norm, physio_norm


def build_tensors(corr_norm, physio_norm, patient_ids_out, split_patients_set, split_name):
    """Build tensors for a given split.

    Input: 48 steps × 46 (7 correlations + 38 physio stats + 1 time position)
    Output (target): 12 steps × 7 (7 correlations)
    """
    mask = np.array([pid in split_patients_set for pid in patient_ids_out])
    split_corr = corr_norm[mask]      # (N, 60, 7)
    split_physio = physio_norm[mask]   # (N, 60, 38)
    N = split_corr.shape[0]

    if N == 0:
        print(f"  WARNING: {split_name} has 0 windows!")
        return None

    # Historical: first 48 steps
    hist_corr = split_corr[:, :WINDOW_HISTORY, :]        # (N, 48, 7)
    hist_physio = split_physio[:, :WINDOW_HISTORY, :]     # (N, 48, 38)

    # Time position for history: linear 0 → 0.75
    time_hist = np.linspace(0.0, 0.75, WINDOW_HISTORY, dtype=np.float32)
    time_hist = np.tile(time_hist, (N, 1))[:, :, np.newaxis]  # (N, 48, 1)

    # Concatenate: 7 corr + 38 physio + 1 time = 46
    historical_ts_numeric = np.concatenate([hist_corr, hist_physio, time_hist], axis=2)  # (N, 48, 46)

    # Future: time position only (known into the future)
    time_future = np.linspace(0.76, 1.0, WINDOW_FORECAST, dtype=np.float32)
    time_future = np.tile(time_future, (N, 1))[:, :, np.newaxis]  # (N, 12, 1)
    future_ts_numeric = time_future

    # Target: 7 correlations for future 12 steps (already normalized)
    target = split_corr[:, WINDOW_HISTORY:, :]  # (N, 12, 7)

    # Target mask: all ones
    target_mask = np.ones((N, WINDOW_FORECAST, NUM_CORR), dtype=np.float32)

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
    print(f"    future_ts_numeric: {data_dict['future_ts_numeric'].shape}")
    print(f"    target: {data_dict['target'].shape}")

    return data_dict


def main():
    print("=" * 70)
    print("Phase 6.1 Data Preparation — Correlation + Physio Features Forecasting")
    print("  Source: data_m3_120s_prediction (2.5-min resolution)")
    print("  Input:  48 steps (2h) × 46 (7 corr + 38 physio stats + 1 time)")
    print("  Output: 12 steps (30min) × 7 correlations")
    print("=" * 70)

    # Load data
    corr_features, x_stats, patient_ids, seg_names, window_times, block_start_times = load_data()

    # Summarize X_stats: (N, 19, 109) → (N, 38)
    physio_stats = summarize_x_stats(x_stats)

    # Group into continuous segments
    continuous_segments = group_into_continuous_segments(
        corr_features, patient_ids, seg_names, window_times, block_start_times)

    # Create sliding windows
    windows_corr, windows_physio, patient_ids_out = create_sliding_windows(
        continuous_segments, corr_features, physio_stats)

    if len(windows_corr) == 0:
        print("ERROR: No valid windows found! Check data.")
        return

    print(f"\n  Correlation window shape: {windows_corr.shape}  (expected: (N, {WINDOW_TOTAL}, 7))")
    print(f"  Physio window shape: {windows_physio.shape}  (expected: (N, {WINDOW_TOTAL}, 38))")

    # Split patients
    train_patients, val_patients, test_patients = split_patients(patient_ids_out)

    # Compute normalization stats from training set
    corr_means, corr_stds, physio_means, physio_stds = compute_norm_stats(
        windows_corr, windows_physio, patient_ids_out, train_patients)

    # Normalize all windows
    corr_norm, physio_norm = normalize_data(
        windows_corr, windows_physio, corr_means, corr_stds, physio_means, physio_stds)

    # Build tensors for each split
    print("\nBuilding tensors...")
    train_data = build_tensors(corr_norm, physio_norm, patient_ids_out, train_patients, "train")
    val_data = build_tensors(corr_norm, physio_norm, patient_ids_out, val_patients, "val")
    test_data = build_tensors(corr_norm, physio_norm, patient_ids_out, test_patients, "test")

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
        'corr_means': corr_means.tolist(),
        'corr_stds': corr_stds.tolist(),
        'physio_means': physio_means.tolist(),
        'physio_stds': physio_stds.tolist(),
        'correlation_names': CORRELATION_NAMES,
        'physio_feature_names': PHYSIO_FEATURE_NAMES,
        'input_feature_names': INPUT_FEATURE_NAMES,
        'num_corr': NUM_CORR,
        'num_physio_stats': NUM_PHYSIO_STATS,
        'num_input_features': NUM_CORR + NUM_PHYSIO_STATS + 1,  # 46
        'corr_transform': 'fisher_z',
        'corr_clip_range': [-0.9999, 0.9999],
        'physio_transform': 'z-score',
        'physio_nan_imputation': 'zero (population mean in z-space)',
        'description': (
            'Correlations: clip ±0.9999 → arctanh (Fisher z) → z-score. '
            'Physio stats: z-score normalize → NaN imputed with 0.'
        ),
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
        'nan_filter': 'windows with >50% NaN in physio features excluded',
        'input_channels': {
            'total': NUM_CORR + NUM_PHYSIO_STATS + 1,
            'correlations': NUM_CORR,
            'physio_stats': NUM_PHYSIO_STATS,
            'time_position': 1,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"  Train windows: {split_info['n_train_windows']:,}")
    print(f"  Val windows:   {split_info['n_val_windows']:,}")
    print(f"  Test windows:  {split_info['n_test_windows']:,}")
    print(f"  Total:         {split_info['n_train_windows'] + split_info['n_val_windows'] + split_info['n_test_windows']:,}")
    print(f"  Input:         {NUM_CORR} corr + {NUM_PHYSIO_STATS} physio + 1 time = {NUM_CORR + NUM_PHYSIO_STATS + 1} channels")
    print(f"  Output:        {NUM_CORR} correlations")
    print(f"  Resolution:    2.5 min (150s)")
    print(f"  History:       {WINDOW_HISTORY} steps = {WINDOW_HISTORY * 2.5:.0f} min")
    print(f"  Forecast:      {WINDOW_FORECAST} steps = {WINDOW_FORECAST * 2.5:.0f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
