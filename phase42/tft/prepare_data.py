#!/usr/bin/env python3
"""
prepare_data.py - Prepare complete-data-only windows for Phase 42 TFT-multi.

Key difference from Phase 41: NO forward-fill, NO masks.
Only windows where ALL 4 vitals have real values at EVERY timestep (100 steps)
are kept. This gives fewer windows but cleaner signal.

Source data: /gpfs/scratch/dk5565/phase41_data (reuses phase41 .npy files)
Output:      /gpfs/scratch/dk5565/phase42_data/processed/

Steps:
    1. Load metadata from phase41
    2. Split patients (80/10/10, seed=42)
    3. Compute normalization stats from training patients (real values only)
    4. Generate sliding windows, SKIP any with NaN, save as .pt tensors

Output:
    - train_data.pt, val_data.pt, test_data.pt
    - norm_params.json
    - split_info.json

Usage:
    python prepare_data.py [--skip-download]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import (
    SIGNAL_NAMES, NUM_SIGNALS,
    PAST_MONTHS, FUTURE_MONTHS, WINDOW_SIZE, STRIDE,
    NUM_HISTORICAL_NUMERIC, NUM_FUTURE_NUMERIC, NUM_STATIC_NUMERIC,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    compute_normalization_stats,
)

# Source: reuse phase41 extracted .npy files
DATA_DIR = '/gpfs/scratch/dk5565/phase41_data'
# Output: phase42 processed tensors
PROCESSED_DIR = '/gpfs/scratch/dk5565/phase42_data/processed'


def generate_windows_for_patients(patient_files, data_dir, norm_params):
    """
    Generate sliding windows for a list of patients, keeping ONLY complete windows.

    A window is complete if ALL 4 signals have real (non-NaN) values at EVERY
    timestep in the 100-step window.

    Returns dict of stacked tensors and (kept_count, discarded_count).
    """
    norm_mean = norm_params['mean'].astype(np.float32)
    norm_std = norm_params['std'].astype(np.float32)

    all_static = []
    all_historical = []
    all_future = []
    all_target = []
    all_target_mask = []

    time_pos = np.linspace(0, 1, WINDOW_SIZE, dtype=np.float32)

    kept = 0
    discarded = 0

    for fname in patient_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        raw_data = np.load(fpath)  # (N, 4)
        n_steps = raw_data.shape[0]

        if n_steps < WINDOW_SIZE:
            continue

        # Generate sliding windows
        num_windows = (n_steps - WINDOW_SIZE) // STRIDE + 1

        for w in range(num_windows):
            start = w * STRIDE
            end = start + WINDOW_SIZE

            window_raw = raw_data[start:end]  # (100, 4)

            # Completeness check: no NaN anywhere in the window
            if np.any(np.isnan(window_raw)):
                discarded += 1
                continue

            kept += 1

            # Normalize (all values are real)
            win_norm = ((window_raw - norm_mean) / norm_std).astype(np.float32)

            # Split into historical and future
            hist_vals = win_norm[:PAST_MONTHS]       # (75, 4)
            hist_time = time_pos[:PAST_MONTHS]       # (75,)

            future_vals = win_norm[PAST_MONTHS:]     # (25, 4)
            future_time = time_pos[PAST_MONTHS:]     # (25,)

            # historical_ts_numeric: (75, 5) — 4 vitals + 1 time position
            historical_ts_numeric = np.concatenate([
                hist_vals,                            # (75, 4)
                hist_time[:, np.newaxis],             # (75, 1)
            ], axis=1)  # (75, 5)

            # future_ts_numeric: (25, 1) — time position only
            future_ts_numeric = future_time[:, np.newaxis]  # (25, 1)

            # target: (25, 4) — normalized future vitals (all real)
            target = future_vals  # (25, 4)

            # target_mask: (25, 4) — all 1.0 (complete data guaranteed)
            target_mask = np.ones((FUTURE_MONTHS, NUM_SIGNALS), dtype=np.float32)

            # static: (1,) — placeholder
            static_feats_numeric = np.array([0.0], dtype=np.float32)

            all_static.append(static_feats_numeric)
            all_historical.append(historical_ts_numeric)
            all_future.append(future_ts_numeric)
            all_target.append(target)
            all_target_mask.append(target_mask)

    if not all_static:
        return None, kept, discarded

    data = {
        'static_feats_numeric': torch.from_numpy(np.stack(all_static)),
        'historical_ts_numeric': torch.from_numpy(np.stack(all_historical)),
        'future_ts_numeric': torch.from_numpy(np.stack(all_future)),
        'target': torch.from_numpy(np.stack(all_target)),
        'target_mask': torch.from_numpy(np.stack(all_target_mask)),
    }
    return data, kept, discarded


def main():
    parser = argparse.ArgumentParser(
        description='Prepare complete-data-only windows for Phase 42 TFT-multi')
    parser.add_argument('--skip-download', action='store_true', default=True,
                        help='Skip extraction (reuse phase41 data, default: True)')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 42 — Data Preparation Pipeline (Complete Windows Only)")
    print("=" * 70)
    print(f"  Source data dir: {DATA_DIR}")
    print(f"  Output dir:      {PROCESSED_DIR}")
    print(f"  Window:          {PAST_MONTHS} input + {FUTURE_MONTHS} output = {WINDOW_SIZE} steps")
    print(f"  Stride:          {STRIDE} steps ({STRIDE * 15 / 60:.1f} hours)")
    print(f"  Resolution:      15-minute intervals")
    print(f"  Key insight:     NO missing data, NO masks, NO forward-fill")
    print(f"                   Only windows with complete data across all 4 vitals")
    print()

    start_time = time.time()

    # =========================================================================
    # Step 1: Load metadata from phase41
    # =========================================================================
    metadata_path = os.path.join(DATA_DIR, 'metadata.json')

    if not os.path.exists(metadata_path):
        print(f"ERROR: metadata.json not found at {metadata_path}")
        print("       Phase 41 data must be extracted first.")
        sys.exit(1)

    print("[STEP 1] Loading metadata from phase41...")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get list of .npy files
    patient_files = [m['patient_id'] + '.npy' for m in metadata]
    print(f"  Total patients from phase41: {len(patient_files)}")

    # =========================================================================
    # Step 2: Patient-level split
    # =========================================================================
    print("[STEP 2] Splitting patients (80/10/10, seed=42)...")

    rng = np.random.RandomState(RANDOM_SEED)
    indices = np.arange(len(patient_files))
    rng.shuffle(indices)

    n_total = len(patient_files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_files = [patient_files[i] for i in indices[:n_train]]
    val_files = [patient_files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [patient_files[i] for i in indices[n_train + n_val:]]

    print(f"  Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

    # =========================================================================
    # Step 3: Compute normalization stats from training patients
    # =========================================================================
    print("[STEP 3] Computing normalization statistics from training data...")
    print("         (using real values only, no masking needed)")
    norm_params = compute_normalization_stats(train_files, DATA_DIR)
    print(f"  Mean: {norm_params['mean']}")
    print(f"  Std:  {norm_params['std']}")

    # =========================================================================
    # Step 4: Generate complete-data windows and save tensors
    # =========================================================================
    print("[STEP 4] Generating sliding windows (complete-only) and saving...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    total_kept = 0
    total_discarded = 0

    for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print(f"\n  Processing {split_name} ({len(split_files)} patients)...")
        data, kept, discarded = generate_windows_for_patients(split_files, DATA_DIR, norm_params)

        total_kept += kept
        total_discarded += discarded

        if data is None:
            print(f"    WARNING: No complete windows generated for {split_name}")
            continue

        n_windows = data['historical_ts_numeric'].shape[0]
        print(f"    Windows kept:      {kept}")
        print(f"    Windows discarded: {discarded} (had NaN)")
        print(f"    Keep rate:         {kept / max(kept + discarded, 1) * 100:.1f}%")
        print(f"    Shapes: historical={data['historical_ts_numeric'].shape}, "
              f"future={data['future_ts_numeric'].shape}, "
              f"target={data['target'].shape}")

        # Save as .pt file
        save_path = os.path.join(PROCESSED_DIR, f'{split_name}_data.pt')
        torch.save(data, save_path)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"    Saved: {save_path} ({size_mb:.1f} MB)")

    # Save normalization params
    norm_path = os.path.join(PROCESSED_DIR, 'norm_params.json')
    with open(norm_path, 'w') as f:
        json.dump({
            'mean': norm_params['mean'].tolist(),
            'std': norm_params['std'].tolist(),
            'signal_names': SIGNAL_NAMES,
        }, f, indent=2)
    print(f"\n  Saved: {norm_path}")

    # Save split info
    split_path = os.path.join(PROCESSED_DIR, 'split_info.json')
    with open(split_path, 'w') as f:
        json.dump({
            'train_files': train_files,
            'val_files': val_files,
            'test_files': test_files,
            'train_count': len(train_files),
            'val_count': len(val_files),
            'test_count': len(test_files),
            'random_seed': RANDOM_SEED,
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'past_months': PAST_MONTHS,
            'future_months': FUTURE_MONTHS,
            'phase': 42,
            'description': 'Complete-data-only windows (no NaN, no forward-fill, no masks)',
        }, f, indent=2)
    print(f"  Saved: {split_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Phase 42 data preparation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)
    print(f"\n  Total windows kept:      {total_kept}")
    print(f"  Total windows discarded: {total_discarded}")
    print(f"  Overall keep rate:       {total_kept / max(total_kept + total_discarded, 1) * 100:.1f}%")
    print(f"\nProcessed data saved to: {PROCESSED_DIR}/")
    print("  train_data.pt, val_data.pt, test_data.pt")
    print("  norm_params.json, split_info.json")
    print("\nTraining can now load these directly with torch.load() — no reprocessing needed.")


if __name__ == '__main__':
    main()
