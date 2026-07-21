#!/usr/bin/env python3
"""
prepare_data.py - Extract, preprocess, and save all data for Phase 41 TFT-multi.

This script runs the full data pipeline and saves pre-processed tensors to disk
so that subsequent model training runs load instantly without reprocessing.

Steps:
    1. Extract vital signs from WFDB numerics (calls download_data.py logic)
    2. Split patients (80/10/10, seed=42)
    3. Compute normalization stats from training patients
    4. Generate all sliding windows, normalize, and save as .pt tensor files

Output (saved to /gpfs/scratch/dk5565/phase41_data/processed/):
    - train_data.pt: dict with keys 'static_feats_numeric', 'historical_ts_numeric',
                     'future_ts_numeric', 'target', 'target_mask'  (all concatenated tensors)
    - val_data.pt: same structure
    - test_data.pt: same structure
    - norm_params.json: normalization mean/std
    - split_info.json: patient split metadata

Usage:
    python prepare_data.py [--num-patients 100] [--skip-download]
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
    DATA_DIR, SIGNAL_NAMES, NUM_SIGNALS,
    PAST_MONTHS, FUTURE_MONTHS, WINDOW_SIZE, STRIDE,
    NUM_HISTORICAL_NUMERIC, NUM_FUTURE_NUMERIC, NUM_STATIC_NUMERIC,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    forward_fill, compute_normalization_stats,
)


PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


def generate_windows_for_patients(patient_files, data_dir, norm_params):
    """
    Generate all sliding windows for a list of patients.

    Returns dict of stacked tensors ready for training.
    """
    norm_mean = norm_params['mean'].astype(np.float32)
    norm_std = norm_params['std'].astype(np.float32)

    all_static = []
    all_historical = []
    all_future = []
    all_target = []
    all_target_mask = []

    time_pos = np.linspace(0, 1, WINDOW_SIZE, dtype=np.float32)

    for fname in patient_files:
        fpath = os.path.join(data_dir, fname)
        raw_data = np.load(fpath)  # (N, 4)
        n_steps = raw_data.shape[0]

        if n_steps < WINDOW_SIZE:
            continue

        # Binary mask
        mask = (~np.isnan(raw_data)).astype(np.float32)

        # Forward fill
        filled = forward_fill(raw_data).astype(np.float32)

        # Normalize
        normalized = (filled - norm_mean) / norm_std

        # Generate sliding windows
        num_windows = (n_steps - WINDOW_SIZE) // STRIDE + 1

        for w in range(num_windows):
            start = w * STRIDE
            end = start + WINDOW_SIZE

            win_norm = normalized[start:end]   # (100, 4)
            win_mask = mask[start:end]         # (100, 4)

            # Split
            hist_vals = win_norm[:PAST_MONTHS]       # (75, 4)
            hist_mask = win_mask[:PAST_MONTHS]       # (75, 4)
            hist_time = time_pos[:PAST_MONTHS]       # (75,)

            future_vals = win_norm[PAST_MONTHS:]     # (25, 4)
            future_mask = win_mask[PAST_MONTHS:]     # (25, 4)
            future_time = time_pos[PAST_MONTHS:]     # (25,)

            # historical_ts_numeric: (75, 9)
            historical_ts_numeric = np.concatenate([
                hist_vals,                            # (75, 4)
                hist_mask,                            # (75, 4)
                hist_time[:, np.newaxis],             # (75, 1)
            ], axis=1)

            # future_ts_numeric: (25, 1)
            future_ts_numeric = future_time[:, np.newaxis]

            # static: (1,)
            static_feats_numeric = np.array([0.0], dtype=np.float32)

            all_static.append(static_feats_numeric)
            all_historical.append(historical_ts_numeric)
            all_future.append(future_ts_numeric)
            all_target.append(future_vals)
            all_target_mask.append(future_mask)

    if not all_static:
        return None

    return {
        'static_feats_numeric': torch.from_numpy(np.stack(all_static)),
        'historical_ts_numeric': torch.from_numpy(np.stack(all_historical)),
        'future_ts_numeric': torch.from_numpy(np.stack(all_future)),
        'target': torch.from_numpy(np.stack(all_target)),
        'target_mask': torch.from_numpy(np.stack(all_target_mask)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Prepare all data for Phase 41 TFT-multi (extract + preprocess + save)')
    parser.add_argument('--num-patients', type=int, default=0,
                        help='Number of patients to extract (0 = all qualified, default: 0)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip extraction if .npy files already exist')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 41 — Data Preparation Pipeline")
    print("=" * 70)
    print(f"  Data dir:      {DATA_DIR}")
    print(f"  Processed dir: {PROCESSED_DIR}")
    print(f"  Num patients:  {args.num_patients}")
    print(f"  Window:        {PAST_MONTHS} input + {FUTURE_MONTHS} output = {WINDOW_SIZE} steps")
    print(f"  Stride:        {STRIDE} steps ({STRIDE * 15 / 60:.1f} hours)")
    print(f"  Resolution:    15-minute intervals")
    print()

    start_time = time.time()

    # =========================================================================
    # Step 1: Extract data from WFDB numerics
    # =========================================================================
    metadata_path = os.path.join(DATA_DIR, 'metadata.json')

    if args.skip_download and os.path.exists(metadata_path):
        print("[SKIP] Extraction — metadata.json already exists")
    else:
        print("[STEP 1] Extracting vital signs from WFDB numerics...")
        import subprocess
        cmd = [
            sys.executable, 'download_data.py',
            '--num-patients', str(args.num_patients)
        ]
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            print("ERROR: download_data.py failed")
            sys.exit(1)
        print()

    # =========================================================================
    # Step 2: Load metadata and get patient file list
    # =========================================================================
    print("[STEP 2] Loading metadata and splitting patients...")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get list of .npy files
    patient_files = [m['patient_id'] + '.npy' for m in metadata]
    print(f"  Total patients: {len(patient_files)}")

    # Patient-level split
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
    # Step 3: Compute normalization stats
    # =========================================================================
    print("[STEP 3] Computing normalization statistics from training data...")
    norm_params = compute_normalization_stats(train_files, DATA_DIR)
    print(f"  Mean: {norm_params['mean']}")
    print(f"  Std:  {norm_params['std']}")

    # =========================================================================
    # Step 4: Generate and save preprocessed tensors
    # =========================================================================
    print("[STEP 4] Generating sliding windows and saving tensors...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print(f"\n  Processing {split_name} ({len(split_files)} patients)...")
        data = generate_windows_for_patients(split_files, DATA_DIR, norm_params)

        if data is None:
            print(f"    WARNING: No windows generated for {split_name}")
            continue

        n_windows = data['historical_ts_numeric'].shape[0]
        print(f"    Windows: {n_windows}")
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
        }, f, indent=2)
    print(f"  Saved: {split_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Data preparation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)
    print(f"\nProcessed data saved to: {PROCESSED_DIR}/")
    print("  train_data.pt, val_data.pt, test_data.pt")
    print("  norm_params.json, split_info.json")
    print("\nTraining can now load these directly with torch.load() — no reprocessing needed.")


if __name__ == '__main__':
    main()
