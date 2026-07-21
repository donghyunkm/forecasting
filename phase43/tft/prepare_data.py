"""
Phase 4.3 Data Preparation
===========================
Generates 3 sets of tensors for different forecast horizons (1h, 3h, 6h).
Uses the same complete-window data source as Phase 4.2.

Forecast horizons (at 15-min resolution):
- 1 hour  = 4 steps
- 3 hours = 12 steps
- 6 hours = 24 steps

Input window: 75 steps (18.75h) for all three models.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
RAW_DATA_DIR = '/gpfs/scratch/dk5565/phase41_data'
OUTPUT_BASE_DIR = '/gpfs/scratch/dk5565/phase43_data/processed'

INPUT_STEPS = 75          # 18.75 hours lookback
HORIZONS = [4, 12, 24]   # 1h, 3h, 6h forecast horizons
STRIDE = 12              # Window stride
NUM_SIGNALS = 4          # mean_bp, pulse, spo2, respiratory_rate

SPLIT_SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

SIGNAL_NAMES = ['mean_bp', 'pulse', 'spo2', 'respiratory_rate']


def load_patient_data(raw_dir):
    """Load all patient .npy files from raw data directory."""
    patients = {}
    npy_files = sorted(Path(raw_dir).glob('*.npy'))
    print(f"Found {len(npy_files)} .npy files in {raw_dir}")
    
    for fpath in npy_files:
        patient_id = fpath.stem
        data = np.load(str(fpath), allow_pickle=True)
        patients[patient_id] = data
    
    return patients


def split_patients(patient_ids, seed=42, train_frac=0.8, val_frac=0.1):
    """Split patients into train/val/test sets (80/10/10)."""
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(patient_ids))
    rng.shuffle(ids)
    
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    
    return train_ids, val_ids, test_ids


def extract_windows(patient_data, window_size, stride):
    """
    Extract sliding windows from patient data.
    Only keeps windows with NO NaN values.
    
    Returns list of (window_array,) tuples where window is (window_size, NUM_SIGNALS).
    """
    windows = []
    n_kept = 0
    n_discarded = 0
    
    # patient_data shape: (T, NUM_SIGNALS) or similar
    T = patient_data.shape[0]
    
    for start in range(0, T - window_size + 1, stride):
        window = patient_data[start:start + window_size, :NUM_SIGNALS]
        
        if window.shape != (window_size, NUM_SIGNALS):
            n_discarded += 1
            continue
        
        # Completeness filter: no NaN in entire window
        if np.any(np.isnan(window)):
            n_discarded += 1
            continue
        
        windows.append(window)
        n_kept += 1
    
    return windows, n_kept, n_discarded


def compute_normalization(train_windows):
    """Compute mean and std from training windows (across all signals)."""
    # Stack all training windows: (N, window_size, NUM_SIGNALS)
    all_data = np.concatenate([w.reshape(-1, NUM_SIGNALS) for w in train_windows], axis=0)
    
    mean = np.nanmean(all_data, axis=0).tolist()
    std = np.nanstd(all_data, axis=0).tolist()
    
    # Avoid division by zero
    std = [s if s > 1e-8 else 1.0 for s in std]
    
    return mean, std


def build_tensors(windows, input_steps, horizon, mean, std):
    """
    Build TFT-compatible tensors from windows.
    
    Args:
        windows: list of arrays, each (input_steps + horizon, NUM_SIGNALS)
        input_steps: number of historical steps (75)
        horizon: number of future steps (H)
        mean, std: normalization parameters (lists of length NUM_SIGNALS)
    
    Returns:
        dict with keys: historical_ts_numeric, future_ts_numeric, target, target_mask, static_feats_numeric
    """
    N = len(windows)
    window_size = input_steps + horizon
    
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    
    # Time position: linearly spaced 0-1 over full window
    time_full = np.linspace(0, 1, window_size, dtype=np.float32)
    
    historical_ts_numeric = np.zeros((N, input_steps, 5), dtype=np.float32)  # 4 vitals + 1 time
    future_ts_numeric = np.zeros((N, horizon, 1), dtype=np.float32)          # time position
    target = np.zeros((N, horizon, NUM_SIGNALS), dtype=np.float32)
    target_mask = np.ones((N, horizon, NUM_SIGNALS), dtype=np.float32)       # all 1.0
    static_feats_numeric = np.zeros((N, 1), dtype=np.float32)                # placeholder
    
    for i, window in enumerate(windows):
        # Normalize vitals
        normed = (window.astype(np.float32) - mean_arr) / std_arr
        
        # Historical: first 75 steps, 4 vitals + time
        historical_ts_numeric[i, :, :NUM_SIGNALS] = normed[:input_steps]
        historical_ts_numeric[i, :, NUM_SIGNALS] = time_full[:input_steps]
        
        # Future: time position for horizon steps
        future_ts_numeric[i, :, 0] = time_full[input_steps:]
        
        # Target: normalized vitals for horizon steps
        target[i] = normed[input_steps:]
    
    return {
        'historical_ts_numeric': torch.from_numpy(historical_ts_numeric),
        'future_ts_numeric': torch.from_numpy(future_ts_numeric),
        'target': torch.from_numpy(target),
        'target_mask': torch.from_numpy(target_mask),
        'static_feats_numeric': torch.from_numpy(static_feats_numeric),
    }


def main():
    print("=" * 70)
    print("Phase 4.3 Data Preparation")
    print("=" * 70)
    print(f"Input steps: {INPUT_STEPS}")
    print(f"Horizons: {HORIZONS} (1h, 3h, 6h)")
    print(f"Stride: {STRIDE}")
    print(f"Split: {TRAIN_FRAC}/{VAL_FRAC}/{TEST_FRAC}, seed={SPLIT_SEED}")
    print()
    
    # Load raw patient data
    print("Loading patient data...")
    patients = load_patient_data(RAW_DATA_DIR)
    patient_ids = list(patients.keys())
    print(f"Loaded {len(patient_ids)} patients")
    print()
    
    # Split patients
    train_ids, val_ids, test_ids = split_patients(patient_ids, seed=SPLIT_SEED,
                                                   train_frac=TRAIN_FRAC, val_frac=VAL_FRAC)
    print(f"Patient split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print()
    
    # Compute normalization from training patients using largest window (24)
    # We use all training data (raw, unwindowed) for normalization
    print("Computing normalization from training patients...")
    train_all_data = []
    for pid in train_ids:
        data = patients[pid][:, :NUM_SIGNALS]
        # Only use non-NaN values for normalization
        train_all_data.append(data)
    
    all_train_concat = np.concatenate(train_all_data, axis=0)
    mean = np.nanmean(all_train_concat, axis=0).tolist()
    std = np.nanstd(all_train_concat, axis=0).tolist()
    std = [s if s > 1e-8 else 1.0 for s in std]
    
    print(f"  Mean: {[f'{m:.4f}' for m in mean]}")
    print(f"  Std:  {[f'{s:.4f}' for s in std]}")
    print()
    
    # Process each horizon
    for horizon in HORIZONS:
        window_size = INPUT_STEPS + horizon
        print("-" * 70)
        print(f"Horizon: {horizon} steps ({horizon * 15} min)")
        print(f"  Window size: {INPUT_STEPS} + {horizon} = {window_size} steps")
        print()
        
        # Extract windows per split
        split_windows = {'train': [], 'val': [], 'test': []}
        split_stats = {'train': [0, 0], 'val': [0, 0], 'test': [0, 0]}
        
        for pid in patient_ids:
            if pid in train_ids:
                split_name = 'train'
            elif pid in val_ids:
                split_name = 'val'
            else:
                split_name = 'test'
            
            windows, kept, discarded = extract_windows(patients[pid], window_size, STRIDE)
            split_windows[split_name].extend(windows)
            split_stats[split_name][0] += kept
            split_stats[split_name][1] += discarded
        
        total_kept = sum(s[0] for s in split_stats.values())
        total_discarded = sum(s[1] for s in split_stats.values())
        
        print(f"  Windows kept:      {total_kept}")
        print(f"  Windows discarded: {total_discarded}")
        print(f"  Train: {split_stats['train'][0]} kept, {split_stats['train'][1]} discarded")
        print(f"  Val:   {split_stats['val'][0]} kept, {split_stats['val'][1]} discarded")
        print(f"  Test:  {split_stats['test'][0]} kept, {split_stats['test'][1]} discarded")
        print()
        
        # Build tensors
        print("  Building tensors...")
        train_tensors = build_tensors(split_windows['train'], INPUT_STEPS, horizon, mean, std)
        val_tensors = build_tensors(split_windows['val'], INPUT_STEPS, horizon, mean, std)
        test_tensors = build_tensors(split_windows['test'], INPUT_STEPS, horizon, mean, std)
        
        # Save
        out_dir = os.path.join(OUTPUT_BASE_DIR, f'horizon_{horizon}')
        os.makedirs(out_dir, exist_ok=True)
        
        torch.save(train_tensors, os.path.join(out_dir, 'train_data.pt'))
        torch.save(val_tensors, os.path.join(out_dir, 'val_data.pt'))
        torch.save(test_tensors, os.path.join(out_dir, 'test_data.pt'))
        
        # Save norm params (same for all horizons, saved per directory for convenience)
        norm_params = {'mean': mean, 'std': std, 'signal_names': SIGNAL_NAMES}
        with open(os.path.join(out_dir, 'norm_params.json'), 'w') as f:
            json.dump(norm_params, f, indent=2)
        
        # Save split info
        split_info = {
            'horizon': horizon,
            'input_steps': INPUT_STEPS,
            'window_size': window_size,
            'stride': STRIDE,
            'seed': SPLIT_SEED,
            'n_train_patients': len(train_ids),
            'n_val_patients': len(val_ids),
            'n_test_patients': len(test_ids),
            'n_train_windows': split_stats['train'][0],
            'n_val_windows': split_stats['val'][0],
            'n_test_windows': split_stats['test'][0],
            'n_discarded_windows': total_discarded,
            'train_patient_ids': sorted(list(train_ids)),
            'val_patient_ids': sorted(list(val_ids)),
            'test_patient_ids': sorted(list(test_ids)),
        }
        with open(os.path.join(out_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"  Saved to: {out_dir}")
        print(f"    historical_ts_numeric: {train_tensors['historical_ts_numeric'].shape}")
        print(f"    future_ts_numeric:     {train_tensors['future_ts_numeric'].shape}")
        print(f"    target:                {train_tensors['target'].shape}")
        print(f"    target_mask:           {train_tensors['target_mask'].shape}")
        print(f"    static_feats_numeric:  {train_tensors['static_feats_numeric'].shape}")
        print()
    
    print("=" * 70)
    print("Phase 4.3 data preparation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
