#!/usr/bin/env python3
"""
preprocess.py - Data preparation for TFT-multi vital sign forecasting (Phase 4).

Adapted from the TFT-multi notebook (rosie068/TFT-multi).
Loads MIMIC-III hourly vital sign data and creates the TimeSeriesDataset
matching the exact format expected by the TFT-multi model.

5 vital signs: mean_BP, pulse, SpO2, respiratory_rate, temperature
Window: 75 hours past → 25 hours future

The dataset returns batches in the TFT-multi format:
    - static_feats_numeric: (1,) — placeholder (BMI-like)
    - historical_ts_numeric: (75, 11) — 5 vitals + 5 masks + 1 time position
    - future_ts_numeric: (25, 1) — time position (known into future)
    - target: (25, 5) — future vital sign values (forward-filled)
    - target_mask: (25, 5) — 1=real recorded, 0=imputed
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = '/gpfs/scratch/dk5565/phase4_data'
SIGNAL_NAMES = ['mean_bp', 'pulse', 'spo2', 'respiratory_rate', 'temperature']
NUM_SIGNALS = 5

# Window sizes (matching TFT-multi paper)
PAST_MONTHS = 75    # 75 hours of history (called "past_months" in their code)
FUTURE_MONTHS = 25  # 25 hours to predict (called "future_months" in their code)
INTERVAL_MINUTES = 60  # hourly resolution

# Historical numeric features: 5 vitals + 5 masks + 1 time = 11
NUM_HISTORICAL_NUMERIC = NUM_SIGNALS * 2 + 1  # vitals + masks + time position
# Future numeric features: 1 (time position, known into the future)
NUM_FUTURE_NUMERIC = 1
# Static numeric features: 1 (placeholder)
NUM_STATIC_NUMERIC = 1

# Data splitting
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
BATCH_SIZE = 64
RANDOM_SEED = 42
STRIDE = 12  # stride between windows (hours)


class TimeSeriesDataset(Dataset):
    """
    TFT-multi compatible dataset for vital sign forecasting.

    Directly adapted from the TFT-multi notebook's TimeSeriesDataset class.
    Returns dict with keys matching the model's expected batch format.
    """

    def __init__(self, static_numeric, historical_ts_numeric, future_ts_numeric,
                 target_arr, target_mask):
        """
        Args:
            static_numeric: (N, 1) — static features per sample
            historical_ts_numeric: (N, 75, 11) — past vitals + masks + time
            future_ts_numeric: (N, 25, 1) — future time position
            target_arr: (N, 25, 5) — future vital sign values
            target_mask: (N, 25, 5) — 1=real, 0=imputed
        """
        self.static_numerical = static_numeric
        self.historical_ts_numeric = historical_ts_numeric
        self.future_ts_numeric = future_ts_numeric
        self.target = target_arr
        self.target_mask = target_mask

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            'static_feats_numeric': torch.tensor(
                self.static_numerical[idx], dtype=torch.float32),
            'historical_ts_numeric': torch.tensor(
                self.historical_ts_numeric[idx], dtype=torch.float32),
            'future_ts_numeric': torch.tensor(
                self.future_ts_numeric[idx], dtype=torch.float32),
            'target': torch.tensor(
                self.target[idx], dtype=torch.float32),
            'target_mask': torch.tensor(
                self.target_mask[idx], dtype=torch.int32),
        }


def load_and_prepare_data():
    """
    Load MIMIC-III vital sign data and prepare arrays matching TFT-multi format.

    Mirrors the notebook's data preparation:
    1. Load per-stay .npy files (hourly vital signs)
    2. Create missingness masks
    3. Forward-fill missing values
    4. Create time position features
    5. Window into (past, future) samples

    Returns:
        List of per-stay dicts, each containing windowed arrays for that stay.
    """
    metadata_path = os.path.join(DATA_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}\nRun download_data.py first.")

    with open(metadata_path) as f:
        metadata = json.load(f)

    window_size = PAST_MONTHS + FUTURE_MONTHS

    # Load all stays and extract windows — keep track of which stay each window belongs to
    stays_windows = []  # list of dicts per stay

    print(f"[INFO] Loading {len(metadata['files'])} ICU stays...")

    for fname in metadata['files']:
        filepath = os.path.join(DATA_DIR, fname)
        raw = np.load(filepath)  # (num_hours, 5) with NaN

        if raw.shape[0] < window_size:
            continue

        # Create mask: 1 where real data, 0 where missing
        mask = (~np.isnan(raw)).astype(np.float32)

        # Forward-fill (matching notebook: .ffill(axis=1).bfill(axis=1))
        filled = raw.copy()
        for col in range(NUM_SIGNALS):
            col_data = filled[:, col]
            valid_idx = np.where(~np.isnan(col_data))[0]
            if len(valid_idx) == 0:
                filled[:, col] = 0
                continue
            # Backfill from first valid
            first_valid = valid_idx[0]
            if first_valid > 0:
                col_data[:first_valid] = col_data[first_valid]
            # Forward-fill
            for i in range(1, len(col_data)):
                if np.isnan(col_data[i]):
                    col_data[i] = col_data[i - 1]

        filled = filled.astype(np.float32)

        # Extract windows with stride
        stay_data = []
        stay_mask = []
        stay_time = []
        num_hours = filled.shape[0]
        for start in range(0, num_hours - window_size + 1, STRIDE):
            window_data = filled[start:start + window_size]
            window_mask = mask[start:start + window_size]
            window_time = np.linspace(0, 1, window_size).reshape(-1, 1).astype(np.float32)

            stay_data.append(window_data)
            stay_mask.append(window_mask)
            stay_time.append(window_time)

        if stay_data:
            stays_windows.append({
                'data': np.array(stay_data),   # (n_windows, 100, 5)
                'mask': np.array(stay_mask),   # (n_windows, 100, 5)
                'time': np.array(stay_time),   # (n_windows, 100, 1)
                'file': fname,
            })

    total_windows = sum(s['data'].shape[0] for s in stays_windows)
    print(f"[INFO] {len(stays_windows)} stays, {total_windows} total windows")

    return stays_windows


def _build_arrays(stays_list):
    """Convert a list of per-stay dicts into flat arrays for TimeSeriesDataset."""
    all_data = np.concatenate([s['data'] for s in stays_list], axis=0)   # (N, 100, 5)
    all_mask = np.concatenate([s['mask'] for s in stays_list], axis=0)   # (N, 100, 5)
    all_time = np.concatenate([s['time'] for s in stays_list], axis=0)   # (N, 100, 1)

    targets = all_data[:, PAST_MONTHS:, :]          # (N, 25, 5)
    targets_masks = all_mask[:, PAST_MONTHS:, :]    # (N, 25, 5)

    hist_vitals = all_data[:, :PAST_MONTHS, :]      # (N, 75, 5)
    hist_masks = all_mask[:, :PAST_MONTHS, :]       # (N, 75, 5)
    hist_time = all_time[:, :PAST_MONTHS, :]        # (N, 75, 1)
    historical_ts_numeric = np.concatenate(
        [hist_vitals, hist_masks, hist_time], axis=-1)  # (N, 75, 11)

    future_ts_numeric = all_time[:, PAST_MONTHS:, :]    # (N, 25, 1)
    static_numeric = np.zeros((len(targets), 1), dtype=np.float32)

    return static_numeric, historical_ts_numeric, future_ts_numeric, targets, targets_masks


def create_dataloaders(batch_size=BATCH_SIZE):
    """
    Create train/val/test DataLoaders with patient-level split (no data leakage).

    All windows from the same ICU stay go to the same split, ensuring no
    overlap between train/val/test patients.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, norm_params)
    """
    stays_windows = load_and_prepare_data()
    n_stays = len(stays_windows)

    # Patient-level split
    np.random.seed(RANDOM_SEED)
    stay_indices = np.random.permutation(n_stays)

    n_train = int(n_stays * TRAIN_RATIO)
    n_val = int(n_stays * VAL_RATIO)

    train_stay_idx = stay_indices[:n_train]
    val_stay_idx = stay_indices[n_train:n_train + n_val]
    test_stay_idx = stay_indices[n_train + n_val:]

    train_stays = [stays_windows[i] for i in train_stay_idx]
    val_stays = [stays_windows[i] for i in val_stay_idx]
    test_stays = [stays_windows[i] for i in test_stay_idx]

    n_train_windows = sum(s['data'].shape[0] for s in train_stays)
    n_val_windows = sum(s['data'].shape[0] for s in val_stays)
    n_test_windows = sum(s['data'].shape[0] for s in test_stays)

    print(f"[INFO] Patient-level split (no leakage):")
    print(f"       Train: {len(train_stays)} stays, {n_train_windows} windows")
    print(f"       Val:   {len(val_stays)} stays, {n_val_windows} windows")
    print(f"       Test:  {len(test_stays)} stays, {n_test_windows} windows")

    # Build arrays for each split
    (train_static, train_hist, train_future,
     train_targets, train_masks) = _build_arrays(train_stays)
    (val_static, val_hist, val_future,
     val_targets, val_masks) = _build_arrays(val_stays)
    (test_static, test_hist, test_future,
     test_targets, test_masks) = _build_arrays(test_stays)

    # Compute normalization from training set only (on real values)
    train_hist_vitals = train_hist[:, :, :NUM_SIGNALS]
    train_hist_mask_vals = train_hist[:, :, NUM_SIGNALS:2*NUM_SIGNALS]

    norm_mean = np.zeros(NUM_SIGNALS, dtype=np.float32)
    norm_std = np.ones(NUM_SIGNALS, dtype=np.float32)
    for i in range(NUM_SIGNALS):
        real_vals = train_hist_vitals[:, :, i][train_hist_mask_vals[:, :, i] > 0]
        if len(real_vals) > 0:
            norm_mean[i] = real_vals.mean()
            norm_std[i] = real_vals.std()
            if norm_std[i] < 1e-6:
                norm_std[i] = 1.0

    print(f"[INFO] Normalization (from train real values):")
    for i, name in enumerate(SIGNAL_NAMES):
        print(f"       {name}: mean={norm_mean[i]:.2f}, std={norm_std[i]:.2f}")

    norm_params = {'mean': norm_mean, 'std': norm_std}

    # Normalize vitals (first 5 channels of historical, and targets)
    train_hist[:, :, :NUM_SIGNALS] = (train_hist[:, :, :NUM_SIGNALS] - norm_mean) / norm_std
    val_hist[:, :, :NUM_SIGNALS] = (val_hist[:, :, :NUM_SIGNALS] - norm_mean) / norm_std
    test_hist[:, :, :NUM_SIGNALS] = (test_hist[:, :, :NUM_SIGNALS] - norm_mean) / norm_std
    train_targets = (train_targets - norm_mean) / norm_std
    val_targets = (val_targets - norm_mean) / norm_std
    test_targets = (test_targets - norm_mean) / norm_std

    # Create datasets
    train_dataset = TimeSeriesDataset(train_static, train_hist, train_future,
                                      train_targets, train_masks)
    val_dataset = TimeSeriesDataset(val_static, val_hist, val_future,
                                    val_targets, val_masks)
    test_dataset = TimeSeriesDataset(test_static, test_hist, test_future,
                                     test_targets, test_masks)

    print(f"[INFO] Dataset sizes: train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, drop_last=False)

    print(f"[INFO] Batches: train={len(train_loader)}, "
          f"val={len(val_loader)}, test={len(test_loader)}")

    return train_loader, val_loader, test_loader, norm_params


if __name__ == '__main__':
    train_loader, val_loader, test_loader, norm_params = create_dataloaders()

    # Show sample batch shapes (matching notebook's verification)
    for data in train_loader:
        print(f"\nSample batch shapes:")
        print(f"  static_feats_numeric:  {data['static_feats_numeric'].shape}")
        print(f"  historical_ts_numeric: {data['historical_ts_numeric'].shape}")
        print(f"  future_ts_numeric:     {data['future_ts_numeric'].shape}")
        print(f"  target:                {data['target'].shape}")
        print(f"  target_mask:           {data['target_mask'].shape}")
        break
