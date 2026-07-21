"""
Phase 41 TFT Preprocessing
---------------------------
Adapted from Phase 4 preprocess.py for 15-minute resolution with 4 vital signs.
Loads per-patient .npy files, applies forward-fill imputation, sliding window,
and Z-score normalization for Temporal Fusion Transformer training.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = '/gpfs/scratch/dk5565/phase41_data'
SIGNAL_NAMES = ['mean_bp', 'pulse', 'spo2', 'respiratory_rate']
NUM_SIGNALS = 4
PAST_MONTHS = 75        # 75 steps × 15 min = 18.75 hours
FUTURE_MONTHS = 25      # 25 steps × 15 min = 6.25 hours
INTERVAL_MINUTES = 15
NUM_HISTORICAL_NUMERIC = 9   # 4 vitals + 4 masks + 1 time = 9
NUM_FUTURE_NUMERIC = 1       # time position only
NUM_STATIC_NUMERIC = 1       # placeholder
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
BATCH_SIZE = 64
RANDOM_SEED = 42
STRIDE = 12             # 3 hours × 4 = 12 steps

WINDOW_SIZE = PAST_MONTHS + FUTURE_MONTHS  # 400 steps total


# ============================================================================
# Helper Functions
# ============================================================================

def forward_fill(data):
    """
    Forward-fill NaN values along axis 0 (time), then backfill leading NaNs.
    Input: (N, 4) array with possible NaN values.
    Output: (N, 4) array with no NaN values.
    """
    filled = data.copy()
    for col in range(filled.shape[1]):
        series = filled[:, col]
        # Forward fill
        mask = np.isnan(series)
        if not mask.all():
            # Find first valid index for backfill
            first_valid = np.where(~mask)[0][0]
            # Forward fill from first valid onward
            for i in range(first_valid + 1, len(series)):
                if mask[i]:
                    series[i] = series[i - 1]
            # Backfill leading NaNs with first valid value
            if first_valid > 0:
                series[:first_valid] = series[first_valid]
        else:
            # Entire column is NaN — fill with 0
            series[:] = 0.0
    return filled


def compute_normalization_stats(patient_files, data_dir):
    """
    Compute per-signal mean and std from training patients' real (non-NaN) values only.
    Returns dict with 'mean' and 'std' arrays of shape (4,).
    """
    sums = np.zeros(NUM_SIGNALS)
    sum_sq = np.zeros(NUM_SIGNALS)
    counts = np.zeros(NUM_SIGNALS)

    for fname in patient_files:
        fpath = os.path.join(data_dir, fname)
        data = np.load(fpath)  # shape: (N, 4)
        for s in range(NUM_SIGNALS):
            col = data[:, s]
            valid = col[~np.isnan(col)]
            sums[s] += valid.sum()
            sum_sq[s] += (valid ** 2).sum()
            counts[s] += len(valid)

    mean = sums / np.maximum(counts, 1)
    std = np.sqrt(sum_sq / np.maximum(counts, 1) - mean ** 2)
    # Prevent division by zero
    std = np.maximum(std, 1e-8)

    return {'mean': mean, 'std': std}


# ============================================================================
# Dataset
# ============================================================================

class TimeSeriesDataset(Dataset):
    """
    Dataset for Phase 41 TFT.
    Loads per-patient .npy files, applies forward-fill, sliding window,
    and Z-score normalization.
    """

    def __init__(self, patient_files, data_dir, norm_params):
        """
        Args:
            patient_files: list of .npy filenames
            data_dir: directory containing .npy files
            norm_params: dict with 'mean' and 'std' arrays of shape (4,)
        """
        self.data_dir = data_dir
        self.norm_mean = norm_params['mean']
        self.norm_std = norm_params['std']

        # Build index of all valid windows across all patients
        self.windows = []  # list of (file_path, raw_mask, filled_data, start_idx)
        self._patient_data = []  # store loaded patient data

        for fname in patient_files:
            fpath = os.path.join(data_dir, fname)
            raw_data = np.load(fpath)  # shape: (N, 4)
            n_steps = raw_data.shape[0]

            if n_steps < WINDOW_SIZE:
                continue

            # Binary mask: 1 where real, 0 where NaN
            mask = (~np.isnan(raw_data)).astype(np.float32)

            # Forward fill
            filled = forward_fill(raw_data).astype(np.float32)

            # Normalize filled data
            normalized = (filled - self.norm_mean) / self.norm_std
            normalized = normalized.astype(np.float32)

            # Store patient data
            patient_idx = len(self._patient_data)
            self._patient_data.append({
                'normalized': normalized,
                'mask': mask,
            })

            # Generate sliding windows
            num_windows = (n_steps - WINDOW_SIZE) // STRIDE + 1
            for w in range(num_windows):
                start = w * STRIDE
                self.windows.append((patient_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        patient_idx, start = self.windows[idx]
        patient = self._patient_data[patient_idx]

        end = start + WINDOW_SIZE
        normalized = patient['normalized'][start:end]  # (400, 4)
        mask = patient['mask'][start:end]              # (400, 4)

        # Time position: linearly spaced 0 to 1 over the full 400-step window
        time_pos = np.linspace(0, 1, WINDOW_SIZE, dtype=np.float32)

        # Split into historical and future
        hist_vals = normalized[:PAST_MONTHS]      # (75, 4)
        hist_mask = mask[:PAST_MONTHS]            # (75, 4)
        hist_time = time_pos[:PAST_MONTHS]        # (75,)

        future_vals = normalized[PAST_MONTHS:]    # (25, 4)
        future_mask = mask[PAST_MONTHS:]          # (25, 4)
        future_time = time_pos[PAST_MONTHS:]      # (25,)

        # historical_ts_numeric: (75, 9) — 4 vitals + 4 masks + 1 time
        historical_ts_numeric = np.concatenate([
            hist_vals,                              # (75, 4)
            hist_mask,                              # (75, 4)
            hist_time[:, np.newaxis],               # (75, 1)
        ], axis=1)  # (75, 9)

        # future_ts_numeric: (25, 1) — time position only
        future_ts_numeric = future_time[:, np.newaxis]  # (25, 1)

        # target: (25, 4) — normalized forward-filled future values
        target = future_vals  # (25, 4)

        # target_mask: (25, 4) — 1=real, 0=imputed
        target_mask = future_mask  # (25, 4)

        # static_feats_numeric: (1,) — placeholder
        static_feats_numeric = np.array([0.0], dtype=np.float32)

        return {
            'static_feats_numeric': torch.from_numpy(static_feats_numeric),
            'historical_ts_numeric': torch.from_numpy(historical_ts_numeric),
            'future_ts_numeric': torch.from_numpy(future_ts_numeric),
            'target': torch.from_numpy(target),
            'target_mask': torch.from_numpy(target_mask),
        }


# ============================================================================
# DataLoader Creation
# ============================================================================

class PreloadedDataset(Dataset):
    """Dataset that wraps pre-saved tensors (from prepare_data.py)."""

    def __init__(self, data_dict):
        self.static = data_dict['static_feats_numeric']
        self.historical = data_dict['historical_ts_numeric']
        self.future = data_dict['future_ts_numeric']
        self.target = data_dict['target']
        self.target_mask = data_dict['target_mask']

    def __len__(self):
        return self.historical.shape[0]

    def __getitem__(self, idx):
        return {
            'static_feats_numeric': self.static[idx],
            'historical_ts_numeric': self.historical[idx],
            'future_ts_numeric': self.future[idx],
            'target': self.target[idx],
            'target_mask': self.target_mask[idx],
        }


def create_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=4):
    """
    Create train, validation, and test DataLoaders.

    If pre-processed .pt files exist (from prepare_data.py), loads those directly.
    Otherwise falls back to on-the-fly processing from .npy files.

    Returns:
        train_loader, val_loader, test_loader, norm_params
    """
    processed_dir = os.path.join(data_dir, 'processed')
    train_pt = os.path.join(processed_dir, 'train_data.pt')
    val_pt = os.path.join(processed_dir, 'val_data.pt')
    test_pt = os.path.join(processed_dir, 'test_data.pt')
    norm_json = os.path.join(processed_dir, 'norm_params.json')

    # Fast path: load pre-processed tensors
    if all(os.path.exists(p) for p in [train_pt, val_pt, test_pt, norm_json]):
        print("[INFO] Loading pre-processed tensors (fast path)...")

        with open(norm_json, 'r') as f:
            norm_data = json.load(f)
        norm_params = {
            'mean': np.array(norm_data['mean']),
            'std': np.array(norm_data['std']),
        }

        train_data = torch.load(train_pt, weights_only=False)
        val_data = torch.load(val_pt, weights_only=False)
        test_data = torch.load(test_pt, weights_only=False)

        train_dataset = PreloadedDataset(train_data)
        val_dataset = PreloadedDataset(val_data)
        test_dataset = PreloadedDataset(test_data)

        print(f"  Train: {len(train_dataset)} windows")
        print(f"  Val:   {len(val_dataset)} windows")
        print(f"  Test:  {len(test_dataset)} windows")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False)

        return train_loader, val_loader, test_loader, norm_params

    # Slow path: process from .npy files on the fly
    print("[INFO] No pre-processed data found. Processing from .npy files...")
    print("       (Run prepare_data.py first for faster subsequent runs)")

    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    patient_files = [m['patient_id'] + '.npy' for m in metadata]

    # Shuffle and split at patient level
    rng = np.random.RandomState(RANDOM_SEED)
    indices = np.arange(len(patient_files))
    rng.shuffle(indices)

    n_total = len(patient_files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_files = [patient_files[i] for i in indices[:n_train]]
    val_files = [patient_files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [patient_files[i] for i in indices[n_train + n_val:]]

    print(f"Patient split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Compute normalization stats from training patients only
    print("Computing normalization statistics from training data...")
    norm_params = compute_normalization_stats(train_files, data_dir)
    print(f"  Mean: {norm_params['mean']}")
    print(f"  Std:  {norm_params['std']}")

    # Create datasets
    print("Building training dataset...")
    train_dataset = TimeSeriesDataset(train_files, data_dir, norm_params)
    print(f"  Training windows: {len(train_dataset)}")

    print("Building validation dataset...")
    val_dataset = TimeSeriesDataset(val_files, data_dir, norm_params)
    print(f"  Validation windows: {len(val_dataset)}")

    print("Building test dataset...")
    test_dataset = TimeSeriesDataset(test_files, data_dir, norm_params)
    print(f"  Test windows: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader, norm_params


# ============================================================================
# Main (for standalone testing)
# ============================================================================

if __name__ == '__main__':
    train_loader, val_loader, test_loader, norm_params = create_dataloaders()
    print(f"\nDataLoaders created successfully.")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Sample a batch
    batch = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    for key, val in batch.items():
        print(f"  {key}: {val.shape}")
