#!/usr/bin/env python3
"""
preprocess.py - Preprocessing module for MIMIC-III waveform forecasting (Diffusion).

Loads raw .npy waveform data, applies z-score normalization, and creates
sliding window datasets. Each dataset uses all 3 signals (ABP, PLETH, II)
as input and targets a single signal for forecasting.

DATA LEAKAGE PREVENTION:
- The raw time series is split into contiguous train/val/test blocks BEFORE
  creating sliding windows. This ensures no temporal overlap between splits.
- Normalization statistics are computed from training data only.

Usage:
    python preprocess.py          # Print dataset statistics
    import preprocess             # Use as module in pipeline
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
INPUT_LENGTH = 125       # 1 second at 125 Hz
FORECAST_HORIZON = 25   # 0.2 seconds at 125 Hz
NUM_SIGNALS = 3          # ABP, PLETH, II
SIGNAL_NAMES = ['ABP', 'PLETH', 'II']
TRAIN_RATIO = 0.70       # 70% train, 15% validation, 15% test
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64
RANDOM_SEED = 42


class MultiSignalForecastDataset(Dataset):
    """
    PyTorch Dataset for waveform forecasting using all signals as input.

    Each sample consists of:
        - Input (condition): INPUT_LENGTH time steps of ALL 3 signals (shape: input_length x 3)
        - Target: Next FORECAST_HORIZON values of a SINGLE target signal
    """

    def __init__(self, signals, target_idx, input_length=INPUT_LENGTH,
                 forecast_horizon=FORECAST_HORIZON):
        """
        Args:
            signals: 2D numpy array of shape (total_samples, num_signals), normalized.
            target_idx: Index of the target signal to forecast (0=ABP, 1=PLETH, 2=II).
            input_length: Number of input time steps (default 125).
            forecast_horizon: Number of steps to forecast (default 25).
        """
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.target_idx = target_idx
        self.window_size = input_length + forecast_horizon

        self.n_samples = len(signals) - self.window_size + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Signal length ({len(signals)}) too short for "
                f"window size ({self.window_size})"
            )

        self.signals = torch.FloatTensor(signals)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            x: Tensor of shape (input_length, num_signals) - all 3 signals as condition
            y: Tensor of shape (forecast_horizon,) - target signal values
        """
        x = self.signals[idx:idx + self.input_length, :]  # (125, 3)
        y = self.signals[idx + self.input_length:idx + self.window_size, self.target_idx]  # (25,)
        return x, y


def load_raw_data():
    """Load all .npy files from the data/ directory."""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            "Run download_data.py first to download waveform data."
        )

    metadata_path = os.path.join(DATA_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Run download_data.py first."
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    all_data = []
    for filename in metadata['files']:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue
        data = np.load(filepath)
        all_data.append(data)
        print(f"[LOADED] {filepath} — shape {data.shape}")

    if not all_data:
        raise RuntimeError("No data files were loaded successfully.")

    return all_data, metadata


def split_contiguous(data, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Split a contiguous time series into train/val/test blocks by time.

    The split is done chronologically:
        - First train_ratio of samples → train
        - Next val_ratio of samples → validation
        - Remaining samples → test

    Args:
        data: numpy array of shape (num_samples, num_signals).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        Tuple of (train_data, val_data, test_data) numpy arrays.
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def normalize_zscore(train_data, val_data, test_data):
    """
    Apply z-score normalization using training data statistics only.

    This prevents information leakage from val/test into normalization.

    Args:
        train_data: numpy array (train samples, num_signals).
        val_data: numpy array (val samples, num_signals).
        test_data: numpy array (test samples, num_signals).

    Returns:
        Tuple of (norm_train, norm_val, norm_test, means, stds).
    """
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    stds[stds == 0] = 1.0

    norm_train = (train_data - means) / stds
    norm_val = (val_data - means) / stds
    norm_test = (test_data - means) / stds

    return norm_train, norm_val, norm_test, means, stds


def create_dataloaders(target_idx, batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO,
                       val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """
    Create train, validation, and test DataLoaders for a target signal.

    The pipeline:
        1. Load raw data (per patient)
        2. Split each patient's data into contiguous train/val/test blocks
        3. Compute normalization stats from training blocks only
        4. Normalize all blocks
        5. Create sliding window datasets within each block
        6. Concatenate across patients

    This ensures zero temporal overlap between train/val/test sets.

    Args:
        target_idx: Index of target signal (0=ABP, 1=PLETH, 2=II).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, normalization_params).
    """
    all_data, metadata = load_raw_data()

    # Step 1: Split each patient's data into contiguous blocks
    train_blocks = []
    val_blocks = []
    test_blocks = []

    for i, patient_data in enumerate(all_data):
        train_part, val_part, test_part = split_contiguous(
            patient_data, train_ratio, val_ratio
        )
        train_blocks.append(train_part)
        val_blocks.append(val_part)
        test_blocks.append(test_part)
        print(f"[INFO] Patient {i+1}: train={len(train_part)}, "
              f"val={len(val_part)}, test={len(test_part)}")

    # Step 2: Compute normalization from training data only
    all_train = np.concatenate(train_blocks, axis=0)
    means = np.mean(all_train, axis=0)
    stds = np.std(all_train, axis=0)
    stds[stds == 0] = 1.0

    print(f"[INFO] Normalization (from training data only):")
    for i, name in enumerate(SIGNAL_NAMES):
        print(f"       {name} — mean: {means[i]:.2f}, std: {stds[i]:.2f}")

    # Step 3: Normalize and create datasets per patient block
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for i in range(len(all_data)):
        # Normalize each block using training statistics
        norm_train = (train_blocks[i] - means) / stds
        norm_val = (val_blocks[i] - means) / stds
        norm_test = (test_blocks[i] - means) / stds

        # Create sliding window datasets within each contiguous block
        train_ds = MultiSignalForecastDataset(norm_train, target_idx)
        val_ds = MultiSignalForecastDataset(norm_val, target_idx)
        test_ds = MultiSignalForecastDataset(norm_test, target_idx)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)

    # Step 4: Concatenate datasets across patients
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    print(f"[INFO] Combined data: {sum(len(d) for d in all_data)} samples")
    print(f"[INFO] Train/Val/Test windows: {train_size}/{val_size}/{test_size} "
          f"(no temporal overlap)")

    # Step 5: Create DataLoaders
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True, generator=generator,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )

    print(f"[INFO] Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    normalization_params = {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'signal_names': SIGNAL_NAMES,
    }

    return train_loader, val_loader, test_loader, normalization_params


def create_dataset(target_idx, input_length=INPUT_LENGTH, forecast_horizon=FORECAST_HORIZON):
    """
    Load data, normalize, and create sliding window dataset for a target signal.
    Uses the full combined data (for standalone statistics display).

    Args:
        target_idx: Index of target signal (0=ABP, 1=PLETH, 2=II).

    Returns:
        Tuple of (dataset, normalization_params).
    """
    all_data, metadata = load_raw_data()

    combined_data = np.concatenate(all_data, axis=0)
    print(f"[INFO] Combined data shape: {combined_data.shape}")

    # For standalone usage, still compute stats from first 70%
    train_end = int(len(combined_data) * TRAIN_RATIO)
    train_portion = combined_data[:train_end]
    means = np.mean(train_portion, axis=0)
    stds = np.std(train_portion, axis=0)
    stds[stds == 0] = 1.0

    normalized_data = (combined_data - means) / stds
    print(f"[INFO] Normalization applied (z-score from training portion)")
    for i, name in enumerate(SIGNAL_NAMES):
        print(f"       {name} — mean: {means[i]:.2f}, std: {stds[i]:.2f}")

    dataset = MultiSignalForecastDataset(
        normalized_data, target_idx, input_length, forecast_horizon
    )
    print(f"[INFO] Dataset created for target '{SIGNAL_NAMES[target_idx]}': "
          f"{len(dataset)} samples")

    normalization_params = {
        'means': means.tolist(),
        'stds': stds.tolist(),
        'signal_names': SIGNAL_NAMES,
    }

    return dataset, normalization_params


def main():
    """Print dataset statistics when run standalone."""
    print("=" * 60)
    print("MIMIC-III Waveform Preprocessing (Diffusion)")
    print("=" * 60)
    print("[INFO] Split strategy: contiguous time blocks (no data leakage)")
    print(f"[INFO] Ratios: {TRAIN_RATIO*100:.0f}% train / "
          f"{VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test")
    print()

    try:
        for target_idx, signal_name in enumerate(SIGNAL_NAMES):
            print(f"\n{'—' * 60}")
            print(f"Target signal: {signal_name} (index {target_idx})")
            print(f"{'—' * 60}")

            train_loader, val_loader, test_loader, norm_params = create_dataloaders(
                target_idx
            )

            # Get a sample
            x, y = next(iter(train_loader))
            print(f"  Sample input shape:  {x.shape[1:]}  (time_steps x signals)")
            print(f"  Sample target shape: {y.shape[1:]}  (forecast_horizon,)")
            print()

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        exit(1)


if __name__ == '__main__':
    main()
