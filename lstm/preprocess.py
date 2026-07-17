#!/usr/bin/env python3
"""
preprocess.py - Preprocessing module for MIMIC-III waveform forecasting.

Loads raw .npy waveform data, applies z-score normalization, and creates
sliding window datasets. Each dataset uses all 3 signals (ABP, PLETH, II)
as input and targets a single signal for forecasting.

Usage:
    python preprocess.py          # Print dataset statistics
    import preprocess             # Use as module in pipeline
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


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
        - Input: INPUT_LENGTH time steps of ALL 3 signals (shape: input_length x 3)
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

        # Store all signals as tensor: (total_samples, num_signals)
        self.signals = torch.FloatTensor(signals)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            x: Tensor of shape (input_length, num_signals) - all 3 signals as input
            y: Tensor of shape (forecast_horizon,) - target signal values
        """
        # Input: all 3 signals for the input window
        x = self.signals[idx:idx + self.input_length, :]  # (125, 3)
        # Target: single signal for the forecast window
        y = self.signals[idx + self.input_length:idx + self.window_size, self.target_idx]  # (25,)
        return x, y


def load_raw_data():
    """
    Load all .npy files from the data/ directory.

    Returns:
        List of numpy arrays, each of shape (num_samples, num_signals).
    """
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


def normalize_zscore(data):
    """
    Apply z-score normalization per signal (column).

    Args:
        data: numpy array of shape (num_samples, num_signals).

    Returns:
        Tuple of (normalized_data, means, stds).
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1.0
    normalized = (data - means) / stds
    return normalized, means, stds


def create_dataset(target_idx, input_length=INPUT_LENGTH, forecast_horizon=FORECAST_HORIZON):
    """
    Load data, normalize, and create sliding window dataset for a target signal.

    Args:
        target_idx: Index of target signal (0=ABP, 1=PLETH, 2=II).
        input_length: Number of input time steps.
        forecast_horizon: Number of steps to forecast.

    Returns:
        Tuple of (dataset, normalization_params).
    """
    all_data, metadata = load_raw_data()

    combined_data = np.concatenate(all_data, axis=0)
    print(f"[INFO] Combined data shape: {combined_data.shape}")

    normalized_data, means, stds = normalize_zscore(combined_data)
    print(f"[INFO] Normalization applied (z-score per signal)")
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


def create_dataloaders(target_idx, batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO,
                       val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """
    Create train, validation, and test DataLoaders for a target signal.

    Args:
        target_idx: Index of target signal (0=ABP, 1=PLETH, 2=II).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, normalization_params).
    """
    dataset, norm_params = create_dataset(target_idx)

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"[INFO] Train/Val/Test split: {train_size}/{val_size}/{test_size} "
          f"({train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%)")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
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

    return train_loader, val_loader, test_loader, norm_params


def main():
    """Print dataset statistics when run standalone."""
    print("=" * 60)
    print("MIMIC-III Waveform Preprocessing (Multi-Signal)")
    print("=" * 60)

    try:
        # Show stats for each target signal
        for target_idx, signal_name in enumerate(SIGNAL_NAMES):
            print(f"\n{'—' * 60}")
            print(f"Target signal: {signal_name} (index {target_idx})")
            print(f"{'—' * 60}")

            dataset, norm_params = create_dataset(target_idx)

            x, y = dataset[0]
            print(f"  Total samples:       {len(dataset)}")
            print(f"  Input length:        {INPUT_LENGTH} ({INPUT_LENGTH/125:.2f}s)")
            print(f"  Input channels:      {NUM_SIGNALS} ({', '.join(SIGNAL_NAMES)})")
            print(f"  Forecast horizon:    {FORECAST_HORIZON} ({FORECAST_HORIZON/125:.2f}s)")
            print(f"  Sample input shape:  {x.shape}  (time_steps x signals)")
            print(f"  Sample target shape: {y.shape}  (forecast_horizon,)")

        train_size = int(len(dataset) * TRAIN_RATIO)
        val_size = int(len(dataset) * VAL_RATIO)
        test_size = len(dataset) - train_size - val_size
        print(f"\n{'=' * 60}")
        print(f"Split (same for all signals):")
        print(f"  Train: {train_size} | Val: {val_size} | Test: {test_size}")
        print(f"  Batch size: {BATCH_SIZE}")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Run download_data.py first to obtain waveform data.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    main()
