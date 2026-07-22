"""
Data loading for Phase 5 iTransformer.

Loads pre-saved .pt files containing:
- static_feats_numeric: (N, 1)
- historical_ts_numeric: (N, 72, 12)  -- 7 correlations + 4 vitals + 1 time
- future_ts_numeric: (N, 24, 1)
- target: (N, 24, 4)                  -- 4 vital signs to predict
- target_mask: (N, 24, 4)             -- validity mask
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "/gpfs/home/dk5565/forecasting/phase5/phase5_data/processed"

# Feature names
CORRELATION_NAMES = [
    'PLETH_ACDC×PLETH_amp', 'ABP_area×ABP_tau', 'ABP_area×ShockIdx',
    'PLETH_amp×ShockIdx', 'PLETH_ACDC×ShockIdx', 'ShockIdx×ABP_tau',
    'PLETH_ACDC×ABP_tau'
]
VITAL_NAMES = ['ABPMean', 'PULSE', 'SpO2', 'RESP']
FEATURE_NAMES = CORRELATION_NAMES + VITAL_NAMES + ['time']

# Vital sign indices in the 12-dim input (and 11-dim for normalization)
VITAL_INDICES = [7, 8, 9, 10]


class PreloadedDataset(Dataset):
    """Dataset for pre-saved .pt files."""

    def __init__(self, data_dict, norm_params=None):
        """
        Args:
            data_dict: dict with keys from .pt file (already normalized by prepare_data.py)
            norm_params: dict with 'vital_mean' and 'vital_std' for denormalization (not used here)
        """
        self.historical = data_dict['historical_ts_numeric'].float()  # (N, 72, 12) — already normalized
        self.target = data_dict['target'].float()                     # (N, 24, 4) — already normalized
        self.target_mask = data_dict['target_mask'].float()           # (N, 24, 4)
        self.future_time = data_dict['future_ts_numeric'].float()     # (N, 24, 1)

        self.n_samples = self.historical.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'historical': self.historical[idx],       # (72, 12)
            'target': self.target[idx],               # (24, 4)
            'target_mask': self.target_mask[idx],     # (24, 4)
            'future_time': self.future_time[idx],     # (24, 1)
        }


def compute_norm_params(train_data):
    """
    Compute normalization parameters from training data.

    Returns:
        dict with 'mean' (12,) and 'std' (12,) tensors for the 12 input features.
        Also includes 'vital_mean' (4,) and 'vital_std' (4,) for denormalization of targets.
    """
    historical = train_data['historical_ts_numeric'].float()  # (N, 72, 12)

    # Compute mean/std across samples and time steps for each feature
    # Handle NaNs by using nanmean/nanstd
    hist_np = historical.numpy()

    mean = np.nanmean(hist_np, axis=(0, 1))  # (12,)
    std = np.nanstd(hist_np, axis=(0, 1))    # (12,)

    # Ensure no zero std
    std = np.where(std < 1e-8, 1.0, std)

    # Vital sign normalization params (indices 7-10 in the 12-dim input)
    vital_mean = mean[VITAL_INDICES]  # (4,)
    vital_std = std[VITAL_INDICES]    # (4,)

    norm_params = {
        'mean': torch.tensor(mean, dtype=torch.float32),
        'std': torch.tensor(std, dtype=torch.float32),
        'vital_mean': torch.tensor(vital_mean, dtype=torch.float32),
        'vital_std': torch.tensor(vital_std, dtype=torch.float32),
    }

    return norm_params


def create_dataloaders(batch_size=64, num_workers=4, data_dir=None):
    """
    Load pre-saved .pt files and create DataLoaders.

    Args:
        batch_size: batch size for DataLoaders
        num_workers: number of data loading workers
        data_dir: path to directory containing .pt files

    Returns:
        train_loader, val_loader, test_loader, norm_params
    """
    if data_dir is None:
        data_dir = DATA_DIR

    print(f"Loading data from: {data_dir}")

    # Load .pt files
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"), map_location='cpu')
    val_data = torch.load(os.path.join(data_dir, "val_data.pt"), map_location='cpu')
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"), map_location='cpu')

    print(f"  Train: {train_data['historical_ts_numeric'].shape[0]:,} samples")
    print(f"  Val:   {val_data['historical_ts_numeric'].shape[0]:,} samples")
    print(f"  Test:  {test_data['historical_ts_numeric'].shape[0]:,} samples")

    # Load normalization params from prepare_data.py output
    norm_json_path = os.path.join(data_dir, "norm_params.json")
    with open(norm_json_path, 'r') as f:
        norm_data = json.load(f)
    means = np.array(norm_data['means'])  # (11,)
    stds = np.array(norm_data['stds'])    # (11,)
    vital_mean = means[VITAL_INDICES]     # (4,)
    vital_std = stds[VITAL_INDICES]       # (4,)
    norm_params = {
        'vital_mean': torch.tensor(vital_mean, dtype=torch.float32),
        'vital_std': torch.tensor(vital_std, dtype=torch.float32),
    }
    print(f"  Norm params loaded from: {norm_json_path}")
    print(f"    Vital means: {norm_params['vital_mean'].numpy()}")
    print(f"    Vital stds:  {norm_params['vital_std'].numpy()}")

    # Create datasets
    train_dataset = PreloadedDataset(train_data)
    val_dataset = PreloadedDataset(val_data)
    test_dataset = PreloadedDataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader, norm_params


if __name__ == "__main__":
    train_loader, val_loader, test_loader, norm_params = create_dataloaders()

    # Check a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  historical:  {batch['historical'].shape}")    # (64, 72, 12)
    print(f"  target:      {batch['target'].shape}")        # (64, 24, 4)
    print(f"  target_mask: {batch['target_mask'].shape}")   # (64, 24, 4)
    print(f"  future_time: {batch['future_time'].shape}")   # (64, 24, 1)
