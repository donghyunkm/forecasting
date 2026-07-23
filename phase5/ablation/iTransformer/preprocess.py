"""
Data loading for Phase 5 iTransformer ablation (vitals-only).

Loads pre-saved .pt files containing:
- static_feats_numeric: (N, 1)
- historical_ts_numeric: (N, 72, 5)  -- 4 vitals + 1 time position
- future_ts_numeric: (N, 24, 1)
- target: (N, 24, 4)                  -- 4 vital signs to predict
- target_mask: (N, 24, 4)             -- validity mask
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "/gpfs/home/dk5565/forecasting/phase5/ablation/data/processed"

# Feature names (vitals-only ablation)
VITAL_NAMES = ['ABPMean', 'PULSE', 'SpO2', 'RESP']
FEATURE_NAMES = VITAL_NAMES + ['time']

# Vital sign indices in the 4-dim target space
VITAL_INDICES = [0, 1, 2, 3]


class PreloadedDataset(Dataset):
    """Dataset for pre-saved .pt files (vitals-only ablation)."""

    def __init__(self, data_dict, norm_params=None):
        """
        Args:
            data_dict: dict with keys from .pt file (already normalized by prepare_data.py)
            norm_params: dict with 'vital_mean' and 'vital_std' for denormalization (not used here)
        """
        self.historical = data_dict['historical_ts_numeric'].float()  # (N, 72, 5) — already normalized
        self.target = data_dict['target'].float()                     # (N, 24, 4) — already normalized
        self.target_mask = data_dict['target_mask'].float()           # (N, 24, 4)
        self.future_time = data_dict['future_ts_numeric'].float()     # (N, 24, 1)

        self.n_samples = self.historical.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'historical': self.historical[idx],       # (72, 5)
            'target': self.target[idx],               # (24, 4)
            'target_mask': self.target_mask[idx],     # (24, 4)
            'future_time': self.future_time[idx],     # (24, 1)
        }


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

    # In ablation, means/stds are directly (4,) for the 4 vitals
    means = np.array(norm_data['means'])  # (4,)
    stds = np.array(norm_data['stds'])    # (4,)

    norm_params = {
        'vital_mean': torch.tensor(means, dtype=torch.float32),  # (4,)
        'vital_std': torch.tensor(stds, dtype=torch.float32),    # (4,)
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
    print(f"  historical:  {batch['historical'].shape}")    # (64, 72, 5)
    print(f"  target:      {batch['target'].shape}")        # (64, 24, 4)
    print(f"  target_mask: {batch['target_mask'].shape}")   # (64, 24, 4)
    print(f"  future_time: {batch['future_time'].shape}")   # (64, 24, 1)

    print(f"\nNorm params:")
    print(f"  vital_mean: {norm_params['vital_mean']} (shape: {norm_params['vital_mean'].shape})")
    print(f"  vital_std:  {norm_params['vital_std']} (shape: {norm_params['vital_std'].shape})")
