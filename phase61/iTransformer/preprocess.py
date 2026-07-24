"""
Data loading for Phase 6.1 iTransformer — Correlation + Physio Features Forecasting.

Loads pre-saved .pt files containing:
- static_feats_numeric: (N, 1)
- historical_ts_numeric: (N, 48, 46)   -- 7 correlations + 38 physio stats + 1 time
- future_ts_numeric: (N, 12, 1)
- target: (N, 12, 7)                   -- 7 correlations to predict (point predictions)
- target_mask: (N, 12, 7)              -- validity mask
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "/gpfs/home/dk5565/forecasting/phase61/phase61_data/processed"

# Feature names
CORRELATION_NAMES = [
    'PLETH_ACDC×PLETH_amp', 'ABP_area×ABP_tau', 'ABP_area×ShockIdx',
    'PLETH_amp×ShockIdx', 'PLETH_ACDC×ShockIdx', 'ShockIdx×ABP_tau',
    'PLETH_ACDC×ABP_tau'
]

PHYSIO_FEATURE_NAMES = [
    "HR", "RR", "SBP", "DBP", "PP",
    "MAP", "ABP_area", "PLETH_ACDC", "PLETH_amp", "ECG_Ramp",
    "HRV_RMSSD", "HR_range", "ShockIdx", "PPV", "PVI",
    "PTT", "dPdt_max", "ABP_tau", "RESP_amp",
]

FEATURE_NAMES = (
    CORRELATION_NAMES +
    [f"{name}_mean" for name in PHYSIO_FEATURE_NAMES] +
    [f"{name}_std" for name in PHYSIO_FEATURE_NAMES] +
    ['time']
)

# All output indices (correlations are both input and output)
CORR_INDICES = [0, 1, 2, 3, 4, 5, 6]
NUM_CORR = 7
NUM_INPUT_FEATURES = 46  # 7 + 38 + 1


class PreloadedDataset(Dataset):
    """Dataset for pre-saved .pt files."""

    def __init__(self, data_dict):
        """
        Args:
            data_dict: dict with keys from .pt file (already normalized by prepare_data.py)
        """
        self.historical = data_dict['historical_ts_numeric'].float()  # (N, 48, 46) — already normalized
        self.target = data_dict['target'].float()                     # (N, 12, 7) — already normalized
        self.target_mask = data_dict['target_mask'].float()           # (N, 12, 7)
        self.future_time = data_dict['future_ts_numeric'].float()     # (N, 12, 1)

        self.n_samples = self.historical.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'historical': self.historical[idx],       # (48, 46)
            'target': self.target[idx],               # (12, 7)
            'target_mask': self.target_mask[idx],     # (12, 7)
            'future_time': self.future_time[idx],     # (12, 1)
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

    corr_means = np.array(norm_data['corr_means'])  # (7,)
    corr_stds = np.array(norm_data['corr_stds'])    # (7,)
    norm_params = {
        'corr_mean': torch.tensor(corr_means, dtype=torch.float32),
        'corr_std': torch.tensor(corr_stds, dtype=torch.float32),
        'corr_means': norm_data['corr_means'],
        'corr_stds': norm_data['corr_stds'],
        'physio_means': norm_data['physio_means'],
        'physio_stds': norm_data['physio_stds'],
        'correlation_names': norm_data['correlation_names'],
        'physio_feature_names': norm_data['physio_feature_names'],
    }
    print(f"  Norm params loaded from: {norm_json_path}")
    print(f"    Corr means: {norm_params['corr_mean'].numpy()}")
    print(f"    Corr stds:  {norm_params['corr_std'].numpy()}")

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
    print(f"  historical:  {batch['historical'].shape}")    # (64, 48, 46)
    print(f"  target:      {batch['target'].shape}")        # (64, 12, 7)
    print(f"  target_mask: {batch['target_mask'].shape}")   # (64, 12, 7)
    print(f"  future_time: {batch['future_time'].shape}")   # (64, 12, 1)
