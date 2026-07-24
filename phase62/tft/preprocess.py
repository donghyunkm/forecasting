#!/usr/bin/env python3
"""
Phase 6.2 TFT Preprocessing: Load pre-computed .pt files into DataLoaders.

Data format:
- historical_ts_numeric: (N, 48, 8) — 7 correlations + 1 time position
- future_ts_numeric: (N, 12, 1) — time position (known into future)
- target: (N, 12) — cluster labels (int64, values 0-6)
- static_feats_numeric: (N, 1) — placeholder
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "/gpfs/home/dk5565/forecasting/phase62/phase62_data/processed/"
BATCH_SIZE = 64
NUM_WORKERS = 4


class PreloadedDataset(Dataset):
    """Dataset that loads from a pre-computed .pt file."""

    def __init__(self, pt_path: str):
        self.data = torch.load(pt_path, map_location='cpu')
        self.n_samples = self.data['target'].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'static_feats_numeric': self.data['static_feats_numeric'][idx],
            'historical_ts_numeric': self.data['historical_ts_numeric'][idx],
            'future_ts_numeric': self.data['future_ts_numeric'][idx],
            'target': self.data['target'][idx],  # (12,) int64
        }


def create_dataloaders(data_dir: str = DATA_DIR, batch_size: int = BATCH_SIZE,
                       num_workers: int = NUM_WORKERS):
    """Load .pt files and return train/val/test DataLoaders + norm_params."""
    train_path = os.path.join(data_dir, "train_data.pt")
    val_path = os.path.join(data_dir, "val_data.pt")
    test_path = os.path.join(data_dir, "test_data.pt")
    norm_path = os.path.join(data_dir, "norm_params.json")

    for path in [train_path, val_path, test_path, norm_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(norm_path, 'r') as f:
        norm_params = json.load(f)

    train_dataset = PreloadedDataset(train_path)
    val_dataset = PreloadedDataset(val_path)
    test_dataset = PreloadedDataset(test_path)

    print(f"Dataset sizes - Train: {len(train_dataset):,}, "
          f"Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader, norm_params


if __name__ == "__main__":
    train_loader, val_loader, test_loader, norm_params = create_dataloaders()
    batch = next(iter(train_loader))
    print("\nSample batch shapes:")
    for key, val in batch.items():
        print(f"  {key}: {val.shape} (dtype={val.dtype})")
    print(f"\nNum classes: {norm_params['num_classes']}")
    print(f"Class weights: {norm_params['class_weights']}")
