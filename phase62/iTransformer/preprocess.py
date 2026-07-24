"""
Data loading for Phase 6.2 iTransformer — Cluster Label Forecasting.

Loads pre-saved .pt files containing:
- historical_ts_numeric: (N, 48, 8)   -- 7 correlations + 1 time
- future_ts_numeric: (N, 12, 1)
- target: (N, 12)                     -- cluster labels (int64, 0-6)
- static_feats_numeric: (N, 1)
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "/gpfs/home/dk5565/forecasting/phase62/phase62_data/processed"
NUM_CLASSES = 7


class PreloadedDataset(Dataset):
    def __init__(self, data_dict):
        self.historical = data_dict['historical_ts_numeric'].float()  # (N, 48, 8)
        self.target = data_dict['target'].long()                      # (N, 12)
        self.future_time = data_dict['future_ts_numeric'].float()     # (N, 12, 1)
        self.n_samples = self.historical.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'historical': self.historical[idx],       # (48, 8)
            'target': self.target[idx],               # (12,)
            'future_time': self.future_time[idx],     # (12, 1)
        }


def create_dataloaders(batch_size=64, num_workers=4, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR

    print(f"Loading data from: {data_dir}")

    train_data = torch.load(os.path.join(data_dir, "train_data.pt"), map_location='cpu')
    val_data = torch.load(os.path.join(data_dir, "val_data.pt"), map_location='cpu')
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"), map_location='cpu')

    print(f"  Train: {train_data['historical_ts_numeric'].shape[0]:,} samples")
    print(f"  Val:   {val_data['historical_ts_numeric'].shape[0]:,} samples")
    print(f"  Test:  {test_data['historical_ts_numeric'].shape[0]:,} samples")

    norm_json_path = os.path.join(data_dir, "norm_params.json")
    with open(norm_json_path, 'r') as f:
        norm_params = json.load(f)

    print(f"  Num classes: {norm_params['num_classes']}")
    print(f"  Class weights: {norm_params['class_weights']}")

    train_dataset = PreloadedDataset(train_data)
    val_dataset = PreloadedDataset(val_data)
    test_dataset = PreloadedDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, norm_params


if __name__ == "__main__":
    train_loader, val_loader, test_loader, norm_params = create_dataloaders()
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  historical:  {batch['historical'].shape}")
    print(f"  target:      {batch['target'].shape} (dtype={batch['target'].dtype})")
    print(f"  future_time: {batch['future_time'].shape}")
