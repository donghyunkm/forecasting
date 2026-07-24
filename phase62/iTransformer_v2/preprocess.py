"""
Data loading for Phase 6.2 v2 iTransformer — with label history input.

historical_ts_numeric: (N, 48, 9) -- 7 corr + 1 time + 1 label history
target: (N, 12) -- cluster labels (int64, 0-6)
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "/gpfs/home/dk5565/forecasting/phase62/phase62_data/processed_v2"
NUM_CLASSES = 7


class PreloadedDataset(Dataset):
    def __init__(self, data_dict):
        self.historical = data_dict['historical_ts_numeric'].float()  # (N, 48, 9)
        self.target = data_dict['target'].long()                      # (N, 12)
        self.n_samples = self.historical.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'historical': self.historical[idx],
            'target': self.target[idx],
        }


def create_dataloaders(batch_size=64, num_workers=4, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR

    print(f"Loading data from: {data_dir}")

    train_data = torch.load(os.path.join(data_dir, "train_data.pt"), map_location='cpu')
    val_data = torch.load(os.path.join(data_dir, "val_data.pt"), map_location='cpu')
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"), map_location='cpu')

    print(f"  Train: {train_data['historical_ts_numeric'].shape[0]:,}")
    print(f"  Val:   {val_data['historical_ts_numeric'].shape[0]:,}")
    print(f"  Test:  {test_data['historical_ts_numeric'].shape[0]:,}")

    with open(os.path.join(data_dir, "norm_params.json"), 'r') as f:
        norm_params = json.load(f)

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
