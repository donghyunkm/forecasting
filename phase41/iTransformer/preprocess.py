"""
Preprocess / Data Loading for iTransformer — Phase 4.1
Fast-load from pre-saved .pt files.
"""

import os, json, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/gpfs/scratch/dk5565/phase41_data'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
NUM_SIGNALS = 4
FUTURE_MONTHS = 25
PAST_MONTHS = 75
SIGNAL_NAMES = ['mean_bp', 'pulse', 'spo2', 'respiratory_rate']
BATCH_SIZE = 64


class PreloadedDataset(Dataset):
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


def create_dataloaders(batch_size=BATCH_SIZE, num_workers=4):
    norm_json = os.path.join(PROCESSED_DIR, 'norm_params.json')
    with open(norm_json) as f:
        norm_data = json.load(f)
    norm_params = {'mean': np.array(norm_data['mean']), 'std': np.array(norm_data['std'])}

    train_data = torch.load(os.path.join(PROCESSED_DIR, 'train_data.pt'), weights_only=False)
    val_data = torch.load(os.path.join(PROCESSED_DIR, 'val_data.pt'), weights_only=False)
    test_data = torch.load(os.path.join(PROCESSED_DIR, 'test_data.pt'), weights_only=False)

    train_loader = DataLoader(PreloadedDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(PreloadedDataset(val_data), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(PreloadedDataset(test_data), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f'  Train: {len(train_data["historical_ts_numeric"])} windows')
    print(f'  Val:   {len(val_data["historical_ts_numeric"])} windows')
    print(f'  Test:  {len(test_data["historical_ts_numeric"])} windows')

    return train_loader, val_loader, test_loader, norm_params
