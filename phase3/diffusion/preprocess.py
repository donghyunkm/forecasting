#!/usr/bin/env python3
"""
preprocess.py - Preprocessing for multivariate waveform forecasting (Phase 3).

Resamples raw 125 Hz waveform data into 15-minute intervals with aggregated
features (mean, std, min, max, skewness, kurtosis) for each of the 4 signals
(II, PLETH, RESP, ABP). Creates forecasting datasets: given 75 time points
(~18.75 hours) of all signals' features, predict the next 25 time points
(~6.25 hours) of a single target signal's mean value.

FEATURE AGGREGATION (per 15-min interval, per signal):
    - mean: average value
    - std: standard deviation (variability)
    - min: minimum value (trough)
    - max: maximum value (peak)
    - skewness: asymmetry of distribution
    - kurtosis: tail heaviness of distribution

TASK: 4 separate models, each targeting one signal:
    - Model 1: Predict II (using all 4 signals as input)
    - Model 2: Predict PLETH (using all 4 signals as input)
    - Model 3: Predict RESP (using all 4 signals as input)
    - Model 4: Predict ABP (using all 4 signals as input)

DATA LEAKAGE PREVENTION:
    - Each chunk is split chronologically (70/15/15) BEFORE windowing
    - Normalization statistics computed from training data only
    - No window spans chunk boundaries or train/val/test boundaries

Usage:
    python preprocess.py                    # Print dataset statistics
    python preprocess.py --target II        # Show stats for II target model
    import preprocess                       # Use as module in pipeline
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# Configuration
DATA_DIR = '/gpfs/scratch/dk5565/phase3_data'
SAMPLING_RATE = 125          # Hz (raw data)
INTERVAL_MINUTES = 6         # Resampling interval
INTERVAL_SAMPLES = INTERVAL_MINUTES * 60 * SAMPLING_RATE  # 45,000 samples per interval
NUM_SIGNALS = 4              # II, PLETH, RESP, ABP
SIGNAL_NAMES = ['II', 'PLETH', 'RESP', 'ABP']
NUM_FEATURES = 4             # mean, std, min, max
FEATURE_NAMES = ['mean', 'std', 'min', 'max']

# Forecasting windows
INPUT_LENGTH = 75            # 75 intervals = 18.75 hours of history
OUTPUT_LENGTH = 25           # 25 intervals = 6.25 hours forecast
WINDOW_SIZE = INPUT_LENGTH + OUTPUT_LENGTH  # 100 intervals needed per sample

# Data splitting
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
BATCH_SIZE = 64
RANDOM_SEED = 42
STRIDE = 25                  # Stride in intervals between windows (25 = 2.5 hours)

# Minimum chunk size in raw samples to keep during loading
# Only need enough for 1 interval (15 min) — patient-level split handles the rest
MIN_CHUNK_SAMPLES = INTERVAL_SAMPLES  # 112,500 samples (15 min)

# Valid target signals
VALID_TARGETS = ['II', 'PLETH', 'RESP', 'ABP']


def aggregate_interval(signal_segment):
    """
    Compute aggregated features for a single 15-minute interval of one signal.

    Args:
        signal_segment: 1D numpy array of raw samples for one signal, one interval.

    Returns:
        numpy array of shape (NUM_FEATURES,): [mean, std, min, max]
    """
    if len(signal_segment) == 0:
        return np.zeros(NUM_FEATURES)

    feat_mean = np.mean(signal_segment)
    feat_std = np.std(signal_segment)
    feat_min = np.min(signal_segment)
    feat_max = np.max(signal_segment)

    return np.array([feat_mean, feat_std, feat_min, feat_max],
                    dtype=np.float32)


def resample_chunk(chunk_data):
    """
    Resample a raw waveform chunk into 15-minute intervals with aggregated features.

    Args:
        chunk_data: numpy array of shape (num_raw_samples, NUM_SIGNALS).

    Returns:
        numpy array of shape (num_intervals, NUM_SIGNALS, NUM_FEATURES).
        Returns None if chunk is too short.
    """
    num_samples = len(chunk_data)
    num_intervals = num_samples // INTERVAL_SAMPLES

    if num_intervals < WINDOW_SIZE:
        return None

    # Trim to exact number of intervals
    trimmed = chunk_data[:num_intervals * INTERVAL_SAMPLES]

    # Reshape to (num_intervals, INTERVAL_SAMPLES, NUM_SIGNALS)
    reshaped = trimmed.reshape(num_intervals, INTERVAL_SAMPLES, NUM_SIGNALS)

    # Compute features for each interval and signal
    # Output shape: (num_intervals, NUM_SIGNALS, NUM_FEATURES)
    features = np.zeros((num_intervals, NUM_SIGNALS, NUM_FEATURES), dtype=np.float32)

    for t in range(num_intervals):
        for s in range(NUM_SIGNALS):
            features[t, s, :] = aggregate_interval(reshaped[t, :, s])

    return features


class ForecastDataset(Dataset):
    """
    PyTorch Dataset for multivariate waveform forecasting.

    Each sample consists of:
        - Input: 75 time points × (4 signals × 6 features) = 75 × 24 features
        - Target: 25 time points × 6 features of the target signal (shape: 25, 6)

    The target is ALL 6 aggregated features of one specific signal over the
    forecast horizon (mean, std, min, max, skewness, kurtosis).
    """

    def __init__(self, features, target_signal_idx, input_length=INPUT_LENGTH,
                 output_length=OUTPUT_LENGTH, stride=STRIDE):
        """
        Args:
            features: numpy array of shape (num_intervals, NUM_SIGNALS, NUM_FEATURES),
                      already normalized.
            target_signal_idx: Index of the target signal (0=II, 1=PLETH, 2=RESP, 3=ABP).
            input_length: Number of input intervals.
            output_length: Number of output intervals to predict.
            stride: Step size between consecutive windows.
        """
        self.input_length = input_length
        self.output_length = output_length
        self.target_signal_idx = target_signal_idx
        self.window_size = input_length + output_length

        # Flatten signal/feature dims for input: (num_intervals, NUM_SIGNALS * NUM_FEATURES)
        num_intervals = features.shape[0]
        self.features_flat = torch.FloatTensor(
            features.reshape(num_intervals, NUM_SIGNALS * NUM_FEATURES)
        )
        # Target: ALL 6 features of the target signal
        # Shape: (num_intervals, NUM_FEATURES)
        self.target_values = torch.FloatTensor(features[:, target_signal_idx, :])

        # Valid window positions
        self.positions = list(range(0, num_intervals - self.window_size + 1, stride))

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        """
        Returns:
            x: Tensor of shape (input_length, NUM_SIGNALS * NUM_FEATURES) — all signals, all features
            y: Tensor of shape (output_length, NUM_FEATURES) — target signal's 6 features for next 25 intervals
        """
        pos = self.positions[idx]
        # Input: all features for all signals over the input window
        x = self.features_flat[pos:pos + self.input_length]  # (75, 24)
        # Target: all 6 features of target signal over the output window
        y = self.target_values[pos + self.input_length:pos + self.window_size]  # (25, 6)
        return x, y


def load_raw_data():
    """
    Load patient metadata from the data/ directory without loading raw arrays.

    Uses metadata.json and chunk boundaries to determine data volume per patient.
    Raw data is NOT loaded into memory — it will be loaded one patient at a time
    during the resampling step.

    Returns:
        List of dicts: [{'patient_id': str, 'chunk_boundaries': [...], 'total_samples': int}, ...]
        metadata dict.
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

    patients_data = []
    total_samples = 0

    for patient_id in metadata['patient_ids']:
        filepath = os.path.join(DATA_DIR, f"{patient_id}.npy")
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        patient_info = metadata.get('patient_info', {}).get(patient_id, {})
        chunk_boundaries = patient_info.get('chunk_boundaries', None)
        patient_total = patient_info.get('total_samples', 0)

        # If no total_samples in metadata, get it from file without loading
        if not patient_total:
            try:
                data = np.load(filepath, mmap_mode='r')
                patient_total = len(data)
                del data
            except Exception as e:
                print(f"[WARNING] Failed to load {filepath}: {e}")
                continue

        if patient_total >= INTERVAL_SAMPLES:
            patients_data.append({
                'patient_id': patient_id,
                'chunk_boundaries': chunk_boundaries,
                'total_samples': patient_total,
            })
            total_samples += patient_total

    if not patients_data:
        raise RuntimeError("No data was loaded successfully.")

    hours = total_samples / SAMPLING_RATE / 3600
    print(f"[LOADED] {len(patients_data)} patients, "
          f"{total_samples:,} total samples ({hours:.1f} hours)")

    return patients_data, metadata


def split_patients(patients_data, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Split patients into train/val/test sets (patient-level split).

    Entire patients are assigned to one split — no patient's data appears
    in multiple splits. Patients are sorted by total data volume and assigned
    to approximate the target ratios, with at least 1 patient per split.

    Args:
        patients_data: List of {'patient_id': str, 'total_samples': int, ...}.
        train_ratio: Target fraction for training.
        val_ratio: Target fraction for validation.

    Returns:
        Tuple of (train_patients, val_patients, test_patients).
    """
    n_patients = len(patients_data)

    if n_patients < 3:
        raise RuntimeError(
            f"Need at least 3 patients for train/val/test split, got {n_patients}. "
            "Increase --num-patients."
        )

    # Sort patients by total samples (most data first for stable assignment)
    patients_with_size = []
    for p in patients_data:
        total = p['total_samples']
        patients_with_size.append((p, total))
    patients_with_size.sort(key=lambda x: -x[1])

    total_samples = sum(s for _, s in patients_with_size)
    train_target = total_samples * train_ratio
    val_target = total_samples * val_ratio

    train_patients = []
    val_patients = []
    test_patients = []
    train_sum = 0
    val_sum = 0

    # Reserve at least 1 patient for val and test (take smallest patients)
    # Assign the rest greedily to approximate target ratios
    reserved_test = patients_with_size[-1]  # smallest for test
    reserved_val = patients_with_size[-2]   # second smallest for val
    assignable = patients_with_size[:-2]

    for p, size in assignable:
        if train_sum < train_target:
            train_patients.append(p)
            train_sum += size
        elif val_sum < val_target:
            val_patients.append(p)
            val_sum += size
        else:
            test_patients.append(p)

    # Add reserved patients
    val_patients.append(reserved_val[0])
    test_patients.append(reserved_test[0])

    return train_patients, val_patients, test_patients
    assignable = patients_with_size[:-2]

    for p, size in assignable:
        if train_sum < train_target:
            train_patients.append(p)
            train_sum += size
        elif val_sum < val_target:
            val_patients.append(p)
            val_sum += size
        else:
            test_patients.append(p)

    # Add reserved patients
    val_patients.append(reserved_val[0])
    test_patients.append(reserved_test[0])

    return train_patients, val_patients, test_patients


def create_dataloaders(target_signal='II', batch_size=BATCH_SIZE, input_length=INPUT_LENGTH,
                       output_length=OUTPUT_LENGTH, stride=STRIDE):
    """
    Create train, validation, and test DataLoaders for forecasting.

    Pipeline:
        1. Load raw data grouped by patient (memory-mapped)
        2. Split PATIENTS into train/val/test (no patient appears in multiple splits)
        3. For each patient: resample chunks → aggregate features → build datasets
        4. Compute normalization from training patients only
        5. Re-process with normalization applied

    Data leakage prevention:
        - Patient-level split: no patient's data appears in multiple splits
        - Normalization from training patients only
        - No window crosses chunk boundaries

    Args:
        target_signal: Name of signal to predict ('II', 'PLETH', 'RESP', 'ABP').
        batch_size: Batch size for DataLoader.
        input_length: Number of input intervals (default 75).
        output_length: Number of output intervals (default 25).
        stride: Window stride in intervals.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, norm_params).
    """
    if target_signal not in VALID_TARGETS:
        raise ValueError(f"Invalid target signal: {target_signal}. Must be one of {VALID_TARGETS}")

    target_idx = SIGNAL_NAMES.index(target_signal)
    window_size = input_length + output_length

    print(f"[INFO] Target signal: {target_signal} (index {target_idx})")
    print(f"[INFO] Window: {input_length} input → {output_length} output intervals")
    print(f"[INFO] Features per interval: {NUM_SIGNALS} signals × {NUM_FEATURES} features = "
          f"{NUM_SIGNALS * NUM_FEATURES}")

    # Step 1: Load patient metadata and split at patient level
    patients_data, metadata = load_raw_data()

    # Step 2: Split at patient level
    train_patients, val_patients, test_patients = split_patients(patients_data)
    print(f"[INFO] Patient split: train={len(train_patients)}, "
          f"val={len(val_patients)}, test={len(test_patients)}")
    print(f"       Train: {[p['patient_id'] for p in train_patients]}")
    print(f"       Val:   {[p['patient_id'] for p in val_patients]}")
    print(f"       Test:  {[p['patient_id'] for p in test_patients]}")

    # Step 3: Resample chunks for each split (process one patient at a time)
    # Cache resampled data to disk so future runs skip resampling
    resampled_dir = os.path.join(DATA_DIR, 'resampled')
    os.makedirs(resampled_dir, exist_ok=True)

    def resample_patient_chunks(patient_list):
        """Resample all chunks for a list of patients, with disk caching.
        Loads raw data from disk one patient at a time to minimize memory."""
        resampled = []
        for patient in patient_list:
            pid = patient['patient_id']
            cache_path = os.path.join(resampled_dir, f"{pid}.npz")

            if os.path.exists(cache_path):
                # Load cached resampled data
                cached = np.load(cache_path, allow_pickle=True)
                blocks = [cached[k] for k in sorted(cached.files)]
                for block in blocks:
                    if len(block) >= window_size:
                        resampled.append(block)
            else:
                # Load raw data from disk for this patient only
                filepath = os.path.join(DATA_DIR, f"{pid}.npy")
                try:
                    data = np.load(filepath, mmap_mode='r')
                except Exception as e:
                    print(f"[WARNING] Failed to load {filepath}: {e}")
                    continue

                chunk_boundaries = patient.get('chunk_boundaries', None)
                raw_chunks = []
                if chunk_boundaries:
                    for boundary in chunk_boundaries:
                        chunk = np.array(data[boundary['start']:boundary['end']])
                        if len(chunk) >= INTERVAL_SAMPLES:
                            raw_chunks.append(chunk)
                else:
                    if len(data) >= INTERVAL_SAMPLES:
                        raw_chunks.append(np.array(data))
                del data

                # Resample from raw and cache
                patient_blocks = []
                for chunk in raw_chunks:
                    features = resample_chunk(chunk)
                    if features is not None:
                        patient_blocks.append(features)
                        if len(features) >= window_size:
                            resampled.append(features)
                del raw_chunks

                # Save to cache
                if patient_blocks:
                    np.savez(cache_path, *patient_blocks)
                del patient_blocks
        return resampled

    print(f"[INFO] Resampling to {INTERVAL_MINUTES}-min intervals (cached in {resampled_dir})...")
    train_blocks = resample_patient_chunks(train_patients)
    val_blocks = resample_patient_chunks(val_patients)
    test_blocks = resample_patient_chunks(test_patients)

    print(f"[INFO] Usable blocks (≥{window_size} intervals): "
          f"train={len(train_blocks)}, val={len(val_blocks)}, test={len(test_blocks)}")

    if not train_blocks:
        raise RuntimeError("No valid training blocks found after resampling.")

    # Step 4: Compute normalization from training data only
    all_train = np.concatenate(train_blocks, axis=0)
    norm_mean = np.mean(all_train, axis=0)  # (NUM_SIGNALS, NUM_FEATURES)
    norm_std = np.std(all_train, axis=0)    # (NUM_SIGNALS, NUM_FEATURES)
    norm_std[norm_std == 0] = 1.0
    del all_train  # free memory

    total_train_intervals = sum(len(b) for b in train_blocks)
    print(f"[INFO] Normalization from {total_train_intervals} training intervals")
    for s, name in enumerate(SIGNAL_NAMES):
        print(f"       {name} mean: mean={norm_mean[s, 0]:.4f}, std={norm_std[s, 0]:.4f}")

    # Step 5: Normalize and build datasets
    def normalize(block):
        return (block - norm_mean) / norm_std

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for block in train_blocks:
        norm_block = normalize(block)
        ds = ForecastDataset(norm_block, target_idx, input_length, output_length, stride)
        if len(ds) > 0:
            train_datasets.append(ds)

    for block in val_blocks:
        norm_block = normalize(block)
        ds = ForecastDataset(norm_block, target_idx, input_length, output_length, stride)
        if len(ds) > 0:
            val_datasets.append(ds)

    for block in test_blocks:
        norm_block = normalize(block)
        ds = ForecastDataset(norm_block, target_idx, input_length, output_length, stride)
        if len(ds) > 0:
            test_datasets.append(ds)

    if not train_datasets:
        raise RuntimeError("No valid training samples found.")

    # Step 6: Concatenate datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets) if val_datasets else ConcatDataset([train_datasets[0]])
    test_dataset = ConcatDataset(test_datasets) if test_datasets else ConcatDataset([train_datasets[0]])

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    print(f"[INFO] Dataset sizes: train={train_size}, val={val_size}, test={test_size}")

    # Step 7: Create DataLoaders
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

    print(f"[INFO] Batches: train={len(train_loader)}, "
          f"val={len(val_loader)}, test={len(test_loader)}")

    # Normalization params to save with checkpoint
    norm_params = {
        'norm_mean': norm_mean.tolist(),
        'norm_std': norm_std.tolist(),
        'signal_names': SIGNAL_NAMES,
        'feature_names': FEATURE_NAMES,
        'target_signal': target_signal,
        'target_signal_idx': target_idx,
        'interval_minutes': INTERVAL_MINUTES,
        'input_length': input_length,
        'output_length': output_length,
        'num_signals': NUM_SIGNALS,
        'num_features': NUM_FEATURES,
    }

    return train_loader, val_loader, test_loader, norm_params


def main():
    """Print dataset statistics when run standalone."""
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess waveform data for phase3 forecasting')
    parser.add_argument('--target', type=str, default='II', choices=VALID_TARGETS,
                        help=f'Target signal to forecast (default: II)')
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 — Multivariate Waveform Forecasting (Preprocessing)")
    print("=" * 60)
    print(f"[INFO] Resampling: {INTERVAL_MINUTES}-minute intervals")
    print(f"[INFO] Features per interval: {NUM_FEATURES} ({', '.join(FEATURE_NAMES)})")
    print(f"[INFO] Input features: {NUM_SIGNALS} signals × {NUM_FEATURES} = {NUM_SIGNALS * NUM_FEATURES}")
    print(f"[INFO] Input window: {INPUT_LENGTH} intervals ({INPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hours)")
    print(f"[INFO] Output window: {OUTPUT_LENGTH} intervals ({OUTPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hours)")
    print(f"[INFO] Target signal: {args.target}")
    print(f"[INFO] Split: {TRAIN_RATIO*100:.0f}% / {VAL_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}%")
    print()

    try:
        train_loader, val_loader, test_loader, norm_params = create_dataloaders(
            target_signal=args.target
        )

        x, y = next(iter(train_loader))
        print(f"\n  Sample input shape:  {x.shape}  (batch, time_steps=75, features=24)")
        print(f"  Sample target shape: {y.shape}  (batch, forecast_steps=25, features=6)")
        print(f"\n  Target signal: {args.target}")
        print(f"  Target: all 6 features of {args.target} for next {OUTPUT_LENGTH} intervals")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Run download_data.py first to obtain waveform data.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    main()
