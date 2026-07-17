#!/usr/bin/env python3
"""
preprocess.py - Preprocessing module for heart rate prediction from waveforms.

Loads raw .npy waveform data, derives heart rate from the PLETH signal using
peak detection, applies z-score normalization, and creates sliding window
datasets. Each dataset uses all 3 signals (ABP, PLETH, II) as input and
targets the heart rate (BPM) computed from the upcoming window.

HEART RATE DERIVATION:
- Uses the PLETH (photoplethysmogram) signal for peak detection
- Detects systolic peaks using scipy.signal.find_peaks
- Computes instantaneous HR from inter-peak intervals (IPI)
- Each window's target HR = 60 / mean(IPI) for peaks in the target window

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
from scipy.signal import find_peaks


# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
INPUT_LENGTH = 7500      # 1 minute at 125 Hz
TARGET_LENGTH = 7500     # 1 minute at 125 Hz (window to compute target HR from)
SAMPLING_RATE = 125      # Hz
NUM_SIGNALS = 4          # II, PLETH, RESP, ABP
SIGNAL_NAMES = ['II', 'PLETH', 'RESP', 'ABP']
PLETH_IDX = 1            # Index of PLETH signal for peak detection
TRAIN_RATIO = 0.70       # 70% train, 15% validation, 15% test
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64
RANDOM_SEED = 42
STRIDE = 3750            # 30 seconds stride between windows
MIN_CHUNK_SAMPLES = 45000  # Minimum chunk size (must fit at least 1 window)

# Heart rate bounds (physiological limits)
HR_MIN = 30.0   # BPM - minimum plausible HR
HR_MAX = 200.0  # BPM - maximum plausible HR


def compute_hr_from_pleth(pleth_segment, sampling_rate=SAMPLING_RATE):
    """
    Compute heart rate (BPM) from a PLETH signal segment using peak detection.

    Quality filters applied:
        1. Minimum 5 peaks required (ensures enough beats for reliable average)
        2. At least 80% of inter-peak intervals must be physiologically valid
        3. Coefficient of variation of valid IPI must be < 0.5 (rejects irregular/noisy)
        4. HR must be within [30, 200] BPM

    Args:
        pleth_segment: 1D numpy array of PLETH values.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Heart rate in BPM, or None if detection fails or quality is too low.
    """
    if len(pleth_segment) < sampling_rate:
        return None

    # Find peaks with physiological constraints
    # At 125 Hz, peaks should be at least 0.3s apart (200 BPM max)
    min_distance = int(0.3 * sampling_rate)  # ~37 samples
    # Use prominence to find real systolic peaks
    prominence = 0.3 * (np.max(pleth_segment) - np.min(pleth_segment))
    if prominence < 1e-6:
        return None

    peaks, properties = find_peaks(
        pleth_segment,
        distance=min_distance,
        prominence=max(prominence, 0.01),
    )

    # Quality filter 1: require at least 5 peaks for reliable HR estimate
    if len(peaks) < 5:
        return None

    # Compute inter-peak intervals
    ipi = np.diff(peaks) / sampling_rate  # in seconds

    # Filter out physiologically implausible intervals
    valid_ipi = ipi[(ipi > 0.3) & (ipi < 2.0)]  # 30–200 BPM range

    # Quality filter 2: at least 80% of intervals must be valid
    if len(valid_ipi) < max(4, 0.8 * len(ipi)):
        return None

    # Quality filter 3: reject high variability (noisy/irregular signal)
    # CV > 0.5 means intervals are too inconsistent for a reliable HR
    ipi_cv = np.std(valid_ipi) / np.mean(valid_ipi)
    if ipi_cv > 0.5:
        return None

    # Heart rate from mean IPI
    mean_ipi = np.mean(valid_ipi)
    hr = 60.0 / mean_ipi

    if HR_MIN <= hr <= HR_MAX:
        return hr
    return None


def compute_hr_series(data, window_size=TARGET_LENGTH, stride=1, sampling_rate=SAMPLING_RATE):
    """
    Compute a heart rate value for sliding windows across the signal.

    For each position, computes HR from the PLETH signal in [pos : pos+window_size].

    Args:
        data: numpy array of shape (total_samples, num_signals).
        window_size: Number of samples in each window for HR computation.
        stride: Stride between windows.
        sampling_rate: Sampling rate in Hz.

    Returns:
        hr_values: numpy array of HR values (BPM) for each valid window.
        valid_mask: boolean array indicating which windows have valid HR.
    """
    pleth = data[:, PLETH_IDX]
    n_windows = (len(pleth) - window_size) // stride + 1

    hr_values = np.full(n_windows, np.nan)
    valid_mask = np.zeros(n_windows, dtype=bool)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        segment = pleth[start:end]

        hr = compute_hr_from_pleth(segment, sampling_rate)
        if hr is not None:
            hr_values[i] = hr
            valid_mask[i] = True

    return hr_values, valid_mask


class HeartRateDataset(Dataset):
    """
    PyTorch Dataset for heart rate prediction from waveform signals.

    Each sample consists of:
        - Input: INPUT_LENGTH time steps of ALL 3 signals (shape: input_length x 3)
        - Target: Heart rate (BPM) computed from the NEXT TARGET_LENGTH samples
    """

    def __init__(self, signals, hr_values, valid_indices, input_length=INPUT_LENGTH):
        """
        Args:
            signals: 2D numpy array of shape (total_samples, num_signals), normalized signals.
            hr_values: 1D numpy array of HR values (one per position).
            valid_indices: Indices where HR is valid and input window is available.
            input_length: Number of input time steps.
        """
        self.input_length = input_length
        self.signals = torch.FloatTensor(signals)
        self.hr_values = torch.FloatTensor(hr_values)
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns:
            x: Tensor of shape (input_length, num_signals) - all 3 signals as input
            y: Tensor of shape (1,) - heart rate in BPM (normalized)
        """
        pos = self.valid_indices[idx]
        x = self.signals[pos:pos + self.input_length, :]  # (INPUT_LENGTH, 3)
        y = self.hr_values[idx:idx+1]  # (1,) — scalar HR
        return x, y


def load_raw_data():
    """
    Load all .npy files from the data/ directory, split into individual chunks.

    Each patient's .npy file may contain multiple concatenated chunks from
    different segments/time periods. Uses chunk_boundaries from metadata
    to split them back into independent contiguous segments.

    Returns:
        List of numpy arrays (one per chunk), each shape (chunk_len, num_signals).
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

    all_chunks = []
    total_samples = 0

    for patient_id in metadata['patient_ids']:
        filepath = os.path.join(DATA_DIR, f"{patient_id}.npy")
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        data = np.load(filepath)
        patient_info = metadata.get('patient_info', {}).get(patient_id, {})
        chunk_boundaries = patient_info.get('chunk_boundaries', None)

        if chunk_boundaries:
            # Split into individual chunks
            for boundary in chunk_boundaries:
                chunk = data[boundary['start']:boundary['end']]
                if len(chunk) >= MIN_CHUNK_SAMPLES:
                    all_chunks.append(chunk)
                    total_samples += len(chunk)
        else:
            # Legacy format: treat entire file as one chunk
            if len(data) >= MIN_CHUNK_SAMPLES:
                all_chunks.append(data)
                total_samples += len(data)

    if not all_chunks:
        raise RuntimeError("No data chunks were loaded successfully.")

    print(f"[LOADED] {len(all_chunks)} chunks from {len(metadata['patient_ids'])} patients, "
          f"{total_samples:,} total samples ({total_samples/SAMPLING_RATE/3600:.1f} hours)")

    return all_chunks, metadata


def split_contiguous(data, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Split a contiguous time series into train/val/test blocks by time.

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


def build_hr_dataset(signals_norm, signals_raw, input_length=INPUT_LENGTH,
                     target_length=TARGET_LENGTH, sampling_rate=SAMPLING_RATE,
                     stride=STRIDE):
    """
    Build a HeartRateDataset from normalized signals and raw signals.

    HR is computed from the RAW PLETH signal (before normalization) in the
    target window following each input window.

    Args:
        signals_norm: Normalized signal array (total_samples, num_signals).
        signals_raw: Raw (unnormalized) signal array for HR computation.
        input_length: Input window length.
        target_length: Target window length for HR computation.
        sampling_rate: Sampling rate in Hz.
        stride: Step size between consecutive windows.

    Returns:
        HeartRateDataset instance (or None if no valid samples).
    """
    total_len = len(signals_norm)
    window_size = input_length + target_length

    if total_len < window_size:
        return None

    # Generate positions with stride
    positions = range(0, total_len - window_size + 1, stride)
    hr_values = []
    valid_indices = []

    pleth_raw = signals_raw[:, PLETH_IDX]

    for i in positions:
        # Target window: the segment AFTER the input window
        target_start = i + input_length
        target_end = target_start + target_length
        pleth_segment = pleth_raw[target_start:target_end]

        hr = compute_hr_from_pleth(pleth_segment, sampling_rate)
        if hr is not None:
            hr_values.append(hr)
            valid_indices.append(i)

    if len(valid_indices) == 0:
        return None

    hr_values = np.array(hr_values, dtype=np.float32)
    valid_indices = np.array(valid_indices, dtype=np.int64)

    return HeartRateDataset(signals_norm, hr_values, valid_indices, input_length)


def create_dataloaders(batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO,
                       val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                       input_length=INPUT_LENGTH, target_length=TARGET_LENGTH,
                       stride=STRIDE):
    """
    Create train, validation, and test DataLoaders for heart rate prediction.

    The pipeline:
        1. Load raw data (chunks from all patients/segments)
        2. Split EACH chunk independently into contiguous train/val/test blocks
           (chunks are independent time segments — no window can cross chunk boundaries)
        3. Compute normalization stats from training blocks only
        4. Normalize all blocks
        5. Compute HR from raw PLETH in target windows
        6. Create datasets within each block
        7. Concatenate across all chunks

    Returns:
        Tuple of (train_loader, val_loader, test_loader, normalization_params).
    """
    all_chunks, metadata = load_raw_data()

    # Step 1: Split each chunk into contiguous train/val/test blocks
    train_blocks_raw = []
    val_blocks_raw = []
    test_blocks_raw = []
    window_size = input_length + target_length

    for i, chunk in enumerate(all_chunks):
        train_part, val_part, test_part = split_contiguous(chunk, train_ratio, val_ratio)
        # Only keep blocks large enough to fit at least one window
        if len(train_part) >= window_size:
            train_blocks_raw.append(train_part)
        if len(val_part) >= window_size:
            val_blocks_raw.append(val_part)
        if len(test_part) >= window_size:
            test_blocks_raw.append(test_part)

    print(f"[INFO] Usable blocks: train={len(train_blocks_raw)}, "
          f"val={len(val_blocks_raw)}, test={len(test_blocks_raw)}")

    if not train_blocks_raw:
        raise RuntimeError("No valid training blocks found. Check data quality.")

    # Step 2: Compute normalization from training data only (for signals)
    all_train = np.concatenate(train_blocks_raw, axis=0)
    signal_means = np.mean(all_train, axis=0)
    signal_stds = np.std(all_train, axis=0)
    signal_stds[signal_stds == 0] = 1.0

    print(f"[INFO] Signal normalization (from training data only):")
    for i, name in enumerate(SIGNAL_NAMES):
        print(f"       {name} — mean: {signal_means[i]:.2f}, std: {signal_stds[i]:.2f}")

    # Step 3: Build datasets per block
    train_datasets = []
    val_datasets = []
    test_datasets = []
    all_train_hr = []

    for block_raw in train_blocks_raw:
        norm_block = (block_raw - signal_means) / signal_stds
        ds = build_hr_dataset(norm_block, block_raw, input_length=input_length,
                              target_length=target_length, stride=stride)
        if ds is not None:
            train_datasets.append(ds)
            all_train_hr.append(ds.hr_values.numpy())

    for block_raw in val_blocks_raw:
        norm_block = (block_raw - signal_means) / signal_stds
        ds = build_hr_dataset(norm_block, block_raw, input_length=input_length,
                              target_length=target_length, stride=stride)
        if ds is not None:
            val_datasets.append(ds)

    for block_raw in test_blocks_raw:
        norm_block = (block_raw - signal_means) / signal_stds
        ds = build_hr_dataset(norm_block, block_raw, input_length=input_length,
                              target_length=target_length, stride=stride)
        if ds is not None:
            test_datasets.append(ds)

    if not train_datasets:
        raise RuntimeError("No valid training samples found. Check data quality.")

    # Step 4: Compute HR normalization from training set
    all_train_hr_arr = np.concatenate(all_train_hr)
    hr_mean = float(np.mean(all_train_hr_arr))
    hr_std = float(np.std(all_train_hr_arr))
    if hr_std == 0:
        hr_std = 1.0
    print(f"[INFO] HR normalization: mean={hr_mean:.2f} BPM, std={hr_std:.2f} BPM")
    print(f"[INFO] HR range in training: [{all_train_hr_arr.min():.1f}, {all_train_hr_arr.max():.1f}] BPM")

    # Normalize HR targets in all datasets
    for ds in train_datasets + val_datasets + test_datasets:
        ds.hr_values = (ds.hr_values - hr_mean) / hr_std

    # Step 5: Concatenate datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    print(f"[INFO] Train/Val/Test samples: {train_size}/{val_size}/{test_size} "
          f"(no temporal overlap)")

    # Step 6: Create DataLoaders
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
        'signal_means': signal_means.tolist(),
        'signal_stds': signal_stds.tolist(),
        'signal_names': SIGNAL_NAMES,
        'hr_mean': hr_mean,
        'hr_std': hr_std,
    }

    return train_loader, val_loader, test_loader, normalization_params


def main():
    """Print dataset statistics when run standalone."""
    print("=" * 60)
    print("MIMIC-III Heart Rate Prediction — Preprocessing")
    print("=" * 60)
    print("[INFO] Split strategy: contiguous time blocks (no data leakage)")
    print(f"[INFO] Ratios: {TRAIN_RATIO*100:.0f}% train / "
          f"{VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test")
    print(f"[INFO] Input: {INPUT_LENGTH} samples ({INPUT_LENGTH/SAMPLING_RATE:.1f}s) of 3 signals")
    print(f"[INFO] Target: HR from next {TARGET_LENGTH} samples ({TARGET_LENGTH/SAMPLING_RATE:.1f}s)")
    print()

    try:
        train_loader, val_loader, test_loader, norm_params = create_dataloaders()

        # Get a sample
        x, y = next(iter(train_loader))
        print(f"\n  Sample input shape:  {x.shape}  (batch, time_steps, signals)")
        print(f"  Sample target shape: {y.shape}  (batch, 1) — normalized HR")
        print(f"\n  HR normalization: mean={norm_params['hr_mean']:.2f}, "
              f"std={norm_params['hr_std']:.2f}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Run download_data.py first to obtain waveform data.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    main()
