#!/usr/bin/env python3
"""
download_data.py - Download MIMIC-III waveform data for time-series forecasting.

Downloads 10 minutes of waveform data (ABP, PLETH, II) from mimic3wdb-matched/1.0
for 2 patients. Validates no NaN values exist; if found, tries different offsets
and fallback patients.

Usage:
    python download_data.py
"""

import os
import json
import numpy as np

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb package is required. Install with: pip install wfdb==4.1.2")


# Configuration
SAMPLING_RATE = 125  # Hz
DURATION_SECONDS = 600  # 10 minutes
NUM_SAMPLES = SAMPLING_RATE * DURATION_SECONDS  # 75000 samples
SIGNALS = ['ABP', 'PLETH', 'II']
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Offsets to try (in seconds) — segments often have NaN at the start
OFFSETS_TO_TRY = [90, 300, 600, 900, 1200]

# Candidate patients with known ABP/PLETH/II availability and long segments
CANDIDATE_PATIENTS = [
    {'patient_id': 'p000160', 'segment': '3531764_0003', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p000160/'},
    {'patient_id': 'p000188', 'segment': '3285727_0007', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p000188/'},
    {'patient_id': 'p000333', 'segment': '3092245_0007', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p000333/'},
    {'patient_id': 'p000543', 'segment': '3047369_0003', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p000543/'},
    {'patient_id': 'p000618', 'segment': '3481389_0008', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p000618/'},
    {'patient_id': 'p000735', 'segment': '3189254_0006', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p000735/'},
    {'patient_id': 'p000801', 'segment': '3054941_0001', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p000801/'},
    {'patient_id': 'p000946', 'segment': '3462211_0001', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p000946/'},
    {'patient_id': 'p001038', 'segment': '3755731_0015', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p001038/'},
    {'patient_id': 'p001049', 'segment': '3988865_0012', 'pn_dir': 'mimic3wdb-matched/1.0/p00/p001049/'},
]

NUM_PATIENTS_NEEDED = 2


def download_patient_data(patient_info, offset_seconds):
    """
    Download waveform data for a single patient at a given offset.

    Args:
        patient_info: dict with 'patient_id', 'segment', 'pn_dir' keys.
        offset_seconds: Number of seconds from segment start to begin reading.

    Returns:
        numpy array of shape (NUM_SAMPLES, 3) with ABP, PLETH, II signals,
        or None if download fails or data contains NaN.
    """
    patient_id = patient_info['patient_id']
    segment = patient_info['segment']
    pn_dir = patient_info['pn_dir']

    sampfrom = SAMPLING_RATE * offset_seconds
    sampto = sampfrom + NUM_SAMPLES

    print(f"  Trying offset={offset_seconds}s (samples {sampfrom}-{sampto})...", end=" ")

    try:
        record = wfdb.rdrecord(
            segment,
            pn_dir=pn_dir,
            sampfrom=sampfrom,
            sampto=sampto,
            channel_names=SIGNALS,
        )

        data = record.p_signal  # numpy array

        # Verify shape
        if data.shape != (NUM_SAMPLES, len(SIGNALS)):
            print(f"bad shape {data.shape}")
            return None

        # Check for NaN values
        if np.isnan(data).any():
            nan_counts = np.isnan(data).sum(axis=0)
            nan_info = ", ".join(f"{SIGNALS[i]}:{nan_counts[i]}" for i in range(len(SIGNALS)) if nan_counts[i] > 0)
            print(f"NaN found ({nan_info})")
            return None

        print("CLEAN!")
        return data

    except Exception as e:
        print(f"error: {e}")
        return None


def download_all_patients():
    """
    Download data for NUM_PATIENTS_NEEDED patients.
    Tries multiple offsets for each patient before moving to next candidate.

    Returns:
        List of tuples: (patient_id, numpy_array, offset_used) for successful downloads.
    """
    successful_downloads = []

    for patient_info in CANDIDATE_PATIENTS:
        if len(successful_downloads) >= NUM_PATIENTS_NEEDED:
            break

        patient_id = patient_info['patient_id']
        print(f"\n[INFO] Patient {patient_id} — segment {patient_info['segment']}")

        # Try different offsets
        for offset in OFFSETS_TO_TRY:
            data = download_patient_data(patient_info, offset)
            if data is not None:
                successful_downloads.append((patient_id, data, offset))
                print(f"  -> SUCCESS: {patient_id} at offset {offset}s")
                break
        else:
            print(f"  -> FAILED: All offsets exhausted for {patient_id}")

    return successful_downloads


def save_data(downloads):
    """
    Save downloaded data to .npy files and metadata JSON.

    Args:
        downloads: List of (patient_id, numpy_array, offset) tuples.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    patient_ids = []
    offsets_used = {}

    for patient_id, data, offset in downloads:
        filepath = os.path.join(DATA_DIR, f"{patient_id}.npy")
        np.save(filepath, data)
        patient_ids.append(patient_id)
        offsets_used[patient_id] = offset
        print(f"[SAVED] {filepath} — shape {data.shape}")

    # Save metadata
    metadata = {
        'patient_ids': patient_ids,
        'signal_names': SIGNALS,
        'sampling_rate_hz': SAMPLING_RATE,
        'duration_seconds': DURATION_SECONDS,
        'num_samples': NUM_SAMPLES,
        'data_source': 'mimic3wdb-matched/1.0',
        'offsets_used': offsets_used,
        'files': [f"{pid}.npy" for pid in patient_ids],
    }

    metadata_path = os.path.join(DATA_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {metadata_path}")

    return metadata


def main():
    """Main entry point for data download."""
    print("=" * 60)
    print("MIMIC-III Waveform Data Download")
    print("=" * 60)
    print(f"Target: {NUM_PATIENTS_NEEDED} patients, {DURATION_SECONDS}s each")
    print(f"Signals: {SIGNALS}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Offsets to try: {OFFSETS_TO_TRY}s")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 60)

    downloads = download_all_patients()

    if len(downloads) < NUM_PATIENTS_NEEDED:
        print(f"\n[WARNING] Only got {len(downloads)}/{NUM_PATIENTS_NEEDED} patients")
        if not downloads:
            print("[FATAL] No data was successfully downloaded.")
            return False

    print(f"\n[INFO] Successfully downloaded data for {len(downloads)} patient(s)")
    metadata = save_data(downloads)

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Patients: {metadata['patient_ids']}")
    print(f"  Offsets:  {metadata['offsets_used']}")
    print(f"  Signals:  {metadata['signal_names']}")
    print(f"  Rate:     {metadata['sampling_rate_hz']} Hz")
    print(f"  Duration: {metadata['duration_seconds']}s per patient")
    print(f"  Samples:  {metadata['num_samples']} per signal per patient")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)
