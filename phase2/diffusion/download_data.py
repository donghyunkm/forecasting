#!/usr/bin/env python3
"""
download_data.py - Load MIMIC-III waveform data from local GPFS storage.

Extracts as much clean waveform data as possible from each patient:
- Uses ALL valid segments per patient (not just one)
- Takes maximum duration from each segment (not a fixed 1-hour window)
- Trims edges to align with the 5min input + 1min target window size
- Skips NaN regions by chunking segments into clean sub-segments

Data source: /gpfs/data/eh3828lab/globus/ICU/mimic3_waveforms_matched/
Format: WFDB (.hea + .dat files), read via wfdb.rdrecord()

Usage:
    python download_data.py
    python download_data.py --num-patients 200
"""

import os
import json
import argparse
import numpy as np

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb package is required. Install with: pip install wfdb==4.1.2")


# Configuration
WAVEFORM_DIR = '/gpfs/data/eh3828lab/globus/ICU/mimic3_waveforms_matched'
SAMPLING_RATE = 125  # Hz
SIGNALS = ['II', 'PLETH', 'RESP', 'ABP']
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NUM_PATIENTS_NEEDED = 5

# Minimum usable chunk: must fit at least one full window (input + target)
# 5 min input + 1 min target = 6 min = 45,000 samples minimum
MIN_CHUNK_SAMPLES = 45000

# Maximum samples to read in a single rdrecord call (avoid memory issues)
MAX_READ_SAMPLES = 2000000  # ~4.4 hours at 125 Hz per segment


def patient_has_signals(patient_dir, required_signals=SIGNALS):
    """
    Quick check via layout header whether patient has all required signals.
    Returns True/False without scanning individual segments.
    """
    records_path = os.path.join(patient_dir, 'RECORDS')
    if not os.path.exists(records_path):
        return False

    with open(records_path, 'r') as f:
        records = [line.strip() for line in f if line.strip()]

    for record_name in records:
        if 'layout' in record_name:
            try:
                hdr = wfdb.rdheader(os.path.join(patient_dir, record_name))
                if hdr.sig_name:
                    return all(sig in hdr.sig_name for sig in required_signals)
            except Exception:
                pass
            break

    # No layout found — can't quickly determine, assume possible
    return True


def find_all_valid_segments(patient_dir, required_signals=SIGNALS, min_samples=MIN_CHUNK_SAMPLES):
    """
    Find ALL segments in a patient directory that contain all required signals
    and have enough samples.

    Returns:
        List of (segment_name, sig_len) tuples, sorted by length (longest first).
    """
    records_path = os.path.join(patient_dir, 'RECORDS')
    if not os.path.exists(records_path):
        return []

    with open(records_path, 'r') as f:
        records = [line.strip() for line in f if line.strip()]

    valid_segments = []

    for record_name in records:
        # Skip master records, layout records, and numeric records
        if '-' in record_name or 'layout' in record_name or record_name.endswith('n'):
            continue

        hea_path = os.path.join(patient_dir, f'{record_name}.hea')
        if not os.path.exists(hea_path):
            continue

        try:
            record_path = os.path.join(patient_dir, record_name)
            hdr = wfdb.rdheader(record_path)
            if hdr.sig_name is None:
                continue
            if hdr.sig_len < min_samples:
                continue
            if all(sig in hdr.sig_name for sig in required_signals):
                valid_segments.append((record_name, hdr.sig_len))
        except Exception:
            continue

    # Sort by length, longest first (maximize data per segment)
    valid_segments.sort(key=lambda x: x[1], reverse=True)
    return valid_segments


def extract_clean_chunks(patient_dir, segment_name, sig_len):
    """
    Read a segment and split into clean (NaN-free) chunks.

    Reads the full segment, finds contiguous NaN-free regions,
    and returns chunks that are at least MIN_CHUNK_SAMPLES long.
    Trims each chunk so its length is a multiple of MIN_CHUNK_SAMPLES
    (aligns with window boundaries for clean splitting later).

    Returns:
        List of numpy arrays, each shape (N, num_signals) where N >= MIN_CHUNK_SAMPLES.
    """
    record_path = os.path.join(patient_dir, segment_name)
    samples_to_read = min(sig_len, MAX_READ_SAMPLES)

    try:
        record = wfdb.rdrecord(
            record_path,
            sampfrom=0,
            sampto=samples_to_read,
            channel_names=SIGNALS,
        )
        data = record.p_signal  # (samples_to_read, num_signals)
    except Exception as e:
        return []

    if data is None or len(data) == 0:
        return []

    # Find NaN-free regions
    nan_mask = np.isnan(data).any(axis=1)  # True where any signal has NaN

    chunks = []
    chunk_start = None

    for i in range(len(nan_mask)):
        if not nan_mask[i]:
            if chunk_start is None:
                chunk_start = i
        else:
            if chunk_start is not None:
                chunk_len = i - chunk_start
                if chunk_len >= MIN_CHUNK_SAMPLES:
                    # Trim to usable length (don't need exact multiple, just min size)
                    chunk = data[chunk_start:i]
                    chunks.append(chunk)
                chunk_start = None

    # Handle last chunk
    if chunk_start is not None:
        chunk_len = len(nan_mask) - chunk_start
        if chunk_len >= MIN_CHUNK_SAMPLES:
            chunk = data[chunk_start:]
            chunks.append(chunk)

    return chunks


def discover_patients():
    """
    Discover all patient directories from the RECORDS file.
    Returns list of (group, patient_id) tuples, shuffled deterministically.
    """
    records_path = os.path.join(WAVEFORM_DIR, 'RECORDS')
    if not os.path.exists(records_path):
        return []

    patients = []
    with open(records_path, 'r') as f:
        for line in f:
            line = line.strip().rstrip('/')
            if not line:
                continue
            parts = line.split('/')
            if len(parts) == 2:
                patients.append((parts[0], parts[1]))

    # Shuffle deterministically for variety
    rng = np.random.default_rng(42)
    rng.shuffle(patients)

    return patients


def load_all_patients(num_patients=NUM_PATIENTS_NEEDED):
    """
    Load waveform data for multiple patients, extracting all valid segments
    and maximum clean data from each.

    Returns:
        List of dicts: {'patient_id': str, 'chunks': [np.array, ...], 'total_samples': int}
    """
    all_patient_data = []
    candidates = discover_patients()

    for group, patient_id in candidates:
        if len(all_patient_data) >= num_patients:
            break

        patient_dir = os.path.join(WAVEFORM_DIR, group, patient_id)
        if not os.path.exists(patient_dir):
            continue

        # Quick signal check via layout
        if not patient_has_signals(patient_dir):
            continue

        # Find all valid segments
        segments = find_all_valid_segments(patient_dir)
        if not segments:
            continue

        # Extract clean chunks from all segments
        patient_chunks = []
        total_samples = 0

        for seg_name, seg_len in segments:
            chunks = extract_clean_chunks(patient_dir, seg_name, seg_len)
            for chunk in chunks:
                patient_chunks.append(chunk)
                total_samples += len(chunk)

        if not patient_chunks:
            continue

        all_patient_data.append({
            'patient_id': patient_id,
            'chunks': patient_chunks,
            'total_samples': total_samples,
            'num_segments': len(segments),
            'num_chunks': len(patient_chunks),
        })

        duration_min = total_samples / SAMPLING_RATE / 60
        print(f"[INFO] Patient {patient_id} ({len(all_patient_data)}/{num_patients}) — "
              f"{len(patient_chunks)} chunks, {total_samples:,} samples ({duration_min:.1f} min)")

    return all_patient_data


def save_data(all_patient_data):
    """
    Save loaded data to local data/ directory.
    
    Each patient gets one .npy file containing all chunks concatenated.
    Metadata records chunk boundaries so preprocess.py can split correctly.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    patient_ids = []
    patient_info = {}

    for pdata in all_patient_data:
        patient_id = pdata['patient_id']
        chunks = pdata['chunks']

        # Concatenate all chunks for storage
        combined = np.concatenate(chunks, axis=0)
        filepath = os.path.join(DATA_DIR, f"{patient_id}.npy")
        np.save(filepath, combined)
        patient_ids.append(patient_id)

        # Record chunk boundaries for later splitting
        chunk_boundaries = []
        offset = 0
        for chunk in chunks:
            chunk_boundaries.append({'start': offset, 'end': offset + len(chunk)})
            offset += len(chunk)

        patient_info[patient_id] = {
            'total_samples': pdata['total_samples'],
            'num_chunks': pdata['num_chunks'],
            'num_segments': pdata['num_segments'],
            'chunk_boundaries': chunk_boundaries,
        }

        print(f"[SAVED] {filepath} — shape {combined.shape}")

    metadata = {
        'patient_ids': patient_ids,
        'signal_names': SIGNALS,
        'sampling_rate_hz': SAMPLING_RATE,
        'data_source': WAVEFORM_DIR,
        'min_chunk_samples': MIN_CHUNK_SAMPLES,
        'files': [f"{pid}.npy" for pid in patient_ids],
        'patient_info': patient_info,
    }

    metadata_path = os.path.join(DATA_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {metadata_path}")

    return metadata


def main(num_patients=None):
    """Load waveform data from local GPFS storage."""
    n_patients = num_patients if num_patients is not None else NUM_PATIENTS_NEEDED

    # Skip if data already exists
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        metadata_path = os.path.join(DATA_DIR, 'metadata.json')
        if os.path.exists(metadata_path):
            print(f"[INFO] Data already exists at {DATA_DIR}")
            print("[INFO] Skipping download. Delete data/ to re-download.")
            return True

    print("=" * 60)
    print("MIMIC-III Waveform Data — Local GPFS Load (Max Extraction)")
    print("=" * 60)
    print(f"Source: {WAVEFORM_DIR}")
    print(f"Target: {n_patients} patients (all valid segments, max duration)")
    print(f"Signals: {SIGNALS}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Min chunk size: {MIN_CHUNK_SAMPLES} samples ({MIN_CHUNK_SAMPLES/SAMPLING_RATE/60:.1f} min)")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 60)

    all_patient_data = load_all_patients(n_patients)

    if not all_patient_data:
        print("[FATAL] No data was successfully loaded.")
        return False

    print(f"\n[INFO] Successfully loaded data for {len(all_patient_data)} patient(s)")

    # Summary stats
    total_samples = sum(p['total_samples'] for p in all_patient_data)
    total_hours = total_samples / SAMPLING_RATE / 3600
    total_chunks = sum(p['num_chunks'] for p in all_patient_data)

    metadata = save_data(all_patient_data)

    print("\n" + "=" * 60)
    print("Load Summary")
    print("=" * 60)
    print(f"  Patients:      {len(all_patient_data)}")
    print(f"  Total chunks:  {total_chunks}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total duration: {total_hours:.1f} hours")
    print(f"  Avg per patient: {total_samples/len(all_patient_data)/SAMPLING_RATE/60:.1f} min")
    print(f"  Signals: {SIGNALS}")
    print("=" * 60)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load MIMIC-III waveform data from local GPFS')
    parser.add_argument('--num-patients', type=int, default=NUM_PATIENTS_NEEDED,
                        help=f'Number of patients to load (default: {NUM_PATIENTS_NEEDED})')
    args = parser.parse_args()
    success = main(num_patients=args.num_patients)
    if not success:
        exit(1)
