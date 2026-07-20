#!/usr/bin/env python3
"""
download_data.py - Load MIMIC-III waveform data from local GPFS storage.

Extracts as much clean waveform data as possible from each patient:
- Uses ALL valid segments per patient (not just one)
- Takes maximum duration from each segment
- Splits around NaN regions into clean sub-segments (chunks)
- Keeps chunks >= MIN_CHUNK_SAMPLES (must fit at least one window after resampling)

Data source: /gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched/
Format: WFDB (.hea + .dat files), read via wfdb.rdrecord()

Usage:
    python download_data.py
    python download_data.py --num-patients 5
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
WAVEFORM_DIR = '/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched'
SAMPLING_RATE = 125  # Hz
SIGNALS = ['II', 'PLETH', 'RESP', 'ABP']
DATA_DIR = '/gpfs/scratch/dk5565/phase3_data'
NUM_PATIENTS_NEEDED = 5

# Minimum usable chunk size (6 min = 1 resampling interval)
MIN_CHUNK_SAMPLES = 45000  # 6 min at 125 Hz

# Maximum samples to read in a single rdrecord call
# No limit — read full segments to maximize data
MAX_READ_SAMPLES = None


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

    # Check records listed in RECORDS file
    for record_name in records:
        if 'layout' in record_name:
            try:
                hdr = wfdb.rdheader(os.path.join(patient_dir, record_name))
                if hdr.sig_name:
                    return all(sig in hdr.sig_name for sig in required_signals)
            except Exception:
                pass
            break

    # Layout not in RECORDS — check layout files on disk directly
    try:
        layout_files = [f for f in os.listdir(patient_dir)
                        if 'layout' in f and f.endswith('.hea')]
        for layout_file in layout_files:
            name = layout_file.replace('.hea', '')
            try:
                hdr = wfdb.rdheader(os.path.join(patient_dir, name))
                if hdr.sig_name:
                    return all(sig in hdr.sig_name for sig in required_signals)
            except Exception:
                continue
    except Exception:
        pass

    return False


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

    valid_segments.sort(key=lambda x: x[1], reverse=True)
    return valid_segments


def extract_clean_chunks(patient_dir, segment_name, sig_len):
    """
    Read a segment and split into clean (NaN-free) chunks.

    Returns:
        List of numpy arrays, each shape (N, num_signals) where N >= MIN_CHUNK_SAMPLES.
    """
    record_path = os.path.join(patient_dir, segment_name)
    samples_to_read = sig_len if MAX_READ_SAMPLES is None else min(sig_len, MAX_READ_SAMPLES)

    try:
        record = wfdb.rdrecord(
            record_path,
            sampfrom=0,
            sampto=samples_to_read,
            channel_names=SIGNALS,
        )
        data = record.p_signal  # (samples_to_read, num_signals)
    except Exception:
        return []

    if data is None or len(data) == 0:
        return []

    # Find NaN-free regions
    nan_mask = np.isnan(data).any(axis=1)

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
    Discover valid patient directories.

    First checks for a pre-computed valid_patients.json (from check_patients.py)
    in the parent directory. If found, uses that for instant discovery.
    Otherwise falls back to scanning RECORDS file.

    Returns list of (group, patient_id) tuples, sorted by data volume (most first).
    """
    # Fast path: use cached valid patient list
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    cached_path = os.path.join(parent_dir, 'valid_patients.json')

    if os.path.exists(cached_path):
        with open(cached_path, 'r') as f:
            cached = json.load(f)
        patients = [(p['group'], p['patient_id']) for p in cached['patients']]
        print(f"[INFO] Using cached patient list: {cached_path} ({len(patients)} patients)")
        return patients

    # Slow path: scan RECORDS file
    print("[INFO] No cached patient list found. Scanning RECORDS file...")
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

    rng = np.random.default_rng(42)
    rng.shuffle(patients)

    return patients


def load_and_save_patients(num_patients=NUM_PATIENTS_NEEDED, skip_ids=None):
    """
    Load waveform data one patient at a time, save to disk immediately,
    and free memory before processing the next patient.

    Args:
        num_patients: Number of patients to load.
        skip_ids: Set of patient_ids to skip (already saved).

    Returns:
        List of dicts with metadata (no chunk arrays):
        {'patient_id': str, 'total_samples': int, 'num_segments': int,
         'num_chunks': int, 'chunk_boundaries': list}
    """
    skip_ids = skip_ids or set()
    saved_patient_info = []
    candidates = discover_patients()
    os.makedirs(DATA_DIR, exist_ok=True)

    for group, patient_id in candidates:
        if len(saved_patient_info) >= num_patients:
            break

        # Skip already-saved patients
        if patient_id in skip_ids:
            continue

        patient_dir = os.path.join(WAVEFORM_DIR, group, patient_id)
        if not os.path.exists(patient_dir):
            continue

        if not patient_has_signals(patient_dir):
            continue

        segments = find_all_valid_segments(patient_dir)
        if not segments:
            continue

        patient_chunks = []
        total_samples = 0

        for seg_name, seg_len in segments:
            chunks = extract_clean_chunks(patient_dir, seg_name, seg_len)
            for chunk in chunks:
                patient_chunks.append(chunk)
                total_samples += len(chunk)
            del chunks

        if not patient_chunks:
            continue

        # Save immediately and record chunk boundaries
        combined = np.concatenate(patient_chunks, axis=0)
        chunk_boundaries = []
        offset = 0
        for chunk in patient_chunks:
            chunk_boundaries.append({'start': offset, 'end': offset + len(chunk)})
            offset += len(chunk)

        # Free chunk list before saving
        del patient_chunks

        filepath = os.path.join(DATA_DIR, f"{patient_id}.npy")
        np.save(filepath, combined)
        del combined

        saved_patient_info.append({
            'patient_id': patient_id,
            'total_samples': total_samples,
            'num_segments': len(segments),
            'num_chunks': len(chunk_boundaries),
            'chunk_boundaries': chunk_boundaries,
        })

        duration_hrs = total_samples / SAMPLING_RATE / 3600
        print(f"[SAVED] Patient {patient_id} ({len(saved_patient_info)}/{num_patients}) — "
              f"{len(chunk_boundaries)} chunks, {total_samples:,} samples ({duration_hrs:.1f} hrs) "
              f"-> {filepath}")

    return saved_patient_info


def write_metadata(patient_info_list, prev_meta=None):
    """
    Write metadata.json from a list of patient info dicts.

    Args:
        patient_info_list: List of dicts with patient_id, total_samples,
                          num_chunks, num_segments, chunk_boundaries.
        prev_meta: Dict of previously-saved patient_info (from old metadata.json).

    Returns:
        The metadata dict that was written.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    patient_ids = [p['patient_id'] for p in patient_info_list]
    patient_info = {}
    for p in patient_info_list:
        patient_info[p['patient_id']] = {
            'total_samples': p['total_samples'],
            'num_chunks': p['num_chunks'],
            'num_segments': p['num_segments'],
            'chunk_boundaries': p['chunk_boundaries'],
        }

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
    """Load waveform data from local GPFS storage. Resumes partial downloads."""
    n_patients = num_patients if num_patients is not None else NUM_PATIENTS_NEEDED

    # Check if complete data already exists
    metadata_path = os.path.join(DATA_DIR, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        existing = len(meta.get('patient_ids', []))
        if existing >= n_patients:
            print(f"[INFO] Data already exists at {DATA_DIR} ({existing} patients)")
            print("[INFO] Skipping download. Delete data/ to re-download.")
            return True
        else:
            print(f"[INFO] Found partial download ({existing}/{n_patients} patients). Resuming...")

    # Find which patients are already saved
    os.makedirs(DATA_DIR, exist_ok=True)
    already_saved_ids = set(
        f.replace('.npy', '') for f in os.listdir(DATA_DIR) if f.endswith('.npy')
    )
    if already_saved_ids:
        print(f"[INFO] Skipping {len(already_saved_ids)} already-saved patients: "
              f"{sorted(already_saved_ids)}")

    print("=" * 60)
    print("MIMIC-III Waveform Data — Local GPFS Load (Phase 3)")
    print("=" * 60)
    print(f"Source: {WAVEFORM_DIR}")
    print(f"Target: {n_patients} patients (all valid segments, max duration)")
    print(f"Signals: {SIGNALS}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Min chunk size: {MIN_CHUNK_SAMPLES:,} samples "
          f"({MIN_CHUNK_SAMPLES/SAMPLING_RATE/60:.1f} min)")
    print(f"Output directory: {DATA_DIR}")
    print("=" * 60)

    # Load and save new patients one at a time
    remaining = n_patients - len(already_saved_ids)
    new_patient_info = load_and_save_patients(remaining, skip_ids=already_saved_ids)

    if not new_patient_info and not already_saved_ids:
        print("[FATAL] No data was successfully loaded.")
        return False

    # Build complete metadata from all saved patients
    all_patient_info = []

    # Reconstruct info for previously-saved patients
    prev_meta = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            prev = json.load(f)
        prev_meta = prev.get('patient_info', {})

    new_patient_ids = {p['patient_id'] for p in new_patient_info}
    for pid in sorted(already_saved_ids):
        if pid in new_patient_ids:
            continue  # handled below with new_patient_info
        filepath = os.path.join(DATA_DIR, f"{pid}.npy")
        if not os.path.exists(filepath):
            continue
        if pid in prev_meta:
            all_patient_info.append({
                'patient_id': pid,
                **prev_meta[pid],
            })
        else:
            # Reconstruct from file using memory-mapped read
            try:
                data = np.load(filepath, mmap_mode='r')
                n_samples = len(data)
                del data
            except Exception:
                continue
            all_patient_info.append({
                'patient_id': pid,
                'total_samples': n_samples,
                'num_chunks': 1,
                'num_segments': 1,
                'chunk_boundaries': [{'start': 0, 'end': n_samples}],
            })

    # Append newly saved patients
    all_patient_info.extend(new_patient_info)

    # Write final metadata
    metadata = write_metadata(all_patient_info)

    total_samples = sum(p['total_samples'] for p in all_patient_info)
    total_hours = total_samples / SAMPLING_RATE / 3600
    total_chunks = sum(p['num_chunks'] for p in all_patient_info)

    print("\n" + "=" * 60)
    print("Load Summary")
    print("=" * 60)
    print(f"  Patients:       {len(all_patient_info)}")
    print(f"  Total chunks:   {total_chunks}")
    print(f"  Total samples:  {total_samples:,}")
    print(f"  Total duration: {total_hours:.1f} hours")
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
