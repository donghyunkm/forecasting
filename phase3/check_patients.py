#!/usr/bin/env python3
"""
check_patients.py - Survey all patients in MIMIC-III waveform database to find
those with valid data for phase3 forecasting.

Checks:
    1. Patient has all 4 required signals (II, PLETH, RESP, ABP)
    2. Patient has at least one NaN-free chunk >= MIN_CHUNK_SAMPLES (25 hours)

Reports how many patients are valid and their data availability.

Usage:
    python check_patients.py
    python check_patients.py --min-hours 10   # Lower the threshold
    python check_patients.py --max-scan 500   # Scan first 500 patients only
"""

import os
import argparse
import numpy as np
import time

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb package is required. Install with: pip install wfdb==4.1.2")


# Configuration
WAVEFORM_DIR = '/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched'
SAMPLING_RATE = 125
SIGNALS = ['II', 'PLETH', 'RESP', 'ABP']
INTERVAL_SAMPLES = 15 * 60 * SAMPLING_RATE  # 112,500 samples per 15-min interval
WINDOW_SIZE = 100  # 75 input + 25 output intervals
DEFAULT_MIN_CHUNK_SAMPLES = INTERVAL_SAMPLES * WINDOW_SIZE  # ~25 hours
MAX_READ_SAMPLES = 5000000  # ~11 hours per segment read


def patient_has_signals(patient_dir):
    """Quick check via layout header."""
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
                    return all(sig in hdr.sig_name for sig in SIGNALS)
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
                    return all(sig in hdr.sig_name for sig in SIGNALS)
            except Exception:
                continue
    except Exception:
        pass

    return False


def get_patient_chunks(patient_dir, min_chunk_samples):
    """
    Find all valid segments and compute clean chunk sizes without reading full data.
    Returns list of chunk durations in hours, or empty list if no valid data.
    """
    records_path = os.path.join(patient_dir, 'RECORDS')
    if not os.path.exists(records_path):
        return []

    with open(records_path, 'r') as f:
        records = [line.strip() for line in f if line.strip()]

    chunk_hours = []

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
            if hdr.sig_len < min_chunk_samples:
                continue
            if not all(sig in hdr.sig_name for sig in SIGNALS):
                continue

            # Read segment to find NaN-free chunks
            samples_to_read = min(hdr.sig_len, MAX_READ_SAMPLES)
            record = wfdb.rdrecord(
                record_path, sampfrom=0, sampto=samples_to_read,
                channel_names=SIGNALS,
            )
            data = record.p_signal
            if data is None or len(data) == 0:
                continue

            # Find NaN-free regions
            nan_mask = np.isnan(data).any(axis=1)
            chunk_start = None

            for i in range(len(nan_mask)):
                if not nan_mask[i]:
                    if chunk_start is None:
                        chunk_start = i
                else:
                    if chunk_start is not None:
                        chunk_len = i - chunk_start
                        if chunk_len >= min_chunk_samples:
                            chunk_hours.append(chunk_len / SAMPLING_RATE / 3600)
                        chunk_start = None

            if chunk_start is not None:
                chunk_len = len(nan_mask) - chunk_start
                if chunk_len >= min_chunk_samples:
                    chunk_hours.append(chunk_len / SAMPLING_RATE / 3600)

        except Exception:
            continue

    return chunk_hours


def discover_patients():
    """Get all patient directories."""
    records_path = os.path.join(WAVEFORM_DIR, 'RECORDS')
    if not os.path.exists(records_path):
        print(f"[ERROR] RECORDS file not found: {records_path}")
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

    return patients


def main():
    parser = argparse.ArgumentParser(description='Check patient data availability for phase3')
    parser.add_argument('--min-hours', type=float, default=DEFAULT_MIN_CHUNK_SAMPLES / SAMPLING_RATE / 3600,
                        help=f'Minimum chunk duration in hours (default: {DEFAULT_MIN_CHUNK_SAMPLES / SAMPLING_RATE / 3600:.1f})')
    parser.add_argument('--max-scan', type=int, default=None,
                        help='Maximum number of patients to scan (default: all)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print details for each valid patient')
    parser.add_argument('--save', type=str, default=None,
                        help='Save valid patient list to JSON file (e.g. --save valid_patients.json)')
    args = parser.parse_args()

    min_chunk_samples = int(args.min_hours * 3600 * SAMPLING_RATE)
    min_intervals = min_chunk_samples // INTERVAL_SAMPLES

    print("=" * 70)
    print("MIMIC-III Waveform Database — Patient Availability Survey (Phase 3)")
    print("=" * 70)
    print(f"Source: {WAVEFORM_DIR}")
    print(f"Required signals: {SIGNALS}")
    print(f"Min chunk duration: {args.min_hours:.1f} hours ({min_chunk_samples:,} samples)")
    print(f"Min intervals after resampling: {min_intervals}")
    print(f"Window size needed: {WINDOW_SIZE} intervals (75 input + 25 output)")
    print(f"Scan limit: {'all' if args.max_scan is None else args.max_scan}")
    print("=" * 70)

    patients = discover_patients()
    total_patients = len(patients)
    print(f"\n[INFO] Total patients in database: {total_patients}")

    if args.max_scan:
        patients = patients[:args.max_scan]
        print(f"[INFO] Scanning first {len(patients)} patients...")

    start_time = time.time()

    # Stage 1: Quick signal check
    has_signals = []
    for group, patient_id in patients:
        patient_dir = os.path.join(WAVEFORM_DIR, group, patient_id)
        if os.path.exists(patient_dir) and patient_has_signals(patient_dir):
            has_signals.append((group, patient_id))

    elapsed_signals = time.time() - start_time
    print(f"\n[STAGE 1] Signal check ({elapsed_signals:.1f}s):")
    print(f"  Scanned: {len(patients)}")
    print(f"  Have all 4 signals (II, PLETH, RESP, ABP): {len(has_signals)} "
          f"({100*len(has_signals)/len(patients):.1f}%)")

    # Stage 2: Check chunk sizes (slower — reads actual data)
    print(f"\n[STAGE 2] Checking chunk sizes for {len(has_signals)} patients with all signals...")
    print(f"  (This reads actual waveform data and may take a while...)")

    valid_patients = []
    total_valid_chunks = 0
    total_valid_hours = 0

    for i, (group, patient_id) in enumerate(has_signals):
        patient_dir = os.path.join(WAVEFORM_DIR, group, patient_id)
        chunk_hours = get_patient_chunks(patient_dir, min_chunk_samples)

        if chunk_hours:
            total_hrs = sum(chunk_hours)
            valid_patients.append({
                'group': group,
                'patient_id': patient_id,
                'num_chunks': len(chunk_hours),
                'total_hours': total_hrs,
                'chunk_hours': chunk_hours,
            })
            total_valid_chunks += len(chunk_hours)
            total_valid_hours += total_hrs

            if args.verbose:
                print(f"  ✓ {patient_id}: {len(chunk_hours)} chunks, "
                      f"{total_hrs:.1f} hrs (largest: {max(chunk_hours):.1f} hrs)")

        if (i + 1) % 50 == 0:
            print(f"  ... checked {i+1}/{len(has_signals)} patients, "
                  f"{len(valid_patients)} valid so far")

    elapsed_total = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total patients scanned:           {len(patients)}")
    print(f"  Have all 4 signals:               {len(has_signals)} "
          f"({100*len(has_signals)/max(len(patients),1):.1f}%)")
    print(f"  Have valid chunks (≥{args.min_hours:.1f} hrs):   {len(valid_patients)} "
          f"({100*len(valid_patients)/max(len(patients),1):.1f}%)")
    print(f"  ---")
    print(f"  Total valid chunks:               {total_valid_chunks}")
    print(f"  Total valid data:                 {total_valid_hours:.1f} hours")
    if valid_patients:
        print(f"  Avg per valid patient:            {total_valid_hours/len(valid_patients):.1f} hours")
        print(f"  Avg chunks per valid patient:     {total_valid_chunks/len(valid_patients):.1f}")

        # Distribution
        hours_list = [p['total_hours'] for p in valid_patients]
        hours_list.sort(reverse=True)
        print(f"  ---")
        print(f"  Top 10 patients by data volume:")
        for j, p in enumerate(sorted(valid_patients, key=lambda x: -x['total_hours'])[:10]):
            print(f"    {j+1:>3}. {p['patient_id']}: {p['num_chunks']} chunks, "
                  f"{p['total_hours']:.1f} hrs")

    print(f"  ---")
    print(f"  Time elapsed: {elapsed_total:.1f}s")
    print("=" * 70)

    # Save results if requested
    if args.save and valid_patients:
        import json
        save_data = {
            'waveform_dir': WAVEFORM_DIR,
            'min_chunk_hours': args.min_hours,
            'min_chunk_samples': min_chunk_samples,
            'signals': SIGNALS,
            'num_valid_patients': len(valid_patients),
            'total_chunks': total_valid_chunks,
            'total_hours': total_valid_hours,
            'patients': [
                {
                    'patient_id': p['patient_id'],
                    'group': p['group'],
                    'patient_dir': os.path.join(WAVEFORM_DIR, p['group'], p['patient_id']),
                    'num_chunks': p['num_chunks'],
                    'total_hours': p['total_hours'],
                    'chunk_hours': p['chunk_hours'],
                }
                for p in sorted(valid_patients, key=lambda x: -x['total_hours'])
            ],
        }
        with open(args.save, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\n[SAVED] Valid patient list: {args.save}")

    # Recommendations
    if valid_patients:
        print(f"\n[RECOMMENDATION]")
        print(f"  {len(valid_patients)} patients available with ≥{args.min_hours:.1f} hr chunks.")
        if len(valid_patients) >= 20:
            print(f"  Consider using --num-patients 20 for more training data.")
        if args.min_hours > 10 and len(valid_patients) < 10:
            print(f"  Consider lowering --min-hours to get more patients.")
    else:
        print(f"\n[WARNING] No valid patients found with ≥{args.min_hours:.1f} hr chunks!")
        print(f"  Try: python check_patients.py --min-hours 10")


if __name__ == '__main__':
    main()
