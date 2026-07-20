#!/usr/bin/env python3
"""
download_data.py - Extract vital signs from MIMIC-III CHARTEVENTS for ICU stays.

Optimized version using pandas chunked reading for ~10x speedup over csv.DictReader.

Extracts 5 vital signs used in TFT-multi paper:
  1. Heart Rate (pulse)
  2. Mean Blood Pressure (non-invasive + arterial)
  3. SpO2 (oxygen saturation)
  4. Respiratory Rate
  5. Temperature (converted to Celsius)

Data source: /gpfs/data/eh3828lab/datasets/mimic_clinical/CHARTEVENTS.csv.gz
Output: Per-ICU-stay .npy files with shape (num_hours, 5) + metadata.json

Usage:
    python download_data.py
    python download_data.py --num-patients 100
"""

import os
import json
import gzip
import argparse
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================
CLINICAL_DIR = '/gpfs/data/eh3828lab/datasets/mimic_clinical'
DATA_DIR = '/gpfs/scratch/dk5565/phase4_data'
NUM_PATIENTS_NEEDED = 100

# Vital sign ITEMIDs from D_ITEMS
# Each vital sign may have multiple ITEMIDs (CareVue + MetaVision)
VITAL_SIGN_ITEMS = {
    'heart_rate': [211, 220045, 1332, 1341, 1725],
    'mean_bp': [456, 52, 443, 6702, 220052, 220181, 225312,
                3320, 3322, 3316, 3318, 3324],
    'sbp': [51, 6, 455, 442, 6701, 220050, 220179, 225309],
    'dbp': [8368, 8364, 8441, 8440, 8555, 220051, 220180, 225310],
    'spo2': [646, 220277, 834, 3288, 5636, 8208, 8209],
    'respiratory_rate': [618, 615, 3603, 220210, 224690],
    'temperature': [676, 677, 678, 679, 223761, 223762,
                    3652, 3654, 3655, 6643],
}

# Temperature ITEMIDs that are in Fahrenheit (need conversion)
FAHRENHEIT_ITEMS = {678, 679, 223761, 3652, 3654}

# All vital sign ITEMIDs for fast lookup
ALL_VITAL_ITEMIDS = set()
for items in VITAL_SIGN_ITEMS.values():
    ALL_VITAL_ITEMIDS.update(items)

# Build reverse lookup: itemid -> vital sign name
ITEMID_TO_VITAL = {}
for vital_name, itemids in VITAL_SIGN_ITEMS.items():
    for itemid in itemids:
        ITEMID_TO_VITAL[itemid] = vital_name

# Vital sign names (output order)
SIGNAL_NAMES = ['heart_rate', 'mean_bp', 'spo2', 'respiratory_rate', 'temperature']
NUM_SIGNALS = 5

# Minimum ICU stay length (hours) to include
MIN_STAY_HOURS = 48  # Need at least 48 hours for 75+25=100 hourly steps

# Physiological bounds for filtering outliers
VITAL_BOUNDS = {
    'heart_rate': (20, 250),       # bpm
    'mean_bp': (20, 200),          # mmHg
    'sbp': (40, 300),              # mmHg
    'dbp': (10, 200),              # mmHg
    'spo2': (50, 100),             # %
    'respiratory_rate': (4, 60),   # breaths/min
    'temperature': (30, 42),       # Celsius
}

# Pandas chunk size for reading CHARTEVENTS
CHUNK_SIZE = 2_000_000  # 2M rows per chunk — good balance of speed vs memory


def load_icustays():
    """Load ICU stays with length of stay >= MIN_STAY_HOURS."""
    icustays_path = os.path.join(CLINICAL_DIR, 'ICUSTAYS.csv.gz')

    df = pd.read_csv(icustays_path, compression='gzip')
    df = df[df['LOS'] * 24 >= MIN_STAY_HOURS].copy()

    stays = {}
    for _, row in df.iterrows():
        icustay_id = int(row['ICUSTAY_ID'])
        stays[icustay_id] = {
            'subject_id': int(row['SUBJECT_ID']),
            'hadm_id': int(row['HADM_ID']),
            'icustay_id': icustay_id,
            'intime': row['INTIME'],
            'outtime': row['OUTTIME'],
            'los_days': float(row['LOS']),
        }

    print(f"[INFO] Found {len(stays)} ICU stays with LOS >= {MIN_STAY_HOURS} hours")
    return stays


def fahrenheit_to_celsius(f_val):
    """Convert Fahrenheit to Celsius."""
    return (f_val - 32) * 5.0 / 9.0


def extract_vital_signs_chunked(icustay_ids):
    """
    Read CHARTEVENTS.csv.gz in large pandas chunks, filtering on ITEMID and ICUSTAY_ID.

    This is ~10x faster than csv.DictReader because:
    - pandas C parser handles gzip decompression + CSV parsing in compiled code
    - We only load needed columns (reduces I/O and memory)
    - Vectorized filtering instead of per-row Python loops

    Args:
        icustay_ids: Set of ICU stay IDs to extract data for.

    Returns:
        Dict mapping icustay_id -> list of (charttime_str, vital_name, value)
    """
    chartevents_path = os.path.join(CLINICAL_DIR, 'CHARTEVENTS.csv.gz')
    vitals_by_stay = defaultdict(list)

    # Only read the columns we need
    usecols = ['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']

    print(f"[INFO] Reading CHARTEVENTS.csv.gz in chunks of {CHUNK_SIZE:,} rows...")
    print(f"[INFO] Filtering for {len(icustay_ids)} ICU stays, {len(ALL_VITAL_ITEMIDS)} ITEMIDs")

    start_time = time.time()
    total_rows = 0
    match_count = 0
    chunk_num = 0

    reader = pd.read_csv(
        chartevents_path,
        compression='gzip',
        chunksize=CHUNK_SIZE,
        usecols=usecols,
        dtype={
            'ICUSTAY_ID': 'float64',  # Has NaN values
            'ITEMID': 'int64',
            'VALUENUM': 'float64',
        },
        low_memory=False,
    )

    for chunk in reader:
        chunk_num += 1
        total_rows += len(chunk)

        # Step 1: Filter by ITEMID (fastest filter - integer comparison)
        chunk = chunk[chunk['ITEMID'].isin(ALL_VITAL_ITEMIDS)]

        if chunk.empty:
            if chunk_num % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  ... chunk {chunk_num}: {total_rows:,} rows processed, "
                      f"{match_count:,} matches, {elapsed:.0f}s elapsed")
            continue

        # Step 2: Drop rows with NaN ICUSTAY_ID or VALUENUM
        chunk = chunk.dropna(subset=['ICUSTAY_ID', 'VALUENUM'])

        # Step 3: Filter by target ICUSTAY_IDs
        chunk['ICUSTAY_ID'] = chunk['ICUSTAY_ID'].astype(int)
        chunk = chunk[chunk['ICUSTAY_ID'].isin(icustay_ids)]

        if chunk.empty:
            if chunk_num % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  ... chunk {chunk_num}: {total_rows:,} rows processed, "
                      f"{match_count:,} matches, {elapsed:.0f}s elapsed")
            continue

        # Step 4: Process matching rows
        for _, row in chunk.iterrows():
            itemid = int(row['ITEMID'])
            value = float(row['VALUENUM'])
            icustay_id = int(row['ICUSTAY_ID'])

            if np.isnan(value):
                continue

            vital_name = ITEMID_TO_VITAL[itemid]

            # Convert Fahrenheit to Celsius
            if itemid in FAHRENHEIT_ITEMS:
                value = fahrenheit_to_celsius(value)

            # Filter physiological bounds
            low, high = VITAL_BOUNDS[vital_name]
            if value < low or value > high:
                continue

            vitals_by_stay[icustay_id].append((row['CHARTTIME'], vital_name, value))
            match_count += 1

        if chunk_num % 20 == 0:
            elapsed = time.time() - start_time
            rate = total_rows / elapsed if elapsed > 0 else 0
            print(f"  ... chunk {chunk_num}: {total_rows:,} rows processed, "
                  f"{match_count:,} matches, {elapsed:.0f}s elapsed "
                  f"({rate:,.0f} rows/s)")

    elapsed = time.time() - start_time
    print(f"[INFO] Finished: {total_rows:,} total rows in {elapsed:.1f}s "
          f"({total_rows/elapsed:,.0f} rows/s)")
    print(f"[INFO] Extracted {match_count:,} vital sign entries for "
          f"{len(vitals_by_stay)} ICU stays")

    return vitals_by_stay


def process_stay_vitals(vitals_list, intime_str, los_days):
    """
    Convert raw vital sign entries to hourly-binned array.
    Computes Mean Arterial Pressure (MAP) from systolic/diastolic when
    direct mean BP is not available: MAP = DBP + 1/3 * (SBP - DBP)

    Args:
        vitals_list: List of (charttime_str, vital_name, value) tuples.
        intime_str: ICU admission time string.
        los_days: Length of stay in days.

    Returns:
        numpy array of shape (num_hours, 5) with NaN for missing values,
        or None if insufficient data.
    """
    from datetime import datetime, timedelta

    intime = datetime.strptime(intime_str, '%Y-%m-%d %H:%M:%S')
    num_hours = int(los_days * 24)

    # Output signals: heart_rate, mean_bp, spo2, respiratory_rate, temperature
    vital_idx = {name: i for i, name in enumerate(SIGNAL_NAMES)}

    # Accumulators for hourly binning (includes sbp/dbp for MAP computation)
    all_vitals = ['heart_rate', 'mean_bp', 'sbp', 'dbp', 'spo2', 'respiratory_rate', 'temperature']
    vital_to_col = {name: i for i, name in enumerate(all_vitals)}

    hour_sums = np.zeros((num_hours, len(all_vitals)), dtype=np.float64)
    hour_counts = np.zeros((num_hours, len(all_vitals)), dtype=np.int32)

    for charttime_str, vital_name, value in vitals_list:
        try:
            charttime = datetime.strptime(charttime_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue

        # Compute hour offset from ICU admission
        delta = charttime - intime
        hour = int(delta.total_seconds() / 3600)

        if hour < 0 or hour >= num_hours:
            continue

        col = vital_to_col.get(vital_name)
        if col is None:
            continue

        hour_sums[hour, col] += value
        hour_counts[hour, col] += 1

    # Compute hourly means for all collected vitals
    hourly_means = np.full((num_hours, len(all_vitals)), np.nan, dtype=np.float32)
    mask = hour_counts > 0
    hourly_means[mask] = (hour_sums[mask] / hour_counts[mask]).astype(np.float32)

    # Build output array with 5 final signals
    data = np.full((num_hours, NUM_SIGNALS), np.nan, dtype=np.float32)

    # Heart rate
    data[:, 0] = hourly_means[:, vital_to_col['heart_rate']]

    # Mean BP: use direct mean if available, otherwise compute from SBP/DBP
    direct_map = hourly_means[:, vital_to_col['mean_bp']]
    sbp = hourly_means[:, vital_to_col['sbp']]
    dbp = hourly_means[:, vital_to_col['dbp']]

    # Computed MAP = DBP + 1/3 * (SBP - DBP)
    has_both = ~np.isnan(sbp) & ~np.isnan(dbp)
    computed_map = np.full(num_hours, np.nan, dtype=np.float32)
    computed_map[has_both] = dbp[has_both] + (sbp[has_both] - dbp[has_both]) / 3.0

    # Prefer direct mean BP, fall back to computed MAP
    data[:, 1] = np.where(~np.isnan(direct_map), direct_map, computed_map)

    # SpO2
    data[:, 2] = hourly_means[:, vital_to_col['spo2']]

    # Respiratory rate
    data[:, 3] = hourly_means[:, vital_to_col['respiratory_rate']]

    # Temperature
    data[:, 4] = hourly_means[:, vital_to_col['temperature']]

    # Quality check: need at least 30% of hours with heart rate data
    hr_available = np.sum(~np.isnan(data[:, 0]))
    if hr_available < num_hours * 0.3:
        return None

    return data


def main(num_patients=None):
    """Extract vital signs from CHARTEVENTS for ICU stays."""
    n_patients = num_patients if num_patients is not None else NUM_PATIENTS_NEEDED

    # Check if data already exists
    metadata_path = os.path.join(DATA_DIR, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        existing = len(meta.get('icustay_ids', []))
        if existing >= n_patients:
            print(f"[INFO] Data already exists at {DATA_DIR} ({existing} ICU stays)")
            print("[INFO] Skipping extraction. Delete data/ to re-extract.")
            return True

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("MIMIC-III Vital Signs — CHARTEVENTS Extraction (Phase 4)")
    print("=" * 60)
    print(f"Source: {CLINICAL_DIR}/CHARTEVENTS.csv.gz")
    print(f"Target: {n_patients} ICU stays with >= {MIN_STAY_HOURS}h LOS")
    print(f"Vital signs: {SIGNAL_NAMES}")
    print(f"Output: {DATA_DIR}")
    print(f"Method: Pandas chunked reading ({CHUNK_SIZE:,} rows/chunk)")
    print("=" * 60)

    overall_start = time.time()

    # Step 1: Load eligible ICU stays
    all_stays = load_icustays()

    # Sort by LOS (longest first) and take top N candidates
    # We request more candidates than needed to account for filtering
    sorted_stays = sorted(all_stays.values(), key=lambda x: -x['los_days'])
    candidate_stays = sorted_stays[:n_patients * 3]
    candidate_ids = set(s['icustay_id'] for s in candidate_stays)

    print(f"[INFO] Targeting {len(candidate_ids)} candidate ICU stays "
          f"(will keep best {n_patients})")

    # Step 2: Extract vital signs from CHARTEVENTS (optimized chunked reading)
    vitals_by_stay = extract_vital_signs_chunked(candidate_ids)

    # Step 3: Process and save one stay at a time
    saved_stays = []

    for stay_info in candidate_stays:
        if len(saved_stays) >= n_patients:
            break

        icustay_id = stay_info['icustay_id']
        if icustay_id not in vitals_by_stay:
            continue

        vitals_list = vitals_by_stay[icustay_id]
        if len(vitals_list) < 50:  # Need minimum vital sign entries
            continue

        data = process_stay_vitals(vitals_list, stay_info['intime'], stay_info['los_days'])
        if data is None:
            continue

        # Save to disk
        filepath = os.path.join(DATA_DIR, f"stay_{icustay_id}.npy")
        np.save(filepath, data)

        saved_stays.append({
            'icustay_id': icustay_id,
            'subject_id': stay_info['subject_id'],
            'hadm_id': stay_info['hadm_id'],
            'los_days': stay_info['los_days'],
            'num_hours': data.shape[0],
            'file': f"stay_{icustay_id}.npy",
        })

        # Free memory
        del vitals_by_stay[icustay_id]
        del data

        if len(saved_stays) % 10 == 0:
            print(f"[SAVED] {len(saved_stays)}/{n_patients} ICU stays processed")

    if not saved_stays:
        print("[FATAL] No valid ICU stays found.")
        return False

    # Step 4: Write metadata
    metadata = {
        'icustay_ids': [s['icustay_id'] for s in saved_stays],
        'signal_names': SIGNAL_NAMES,
        'num_signals': NUM_SIGNALS,
        'sampling_interval': 'hourly',
        'data_source': f'{CLINICAL_DIR}/CHARTEVENTS.csv.gz',
        'min_stay_hours': MIN_STAY_HOURS,
        'vital_bounds': VITAL_BOUNDS,
        'files': [s['file'] for s in saved_stays],
        'stay_info': {str(s['icustay_id']): s for s in saved_stays},
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {metadata_path}")

    total_hours = sum(s['num_hours'] for s in saved_stays)
    total_time = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print("Extraction Summary")
    print("=" * 60)
    print(f"  ICU stays:     {len(saved_stays)}")
    print(f"  Total hours:   {total_hours:,}")
    print(f"  Avg LOS:       {total_hours/len(saved_stays):.1f} hours")
    print(f"  Vital signs:   {SIGNAL_NAMES}")
    print(f"  Total time:    {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 60)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract vital signs from MIMIC-III CHARTEVENTS')
    parser.add_argument('--num-patients', type=int, default=NUM_PATIENTS_NEEDED,
                        help=f'Number of ICU stays to extract (default: {NUM_PATIENTS_NEEDED})')
    args = parser.parse_args()
    success = main(num_patients=args.num_patients)
    if not success:
        exit(1)
