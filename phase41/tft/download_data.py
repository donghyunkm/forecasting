#!/usr/bin/env python3
"""
Phase 41 - Extract vital signs from MIMIC-III waveform numerics data at 15-minute resolution.

Extracts 4 vital signs: mean_bp, pulse, spo2, respiratory_rate
from MIMIC-III matched waveform numerics records, aggregates to 15-min bins,
applies physiological bounds filtering, and selects top patients by duration.

Output: per-patient .npy files (N x 4) + metadata.json
"""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import wfdb


# ============================================================================
# Configuration
# ============================================================================

WAVEFORM_BASE = "/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched"
RECORDS_FILE = os.path.join(WAVEFORM_BASE, "RECORDS-numerics")
OUTPUT_DIR = "/gpfs/scratch/dk5565/phase41_data/"

# 15-minute bin size in samples (assuming ~1 sample/min baseline)
BIN_SIZE_MINUTES = 15

# Physiological bounds
BOUNDS = {
    "pulse": (20, 250),
    "mean_bp": (20, 200),
    "sbp": (40, 300),
    "dbp": (10, 200),
    "spo2": (50, 100),
    "resp": (4, 60),
}

# Quality thresholds
MIN_DURATION_HOURS = 25
MIN_PULSE_COVERAGE = 0.30

# Signal name normalization mapping
SIGNAL_NAME_MAP = {
    # Heart rate / Pulse
    "HR": "HR",
    "Heart Rate": "HR",
    "PULSE": "PULSE",
    "Pulse": "PULSE",
    # Arterial blood pressure
    "ABPSys": "ABPSys",
    "ABP Sys": "ABPSys",
    "ART Sys": "ABPSys",
    "ABPDias": "ABPDias",
    "ABP Dias": "ABPDias",
    "ART Dias": "ABPDias",
    "ABPMean": "ABPMean",
    "ABP Mean": "ABPMean",
    "ART Mean": "ABPMean",
    # Non-invasive blood pressure
    "NBPSys": "NBPSys",
    "NBP Sys": "NBPSys",
    "NBPDias": "NBPDias",
    "NBP Dias": "NBPDias",
    "NBPMean": "NBPMean",
    "NBP Mean": "NBPMean",
    # Respiratory
    "RESP": "RESP",
    "Resp": "RESP",
    # SpO2
    "SpO2": "SpO2",
}


# ============================================================================
# Helper functions
# ============================================================================


def normalize_signal_name(name):
    """Normalize a signal name using the mapping, return None if not recognized."""
    name = name.strip()
    return SIGNAL_NAME_MAP.get(name, None)


def parse_records_file(records_path):
    """Parse the RECORDS-numerics file to get all numeric record paths."""
    records = []
    with open(records_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(line)
    print(f"Parsed {len(records)} numeric record paths from RECORDS-numerics")
    return records


def extract_patient_id(record_path):
    """Extract patient ID from record path like 'p00/p000020/p000020-2183-04-28-17-47n'."""
    parts = record_path.split("/")
    if len(parts) >= 2:
        return parts[1]  # e.g., 'p000020'
    return None


def extract_datetime_from_path(record_path):
    """Extract datetime from record path for chronological sorting.
    Path format: p00/p000020/p000020-2183-04-28-17-47n
    """
    basename = record_path.split("/")[-1]
    # Match pattern like p000020-2183-04-28-17-47n
    match = re.search(r"-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})n?$", basename)
    if match:
        dt_str = match.group(1)
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d-%H-%M")
        except ValueError:
            pass
    return datetime.min


def group_records_by_patient(records):
    """Group record paths by patient ID."""
    patient_records = defaultdict(list)
    for rec_path in records:
        patient_id = extract_patient_id(rec_path)
        if patient_id:
            patient_records[patient_id].append(rec_path)

    # Sort each patient's records chronologically
    for patient_id in patient_records:
        patient_records[patient_id].sort(key=extract_datetime_from_path)

    print(f"Grouped records into {len(patient_records)} patients")
    return patient_records


def apply_bounds(values, low, high):
    """Set values outside [low, high] to NaN."""
    values = values.copy()
    values[(values < low) | (values > high)] = np.nan
    return values


def compute_map_from_sys_dias(sbp, dbp):
    """Compute mean arterial pressure from systolic and diastolic: MAP = DBP + (SBP - DBP) / 3."""
    # Apply bounds first
    sbp = apply_bounds(sbp, *BOUNDS["sbp"])
    dbp = apply_bounds(dbp, *BOUNDS["dbp"])
    # Compute MAP where both are valid
    valid = ~np.isnan(sbp) & ~np.isnan(dbp)
    result = np.full_like(sbp, np.nan)
    result[valid] = dbp[valid] + (sbp[valid] - dbp[valid]) / 3.0
    return result


def read_record_signals(record_path):
    """Read a single numeric record and return normalized signal data.

    Returns:
        dict: mapping normalized signal names to 1D arrays (1 sample/min),
              or None if the record cannot be read.
        int: number of samples (at ~1 sample/min resolution)
    """
    full_path = os.path.join(WAVEFORM_BASE, record_path)
    try:
        rec = wfdb.rdrecord(full_path)
    except Exception:
        return None, 0

    if rec.p_signal is None or rec.sig_name is None:
        return None, 0

    n_samples = rec.p_signal.shape[0]
    fs = rec.fs if rec.fs else 0.0167

    signals = {}
    for i, raw_name in enumerate(rec.sig_name):
        norm_name = normalize_signal_name(raw_name)
        if norm_name is not None:
            channel_data = rec.p_signal[:, i].astype(np.float64)
            # Replace invalid values (wfdb uses specific sentinel values)
            channel_data[channel_data < -1e6] = np.nan
            channel_data[channel_data > 1e6] = np.nan

            # If fs=1 (1 Hz), downsample to ~1 sample/min by taking minute means
            if fs >= 0.5:  # fs=1 means 1 sample/sec
                samples_per_min = int(round(fs * 60))
                n_minutes = n_samples // samples_per_min
                if n_minutes > 0:
                    trimmed = channel_data[: n_minutes * samples_per_min]
                    reshaped = trimmed.reshape(n_minutes, samples_per_min)
                    with np.errstate(all="ignore"):
                        channel_data = np.nanmean(reshaped, axis=1)
                    # If all values in a minute were NaN, nanmean gives NaN (correct)
                else:
                    channel_data = np.array([np.nanmean(channel_data)])
            # else: fs~0.0167 means already ~1 sample/min, keep as is

            signals[norm_name] = channel_data

    # Determine number of minutes
    if fs >= 0.5:
        n_minutes = n_samples // int(round(fs * 60))
        n_minutes = max(n_minutes, 1)
    else:
        n_minutes = n_samples

    return signals, n_minutes


def extract_vitals_per_minute(signals, n_minutes):
    """Extract the 4 vital signs at per-minute resolution from normalized signals.

    Returns:
        np.ndarray of shape (n_minutes, 4): [mean_bp, pulse, spo2, resp_rate]
    """
    vitals = np.full((n_minutes, 4), np.nan)

    # --- Mean BP (column 0) ---
    # Priority: ABPMean > MAP(ABPSys, ABPDias) > NBPMean > MAP(NBPSys, NBPDias)
    mean_bp = np.full(n_minutes, np.nan)

    if "ABPMean" in signals:
        abp_mean = signals["ABPMean"][:n_minutes]
        padded = np.full(n_minutes, np.nan)
        padded[: len(abp_mean)] = abp_mean
        padded = apply_bounds(padded, *BOUNDS["mean_bp"])
        valid = ~np.isnan(padded)
        mean_bp[valid] = padded[valid]

    if "ABPSys" in signals and "ABPDias" in signals:
        abp_sys = signals["ABPSys"][:n_minutes]
        abp_dias = signals["ABPDias"][:n_minutes]
        min_len = min(len(abp_sys), len(abp_dias), n_minutes)
        sys_padded = np.full(n_minutes, np.nan)
        dias_padded = np.full(n_minutes, np.nan)
        sys_padded[:min_len] = abp_sys[:min_len]
        dias_padded[:min_len] = abp_dias[:min_len]
        computed_map = compute_map_from_sys_dias(sys_padded, dias_padded)
        computed_map = apply_bounds(computed_map, *BOUNDS["mean_bp"])
        # Fill only where ABPMean didn't provide values
        fill_mask = np.isnan(mean_bp) & ~np.isnan(computed_map)
        mean_bp[fill_mask] = computed_map[fill_mask]

    if "NBPMean" in signals:
        nbp_mean = signals["NBPMean"][:n_minutes]
        padded = np.full(n_minutes, np.nan)
        padded[: len(nbp_mean)] = nbp_mean
        padded = apply_bounds(padded, *BOUNDS["mean_bp"])
        fill_mask = np.isnan(mean_bp) & ~np.isnan(padded)
        mean_bp[fill_mask] = padded[fill_mask]

    if "NBPSys" in signals and "NBPDias" in signals:
        nbp_sys = signals["NBPSys"][:n_minutes]
        nbp_dias = signals["NBPDias"][:n_minutes]
        min_len = min(len(nbp_sys), len(nbp_dias), n_minutes)
        sys_padded = np.full(n_minutes, np.nan)
        dias_padded = np.full(n_minutes, np.nan)
        sys_padded[:min_len] = nbp_sys[:min_len]
        dias_padded[:min_len] = nbp_dias[:min_len]
        computed_map = compute_map_from_sys_dias(sys_padded, dias_padded)
        computed_map = apply_bounds(computed_map, *BOUNDS["mean_bp"])
        fill_mask = np.isnan(mean_bp) & ~np.isnan(computed_map)
        mean_bp[fill_mask] = computed_map[fill_mask]

    vitals[:, 0] = mean_bp

    # --- Pulse (column 1) ---
    # Priority: PULSE > HR
    pulse = np.full(n_minutes, np.nan)

    if "PULSE" in signals:
        p = signals["PULSE"][:n_minutes]
        padded = np.full(n_minutes, np.nan)
        padded[: len(p)] = p
        padded = apply_bounds(padded, *BOUNDS["pulse"])
        valid = ~np.isnan(padded)
        pulse[valid] = padded[valid]

    if "HR" in signals:
        hr = signals["HR"][:n_minutes]
        padded = np.full(n_minutes, np.nan)
        padded[: len(hr)] = hr
        padded = apply_bounds(padded, *BOUNDS["pulse"])
        fill_mask = np.isnan(pulse) & ~np.isnan(padded)
        pulse[fill_mask] = padded[fill_mask]

    vitals[:, 1] = pulse

    # --- SpO2 (column 2) ---
    if "SpO2" in signals:
        spo2 = signals["SpO2"][:n_minutes]
        padded = np.full(n_minutes, np.nan)
        padded[: len(spo2)] = spo2
        padded = apply_bounds(padded, *BOUNDS["spo2"])
        vitals[:, 2] = padded

    # --- Respiratory rate (column 3) ---
    if "RESP" in signals:
        resp = signals["RESP"][:n_minutes]
        padded = np.full(n_minutes, np.nan)
        padded[: len(resp)] = resp
        padded = apply_bounds(padded, *BOUNDS["resp"])
        vitals[:, 3] = padded

    return vitals


def aggregate_to_15min(minute_data):
    """Aggregate per-minute data (N_minutes x 4) into 15-minute bins.

    Takes the mean of available (non-NaN) values in each 15-min window.
    """
    n_minutes = minute_data.shape[0]
    n_bins = n_minutes // BIN_SIZE_MINUTES
    if n_bins == 0:
        return np.full((1, 4), np.nan)

    # Trim to exact multiple of 15
    trimmed = minute_data[: n_bins * BIN_SIZE_MINUTES]
    reshaped = trimmed.reshape(n_bins, BIN_SIZE_MINUTES, 4)

    with np.errstate(all="ignore"):
        binned = np.nanmean(reshaped, axis=1)

    # If all values in a bin were NaN, nanmean returns NaN (correct behavior)
    # But numpy may produce RuntimeWarning for all-NaN slices; suppress above
    return binned


def process_patient(patient_id, record_paths):
    """Process all records for a single patient.

    Returns:
        np.ndarray: shape (N_15min_bins, 4) or None if insufficient data
        dict: metadata for this patient
    """
    all_minute_data = []
    total_minutes = 0
    failed_records = 0

    for rec_path in record_paths:
        signals, n_minutes = read_record_signals(rec_path)
        if signals is None or n_minutes == 0:
            failed_records += 1
            continue

        vitals = extract_vitals_per_minute(signals, n_minutes)
        all_minute_data.append(vitals)
        total_minutes += n_minutes

    if not all_minute_data:
        return None, None

    # Concatenate all records chronologically
    concatenated = np.vstack(all_minute_data)

    # Aggregate to 15-minute bins
    binned = aggregate_to_15min(concatenated)

    # Compute coverage stats
    n_bins = binned.shape[0]
    duration_hours = (n_bins * BIN_SIZE_MINUTES) / 60.0

    coverage = {}
    col_names = ["mean_bp", "pulse", "spo2", "respiratory_rate"]
    for i, name in enumerate(col_names):
        valid_count = np.sum(~np.isnan(binned[:, i]))
        coverage[name] = float(valid_count) / n_bins if n_bins > 0 else 0.0

    metadata = {
        "patient_id": patient_id,
        "num_intervals": int(n_bins),
        "duration_hours": round(duration_hours, 2),
        "total_records": len(record_paths),
        "failed_records": failed_records,
        "signal_coverage": {k: round(v, 4) for k, v in coverage.items()},
    }

    return binned, metadata


def quality_check(metadata):
    """Check if patient meets quality thresholds."""
    if metadata is None:
        return False
    if metadata["duration_hours"] < MIN_DURATION_HOURS:
        return False
    if metadata["signal_coverage"]["pulse"] < MIN_PULSE_COVERAGE:
        return False
    return True


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract vital signs from MIMIC-III waveform numerics at 15-min resolution."
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=0,
        help="Number of top patients to select (by duration). 0 = all qualified. Default: 0",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Parse records file
    print("=" * 60)
    print("Phase 41: MIMIC-III Waveform Numerics Vital Signs Extraction")
    print("=" * 60)
    print(f"\nWaveform base: {WAVEFORM_BASE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Target patients: {'all qualified' if args.num_patients == 0 else args.num_patients}")
    print()

    records = parse_records_file(RECORDS_FILE)

    # Step 2-3: Group by patient
    patient_records = group_records_by_patient(records)

    # Step 4-7: Process each patient
    print(f"\nProcessing {len(patient_records)} patients...")
    print("-" * 60)

    qualified_patients = []
    total_patients = len(patient_records)

    for idx, (patient_id, rec_paths) in enumerate(patient_records.items()):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(
                f"  Processing patient {idx + 1}/{total_patients}: {patient_id} "
                f"({len(rec_paths)} records)..."
            )

        binned_data, metadata = process_patient(patient_id, rec_paths)

        if binned_data is not None and quality_check(metadata):
            qualified_patients.append((binned_data, metadata))

        if (idx + 1) % 500 == 0:
            print(
                f"    -> {len(qualified_patients)} patients qualified so far"
            )

    print(f"\n{len(qualified_patients)} patients passed quality checks")
    print(
        f"  (required: >= {MIN_DURATION_HOURS}h duration, "
        f">= {MIN_PULSE_COVERAGE*100:.0f}% pulse coverage)"
    )

    # Step 8: Select patients (all qualified, or top N by duration)
    qualified_patients.sort(key=lambda x: x[1]["duration_hours"], reverse=True)
    if args.num_patients > 0:
        selected = qualified_patients[: args.num_patients]
        print(f"\nSelected top {len(selected)} patients by duration")
    else:
        selected = qualified_patients
        print(f"\nSelected all {len(selected)} qualified patients")
    if selected:
        durations = [m["duration_hours"] for _, m in selected]
        print(
            f"  Duration range: {min(durations):.1f}h - {max(durations):.1f}h"
        )

    # Step 9: Save outputs
    print(f"\nSaving outputs to {OUTPUT_DIR}...")
    all_metadata = []

    for binned_data, metadata in selected:
        patient_id = metadata["patient_id"]
        npy_path = os.path.join(OUTPUT_DIR, f"{patient_id}.npy")
        np.save(npy_path, binned_data)
        all_metadata.append(metadata)

    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"  Saved {len(selected)} .npy files")
    print(f"  Saved metadata.json")
    print("\nDone!")

    # Summary statistics
    if all_metadata:
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        avg_duration = np.mean([m["duration_hours"] for m in all_metadata])
        avg_intervals = np.mean([m["num_intervals"] for m in all_metadata])
        print(f"  Avg duration: {avg_duration:.1f} hours")
        print(f"  Avg intervals: {avg_intervals:.0f}")
        for sig in ["mean_bp", "pulse", "spo2", "respiratory_rate"]:
            avg_cov = np.mean(
                [m["signal_coverage"][sig] for m in all_metadata]
            )
            print(f"  Avg {sig} coverage: {avg_cov*100:.1f}%")


if __name__ == "__main__":
    main()
