#!/usr/bin/env python3
"""
Phase 41 - Extract vital signs from MIMIC-III waveform numerics data at 15-minute resolution.

Extracts 4 vital signs: mean_bp, pulse, spo2, respiratory_rate
from MIMIC-III matched waveform numerics records, aggregates to 15-min bins,
applies physiological bounds filtering, and saves per-CHUNK .npy files.

A "chunk" is a continuous recording segment. Gaps between records (detected by
comparing record end time vs. next record start time) split the data into
separate chunks. Each chunk is saved as an independent .npy file.

Output: per-chunk .npy files (N x 4) + metadata.json
        Filenames: {patient_id}_chunk{i}.npy (e.g., p000033_chunk0.npy)
"""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
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

# Maximum gap (in minutes) between end of one record and start of next
# to still consider them part of the same continuous chunk.
# If the gap exceeds this, a new chunk starts.
MAX_GAP_MINUTES = 30

# Physiological bounds
BOUNDS = {
    "pulse": (20, 250),
    "mean_bp": (20, 200),
    "sbp": (40, 300),
    "dbp": (10, 200),
    "spo2": (50, 100),
    "resp": (4, 60),
}

# Quality thresholds (per chunk)
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
        int: number of minutes (at ~1 sample/min resolution)
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
    return binned


def get_record_duration_minutes(record_path):
    """Get the duration of a record in minutes without reading full data."""
    full_path = os.path.join(WAVEFORM_BASE, record_path)
    try:
        rec = wfdb.rdrecord(full_path)
    except Exception:
        return 0

    if rec.p_signal is None:
        return 0

    n_samples = rec.p_signal.shape[0]
    fs = rec.fs if rec.fs else 0.0167

    if fs >= 0.5:
        # fs=1 Hz → n_samples seconds → n_samples/60 minutes
        return n_samples / (fs * 60)
    else:
        # fs~0.0167 → already 1 sample/min
        return n_samples


def group_records_into_chunks(record_paths):
    """Group a patient's records into continuous chunks based on time gaps.

    Records within MAX_GAP_MINUTES of each other are considered part of the
    same continuous chunk.

    Returns:
        list of lists: each inner list is a group of record paths forming
                       one continuous chunk.
    """
    if not record_paths:
        return []

    if len(record_paths) == 1:
        return [record_paths]

    # Get start datetime for each record
    record_info = []
    for rec_path in record_paths:
        start_dt = extract_datetime_from_path(rec_path)
        record_info.append((rec_path, start_dt))

    # Sort by start time (should already be sorted, but ensure)
    record_info.sort(key=lambda x: x[1])

    # Group into chunks: a new chunk starts when the gap between end of
    # previous record and start of current record exceeds MAX_GAP_MINUTES
    chunks = [[record_info[0][0]]]
    prev_start = record_info[0][1]
    prev_duration = get_record_duration_minutes(record_info[0][0])

    for i in range(1, len(record_info)):
        curr_path, curr_start = record_info[i]

        # Estimate end time of previous record
        prev_end = prev_start + timedelta(minutes=prev_duration)

        # Gap = current start - previous end
        gap_minutes = (curr_start - prev_end).total_seconds() / 60.0

        if gap_minutes > MAX_GAP_MINUTES:
            # Start a new chunk
            chunks.append([curr_path])
        else:
            # Continue current chunk
            chunks[-1].append(curr_path)

        prev_start = curr_start
        prev_duration = get_record_duration_minutes(curr_path)

    return chunks


def process_chunk(record_paths):
    """Process a group of records forming one continuous chunk.

    Returns:
        np.ndarray: shape (N_15min_bins, 4) or None if insufficient data
        int: total minutes processed
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
        return None, 0, failed_records

    # Concatenate records within this chunk (they are continuous)
    concatenated = np.vstack(all_minute_data)

    # Aggregate to 15-minute bins
    binned = aggregate_to_15min(concatenated)

    return binned, total_minutes, failed_records


def process_patient(patient_id, record_paths):
    """Process all records for a single patient, splitting into continuous chunks.

    Returns:
        list of tuples: [(binned_data, metadata), ...] — one per qualified chunk
    """
    # Group records into continuous chunks based on time gaps
    chunks = group_records_into_chunks(record_paths)

    results = []

    for chunk_idx, chunk_records in enumerate(chunks):
        binned_data, total_minutes, failed_records = process_chunk(chunk_records)

        if binned_data is None:
            continue

        n_bins = binned_data.shape[0]
        duration_hours = (n_bins * BIN_SIZE_MINUTES) / 60.0

        # Compute coverage stats
        coverage = {}
        col_names = ["mean_bp", "pulse", "spo2", "respiratory_rate"]
        for i, name in enumerate(col_names):
            valid_count = np.sum(~np.isnan(binned_data[:, i]))
            coverage[name] = float(valid_count) / n_bins if n_bins > 0 else 0.0

        metadata = {
            "patient_id": patient_id,
            "chunk_index": chunk_idx,
            "chunk_id": f"{patient_id}_chunk{chunk_idx}",
            "num_intervals": int(n_bins),
            "duration_hours": round(duration_hours, 2),
            "num_records_in_chunk": len(chunk_records),
            "total_records": len(record_paths),
            "total_chunks": len(chunks),
            "failed_records": failed_records,
            "signal_coverage": {k: round(v, 4) for k, v in coverage.items()},
        }

        results.append((binned_data, metadata))

    return results


def quality_check(metadata):
    """Check if a chunk meets quality thresholds."""
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
        help="Number of patients to process (0 = all). Default: 0",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Parse records file
    print("=" * 60)
    print("Phase 41: MIMIC-III Waveform Numerics Vital Signs Extraction")
    print("         (Per-Chunk Output with Gap Detection)")
    print("=" * 60)
    print(f"\nWaveform base: {WAVEFORM_BASE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Max gap between records: {MAX_GAP_MINUTES} minutes")
    print(f"Min chunk duration: {MIN_DURATION_HOURS} hours")
    print(f"Min pulse coverage: {MIN_PULSE_COVERAGE*100:.0f}%")
    print(f"Target patients: {'all' if args.num_patients == 0 else args.num_patients}")
    print()

    records = parse_records_file(RECORDS_FILE)

    # Step 2-3: Group by patient
    patient_records = group_records_by_patient(records)

    # Step 4-7: Process each patient → save per-chunk files
    print(f"\nProcessing {len(patient_records)} patients...")
    print("-" * 60)

    qualified_chunks = []
    total_patients = len(patient_records)
    total_chunks_found = 0
    total_chunks_qualified = 0

    for idx, (patient_id, rec_paths) in enumerate(patient_records.items()):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(
                f"  Processing patient {idx + 1}/{total_patients}: {patient_id} "
                f"({len(rec_paths)} records)..."
            )

        chunk_results = process_patient(patient_id, rec_paths)
        total_chunks_found += len(chunk_results)

        for binned_data, metadata in chunk_results:
            if quality_check(metadata):
                qualified_chunks.append((binned_data, metadata))
                total_chunks_qualified += 1

        if (idx + 1) % 500 == 0:
            print(
                f"    -> {total_chunks_qualified} chunks qualified so far "
                f"(from {total_chunks_found} total chunks)"
            )

    print(f"\nChunk statistics:")
    print(f"  Total chunks found: {total_chunks_found}")
    print(f"  Chunks qualified:   {total_chunks_qualified}")
    print(
        f"  (required: >= {MIN_DURATION_HOURS}h duration, "
        f">= {MIN_PULSE_COVERAGE*100:.0f}% pulse coverage)"
    )

    # Sort by duration (longest first)
    qualified_chunks.sort(key=lambda x: x[1]["duration_hours"], reverse=True)

    if qualified_chunks:
        durations = [m["duration_hours"] for _, m in qualified_chunks]
        print(
            f"  Duration range: {min(durations):.1f}h - {max(durations):.1f}h"
        )
        print(f"  Mean duration: {np.mean(durations):.1f}h")

    # Step 8: Save outputs
    print(f"\nSaving outputs to {OUTPUT_DIR}...")
    all_metadata = []

    for binned_data, metadata in qualified_chunks:
        chunk_id = metadata["chunk_id"]
        npy_path = os.path.join(OUTPUT_DIR, f"{chunk_id}.npy")
        np.save(npy_path, binned_data)
        all_metadata.append(metadata)

    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"  Saved {len(qualified_chunks)} .npy files (per-chunk)")
    print(f"  Saved metadata.json")
    print("\nDone!")

    # Summary statistics
    if all_metadata:
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        unique_patients = len(set(m["patient_id"] for m in all_metadata))
        avg_duration = np.mean([m["duration_hours"] for m in all_metadata])
        avg_intervals = np.mean([m["num_intervals"] for m in all_metadata])
        print(f"  Unique patients: {unique_patients}")
        print(f"  Total chunks: {len(all_metadata)}")
        print(f"  Avg chunks per patient: {len(all_metadata) / max(unique_patients, 1):.2f}")
        print(f"  Avg duration: {avg_duration:.1f} hours")
        print(f"  Avg intervals: {avg_intervals:.0f}")
        for sig in ["mean_bp", "pulse", "spo2", "respiratory_rate"]:
            avg_cov = np.mean(
                [m["signal_coverage"][sig] for m in all_metadata]
            )
            print(f"  Avg {sig} coverage: {avg_cov*100:.1f}%")


if __name__ == "__main__":
    main()
