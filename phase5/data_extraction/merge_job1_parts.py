#!/usr/bin/env python3
"""
Merge the 8 sub-job outputs (part_001_sub_000 .. part_001_sub_007) into part_001.

Run this after all sub-jobs complete:
  python3 merge_job1_parts.py
"""

import json
import shutil
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).parent / "output"
NUM_SUBS = 8
TARGET = "part_001"


def main():
    sub_dirs = []
    for i in range(NUM_SUBS):
        d = OUTPUT_DIR / f"part_001_sub_{i:03d}"
        if d.exists():
            sub_dirs.append(d)
        else:
            print(f"WARNING: {d} does not exist — skipping")

    if not sub_dirs:
        print("ERROR: No sub-job output directories found.")
        return

    print(f"Merging {len(sub_dirs)} sub-job directories into {TARGET}...")

    # Load and concatenate arrays
    all_features = []
    all_patient_ids = []
    all_seg_names = []
    all_window_times = []

    for d in sorted(sub_dirs):
        features_path = d / "features.npy"
        if not features_path.exists():
            print(f"  Skipping {d.name}: no features.npy")
            continue
        features = np.load(features_path)
        patient_ids = np.load(d / "patient_ids.npy", allow_pickle=True)
        seg_names = np.load(d / "seg_names.npy", allow_pickle=True)
        window_times = np.load(d / "window_times.npy")

        all_features.append(features)
        all_patient_ids.append(patient_ids)
        all_seg_names.append(seg_names)
        all_window_times.append(window_times)
        print(f"  {d.name}: {features.shape[0]} windows, {len(set(patient_ids))} patients")

    if not all_features:
        print("ERROR: No valid data found in sub-job directories.")
        return

    merged_features = np.concatenate(all_features, axis=0)
    merged_patient_ids = np.concatenate(all_patient_ids, axis=0)
    merged_seg_names = np.concatenate(all_seg_names, axis=0)
    merged_window_times = np.concatenate(all_window_times, axis=0)

    n_windows = merged_features.shape[0]
    n_patients = len(set(merged_patient_ids))
    print(f"\n  Total: {n_windows} windows across {n_patients} patients")

    # Save merged output
    target_dir = OUTPUT_DIR / TARGET
    target_dir.mkdir(parents=True, exist_ok=True)

    np.save(target_dir / "features.npy", merged_features)
    np.save(target_dir / "patient_ids.npy", merged_patient_ids)
    np.save(target_dir / "seg_names.npy", merged_seg_names)
    np.save(target_dir / "window_times.npy", merged_window_times)

    # Load one metadata file as template and update counts
    meta_path = sorted(sub_dirs)[0] / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    metadata["n_windows"] = int(n_windows)
    metadata["n_patients"] = int(n_patients)

    # Recompute nan_fraction and valid_counts across merged data
    nan_mask = np.isnan(merged_features)
    metadata["nan_fraction"] = float(nan_mask.sum()) / merged_features.size if merged_features.size > 0 else 0.0
    metadata["valid_counts"] = int((~nan_mask).all(axis=1).sum())

    with open(target_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Written to: {target_dir}")
    print("  DONE")

    # Optionally remove sub-directories
    response = input("\nRemove sub-job directories? [y/N] ")
    if response.lower() == "y":
        for d in sub_dirs:
            shutil.rmtree(d)
            print(f"  Removed {d.name}")


if __name__ == "__main__":
    main()
