#!/usr/bin/env python3
"""
Merge outputs from parallel extraction jobs into a single dataset.

Usage:
    python3 merge_outputs.py --config ../config/pipeline_config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="Merge parallel extraction outputs")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["output_dir"])
    num_jobs = cfg["processing"]["num_jobs"]

    print("=" * 80)
    print("MERGING PARALLEL OUTPUTS")
    print("=" * 80)

    # Collect all parts
    all_features = []
    all_patient_ids = []
    all_seg_names = []
    all_window_times = []
    missing_parts = []

    for j in range(num_jobs):
        part_dir = output_dir / f"part_{j:03d}"
        feat_path = part_dir / "features.npy"
        if not feat_path.exists():
            missing_parts.append(j)
            continue

        features = np.load(feat_path)
        patient_ids = np.load(part_dir / "patient_ids.npy", allow_pickle=True)
        seg_names = np.load(part_dir / "seg_names.npy", allow_pickle=True)
        window_times = np.load(part_dir / "window_times.npy")

        all_features.append(features)
        all_patient_ids.append(patient_ids)
        all_seg_names.append(seg_names)
        all_window_times.append(window_times)

        print(f"  Part {j:03d}: {len(features)} windows")

    if missing_parts:
        print(f"\n  WARNING: Missing parts: {missing_parts}")

    if not all_features:
        print("  ERROR: No parts found to merge.")
        return

    # Concatenate
    features = np.concatenate(all_features, axis=0)
    patient_ids = np.concatenate(all_patient_ids)
    seg_names = np.concatenate(all_seg_names)
    window_times = np.concatenate(all_window_times)

    # Save merged
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    np.save(merged_dir / "features.npy", features)
    np.save(merged_dir / "patient_ids.npy", patient_ids)
    np.save(merged_dir / "seg_names.npy", seg_names)
    np.save(merged_dir / "window_times.npy", window_times)

    # Summary metadata
    pair_names = [f"{a} × {b}" for a, b in cfg["correlation_pairs"]]
    vital_names = [v["name"] for v in cfg["vital_signs"]["signals"]]

    metadata = {
        "n_windows": len(features),
        "n_patients": len(np.unique(patient_ids)),
        "feature_dim": features.shape[1],
        "feature_names": pair_names + vital_names,
        "window_duration_min": cfg["window"]["duration_min"],
        "nan_fraction": float(np.isnan(features).sum() / features.size),
        "parts_merged": num_jobs - len(missing_parts),
        "parts_missing": missing_parts,
    }

    with open(merged_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Merged: {features.shape[0]} windows from "
          f"{len(np.unique(patient_ids))} patients")
    print(f"  Shape: {features.shape}")
    print(f"  NaN fraction: {metadata['nan_fraction']:.3f}")
    print(f"  Saved to: {merged_dir}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
