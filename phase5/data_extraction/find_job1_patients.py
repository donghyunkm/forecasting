#!/usr/bin/env python3
"""
Identify the patients assigned to job index 1 (out of 50) using the same
partitioning logic as extract_features.py, and write them to a text file.

This replicates:
  1. Scanning all qualifying patients (same as extract_features.py Pass 1)
  2. Sorting patient IDs
  3. Selecting patients where index % 50 == 1

Output: job1_patients.txt (one patient ID per line)
"""

import sys
from pathlib import Path

import yaml

# Add project dir to path for importing scan logic
sys.path.insert(0, str(Path(__file__).parent))
from extract_features import load_config, setup_params, scan_patients


def main():
    config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"
    cfg = load_config(str(config_path))
    params = setup_params(cfg)

    print("Scanning for qualifying patients...")
    patient_segs = scan_patients(cfg, params)
    patient_ids_sorted = sorted(patient_segs.keys())

    print(f"Total qualifying patients: {len(patient_ids_sorted)}")

    # Same partitioning as extract_features.py: i % num_jobs == job_idx
    num_jobs = 50
    job_idx = 1
    job1_patients = [p for i, p in enumerate(patient_ids_sorted)
                     if i % num_jobs == job_idx]

    print(f"Job {job_idx}/{num_jobs}: {len(job1_patients)} patients")

    output_file = Path(__file__).parent / "job1_patients.txt"
    with open(output_file, "w") as f:
        for pid in job1_patients:
            f.write(pid + "\n")

    print(f"Written to: {output_file}")


if __name__ == "__main__":
    main()
