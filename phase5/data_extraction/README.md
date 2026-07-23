# Phase 5 — Data Extraction Pipeline

Feature extraction from MIMIC-III waveform data. Produces 11-dim feature vectors
(7 waveform correlations + 4 vital signs) at 5-min stride.

## Source

- **Waveforms:** `/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched`
- **Clinical (for segment alignment):** `/gpfs/data/eh3828lab/datasets/mimic_clinical`

## Pipeline

```
125 Hz waveforms (II, PLETH, RESP, ABP)
    │
    ▼  20-min sliding window, 5-min stride
    │
    ├── 118 sub-windows (30s each, 10s stride)
    │       │
    │       ▼  19 beat-by-beat features per sub-window
    │       │   (HR, RR, SBP, DBP, MAP, ABP_area, PLETH_ACDC, ...)
    │       │
    │       ▼  Pearson r across sub-windows for 7 feature pairs
    │
    ├── Numerics (~1/min) → median vitals over 20-min window
    │
    ▼  Output: (N, 11) per extraction part
```

## Usage

```bash
# Step 1: Run extraction (50 parallel jobs, ~2h each)
sbatch slurm_extract.sh

# Step 2: Merge outputs (after all parts complete)
sbatch slurm_merge.sh

# Step 3: Then run Phase 5 data preparation
cd ../tft && python prepare_data.py
```

## Output

```
output/
├── part_000/ through part_049/    # Individual job outputs
│   ├── features.npy              # (N_part, 11)
│   ├── patient_ids.npy           # (N_part,)
│   ├── seg_names.npy             # (N_part,)
│   ├── window_times.npy          # (N_part,)
│   └── metadata.json
└── merged/                        # Combined from all parts
    ├── features.npy              # (N_total, 11)
    ├── patient_ids.npy
    ├── seg_names.npy
    ├── window_times.npy
    └── metadata.json
```

## Current Status

- 44/50 parts extracted (missing: 0, 1, 3, 4, 5, 6)
- Merged data currently at: `/gpfs/home/dk5565/mimicEran/output/merged/`
- 1,153,097 total windows from 1,725 patients

## Notes

- Extraction requires all 4 waveform channels (II, PLETH, RESP, ABP) simultaneously
- Only ~2,060 patients qualify (not all have arterial lines)
- Each job takes ~1.5-3 hours on cpu_short
- Conda env: `/gpfs/home/dk5565/.conda/envs/CSDI`
