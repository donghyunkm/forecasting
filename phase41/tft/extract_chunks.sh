#!/bin/bash
#SBATCH --job-name=phase41_extract
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_extract_%j.out
#SBATCH --error=logs/slurm_extract_%j.err

mkdir -p logs

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

cd /gpfs/home/dk5565/forecasting/phase41/tft

echo "========================================="
echo "Phase 4.1 — Re-extraction with chunk logic"
echo "========================================="
echo ""

# Step 1: Remove old per-patient .npy files and metadata
echo "[CLEANUP] Removing old per-patient .npy files..."
rm -f /gpfs/scratch/dk5565/phase41_data/p*.npy
rm -f /gpfs/scratch/dk5565/phase41_data/metadata.json
echo "  Done."

# Step 2: Remove old processed tensors (will be regenerated)
echo "[CLEANUP] Removing old processed tensors..."
rm -f /gpfs/scratch/dk5565/phase41_data/processed/train_data.pt
rm -f /gpfs/scratch/dk5565/phase41_data/processed/val_data.pt
rm -f /gpfs/scratch/dk5565/phase41_data/processed/test_data.pt
rm -f /gpfs/scratch/dk5565/phase41_data/processed/norm_params.json
rm -f /gpfs/scratch/dk5565/phase41_data/processed/split_info.json
echo "  Done."
echo ""

# Step 3: Run new extraction (per-chunk output)
echo "[EXTRACT] Running download_data.py (per-chunk extraction with gap detection)..."
python download_data.py
EXTRACT_STATUS=$?

if [ $EXTRACT_STATUS -ne 0 ]; then
    echo "ERROR: download_data.py failed with exit code $EXTRACT_STATUS"
    exit 1
fi

echo ""
echo "[PREPROCESS] Running prepare_data.py (chunk-level splits)..."
python prepare_data.py --skip-download
PREP_STATUS=$?

if [ $PREP_STATUS -ne 0 ]; then
    echo "ERROR: prepare_data.py failed with exit code $PREP_STATUS"
    exit 1
fi

echo ""
echo "Phase 4.1 extraction + preprocessing complete."
