#!/bin/bash
#SBATCH --job-name=phase42_data
#SBATCH --partition=cpu_medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_data_%j.out
#SBATCH --error=logs/slurm_data_%j.err

mkdir -p logs

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

cd /gpfs/home/dk5565/forecasting/phase42/tft

echo "========================================="
echo "Phase 4.2 — Data Preparation (chunk-level splits)"
echo "========================================="
echo ""

# Remove old processed tensors
echo "[CLEANUP] Removing old processed tensors..."
rm -f /gpfs/scratch/dk5565/phase42_data/processed/train_data.pt
rm -f /gpfs/scratch/dk5565/phase42_data/processed/val_data.pt
rm -f /gpfs/scratch/dk5565/phase42_data/processed/test_data.pt
rm -f /gpfs/scratch/dk5565/phase42_data/processed/norm_params.json
rm -f /gpfs/scratch/dk5565/phase42_data/processed/split_info.json
echo "  Done."
echo ""

python prepare_data.py
