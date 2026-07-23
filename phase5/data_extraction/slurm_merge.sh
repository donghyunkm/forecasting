#!/bin/bash
#SBATCH --job-name=p5_merge
#SBATCH --partition=cpu_short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

set -euo pipefail

PROJECT=/gpfs/home/dk5565/forecasting/phase5/data_extraction
cd "$PROJECT"

mkdir -p logs

echo "========================================"
echo "Phase 5 — Merge Extraction Outputs"
echo "Start: $(date)"
echo "========================================"

source /gpfs/share/apps/anaconda3/cpu/5.3.1/etc/profile.d/conda.sh
conda activate /gpfs/home/dk5565/.conda/envs/CSDI

python3 merge_outputs.py --config config/pipeline_config.yaml

echo "========================================"
echo "End: $(date)"
echo "========================================"
