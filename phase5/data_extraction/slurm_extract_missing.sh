#!/bin/bash
#SBATCH --job-name=p5_extract_missing
#SBATCH --partition=cpu_medium
#SBATCH --array=0,1,3,4,5,6
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/extract_missing_%A_%a.out
#SBATCH --error=logs/extract_missing_%A_%a.err

set -euo pipefail

PROJECT=/gpfs/home/dk5565/forecasting/phase5/data_extraction
cd "$PROJECT"

mkdir -p logs output

echo "========================================"
echo "Phase 5 — Feature Extraction (Missing Parts)"
echo "Start:    $(date)"
echo "Node:     $(hostname)"
echo "Job:      $SLURM_ARRAY_JOB_ID"
echo "Task:     $SLURM_ARRAY_TASK_ID"
echo "========================================"

# Activate conda environment
source /gpfs/share/apps/anaconda3/cpu/5.3.1/etc/profile.d/conda.sh
conda activate /gpfs/home/dk5565/.conda/envs/CSDI

python3 extract_features.py \
    --config config/pipeline_config.yaml \
    --job-idx "$SLURM_ARRAY_TASK_ID" \
    --num-jobs 50

echo "========================================"
echo "End:      $(date)"
echo "========================================"
