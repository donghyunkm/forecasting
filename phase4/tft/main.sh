#!/bin/bash
#SBATCH --job-name=tft_forecast_p4
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/home/dk5565/forecasting/phase4/tft/logs/train_%j.out
#SBATCH --error=/gpfs/home/dk5565/forecasting/phase4/tft/logs/train_%j.err

# Create logs directory
mkdir -p /gpfs/home/dk5565/forecasting/phase4/tft/logs

# Activate conda environment
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

# Set working directory
cd /gpfs/home/dk5565/forecasting/phase4/tft

# Configuration
EPOCHS=100

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "Start time: $(date)"
echo "Task: Phase 4 TFT-multi Vital Sign Forecasting"
echo "Epochs: $EPOCHS"
echo "============================================"

# Run full pipeline
python -u run_pipeline.py --epochs $EPOCHS

echo "============================================"
echo "End time: $(date)"
echo "============================================"
