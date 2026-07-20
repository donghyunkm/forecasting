#!/bin/bash
#SBATCH --job-name=lstm_forecast_p3
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home/dk5565/forecasting/phase3/lstm/logs/train_%j.out
#SBATCH --error=/gpfs/home/dk5565/forecasting/phase3/lstm/logs/train_%j.err

# Create logs directory
mkdir -p /gpfs/home/dk5565/forecasting/phase3/lstm/logs

# Activate conda environment
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

# Set working directory
cd /gpfs/home/dk5565/forecasting/phase3/lstm

# Configuration
EPOCHS=100
TARGET=${1:-II}  # Default target signal, can override: sbatch main.sh PLETH

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "Start time: $(date)"
echo "Task: Phase 3 Multivariate Waveform Forecasting (LSTM)"
echo "Target signal: $TARGET"
echo "Epochs: $EPOCHS"
echo "============================================"

# Download/load data (skips if data/ already exists)
python -u download_data.py --num-patients 100

# Train model for the target signal
python -u model.py --target $TARGET --epochs $EPOCHS

# Test with best checkpoint
python -u test.py --target $TARGET --epochs $EPOCHS

# Generate plots
python -u plot_predictions.py --target $TARGET --epochs $EPOCHS

echo "============================================"
echo "End time: $(date)"
echo "============================================"
