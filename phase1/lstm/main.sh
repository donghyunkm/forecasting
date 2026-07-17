#!/bin/bash
#SBATCH --job-name=lstm_forecast
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home/dk5565/forecasting/lstm/logs/train_%j.out
#SBATCH --error=/gpfs/home/dk5565/forecasting/lstm/logs/train_%j.err

# Create logs directory
mkdir -p /gpfs/home/dk5565/forecasting/lstm/logs

# Activate conda environment
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

# Set working directory
cd /gpfs/home/dk5565/forecasting/lstm

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "Start time: $(date)"
echo "============================================"

# Train LSTM models for 20 epochs
python -u model.py --epochs 20

# Test with best checkpoints
python -u test.py --epochs 20

# Generate plots
python -u plot_predictions.py --epochs 20

echo "============================================"
echo "End time: $(date)"
echo "============================================"
