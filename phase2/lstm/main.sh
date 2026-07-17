#!/bin/bash
#SBATCH --job-name=lstm_hr_predict
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home/dk5565/forecasting/phase2/lstm/logs/train_%j.out
#SBATCH --error=/gpfs/home/dk5565/forecasting/phase2/lstm/logs/train_%j.err

# Create logs directory
mkdir -p /gpfs/home/dk5565/forecasting/phase2/lstm/logs

# Activate conda environment
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

# Set working directory
cd /gpfs/home/dk5565/forecasting/phase2/lstm

# Configuration
EPOCHS=20
INPUT_LENGTH=37500
TARGET_LENGTH=7500

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "Start time: $(date)"
echo "Task: Heart Rate Prediction (LSTM)"
echo "Input length: $INPUT_LENGTH samples"
echo "Target length: $TARGET_LENGTH samples"
echo "Epochs: $EPOCHS"
echo "============================================"

# Download/load data (skips if data/ already exists)
python -u download_data.py --num-patients 5

# Train LSTM model
python -u model.py --epochs $EPOCHS --input-length $INPUT_LENGTH --target-length $TARGET_LENGTH

# Test with best checkpoint
python -u test.py --epochs $EPOCHS --input-length $INPUT_LENGTH --target-length $TARGET_LENGTH

# Generate plots
python -u plot_predictions.py --epochs $EPOCHS --input-length $INPUT_LENGTH --target-length $TARGET_LENGTH

echo "============================================"
echo "End time: $(date)"
echo "============================================"
