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
python prepare_data.py
