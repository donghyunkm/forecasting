#!/bin/bash
#SBATCH --job-name=phase62v2_data
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_data_v2_%j.out
#SBATCH --error=logs/slurm_data_v2_%j.err

mkdir -p logs
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

cd /gpfs/home/dk5565/forecasting/phase62
python prepare_data_v2.py
