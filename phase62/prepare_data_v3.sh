#!/bin/bash
#SBATCH --job-name=phase62v3_data
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_data_v3_%j.out
#SBATCH --error=logs/slurm_data_v3_%j.err

mkdir -p logs
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

cd /gpfs/home/dk5565/forecasting/phase62
python prepare_data_v3.py
