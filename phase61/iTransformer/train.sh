#!/bin/bash
#SBATCH --job-name=phase61_itrans
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_train_%j.out
#SBATCH --error=logs/slurm_train_%j.err

mkdir -p logs
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate CSDI

cd /gpfs/home/dk5565/forecasting/phase61/iTransformer
python train.py --epochs 100
python test.py --epochs 100
