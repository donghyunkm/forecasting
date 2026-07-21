#!/bin/bash
#SBATCH --job-name=phase43_itrans
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

cd /gpfs/home/dk5565/forecasting/phase43/iTransformer

for HORIZON in 1h 3h 6h; do
    echo "========================================"
    echo "Training iTransformer for horizon: $HORIZON"
    echo "========================================"
    python train.py --epochs 100 --horizon $HORIZON
    python test.py --epochs 100 --horizon $HORIZON
    python plot_predictions.py --epochs 100 --horizon $HORIZON
    python plot_full_forecast.py --epochs 100 --horizon $HORIZON
done
