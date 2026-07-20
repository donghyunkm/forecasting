#!/usr/bin/env python3
"""
run_pipeline.py - Orchestrator for multivariate waveform forecasting (LSTM, Phase 3).

Runs the full pipeline for a given target signal:
    1. Download data (if not already present)
    2. Train LSTM model for forecasting
    3. Test model with saved checkpoint
    4. Plot predictions vs ground truth

Usage:
    python run_pipeline.py --target II
    python run_pipeline.py --target PLETH --epochs 50
    python run_pipeline.py --target ABP
"""

import os
import sys
import time
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/gpfs/scratch/dk5565/phase3_data'


def run_download():
    """Run data download step if data/ doesn't exist."""
    print("\n" + "=" * 60)
    print("STEP 1: Data Download")
    print("=" * 60)

    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        print(f"[INFO] Data already exists at {DATA_DIR}")
        print("[INFO] Skipping download. Delete data/ to re-download.")
        return True

    print("[INFO] Data directory not found. Starting download...")
    try:
        import download_data
        success = download_data.main()
        if not success:
            print("[ERROR] Data download failed.")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Download step failed: {e}")
        return False


def run_training(target_signal, num_epochs):
    """Train LSTM model for a target signal."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Training ({target_signal}, {num_epochs} epochs)")
    print("=" * 60)

    try:
        import model as model_module
        model_module.main(target_signal=target_signal, num_epochs=num_epochs)
        print("[SUCCESS] Model trained.")
        return True
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_testing(target_signal, num_epochs):
    """Test model with saved checkpoint."""
    print("\n" + "=" * 60)
    print(f"STEP 3: Testing ({target_signal})")
    print("=" * 60)

    try:
        import test as test_module
        test_module.run_test(target_signal=target_signal, num_epochs=num_epochs)
        print("[SUCCESS] Testing complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_plotting(target_signal, num_epochs):
    """Generate prediction plots."""
    print("\n" + "=" * 60)
    print(f"STEP 4: Plotting ({target_signal})")
    print("=" * 60)

    try:
        import plot_predictions
        plot_predictions.main(target_signal=target_signal, num_epochs=num_epochs)
        print("[SUCCESS] Plots generated.")
        return True
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the full forecasting pipeline."""
    from preprocess import VALID_TARGETS, INPUT_LENGTH, OUTPUT_LENGTH, INTERVAL_MINUTES
    from model import NUM_EPOCHS

    parser = argparse.ArgumentParser(description='Run full forecasting pipeline (LSTM)')
    parser.add_argument('--target', type=str, default='II', choices=VALID_TARGETS,
                        help=f'Target signal to forecast (default: II)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    args = parser.parse_args()

    target_signal = args.target
    num_epochs = args.epochs

    print("=" * 60)
    print(f"Phase 3 — Multivariate Waveform Forecasting Pipeline (LSTM)")
    print("=" * 60)
    print(f"Working directory: {BASE_DIR}")
    print(f"Target signal: {target_signal}")
    print(f"Input: {INPUT_LENGTH} intervals ({INPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hrs)")
    print(f"Output: {OUTPUT_LENGTH} intervals ({OUTPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hrs)")
    print(f"Epochs: {num_epochs}")

    start_time = time.time()

    if not run_download():
        print("\n[FATAL] Pipeline aborted at download step.")
        sys.exit(1)

    if not run_training(target_signal, num_epochs):
        print("\n[FATAL] Pipeline aborted at training step.")
        sys.exit(1)

    if not run_testing(target_signal, num_epochs):
        print("\n[FATAL] Pipeline aborted at testing step.")
        sys.exit(1)

    if not run_plotting(target_signal, num_epochs):
        print("\n[WARNING] Plotting failed but training/testing succeeded.")

    from model import get_checkpoint_dir, get_output_dir
    checkpoint_dir = get_checkpoint_dir(target_signal, num_epochs)
    output_dir = get_output_dir(target_signal, num_epochs)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Target signal: {target_signal}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Total time:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Data:          {DATA_DIR}/")
    print(f"  Checkpoint:    {checkpoint_dir}/best_model.pt")
    print(f"  Outputs:       {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
