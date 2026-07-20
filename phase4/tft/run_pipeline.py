#!/usr/bin/env python3
"""
run_pipeline.py - End-to-end orchestrator for TFT-multi (Phase 4).

Steps:
    1. Check/extract vital signs from MIMIC-III CHARTEVENTS
    2. Train TFT-multi model
    3. Evaluate on test set
    4. Generate plots

Usage:
    python run_pipeline.py --epochs 100
"""

import os
import sys
import time
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/gpfs/scratch/dk5565/phase4_data'


def run_download():
    """Check vital signs data availability."""
    print("\n" + "=" * 60)
    print("STEP 1: Data Check")
    print("=" * 60)

    if os.path.exists(os.path.join(DATA_DIR, 'metadata.json')):
        print(f"[OK] Vital sign data available at {DATA_DIR}")
        return True

    print("[INFO] Data not found. Running extraction...")
    try:
        import download_data
        success = download_data.main()
        return success
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_training(num_epochs):
    """Train TFT-multi."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Training ({num_epochs} epochs)")
    print("=" * 60)

    try:
        from train import main as train_main
        train_main(num_epochs=num_epochs)
        return True
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_testing(num_epochs):
    """Evaluate model."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing")
    print("=" * 60)

    try:
        from test import run_test
        run_test(num_epochs=num_epochs)
        return True
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_plotting(num_epochs):
    """Generate plots."""
    print("\n" + "=" * 60)
    print("STEP 4: Plotting")
    print("=" * 60)

    try:
        from plot_predictions import main as plot_main
        plot_main(num_epochs=num_epochs)
        return True
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    from train import NUM_EPOCHS
    from preprocess import PAST_MONTHS, FUTURE_MONTHS, SIGNAL_NAMES

    parser = argparse.ArgumentParser(description='Run TFT-multi pipeline')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    num_epochs = args.epochs

    print("=" * 60)
    print("TFT-multi — Simultaneous Vital Sign Forecasting (Phase 4)")
    print("=" * 60)
    print(f"Vital signs: {SIGNAL_NAMES}")
    print(f"Input: {PAST_MONTHS} hours → Output: {FUTURE_MONTHS} hours")
    print(f"Epochs: {num_epochs}")

    start_time = time.time()

    if not run_download():
        print("\n[FATAL] Pipeline aborted: data not available.")
        sys.exit(1)

    if not run_training(num_epochs):
        print("\n[FATAL] Pipeline aborted: training failed.")
        sys.exit(1)

    if not run_testing(num_epochs):
        print("\n[FATAL] Pipeline aborted: testing failed.")
        sys.exit(1)

    if not run_plotting(num_epochs):
        print("\n[WARNING] Plotting failed but training/testing succeeded.")

    from train import get_checkpoint_dir, get_output_dir
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Epochs:     {num_epochs}")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Data:       {DATA_DIR}/")
    print(f"  Checkpoint: {get_checkpoint_dir(num_epochs)}/best_model.pt")
    print(f"  Outputs:    {get_output_dir(num_epochs)}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
