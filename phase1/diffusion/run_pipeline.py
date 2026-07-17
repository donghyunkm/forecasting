#!/usr/bin/env python3
"""
run_pipeline.py - Orchestrator for MIMIC-III waveform forecasting (Diffusion).

Runs the full pipeline:
    1. Download data (if not already present)
    2. Train 3 DDPM models (one per signal)
    3. Test all models with saved checkpoints (reverse diffusion)
    4. Plot predictions vs ground truth

Usage:
    python run_pipeline.py               # Default 20 epochs
    python run_pipeline.py --epochs 200  # Custom epoch count
"""

import os
import sys
import time
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def run_download():
    """Run data download step if data/ doesn't exist."""
    print("\n" + "=" * 60)
    print("STEP 1: Data Download")
    print("=" * 60)

    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        print("[INFO] Data directory already exists and is not empty.")
        print(f"       Path: {DATA_DIR}")
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


def run_training(num_epochs):
    """Train all 3 DDPM models."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Diffusion Model Training (3 signals, {num_epochs} epochs)")
    print("=" * 60)

    try:
        import model as model_module
        model_module.main(num_epochs=num_epochs)
        print("[SUCCESS] All DDPM models trained.")
        return True
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return False


def run_testing(num_epochs):
    """Test all 3 models with saved checkpoints."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing (reverse diffusion with best checkpoints)")
    print("=" * 60)

    try:
        import test as test_module
        test_module.run_test(num_epochs=num_epochs)
        print("[SUCCESS] Testing complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        return False


def run_plotting(num_epochs):
    """Generate prediction plots."""
    print("\n" + "=" * 60)
    print("STEP 4: Plotting")
    print("=" * 60)

    try:
        import plot_predictions
        plot_predictions.main(num_epochs=num_epochs)
        print("[SUCCESS] Plots generated.")
        return True
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")
        return False


def main():
    """Run the full diffusion forecasting pipeline."""
    parser = argparse.ArgumentParser(description='Run full diffusion forecasting pipeline')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    args = parser.parse_args()
    num_epochs = args.epochs

    print("=" * 60)
    print("MIMIC-III Waveform Forecasting — Diffusion (DDPM) Pipeline")
    print("=" * 60)
    print(f"Working directory: {BASE_DIR}")
    print(f"Signals: ABP, PLETH, II")
    print(f"Architecture: 3 separate conditional DDPMs")
    print(f"Epochs: {num_epochs}")

    start_time = time.time()

    # Step 1: Download
    if not run_download():
        print("\n[FATAL] Pipeline aborted at download step.")
        sys.exit(1)

    # Step 2: Train
    if not run_training(num_epochs):
        print("\n[FATAL] Pipeline aborted at training step.")
        sys.exit(1)

    # Step 3: Test
    if not run_testing(num_epochs):
        print("\n[FATAL] Pipeline aborted at testing step.")
        sys.exit(1)

    # Step 4: Plot
    if not run_plotting(num_epochs):
        print("\n[WARNING] Plotting failed but training/testing succeeded.")

    # Summary
    from model import get_checkpoint_dir, get_output_dir
    checkpoint_dir = get_checkpoint_dir(num_epochs)
    output_dir = get_output_dir(num_epochs)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Epochs:           {num_epochs}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Data:             {DATA_DIR}/")
    print(f"  Checkpoints:      {checkpoint_dir}/best_model_{{signal}}.pt")
    print(f"  Test predictions: {output_dir}/test_predictions_{{signal}}.npy")
    print(f"  Test targets:     {output_dir}/test_targets_{{signal}}.npy")
    print(f"  Metrics:          {output_dir}/test_metrics.json")
    print(f"  Plots:            {output_dir}/plot_*.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
