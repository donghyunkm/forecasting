#!/usr/bin/env python3
"""
run_pipeline.py - Orchestrator for MIMIC-III waveform forecasting pipeline.

Runs the full pipeline:
    1. Download data (if not already present)
    2. Train 3 LSTM models (one per signal)
    3. Test all models with saved checkpoints
    4. Plot predictions vs ground truth

Usage:
    python run_pipeline.py
"""

import os
import sys
import time


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


def run_training():
    """Train all 3 LSTM models."""
    print("\n" + "=" * 60)
    print("STEP 2: Model Training (3 signals)")
    print("=" * 60)

    try:
        import model as model_module
        model_module.main()
        print("[SUCCESS] All models trained.")
        return True
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return False


def run_testing():
    """Test all 3 models with saved checkpoints."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing (loading best checkpoints)")
    print("=" * 60)

    try:
        import test as test_module
        test_module.run_test()
        print("[SUCCESS] Testing complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        return False


def run_plotting():
    """Generate prediction plots."""
    print("\n" + "=" * 60)
    print("STEP 4: Plotting")
    print("=" * 60)

    try:
        import plot_predictions
        plot_predictions.main()
        print("[SUCCESS] Plots generated.")
        return True
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")
        return False


def main():
    """Run the full forecasting pipeline."""
    print("=" * 60)
    print("MIMIC-III Waveform Forecasting Pipeline (Multi-Signal)")
    print("=" * 60)
    print(f"Working directory: {BASE_DIR}")
    print(f"Signals: ABP, PLETH, II")
    print(f"Architecture: 3 separate LSTMs (input_size=3 each)")

    start_time = time.time()

    # Step 1: Download
    if not run_download():
        print("\n[FATAL] Pipeline aborted at download step.")
        sys.exit(1)

    # Step 2: Train
    if not run_training():
        print("\n[FATAL] Pipeline aborted at training step.")
        sys.exit(1)

    # Step 3: Test
    if not run_testing():
        print("\n[FATAL] Pipeline aborted at testing step.")
        sys.exit(1)

    # Step 4: Plot
    if not run_plotting():
        print("\n[WARNING] Plotting failed but training/testing succeeded.")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Data:             {DATA_DIR}/")
    print(f"  Checkpoints:      {os.path.join(BASE_DIR, 'checkpoints')}/best_model_{{signal}}.pt")
    print(f"  Test predictions: {os.path.join(BASE_DIR, 'outputs')}/test_predictions_{{signal}}.npy")
    print(f"  Test targets:     {os.path.join(BASE_DIR, 'outputs')}/test_targets_{{signal}}.npy")
    print(f"  Metrics:          {os.path.join(BASE_DIR, 'outputs')}/test_metrics.json")
    print(f"  Plots:            {os.path.join(BASE_DIR, 'outputs')}/plot_*.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
