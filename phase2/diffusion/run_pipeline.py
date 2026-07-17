#!/usr/bin/env python3
"""
run_pipeline.py - Orchestrator for heart rate prediction pipeline (Diffusion).

Runs the full pipeline:
    1. Download data (if not already present)
    2. Train DDPM model for HR prediction
    3. Test model with saved checkpoint (reverse diffusion)
    4. Plot predictions vs ground truth

Usage:
    python run_pipeline.py
    python run_pipeline.py --epochs 100 --input-length 7500 --target-length 7500
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


def run_training(num_epochs, input_length, target_length):
    """Train DDPM model for HR prediction."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Diffusion Model Training ({num_epochs} epochs, in={input_length}, tgt={target_length})")
    print("=" * 60)

    try:
        import model as model_module
        model_module.main(num_epochs=num_epochs, input_length=input_length, target_length=target_length)
        print("[SUCCESS] DDPM model trained.")
        return True
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return False


def run_testing(num_epochs, input_length, target_length):
    """Test model with saved checkpoint."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing (reverse diffusion with best checkpoint)")
    print("=" * 60)

    try:
        import test as test_module
        test_module.run_test(num_epochs=num_epochs, input_length=input_length, target_length=target_length)
        print("[SUCCESS] Testing complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        return False


def run_plotting(num_epochs, input_length, target_length):
    """Generate prediction plots."""
    print("\n" + "=" * 60)
    print("STEP 4: Plotting")
    print("=" * 60)

    try:
        import plot_predictions
        plot_predictions.main(num_epochs=num_epochs, input_length=input_length, target_length=target_length)
        print("[SUCCESS] Plots generated.")
        return True
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")
        return False


def main():
    """Run the full diffusion HR prediction pipeline."""
    from preprocess import INPUT_LENGTH, TARGET_LENGTH
    from model import NUM_EPOCHS

    parser = argparse.ArgumentParser(description='Run full HR prediction pipeline (Diffusion)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--input-length', type=int, default=INPUT_LENGTH,
                        help=f'Input window length in samples (default: {INPUT_LENGTH})')
    parser.add_argument('--target-length', type=int, default=TARGET_LENGTH,
                        help=f'Target window length in samples (default: {TARGET_LENGTH})')
    args = parser.parse_args()

    num_epochs = args.epochs
    input_length = args.input_length
    target_length = args.target_length

    print("=" * 60)
    print("Heart Rate Prediction — Diffusion (DDPM) Pipeline")
    print("=" * 60)
    print(f"Working directory: {BASE_DIR}")
    print(f"Input: {input_length} samples ({input_length/125:.1f}s) of ABP, PLETH, II")
    print(f"Target: HR from next {target_length} samples ({target_length/125:.1f}s)")
    print(f"Epochs: {num_epochs}")

    start_time = time.time()

    if not run_download():
        print("\n[FATAL] Pipeline aborted at download step.")
        sys.exit(1)

    if not run_training(num_epochs, input_length, target_length):
        print("\n[FATAL] Pipeline aborted at training step.")
        sys.exit(1)

    if not run_testing(num_epochs, input_length, target_length):
        print("\n[FATAL] Pipeline aborted at testing step.")
        sys.exit(1)

    if not run_plotting(num_epochs, input_length, target_length):
        print("\n[WARNING] Plotting failed but training/testing succeeded.")

    from model import get_checkpoint_dir, get_output_dir
    checkpoint_dir = get_checkpoint_dir(num_epochs, input_length, target_length)
    output_dir = get_output_dir(num_epochs, input_length, target_length)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Epochs:           {num_epochs}")
    print(f"  Input length:     {input_length} samples ({input_length/125:.1f}s)")
    print(f"  Target length:    {target_length} samples ({target_length/125:.1f}s)")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Data:             {DATA_DIR}/")
    print(f"  Checkpoint:       {checkpoint_dir}/best_model_hr.pt")
    print(f"  Outputs:          {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
