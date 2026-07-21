#!/usr/bin/env python3
"""run_pipeline.py - End-to-end pipeline for Phase 4.1 TFT-multi waveform forecasting."""
import os, sys, argparse, subprocess

def run_step(script, args=[]):
    cmd = [sys.executable, script] + args
    print(f'\n{"="*60}\nRunning: {" ".join(cmd)}\n{"="*60}')
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f'FAILED: {script}')
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-patients', type=int, default=0)
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    
    if not args.skip_download:
        run_step('download_data.py', ['--num-patients', str(args.num_patients)])
    run_step('train.py', ['--epochs', str(args.epochs)])
    run_step('test.py', ['--epochs', str(args.epochs)])
    run_step('plot_predictions.py', ['--epochs', str(args.epochs)])
    print('\nPipeline complete!')

if __name__ == '__main__':
    main()
