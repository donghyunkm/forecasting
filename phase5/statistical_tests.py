#!/usr/bin/env python3
"""
statistical_tests.py - Compare Vitals+Waveform vs Vitals-Only model performance.

Runs paired t-test and Wilcoxon signed-rank test on per-sample MAE to determine
whether the difference between models is statistically significant.

Both models use the exact same test windows and targets, so a paired test is appropriate.

Usage:
    python statistical_tests.py
"""

import numpy as np
from scipy import stats

VITAL_NAMES = ['ABPMean', 'PULSE', 'SpO2', 'RESP']

# Paths to model outputs
MODELS = {
    'iTransformer': {
        'vw_pred': '/gpfs/home/dk5565/forecasting/phase5/iTransformer/outputs/itransformer_epochs_100/test_predictions.npy',
        'vw_tgt': '/gpfs/home/dk5565/forecasting/phase5/iTransformer/outputs/itransformer_epochs_100/test_targets.npy',
        'vo_pred': '/gpfs/home/dk5565/forecasting/phase5/ablation/iTransformer/outputs/itransformer_epochs_100/test_predictions.npy',
        'vo_tgt': '/gpfs/home/dk5565/forecasting/phase5/ablation/iTransformer/outputs/itransformer_epochs_100/test_targets.npy',
    },
    'TFT': {
        'vw_pred': '/gpfs/home/dk5565/forecasting/phase5/tft/outputs/tft_epochs_100/test_predictions.npy',
        'vw_tgt': '/gpfs/home/dk5565/forecasting/phase5/tft/outputs/tft_epochs_100/test_targets.npy',
        'vo_pred': '/gpfs/home/dk5565/forecasting/phase5/ablation/tft/outputs/tft_epochs_100/test_predictions.npy',
        'vo_tgt': '/gpfs/home/dk5565/forecasting/phase5/ablation/tft/outputs/tft_epochs_100/test_targets.npy',
    },
}


def run_tests(model_name, paths):
    """Run statistical tests for one model type."""
    print(f"\n{'='*70}")
    print(f"  {model_name}: Vitals+Waveform vs Vitals-Only")
    print(f"{'='*70}")

    # Load predictions (N, 24, 4, 3) and targets (N, 24, 4)
    vw_pred = np.load(paths['vw_pred'])
    vw_tgt = np.load(paths['vw_tgt'])
    vo_pred = np.load(paths['vo_pred'])
    vo_tgt = np.load(paths['vo_tgt'])

    print(f"  Samples: {len(vw_pred)}")
    print(f"  Same targets: {np.allclose(vw_tgt, vo_tgt)}")

    # Median predictions (quantile index 1 = 50th percentile)
    vw_median = vw_pred[:, :, :, 1]  # (N, 24, 4)
    vo_median = vo_pred[:, :, :, 1]  # (N, 24, 4)

    # --- Overall ---
    vw_sample_mae = np.abs(vw_median - vw_tgt).mean(axis=(1, 2))  # (N,)
    vo_sample_mae = np.abs(vo_median - vo_tgt).mean(axis=(1, 2))  # (N,)

    t_stat, p_ttest = stats.ttest_rel(vo_sample_mae, vw_sample_mae)
    w_stat, p_wilcoxon = stats.wilcoxon(vo_sample_mae, vw_sample_mae)

    delta = (vo_sample_mae - vw_sample_mae).mean()

    print(f"\n  Overall MAE:")
    print(f"    V+W:         {vw_sample_mae.mean():.4f}")
    print(f"    Vitals-Only: {vo_sample_mae.mean():.4f}")
    print(f"    Δ (VO - V+W): {delta:.4f} ({'correlations help' if delta > 0 else 'correlations hurt'})")
    print(f"    Paired t-test:      t={t_stat:.3f}, p={p_ttest:.6f} {'✓' if p_ttest < 0.05 else '✗'}")
    print(f"    Wilcoxon signed-rank: W={w_stat:.0f}, p={p_wilcoxon:.6f} {'✓' if p_wilcoxon < 0.05 else '✗'}")

    # --- Per-vital ---
    print(f"\n  Per-Vital (paired t-test, two-sided):")
    print(f"  {'Vital':<10} {'V+W MAE':<10} {'VO MAE':<10} {'Δ':<10} {'t-stat':<10} {'p-value':<12} {'Sig?'}")
    print(f"  {'-'*70}")

    for v in range(4):
        vw_v = np.abs(vw_median[:, :, v] - vw_tgt[:, :, v]).mean(axis=1)  # (N,)
        vo_v = np.abs(vo_median[:, :, v] - vo_tgt[:, :, v]).mean(axis=1)  # (N,)
        t, p = stats.ttest_rel(vo_v, vw_v)
        d = (vo_v - vw_v).mean()
        sig = '✓' if p < 0.05 else '✗'
        print(f"  {VITAL_NAMES[v]:<10} {vw_v.mean():<10.4f} {vo_v.mean():<10.4f} {d:<+10.4f} {t:<10.3f} {p:<12.6f} {sig}")

    # Effect size (Cohen's d)
    diff = vo_sample_mae - vw_sample_mae
    cohens_d = diff.mean() / diff.std()
    print(f"\n  Effect size (Cohen's d): {cohens_d:.4f} ({'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium'})")


def main():
    print("=" * 70)
    print("  Statistical Significance: V+W vs Vitals-Only")
    print("  Test: Two-sided paired t-test + Wilcoxon signed-rank")
    print("  H0: No difference in per-sample MAE between models")
    print("  Samples are paired (same windows, same targets)")
    print("=" * 70)

    for model_name, paths in MODELS.items():
        run_tests(model_name, paths)

    print(f"\n{'='*70}")
    print("  Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()
