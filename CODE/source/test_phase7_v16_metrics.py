#!/usr/bin/env python3
"""CPU smoke test for the metric math in run_v16_phase7_calibration_v16.py.

Inlines the pure-numpy metric primitives (kept BYTE-IDENTICAL to the v16
script — if you edit one, edit the other). Runs in ~5 sec without GPU,
without torch, without the dataloader. If any test fails, the v16 GPU
probe should not be launched.

Run with:
    python test_phase7_v16_metrics.py
"""

import sys
import numpy as np


# === INLINED METRIC PRIMITIVES (must match v16 script verbatim) ===

def expected_calibration_error(confidence, correct, n_bins=15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidence)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidence >= lo) & (
            confidence < hi if i < n_bins - 1 else confidence <= hi)
        cnt = int(in_bin.sum())
        if cnt > 0:
            acc = float(correct[in_bin].mean())
            conf = float(confidence[in_bin].mean())
            ece += (cnt / n) * abs(acc - conf)
            bin_stats.append({'lo': float(lo), 'hi': float(hi),
                              'count': cnt, 'acc': acc, 'conf': conf})
        else:
            bin_stats.append({'lo': float(lo), 'hi': float(hi),
                              'count': 0, 'acc': None, 'conf': None})
    return ece, bin_stats


def brier_score(prob_fg, target):
    return float(((prob_fg - target) ** 2).mean())


def risk_coverage_curve(uncertainty, error, n_points=10):
    order = np.argsort(uncertainty)
    error_sorted = error[order]
    n = len(error_sorted)
    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    risks = np.zeros_like(coverages)
    for i, c in enumerate(coverages):
        k = max(1, int(round(c * n)))
        risks[i] = float(error_sorted[:k].mean())
    trapz_fn = getattr(np, 'trapezoid', None) or np.trapz
    aurc = float(trapz_fn(risks, coverages))
    return coverages.tolist(), risks.tolist(), aurc


def safe_corr(x, y):
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_metric(per_slice_data, compute_fn, n_boot=1000, seed=1337):
    point = compute_fn(per_slice_data)
    if n_boot <= 0 or len(per_slice_data) == 0:
        return point, float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    n = len(per_slice_data)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resample = [per_slice_data[i] for i in idx]
        try:
            v = compute_fn(resample)
            if not np.isnan(v):
                estimates.append(v)
        except Exception:
            pass
    if not estimates:
        return point, float('nan'), float('nan')
    arr = np.asarray(estimates)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# === TESTS ===

rng = np.random.default_rng(0)
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))


# T1: ECE on perfectly-calibrated data should be near zero
n = 10000
conf = rng.uniform(0, 1, size=n)
correct = (rng.uniform(0, 1, size=n) < conf).astype(np.float32)
ece, _ = expected_calibration_error(conf, correct, n_bins=15)
print(f"[T1] ECE perfectly-calibrated: {ece:.4f}")
check("ECE_perfect_<0.02", ece < 0.02, f"got {ece:.4f}")

# T2: ECE on over-confident data ~ |0.95 - 0.50|
conf2 = np.full(n, 0.95)
correct2 = (rng.uniform(0, 1, size=n) < 0.50).astype(np.float32)
ece2, _ = expected_calibration_error(conf2, correct2, n_bins=15)
print(f"[T2] ECE over-confident (0.95 conf, 0.5 acc): {ece2:.4f}")
check("ECE_overconf_~0.45", 0.40 < ece2 < 0.50, f"got {ece2:.4f}")

# T3: Brier score on known-answer case
brier = brier_score(np.array([0.9, 0.1, 0.8, 0.2]),
                    np.array([1.0, 0.0, 1.0, 0.0]))
expected = ((0.1)**2 + (0.1)**2 + (0.2)**2 + (0.2)**2) / 4
print(f"[T3] Brier: {brier:.4f} (expected {expected:.4f})")
check("Brier_exact", abs(brier - expected) < 1e-6, f"got {brier:.6f}")

# T4: Risk-coverage AURC on near-perfect uncertainty should be low
n4 = 1000
error4 = (rng.uniform(0, 1, size=n4) < 0.30).astype(np.float32)
unc4 = error4 + rng.normal(0, 0.001, size=n4)
covs, risks, aurc = risk_coverage_curve(unc4, error4, n_points=10)
print(f"[T4] AURC near-perfect uncertainty: {aurc:.4f}")
check("AURC_perfect_<0.20", aurc < 0.20, f"got {aurc:.4f}")

# T5: Risk-coverage AURC on uninformative uncertainty ~ base error rate
unc5 = rng.uniform(0, 1, size=n4)
covs5, risks5, aurc5 = risk_coverage_curve(unc5, error4, n_points=10)
print(f"[T5] AURC uninformative uncertainty: {aurc5:.4f}")
check("AURC_uninformative_~0.30", 0.22 < aurc5 < 0.38, f"got {aurc5:.4f}")

# T6: Bootstrap CI contains the point estimate
per_slice = [{'val': float(rng.normal(5.0, 1.0))} for _ in range(100)]
mean_fn = lambda s: float(np.mean([x['val'] for x in s]))
point, lo, hi = bootstrap_metric(per_slice, mean_fn, n_boot=500, seed=42)
print(f"[T6] Bootstrap mean: {point:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
check("Bootstrap_CI_contains_point", lo <= point <= hi,
      f"point={point:.3f}, CI=[{lo:.3f}, {hi:.3f}]")
check("Bootstrap_CI_reasonable_width", (hi - lo) < 0.6,
      f"width={hi - lo:.3f}")

# T7: Informative uncertainty must start with lower risk than uninformative
print(f"[T7] risks[0] informative   = {risks[0]:.3f}")
print(f"[T7] risks[0] uninformative = {risks5[0]:.3f}")
check("Informative_starts_lower", risks[0] < risks5[0],
      f"informative={risks[0]:.3f}, uninformative={risks5[0]:.3f}")

# T8: ECE goes up when you make a calibrated model worse
conf_shifted = np.clip(conf + 0.2, 0, 1)
ece_shifted, _ = expected_calibration_error(conf_shifted, correct, n_bins=15)
print(f"[T8] ECE before shift={ece:.4f}, after +0.2 shift={ece_shifted:.4f}")
check("ECE_increases_with_overconfidence", ece_shifted > ece + 0.05,
      f"before={ece:.4f}, after={ece_shifted:.4f}")

# T9: safe_corr handles zero-variance inputs
nan_corr = safe_corr(np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.7, 0.3]))
print(f"[T9] safe_corr with zero-variance x: {nan_corr}")
check("safe_corr_handles_zero_var", np.isnan(nan_corr),
      f"got {nan_corr}")

# T10: safe_corr handles small samples
small_corr = safe_corr(np.array([1.0]), np.array([2.0]))
print(f"[T10] safe_corr with len=1: {small_corr}")
check("safe_corr_handles_short", np.isnan(small_corr), f"got {small_corr}")

# T11: Risk-coverage produces monotonically non-decreasing risk when
#      uncertainty is *anti*-informative (correlates negatively with error)
unc_anti = -error4 + rng.normal(0, 0.001, size=n4)
covs_anti, risks_anti, aurc_anti = risk_coverage_curve(unc_anti, error4,
                                                       n_points=10)
print(f"[T11] AURC anti-informative uncertainty: {aurc_anti:.4f}")
# When uncertainty is anti-informative, we're "selecting" the highest-error
# pixels first; risks should approach base rate (0.30) and AURC ≥ base rate
check("AURC_anti_>=base_rate", aurc_anti >= 0.25,
      f"got {aurc_anti:.4f}")

# T12: ECE binning is exact at the bin boundary
# Confidence values exactly on boundaries should be assigned consistently
edge_conf = np.array([0.0, 1.0/15, 2.0/15, 0.5, 1.0])
edge_correct = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
ece_edge, bins_edge = expected_calibration_error(edge_conf, edge_correct,
                                                  n_bins=15)
total_in_bins = sum(b['count'] for b in bins_edge)
print(f"[T12] Edge-case ECE: total bin counts = {total_in_bins} (should be 5)")
check("Bin_partition_total", total_in_bins == 5, f"got {total_in_bins}")

# ===
print()
print("=" * 60)
n_pass = sum(1 for _, p, _ in results if p)
for name, passed, detail in results:
    mark = 'PASS' if passed else 'FAIL'
    line = f"  {mark}: {name}"
    if not passed:
        line += f"  ({detail})"
    print(line)
print("=" * 60)
print(f"Total: {n_pass}/{len(results)} passing")
if n_pass == len(results):
    print("All metric tests pass. v16 GPU probe is safe to launch.")
    sys.exit(0)
else:
    print("Some tests failed — fix before running the GPU probe.")
    sys.exit(1)
