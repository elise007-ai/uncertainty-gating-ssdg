# v17 Alternative Gates — Results & Decision

## What we tested
To check whether *any* gate form beats the fixed confidence threshold (and to
test the "second-paper" consistency-gate idea), we ran three alternative gates
under the identical seed-split testbed:
- **adaptive** (FreeMatch-style self-adaptive threshold) — 3 seeds
- **percentile_gate** (keep top-confidence fraction per image; rank-based,
  scale-free) — seed 1337 only
- **consistency_gate** (trust pixels stable across FDA style views; scale-free,
  domain-aware) — seed 1337 only

## Results (Dice %, Fundus / Prostate)

| Gate | Fundus | Prostate |
|---|---|---|
| Fixed threshold (τ=0.80) | 88.47 ± 0.23 | 85.71 ± 0.57 |
| Adaptive (FreeMatch) | 88.17 ± 0.36 | 83.60 ± 0.85 |
| Percentile gate † | 86.97 | 85.63 |
| Consistency gate † | 87.40 | **67.95** |
| Evidential (DAEP) | 88.54 ± 0.17 | 85.38 ± 0.58 |

† single seed (1337), exploratory.

Consistency-gate Prostate per-domain (s1337): d1=40.1, d2=84.9, **d3=34.3**,
d4=85.3, d5=79.0, d6=84.1 → two vendors collapse; DSDV=0.048 (huge).

## Decision (lead author)
The consistency-gate hypothesis (my proposed "second paper" idea) **failed**: it
is worse than a plain threshold on Fundus and catastrophically unstable on
Prostate, where its agreement mask becomes too sparse on the most-shifted vendors
and training collapses on two domains. The percentile gate is also no better.

Rather than spin this into a separate paper, the failure is **most valuable
inside the current paper**: it converts the message from "this one uncertainty
signal (evidential) fails" to "multiple architecturally different gates —
evidential, adaptive, rank-based, and a purpose-built scale-free consistency gate
— all fail to beat a threshold." That makes the central claim far more robust: the
bottleneck is not the evidential parameterization specifically; a useful gate must
be both well-ranked AND stably dense, and none of the tested signals are both.

This is reported in Section 4.4 / Table 5 of the paper, with the single-seed
status of the percentile/consistency gates explicitly marked, and the Prostate
collapse disclosed honestly (not hidden as an averaged number).

## Honesty notes
- The percentile/consistency gates are single-seed; the paper marks them as
  exploratory (†) and does not report them as mean±std. If they are to appear as
  a firm result, run seeds 42 and 2024 (the run script supports this).
- The consistency-gate collapse is a real instability, not a tuning artifact we
  hid; it is stated plainly in the table caption and text.

## Files
- `v17_logs.txt` — raw best-Dice + per-domain for all v17 runs.
- Code: `CODE/source/train_fourierfuse_v17_consistency.py` (adds --method
  consistency_gate / percentile_gate); `run_v17_consistency_sidequest.sh`.
