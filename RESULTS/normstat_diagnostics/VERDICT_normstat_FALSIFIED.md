# Normalization-Statistics Mismatch — Stage-0 Kill-Switch: FALSIFIED

**Status: hypothesis tested and rejected. No further work on this mechanism.**

## What was tested

The proposed next-project mechanism (`NEXT_PROJECT/PROPOSAL.md`) hypothesized
that a per-region **BatchNorm-statistics mismatch** signal — how far a target
image's local feature statistics at each BN layer deviate from the stored source
running statistics — would predict pseudo-label error *better than confidence*,
especially on the most-shifted vendor (BIDMC), and could therefore replace the
collapsed evidential gate.

Pre-registered pass criterion (P1): on the shifted vendor BIDMC (domain 1),
`AUROC(mismatch, error)` must exceed `AUROC(confidence, error)`.

The kill-switch (`CODE/source/normstat_diagnostics.py`,
`run_normstat_diagnostics.sh`) hooks the BN layers of the existing converged
`best_student.pth` checkpoints and computes, per pixel/region, the KL mismatch
between local target stats and stored source stats, then compares its
error-ranking AUROC against confidence and vacuity. No training required.

## Result (3 seeds, 6 Prostate vendors)

Raw JSONs: `RESULTS/normstat_diagnostics/normstat_prostate_s{1337,42,2024}.json`.

| Seed  | mean AUROC(mismatch) | mean AUROC(low-conf) | P1 |
|-------|----------------------|----------------------|----|
| 1337  | 0.451                | 0.829                | FAIL |
| 42    | 0.356                | 0.840                | FAIL |
| 2024  | 0.512                | 0.875                | FAIL |
| **mean** | **0.440**         | **0.848**            | **FAIL** |

**BIDMC (domain 1), the vendor the hypothesis specifically predicted would win:**

| Seed  | AUROC(mismatch) | AUROC(low-conf) |
|-------|-----------------|-----------------|
| 1337  | 0.414           | 0.841           |
| 42    | 0.338           | 0.831           |
| 2024  | 0.445           | 0.861           |

## Verdict

P1 failed decisively on every seed and every vendor. Two facts settle it:

1. **Mismatch is far worse than confidence**, not better — confidence ranks
   error at ~0.85 AUROC; mismatch sits at ~0.44.
2. **Mismatch is at or below chance (0.5) on BIDMC**, the exact vendor where it
   was predicted to excel. A signal reliably below 0.5 is *anti-correlated* with
   error: high BN-statistics mismatch tends to occur where the model is
   **correct** (high-contrast anatomical boundaries — the easy regions), while
   errors concentrate in low-signal ambiguous areas where mismatch is small. The
   hypothesis had the direction backwards.

## What this changes

- The normalization-mismatch mechanism is **dropped**. Like CADG before it, it
  was killed by a cheap forward-pass diagnostic before any training was spent.
- The two falsified mechanisms (CADG, normstat) jointly reinforce the paper's
  conclusion: the SSDG bottleneck is **not** a missing per-pixel error signal —
  confidence already ranks error at ~0.85 AUROC. Neither a recalibrated
  evidential signal nor an architectural mismatch signal beats it.
- A methodological caveat surfaced during this run and is recorded for any future
  work: these diagnostics measure signals against **all-pixel** error, where the
  converged model is ~99% accurate everywhere, so they do not probe the
  **boundary/Dice** regime that actually determines SSDG performance. Any future
  mechanism work should re-target the diagnostic at boundary-band error before
  drawing conclusions.

## Disposition

The decision after this result was **not** to chase a third mechanism but to
consolidate and ship the strengthened Option-A paper (now the TMI submission in
`PAPER/`). This document and the JSONs are retained as the record of the
falsification.
