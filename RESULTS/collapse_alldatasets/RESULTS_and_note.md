# Cross-dataset collapse measurement — VALIDATED (all four datasets)

## Final numbers (corrected script, single seed s1337, v16_*_s1337 checkpoints)
Dataset   AUROC(U_epi,err) mean   per-domain AUROC            central-80% range mean
Prostate  0.87                    0.88,0.79,0.90,0.98,0.83,0.85   9.2e-5
Fundus    0.86                    0.90,0.92,0.81,0.84             4.6e-4
M&Ms      0.92                    0.86,0.94,0.90,0.96             8.5e-5
BUSI      0.72                    0.69,0.74                       9.6e-4

=> On ALL FOUR: AUROC well above chance (0.72-0.92) AND range collapsed to 1e-4..1e-3.
   The "usable ranking + collapsed range" mechanism is now general, not Prostate-specific.

## History of the bug (resolved)
1. v1 run: sub-chance AUROC (Prostate 0.24). Misattributed to resolution alignment.
2. v2 run (alignment "fixed"): STILL sub-chance (Prostate 0.18) -> prediction falsified.
3. Real cause found: missing per-dataset LABEL REMAP (prostate_mask) -> for Prostate the
   raw label is the binary complement, inverting AUROC (0.84 -> ~0.16). MNMS unaffected
   (identity remap), which is why it alone looked fine.
4. v3 (corrected: remap added). Prediction: Prostate returns to ~0.8x. CONFIRMED at 0.87.
   Fundus NaNs disappeared (nan,0.01,0.40,nan -> 0.90,0.92,0.81,0.84). BUSI 0.72.

## Consistency checks
- Prostate single-seed 0.87 / 9.2e-5 vs Table 6 three-seed 0.84 / 9.9e-5: consistent.
- M&Ms 0.92 stable across runs (identity remap was always correct).

## Paper edits applied
- New Table (tab:collapse-cross): four-dataset AUROC + range summary (single seed).
- Cross-dataset paragraph rewritten from "M&Ms only" to "all four above chance,
  range collapsed", establishing generality of the mechanism.
- Table 6 (Prostate, three-seed per-vendor) unchanged.
