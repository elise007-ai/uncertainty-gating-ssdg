# Adaptive-Threshold Experiment — Results & Analysis

**Question.** Is the paper's "evidential gating does not beat a confidence
threshold" conclusion an artifact of comparing only against a *fixed* threshold?
We add a FreeMatch-style **adaptive** threshold as a third gate and compare
fixed vs. adaptive vs. evidential (DAEP) under one shared configuration.

## Setup
- Three gates: `fixmatch` (fixed τ=0.80), `adaptive` (FreeMatch-style per-class
  self-adaptive threshold), `daep` (evidential head).
- Two benchmarks: Fundus, Prostate. Three seeds each (1337, 42, 2024).
- **Single shared labeled-domain configuration** (Fundus `--lb_domain 3`,
  Prostate `--lb_domain 5`), identical across the three gates.

## Results — average Dice (%), 3-seed mean ± std

| Gate | Fundus | Prostate |
|---|---|---|
| Fixed threshold (τ=0.80) | 88.47 ± 0.23 | 85.71 ± 0.57 |
| Adaptive (FreeMatch-style) | 88.17 ± 0.36 | 83.60 ± 0.85 |
| Evidential (DAEP) | 88.54 ± 0.17 | 85.38 ± 0.58 |

Paired t-tests (df=2):
- Fundus: adaptive − fixed = −0.30 (t=−3.44, p=0.075); daep − fixed = +0.07 (p=0.63)
- Prostate: adaptive − fixed = −2.11 (p=0.12); adaptive − daep = −1.78 (p=0.05); daep − fixed = −0.33 (p=0.55)
- Fundus cup, cell-level (n=12): adaptive − fixed = −0.05 Dice (5/12 cells); daep − fixed = +0.03 (6/12)

## Conclusion (what went into the paper, Table 5 / §4.4)
**The adaptive threshold does not beat the fixed threshold** — indistinguishable
on Fundus, and **worse on Prostate**. No gate variant beats a standard fixed
threshold. This closes the "you only tried a dumb threshold" objection and
*reinforces* the paper's thesis: the limiting factor is the absence of a
per-pixel reliability signal with usable dynamic range, not the sophistication
of the threshold.

## IMPORTANT caveat (disclosed in the paper)
These runs use a **single fixed labeled-domain configuration**, which differs
from the **leave-one-domain-out averaging** used in the paper's main tables
(Table 2). Consequences, handled honestly in the manuscript:

1. The absolute Dice here is **not** directly comparable to Table 2 (both gates
   sit ~1–1.7 Dice higher than the leave-one-out averages). Only the
   **within-table gate differences** are interpreted.
2. The paper's headline **+1.19 Dice cup gain for DAEP** comes from the
   leave-one-domain-out protocol and is **not** reproduced or contradicted by
   this single-domain run (here DAEP's cup is +0.03 over fixed). The two are
   different configurations; we do not splice these numbers into the main tables
   and we do not present this run as evidence about the cup claim either way.

The manuscript states the configuration difference explicitly in both the §4.4
paragraph and the Table 5 caption, so a reviewer comparing tables will not be
misled.

## Files
- `fundus_logs.txt`, `prostate_logs.txt` — raw training-summary logs for all
  3 gates × 3 seeds.
