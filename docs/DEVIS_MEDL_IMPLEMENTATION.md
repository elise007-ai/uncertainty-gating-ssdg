# DEviS and MEDL as implemented here, checked against the original papers

This is the authors' line-by-line comparison of the two re-implemented baselines with their papers. Identifiers such as E-38 or R-33 refer to the authors' revision log and can be ignored.

## 1. DEviS

| Paper | Code | Status |
|---|---|---|
| Eq. 2 E = softplus(Z); Eq. 6 α = E + 1 (r = W = 1); Eq. 8 b = e/S, u = C/S | `EvidentialHeadV18` defaults (softplus, W = 1, T = 1, no cap) | faithful |
| Eq. 15 L_ice = Σ_c y_c (ψ(S) − ψ(α_c)) | `evidential_bayes_risk(..., 'ce')`; `--evid_loss` forced to `ce` | faithful |
| Eq. 16 KL(Dir(α̃)‖Dir(1)), α̃ = y + (1−y)α; Eq. 17 λ = 0.02 **fixed** | `kl_dirichlet_to_prior`, α̃ as in `evidential_seg_loss_v18`; schedule forced `const`, `evid_lambda_kl_max = 0.02` (log line "[v18] DEviS keeps lambda_KL fixed at 0.02") | faithful |
| Eq. 18 soft Dice on b (belief); caption calls the same symbol "probability" | `soft_dice_from_alpha` on p = α/S | reading (b = (α−1)/S vs α/S; identical for large S) |
| Dice weight: Eq. 19/21 → weight 1; **Appendix B eq. S1 → (1 − β_t)** | `(1.0 − beta_t) * soft_dice` (Dice weight 0.99 → 0.90 at T/2 → 0 at T) | **reading — the paper contradicts itself; the authors' code (github Cocofeat/DEviS) could not be fetched this session, so it stays unresolved at code level** |
| Eq. 20 CUP = −β_t Σ_{correct} Σ_c b_c log(1−u) − (1−β_t) Σ_{incorrect} Σ_c (1−b_c) log u; β_t = β₀ e^{−(ln β₀) t/T}, β₀ = 0.01; Eq. 21 weight 1 | `devis_cup_loss`, `devis_beta_t(iter, max_iter, 0.01)`, `--devis_cup_weight 1.0` | faithful (β₀ and the CUP weight were listed as unverifiable in HANDOFF §5 — now verified) |
| CUP reduction: sums over pixels (relative scale to the per-pixel-mean terms unstated) | means over the correct set and over the incorrect set | reading — **tested, immaterial**: logged `devis_cup` is 0.09–0.22 at 500 it, 0.02–0.05 at 5–10 k, ≤ 0.01 after 20 k in all 12 runs, i.e. ≤ 10 % of the supervised loss after 2 k it |
| T, t in epochs | iterations (same fraction) | faithful |
| Semi-supervised extension (the journal version's gate) | `--devis_gate soft`: w = 1 − u | **[UNVERIFIED]** — the extension is not in any arXiv version; `devisobj_threshold` rows do not use it |

**Finding for O-03.** No implementation defect. Every term the paper specifies unambiguously is implemented as
written; the three readings above are documented choices on paper-internal ambiguities. The 1.4–1.9 Dice deficit
of `devisobj_threshold` (standard 0.80 gate, so the gate is not involved) is therefore attributable to **the DEviS
objective as published** — with the explicit caveat that under the main-text reading (Dice weight 1) a different
run would be "DEviS" too. Candidates inside the objective, *not* separable with the existing runs: digamma risk vs
MSE risk; KL at 0.02 from iteration 0 vs the testbed's linear anneal; Dice annealed to zero late in training; CUP.

**Finding for the two stalls** (`prostate_devis_s42` best 0.2542, `fundus_devis_s2024` best 0.6754). Both are
**soft-gated** runs; the threshold-gated run of the same seed/split/objective is healthy in both cases. Supervised
loss at 10 k / 20 k / 45 k (Prostate s42): soft 0.456 / 0.390 / 0.280 vs threshold 0.187 / 0.095 / 0.046; (Fundus
s2024, 5 k / 10 k / 20 k): soft 0.778 / 0.661 / 0.535 vs threshold 0.420 / 0.260 / 0.137. The two programs differ
only in the gate (w = 1 − u, mean 0.61–0.65 at 500 it vs 0.22–0.29 for the threshold), and the consistency weight is
≤ 0.01 for the first 10 % of training, so the gate perturbs the trajectory early and weakly; whether the stall is
"caused by the soft gate" or is a seed-dependent basin instability of the DEviS objective that the perturbation
exposes cannot be decided with n = 2 (both stalls landing in the soft group has chance 1/4 if the gate were
irrelevant). **Wording allowed:** "two of six soft-gated DEviS runs stalled in the supervised branch; none of the six
threshold-gated runs of the same objective did". **Not allowed:** "the soft gate causes the stall", "DEviS collapses".

**Dead hypothesis (do not retry):** "the CUP term, reduced as a mean over incorrect pixels, dominates the supervised
objective" — the logged CUP values above kill it.

## 2. MEDL

| Paper | Code | Status |
|---|---|---|
| Eq. 1 b = e/S, u = K/S, p = α/S, α = e + 1 | head defaults | faithful |
| Eq. 3 CAEF: b_n = b¹_n b²_n + |C_n|/(|C_n|+|C_K|)(b¹_n b²_K + b²_n b¹_K), b_K = b¹_K b²_K, |C_n| = 1, |C_K| = K, then normalise | `caef_fuse`: coef = 1/(1+K), u = u1·u2, all K+1 masses normalised | faithful (the paper also says C_K "has cardinality K−1" once; code uses K) |
| Eq. 4 R = exp(b_K Σ_n ζ_n log₂ ζ_n), ζ = fused belief | `medl_reliability` | faithful |
| "reliability … mask the generated pseudo-labels" (UPCoL-style); threshold not stated | `w_gate = (R > 0.5)` | **[UNVERIFIED]** threshold; with K = 2, R ≥ e^{−u} so R > 0.5 whenever u < 0.69 — the 100 % pass rate (E-37) follows from the formula + this threshold + the runs' vacuity, not from a bug |
| Eq. 5 u^final = λ₁ u_own + λ₂ u_fused, (0.5, 0.5) best (Table VII) | `medl_lambda1 = 0.5` | faithful |
| Eq. 6 ω = Ξ tanh(ψ ζ) + 1, ψ = 2h/V − 1 with h = **descending** uncertainty rank, ζ = 2q/Q − 1 | ψ from the **ascending** rank, ζ = 1 − 2·progress | **reading** — read literally, the paper's formula up-weights the most *uncertain* voxels at the start and the most certain at the end, the opposite of its stated intent ("initially focus on voxels with smaller uncertainties"); the code follows the intent. Ξ not given in the paper → Ξ = 1 **[UNVERIFIED]** (ω ∈ [0.24, 1.76]) |
| Eq. 7 I-EDL term Σ_n[(ŷ_n − α_n/S)² + α_n(S−α_n)/(S²(S+1))]ψ₁(α_n) − λ₂ log|I(α)| | `iedl_pixel_loss`, `dirichlet_log_fisher_det` (|I| = Π ψ₁(α_k)·(1 − ψ₁(S) Σ 1/ψ₁(α_k))) | faithful; λ₂ = 0.1 **[UNVERIFIED]** (not in the paper) |
| Eq. 8 L_{w,l} = Σ_v ω L_v / V; L^I_u = Σ_v L^I_v / V′ | weighted means Σ ω L / Σ ω and Σ w ω L^I / Σ w ω | reading — immaterial: ψ is uniform on [−1, 1] so mean ω ≈ 1 |
| Eq. 9 L = L_l + L_u + L_{w,l} + λ_GWU L^I_u; L_l, L_u = "Dice and Cross-Entropy" (form unspecified); λ_GWU Gaussian warm-up [SASSNet] | L_l = testbed evidential loss (mse risk + annealed KL + Dice); L_u = gated testbed loss vs the fused pseudo-label; L_{w,l} = ω-weighted risk + Dice (KL omitted); the **whole** unsupervised part (L_u + ω·L^I) multiplied by the testbed's consistency weight cw (sigmoid ramp, cap 0.5) | testbed (R-33) for L_l/L_u/the ramp; **reading**: the paper gives L_u weight 1 and ramps only L^I — ours ramps both |
| Pseudo-labels from the two sub-networks' fused belief (no EMA mentioned) | argmax of the fusion of the two **EMA** teachers | testbed (mean-teacher target) — disclose |
| Final output = **mean of both sub-networks** (Sec. III) | headline `--medl_report net1`; net2 and the ensemble logged every evaluation | **the paper's headline is the ensemble** → D-05 must be decided before the row is written; the ensemble values exist in every MEDL log |
| N₁ = VNet, N₂ = ResNet34-encoder VNet | net1 = testbed UNet, net2 = residual-block-encoder UNet | testbed analogue (disclose) |
| SGD, lr 0.01, 6 000 it, batch 2+2 | testbed optimiser/schedule | testbed |

**Finding for O-03/O-02 (objective vs implementation).** No defect in the MEDL-specific arithmetic (eqs. 3, 4, 5,
7); the ω sign convention, the reliability threshold, Ξ and λ₂ are documented readings/unverified constants; the
testbed's loss form, ramp and mean-teacher target are retained by R-33 and must be disclosed next to the row. The
evidence deflation observed in all MEDL runs (student mean Dirichlet strength 425 → 88–150 vs 3 600 for the
threshold run, ADJUDICATION_S10 O-02 iii) is what the −λ₂ log|I(α)| term does by construction (minimising it
maximises |I|, i.e. pushes α down) — a property of the published objective, not a bug.

