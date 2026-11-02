# Where each table and figure of the paper comes from

All paths are relative to `BSPC_results/derived/derived_v04_S12/` unless they start with another folder. Tables are named by their LaTeX label.

| Paper item | File(s) | Produced by |
|---|---|---|
| Table `main` (testbed and published reference) | `extra_s17/table_main_gate_tests_s17.csv` (per-seed values, means, paired tests); `extra_s16/table_synfoc_like_for_like_s16.csv`; `REFERENCE/synfoc_published_tables.csv` | `make_extra_s17.py` (not included here: it also reads a search of our server's folders, which is not published), `make_extra_s16.py` |
| Table `ablation`, seed-split rows | same as above | same |
| Table `ablation`, term-removal rows; Table `components` and the weight sensitivity (Supplementary Section S7) | no record here: values of the submitted version, from fixed-split runs of the original submission whose training logs are not in this repository | their launchers: `CODE/source/run_v16_experiments.sh`, `CODE/source/run_v16_experiments_v2.sh` |
| Table `fundus-paired` | cup column: per-domain cup Dice of the original runs, `RESULTS/adaptive_threshold/fundus_logs.txt` (12 seed-paired cells); disc column: the per-domain disc values of the original runs are not in this repository (that digest keeps the cup only); the same analysis on the code-matched runs: `extra_s25_selection/table_d30_cup_s25.csv` | printed columns: none; code-matched runs: `make_selection_s25.py` |
| Table `adaptive` (alternative gates and the two controls) | adaptive and consistency gates: `RESULTS/adaptive_threshold/ANALYSIS.md`, `RESULTS/v17_alternative_gates/ANALYSIS.md`; percentile gate: `extra_s29/`; percentile gate at the oracle's removal rate: `extra_s41/`; no gate: `extra_s30/`; oracle gate: `extra_s31/`; threshold gate trained again: `table_phaseB_per_seed.csv`, `table_baselines.csv`; the fixed-threshold and evidential rows repeat Table `main` | `make_extra_s29.py`, `make_extra_s41_d34.py`, `make_extra_s30_d32.py`, `make_extra_s31_d33.py`, `make_derived_v04.py` |
| Table `baselines` (published alternatives and controls) | `table_baselines.csv`, `table_phaseB_per_seed.csv`, `table_nonfinite.csv`, `extra_s13/table_medl_heads.csv`, `extra_s14/table_mcdropout_signal_s14.csv`; the released form of the DEviS objective: `extra_s23/`, `extra_s24/` | `make_derived_v04.py`, `make_medl_heads_s13.py`, `make_extra_s14.py`, `make_extra_s23.py`, `make_extra_s24.py` |
| Table `evidsweep` (evidential configuration sweep) | `table_evid_sweep.csv`; the three trainings of the default: `extra_s18/table_replicates_s18.csv` | `make_derived_v04.py`, `make_extra_s18.py` |
| Tables `collapse`, `band-auroc` | `extra_s16/table_collapse_prostate_s16.csv` | `make_extra_s16.py` |
| Table `collapse-cross` | `extra_s16/table_collapse_cross_s16.csv` | `make_extra_s16.py` |
| Table `banddef` (band definitions, mm bands) | `extra_s15/table_band_definitions_s15.csv`, `table_mm_bands.csv`, `table_mm_bands_summary.csv` | `make_extra_s15.py`, `make_derived_v04.py` |
| Signal identity, calibration, ensembles | `table_signal_identity.csv`, `table_calibration.csv`, `extra_s14/table_ensemble_s14.csv` | `make_derived_v04.py`, `make_extra_s14.py` |
| Table `prostate-domains` | `extra_s15/table_prostate_per_vendor_seedsplit_s15.csv` | `make_extra_s15.py` |
| Table `spectral` and its robustness checks | `extra_s14/table_spectral_s14.csv`, `extra_s15/table_spectral_robustness_s15.csv`, `extra_s15/table_spectral_zconventions_s15.csv`, `table_spectral_loo.csv` | `spectral_distance_analysis.py`, `make_extra_s14.py`, `make_extra_s15.py` |
| Table `fda` (augmentation strength) | `table_fda_sweep.csv`, `extra_s14/table_fda_sweep_s14.csv` | `make_derived_v04.py`, `make_extra_s14.py` |
| Threshold sweep, oracle band (Figures 4, 5) | `extra_s16/table_threshold_sweep_s16.csv`, `extra_s16/table_oracle_band_s16.csv` | `make_extra_s16.py` |
| Figures 3, 4, 5 | drawn from the three `extra_s16` tables above | `make_figures_s19.py` (PDF) or `make_figures_s16.py` (PNG) |
| Figure 2 (qualitative maps) | needs the two checkpoints | `make_uncertainty_figure.py` |
| Run-to-run variation; reproduction of the original runs | `extra_s18/table_replicates_s18.csv`, `extra_s18/table_reproduction_s18.csv`, `extra_s25_selection/table_d30_codematched_s25.csv` | `make_extra_s18.py`, `make_selection_s25.py` |
| Consistency schedule: the single-seed control | `table_ramp_control.csv` | `make_derived_v04.py` |
| Table `shortramp` (Supplementary Table S4): the short ramp with three seeds | `extra_s43/table_d35_rule_s43.csv`, `extra_s43/table_d35_arms_s43.csv`, `extra_s43/facts_s43.json` | `make_extra_s43_d35.py` |
| Level and shape of the spectral distance (Supplementary Section S5, paragraph "Level and shape"; Section 4.7) | `extra_s22/table_spectral_level_s22.csv`, `extra_s22/facts_s22.json` | `make_extra_s22.py` |
| Published SynFoC tables: owner's sheet against `REFERENCE/synfoc_published_tables.csv` | `extra_s22/table_synfoc_sheet_check_s22.csv` | `make_extra_s22.py` |
| Checkpoint selection: the best-minus-last gaps of Section 4.2 and the comparisons at three checkpoints of Supplementary Section S7 | gaps over all held logs: key `selection` of `extra_s43/facts_s43.json`; comparisons: `extra_s24_selection/`, `extra_s25_selection/`, key `percentile_checkpoints` of `extra_s29/facts_s30_counts.json`, `extra_s30/table_d32_checkpoints_s30.csv`, `extra_s31/table_d33_checkpoints_s31.csv`, `extra_s41/table_d34_checkpoints_s41.csv`, `extra_s43/table_d35_checkpoints_s43.csv` | `make_extra_s43_d35.py`, `make_selection_s24.py`, `make_selection_s25.py`, `make_extra_s30.py`, `make_extra_s30_d32.py`, `make_extra_s31_d33.py`, `make_extra_s41_d34.py` |
| What three seeds can resolve (equivalence tests, minimum detectable effect) | `extra_s29/facts_s29_power.json` | `make_power_s29.py` |
| Vacuity-confidence identity and band ranking of every trained teacher (Section 4.5, Supplementary Section S9) | current totals: key `s9` of `extra_s43/facts_s43.json`; per teacher: every `table_signal_identity*.csv` and `table_band_summary*.csv` | `make_derived_v04.py` and the reader of each `extra_*` folder |
| Statements of the text that rest on several files | `extra_s43b/facts_s43b.json`, `extra_s44/facts_s44.json` | `make_extra_s43b.py`, `make_extra_s44.py` |

Every training run of the revision has its `log.txt` and `run_config.json` under `BSPC_results/RETURNS/<session>/runs/<dataset>/<run>/` and its `split.json` there or, for 45 runs of session S10, its labeled indices in the `[seed_split]` line of the log, and every measured checkpoint has one JSON under `BSPC_results/RETURNS/<session>/diag_out/` whose `meta` block records the checkpoint hash, the iteration and the exact harness arguments.
