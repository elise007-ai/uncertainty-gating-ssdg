# Where each table and figure of the paper comes from

All paths are relative to `BSPC_results/derived/derived_v04_S12/` unless they start with another folder. Tables are named by their LaTeX label.

| Paper item | File(s) | Produced by |
|---|---|---|
| Table `main` (testbed and published reference) | `extra_s17/table_main_gate_tests_s17.csv` (per-seed values, means, paired tests); `extra_s16/table_synfoc_like_for_like_s16.csv`; `REFERENCE/synfoc_published_tables.csv` | `make_extra_s17.py`, `make_extra_s16.py` |
| Table `ablation`, seed-split rows | same as above | same |
| Table `ablation`, term-removal rows; component ablation; alternative gates | `RESULTS/` (fixed-split experiments of the original submission) | original scripts under `CODE/source/run_*.sh` |
| Table `baselines` (published alternatives and controls) | `table_baselines.csv`, `table_phaseB_per_seed.csv`, `table_nonfinite.csv`, `extra_s13/table_medl_heads.csv`, `extra_s14/table_mcdropout_signal_s14.csv` | `make_derived_v04.py`, `make_medl_heads_s13.py`, `make_extra_s14.py` |
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
| Run-to-run variation; reproduction of the original runs | `extra_s18/table_replicates_s18.csv`, `extra_s18/table_reproduction_s18.csv` | `make_extra_s18.py` |
| Ramp-schedule control | `table_ramp_control.csv` | `make_derived_v04.py` |

Every training run of the revision has its `log.txt`, `run_config.json` and `split.json` under `BSPC_results/RETURNS/<session>/runs/<dataset>/<run>/`, and every measured checkpoint has one JSON under `BSPC_results/RETURNS/<session>/diag_out/` whose `meta` block records the checkpoint hash, the iteration and the exact harness arguments.
