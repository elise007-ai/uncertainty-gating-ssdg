# When does uncertainty-aware pseudo-labeling help?

Code, training logs and measurement files for the paper
**"When Does Uncertainty-Aware Pseudo-Labeling Help? A Foundation-Free Study of Mixed-Domain Semi-Supervised Medical Image Segmentation"** (Kasereka Kiombwe Elise, Qingling Zhao; under review).

The paper compares, inside one foundation-free testbed (from-scratch UNet, Fourier augmentation, EMA mean teacher, SWAD), a plain confidence threshold with an evidential (Dirichlet) pseudo-label gate and with several other gates, and then measures the teacher's uncertainty signal directly. This repository contains what is needed to retrain the models, to repeat the measurements, and to check the tables against the files they were computed from; `docs/TABLE_MAP.md` also names the values that come from the submitted version and have no record here.

## Contents

| Folder | What it holds |
|---|---|
| `CODE/source/` | training scripts (`train_fourierfuse_v18_revision.py` trained all revision experiments, the later ones in the version under `v12/`; see Training), model and loss modules, data loaders, the uncertainty-measurement harness (`diag_uncertainty_harness_v18.py`), the job runner, and the analysis scripts that turn logs and measurements into tables and figures |
| `BSPC_results/RETURNS/<session>/runs/` | `log.txt`, `run_config.json` and `split.json` of the 149 training runs of the revision (45 runs of session S10 have no `split.json`; their log prints the labeled indices in its `[seed_split]` line); the logs of 14 runs of the original experiments; the incomplete logs of two first attempts of the MC-dropout gate, whose pseudo-label target differed from that of UA-MT (the `mcdropout2` runs replace them); and the logs of 32 smoke tests (short test runs of the pipeline). A run collected in more than one session has its complete log in the latest one |
| `BSPC_results/RETURNS/<session>/diag_out/` | one JSON per measured checkpoint (error-ranking AUROC, ranges, boundary bands, threshold sweep, oracle band, calibration) |
| `BSPC_results/RETURNS/<session>/jobs/` | the exact job lines that were run |
| `BSPC_results/derived/derived_v04_S12/` | every derived table as CSV, with the facts files the manuscript text was generated from |
| `RESULTS/` | per-seed records of the original experiments (see its README) |
| `REFERENCE/` | the published SynFoC tables as data, with the rule for reading them |
| `docs/TABLE_MAP.md` | which file and which script stand behind each table and figure |
| `docs/DEVIS_MEDL_IMPLEMENTATION.md` | how the DEviS and MEDL baselines were implemented, compared line by line with their papers |

Model checkpoints and the datasets are not included.

## Setup

```
pip install -r requirements.txt
```

Experiments were run with PyTorch 2 on one NVIDIA RTX 3090 (24 GB) per job.

## Data

The four benchmarks are public: Prostate (six sites), Fundus (four domains), M&Ms (four vendors) and BUSI. We use them in the pre-processed, pre-split form distributed with SynFoC (https://github.com/MQinghe/SynFoC, section "DataSets" of its README). The code expects them next to the repository:

```
../../data/ProstateSlice     ../../data/Fundus     ../../data/mnms     ../../data/Dataset_BUSI_with_GT
```

(paths are relative to `CODE/source/`; they are set near the top of the dataset section of each training script). Labeled images come from one domain per benchmark: RUNMC (`--lb_domain 5`, 20 images) on Prostate, Domain 3 (20) on Fundus, Vendor D (`--lb_domain 4`, 5) on M&Ms, malignant (`--lb_domain 2`, 64) on BUSI. Unlabeled images come from the training split of every domain; evaluation is on the test split of every domain.

## Training

Run from `CODE/source/`. The two gates of the main comparison on Prostate, seed 1337:

```
python train_fourierfuse_v18_revision.py --dataset prostate --lb_num 20 --lb_domain 5 --domain_num 6 --max_iterations 60000 --num_eval_iter 500 --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 --swad_start_iter 15000 --swad_update_every 100 --swad_bn_recalib_batches 20 --warmup --warmup_period 150 --AdamW --amp 1 --ff_base_features 64 --ff_spec_consistency_weight 0.5 --ff_fda_beta 0.05 --consistency 1.0 --consistency_rampup 150.0 --ema_decay 0.995 --daep_alpha 1.0 --daep_beta 1.0 --daep_gamma 1.0 --proto_per_domain 2 --proto_ema 0.99 --evid_lambda_kl_max 0.02 --evid_anneal_iters 10000 --save_model --overwrite --method daep --seed 1337 --save_name my_prostate_daep_s1337 --gpu 0
```

Replace `--method daep` by `--method fixmatch` for the confidence threshold. Every other run of the revision is one line of a job file under `BSPC_results/RETURNS/*/jobs/`; the first line of each `log.txt` repeats the command that produced it (the runs of the original experiments used `train_fourierfuse_v16.py`). A job file is run with

```
python job_runner.py --jobs <jobs file> --gpus 0 --per_gpu 1 --session <name>
```

A 60,000-iteration Prostate run takes about 11 to 14 hours on one RTX 3090.

`CODE/source/v12/train_fourierfuse_v18_revision.py` is the same script with one method added (`--method oracle_gate`) and runs every other method exactly as the file above does; it trained every run of the sessions S31, S41, S41b and S43. The job files of S31 and S43 use the oracle gate and need it: copy it over `CODE/source/train_fourierfuse_v18_revision.py` first. The `run_config.json` of every run of the revision records the sha256 of the script that trained it.

If a run is interrupted, start it again from iteration 0 (`--resume off`, or `--restart` with the job runner): continuing it from its saved state (`--resume auto`, the default) restarts its learning-rate schedule.

## Measuring a trained model

```
python diag_uncertainty_harness_v18.py --dataset prostate --ckpt <run folder>/best_student.pth --which ema --out diag_out/<name>.json --calibration 1 --lb_domain 5
```

The `meta.args` block of every JSON under `diag_out/` records the full argument list used for that file.

## From measurements to tables and figures

`docs/TABLE_MAP.md` lists the file behind each table. Two analysis steps run on this repository as it is and were tested on it:

```
python CODE/source/make_extra_s16.py --derived BSPC_results/derived/derived_v04_S12 --release . --out /tmp/check_s16
python CODE/source/make_figures_s19.py --derived BSPC_results/derived/derived_v04_S12 --out /tmp/figures --no_json
```

The first recomputes the headline measurement tables and the comparison with the published reference and stops with an error if any statement made in the paper about them no longer holds. The second redraws Figures 3 to 5 as vector PDF (`make_figures_s16.py` writes the same figures as PNG). The TikZ source of Figure 1 is `docs/figures/Figure1.tex`. The other analysis scripts are included as the record of how each table was derived; some of them also read console logs and transfer manifests that are too large or too specific to our server to be published here.

## What to expect when re-running

Training is not bit-reproducible on our hardware even with fixed seeds: three trainings of the same Prostate job line reached 86.30, 86.35 and 85.91 Dice (`extra_s18/table_replicates_s18.csv`). Re-running ten of the original configurations with this code gave values within 0.01 to 1.57 Dice points of the original runs, nine of them within 0.56 (`extra_s18/table_reproduction_s18.csv`, `extra_s25_selection/table_d30_codematched_s25.csv`). The training logs of the original Prostate and Fundus runs, and of the original threshold runs on M&Ms and BUSI, no longer exist; their per-seed values are kept in `RESULTS/`. The repository does not contain the logs of the term-removal, component-ablation and weight-sensitivity runs of the original submission either, and the Fundus digest keeps cup values only; `docs/TABLE_MAP.md` lists the values of the paper that rest on these runs.

## Licence and acknowledgement

Apache License 2.0 (see `LICENSE` and `NOTICE`). The data pipeline and the structure of the training loop derive from SynFoC (Ma et al., CVPR 2025), which builds on SSL4MIS and SAMed; we thank their authors for releasing their code. No foundation-model code or weights are used.

## Citation

See `CITATION.cff`. The paper is under review; this section will be updated with the reference. Every archived version of this repository is reachable through the concept DOI https://doi.org/10.5281/zenodo.23268781.
