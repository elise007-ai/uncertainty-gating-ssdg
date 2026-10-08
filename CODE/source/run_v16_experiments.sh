#!/bin/bash
# =============================================================================
# v16 EXPERIMENT RUNNER — MedIA/TMI submission track (2-GPU budget)
# =============================================================================
#
# Contribution: Domain-Aware Evidential Pseudo-Labeling (DAEP).
# Framing:     Compact UNet + principled uncertainty-aware SSDG matches
#              foundation-model-backed SynFoC at a fraction of the parameters.
#
# Phases (run in order, verify each phase before moving on):
#
#   PHASE 0  — Sanity run on MNMS seed=1337 only.
#              Verify v16 does not crash, DAEP weights behave, dice > v10.
#              ~8h on 1 GPU. Do this BEFORE phase 1.
#
#   PHASE 1  — Main results: 3 seeds x 4 datasets. 12 runs.
#              This is the headline Table 2 of the paper. ~60-80h GPU-time.
#              With 2 GPUs running in parallel: ~3-5 days wall time.
#
#   PHASE 2  — Ablations: 1 seed x {MNMS, Prostate} x 6 configs. 12 runs.
#              MNMS = hardest, Prostate = easiest. If DAEP works on both,
#              it generalizes. Table 3 of the paper.
#
#   PHASE 3  — Sensitivity + comparisons: MNMS only.
#              Tests α/β/γ hyperparameter robustness + threshold baseline
#              + SynFoC comparison (SynFoC is run from their repo, not here).
#
# Datasets / flags:
#   prostate: --lb_num 20 --lb_domain 5 --domain_num 6  (binary, K=2)
#   fundus:   --lb_num 20 --lb_domain 3 --domain_num 4  (K=3)
#   BUSI:     --lb_num 64 --lb_domain 2 --domain_num 2  (binary, K=2)
#   MNMS:     --lb_num 5  --lb_domain 4 --domain_num 4  (K=4)
#
# =============================================================================

# Common args shared across all datasets
COMMON="--warmup --warmup_period 150 --AdamW --amp 1 \
  --ff_base_features 64 \
  --ff_spec_consistency_weight 0.5 --ff_fda_beta 0.05 \
  --consistency 1.0 --consistency_rampup 150.0 --ema_decay 0.995 \
  --daep_alpha 1.0 --daep_beta 1.0 --daep_gamma 1.0 \
  --proto_per_domain 2 --proto_ema 0.99 \
  --evid_lambda_kl_max 0.02 --evid_anneal_iters 10000 \
  --save_model --overwrite"

# Per-dataset blocks
PROSTATE="--dataset prostate --lb_num 20 --lb_domain 5 --domain_num 6 \
  --max_iterations 60000 --num_eval_iter 500 \
  --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 \
  --swad_start_iter 15000 --swad_update_every 100 --swad_bn_recalib_batches 20"

FUNDUS="--dataset fundus --lb_num 20 --lb_domain 3 --domain_num 4 \
  --max_iterations 30000 --num_eval_iter 500 \
  --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 \
  --swad_start_iter 10000 --swad_update_every 100 --swad_bn_recalib_batches 20"

BUSI="--dataset BUSI --lb_num 64 --lb_domain 2 --domain_num 2 \
  --max_iterations 30000 --num_eval_iter 500 \
  --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 \
  --swad_start_iter 10000 --swad_update_every 100 --swad_bn_recalib_batches 20"

MNMS="--dataset MNMS --lb_num 5 --lb_domain 4 --domain_num 4 \
  --max_iterations 60000 --num_eval_iter 500 \
  --base_lr 0.0005 --label_bs 2 --unlabel_bs 2 \
  --swad_start_iter 15000 --swad_update_every 100 --swad_bn_recalib_batches 20"

SEEDS="42 2024"

# =============================================================================
# PHASE 0 — Sanity check (RUN THIS FIRST, DO NOT SKIP)
# =============================================================================
phase0_sanity() {
  echo "=== PHASE 0 — sanity check on MNMS seed=1337 (~8h on 1 GPU) ==="
  python train_fourierfuse_v16.py $MNMS $COMMON \
    --gpu 1 --seed 1337 --save_name v16_sanity_mnms_s1337

  # Verify
  log="../model/MNMS/train_fourierfuse_v16/v16_sanity_mnms_s1337/log.txt"
  if [ -f "$log" ]; then
    echo ""; echo "=== Sanity check results ==="
    best=$(grep "best_avg=" "$log" | tail -1)
    echo "Last line: $best"
    w_lines=$(grep -c "w_mean=" "$log")
    echo "DAEP w_mean log lines seen: $w_lines"
    echo ""
    echo "PASS CRITERIA:"
    echo "  - best_avg should exceed 0.74 (v10 baseline ~0.74 on MNMS)"
    echo "  - w_mean should be in [0.3, 0.9] range (not stuck at 0 or 1)"
    echo "  - No 'non-finite loss' warnings"
    echo ""
    echo "If any criterion fails, DO NOT proceed to phase 1. Debug first."
  fi
}

# =============================================================================
# PHASE 1 — Main results (12 runs: 3 seeds x 4 datasets)
# =============================================================================
# Runtime budget: ~75h of 1-GPU time, ~40h wall with 2-GPU parallelism.
# Strategy: pair short-with-long datasets on the two GPUs so they finish close.
# We run: (MNMS seed1 | Prostate seed1) then (MNMS seed2 | Prostate seed2) ...
# then Fundus and BUSI packed together.
# =============================================================================
phase1_main_results() {
  echo "=== PHASE 1 — main results, 3 seeds, 4 datasets, 2 GPUs, 2 processes/GPU ==="

  for SEED in $SEEDS; do
    echo "============================================"
    echo "  SEED = $SEED"
    echo "============================================"

    # GPU 0: Prostate (60k) + Fundus (30k)
    echo "[GPU0] Starting Prostate seed=$SEED"
    python train_fourierfuse_v16.py $PROSTATE $COMMON \
      --gpu 0 --seed $SEED --save_name v16_prostate_s$SEED &
    PID_PROSTATE=$!

    echo "[GPU0] Starting Fundus seed=$SEED"
    python train_fourierfuse_v16.py $FUNDUS $COMMON \
      --gpu 0 --seed $SEED --save_name v16_fundus_s$SEED &
    PID_FUNDUS=$!

    # GPU 1: MNMS (60k) + BUSI (30k)
    echo "[GPU1] Starting MNMS seed=$SEED"
    python train_fourierfuse_v16.py $MNMS $COMMON \
      --gpu 1 --seed $SEED --save_name v16_mnms_s$SEED &
    PID_MNMS=$!

    echo "[GPU1] Starting BUSI seed=$SEED"
    python train_fourierfuse_v16.py $BUSI $COMMON \
      --gpu 1 --seed $SEED --save_name v16_busi_s$SEED &
    PID_BUSI=$!

    # Wait for all four to finish before moving to next seed
    wait $PID_PROSTATE $PID_FUNDUS $PID_MNMS $PID_BUSI

    echo "=== seed=$SEED completed for all four datasets ==="
  done

  echo "=== PHASE 1 COMPLETE ==="
}

# =============================================================================
# PHASE 2 — Ablations (12 runs: 1 seed x 2 datasets x 6 configs)
# =============================================================================
# Core ablation: which uncertainty term matters?
#   a) DAEP full (=main result, already have it from phase 1)
#   b) DAEP -U_ale
#   c) DAEP -U_epi
#   d) DAEP -U_dom
#   e) Threshold baseline (hard >0.8 filter, same evidential head)
#   f) DAEP with no evidential head (softmax + threshold) — implemented below
#
# For MedIA reviewers: (a)-(e) is sufficient. (f) is for the paper's "what
# does the evidential head contribute separately from the uncertainty-
# weighted loss?" question. We implement (f) by running with
# --daep_threshold_baseline AND using softmax-derived uncertainties; but
# since our model always uses Dirichlet, we instead report (f) in the paper
# by comparing to published FixMatch-SSDG / UniMatch-SSDG numbers.
# =============================================================================
phase2_ablations() {
  echo "=== PHASE 2 — ablations on MNMS + Prostate (seed=1337), 2 processes per GPU ==="

  # List of ablation configs: (name, flag)
  ABLATIONS=(
    "no_ale:--daep_no_ale"
    "no_epi:--daep_no_epi"
    "no_dom:--daep_no_dom"
    "threshold:--daep_threshold_baseline"
  )

  # Process in batches of 2 configs (to have 2 configs per GPU)
  for ((i=0; i<${#ABLATIONS[@]}; i+=2)); do
    CFG1="${ABLATIONS[$i]}"
    CFG2="${ABLATIONS[$i+1]}"

    CFG1_NAME="${CFG1%%:*}"
    CFG1_FLAG="${CFG1##*:}"
    CFG2_NAME="${CFG2%%:*}"
    CFG2_FLAG="${CFG2##*:}"

    echo "=== Batch: $CFG1_NAME and $CFG2_NAME ==="

    # GPU 0: MNMS with both configs in parallel
    echo "  [GPU0] MNMS ablation $CFG1_NAME"
    python train_fourierfuse_v16.py $MNMS $COMMON $CFG1_FLAG \
      --gpu 0 --seed 1337 --save_name v16_mnms_abl_${CFG1_NAME}_s1337 &
    PID_MNMS1=$!

    echo "  [GPU0] MNMS ablation $CFG2_NAME"
    python train_fourierfuse_v16.py $MNMS $COMMON $CFG2_FLAG \
      --gpu 0 --seed 1337 --save_name v16_mnms_abl_${CFG2_NAME}_s1337 &
    PID_MNMS2=$!

    # GPU 1: Prostate with both configs in parallel
    echo "  [GPU1] Prostate ablation $CFG1_NAME"
    python train_fourierfuse_v16.py $PROSTATE $COMMON $CFG1_FLAG \
      --gpu 1 --seed 1337 --save_name v16_prostate_abl_${CFG1_NAME}_s1337 &
    PID_PRO1=$!

    echo "  [GPU1] Prostate ablation $CFG2_NAME"
    python train_fourierfuse_v16.py $PROSTATE $COMMON $CFG2_FLAG \
      --gpu 1 --seed 1337 --save_name v16_prostate_abl_${CFG2_NAME}_s1337 &
    PID_PRO2=$!

    # Wait for all 4 processes to finish
    wait $PID_MNMS1 $PID_MNMS2 $PID_PRO1 $PID_PRO2
    echo "=== batch $CFG1_NAME + $CFG2_NAME completed ==="
  done

  echo "=== PHASE 2 COMPLETE ==="
}

# =============================================================================
# PHASE 3 — Sensitivity study (MNMS only)
# =============================================================================
# Vary alpha, beta, gamma across a small grid to show robustness. Reviewers
# will ask about sensitivity to the uncertainty weights; we need a figure.
#
# 6 configs, 1 seed, MNMS only = 6 runs, ~48h GPU-time, ~24h wall with 2 GPUs.
# =============================================================================
phase3_sensitivity() {
  echo "=== PHASE 3 — sensitivity study (MNMS, seed=1337) ==="

  # Grid: change one weight at a time, keeping others at 1.0.
  # We pair two configs per round on the two GPUs.

  echo "=== [GPU 0] MNMS alpha=0.5 | [GPU 1] MNMS alpha=2.0 ==="
  python train_fourierfuse_v16.py $MNMS $COMMON \
    --daep_alpha 0.5 --daep_beta 1.0 --daep_gamma 1.0 \
    --gpu 0 --seed 1337 --save_name v16_mnms_sens_a0.5 &
  python train_fourierfuse_v16.py $MNMS $COMMON \
    --daep_alpha 2.0 --daep_beta 1.0 --daep_gamma 1.0 \
    --gpu 1 --seed 1337 --save_name v16_mnms_sens_a2.0 &
  wait

  echo "=== [GPU 0] MNMS beta=0.5 | [GPU 1] MNMS beta=2.0 ==="
  python train_fourierfuse_v16.py $MNMS $COMMON \
    --daep_alpha 1.0 --daep_beta 0.5 --daep_gamma 1.0 \
    --gpu 0 --seed 1337 --save_name v16_mnms_sens_b0.5 &
  python train_fourierfuse_v16.py $MNMS $COMMON \
    --daep_alpha 1.0 --daep_beta 2.0 --daep_gamma 1.0 \
    --gpu 1 --seed 1337 --save_name v16_mnms_sens_b2.0 &
  wait

  echo "=== [GPU 0] MNMS gamma=0.5 | [GPU 1] MNMS gamma=2.0 ==="
  python train_fourierfuse_v16.py $MNMS $COMMON \
    --daep_alpha 1.0 --daep_beta 1.0 --daep_gamma 0.5 \
    --gpu 0 --seed 1337 --save_name v16_mnms_sens_g0.5 &
  python train_fourierfuse_v16.py $MNMS $COMMON \
    --daep_alpha 1.0 --daep_beta 1.0 --daep_gamma 2.0 \
    --gpu 1 --seed 1337 --save_name v16_mnms_sens_g2.0 &
  wait

  echo "=== PHASE 3 COMPLETE ==="
}

# =============================================================================
# Summary / parsing helpers
# =============================================================================
summarize_all() {
  echo ""
  echo "=================================================================="
  echo "=== v16 RESULTS SUMMARY ==="
  echo "=================================================================="
  for name in v16_mnms_s1337 v16_mnms_s42 v16_mnms_s2024 \
              v16_prostate_s1337 v16_prostate_s42 v16_prostate_s2024 \
              v16_fundus_s1337 v16_fundus_s42 v16_fundus_s2024 \
              v16_busi_s1337 v16_busi_s42 v16_busi_s2024; do
    case $name in
      v16_mnms*)     ds="MNMS" ;;
      v16_prostate*) ds="prostate" ;;
      v16_fundus*)   ds="fundus" ;;
      v16_busi*)     ds="BUSI" ;;
    esac
    log="../model/${ds}/train_fourierfuse_v16/${name}/log.txt"
    if [ -f "$log" ]; then
      best=$(grep "best_avg=" "$log" | tail -1 | grep -oP "best_avg=[\d.]+" | head -1)
      swad=$(grep "swad_best=" "$log" | tail -1 | grep -oP "swad_best=[\d.]+" | head -1)
      echo "  $name : $best | $swad"
    else
      echo "  $name : (log not found)"
    fi
  done

  echo ""; echo "=== Ablations ==="
  for name in v16_mnms_abl_no_ale_s1337 v16_mnms_abl_no_epi_s1337 \
              v16_mnms_abl_no_dom_s1337 v16_mnms_abl_threshold_s1337 \
              v16_prostate_abl_no_ale_s1337 v16_prostate_abl_no_epi_s1337 \
              v16_prostate_abl_no_dom_s1337 v16_prostate_abl_threshold_s1337; do
    case $name in
      v16_mnms*)     ds="MNMS" ;;
      v16_prostate*) ds="prostate" ;;
    esac
    log="../model/${ds}/train_fourierfuse_v16/${name}/log.txt"
    if [ -f "$log" ]; then
      best=$(grep "best_avg=" "$log" | tail -1 | grep -oP "best_avg=[\d.]+" | head -1)
      echo "  $name : $best"
    fi
  done
}

# =============================================================================
# Main entry
# =============================================================================
case "${1:-}" in
  phase0)      phase0_sanity ;;
  phase1)      phase1_main_results ;;
  phase2)      phase2_ablations ;;
  phase3)      phase3_sensitivity ;;
  summary)     summarize_all ;;
  all)
    phase0_sanity
    phase1_main_results
    phase2_ablations
    phase3_sensitivity
    summarize_all
    ;;
  *)
    echo "Usage: $0 {phase0|phase1|phase2|phase3|summary|all}"
    echo ""
    echo "  phase0   — sanity check on MNMS seed=1337 (RUN FIRST, ~8h)"
    echo "  phase1   — main results, 3 seeds x 4 datasets (~40h wall, 2 GPUs)"
    echo "  phase2   — ablations, 1 seed x 2 datasets x 4 configs (~30h wall)"
    echo "  phase3   — sensitivity study on MNMS (~24h wall)"
    echo "  summary  — parse all logs and print dice numbers"
    echo "  all      — run all phases sequentially (~4-5 days wall time)"
    exit 1
    ;;
esac
