#!/bin/bash
# =============================================================================
# v16 EXPERIMENT RUNNER — v2, MedIA/TMI submission track (2-GPU budget)
# =============================================================================
#
# v2 changes (vs original run_v16_experiments.sh):
#
#   - PHASE 4 (NEW): gap-fill the eight (dataset, ablation, seed=1337) cells
#                    that didn't run in phase 2 due to GPU re-allocation.
#                    This brings every ablation cell to n=3 seeds.
#                    ~30 GPU-hours total; ~16h wall time on 2 GPUs.
#
#   - PHASE 5 (NEW): component ablation on M&Ms, 3 seeds × 3 configs.
#                    Tests {FF − FDA, FF − SWAD, FixMatch baseline}; FF-full
#                    is reused from phase 1. ~144 GPU-hours; ~72h wall time
#                    on 2 GPUs. Generates a new "what does each module add?"
#                    table for the paper revision.
#
#   - SUMMARY parser updated to include the new runs.
#
# All changes are scheduling-only — NO modifications to train_fourierfuse_v16.py
# or fourierfuse_v16_modules.py. Every flag used below was already present in
# the v16 training script.
#
# =============================================================================
#
# Phase index:
#
#   PHASE 0  — sanity check on MNMS seed=1337 (run first, ~8h)
#   PHASE 1  — main results, 3 seeds × 4 datasets (12 runs, ~40h wall)
#   PHASE 2  — ablations, 1 seed × 2 datasets × 4 configs (12 runs, ~30h wall)
#   PHASE 3  — sensitivity study on MNMS (6 runs, ~24h wall)
#   PHASE 4  — gap-fill 8 missing seed-1337 ablation cells          [NEW]
#   PHASE 5  — component ablation on M&Ms, 3 seeds × 3 configs      [NEW]
#   SUMMARY  — parse all logs and print Dice numbers                [extended]
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

SEEDS_NONSEED1337="42 2024"
SEEDS_ALL="1337 42 2024"

# =============================================================================
# PHASE 0 — Sanity check (run before phase 1)
# =============================================================================
phase0_sanity() {
  echo "=== PHASE 0 — sanity check on MNMS seed=1337 (~8h on 1 GPU) ==="
  python train_fourierfuse_v16.py $MNMS $COMMON \
    --gpu 1 --seed 1337 --save_name v16_sanity_mnms_s1337
  log="../model/MNMS/train_fourierfuse_v16/v16_sanity_mnms_s1337/log.txt"
  if [ -f "$log" ]; then
    echo ""; echo "=== Sanity check results ==="
    best=$(grep "best_avg=" "$log" | tail -1)
    echo "Last line: $best"
    w_lines=$(grep -c "w_mean=" "$log")
    echo "DAEP w_mean log lines seen: $w_lines"
    echo ""
    echo "PASS CRITERIA: best_avg > 0.74; w_mean ∈ [0.3, 0.9]; no non-finite-loss warnings."
  fi
}

# =============================================================================
# PHASE 1 — Main results (12 runs: 3 seeds × 4 datasets)
# =============================================================================
phase1_main_results() {
  echo "=== PHASE 1 — main results, 3 seeds, 4 datasets, 2 GPUs, 2 processes/GPU ==="
  for SEED in $SEEDS_NONSEED1337; do
    echo "============================================"
    echo "  SEED = $SEED"
    echo "============================================"

    python train_fourierfuse_v16.py $PROSTATE $COMMON \
      --gpu 0 --seed $SEED --save_name v16_prostate_s$SEED &
    PID_PROSTATE=$!
    python train_fourierfuse_v16.py $FUNDUS $COMMON \
      --gpu 0 --seed $SEED --save_name v16_fundus_s$SEED &
    PID_FUNDUS=$!
    python train_fourierfuse_v16.py $MNMS $COMMON \
      --gpu 1 --seed $SEED --save_name v16_mnms_s$SEED &
    PID_MNMS=$!
    python train_fourierfuse_v16.py $BUSI $COMMON \
      --gpu 1 --seed $SEED --save_name v16_busi_s$SEED &
    PID_BUSI=$!
    wait $PID_PROSTATE $PID_FUNDUS $PID_MNMS $PID_BUSI
    echo "=== seed=$SEED completed for all four datasets ==="
  done
  echo "=== PHASE 1 COMPLETE ==="
}

# =============================================================================
# PHASE 2 — Ablations (existing, 12 runs at seed=1337 across MNMS+Prostate)
# =============================================================================
phase2_ablations() {
  echo "=== PHASE 2 — ablations on MNMS + Prostate (seed=1337), 2 processes per GPU ==="
  ABLATIONS=(
    "no_ale:--daep_no_ale"
    "no_epi:--daep_no_epi"
    "no_dom:--daep_no_dom"
    "threshold:--daep_threshold_baseline"
  )
  for ((i=0; i<${#ABLATIONS[@]}; i+=2)); do
    CFG1="${ABLATIONS[$i]}"; CFG2="${ABLATIONS[$i+1]}"
    CFG1_NAME="${CFG1%%:*}"; CFG1_FLAG="${CFG1##*:}"
    CFG2_NAME="${CFG2%%:*}"; CFG2_FLAG="${CFG2##*:}"
    echo "=== Batch: $CFG1_NAME and $CFG2_NAME ==="

    python train_fourierfuse_v16.py $MNMS $COMMON $CFG1_FLAG \
      --gpu 0 --seed 1337 --save_name v16_mnms_abl_${CFG1_NAME}_s1337 &
    python train_fourierfuse_v16.py $MNMS $COMMON $CFG2_FLAG \
      --gpu 0 --seed 1337 --save_name v16_mnms_abl_${CFG2_NAME}_s1337 &
    python train_fourierfuse_v16.py $PROSTATE $COMMON $CFG1_FLAG \
      --gpu 1 --seed 1337 --save_name v16_prostate_abl_${CFG1_NAME}_s1337 &
    python train_fourierfuse_v16.py $PROSTATE $COMMON $CFG2_FLAG \
      --gpu 1 --seed 1337 --save_name v16_prostate_abl_${CFG2_NAME}_s1337 &
    wait
    echo "=== batch $CFG1_NAME + $CFG2_NAME completed ==="
  done
  echo "=== PHASE 2 COMPLETE ==="
}

# =============================================================================
# PHASE 3 — Sensitivity study (existing)
# =============================================================================
phase3_sensitivity() {
  echo "=== PHASE 3 — sensitivity study (MNMS, seed=1337) ==="
  for tag_pair in "alpha:0.5:2.0" "beta:0.5:2.0" "gamma:0.5:2.0"; do
    name="${tag_pair%%:*}"; rest="${tag_pair#*:}"; lo="${rest%%:*}"; hi="${rest##*:}"
    echo "=== $name=$lo on GPU 0 | $name=$hi on GPU 1 ==="
    a_lo=1.0; b_lo=1.0; g_lo=1.0; a_hi=1.0; b_hi=1.0; g_hi=1.0
    case "$name" in
      alpha) a_lo=$lo; a_hi=$hi ;;
      beta)  b_lo=$lo; b_hi=$hi ;;
      gamma) g_lo=$lo; g_hi=$hi ;;
    esac
    python train_fourierfuse_v16.py $MNMS $COMMON \
      --daep_alpha $a_lo --daep_beta $b_lo --daep_gamma $g_lo \
      --gpu 0 --seed 1337 --save_name v16_mnms_sens_${name:0:1}${lo} &
    python train_fourierfuse_v16.py $MNMS $COMMON \
      --daep_alpha $a_hi --daep_beta $b_hi --daep_gamma $g_hi \
      --gpu 1 --seed 1337 --save_name v16_mnms_sens_${name:0:1}${hi} &
    wait
  done
  echo "=== PHASE 3 COMPLETE ==="
}

# =============================================================================
# PHASE 4 — GAP FILL: eight missing seed-1337 ablation cells          [NEW]
# =============================================================================
#
# Inventory of missing cells (every cell is at seed=1337):
#
#   BUSI   --daep_no_ale   v16_busi_abl_no_ale_s1337
#   BUSI   --daep_no_epi   v16_busi_abl_no_epi_s1337
#   BUSI   --daep_no_dom   v16_busi_abl_no_dom_s1337
#   Fundus --daep_no_ale   v16_fundus_abl_no_ale_s1337
#   Fundus --daep_no_epi   v16_fundus_abl_no_epi_s1337
#   Fundus --daep_no_dom   v16_fundus_abl_no_dom_s1337
#   Prostate --daep_no_dom v16_prostate_abl_no_dom_s1337
#   Prostate --daep_threshold_baseline  v16_prostate_abl_threshold_s1337
#
# Total ≈ 30 GPU-hours: 6 × short (BUSI/Fundus, ~30k iter) + 2 × long
# (Prostate, ~60k iter). Strategy: pair one BUSI + one Fundus on each GPU
# to keep BUSI's ~6h finish from idling the GPU; then the two Prostate
# runs share the GPUs in the final batch.
#
# Wall time on 2 × 4090: ~16 hours.
# =============================================================================
phase4_gap_fill() {
  echo "=== PHASE 4 — gap-fill missing seed-1337 ablation cells ==="
  echo "    eight runs across BUSI/Fundus/Prostate, ~16h wall on 2 GPUs"

  # ---- Round A: BUSI no_ale + Fundus no_ale (GPU 0)
  #             BUSI no_epi + Fundus no_epi (GPU 1)
  echo ""
  echo "=== Round A: no_ale on GPU 0, no_epi on GPU 1 ==="
  python train_fourierfuse_v16.py $BUSI $COMMON --daep_no_ale \
    --gpu 0 --seed 1337 --save_name v16_busi_abl_no_ale_s1337 &
  python train_fourierfuse_v16.py $FUNDUS $COMMON --daep_no_ale \
    --gpu 0 --seed 1337 --save_name v16_fundus_abl_no_ale_s1337 &
  python train_fourierfuse_v16.py $BUSI $COMMON --daep_no_epi \
    --gpu 1 --seed 1337 --save_name v16_busi_abl_no_epi_s1337 &
  python train_fourierfuse_v16.py $FUNDUS $COMMON --daep_no_epi \
    --gpu 1 --seed 1337 --save_name v16_fundus_abl_no_epi_s1337 &
  wait
  echo "=== Round A complete ==="

  # ---- Round B: BUSI no_dom + Fundus no_dom (GPU 0)
  #             Prostate no_dom (GPU 1, sole occupant — long run)
  echo ""
  echo "=== Round B: no_dom × {BUSI, Fundus} on GPU 0; Prostate no_dom on GPU 1 ==="
  python train_fourierfuse_v16.py $BUSI $COMMON --daep_no_dom \
    --gpu 0 --seed 1337 --save_name v16_busi_abl_no_dom_s1337 &
  python train_fourierfuse_v16.py $FUNDUS $COMMON --daep_no_dom \
    --gpu 0 --seed 1337 --save_name v16_fundus_abl_no_dom_s1337 &
  python train_fourierfuse_v16.py $PROSTATE $COMMON --daep_no_dom \
    --gpu 1 --seed 1337 --save_name v16_prostate_abl_no_dom_s1337 &
  wait
  echo "=== Round B complete ==="

  # ---- Round C: Prostate threshold (sole long run, runs on whichever GPU is free)
  echo ""
  echo "=== Round C: Prostate threshold on GPU 1 ==="
  python train_fourierfuse_v16.py $PROSTATE $COMMON --daep_threshold_baseline \
    --gpu 1 --seed 1337 --save_name v16_prostate_abl_threshold_s1337
  echo "=== Round C complete ==="

  echo ""
  echo "=== PHASE 4 COMPLETE — eight gap-fill cells finished ==="
  echo "    All ablation cells now have n = 3 seeds."
}

# =============================================================================
# PHASE 5 — Component ablation on M&Ms (3 seeds × 3 configs)        [NEW]
# =============================================================================
#
# Question: "what does each component contribute?" — the natural
# follow-up to §IV.D's pseudo-label-gating ablation. M&Ms is the headline
# benchmark for this paper (largest win, multi-vendor, K=4) so it's the
# right dataset to anchor the component ablation.
#
# Configurations (FF-full = 3 already-completed phase-1 runs, reused):
#   FF − FDA           : --ff_no_fda                                      (FDA off, DAEP on, SWAD on)
#   FF − SWAD          : --swad_disable                                    (FDA on, DAEP on, SWAD off)
#   FixMatch baseline  : --ff_no_fda --daep_threshold_baseline             (FDA off, DAEP off → threshold,
#                                                                          SWAD on; SWAD is just a
#                                                                          final-checkpoint averaging
#                                                                          and not a training-dynamics
#                                                                          component, so we keep it)
#
# 3 seeds × 3 configs = 9 NEW runs (FF-full reuses phase 1).
# Each MNMS run is ~16 GPU-hours (60k iters @ batch=2/2). Total ≈ 144 GPU-h.
# Wall time on 2 × 4090: ~72 hours (3 days). Pair one config-pair per round
# across the two GPUs.
# =============================================================================
phase5_components() {
  echo "=== PHASE 5 — component ablation on M&Ms, 3 seeds × 3 configs ==="
  echo "    9 runs, ~144 GPU-hours, ~72h wall on 2 GPUs."
  echo "    FF-full numbers reused from phase 1."

  for SEED in $SEEDS_ALL; do
    echo ""
    echo "============================================"
    echo "  SEED = $SEED  (component ablation on M&Ms)"
    echo "============================================"

    # GPU 0: FF − FDA + FixMatch baseline (both ~16h)
    echo "[GPU0] M&Ms FF−FDA seed=$SEED"
    python train_fourierfuse_v16.py $MNMS $COMMON --ff_no_fda \
      --gpu 0 --seed $SEED --save_name v16_mnms_comp_noFDA_s$SEED &
    PID1=$!

    echo "[GPU0] M&Ms FixMatch baseline seed=$SEED  (no FDA, no DAEP)"
    python train_fourierfuse_v16.py $MNMS $COMMON --ff_no_fda --daep_threshold_baseline \
      --gpu 0 --seed $SEED --save_name v16_mnms_comp_fixmatch_s$SEED &
    PID2=$!

    # GPU 1: FF − SWAD (one run, GPU 1 sits at ~50% utilisation —
    #        acceptable, simpler scheduling).
    echo "[GPU1] M&Ms FF−SWAD seed=$SEED"
    python train_fourierfuse_v16.py $MNMS $COMMON --swad_disable \
      --gpu 1 --seed $SEED --save_name v16_mnms_comp_noSWAD_s$SEED &
    PID3=$!

    wait $PID1 $PID2 $PID3
    echo "=== seed=$SEED component ablation completed ==="
  done

  echo ""
  echo "=== PHASE 5 COMPLETE ==="
}

# =============================================================================
# Summary / parsing helpers (extended for v3 runs)
# =============================================================================
summarize_all() {
  echo ""
  echo "=================================================================="
  echo "=== v16 RESULTS SUMMARY (all phases, including v3 additions) ==="
  echo "=================================================================="

  # Main + phase-2 ablations + phase-4 gap-fill cells
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

  echo ""; echo "=== Pseudo-label-gating ablations (phase 2 + phase 4 gap-fill) ==="
  for ds_root in MNMS:v16_mnms prostate:v16_prostate fundus:v16_fundus BUSI:v16_busi; do
    ds="${ds_root%%:*}"; pre="${ds_root##*:}"
    for cfg in no_ale no_epi no_dom threshold; do
      for seed in 1337 42 2024; do
        name="${pre}_abl_${cfg}_s${seed}"
        log="../model/${ds}/train_fourierfuse_v16/${name}/log.txt"
        if [ -f "$log" ]; then
          best=$(grep "best_avg=" "$log" | tail -1 | grep -oP "best_avg=[\d.]+" | head -1)
          echo "  $name : $best"
        fi
      done
    done
  done

  echo ""; echo "=== Component ablations on M&Ms (phase 5) ==="
  for cfg in noFDA noSWAD fixmatch; do
    for seed in 1337 42 2024; do
      name="v16_mnms_comp_${cfg}_s${seed}"
      log="../model/MNMS/train_fourierfuse_v16/${name}/log.txt"
      if [ -f "$log" ]; then
        best=$(grep "best_avg=" "$log" | tail -1 | grep -oP "best_avg=[\d.]+" | head -1)
        echo "  $name : $best"
      fi
    done
  done

  echo ""; echo "=== Sensitivity (phase 3) ==="
  for tag in a0.5 a2.0 b0.5 b2.0 g0.5 g2.0; do
    name="v16_mnms_sens_${tag}"
    log="../model/MNMS/train_fourierfuse_v16/${name}/log.txt"
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
  phase4)      phase4_gap_fill ;;
  phase5)      phase5_components ;;
  summary)     summarize_all ;;
  v3)
    # Convenience: phase 4 + phase 5 + summary, the new work for the v3 paper.
    phase4_gap_fill
    phase5_components
    summarize_all
    ;;
  all)
    phase0_sanity
    phase1_main_results
    phase2_ablations
    phase3_sensitivity
    phase4_gap_fill
    phase5_components
    summarize_all
    ;;
  *)
    echo "Usage: $0 {phase0|phase1|phase2|phase3|phase4|phase5|v3|summary|all}"
    echo ""
    echo "  phase0   — sanity check on MNMS seed=1337 (~8h)"
    echo "  phase1   — main results, 3 seeds × 4 datasets (~40h wall, 2 GPUs)"
    echo "  phase2   — ablations, 1 seed × 2 datasets × 4 configs (~30h wall)"
    echo "  phase3   — sensitivity study on MNMS (~24h wall)"
    echo "  phase4   — gap-fill 8 missing seed-1337 ablation cells (~16h wall)  [NEW]"
    echo "  phase5   — component ablation on M&Ms, 3 seeds × 3 configs (~72h)   [NEW]"
    echo "  v3       — phase4 + phase5 + summary (run this for the paper revision)"
    echo "  summary  — parse all logs and print Dice numbers"
    echo "  all      — run every phase sequentially (~7-8 days wall time)"
    exit 1
    ;;
esac
