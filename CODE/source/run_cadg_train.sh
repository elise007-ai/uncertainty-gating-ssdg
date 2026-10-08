#!/usr/bin/env bash
# =============================================================================
# run_cadg_train.sh — Option-B main training (CADG + ablations), 2 GPUs.
# Run ONLY after run_cadg_diagnostics.sh passes.
# =============================================================================
# Uses the EXACT v16 per-dataset configs from run_v16_experiments.sh, adding the
# --method ablation axis:
#   cadg         full Option B (per-domain label-free recalibrated gate)  [NOVEL]
#   cadg_global  single shared calibration map (isolates per-domain)      [ablation]
#   daep         evidential gate, NO recalibration (== Option A)          [ablation]
#   fixmatch     fixed-threshold baseline                                 [baseline]
#
# Seeds actually vary the labeled split now (seed_split patch; on by default).
# Primary endpoint: per-domain Dice on the most-shifted vendors (see test.py).
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/../CODE/source"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-$USER}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

COMMON="--warmup --warmup_period 150 --AdamW --amp 1 \
  --ff_base_features 64 \
  --ff_spec_consistency_weight 0.5 --ff_fda_beta 0.05 \
  --consistency 1.0 --consistency_rampup 150.0 --ema_decay 0.995 \
  --daep_alpha 1.0 --daep_beta 1.0 --daep_gamma 1.0 \
  --proto_per_domain 2 --proto_ema 0.99 \
  --evid_lambda_kl_max 0.02 --evid_anneal_iters 10000 \
  --cadg_n_bins 20 --cadg_refit_every 200 --cadg_min_count 50000 \
  --save_model --overwrite"

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

declare -A CFG=( [prostate]="$PROSTATE" [fundus]="$FUNDUS" [BUSI]="$BUSI" [MNMS]="$MNMS" )

# ---- choose scope here ----
DATASETS=(prostate fundus MNMS BUSI)
METHODS=(cadg cadg_global daep fixmatch)
SEEDS=(1337 42 2024)
# ---------------------------

JOBS=()
for ds in "${DATASETS[@]}"; do
  for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
      JOBS+=("$ds|$m|$s")
    done
  done
done
echo "Total jobs: ${#JOBS[@]}"

run_one () {
  local gpu="$1" spec="$2"
  IFS='|' read -r ds m s <<< "$spec"
  local name="cadg_${ds}_${m}_s${s}"
  echo "[GPU $gpu] START $name"
  python train_fourierfuse_v16.py \
      ${CFG[$ds]} $COMMON \
      --method "$m" \
      --gpu "$gpu" --seed "$s" \
      --save_name "$name"
  echo "[GPU $gpu] DONE  $name"
}

# two-at-a-time across the two GPUs
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  run_one 0 "${JOBS[$i]}" &
  pid0=$!
  pid1=""
  if [ $((i+1)) -lt ${#JOBS[@]} ]; then
    run_one 1 "${JOBS[$((i+1))]}" &
    pid1=$!
  fi
  wait $pid0
  [ -n "$pid1" ] && wait $pid1
  i=$((i+2))
done

echo "All Option-B runs finished. Use test.py per domain and compare"
echo "cadg vs {cadg_global, daep, fixmatch} on the shifted vendors."
