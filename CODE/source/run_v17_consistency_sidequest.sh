#!/bin/bash
# =============================================================================
# v17 SIDEQUEST — scale-free pseudo-label gates (toward a SECOND paper)
# =============================================================================
# Motivation (from paper 1): the teacher's Dirichlet vacuity K/S RANKS error
# well (AUROC ~0.84) but its dynamic RANGE collapses to ~1e-4, so a fixed gate
# cannot use it. These gates avoid depending on the absolute uncertainty scale:
#   - consistency_gate : trust a pixel only where the teacher's hard prediction
#                        is STABLE across FDA style views (scale-free, domain-aware)
#   - percentile_gate  : keep the top-confidence fraction PER IMAGE (rank-based)
#
# We compare both against the two established baselines (fixmatch, daep) on the
# two most diagnostic benchmarks (Fundus, Prostate), 3 seeds each, using the
# SAME seed-split protocol and SAME config as the paper's corrected runs so the
# numbers are directly comparable to the adaptive_cmp_* / Table-2 results.
#
# IMPORTANT: run with train_fourierfuse_v17_consistency.py (the patched file that
# adds --method consistency_gate / percentile_gate). fixmatch & daep paths are
# byte-identical to the original, so those columns reproduce the paper.
#
# Hardware: designed for 2x RTX 3090 (or 4090). Fundus on GPU0, Prostate on GPU1.
# Run inside tmux. ~ each Fundus run 30k iter, each Prostate run 60k iter.
# =============================================================================
set -e
export MPLCONFIGDIR=/tmp/mpl_cache
mkdir -p "$MPLCONFIGDIR"

TRAIN=train_fourierfuse_v17_consistency.py

COMMON="--warmup --warmup_period 150 --AdamW --amp 1 \
  --ff_base_features 64 \
  --ff_spec_consistency_weight 0.5 --ff_fda_beta 0.05 \
  --consistency 1.0 --consistency_rampup 150.0 --ema_decay 0.995 \
  --daep_alpha 1.0 --daep_beta 1.0 --daep_gamma 1.0 \
  --proto_per_domain 2 --proto_ema 0.99 \
  --evid_lambda_kl_max 0.02 --evid_anneal_iters 10000 \
  --save_model --overwrite"

FUNDUS="--dataset fundus --lb_num 20 --lb_domain 3 --domain_num 4 \
  --max_iterations 30000 --num_eval_iter 500 \
  --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 \
  --swad_start_iter 10000 --swad_update_every 100 --swad_bn_recalib_batches 20"

PROSTATE="--dataset prostate --lb_num 20 --lb_domain 5 --domain_num 6 \
  --max_iterations 60000 --num_eval_iter 500 \
  --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 \
  --swad_start_iter 15000 --swad_update_every 100 --swad_bn_recalib_batches 20"

# The two NEW gates. (fixmatch/daep already exist from adaptive_cmp_* — only
# re-run them here if you want them under an identical software snapshot.)
NEW_METHODS=(consistency_gate percentile_gate)
SEEDS=(1337 42 2024)

for METHOD in "${NEW_METHODS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "[GPU0] FUNDUS   ${METHOD} seed=${SEED}"
    python $TRAIN $FUNDUS $COMMON --method "$METHOD" \
      --gpu 0 --seed "$SEED" \
      --save_name "v17_fundus_${METHOD}_s${SEED}" &

    echo "[GPU1] PROSTATE ${METHOD} seed=${SEED}"
    python $TRAIN $PROSTATE $COMMON --method "$METHOD" \
      --gpu 1 --seed "$SEED" \
      --save_name "v17_prostate_${METHOD}_s${SEED}" &

    wait   # one Fundus + one Prostate at a time (one run per GPU)
  done
done

echo "============================================================"
echo "v17 sidequest complete. Runs under:"
echo "  ../model/fundus/.../v17_fundus_{consistency_gate,percentile_gate}_s*"
echo "  ../model/prostate/.../v17_prostate_{consistency_gate,percentile_gate}_s*"
echo ""
echo "Compare 'Best avg dice (student)' against the adaptive_cmp_* fixmatch/daep"
echo "numbers (same protocol). For Fundus also compare per-domain CUP dice."
echo "Expected per the hypothesis: consistency_gate >= fixmatch if scale-free"
echo "agreement carries usable signal; if NOT, that is also a publishable result."
echo "============================================================"
