#!/usr/bin/env bash
# run_adaptive_threshold.sh
# =============================================================================
# CMIG revision experiment: FreeMatch-style ADAPTIVE confidence threshold as a
# third pseudo-label gate, vs the fixed threshold and the evidential head (DAEP),
# to answer "is the comparison fair against a *smarter* threshold?".
#
# Produces the numbers for Table 'adaptive' (Fundus + Prostate, 3 seeds).
#
# IMPORTANT — use the PATCHED training script before running this:
#   This experiment needs `--method adaptive`, which the original
#   train_fourierfuse_v16.py does not have. A patched copy is provided:
#     PAPER_CMIG/latex_source/scripts/train_fourierfuse_v16.py   (drop-in), and
#     CODE/source/train_fourierfuse_v16_adaptive.py              (same file, renamed)
#   The ONLY change vs the original is the added 'adaptive' method; fixmatch and
#   daep are byte-for-byte unchanged, so their columns reproduce the paper.
#   Before running, from ProtoSynFoC/source:
#     cp /path/to/train_fourierfuse_v16_adaptive.py ./train_fourierfuse_v16.py
#   (or back up your original first). Then run this script from that directory.
#
# Run from:  ProtoSynFoC/source   (same place you run the other experiments)
# Data paths are resolved INTERNALLY by the training script (../../data/...),
# so there is no --data_root argument.
# =============================================================================
set -e
export MPLCONFIGDIR=/tmp/mpl_cache
mkdir -p "$MPLCONFIGDIR"

# Exact training config used for the paper's numbers (COMMON block), so the
# fixmatch/daep columns match Table 'main'/'ablation' and only the gate varies.
COMMON="--warmup --warmup_period 150 --AdamW --amp 1 \
  --ff_base_features 64 \
  --ff_spec_consistency_weight 0.5 --ff_fda_beta 0.05 \
  --consistency 1.0 --consistency_rampup 150.0 --ema_decay 0.995 \
  --daep_alpha 1.0 --daep_beta 1.0 --daep_gamma 1.0 \
  --proto_per_domain 2 --proto_ema 0.99 \
  --evid_lambda_kl_max 0.02 --evid_anneal_iters 10000 \
  --save_model --overwrite"

PROSTATE="--dataset prostate --lb_num 20 --lb_domain 5 --domain_num 6 \
  --max_iterations 60000 --num_eval_iter 500 \
  --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 \
  --swad_start_iter 15000 --swad_update_every 100 --swad_bn_recalib_batches 20"

FUNDUS="--dataset fundus --lb_num 20 --lb_domain 3 --domain_num 4 \
  --max_iterations 30000 --num_eval_iter 500 \
  --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 \
  --swad_start_iter 10000 --swad_update_every 100 --swad_bn_recalib_batches 20"

SEEDS="1337 42 2024"
GPU=0

for SEED in $SEEDS; do
  for METHOD in fixmatch adaptive daep; do
    echo "=== Fundus | method=$METHOD | seed=$SEED ==="
    python train_fourierfuse_v16.py $FUNDUS $COMMON \
      --method $METHOD --gpu $GPU --seed $SEED \
      --save_name adaptive_cmp_fundus_${METHOD}_s${SEED}

    echo "=== Prostate | method=$METHOD | seed=$SEED ==="
    python train_fourierfuse_v16.py $PROSTATE $COMMON \
      --method $METHOD --gpu $GPU --seed $SEED \
      --save_name adaptive_cmp_prostate_${METHOD}_s${SEED}
  done
done

echo
echo "Done. Models + eval logs are under:"
echo "  ../model/fundus/train_fourierfuse_v16/adaptive_cmp_fundus_<method>_s<seed>/"
echo "  ../model/prostate/train_fourierfuse_v16/adaptive_cmp_prostate_<method>_s<seed>/"
echo "Read the best average Dice from each run's log and fill Table 'adaptive'"
echo "(mean +/- std over the 3 seeds). The fixmatch and daep columns should match"
echo "the numbers already in the main/ablation tables (a setup sanity check)."
