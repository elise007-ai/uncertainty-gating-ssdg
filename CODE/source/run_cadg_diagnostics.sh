#!/usr/bin/env bash
# =============================================================================
# run_cadg_diagnostics.sh  — RUN FIRST (no training; uses your v16 checkpoints)
# =============================================================================
# Kill-switch Experiments 0 + 1 on the REAL FourierFuse v16 release.
# Decides whether Option B is viable before any training. See docs/DESIGN.md §5.
#
# Edit DATA_ROOT and CKPT_DIR to your machine. The checkpoint you gave:
#   .../model/prostate/train_fourierfuse_v16/v16_prostate_s{1337,42,2024}/best_student.pth
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../CODE/source"   # run from CODE/source

# Silence the matplotlib "/home/abc/.config" permission warning (HOME is
# mis-set in some shells on this box) by pointing caches at a writable dir.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-$USER}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

# ---- EDIT THESE ----
DATASET=prostate
DATA_ROOT=/home/yahyaisaac/root_eli/ProtoSynFoC/data/prostate
CKPT_DIR=/home/yahyaisaac/root_eli/ProtoSynFoC/model/prostate/train_fourierfuse_v16
DOMAIN_NUM=6
IMG_SIZE=512
FDA_BETA=0.05
# 1-indexed id of the most-shifted vendor (BIDMC). The prostate loader orders
# vendors alphabetically: 1=BIDMC, 2=BMC, 3=HK, 4=I2CVB, 5=RUNMC, 6=UCL.
# BIDMC (the legacy 1.5T vendor the BIDMC analysis is about) is domain 1.
SHIFTED=1
OUTDIR=../../diag_out
# --------------------

mkdir -p "$OUTDIR"

for SEED in 1337 42 2024; do
  CKPT="$CKPT_DIR/v16_${DATASET}_s${SEED}/best_student.pth"
  if [ ! -f "$CKPT" ]; then
    echo "SKIP seed $SEED — checkpoint not found: $CKPT"
    continue
  fi
  echo "=== diagnostics on seed $SEED ==="
  CUDA_VISIBLE_DEVICES=0 python cadg_diagnostics.py \
      --dataset "$DATASET" \
      --data_root "$DATA_ROOT" \
      --ckpt "$CKPT" \
      --domain_num "$DOMAIN_NUM" \
      --img_size "$IMG_SIZE" \
      --fda_beta "$FDA_BETA" \
      --shifted_domain "$SHIFTED" \
      --out "$OUTDIR/diag_${DATASET}_s${SEED}.json"
  echo ""
done

echo "=== DECISION ==="
echo "Inspect $OUTDIR/diag_${DATASET}_s*.json -> summary.EXP0_VERDICT / EXP1_VERDICT"
echo "Both SUPPORTED/VALIDATED (or WEAK/PARTIAL with a clear per-vendor story,"
echo "especially on BIDMC) -> run run_cadg_train.sh."
echo "Either FAILED -> stop Option B, ship Option A."
