#!/usr/bin/env bash
# =============================================================================
# run_normstat_diagnostics.sh — NEXT-PROJECT Stage-0 kill-switch (no training)
# =============================================================================
# Tests whether BatchNorm-statistics mismatch predicts pseudo-label error better
# than confidence / vacuity, on the SAME v16 checkpoints. See NEXT_PROJECT/PROPOSAL.md.
# No training. Decision gate before building the next project.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../CODE/source" 2>/dev/null || cd "$(dirname "$0")"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-$USER}"; mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

# ---- EDIT THESE ----
DATASET=prostate
DATA_ROOT=/home/yahyaisaac/root_eli/data/ProstateSlice
CKPT_DIR=/home/yahyaisaac/root_eli/ProtoSynFoC/model/prostate/train_fourierfuse_v16
DOMAIN_NUM=6
IMG_SIZE=384
SHIFTED=1            # BIDMC is domain 1 (alphabetical loader order)
OUTDIR=../../normstat_out
# --------------------

mkdir -p "$OUTDIR"
for SEED in 1337 42 2024; do
  CKPT="$CKPT_DIR/v16_${DATASET}_s${SEED}/best_student.pth"
  [ -f "$CKPT" ] || { echo "SKIP s$SEED ($CKPT missing)"; continue; }
  echo "=== normstat Stage-0 on seed $SEED ==="
  python normstat_diagnostics.py \
      --dataset "$DATASET" --data_root "$DATA_ROOT" --ckpt "$CKPT" \
      --domain_num "$DOMAIN_NUM" --img_size "$IMG_SIZE" --shifted_domain "$SHIFTED" \
      --out "$OUTDIR/normstat_${DATASET}_s${SEED}.json"
  echo ""
done

echo "=== DECISION ==="
echo "Inspect $OUTDIR/normstat_${DATASET}_s*.json -> summary.VERDICT"
echo "P1 SUPPORTED on shifted vendor (BIDMC) -> proceed to Stage 1."
echo "P1 FAILED -> stop; ship the strengthened Option A."
