#!/bin/bash
# Phase 7 v16 launcher — rebuilt-from-scratch calibration probe.
# =============================================================================
# v14/v15 had a fatal bug: they passed normal_toTensor=None to
# ProstateSegmentation, but the dataloader unconditionally calls
# self.normal_toTensor(sample) at test time. This crashed at the
# first batch with TypeError. v16 fixes this by constructing the
# correct transform (matches train_fourierfuse_v16.py:668 exactly).
#
# Usage:
#   GPU=0 CKPT_DAEP=path/to/daep.pth ./run_phase7_v16.sh
# Or with both checkpoints:
#   CKPT_DAEP=...daep.pth CKPT_THR=...thr.pth ./run_phase7_v16.sh
# =============================================================================

set -e

# 1. Always run smoke test first; fail fast on math bugs.
echo "=== Phase 7 v16: synthetic smoke test ==="
python test_phase7_v16_metrics.py
if [ $? -ne 0 ]; then
    echo "Smoke test failed; aborting GPU probe." >&2
    exit 1
fi

CKPT_DAEP="${CKPT_DAEP:-../model/prostate/train_fourierfuse_v16/v16_prostate_daep_s1337/best_model.pth}"
CKPT_THR="${CKPT_THR:-../model/prostate/train_fourierfuse_v16/v16_prostate_thr_s1337/best_model.pth}"
DATA_ROOT="${DATA_ROOT:-../../data/ProstateSlice}"
OUT_DIR="${OUT_DIR:-./phase7_v16_out}"
GPU="${GPU:-0}"

# 2. Confirm the DAEP checkpoint exists.
if [ ! -f "$CKPT_DAEP" ]; then
    echo "ERROR: DAEP checkpoint not found at $CKPT_DAEP" >&2
    echo "Override with CKPT_DAEP=<path> ./run_phase7_v16.sh" >&2
    exit 1
fi

echo ""
echo "=== Phase 7 v16: launching GPU probe ==="
echo "  DAEP checkpoint:      $CKPT_DAEP"
echo "  Threshold checkpoint: $CKPT_THR (optional)"
echo "  Data root:            $DATA_ROOT"
echo "  Output dir:           $OUT_DIR"
echo "  GPU:                  $GPU"
echo ""

if [ -f "$CKPT_THR" ]; then
    python run_v16_phase7_calibration_v16.py \
        --checkpoint           "$CKPT_DAEP" \
        --threshold_checkpoint "$CKPT_THR"  \
        --data_root            "$DATA_ROOT" \
        --output_dir           "$OUT_DIR"   \
        --gpu                  "$GPU"
else
    echo "Note: threshold checkpoint not found at $CKPT_THR; DAEP-only probe."
    python run_v16_phase7_calibration_v16.py \
        --checkpoint  "$CKPT_DAEP" \
        --data_root   "$DATA_ROOT" \
        --output_dir  "$OUT_DIR"   \
        --gpu         "$GPU"
fi

echo ""
echo "=== Phase 7 v16 complete ==="
echo "  Summary: $OUT_DIR/calibration_probe_v16_summary.json"
echo "  Plots:   $OUT_DIR/{reliability_diagrams,risk_coverage}.pdf"
