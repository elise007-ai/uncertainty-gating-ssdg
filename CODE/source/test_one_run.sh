#!/bin/bash
# =============================================================================
# test_one_run.sh — convenience wrapper for evaluating one FourierFuse run
# =============================================================================
#
# `test.py` (inherited from the SynFoC codebase) hard-codes a specific
# directory layout for checkpoints and data:
#
#   <REPO_ROOT>/
#   ├── code_3/                       <- this directory
#   │   └── test.py
#   ├── model/
#   │   └── <dataset>/<save_name>/
#   │       └── unet_avg_dice_best_model.pth
#   ├── data/
#   │   ├── Fundus/
#   │   ├── ProstateSlice/
#   │   ├── MNMS/mnms/
#   │   └── BUSI/
#   └── checkpoints/
#       └── medsam_vit_b.pth          <- only required for SAM-branch evaluation
#
# That is, `test.py` expects to be run *from* `code_3/`, and resolves
# `../model/...` and `../../data/...` relative to that working directory.
#
# This wrapper just changes into the code_3/ directory and forwards your
# arguments. It does NOT relocate the model/ and data/ directories —
# please set them up at the locations shown above before running.
#
# =============================================================================
# Usage
# =============================================================================
#
#   ./test_one_run.sh <dataset> <save_name> [extra args]
#
# Examples:
#
#   # Test the seed-1337 FourierFuse DAEP run on Prostate
#   ./test_one_run.sh prostate v16_prostate_daep_s1337
#
#   # Test the seed-42 threshold variant on Fundus
#   ./test_one_run.sh fundus v16_fundus_thr_s42 --domain_num 4
#
#   # Test our SynFoC-UNet 3-seed re-runs (Phase 8a/b/c/d)
#   ./test_one_run.sh fundus synfoc_fundus_s1337 --domain_num 4
#   ./test_one_run.sh prostate synfoc_prostate_s2024
#
# =============================================================================

set -e

if [ "$#" -lt 2 ]; then
  cat >&2 <<USAGE
Usage: $0 <dataset> <save_name> [extra args]
  dataset:    one of {fundus, prostate, MNMS, BUSI}
  save_name:  the directory name under model/<dataset>/ where the
              best_model.pth lives. Example: v16_prostate_daep_s1337
  extra args: forwarded to test.py (e.g., --domain_num 4 --gpu 1)

See the comment header inside this script for the expected directory
layout. The repo must be set up like:
    <REPO_ROOT>/
    ├── code_3/   <- run from here (this script handles cd)
    ├── model/<dataset>/<save_name>/{unet,SAM}_avg_dice_best_model.pth
    ├── data/<dataset>/
    └── checkpoints/medsam_vit_b.pth  (only for SAM-branch eval)
USAGE
  exit 1
fi

DATASET="$1"
SAVE_NAME="$2"
shift 2

# Resolve script location -> we expect to be in the same dir as test.py.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -f "$SCRIPT_DIR/test.py" ]; then
  echo "ERROR: $SCRIPT_DIR/test.py not found." >&2
  echo "This wrapper must live alongside test.py inside code_3/." >&2
  exit 1
fi

# Pre-flight: warn (not abort) if the expected checkpoint isn't where
# test.py will look. test.py will raise a clearer FileNotFoundError if
# it really isn't there.
EXPECTED_CKPT="$SCRIPT_DIR/../model/$DATASET/$SAVE_NAME/unet_avg_dice_best_model.pth"
if [ ! -f "$EXPECTED_CKPT" ]; then
  echo "WARNING: expected checkpoint not found at:" >&2
  echo "    $EXPECTED_CKPT" >&2
  echo "test.py will fail if you proceed and this file is missing." >&2
  echo "(Continuing in case test.py finds it elsewhere or you're testing the SAM branch.)" >&2
  echo "" >&2
fi

cd "$SCRIPT_DIR"
echo "[test_one_run] Working dir: $(pwd)"
echo "[test_one_run] dataset=$DATASET  save_name=$SAVE_NAME"
echo "[test_one_run] extra args: $*"
echo ""

# Note: test.py reads --overwrite to allow re-running into an existing
# snapshot dir. We always pass it because we're testing a pre-trained
# checkpoint, not training a new one.
python test.py \
  --dataset "$DATASET" \
  --save_name "$SAVE_NAME" \
  --overwrite \
  --eval True \
  "$@"

echo ""
echo "[test_one_run] Output log: $SCRIPT_DIR/../model/$DATASET/$SAVE_NAME/log.txt"
