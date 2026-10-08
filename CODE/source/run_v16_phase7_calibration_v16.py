#!/usr/bin/env python3
# =============================================================================
# Phase 7 calibration probe — v16, rebuilt from scratch.
# =============================================================================
#
# Tests the §V.C hypothesis that pseudo-label uncertainty is miscalibrated
# on BIDMC by comparing ECE / Brier / AURC / corr-with-error between BIDMC
# (the failure case) and HK (a non-failure Prostate vendor).
#
# Why a from-scratch rebuild was needed:
#   v14 and v15 of this script both passed `normal_toTensor=None` to
#   ProstateSegmentation, with a comment claiming "the dataloader uses
#   its own built-in test-time preprocessing." That comment was wrong.
#   At test time, dataloaders/dataloader.py:ProstateSegmentation.__getitem__
#   calls `anco_sample = self.normal_toTensor(anco_sample)` unconditionally.
#   With `None`, this is TypeError: 'NoneType' object is not callable at
#   the first item — so neither v14 nor v15 ever produced a batch.
#
# What v16 does differently:
#
#   1. CONSTRUCTS normal_toTensor CORRECTLY.
#      Uses dataloaders.custom_transforms.Normalize_tf and
#      dataloaders.custom_transforms.ToTensor(unet_size=patch_size) —
#      the exact same composition the training script builds at
#      train_fourierfuse_v16.py:668. This is what produces the
#      `unet_size_img` key the model forward reads.
#
#   2. PROBES ALL THREE DIRICHLET UNCERTAINTIES.
#      The model forward returns `U_ale` and `U_epi` directly. `U_dom`
#      is NOT in the forward output — it's computed by
#      `model.prototype_bank.compute_u_dom(bottleneck)` after a
#      `return_features=True` forward pass. v15 expected `out['U_dom']`
#      and silently dropped U_dom via its `if uk in out` filter, so
#      U_dom was never actually probed in v15. v16 fixes this by
#      explicitly calling compute_u_dom on the bottleneck.
#
#   3. MATCHES TRAINING'S TEST LOOP MASK CONVENTION.
#      train_fourierfuse_v16.py:376 does `mask = mask.eq(0).long()` for
#      Prostate. After this flip: original 0 → 1 (background),
#      original non-zero → 0 (prostate). The model's class-0 output
#      head is therefore the prostate-positive channel. The probe
#      reports per-pixel Dice on class-0.
#
#   4. ROBUST DECISION RULE (graded).
#      Reports STRONG / PARTIAL / WEAK / NULL support for the
#      miscalibration hypothesis, with per-metric BIDMC-vs-HK gaps and
#      bootstrap CIs. Manuscript action is mechanical given the verdict.
#
#   5. NUMPY 2.0+ COMPATIBLE.
#      np.trapz was removed in NumPy 2.0 in favor of np.trapezoid; v16
#      uses a fallback so both old and new NumPy work.
#
#   6. PRE-FLIGHT SANITY CHECKS.
#      Before running the calibration metrics, the script:
#       - verifies the checkpoint loads and produces the expected output
#         keys ('prob', 'U_epi', 'U_ale');
#       - runs one mini-batch through the training-time test loop and
#         compares the macro-Dice against an `--expected_dice` argument
#         (with a clear, configurable tolerance); failure halts the run
#         before producing any calibration numbers.
#      This protects against the kind of silent failure that v13/v14
#      had (vacuous Dice = 1.0 on background slices was hiding garbage
#      preds).
#
# Usage:
#
#   python run_v16_phase7_calibration_v16.py \
#     --checkpoint  ../model/prostate/.../v16_prostate_daep_s1337/best_model.pth \
#     --data_root   ../../data/ProstateSlice \
#     --output_dir  ./phase7_v16_out \
#     --gpu 0
#
# Optional:
#   --threshold_checkpoint   path to a threshold-variant checkpoint;
#                            adds a max-softmax-confidence baseline probe
#                            for direct comparison with the evidential head.
#   --expected_dice_BIDMC    expected BIDMC Dice on this checkpoint
#                            (default: 78.3, from v9 manuscript)
#   --expected_dice_HK       expected HK Dice (default: 88.3)
#   --dice_tolerance         absolute Dice tolerance for parity check
#                            (default: 2.5)
#   --skip_parity            skip parity check (use only if you've
#                            verified the model is producing reasonable
#                            preds and want raw metrics anyway).
#
# Outputs (all in --output_dir):
#   - calibration_probe_v16_summary.json
#   - reliability_diagrams.pdf
#   - risk_coverage.pdf
#
# Smoke test the math without GPU:
#   python test_phase7_v16_metrics.py
# =============================================================================

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, '.')


# --------------------------------------------------------------------------
# Pure-numpy metric primitives (kept identical to test_phase7_v16_metrics.py)
# --------------------------------------------------------------------------

def expected_calibration_error(confidence, correct, n_bins=15):
    """Standard ECE with equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidence)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidence >= lo) & (
            confidence < hi if i < n_bins - 1 else confidence <= hi)
        cnt = int(in_bin.sum())
        if cnt > 0:
            acc = float(correct[in_bin].mean())
            conf = float(confidence[in_bin].mean())
            ece += (cnt / n) * abs(acc - conf)
            bin_stats.append({'lo': float(lo), 'hi': float(hi),
                              'count': cnt, 'acc': acc, 'conf': conf})
        else:
            bin_stats.append({'lo': float(lo), 'hi': float(hi),
                              'count': 0, 'acc': None, 'conf': None})
    return ece, bin_stats


def brier_score(prob_fg, target):
    return float(((prob_fg - target) ** 2).mean())


def risk_coverage_curve(uncertainty, error, n_points=10):
    """Sort pixels by ascending uncertainty; report error rate on the
    top-coverage*N most-confident pixels."""
    order = np.argsort(uncertainty)
    error_sorted = error[order]
    n = len(error_sorted)
    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    risks = np.zeros_like(coverages)
    for i, c in enumerate(coverages):
        k = max(1, int(round(c * n)))
        risks[i] = float(error_sorted[:k].mean())
    # NumPy 2.0 renamed trapz -> trapezoid.
    trapz_fn = getattr(np, 'trapezoid', None) or np.trapz
    aurc = float(trapz_fn(risks, coverages))
    return coverages.tolist(), risks.tolist(), aurc


def safe_corr(x, y):
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_metric(per_slice_data, compute_fn, n_boot=1000, seed=1337):
    """1000-resample bootstrap OVER SLICES (preserves within-slice
    spatial correlation that a pixel-level bootstrap would destroy)."""
    point = compute_fn(per_slice_data)
    if n_boot <= 0 or len(per_slice_data) == 0:
        return point, float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    n = len(per_slice_data)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resample = [per_slice_data[i] for i in idx]
        try:
            v = compute_fn(resample)
            if not np.isnan(v):
                estimates.append(v)
        except Exception:
            pass
    if not estimates:
        return point, float('nan'), float('nan')
    arr = np.asarray(estimates)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# --------------------------------------------------------------------------
# Build the EXACT training-time test transform.
# This mirrors train_fourierfuse_v16.py:668 line-for-line. If you edit
# training's transform, edit this too.
# --------------------------------------------------------------------------

def build_test_transform(patch_size):
    """Return the same `normal_toTensor` composition the training
    script uses at test time. Crucially this populates
    sample['unet_size_img']."""
    from dataloaders import custom_transforms as tr
    return transforms.Compose([
        tr.Normalize_tf(dataRange=[0, 1]),
        tr.ToTensor(low_res=0, unet_size=patch_size),
    ])


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model(ckpt_path, n_classes=1, num_domains=6, use_teacher=True):
    """Load FourierFuseNet from a training checkpoint. Returns
    (model, train_args_dict_or_None)."""
    from fourierfuse_v16_modules import FourierFuseNet
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model = FourierFuseNet(
        in_channels=1,
        n_classes=n_classes,
        base_features=64,
        bilinear=False,
        num_domains=num_domains,
        num_protos_per_domain=2,
        proto_ema_decay=0.99,
    ).cuda()
    sd_key = ('ema_state_dict' if (use_teacher and 'ema_state_dict' in ckpt)
              else 'state_dict')
    if sd_key not in ckpt:
        # Some checkpoints save directly as a dict of tensors.
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            model.load_state_dict(ckpt, strict=True)
            print(f"[phase7-v16] Loaded raw state_dict from {ckpt_path}")
            return model, None
        raise KeyError(
            f"Checkpoint at {ckpt_path} has no '{sd_key}' key. "
            f"Available keys: {list(ckpt.keys())}"
        )
    model.load_state_dict(ckpt[sd_key], strict=True)
    model.eval()

    train_args = ckpt.get('args', None)
    if train_args is not None and hasattr(train_args, '__dict__'):
        train_args = vars(train_args)
    print(f"[phase7-v16] Loaded {sd_key} from {ckpt_path}")
    if train_args:
        print(f"[phase7-v16] Training args: dataset={train_args.get('dataset')}, "
              f"lb_domain={train_args.get('lb_domain')}, "
              f"seed={train_args.get('seed')}")
    return model, train_args


def verify_model_interface(model):
    """Run one tiny forward pass to verify the output dict has the
    keys we'll need downstream."""
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 384, 384, device='cuda')
        out = model(dummy, return_features=True)
    expected = {'prob', 'U_ale', 'U_epi', 'logits', 'alpha', 'bottleneck'}
    missing = expected - set(out.keys())
    if missing:
        raise RuntimeError(
            f"Model forward(return_features=True) is missing keys {missing}. "
            f"Available: {list(out.keys())}. "
            f"Verify the checkpoint was trained with the DAEP-variant model "
            f"and that fourierfuse_v16_modules.py matches the training repo."
        )
    print(f"[phase7-v16] Model interface OK; output keys: {sorted(out.keys())}")


# --------------------------------------------------------------------------
# Build the Prostate test dataloader.
# --------------------------------------------------------------------------

def build_test_loader(data_root, domain_id, patch_size):
    from dataloaders.dataloader import ProstateSegmentation
    normal_toTensor = build_test_transform(patch_size)
    ds = ProstateSegmentation(
        base_dir=data_root,
        phase='test',
        splitid=-1,                 # -1 = use all available slices
        domain=[domain_id],
        weak_transform=None,
        strong_tranform=None,       # note: SIC, the dataloader uses this typo
        normal_toTensor=normal_toTensor,
        selected_idxs=None,
        img_size=patch_size,
        is_RGB=False,
    )
    print(f"[phase7-v16] Test loader for domain {domain_id}: {len(ds)} slices")
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


# --------------------------------------------------------------------------
# Probe one vendor, all three Dirichlet uncertainties.
# --------------------------------------------------------------------------

def probe_one_vendor(model, loader, vendor_name, expected_dice,
                     dice_tolerance, skip_parity):
    """Walk the loader once, collect per-slice signals, return a dict
    of (point, CI) for every metric of interest."""
    slice_dices_all = []
    slice_dices_fg = []
    per_slice_data = []

    # Diagnostic counters.
    n_no_proto_warning_emitted = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # ---- Sanity: dataloader must produce unet_size_img ----
            if 'unet_size_img' not in batch:
                raise KeyError(
                    "Dataloader batch is missing 'unet_size_img'. The training "
                    "script reads sample['unet_size_img'] at the test() loop. "
                    "This means build_test_transform() didn't produce the key, "
                    "which can happen if the imported custom_transforms.ToTensor "
                    "differs from the training-time version. "
                    f"Available keys: {list(batch.keys())}"
                )

            data = batch['unet_size_img'].cuda()           # [B, 1, H, W]
            mask = batch['label']                          # [B, H, W] or [B, 1, H, W]
            if mask.ndim == 4:
                mask = mask.squeeze(1)
            # Match training test loop: prostate uses mask.eq(0).long().
            # After this flip:
            #   class 0 (prostate-positive) = where original mask was nonzero
            #   class 1 (background)        = where original mask was zero
            mask = mask.eq(0).long().cuda()

            if batch_idx == 0:
                print(f"[phase7-v16] {vendor_name}: first batch — data "
                      f"{tuple(data.shape)} [{float(data.min()):.2f}, "
                      f"{float(data.max()):.2f}]; mask {tuple(mask.shape)} "
                      f"uniques={torch.unique(mask).tolist()}")

            out = model(data, return_features=True)
            prob = out['prob']                              # [B, K, H, W]
            u_ale = out['U_ale']                            # [B, 1, H, W]
            u_epi = out['U_epi']                            # [B, 1, H, W]
            bottleneck = out['bottleneck']                  # [B, C, H', W']

            # U_dom is per-image; broadcast to [B, 1, H, W] for per-pixel
            # comparison with U_ale and U_epi.
            try:
                u_dom_img = model.prototype_bank.compute_u_dom(bottleneck)
                # compute_u_dom returns [B]. Broadcast to pixel grid.
                u_dom_map = u_dom_img.view(-1, 1, 1, 1).expand(
                    -1, 1, data.shape[-2], data.shape[-1])
            except Exception as e:
                if not n_no_proto_warning_emitted:
                    print(f"[phase7-v16] WARNING: compute_u_dom failed: {e}; "
                          f"U_dom will be skipped for this vendor.")
                    n_no_proto_warning_emitted = True
                u_dom_map = None

            pred = prob.argmax(dim=1)                       # [B, H, W]
            # Class 0 = prostate. So foreground probability is prob[:, 0].
            prob_fg = prob[:, 0]                            # [B, H, W]

            # Per-slice Dice on class-0 (prostate). Matches the training-loop
            # convention in train_fourierfuse_v16.py:402-403.
            for j in range(pred.shape[0]):
                pred_j = (pred[j] == 0).float()
                mask_j = (mask[j] == 0).float()
                p_sum, m_sum = pred_j.sum().item(), mask_j.sum().item()
                inter = (pred_j * mask_j).sum().item()
                if p_sum == 0 and m_sum == 0:
                    slice_dices_all.append(1.0)
                    has_fg = False
                else:
                    d = (2 * inter + 1e-6) / (p_sum + m_sum + 1e-6)
                    slice_dices_all.append(d)
                    slice_dices_fg.append(d)
                    has_fg = True

                # Calibration: aggregate over pixels in the foreground union
                # (either GT or pred says prostate). Storing per-slice for
                # slice-level bootstrap.
                fg_gt = (mask[j] == 0)
                fg_pr = (pred[j] == 0)
                fg = (fg_gt | fg_pr)
                if not fg.any():
                    continue
                slice_signals = {
                    'has_fg': has_fg,
                    'prob_fg':      prob_fg[j][fg].cpu().numpy().reshape(-1).astype(np.float32),
                    'target':       (mask[j] == 0).float()[fg].cpu().numpy().reshape(-1).astype(np.float32),
                    'pred_correct': (pred[j] == mask[j])[fg].cpu().numpy().reshape(-1).astype(np.float32),
                    'U_ale':        u_ale[j].squeeze(0)[fg].cpu().numpy().reshape(-1).astype(np.float32),
                    'U_epi':        u_epi[j].squeeze(0)[fg].cpu().numpy().reshape(-1).astype(np.float32),
                }
                if u_dom_map is not None:
                    slice_signals['U_dom'] = u_dom_map[j].squeeze(0)[fg].cpu().numpy().reshape(-1).astype(np.float32)
                per_slice_data.append(slice_signals)

    n_total = len(slice_dices_all)
    n_fg = len(slice_dices_fg)
    dice_all = (float(np.mean(slice_dices_all)) * 100.0) if slice_dices_all else float('nan')
    dice_fg = (float(np.mean(slice_dices_fg)) * 100.0) if slice_dices_fg else float('nan')
    print(f"[phase7-v16] {vendor_name}: Dice all={dice_all:.2f}% ({n_total} slices), "
          f"fg={dice_fg:.2f}% ({n_fg} slices)")

    # ---- Parity check ----
    if not skip_parity and expected_dice is not None:
        if (abs(dice_all - expected_dice) > dice_tolerance
                and abs(dice_fg - expected_dice) > dice_tolerance):
            raise RuntimeError(
                f"Parity failure on {vendor_name}: expected ≈ {expected_dice:.2f}%, "
                f"got all={dice_all:.2f}% / fg={dice_fg:.2f}%. "
                f"Calibration metrics suppressed (likely model-loading or "
                f"preprocessing mismatch). Pass --skip_parity to override."
            )

    # Identify which uncertainty signals are present across ALL slices.
    available_signals = [k for k in ('U_ale', 'U_epi', 'U_dom')
                         if all(k in s for s in per_slice_data)]
    print(f"[phase7-v16] {vendor_name}: probing signals = {available_signals}")

    # ---- Metric computation, with slice-bootstrap CI ----
    def concat(field, slices):
        return np.concatenate([s[field] for s in slices])

    def compute_all_metrics(slices):
        if not slices:
            return {}
        prob_fg_arr = concat('prob_fg', slices)
        target_arr = concat('target', slices)
        correct_arr = concat('pred_correct', slices)
        out = {'brier': brier_score(prob_fg_arr, target_arr)}
        for sig in available_signals:
            u_arr = concat(sig, slices)
            conf_arr = 1.0 - u_arr
            err_arr = 1.0 - correct_arr
            ece, _ = expected_calibration_error(conf_arr, correct_arr, n_bins=15)
            _, _, aurc = risk_coverage_curve(u_arr, err_arr, n_points=10)
            corr = safe_corr(u_arr, err_arr)
            out[f'ece_{sig}'] = ece
            out[f'aurc_{sig}'] = aurc
            out[f'corr_{sig}_error'] = corr
        return out

    # Point estimates.
    point = compute_all_metrics(per_slice_data)
    # Bootstrap.
    bootstrap = {}
    for k in point:
        _, lo, hi = bootstrap_metric(
            per_slice_data,
            lambda s, _k=k: compute_all_metrics(s).get(_k, float('nan')),
            n_boot=1000, seed=1337,
        )
        bootstrap[k] = {'point': point[k], 'lo95': lo, 'hi95': hi}

    # Reliability bin stats (point estimate only — bootstrap on bin
    # counts is messy because empty bins differ across resamples).
    bin_stats_by_signal = {}
    correct_arr = concat('pred_correct', per_slice_data)
    for sig in available_signals:
        u_arr = concat(sig, per_slice_data)
        conf_arr = 1.0 - u_arr
        _, bs = expected_calibration_error(conf_arr, correct_arr, n_bins=15)
        bin_stats_by_signal[sig] = bs

    # Risk-coverage curves at high resolution for plotting.
    rc_curves = {}
    err_arr = 1.0 - correct_arr
    for sig in available_signals:
        u_arr = concat(sig, per_slice_data)
        covs, risks, aurc = risk_coverage_curve(u_arr, err_arr, n_points=20)
        rc_curves[sig] = {'coverages': covs, 'risks': risks, 'aurc': aurc}

    return {
        'dice_all_slices': dice_all,
        'dice_fg_only': dice_fg,
        'n_total_slices': n_total,
        'n_fg_slices': n_fg,
        'n_foreground_pixels': int(sum(len(s['target']) for s in per_slice_data)),
        'available_signals': available_signals,
        'metrics': bootstrap,
        'bin_stats': bin_stats_by_signal,
        'rc_curves': rc_curves,
    }


# --------------------------------------------------------------------------
# Optional threshold-variant probe (max-softmax confidence)
# --------------------------------------------------------------------------

def probe_threshold_variant(model, loader, vendor_name, expected_dice,
                             dice_tolerance, skip_parity):
    """Same as probe_one_vendor but with max-softmax probability as the
    confidence signal. The model is assumed to expose `prob` in its
    output dict regardless of variant (since the DAEP model's
    Dirichlet mean and the threshold model's softmax both live there)."""
    slice_dices_all, slice_dices_fg = [], []
    per_slice_data = []

    with torch.no_grad():
        for batch in loader:
            data = batch['unet_size_img'].cuda()
            mask = batch['label']
            if mask.ndim == 4:
                mask = mask.squeeze(1)
            mask = mask.eq(0).long().cuda()

            out = model(data, return_features=False)
            prob = out.get('prob') if isinstance(out, dict) else torch.softmax(out, dim=1)
            if prob is None:
                raise RuntimeError(
                    "Threshold-variant model did not expose 'prob' in its "
                    "output dict; cannot compute max-softmax confidence."
                )
            pred = prob.argmax(dim=1)
            prob_fg = prob[:, 0]
            max_p = prob.max(dim=1).values                # max softmax confidence

            for j in range(pred.shape[0]):
                pred_j = (pred[j] == 0).float()
                mask_j = (mask[j] == 0).float()
                p_sum, m_sum = pred_j.sum().item(), mask_j.sum().item()
                inter = (pred_j * mask_j).sum().item()
                if p_sum == 0 and m_sum == 0:
                    slice_dices_all.append(1.0); has_fg = False
                else:
                    d = (2 * inter + 1e-6) / (p_sum + m_sum + 1e-6)
                    slice_dices_all.append(d)
                    slice_dices_fg.append(d)
                    has_fg = True

                fg = ((mask[j] == 0) | (pred[j] == 0))
                if not fg.any():
                    continue
                per_slice_data.append({
                    'prob_fg':      prob_fg[j][fg].cpu().numpy().reshape(-1).astype(np.float32),
                    'target':       (mask[j] == 0).float()[fg].cpu().numpy().reshape(-1).astype(np.float32),
                    'pred_correct': (pred[j] == mask[j])[fg].cpu().numpy().reshape(-1).astype(np.float32),
                    'softmax_conf': max_p[j][fg].cpu().numpy().reshape(-1).astype(np.float32),
                })

    n_total = len(slice_dices_all)
    n_fg = len(slice_dices_fg)
    dice_all = (float(np.mean(slice_dices_all)) * 100.0) if slice_dices_all else float('nan')
    dice_fg = (float(np.mean(slice_dices_fg)) * 100.0) if slice_dices_fg else float('nan')
    print(f"[phase7-v16] {vendor_name} (threshold): Dice all={dice_all:.2f}% "
          f"fg={dice_fg:.2f}%")

    if not skip_parity and expected_dice is not None:
        if (abs(dice_all - expected_dice) > dice_tolerance
                and abs(dice_fg - expected_dice) > dice_tolerance):
            print(f"[phase7-v16] WARNING: parity drift on {vendor_name}-threshold: "
                  f"expected ≈ {expected_dice:.2f}, got all={dice_all:.2f} / "
                  f"fg={dice_fg:.2f}. Continuing anyway (non-fatal for the "
                  f"threshold baseline).")

    def concat(field, slices):
        return np.concatenate([s[field] for s in slices])

    def compute_all_metrics(slices):
        if not slices:
            return {}
        prob_fg_arr = concat('prob_fg', slices)
        target_arr = concat('target', slices)
        correct_arr = concat('pred_correct', slices)
        conf_arr = concat('softmax_conf', slices)
        err_arr = 1.0 - correct_arr
        u_arr = 1.0 - conf_arr
        ece, _ = expected_calibration_error(conf_arr, correct_arr, n_bins=15)
        _, _, aurc = risk_coverage_curve(u_arr, err_arr, n_points=10)
        return {
            'brier_softmax_threshold': brier_score(prob_fg_arr, target_arr),
            'ece_softmax_conf': ece,
            'aurc_softmax_unc': aurc,
            'corr_softmax_unc_error': safe_corr(u_arr, err_arr),
        }

    point = compute_all_metrics(per_slice_data)
    bootstrap = {}
    for k in point:
        _, lo, hi = bootstrap_metric(
            per_slice_data,
            lambda s, _k=k: compute_all_metrics(s).get(_k, float('nan')),
            n_boot=1000, seed=1337,
        )
        bootstrap[k] = {'point': point[k], 'lo95': lo, 'hi95': hi}

    return {
        'dice_all_slices': dice_all,
        'dice_fg_only': dice_fg,
        'n_total_slices': n_total,
        'n_fg_slices': n_fg,
        'metrics': bootstrap,
    }


# --------------------------------------------------------------------------
# Graded decision rule
# --------------------------------------------------------------------------

def graded_verdict(daep_results):
    """Per-metric BIDMC-vs-HK comparisons. Each metric has a
    "hypothesis-supporting direction" (BIDMC worse than HK). Count the
    metrics that agree with the hypothesis AND whose CIs don't overlap."""
    sb = daep_results.get('BIDMC')
    sh = daep_results.get('HK')
    if not sb or not sh:
        return {'verdict': 'INCOMPLETE',
                'reason': 'Probe did not complete on both vendors.'}

    higher_is_worse = []
    higher_is_better = []
    for sig in ('U_ale', 'U_epi', 'U_dom'):
        higher_is_worse.append(f'ece_{sig}')
        higher_is_worse.append(f'aurc_{sig}')
        higher_is_better.append(f'corr_{sig}_error')
    higher_is_worse.append('brier')

    rows = []
    for k in higher_is_worse + higher_is_better:
        if k not in sb['metrics'] or k not in sh['metrics']:
            continue
        b = sb['metrics'][k]; h = sh['metrics'][k]
        gap = b['point'] - h['point']
        # CI non-overlap: a conservative, paired-comparison-free test.
        ci_b_above_h = b['lo95'] > h['hi95']
        ci_h_above_b = h['lo95'] > b['hi95']
        ci_non_overlap = ci_b_above_h or ci_h_above_b
        if k in higher_is_worse:
            supports = (gap > 0) and ci_b_above_h
        else:
            supports = (gap < 0) and ci_h_above_b
        rows.append({
            'metric': k, 'bidmc_point': b['point'], 'hk_point': h['point'],
            'bidmc_ci': [b['lo95'], b['hi95']],
            'hk_ci': [h['lo95'], h['hi95']],
            'gap': gap,
            'ci_non_overlap': ci_non_overlap,
            'supports_hypothesis': supports,
        })

    n = len(rows)
    if n == 0:
        return {'verdict': 'INCOMPLETE', 'rows': rows}
    n_supp = sum(1 for r in rows if r['supports_hypothesis'])
    frac = n_supp / n

    if frac >= 0.85:
        verdict = 'STRONG SUPPORT'
        action = ("§V.C: change 'consistent with — but does not directly "
                  "establish' to 'directly supported by a calibration probe', "
                  "with specific BIDMC-vs-HK metric gaps cited.")
    elif frac >= 0.50:
        verdict = 'PARTIAL SUPPORT'
        action = ("§V.C: keep current 'consistent with' wording, append a "
                  "sentence naming the subset of metrics for which the "
                  "calibration gap was statistically detectable on BIDMC.")
    elif frac > 0:
        verdict = 'WEAK SUPPORT'
        action = ("§V.E: mention as a second-order observation; do not "
                  "modify §V.C — the binary 'consistent with' framing "
                  "already accommodates this.")
    else:
        verdict = 'NULL'
        action = ("§V.C: edit to acknowledge the null. The miscalibration "
                  "hypothesis is not supported by direct measurement; the "
                  "BIDMC pattern remains unexplained and is a candidate "
                  "for follow-on investigation.")

    return {
        'verdict': verdict,
        'n_supports': n_supp,
        'n_total': n,
        'fraction': frac,
        'manuscript_action': action,
        'rows': rows,
    }


# --------------------------------------------------------------------------
# Plotting (matplotlib optional; non-fatal if unavailable)
# --------------------------------------------------------------------------

def make_plots(daep_results, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[phase7-v16] matplotlib not installed; skipping plots.")
        return

    sb = daep_results.get('BIDMC'); sh = daep_results.get('HK')
    if not (sb and sh):
        return

    # Reliability diagrams
    signals = sorted(set(sb.get('bin_stats', {}).keys())
                     & set(sh.get('bin_stats', {}).keys()))
    if signals:
        n = len(signals)
        fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7), sharey=True,
                                 squeeze=False)
        for j, sig in enumerate(signals):
            for i, (vname, vres) in enumerate([('HK', sh), ('BIDMC', sb)]):
                ax = axes[i, j]
                stats = vres['bin_stats'][sig]
                confs = [s['conf'] for s in stats if s['conf'] is not None]
                accs = [s['acc'] for s in stats if s['acc'] is not None]
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                if confs:
                    ax.bar(confs, accs, width=0.04, alpha=0.6, edgecolor='k')
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xlabel(f'1 − {sig}')
                ax.set_title(f'{vname} · {sig}')
                if j == 0:
                    ax.set_ylabel('accuracy')
        fig.tight_layout()
        out = os.path.join(output_dir, 'reliability_diagrams.pdf')
        fig.savefig(out, dpi=200); plt.close(fig)
        print(f"[phase7-v16] Saved {out}")

    # Risk-coverage curves
    signals = sorted(set(sb.get('rc_curves', {}).keys())
                     & set(sh.get('rc_curves', {}).keys()))
    if signals:
        n = len(signals)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True,
                                 squeeze=False)
        for j, sig in enumerate(signals):
            ax = axes[0, j]
            rc_h = sh['rc_curves'][sig]; rc_b = sb['rc_curves'][sig]
            ax.plot(rc_h['coverages'], rc_h['risks'], 'o-',
                    label=f"HK (AURC={rc_h['aurc']:.3f})")
            ax.plot(rc_b['coverages'], rc_b['risks'], 's-',
                    label=f"BIDMC (AURC={rc_b['aurc']:.3f})")
            ax.set_xlabel('coverage')
            ax.set_title(sig)
            ax.legend(loc='upper left', fontsize=8)
            if j == 0:
                ax.set_ylabel('error rate on covered pixels')
        fig.tight_layout()
        out = os.path.join(output_dir, 'risk_coverage.pdf')
        fig.savefig(out, dpi=200); plt.close(fig)
        print(f"[phase7-v16] Saved {out}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='DAEP-variant checkpoint to probe.')
    parser.add_argument('--threshold_checkpoint', type=str, default=None,
                        help='Optional threshold-variant checkpoint for a '
                             'max-softmax baseline.')
    parser.add_argument('--data_root', type=str,
                        default='../../data/ProstateSlice')
    parser.add_argument('--output_dir', type=str,
                        default='./phase7_v16_out')
    parser.add_argument('--patch_size', type=int, default=384)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dice_tolerance', type=float, default=2.5)
    parser.add_argument('--expected_dice_BIDMC', type=float, default=78.3)
    parser.add_argument('--expected_dice_HK', type=float, default=88.3)
    parser.add_argument('--skip_parity', action='store_true')
    parser.add_argument('--use_student', action='store_true',
                        help='Load state_dict instead of ema_state_dict. '
                             'Default: ema_state_dict (matches training '
                             'test loop).')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[phase7-v16] Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[phase7-v16] PyTorch {torch.__version__}, "
          f"CUDA {torch.cuda.is_available()}, "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
    print(f"[phase7-v16] Output → {args.output_dir}")

    # =========================
    # DAEP probe
    # =========================
    print("\n" + "=" * 64)
    print("DAEP probe (full Dirichlet uncertainties)")
    print("=" * 64)
    model_daep, _ = load_model(args.checkpoint, n_classes=1, num_domains=6,
                                use_teacher=not args.use_student)
    verify_model_interface(model_daep)

    expected = {'BIDMC': args.expected_dice_BIDMC, 'HK': args.expected_dice_HK}
    domains = {'BIDMC': 1, 'HK': 3}    # Prostate vendor → domain-id mapping

    daep_results = {}
    for vname in ('HK', 'BIDMC'):
        print(f"\n--- {vname} (DAEP) ---")
        try:
            loader = build_test_loader(args.data_root, domains[vname],
                                        args.patch_size)
            daep_results[vname] = probe_one_vendor(
                model_daep, loader, vname,
                expected[vname], args.dice_tolerance, args.skip_parity,
            )
        except Exception as e:
            print(f"[phase7-v16] {vname} probe failed: {e}")
            traceback.print_exc()
            raise

    del model_daep; torch.cuda.empty_cache()

    # =========================
    # Threshold probe (optional)
    # =========================
    threshold_results = {}
    if args.threshold_checkpoint:
        print("\n" + "=" * 64)
        print("Threshold probe (max-softmax confidence)")
        print("=" * 64)
        model_thr, _ = load_model(args.threshold_checkpoint, n_classes=1,
                                    num_domains=6,
                                    use_teacher=not args.use_student)
        verify_model_interface(model_thr)  # threshold model still has 'prob'
        for vname in ('HK', 'BIDMC'):
            print(f"\n--- {vname} (threshold) ---")
            loader = build_test_loader(args.data_root, domains[vname],
                                        args.patch_size)
            threshold_results[vname] = probe_threshold_variant(
                model_thr, loader, vname, expected[vname],
                args.dice_tolerance, args.skip_parity,
            )
        del model_thr; torch.cuda.empty_cache()

    # =========================
    # Save + decision
    # =========================
    full = {
        'daep': daep_results,
        'threshold': threshold_results,
        'cli_args': vars(args),
        'verdict': graded_verdict(daep_results),
    }
    out_json = os.path.join(args.output_dir, 'calibration_probe_v16_summary.json')

    def to_jsonable(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return str(o)

    with open(out_json, 'w') as f:
        json.dump(full, f, indent=2, default=to_jsonable)
    print(f"\n[phase7-v16] Summary → {out_json}")

    make_plots(daep_results, args.output_dir)

    # Print the decision.
    v = full['verdict']
    print("\n" + "=" * 64)
    print("DECISION (graded)")
    print("=" * 64)
    print(f"  Verdict:                {v.get('verdict', '?')}")
    if 'fraction' in v:
        print(f"  Metrics supporting:     {v['n_supports']}/{v['n_total']} "
              f"({v['fraction']:.0%})")
    if 'manuscript_action' in v:
        print(f"  → Manuscript action:    {v['manuscript_action']}")
    print("=" * 64)


if __name__ == '__main__':
    main()
