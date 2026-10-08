#!/usr/bin/env python3
"""
diag_uncertainty_harness_v18.py
============================================================================
BSPC-D-26-12287 revision: ONE inference-only harness that measures every
uncertainty signal the reviewer asked about, on saved checkpoints, per test
domain, and writes machine-readable JSON (nothing is transcribed by hand).

What it computes, per (checkpoint, test domain, signal)
--------------------------------------------------------
  signals : conf (max prob), u_epi (vacuity; 1-conf for a softmax head),
            u_ale, entropy; with --mc_T>0 on a dropout model: mc_entropy, mc_mi;
            with --ensemble_ckpts: ens_entropy, ens_mi, ens_var
  A. all-pixel error-ranking AUROC and central-80% range         (Tables 6/7)
  B. boundary-band AUROC + in-band central-80% range for
       - Euclidean-distance bands  r px in --bands            (paper: +-5 px)
       - morphological (Chebyshev) bands, same radii          (R1 #7)
       - object-size-relative bands  r = f * sqrt(area/pi)    (R1 #7)
       - millimetre bands when a spacing index is given       (R1 #7, D-1)
     around the GT boundary (default) or the predicted / union boundary
  C. error distribution vs distance to the GT boundary            (R1 #7)
  D. oracle band ceiling: Dice as-is vs band replaced by GT       (Fig. 5)
  E. foreground-threshold sweep of Dice (binary datasets)         (Fig. 4)
  F. post-hoc calibration of the teacher fitted ONLY on the run's labelled
     source split (temperature scaling on the predictive mean; isotonic
     regression of confidence -> accuracy): ECE before/after, AUROC and
     range of the calibrated confidence                          (R1 #8)

Conventions (pinned to the submitted paper's harness):
  * teacher = 'ema_state_dict' of best_student.pth (override --which)
  * everything on the patch grid the model predicts on; GT resized nearest
  * AUROC by rank statistic on a fixed random subsample of 3e5 pixels
    (RandomState(0)) -- identical convention to Tables 6-8
  * error = (argmax != GT) after the per-dataset label remap

Usage
-----
  python diag_uncertainty_harness_v18.py --dataset prostate \
      --ckpt ../model/prostate/train_fourierfuse_v16/adaptive_cmp_prostate_fixmatch_s1337/best_student.pth \
      --out diag_out/prostate_fixmatch_s1337.json [--spacing_index prostate_spacing_index.json]
  python diag_uncertainty_harness_v18.py --selftest      # CPU, synthetic data

Outputs one JSON with a 'per_domain' dict and a 'meta' block (ckpt sha256,
config, harness sha256, timestamp) so every number is provenance-tagged.
============================================================================
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import argparse, glob, hashlib, json, math, re, sys, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Dataset constants (mirror train_fourierfuse_v16.py exactly)
# ---------------------------------------------------------------------------
LB_NUM_DEFAULT = {'prostate': 20, 'fundus': 20, 'MNMS': 5, 'BUSI': 64}   # paper protocol
DATASET_CFG = {
    "prostate": dict(num_channels=1, num_classes=1, patch_size=384, domain_num=6,
                     data_path="../../data/ProstateSlice", img_size=512,
                     domain_len=[225, 305, 136, 373, 338, 133],
                     names={1: 'BIDMC', 2: 'BMC', 3: 'HK', 4: 'I2CVB', 5: 'RUNMC', 6: 'UCL'}),
    "fundus":   dict(num_channels=3, num_classes=2, patch_size=256, domain_num=4,
                     data_path="../../data/Fundus", img_size=512,
                     domain_len=[50, 99, 320, 320], names={1: 'D1', 2: 'D2', 3: 'D3', 4: 'D4'}),
    "MNMS":     dict(num_channels=1, num_classes=3, patch_size=288, domain_num=4,
                     data_path="../../data/mnms", img_size=512,
                     domain_len=[1030, 1342, 525, 550], names={1: 'A', 2: 'B', 3: 'C', 4: 'D'}),
    "BUSI":     dict(num_channels=1, num_classes=1, patch_size=256, domain_num=2,
                     data_path="../../data/Dataset_BUSI_with_GT", img_size=512,
                     domain_len=[350, 168], names={1: 'benign', 2: 'malignant'}),
}


def remap_label(label: torch.Tensor, dataset: str) -> torch.Tensor:
    """Loader label -> model convention (0=bg, 1..K-1=fg). Same as
    measure_collapse_alldatasets.py / cadg_diagnostics.prostate_mask."""
    if dataset == "prostate":
        return label.eq(0).long()
    if dataset == "BUSI":
        return label.eq(255).long()
    if dataset == "MNMS":
        return label.long()
    lab = (label <= 128).long() * 2
    lab[label == 0] = 1
    return lab.long()


# ---------------------------------------------------------------------------
# Metrics (numpy)
# ---------------------------------------------------------------------------

def fast_auroc(scores: np.ndarray, labels: np.ndarray, max_pts: int = 300000,
               seed: int = 0) -> float:
    """Rank AUROC of `scores` for `labels` (1 = error). Fixed subsample."""
    scores = scores.ravel().astype(np.float64); labels = labels.ravel()
    if scores.shape[0] > max_pts:
        idx = np.random.RandomState(seed).choice(scores.shape[0], max_pts, replace=False)
        scores, labels = scores[idx], labels[idx]
    pos = labels == 1; neg = labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    from scipy.stats import rankdata
    ranks = rankdata(scores)  # average ranks (ties handled)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def central_range(x: np.ndarray, lo: float = 10, hi: float = 90) -> float:
    if x.size == 0:
        return float('nan')
    a, b = np.percentile(x, [lo, hi])
    return float(b - a)


def ece_score(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    conf = conf.ravel(); correct = correct.ravel().astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def dice_per_image(pred: np.ndarray, gt: np.ndarray, K: int) -> float:
    """Mean over foreground classes of 2|A n B| / (|A|+|B|); 1.0 if both empty
    (mirrors metrics.dice_coeff_* conventions on empty foreground)."""
    ds = []
    for c in range(1, K):
        a = pred == c; b = gt == c
        denom = a.sum() + b.sum()
        ds.append(1.0 if denom == 0 else 2.0 * np.logical_and(a, b).sum() / denom)
    return float(np.mean(ds))


# ---------------------------------------------------------------------------
# Boundary geometry
# ---------------------------------------------------------------------------

def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Pixels of `mask` (any label map) that touch a different label (4-conn)."""
    b = np.zeros(mask.shape, dtype=bool)
    b[:-1, :] |= mask[:-1, :] != mask[1:, :]
    b[1:, :] |= mask[1:, :] != mask[:-1, :]
    b[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    b[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return b


def edt_to_boundary(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) from every pixel to the nearest label boundary
    of `mask`; inf if the mask has no boundary (single label)."""
    b = boundary_pixels(mask)
    if not b.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return ndi.distance_transform_edt(~b).astype(np.float32)


def chebyshev_to_boundary(mask: np.ndarray) -> np.ndarray:
    """Chebyshev (8-connected morphological) distance to the label boundary:
    r iterations of a 3x3 dilation == distance <= r."""
    b = boundary_pixels(mask)
    if not b.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return ndi.distance_transform_cdt(~b, metric='chessboard').astype(np.float32)


def equivalent_radius(gt: np.ndarray) -> float:
    area = float((gt > 0).sum())
    return math.sqrt(area / math.pi) if area > 0 else float('nan')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def sha256_of(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def load_model(dataset: str, ckpt_path: str, which: str, device: str,
               override: Dict = None):
    """Build the network from the checkpoint's own config (v18) or the v16
    defaults, and load the requested weights."""
    from revision_modules import FourierFuseNetV18
    cfg = DATASET_CFG[dataset]
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    vc = dict(ck.get('v18_config', {}))
    if override:
        vc.update(override)
    net_kwargs = dict(
        in_channels=cfg['num_channels'], n_classes=cfg['num_classes'],
        base_features=int(vc.get('ff_base_features', 64)),
        bilinear=bool(vc.get('ff_bilinear', False)),
        num_domains=cfg['domain_num'],
        num_protos_per_domain=int(vc.get('proto_per_domain', 2)),
        proto_ema_decay=float(vc.get('proto_ema', 0.99)),
        head=vc.get('head', 'evidential'),
        evid_activation=vc.get('evid_activation', 'softplus'),
        evid_temperature=float(vc.get('evid_temperature', 1.0)),
        evid_prior_w=float(vc.get('evid_prior_w', 1.0)),
        evid_max_evidence=float(vc.get('evid_max_evidence', 0.0)),
        dropout_p=float(vc.get('mc_dropout_p', 0.0)),
        encoder_arch='unet',
        dropout_place=vc.get('mc_dropout_place', 'v18'),   # S06 checkpoints carry no key -> 'v18'
    )
    net2_arch = vc.get('medl_arch2', 'resunet')
    models = []
    keys = {'ema': 'ema_state_dict', 'student': 'state_dict', 'swad': 'state_dict'}
    key = keys[which]
    if key not in ck:
        raise KeyError(f"{ckpt_path} has no '{key}' (keys: {list(ck.keys())})")
    m = FourierFuseNetV18(**net_kwargs).to(device)
    missing, unexpected = m.load_state_dict(ck[key], strict=False)
    if missing or unexpected:
        print(f"[load] WARNING non-strict load: missing={missing[:5]} unexpected={unexpected[:5]}")
    m.eval(); models.append(m)
    if vc.get('method') == 'medl' and (key + '2') in ck:
        m2 = FourierFuseNetV18(**{**net_kwargs, 'encoder_arch': net2_arch}).to(device)
        m2.load_state_dict(ck[key + '2'], strict=False); m2.eval(); models.append(m2)
    meta = {'ckpt': ckpt_path, 'ckpt_sha256': sha256_of(ckpt_path), 'which': which,
            'iter': int(ck.get('iter', -1)), 'best_avg_dice': float(ck.get('best_avg_dice', float('nan'))),
            'net_kwargs': net_kwargs, 'v18_config': vc, 'n_models': len(models)}
    return models, meta


def build_test_loader(dataset: str, domain: int):
    from dataloaders.dataloader import (FundusSegmentation, ProstateSegmentation,
                                        MNMSSegmentation, BUSISegmentation)
    import dataloaders.custom_transforms as tr
    from torchvision import transforms
    from torch.utils.data import DataLoader
    DS = {"prostate": ProstateSegmentation, "fundus": FundusSegmentation,
          "MNMS": MNMSSegmentation, "BUSI": BUSISegmentation}[dataset]
    cfg = DATASET_CFG[dataset]
    ntt = transforms.Compose([tr.Normalize_tf(dataRange=[0, 1]),
                              tr.ToTensor(low_res=cfg['patch_size'], unet_size=cfg['patch_size'])])
    ds = DS(base_dir=cfg['data_path'], phase='test', splitid=-1, domain=[domain],
            normal_toTensor=ntt, img_size=cfg['img_size'])
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


def build_labelled_loader(dataset: str, lb_domain: int, lb_idxs: List[int]):
    """The run's OWN labelled source images, unaugmented (for calibration fit)."""
    from dataloaders.dataloader import (FundusSegmentation, ProstateSegmentation,
                                        MNMSSegmentation, BUSISegmentation)
    import dataloaders.custom_transforms as tr
    from torchvision import transforms
    from torch.utils.data import DataLoader
    DS = {"prostate": ProstateSegmentation, "fundus": FundusSegmentation,
          "MNMS": MNMSSegmentation, "BUSI": BUSISegmentation}[dataset]
    cfg = DATASET_CFG[dataset]
    ntt = transforms.Compose([tr.Normalize_tf(dataRange=[0, 1]),
                              tr.ToTensor(low_res=cfg['patch_size'], unet_size=cfg['patch_size'])])
    ds = DS(base_dir=cfg['data_path'], phase='train', splitid=lb_domain, domain=[lb_domain],
            selected_idxs=lb_idxs, weak_transform=None, strong_tranform=None,
            normal_toTensor=ntt, img_size=cfg['img_size'])
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_signals(models, x: torch.Tensor, mc_T: int) -> Dict[str, torch.Tensor]:
    """Returns prob [K,H,W] and a dict of [H,W] uncertainty maps (float32)."""
    from revision_modules import mc_dropout_predict
    outs = [m.forward(x, return_features=False) for m in models]
    prob = outs[0]['prob'].float()
    sig = {}
    if len(models) > 1:                       # MEDL pair / explicit ensemble
        P = torch.stack([o['prob'].float() for o in outs], 0)
        mean = P.mean(0)
        H_mean = -(mean * torch.log(mean.clamp(min=1e-10))).sum(1)
        H_each = -(P * torch.log(P.clamp(min=1e-10))).sum(2).mean(0)
        sig['ens_entropy'] = H_mean[0]
        sig['ens_mi'] = (H_mean - H_each)[0]
        sig['ens_var'] = P.max(2).values.var(0, unbiased=False)[0]
        if len(models) == 2 and all(getattr(m, 'head_type', '') == 'evidential' for m in models):
            # MEDL pair: class-aware evidential fusion uncertainty (eq.3) as a signal
            from revision_modules import caef_fuse
            _, u_f = caef_fuse(outs[0]['belief'].float(), outs[0]['U_epi'].float(),
                               outs[1]['belief'].float(), outs[1]['U_epi'].float())
            sig['caef_u'] = u_f[0, 0]
        prob = mean
    sig['conf'] = 1.0 - prob.max(1).values[0]            # deficit, so "high = uncertain"
    sig['u_epi'] = outs[0]['U_epi'].float()[0, 0]
    sig['u_ale'] = outs[0]['U_ale'].float()[0, 0]
    sig['entropy'] = -(prob * torch.log(prob.clamp(min=1e-10))).sum(1)[0]
    if mc_T > 0 and getattr(models[0], 'dropout_p', 0.0) > 0:
        mc = mc_dropout_predict(models[0], x, T=mc_T)
        sig['mc_entropy'] = mc['entropy'][0, 0]
        sig['mc_mi'] = mc['mi'][0, 0]
        sig['mc_std'] = mc['std'][0, 0]
    return {'prob': prob[0], **{k: v for k, v in sig.items()}}


# ---------------------------------------------------------------------------
# Per-domain measurement
# ---------------------------------------------------------------------------

def measure_domain(models, loader, dataset: str, K: int, args, spacing_lookup=None
                   ) -> Dict:
    dev = next(models[0].parameters()).device
    bands = [int(b) for b in args.bands.split(',')]
    rel_fracs = [float(f) for f in args.rel_fracs.split(',')]
    mm_bands = [float(m) for m in args.mm_bands.split(',')] if spacing_lookup else []
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
    binary = (K == 2)

    acc = {}          # signal -> list of arrays (all pixels)
    err_all = []
    dist_gt_all, dist_cheb_all, dist_pred_all, dist_union_all, dist_rel_all, dist_mm_all = [], [], [], [], [], []
    dice_asis, dice_oracle = [], {r: [] for r in bands}
    dice_tau = {float(t): [] for t in taus} if binary else None
    n_img = 0; n_mm = 0; seen_names = []

    for sample in loader:
        x = sample['unet_size_img'].to(dev)
        lab = sample['label']
        if lab.dim() == 4:
            lab = lab.squeeze(1)
        lab = remap_label(lab.float(), dataset)
        sig = forward_signals(models, x, args.mc_T)
        prob = sig.pop('prob')                              # [K,H,W]
        pred = prob.argmax(0).cpu().numpy()
        if lab.shape[-2:] != pred.shape:
            lab = F.interpolate(lab.unsqueeze(1).float(), size=pred.shape, mode='nearest').squeeze(1).long()
        gt = lab[0].cpu().numpy().astype(np.int64)
        err = (pred != gt)
        d_gt = edt_to_boundary(gt); d_cheb = chebyshev_to_boundary(gt)
        d_pred = edt_to_boundary(pred); d_union = np.minimum(d_gt, d_pred)
        r_eq = equivalent_radius(gt)
        d_rel = d_gt / r_eq if (r_eq == r_eq and r_eq > 0) else np.full(gt.shape, np.inf, np.float32)
        if spacing_lookup is not None:
            mmpp = spacing_lookup(sample['img_name'][0], pred.shape[-1], args._site)
            if len(seen_names) < 3:
                seen_names.append(str(sample['img_name'][0]))
            if mmpp is not None:
                d_mm = d_gt * mmpp; n_mm += 1
            else:
                d_mm = np.full(gt.shape, np.nan, np.float32)
            dist_mm_all.append(d_mm.ravel())
        for k, v in sig.items():
            acc.setdefault(k, []).append(v.cpu().numpy().ravel().astype(np.float32))
        err_all.append(err.ravel())
        dist_gt_all.append(d_gt.ravel()); dist_cheb_all.append(d_cheb.ravel())
        dist_pred_all.append(d_pred.ravel()); dist_union_all.append(d_union.ravel())
        dist_rel_all.append(d_rel.ravel())
        # D. oracle band ceiling (band around GT boundary replaced by GT)
        dice_asis.append(dice_per_image(pred, gt, K))
        for r in bands:
            p2 = pred.copy(); m = d_gt <= r; p2[m] = gt[m]
            dice_oracle[r].append(dice_per_image(p2, gt, K))
        # E. threshold sweep (binary)
        if binary:
            fg = prob[1].cpu().numpy()
            for t in taus:
                dice_tau[float(t)].append(dice_per_image((fg >= t).astype(np.int64), gt, K))
        n_img += 1

    err = np.concatenate(err_all); d_gt = np.concatenate(dist_gt_all)
    d_cheb = np.concatenate(dist_cheb_all); d_pred = np.concatenate(dist_pred_all)
    d_union = np.concatenate(dist_union_all); d_rel = np.concatenate(dist_rel_all)
    d_mm = np.concatenate(dist_mm_all) if dist_mm_all else None
    sigs = {k: np.concatenate(v) for k, v in acc.items()}

    res = {'n_images': n_img, 'n_pixels': int(err.size), 'error_rate': float(err.mean()),
           'n_images_with_mm': n_mm, 'signals': {}, 'bands_px': bands}
    if spacing_lookup is not None:
        # R-29 / E-21: a name-matching parser reports what it saw; zero matches is a hard error,
        # never a silent NaN column.
        print(f"[spacing] site {getattr(args, '_site', '?')}: mm spacing matched {n_mm}/{n_img} images; "
              f"first names seen {seen_names}; index sample keys {list(spacing_lookup.table)[:2]}")
        if n_mm == 0:
            raise SystemExit(f"[spacing] ERROR: 0/{n_img} images matched the spacing index for site "
                             f"{getattr(args, '_site', '?')} (names seen {seen_names}) -> fix the key "
                             f"contract before quoting mm bands (E-21)")
    for k, s in sigs.items():
        row = {'auroc_all': fast_auroc(s, err), 'range80_all': central_range(s),
               'median_all': float(np.median(s)), 'band': {}}
        for name, d, radii in (('edt_gt', d_gt, bands), ('cheb_gt', d_cheb, bands),
                               ('edt_pred', d_pred, bands), ('edt_union', d_union, bands),
                               ('rel_gt', d_rel, rel_fracs)):
            for r in radii:
                m = d <= r
                row['band'][f'{name}_{r}'] = {
                    'auroc': fast_auroc(s[m], err[m]) if m.sum() > 100 else float('nan'),
                    'range80': central_range(s[m]) if m.sum() > 100 else float('nan'),
                    'frac_pixels': float(m.mean()),
                    'frac_errors': float(err[m].sum() / max(err.sum(), 1)),
                    'error_rate': float(err[m].mean()) if m.any() else float('nan')}
        if d_mm is not None:
            for r in mm_bands:
                m = (d_mm <= r) & np.isfinite(d_mm)
                row['band'][f'mm_gt_{r}'] = {
                    'auroc': fast_auroc(s[m], err[m]) if m.sum() > 100 else float('nan'),
                    'range80': central_range(s[m]) if m.sum() > 100 else float('nan'),
                    'frac_pixels': float(m.mean()),
                    'frac_errors': float(err[m].sum() / max(err.sum(), 1)),
                    'error_rate': float(err[m].mean()) if m.any() else float('nan')}
        res['signals'][k] = row
    # C. error distribution vs distance to GT boundary
    edges = [0, 1, 2, 3, 5, 8, 12, 20, 40, np.inf]
    dist_rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_gt > lo - 1e-9) & (d_gt <= hi) if lo > 0 else (d_gt <= hi)
        if lo > 0:
            m = (d_gt > lo) & (d_gt <= hi)
        dist_rows.append({'d_lo': float(lo), 'd_hi': float(hi) if hi != np.inf else None,
                          'frac_pixels': float(m.mean()),
                          'frac_errors': float(err[m].sum() / max(err.sum(), 1)),
                          'error_rate': float(err[m].mean()) if m.any() else float('nan')})
    res['error_by_distance'] = dist_rows
    res['oracle_band'] = {'dice_asis': float(np.mean(dice_asis)),
                          **{f'dice_band_{r}': float(np.mean(v)) for r, v in dice_oracle.items()}}
    if binary:
        res['threshold_sweep'] = {str(t): float(np.mean(v)) for t, v in dice_tau.items()}
        vals = list(res['threshold_sweep'].values())
        res['threshold_sweep_span'] = float(max(vals) - min(vals))
    return res, sigs, err


# ---------------------------------------------------------------------------
# F. Post-hoc calibration fitted on the labelled source split
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_probs(models, loader, dataset: str, dev):
    probs, gts = [], []
    for sample in loader:
        x = sample['unet_size_img'].to(dev)
        lab = sample.get('unet_size_label', sample['label'])
        if lab.dim() == 4:
            lab = lab.squeeze(1)
        lab = remap_label(lab.float(), dataset)
        p = forward_signals(models, x, 0)['prob']          # [K,H,W]
        if lab.shape[-2:] != p.shape[-2:]:
            lab = F.interpolate(lab.unsqueeze(1).float(), size=p.shape[-2:], mode='nearest').squeeze(1).long()
        probs.append(p.permute(1, 2, 0).reshape(-1, p.shape[0]).cpu().numpy().astype(np.float32))
        gts.append(lab[0].cpu().numpy().ravel().astype(np.int64))
    return np.concatenate(probs), np.concatenate(gts)


def fit_temperature(probs: np.ndarray, gt: np.ndarray, max_pts: int = 500000) -> float:
    """Temperature on log-probabilities (p_T = softmax(log p / T)), fitted by
    NLL on a fixed subsample. For a Dirichlet mean this is a monotone
    reparameterisation of the same ranking."""
    rs = np.random.RandomState(0)
    if probs.shape[0] > max_pts:
        idx = rs.choice(probs.shape[0], max_pts, replace=False); probs, gt = probs[idx], gt[idx]
    logp = np.log(np.clip(probs, 1e-8, 1.0))
    ts = np.exp(np.linspace(np.log(0.05), np.log(20.0), 121))
    best_T, best_nll = 1.0, np.inf
    for T in ts:
        z = logp / T
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        nll = float(np.mean(lse - z[np.arange(len(gt)), gt]))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-8, 1.0)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_isotonic(conf: np.ndarray, correct: np.ndarray, max_pts: int = 500000):
    from sklearn.isotonic import IsotonicRegression
    rs = np.random.RandomState(0)
    if conf.shape[0] > max_pts:
        idx = rs.choice(conf.shape[0], max_pts, replace=False); conf, correct = conf[idx], correct[idx]
    ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    ir.fit(conf.astype(np.float64), correct.astype(np.float64))
    return ir


def calibration_block(models, dataset, K, args, domains, dev) -> Dict:
    """Fit on the run's labelled split; evaluate on every test domain."""
    lb_domain, lb_idxs = args.lb_domain, None
    split_path = os.path.join(os.path.dirname(args.ckpt), 'split.json')
    if os.path.exists(split_path):
        sp = json.load(open(split_path)); lb_domain = int(sp['lb_domain']); lb_idxs = sp['lb_idxs']
    else:
        from dataloaders.seed_split import select_labeled_idxs
        cfg = DATASET_CFG[dataset]
        # v16 checkpoints carry no split.json: rebuild the run's OWN split from the paper's
        # per-dataset protocol (run_v16_experiments_v2.sh / run_adaptive_threshold.sh):
        # lb_num prostate 20, fundus 20, MNMS 5, BUSI 64 (E-12: an earlier version used 20 for all).
        lb_num = int(args.lb_num) if args.lb_num > 0 else LB_NUM_DEFAULT[dataset]
        seed_for_split = int(re.search(r'_s(\d+)$', os.path.basename(os.path.dirname(args.ckpt))).group(1)) \
            if re.search(r'_s(\d+)$', os.path.basename(os.path.dirname(args.ckpt))) else args.split_seed
        lb_idxs, _ = select_labeled_idxs(cfg['domain_len'][lb_domain - 1], lb_num, seed_for_split)
        print(f"[calibration] no split.json -> protocol split: domain {lb_domain}, lb_num {lb_num}, seed {seed_for_split}")
    lb_loader = build_labelled_loader(dataset, lb_domain, lb_idxs)
    p_lb, g_lb = collect_probs(models, lb_loader, dataset, dev)
    T = fit_temperature(p_lb, g_lb)
    conf_lb = p_lb.max(1); corr_lb = (p_lb.argmax(1) == g_lb)
    iso = fit_isotonic(conf_lb, corr_lb)
    out = {'lb_domain': lb_domain, 'n_lb_images': len(lb_idxs), 'temperature': T,
           'lb_ece_raw': ece_score(conf_lb, corr_lb),
           'lb_ece_ts': ece_score(apply_temperature(p_lb, T).max(1), corr_lb),
           'lb_ece_iso': ece_score(iso.predict(conf_lb.astype(np.float64)), corr_lb),
           'per_domain': {}}
    for d in domains:
        p, g = collect_probs(models, build_test_loader(dataset, d), dataset, dev)
        conf = p.max(1); corr = (p.argmax(1) == g); err = (~corr).astype(np.int64)
        pT = apply_temperature(p, T); confT = pT.max(1)
        confI = iso.predict(conf.astype(np.float64))
        out['per_domain'][str(d)] = {
            'ece_raw': ece_score(conf, corr), 'ece_ts': ece_score(confT, corr), 'ece_iso': ece_score(confI, corr),
            'auroc_raw': fast_auroc(1 - conf, err), 'auroc_ts': fast_auroc(1 - confT, err),
            'auroc_iso': fast_auroc(1 - confI, err),
            'range80_raw': central_range(1 - conf), 'range80_ts': central_range(1 - confT),
            'range80_iso': central_range(1 - confI),
            'acc': float(corr.mean())}
    return out


# ---------------------------------------------------------------------------
# Spacing lookup for millimetre bands (D-1)
# ---------------------------------------------------------------------------

def spacing_index_keys(img_name: str, site: str) -> List[str]:
    """Candidate keys of prostate_spacing_index.json (v3 keys are "<site>/<png stem>") for a
    loader img_name. ProstateSegmentation names its samples "<site>_<stem>.png" (E-21: the S06
    lookup tried "<site>/<site>_<stem>" and matched nothing, silently)."""
    stem = os.path.splitext(os.path.basename(str(img_name)))[0]
    keys = []
    if site and stem.startswith(site + '_'):
        keys.append(f"{site}/{stem[len(site) + 1:]}")
    keys += [f"{site}/{stem}", stem]
    return keys


def make_spacing_lookup(index_path: str):
    """The index (built by prostate_spacing_index.py v3) maps "<site>/<stem>" -> dict(
    mm_per_px_at_grid: {grid_size: value}) after the brute-force slice verification."""
    idx = json.load(open(index_path))
    table = idx['slices']
    print(f"[spacing] index {index_path}: {len(table)} verified slices, grids {idx.get('meta', {}).get('grids')}, "
          f"sample keys {list(table)[:3]}")
    def lookup(img_name: str, grid: int, site: str = ''):
        rec = None
        for k in spacing_index_keys(img_name, site):
            rec = table.get(k)
            if rec is not None:
                break
        if rec is None or not rec.get('pass', True):
            return None
        v = rec.get('mm_per_px_at_grid', {}).get(str(grid))
        return float(v) if v is not None else None
    lookup.table = table
    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASET_CFG))
    ap.add_argument('--ckpt', help='best_student.pth (or glob; first match)')
    ap.add_argument('--ensemble_ckpts', default='', help='comma-separated extra checkpoints '
                    '(same split, different init) -> ens_* signals')
    ap.add_argument('--which', default='ema', choices=['ema', 'student', 'swad'])
    ap.add_argument('--out', default='')
    ap.add_argument('--bands', default='1,2,3,5,8,12')
    ap.add_argument('--rel_fracs', default='0.025,0.05,0.1,0.2')
    ap.add_argument('--mm_bands', default='0.5,1,2,3,5')
    ap.add_argument('--spacing_index', default='')
    ap.add_argument('--mc_T', type=int, default=0)
    ap.add_argument('--domains', default='', help='subset, e.g. 1,3 (default all)')
    ap.add_argument('--calibration', type=int, default=1)
    ap.add_argument('--lb_domain', type=int, default=0, help='needed for v16 ckpts (no split.json)')
    ap.add_argument('--lb_num', type=int, default=0, help='0 = paper protocol per dataset (20/20/5/64)')
    ap.add_argument('--split_seed', type=int, default=1337)
    ap.add_argument('--head', default='', help='override head for old ckpts if needed')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    assert args.dataset and args.ckpt and args.out
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = sorted(glob.glob(args.ckpt))
    assert ck, f"no checkpoint matches {args.ckpt}"
    args.ckpt = ck[0]
    override = {'head': args.head} if args.head else None
    models, meta = load_model(args.dataset, args.ckpt, args.which, dev, override)
    for extra in [c for c in args.ensemble_ckpts.split(',') if c.strip()]:
        m_extra, _ = load_model(args.dataset, extra, args.which, dev, override)
        models += m_extra
        meta.setdefault('ensemble_members', []).append({'ckpt': extra, 'sha256': sha256_of(extra)})
    if args.lb_domain == 0:
        args.lb_domain = int(meta['v18_config'].get('lb_domain', {'prostate': 5, 'fundus': 3, 'MNMS': 4, 'BUSI': 2}[args.dataset]))
    K = DATASET_CFG[args.dataset]['num_classes'] + 1
    domains = [int(d) for d in args.domains.split(',')] if args.domains else list(range(1, DATASET_CFG[args.dataset]['domain_num'] + 1))
    spacing_lookup = make_spacing_lookup(args.spacing_index) if args.spacing_index else None
    out = {'meta': {**meta, 'harness_sha256': sha256_of(os.path.abspath(__file__)),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'args': vars(args),
                    'domain_names': DATASET_CFG[args.dataset]['names']},
           'per_domain': {}}
    t0 = time.time()
    for d in domains:
        loader = build_test_loader(args.dataset, d)
        args._site = DATASET_CFG[args.dataset]['names'][d]
        res, _, _ = measure_domain(models, loader, args.dataset, K, args, spacing_lookup)
        out['per_domain'][str(d)] = res
        s = res['signals'].get('u_epi', res['signals']['conf'])
        b5 = s['band'].get('edt_gt_5', {}).get('auroc', float('nan'))
        print(f"[{args.dataset} d{d} {DATASET_CFG[args.dataset]['names'][d]}] n={res['n_images']} "
              f"err={res['error_rate']:.4f} AUROC(u_epi) all={s['auroc_all']:.3f} band5={b5:.3f} "
              f"range80={s['range80_all']:.2e} oracle5={res['oracle_band'].get('dice_band_5', float('nan')):.4f} "
              f"mm_cov={res['n_images_with_mm']}/{res['n_images']} ({time.time()-t0:.0f}s)")
    if args.calibration:
        try:
            out['calibration'] = calibration_block(models, args.dataset, K, args, domains, dev)
            print(f"[calibration] T={out['calibration']['temperature']:.3f} "
                  f"lb_ece raw/ts/iso = {out['calibration']['lb_ece_raw']:.4f}/"
                  f"{out['calibration']['lb_ece_ts']:.4f}/{out['calibration']['lb_ece_iso']:.4f}")
        except Exception as e:  # never lose the rest of the measurement
            out['calibration'] = {'error': repr(e)}
            print(f"[calibration] FAILED: {e!r}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f"[done] wrote {args.out}")


# ---------------------------------------------------------------------------
# Self-test (CPU, synthetic): geometry, metrics, calibration helpers
# ---------------------------------------------------------------------------

def _selftest_spacing_keys():
    ks = spacing_index_keys('BIDMC_00_07.png', 'BIDMC')
    assert ks[0] == 'BIDMC/00_07', ks
    assert spacing_index_keys('00_07.png', 'BIDMC')[0] == 'BIDMC/00_07'
    assert spacing_index_keys('/x/y/HK_12_03.png', 'HK')[0] == 'HK/12_03'
    import tempfile
    d = tempfile.mkdtemp(); p = os.path.join(d, 'idx.json')
    json.dump({'meta': {'grids': [384]}, 'slices': {'BIDMC/00_07': {'pass': True, 'mm_per_px_at_grid': {'384': 0.3646}},
                                                     'BIDMC/00_08': {'pass': False, 'mm_per_px_at_grid': {'384': 0.3646}}}},
              open(p, 'w'))
    lk = make_spacing_lookup(p)
    assert abs(lk('BIDMC_00_07.png', 384, 'BIDMC') - 0.3646) < 1e-9
    assert lk('BIDMC_00_07.png', 512, 'BIDMC') is None      # grid not indexed
    assert lk('BIDMC_00_08.png', 384, 'BIDMC') is None      # failed verification -> no mm
    assert lk('BIDMC_00_09.png', 384, 'BIDMC') is None      # unknown slice
    print('[harness] spacing-key self-test PASSED')


def _selftest():
    _selftest_spacing_keys()
    rs = np.random.RandomState(0)
    # boundary geometry on a disc
    gt = np.zeros((64, 64), np.int64); yy, xx = np.mgrid[:64, :64]
    gt[(yy - 32) ** 2 + (xx - 32) ** 2 <= 15 ** 2] = 1
    d = edt_to_boundary(gt); c = chebyshev_to_boundary(gt)
    assert d[32, 32] > 13 and d[32, 32] < 16, d[32, 32]
    assert (c >= d - 1e-6).all() or (c <= d + 1e-6).all()  # chebyshev <= euclid
    assert (c <= d + 1e-6).all()
    assert abs(equivalent_radius(gt) - 15) < 0.5
    # a prediction that is the disc dilated by 3 px: all errors within 3 px of GT boundary
    pred = ndi.binary_dilation(gt, iterations=3).astype(np.int64)
    err = pred != gt
    assert (d[err] <= 3 + 1e-6).all()
    assert dice_per_image(gt, gt, 2) == 1.0 and 0.8 < dice_per_image(pred, gt, 2) < 1.0
    p2 = pred.copy(); m = d <= 3; p2[m] = gt[m]; assert dice_per_image(p2, gt, 2) == 1.0
    # AUROC: perfect / chance / inverted
    s = rs.rand(10000); e = (s > 0.7).astype(np.int64)
    assert fast_auroc(s, e) > 0.999 and fast_auroc(-s, e) < 0.001
    assert abs(fast_auroc(rs.rand(10000), (rs.rand(10000) > 0.5).astype(int)) - 0.5) < 0.03
    assert abs(central_range(np.linspace(0, 1, 1001)) - 0.8) < 1e-6
    # ECE / temperature / isotonic
    conf = rs.uniform(0.5, 1.0, 20000); correct = (rs.rand(20000) < conf).astype(int)
    assert ece_score(conf, correct) < 0.03
    probs = np.stack([1 - conf, conf], 1); gt2 = np.where(correct == 1, 1, 0)
    T = fit_temperature(probs, gt2); assert 0.5 < T < 2.0, T
    over = np.stack([1 - conf ** 3, conf ** 3], 1)  # overconfident
    assert fit_temperature(over, gt2) > 1.0
    iso = fit_isotonic(conf, correct); assert ece_score(iso.predict(conf), correct) < 0.03
    print('[diag_uncertainty_harness_v18] self-test PASSED')


if __name__ == '__main__':
    main()
