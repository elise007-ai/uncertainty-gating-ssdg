#!/usr/bin/env python3
"""measure_collapse_alldatasets.py

Extends the vacuity range-collapse measurement (paper Table 6) from Prostate to
all four benchmarks (Prostate, Fundus, M&Ms, BUSI), per the reviewer request to
demonstrate the collapse is general, not Prostate-specific.

For each test domain of each dataset it computes, on the converged EMA teacher:
  (1) AUROC(U_epi, error)  -- does vacuity RANK pixel error?  (rank quality)
  (2) central-80% range of U_epi -- the width of the 10th-90th percentile band
      of the per-pixel vacuity distribution (usable dynamic range).

It reuses the SAME model loader, forward call, and dataloaders as
make_uncertainty_figure.py, so it runs against the SAME checkpoints with no
retraining -- inference only.

Usage (run from ProtoSynFoC/source, one dataset at a time):
  python measure_collapse_alldatasets.py --dataset prostate \
      --ckpt_glob "../model/prostate/train_fourierfuse_v16/v16_prostate_s1337/best_student.pth"
  python measure_collapse_alldatasets.py --dataset fundus \
      --ckpt_glob "../model/fundus/train_fourierfuse_v16/v16_fundus_s1337/best_student.pth"
  python measure_collapse_alldatasets.py --dataset MNMS  --ckpt_glob "...best_student.pth"
  python measure_collapse_alldatasets.py --dataset BUSI  --ckpt_glob "...best_student.pth"

Prints a per-domain table and the dataset mean (AUROC, central-80% range), ready
to drop into Table 6.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F

from fourierfuse_v16_modules import FourierFuseNet
from dataloaders.dataloader import (FundusSegmentation, ProstateSegmentation,
                                    MNMSSegmentation, BUSISegmentation)
import dataloaders.custom_transforms as tr
from torchvision import transforms
from torch.utils.data import DataLoader

# Per-dataset constants, matching train_fourierfuse_v16.py / the figure script.
DATASET_CFG = {
    "prostate": dict(num_channels=1, num_classes=1, patch_size=384, domain_num=6,
                     data_path="../../data/ProstateSlice", DS=ProstateSegmentation),
    "fundus":   dict(num_channels=3, num_classes=2, patch_size=256, domain_num=4,
                     data_path="../../data/Fundus", DS=FundusSegmentation),
    "MNMS":     dict(num_channels=1, num_classes=3, patch_size=288, domain_num=4,
                     data_path="../../data/mnms", DS=MNMSSegmentation),
    "BUSI":     dict(num_channels=1, num_classes=1, patch_size=256, domain_num=2,
                     data_path="../../data/Dataset_BUSI_with_GT", DS=BUSISegmentation),
}


def load_model(dataset, ckpt, base_features=64):
    cfg = DATASET_CFG[dataset]
    model = FourierFuseNet(in_channels=cfg["num_channels"], n_classes=cfg["num_classes"],
                           base_features=base_features, bilinear=False,
                           num_domains=cfg["domain_num"], num_protos_per_domain=2,
                           proto_ema_decay=0.99).cuda()
    ckpt_d = torch.load(ckpt, map_location="cuda")
    sd = ckpt_d.get("ema_state_dict") or ckpt_d.get("state_dict") or ckpt_d
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def build_loader(dataset, domain):
    cfg = DATASET_CFG[dataset]
    patch = cfg["patch_size"]
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0, 1]),
        tr.ToTensor(low_res=patch, unet_size=patch),
    ])
    ds = cfg["DS"](base_dir=cfg["data_path"], phase='test', splitid=-1,
                   domain=[domain], normal_toTensor=normal_toTensor, img_size=patch)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


def remap_label(label, dataset):
    """Put the loaded label into the SAME convention the model was trained/
    evaluated against, matching prostate_mask() in cadg_diagnostics.py.

    The dataloaders return raw label values whose foreground convention differs
    per dataset; the model's argmax prediction is always 0=bg, 1..K-1=fg. Without
    this remap, (pred != label) is computed across incompatible conventions and
    the error mask is wrong -- for Prostate it is the exact binary complement,
    which INVERTS the AUROC (AUROC(U,1-y) = 1-AUROC(U,y)), turning the true ~0.84
    into ~0.16. MNMS is the only dataset whose remap is the identity, which is why
    it was the only one unaffected by the omission.
    """
    if dataset == "prostate":
        return label.eq(0).long()
    if dataset == "BUSI":
        return label.eq(255).long()
    if dataset == "MNMS":
        return label.long()
    # fundus: grayscale {0,128,255} -> {cup=2 inside disc, disc=1, bg=0}
    lab = (label <= 128).long() * 2
    lab[label == 0] = 1
    return lab.long()


def fast_auroc(scores, labels, max_pts=200000):
    """AUROC of `scores` predicting binary `labels` (1=error). Subsampled for speed."""
    scores = scores.ravel(); labels = labels.ravel()
    if scores.shape[0] > max_pts:
        idx = np.random.RandomState(0).choice(scores.shape[0], max_pts, replace=False)
        scores, labels = scores[idx], labels[idx]
    pos = labels == 1; neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float('nan')
    # rank-based AUROC (Mann-Whitney U)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    return float(auc)


@torch.no_grad()
def measure_domain(model, loader, dataset):
    cfg = DATASET_CFG[dataset]
    all_u, all_err = [], []
    for sample in loader:
        x = sample['unet_size_img'].cuda()
        out = model.forward(x, return_features=False)
        prob = out['prob']
        u_epi = out['U_epi']
        u_epi = u_epi.squeeze(1) if u_epi.dim() == 4 else u_epi   # [B,H,W]
        pred = torch.argmax(prob, dim=1)                          # [B,H,W]
        lab = sample['label'].cuda().float()
        if lab.dim() == 4:
            lab = lab.squeeze(1)
        # Put the label into the model's prediction convention BEFORE comparing.
        # (This was the missing step: the previous version compared argmax against
        # the raw loader label, which for Prostate/BUSI/Fundus is a different
        # convention and produced inverted/garbage error masks. MNMS happened to
        # be unaffected because its remap is the identity.)
        lab = remap_label(lab, dataset)
        # Align label to the prediction grid (nearest) so the error mask sits on
        # the same grid as U_epi.
        if lab.shape[-2:] != pred.shape[-2:]:
            lab = F.interpolate(lab.unsqueeze(1).float(), size=pred.shape[-2:],
                                mode='nearest').squeeze(1).long()
        err = (pred != lab.long()).long().cpu().numpy()
        all_u.append(u_epi.cpu().numpy().ravel())
        all_err.append(err.ravel())
    u = np.concatenate(all_u); err = np.concatenate(all_err)
    auroc = fast_auroc(u, err)
    lo, hi = np.percentile(u, [10, 90])
    return auroc, float(hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    ap.add_argument("--ckpt_glob", required=True,
                    help="path (or glob) to best_student.pth for this dataset")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"

    ckpt = sorted(glob.glob(args.ckpt_glob))[0]
    model = load_model(args.dataset, ckpt)
    cfg = DATASET_CFG[args.dataset]

    print(f"\n=== {args.dataset}  (ckpt: {ckpt}) ===")
    print(f"{'domain':>8s}{'AUROC':>10s}{'central-80% range':>22s}")
    aurocs, ranges = [], []
    for d in range(1, cfg["domain_num"] + 1):
        try:
            loader = build_loader(args.dataset, d)
            a, r = measure_domain(model, loader, args.dataset)
        except Exception as e:
            print(f"{d:>8d}   (skipped: {e})")
            continue
        aurocs.append(a); ranges.append(r)
        print(f"{d:>8d}{a:>10.2f}{r:>22.2e}")
    if aurocs:
        print(f"{'MEAN':>8s}{np.nanmean(aurocs):>10.2f}{np.nanmean(ranges):>22.2e}")
        print("\nDrop the MEAN row (and per-domain rows) into Table 6.")


if __name__ == "__main__":
    main()
