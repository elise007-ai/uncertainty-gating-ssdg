"""
cadg_diagnostics.py  (Option B kill-switch experiments, fully wired)
====================================================================
Runs Experiment 0 (premise) and Experiment 1 (label-free vs oracle calibration
map) on the REAL FourierFuse v16 release — no stubs to fill in. It:

  * builds the per-domain TEST loaders exactly as train_fourierfuse_v16.py does,
  * loads the EMA teacher weights from a best_student.pth checkpoint,
  * for each domain (vendor), computes:
      - Spearman( teacher U_epi , true pseudo-label error )         [rankability]
      - Spearman( 1 - stability-under-FDA , true error )            [proxy validity]
      - AUROC for both vs error
      - the per-domain LABEL-FREE reliability curve g_free and the LABEL-BASED
        (oracle) curve g_lab, and their weighted MAE / Spearman
  * prints per-domain tables and overall VERDICTs for Exp 0 and Exp 1.

This is the FIRST thing to run for Option B. It needs no training.

Example (prostate, the checkpoint you provided):

  cd CODE/source
  python cadg_diagnostics.py \
      --dataset prostate \
      --data_root /home/yahyaisaac/root_eli/ProtoSynFoC/data/prostate \
      --ckpt /home/yahyaisaac/root_eli/ProtoSynFoC/model/prostate/train_fourierfuse_v16/v16_prostate_s1337/best_student.pth \
      --domain_num 6 --img_size 512 --fda_beta 0.05 \
      --shifted_domain 6 \
      --out ../../diag_prostate_s1337.json

(Repeat for s42 / s2024 by pointing --ckpt at their best_student.pth.)

VERDICTS:
  Exp0: PREMISE SUPPORTED / WEAK / FAILED
  Exp1: TRICK VALIDATED / PARTIAL / FAILED
If Exp0 FAILED or Exp1 FAILED -> stop Option B, ship Option A.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# project imports (run from CODE/source)
from dataloaders.dataloader import (FundusSegmentation, ProstateSegmentation,
                                    MNMSSegmentation, BUSISegmentation)
import dataloaders.custom_transforms as tr
from torch.utils.data import DataLoader

from fourierfuse_v16_modules import FourierFuseNet, fda_spectral_augment
from cadg_modules import PerDomainReliabilityCalibrator


# --------------------------------------------------------------------------
# stats helpers
# --------------------------------------------------------------------------
def spearman(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d > 0 else float("nan")


def auroc(score, label):
    score = np.asarray(score); label = np.asarray(label).astype(int)
    n_pos = int(label.sum()); n_neg = len(label)-n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    return float((ranks[label == 1].sum() - n_pos*(n_pos+1)/2)/(n_pos*n_neg))


def isotonic_from_counts(stable_counts, unstable_counts):
    total = stable_counts + unstable_counts
    with np.errstate(invalid="ignore", divide="ignore"):
        q = unstable_counts/np.maximum(total, 1e-8)
    g = PerDomainReliabilityCalibrator._pava(np.nan_to_num(q), total)
    return np.clip(g, 0, 1)


# --------------------------------------------------------------------------
# build per-domain test loaders exactly like train does
# --------------------------------------------------------------------------
def build_test_loaders(args):
    ds_cls = {"prostate": ProstateSegmentation, "fundus": FundusSegmentation,
              "MNMS": MNMSSegmentation, "BUSI": BUSISegmentation}[args.dataset]
    patch_size = {"prostate": 384, "fundus": 256, "MNMS": 288, "BUSI": 256}[args.dataset]
    low_res = patch_size
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0, 1]),
        tr.ToTensor(low_res=low_res, unet_size=patch_size),
    ])
    loaders = []
    for i in range(1, args.domain_num + 1):
        cur = ds_cls(base_dir=args.data_root, phase='test', splitid=-1,
                     domain=[i], normal_toTensor=normal_toTensor,
                     img_size=args.img_size)
        loaders.append(DataLoader(cur, batch_size=args.test_bs, shuffle=False,
                                  num_workers=2, pin_memory=True))
    return loaders


def load_teacher(args):
    # num_classes here = FOREGROUND class count (model output K = n_classes + 1),
    # matching train_fourierfuse_v16.py.
    num_classes = {"prostate": 1, "fundus": 2, "MNMS": 3, "BUSI": 1}[args.dataset]
    num_channels = {"prostate": 1, "fundus": 3, "MNMS": 1, "BUSI": 1}[args.dataset]
    model = FourierFuseNet(
        in_channels=num_channels,
        n_classes=num_classes,
        base_features=args.ff_base_features,
        bilinear=False,
        num_domains=args.domain_num,
        num_protos_per_domain=2,
        proto_ema_decay=0.99,
    ).cuda()
    ckpt = torch.load(args.ckpt, map_location="cuda")
    sd = ckpt.get("ema_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, num_classes + 1


def prostate_mask(sample_label, dataset):
    if dataset == "prostate":
        return sample_label.eq(0).long()
    if dataset == "BUSI":
        return sample_label.eq(255).long()
    if dataset == "MNMS":
        return sample_label.long()
    # fundus handled separately (2-channel); for diagnostics we use argmax match
    lab = (sample_label <= 128) * 2
    lab[sample_label == 0] = 1
    return lab.long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain_num", type=int, required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--test_bs", type=int, default=1)
    ap.add_argument("--ff_base_features", type=int, default=64)
    ap.add_argument("--fda_beta", type=float, default=0.05)
    ap.add_argument("--n_bins", type=int, default=20)
    ap.add_argument("--shifted_domain", type=int, default=None,
                    help="1-indexed vendor id of the most-shifted domain "
                         "(e.g. BIDMC) for the headline Exp-1 check")
    ap.add_argument("--max_imgs_per_domain", type=int, default=200)
    ap.add_argument("--out", default="cadg_diag.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    model, K = load_teacher(args)
    loaders = build_test_loaders(args)

    nb = args.n_bins
    per_domain = {}      # 1-indexed domain -> dict of arrays/counts
    prev_img_buf = {}    # domain -> last image tensor, used as a style reference

    for di, loader in enumerate(loaders, start=1):
        u_list, err_list, unstable_list, conf_list = [], [], [], []
        lab_stable = np.zeros(nb); lab_unstable = np.zeros(nb)
        free_stable = np.zeros(nb); free_unstable = np.zeros(nb)
        seen = 0
        for sample in loader:
            if seen >= args.max_imgs_per_domain:
                break
            img = sample['unet_size_img'].cuda()
            lab = prostate_mask(sample['label'], args.dataset).cuda()
            if lab.dim() == 4:
                lab = lab.squeeze(1)

            with torch.no_grad():
                out = model.forward(img, return_features=False)
                prob = out['prob']                      # [B,K,H,W]
                u = out['U_epi']                        # [B,1,H,W] or [B,H,W]
                u = u.squeeze(1) if u.dim() == 4 else u
                pred = prob.argmax(dim=1)               # [B,H,W]
                # align label spatial size to pred if needed
                if lab.shape[-2:] != pred.shape[-2:]:
                    lab = F.interpolate(lab.unsqueeze(1).float(),
                                        size=pred.shape[-2:],
                                        mode='nearest').squeeze(1).long()
                wrong = (pred != lab).float()

                # label-free stability under FDA style swap. Use a DIFFERENT
                # image as the style reference (buffered from the previous
                # batch) so that at test_bs=1 the swap is not a no-op self-swap.
                ref = prev_img_buf.get(di, img.flip(0))
                if ref.shape != img.shape:
                    ref = img.flip(0)
                styled = fda_spectral_augment(img, ref, beta=args.fda_beta)
                prob2 = model.forward(styled, return_features=False)['prob']
                unstable = (pred != prob2.argmax(dim=1)).float()
                conf = prob.max(dim=1).values
            prev_img_buf[di] = img.detach()

            ub = (u.clamp(0, 1-1e-6) * nb).long().reshape(-1).cpu().numpy()
            w = wrong.reshape(-1).cpu().numpy()
            us = unstable.reshape(-1).cpu().numpy()
            # accumulate calibration-curve counts (all pixels)
            np.add.at(lab_unstable, ub, w);   np.add.at(lab_stable, ub, 1.0-w)
            np.add.at(free_unstable, ub, us); np.add.at(free_stable, ub, 1.0-us)
            # subsample for correlation/AUROC arrays
            uf = u.reshape(-1).cpu().numpy()
            n = len(uf); sel = np.random.choice(n, size=min(n, 4000), replace=False)
            u_list.extend(uf[sel]); err_list.extend(w[sel])
            unstable_list.extend(us[sel]); conf_list.extend(conf.reshape(-1).cpu().numpy()[sel])
            seen += img.shape[0]

        u_arr = np.asarray(u_list); err_arr = np.asarray(err_list)
        uns_arr = np.asarray(unstable_list)
        g_lab = isotonic_from_counts(lab_stable, lab_unstable)
        g_free = isotonic_from_counts(free_stable, free_unstable)
        mass = lab_stable + lab_unstable
        mask = mass > 0
        mae = (float(np.average(np.abs(g_lab[mask]-g_free[mask]), weights=mass[mask]))
               if mask.sum() >= 3 else float("nan"))
        u_lo, u_hi = (np.percentile(u_arr, [10, 90]) if len(u_arr) else (0, 0))
        per_domain[di] = {
            "n_pixels_sampled": int(len(u_arr)),
            "error_rate": float(err_arr.mean()) if len(err_arr) else float("nan"),
            "spearman_u_vs_error": spearman(u_arr, err_arr),
            "spearman_proxy_vs_error": spearman(uns_arr, err_arr),
            "auroc_u_vs_error": auroc(u_arr, err_arr),
            "auroc_proxy_vs_error": auroc(uns_arr, err_arr),
            "u_central80_range": float(u_hi - u_lo),
            "calmap_weighted_MAE_free_vs_lab": mae,
            "g_lab": [float(x) for x in g_lab],
            "g_free": [float(x) for x in g_free],
        }
        print(f"[domain {di}] err={per_domain[di]['error_rate']:.3f} "
              f"rho(u,err)={per_domain[di]['spearman_u_vs_error']:.3f} "
              f"rho(proxy,err)={per_domain[di]['spearman_proxy_vs_error']:.3f} "
              f"calMAE={mae:.3f} u80range={per_domain[di]['u_central80_range']:.3f}")

    # ---- verdicts ----
    rho_u = [abs(d["spearman_u_vs_error"]) for d in per_domain.values()
             if not np.isnan(d["spearman_u_vs_error"])]
    rho_p = [abs(d["spearman_proxy_vs_error"]) for d in per_domain.values()
             if not np.isnan(d["spearman_proxy_vs_error"])]
    maes = [d["calmap_weighted_MAE_free_vs_lab"] for d in per_domain.values()
            if not np.isnan(d["calmap_weighted_MAE_free_vs_lab"])]
    mean_rho_u = float(np.mean(rho_u)) if rho_u else 0.0
    mean_rho_p = float(np.mean(rho_p)) if rho_p else 0.0
    mean_mae = float(np.mean(maes)) if maes else float("nan")
    shifted_mae = (per_domain.get(args.shifted_domain, {})
                   .get("calmap_weighted_MAE_free_vs_lab")
                   if args.shifted_domain else None)

    if mean_rho_u >= 0.15 and mean_rho_p >= max(0.08, 0.6*mean_rho_u):
        exp0 = "PREMISE SUPPORTED"
    elif mean_rho_u >= 0.10:
        exp0 = "PREMISE WEAK"
    else:
        exp0 = "PREMISE FAILED"

    if not np.isnan(mean_mae) and mean_mae < 0.10 and (shifted_mae is None or shifted_mae < 0.15):
        exp1 = "TRICK VALIDATED"
    elif not np.isnan(mean_mae) and mean_mae < 0.18:
        exp1 = "TRICK PARTIAL"
    else:
        exp1 = "TRICK FAILED"

    summary = {
        "dataset": args.dataset, "ckpt": args.ckpt,
        "mean_abs_spearman_u_vs_error": mean_rho_u,
        "mean_abs_spearman_proxy_vs_error": mean_rho_p,
        "mean_calmap_MAE_free_vs_lab": mean_mae,
        "shifted_domain": args.shifted_domain,
        "shifted_domain_calMAE": shifted_mae,
        "EXP0_VERDICT": exp0,
        "EXP1_VERDICT": exp1,
    }
    result = {"summary": summary, "per_domain": per_domain}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    print(f"EXP0: {exp0}   EXP1: {exp1}")
    print("If EXP0 FAILED or EXP1 FAILED -> stop Option B, ship Option A.")
    print("Otherwise -> run scripts/run_exp2.sh (full CADG training).")


if __name__ == "__main__":
    main()
