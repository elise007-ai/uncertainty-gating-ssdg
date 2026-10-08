"""
normstat_diagnostics.py  (NEXT-PROJECT Stage-0 kill-switch)
===========================================================
Tests the premise of the normalization-statistics hypothesis (NEXT_PROJECT/
PROPOSAL.md) on EXISTING v16 checkpoints, with NO training.

Question: does a per-region BatchNorm-statistics MISMATCH signal predict true
pseudo-label error better than the model's own confidence or evidential vacuity —
especially on the shifted vendors where confidence is known to fail?

For each BN layer we have stored SOURCE statistics (running_mean, running_var).
For a given target image we compute that layer's ACTUAL feature statistics
(per-sample, and spatially-local via average pooling), then a mismatch map:
    m = 0.5 * [ (mu_t - mu_s)^2 / var_s + var_t/var_s - 1 - log(var_t/var_s) ]
(symmetric-ish KL between Gaussians, per channel), averaged over channels and
upsampled to pixel space, aggregated across a set of BN layers.

We then compare, per vendor:
    AUROC( mismatch ,    pseudo-label error )
    AUROC( 1 - confidence, error )
    AUROC( vacuity U_epi , error )

P1 passes if mismatch AUROC > confidence/vacuity AUROC on the shifted vendors.

Usage (same checkpoints as before):
  python normstat_diagnostics.py \
      --dataset prostate \
      --data_root /home/yahyaisaac/root_eli/data/ProstateSlice \
      --ckpt .../v16_prostate_s1337/best_student.pth \
      --domain_num 6 --img_size 384 \
      --shifted_domain 1 \
      --out ../../normstat_prostate_s1337.json

Decision: P1 green on shifted vendors -> proceed to Stage 1.
          P1 red -> stop; the signal is redundant with confidence; ship Option A.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from dataloaders.dataloader import (FundusSegmentation, ProstateSegmentation,
                                    MNMSSegmentation, BUSISegmentation)
import dataloaders.custom_transforms as tr
from torch.utils.data import DataLoader
from fourierfuse_v16_modules import FourierFuseNet


def auroc(score, label):
    score = np.asarray(score); label = np.asarray(label).astype(int)
    n_pos = int(label.sum()); n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(score)) + 1
    return float((ranks[label == 1].sum() - n_pos*(n_pos+1)/2)/(n_pos*n_neg))


# --------------------------------------------------------------------------
# BN-statistics mismatch via forward hooks
# --------------------------------------------------------------------------
class BNMismatch:
    """Hooks BatchNorm2d layers; on each forward, computes a per-pixel mismatch
    map between the layer's INPUT feature statistics (local) and the layer's
    stored running (source) statistics, upsampled to the input image size and
    averaged across the hooked layers.
    """
    def __init__(self, model, img_size, max_layers=8, local_window=8):
        self.img_size = img_size
        self.local_window = local_window
        self.maps = []
        self.handles = []
        bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        # spread across depth: take up to max_layers evenly
        if len(bns) > max_layers:
            idx = np.linspace(0, len(bns) - 1, max_layers).astype(int)
            bns = [bns[i] for i in idx]
        for bn in bns:
            self.handles.append(bn.register_forward_pre_hook(self._make_hook(bn)))

    def _make_hook(self, bn):
        def hook(module, inputs):
            x = inputs[0]                              # [B,C,H,W] (pre-norm)
            mu_s = bn.running_mean.view(1, -1, 1, 1)
            var_s = bn.running_var.view(1, -1, 1, 1).clamp(min=1e-5)
            # local target statistics via average pooling
            k = self.local_window
            mu_t = F.avg_pool2d(x, k, stride=1, padding=k // 2)
            mu_t = mu_t[..., :x.shape[-2], :x.shape[-1]]
            ex2 = F.avg_pool2d(x * x, k, stride=1, padding=k // 2)
            ex2 = ex2[..., :x.shape[-2], :x.shape[-1]]
            var_t = (ex2 - mu_t * mu_t).clamp(min=1e-5)
            # per-channel symmetric-ish KL(Gaussian) between (mu_t,var_t)||(mu_s,var_s)
            term = 0.5 * ((mu_t - mu_s) ** 2 / var_s + var_t / var_s
                          - 1.0 - torch.log(var_t / var_s))
            m = term.mean(dim=1, keepdim=True)          # [B,1,H,W]
            m = F.interpolate(m, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
            self.maps.append(m.detach())
        return hook

    def consume(self):
        """Aggregate hooked maps for the last forward into one [B,H,W] map."""
        if not self.maps:
            return None
        stacked = torch.stack(self.maps, dim=0).mean(dim=0)   # [B,1,H,W]
        self.maps = []
        return stacked.squeeze(1)

    def remove(self):
        for h in self.handles:
            h.remove()


def build_loaders(args):
    ds_cls = {"prostate": ProstateSegmentation, "fundus": FundusSegmentation,
              "MNMS": MNMSSegmentation, "BUSI": BUSISegmentation}[args.dataset]
    patch = {"prostate": 384, "fundus": 256, "MNMS": 288, "BUSI": 256}[args.dataset]
    tfm = transforms.Compose([tr.Normalize_tf(dataRange=[0, 1]),
                              tr.ToTensor(low_res=patch, unet_size=patch)])
    loaders = []
    for i in range(1, args.domain_num + 1):
        cur = ds_cls(base_dir=args.data_root, phase='test', splitid=-1,
                     domain=[i], normal_toTensor=tfm, img_size=args.img_size)
        loaders.append(DataLoader(cur, batch_size=1, shuffle=False,
                                  num_workers=2, pin_memory=True))
    return loaders, patch


def load_model(args):
    num_classes = {"prostate": 1, "fundus": 2, "MNMS": 3, "BUSI": 1}[args.dataset]
    num_channels = {"prostate": 1, "fundus": 3, "MNMS": 1, "BUSI": 1}[args.dataset]
    model = FourierFuseNet(in_channels=num_channels, n_classes=num_classes,
                           base_features=args.ff_base_features, bilinear=False,
                           num_domains=args.domain_num, num_protos_per_domain=2,
                           proto_ema_decay=0.99).cuda()
    ckpt = torch.load(args.ckpt, map_location="cuda")
    sd = ckpt.get("ema_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def label_to_fg(label, dataset):
    if dataset == "prostate":
        return label.eq(0).long()
    if dataset == "BUSI":
        return label.eq(255).long()
    if dataset == "MNMS":
        return label.long()
    lab = (label <= 128) * 2
    lab[label == 0] = 1
    return lab.long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain_num", type=int, required=True)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--ff_base_features", type=int, default=64)
    ap.add_argument("--shifted_domain", type=int, default=1)
    ap.add_argument("--max_imgs_per_domain", type=int, default=150)
    ap.add_argument("--out", default="normstat_diag.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs GPU"
    model = load_model(args)
    loaders, patch = build_loaders(args)
    mism = BNMismatch(model, img_size=patch, max_layers=8, local_window=8)

    per_domain = {}
    for di, loader in enumerate(loaders, start=1):
        s_mism, s_conf, s_vac, s_err = [], [], [], []
        seen = 0
        for sample in loader:
            if seen >= args.max_imgs_per_domain:
                break
            img = sample['unet_size_img'].cuda()
            lab = label_to_fg(sample['label'], args.dataset).cuda()
            if lab.dim() == 4:
                lab = lab.squeeze(1)
            with torch.no_grad():
                out = model.forward(img, return_features=False)
                prob = out['prob']
                vac = out['U_epi']
                vac = vac.squeeze(1) if vac.dim() == 4 else vac
                pred = prob.argmax(dim=1)
                conf = prob.max(dim=1).values
                mmap = mism.consume()                    # [B,H,W] mismatch
            if lab.shape[-2:] != pred.shape[-2:]:
                lab = F.interpolate(lab.unsqueeze(1).float(), size=pred.shape[-2:],
                                    mode='nearest').squeeze(1).long()
            if mmap is not None and mmap.shape[-2:] != pred.shape[-2:]:
                mmap = F.interpolate(mmap.unsqueeze(1), size=pred.shape[-2:],
                                     mode='bilinear', align_corners=False).squeeze(1)
            err = (pred != lab).float()

            n = err.numel()
            sel = np.random.choice(n, size=min(n, 4000), replace=False)
            s_err.extend(err.reshape(-1).cpu().numpy()[sel])
            s_conf.extend((1.0 - conf).reshape(-1).cpu().numpy()[sel])  # low conf -> high "uncertainty"
            s_vac.extend(vac.reshape(-1).cpu().numpy()[sel])
            s_mism.extend(mmap.reshape(-1).cpu().numpy()[sel] if mmap is not None
                          else np.zeros(len(sel)))
            seen += 1

        err = np.asarray(s_err)
        a_m = auroc(np.asarray(s_mism), err)
        a_c = auroc(np.asarray(s_conf), err)
        a_v = auroc(np.asarray(s_vac), err)
        per_domain[di] = {
            "error_rate": float(err.mean()),
            "auroc_mismatch_vs_error": a_m,
            "auroc_lowconf_vs_error": a_c,
            "auroc_vacuity_vs_error": a_v,
            "mismatch_beats_conf": bool(a_m > a_c),
            "mismatch_beats_vac": bool(a_m > a_v),
        }
        print(f"[domain {di}] err={err.mean():.3f}  "
              f"AUROC mismatch={a_m:.3f}  lowconf={a_c:.3f}  vacuity={a_v:.3f}  "
              f"{'<-- BIDMC/shifted' if di==args.shifted_domain else ''}")

    mism.remove()

    # verdict: P1 = mismatch beats confidence AND vacuity on the shifted vendor,
    # and on average across vendors.
    shifted = per_domain.get(args.shifted_domain, {})
    mean_m = np.mean([d["auroc_mismatch_vs_error"] for d in per_domain.values()
                      if not np.isnan(d["auroc_mismatch_vs_error"])])
    mean_c = np.mean([d["auroc_lowconf_vs_error"] for d in per_domain.values()
                      if not np.isnan(d["auroc_lowconf_vs_error"])])
    shifted_ok = (shifted.get("mismatch_beats_conf") and
                  shifted.get("mismatch_beats_vac"))
    if shifted_ok and mean_m > mean_c:
        verdict = "P1 SUPPORTED — mismatch predicts error better than confidence"
    elif mean_m > mean_c:
        verdict = "P1 WEAK — better on average but not decisively on shifted vendor"
    else:
        verdict = "P1 FAILED — mismatch is not better than confidence; stop, ship Option A"

    summary = {
        "dataset": args.dataset, "ckpt": args.ckpt,
        "shifted_domain": args.shifted_domain,
        "mean_auroc_mismatch": float(mean_m),
        "mean_auroc_lowconf": float(mean_c),
        "shifted_mismatch_beats_conf": shifted.get("mismatch_beats_conf"),
        "shifted_mismatch_beats_vac": shifted.get("mismatch_beats_vac"),
        "VERDICT": verdict,
    }
    json.dump({"summary": summary, "per_domain": per_domain},
              open(args.out, "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
