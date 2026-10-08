"""
make_uncertainty_figure.py  --- generates CMIG Figure 1 (qualitative vacuity collapse)
=====================================================================================
Produces a multi-panel figure that makes the evidential-uncertainty *range
collapse* visible. For one representative Prostate BIDMC slice and one Fundus
optic-cup slice it shows, left to right:

    input | teacher prediction | true error map |
    U_epi rescaled to its own min-max | U_epi on a fixed [0,1] gate scale

The two right-most panels are the point of the figure: the first shows that
vacuity, after per-image rescaling, *does* track error (matching the AUROC in
Table 'collapse'); the second shows that on a fixed gate scale the same map is
nearly uniform, because its true dynamic range is ~1e-4. That visual contrast
is the collapse.

This script mirrors normstat_diagnostics.py exactly (same model loader, same
forward call, same label_to_fg) so it runs against the SAME checkpoints with no
retraining.

USAGE (run once for prostate, once for fundus, or pass both ckpts):
  python make_uncertainty_figure.py \
      --prostate_ckpt /path/v16_prostate_s1337/best_student.pth \
      --prostate_root /home/.../data/ProstateSlice \
      --fundus_ckpt   /path/v16_fundus_s1337/best_student.pth \
      --fundus_root   /home/.../data/Fundus \
      --out Figure_2.png

Then copy Figure_2.png into the paper's figures/ directory.

Dependencies: torch, torchvision, numpy, matplotlib, and the repo's
fourierfuse_v16_modules + dataloaders on PYTHONPATH (run from CODE/source/).
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourierfuse_v16_modules import FourierFuseNet
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation
import dataloaders.custom_transforms as tr
from torchvision import transforms
from torch.utils.data import DataLoader


# Per-dataset constants, copied verbatim from train_fourierfuse_v16.py
DATASET_CFG = {
    "prostate": dict(num_channels=1, num_classes=1, patch_size=384, domain_num=6,
                     data_path="../../data/ProstateSlice"),
    "fundus":   dict(num_channels=3, num_classes=2, patch_size=256, domain_num=4,
                     data_path="../../data/Fundus"),
}


def load_model(dataset, ckpt, domain_num, base_features=64):
    cfg = DATASET_CFG[dataset]
    model = FourierFuseNet(in_channels=cfg["num_channels"], n_classes=cfg["num_classes"],
                           base_features=base_features, bilinear=False,
                           num_domains=domain_num, num_protos_per_domain=2,
                           proto_ema_decay=0.99).cuda()
    ckpt_d = torch.load(ckpt, map_location="cuda")
    sd = ckpt_d.get("ema_state_dict") or ckpt_d.get("state_dict") or ckpt_d
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def label_to_fg(label, dataset):
    if dataset == "prostate":
        return label.eq(0).long()
    lab = (label <= 128) * 2          # fundus: cup
    lab[label == 0] = 1               # disc
    return lab.long()


def build_loader(dataset, root, domain, img_size):
    """Mirror the test-loader construction in train_fourierfuse_v16.py exactly.
    The test dataset uses phase='test', splitid=-1, domain=[i], and a
    normal_toTensor transform that emits the 'unet_size_img' key the model reads."""
    cfg = DATASET_CFG[dataset]
    patch = cfg["patch_size"]
    base = root if root else cfg["data_path"]
    DS = ProstateSegmentation if dataset == "prostate" else FundusSegmentation
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0, 1]),
        tr.ToTensor(low_res=patch, unet_size=patch),
    ])
    ds = DS(base_dir=base, phase='test', splitid=-1, domain=[domain],
            normal_toTensor=normal_toTensor, img_size=img_size)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


def pick_informative_slice(model, loader, dataset, want_cup=False, max_scan=120):
    """Return (img, pred, err, vac) for the slice with the most foreground error,
    so the panel is visually meaningful rather than an empty background slice."""
    best = None; best_score = -1
    for i, sample in enumerate(loader):
        if i >= max_scan:
            break
        img = sample['unet_size_img'].cuda()
        lab = label_to_fg(sample['label'], dataset).cuda()
        if lab.dim() == 4:
            lab = lab.squeeze(1)
        with torch.no_grad():
            out = model.forward(img, return_features=False)
            prob = out['prob']; vac = out['U_epi']
            vac = vac.squeeze(1) if vac.dim() == 4 else vac
            pred = prob.argmax(dim=1)
            conf = prob.max(dim=1).values
        if lab.shape[-2:] != pred.shape[-2:]:
            lab = F.interpolate(lab.unsqueeze(1).float(), size=pred.shape[-2:],
                                mode='nearest').squeeze(1).long()
        err = (pred != lab).float()
        # score: prefer slices with real foreground error; for cup, require class-2 present
        fg = (lab > 0).float().sum().item()
        if want_cup and (lab == 2).float().sum().item() < 50:
            continue
        score = err.sum().item() * (1.0 if fg > 50 else 0.1)
        if score > best_score:
            best_score = score
            best = (img.detach().cpu(), pred.detach().cpu(),
                    err.detach().cpu(), vac.detach().cpu())
    return best


def row_panels(ax_row, packed, title):
    img, pred, err, vac = packed
    img0 = img[0].numpy()
    if img0.shape[0] == 1:
        disp = img0[0]; cmap_in = 'gray'
    else:
        disp = np.transpose(img0, (1, 2, 0)); disp = (disp - disp.min()) / (disp.ptp() + 1e-8); cmap_in = None
    pred0 = pred[0].numpy(); err0 = err[0].numpy(); vac0 = vac[0].numpy()

    ax_row[0].imshow(disp, cmap=cmap_in); ax_row[0].set_ylabel(title, fontsize=11)
    ax_row[1].imshow(pred0, cmap='viridis')
    ax_row[2].imshow(err0, cmap='Reds', vmin=0, vmax=1)
    # vacuity rescaled to its own range
    lo, hi = np.percentile(vac0, 1), np.percentile(vac0, 99)
    ax_row[3].imshow(np.clip((vac0 - lo) / (hi - lo + 1e-12), 0, 1), cmap='magma')
    # vacuity on a fixed [0,1] gate scale (the collapse: near-uniform)
    ax_row[4].imshow(vac0, cmap='magma', vmin=0.0, vmax=1.0)
    # annotate the true range
    ax_row[3].set_xlabel(f"range=[{vac0.min():.1e},{vac0.max():.1e}]", fontsize=7)
    for a in ax_row:
        a.set_xticks([]); a.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prostate_ckpt"); ap.add_argument("--prostate_root")
    ap.add_argument("--fundus_ckpt");   ap.add_argument("--fundus_root")
    ap.add_argument("--prostate_bidmc_domain", type=int, default=1)
    ap.add_argument("--fundus_domain", type=int, default=1)
    ap.add_argument("--domain_num_prostate", type=int, default=6)
    ap.add_argument("--domain_num_fundus", type=int, default=4)
    ap.add_argument("--prostate_img_size", type=int, default=384,
                    help="match the --img_size used at training for prostate")
    ap.add_argument("--fundus_img_size", type=int, default=256,
                    help="match the --img_size used at training for fundus")
    ap.add_argument("--out", default="Figure_2.png")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs GPU"

    rows = []
    if args.prostate_ckpt:
        m = load_model("prostate", args.prostate_ckpt, args.domain_num_prostate)
        ld = build_loader("prostate", args.prostate_root, args.prostate_bidmc_domain, args.prostate_img_size)
        rows.append(("Prostate (BIDMC)", pick_informative_slice(m, ld, "prostate")))
    if args.fundus_ckpt:
        m = load_model("fundus", args.fundus_ckpt, args.domain_num_fundus)
        ld = build_loader("fundus", args.fundus_root, args.fundus_domain, args.fundus_img_size)
        rows.append(("Fundus (optic cup)", pick_informative_slice(m, ld, "fundus", want_cup=True)))

    rows = [(t, p) for t, p in rows if p is not None]
    assert rows, "no slices found; check ckpt/root/domain args"

    fig, axes = plt.subplots(len(rows), 5, figsize=(13, 2.7 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    col_titles = ["Input", "Prediction", "Error",
                  r"$U_{\mathrm{epi}}$ (own range)",
                  r"$U_{\mathrm{epi}}$ (fixed [0,1])"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=10)
    for i, (t, packed) in enumerate(rows):
        row_panels(axes[i], packed, t)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
