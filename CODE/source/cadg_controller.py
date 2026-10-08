"""
cadg_controller.py  (Option B integration for FourierFuse v16)
==============================================================
Drop-in controller that adds Calibration-Aware Domain-Generalized pseudo-labeling
(CADG) to the FourierFuse v16 training loop, WITHOUT a foundation model.

It is wired against the REAL fourierfuse_v16_modules / train_fourierfuse_v16.py
data flow:

  * the EMA teacher forward already returns {'prob','U_ale','U_epi','bottleneck'}
  * fda_spectral_augment(x, ref, beta) is the project's own FDA op (content
    preserved, style swapped) — we reuse it to make a label-free style view.

The mechanism (see docs/DESIGN.md):

  1. Infer a soft domain id per unlabeled image from the teacher bottleneck
     (OnlineDomainCluster, Component B).
  2. Build a label-free correctness proxy: teacher argmax STABILITY across an
     FDA style swap (Component A signal).
  3. Maintain per-domain EMA reliability histograms over (U_epi bin x stable?)
     and fit per-domain isotonic recalibration maps g_d (Component A, NOVEL).
  4. Recalibrate the teacher's per-pixel U_epi (and optionally U_ale) with g_d,
     so the DAEP weight  w = exp(-a*U_ale - b*U_epi_recal - g*U_dom)  operates at
     a per-domain-correct scale.

The controller returns recalibrated U_ale / U_epi maps that the training loop
feeds straight into the EXISTING daep_pseudo_label_loss — so the change to the
loop is minimal and the rest of FourierFuse is untouched.

All methods are no_grad; this adds negligible overhead (one extra teacher
forward on the styled view, every step).
"""
from __future__ import annotations
from typing import Dict, Optional
import torch

from cadg_modules import (OnlineDomainCluster, PerDomainReliabilityCalibrator,
                          stability_proxy)


class CADGController:
    def __init__(self,
                 feat_dim: int,
                 n_domain_clusters: int,
                 n_bins: int = 20,
                 refit_every: int = 200,
                 min_count: float = 50000.0,
                 recalibrate_ale: bool = False,
                 per_domain: bool = True,
                 device: str = "cuda"):
        """
        feat_dim          : teacher bottleneck channels (1024 for base=64).
        n_domain_clusters : >= number of domains expected (use args.domain_num).
        per_domain        : if False, collapse to a single shared map
                            (this is the `cadg_global` ablation).
        recalibrate_ale   : also recalibrate U_ale (default: only U_epi, the
                            epistemic/vacuity term that the BIDMC analysis
                            implicated).
        """
        self.per_domain = per_domain
        n_dom = n_domain_clusters if per_domain else 1
        self.cluster = OnlineDomainCluster(feat_dim, n_domain_clusters).to(device)
        self.cal_epi = PerDomainReliabilityCalibrator(
            n_domains=n_dom, n_bins=n_bins, refit_every=refit_every,
            min_count=min_count).to(device)
        self.recalibrate_ale = recalibrate_ale
        self.cal_ale = (PerDomainReliabilityCalibrator(
            n_domains=n_dom, n_bins=n_bins, refit_every=refit_every,
            min_count=min_count).to(device) if recalibrate_ale else None)
        self.refit_every = refit_every
        self.device = device
        self._last: Dict[str, float] = {}

    @torch.no_grad()
    def recalibrate(self,
                    teacher_prob: torch.Tensor,        # [B,K,H,W]
                    teacher_prob_styled: torch.Tensor, # [B,K,H,W] (FDA view)
                    teacher_U_epi: torch.Tensor,       # [B,1,H,W] or [B,H,W]
                    teacher_U_ale: torch.Tensor,       # [B,1,H,W] or [B,H,W]
                    teacher_bottleneck: torch.Tensor,  # [B,C,h,w]
                    iter_num: int) -> Dict[str, torch.Tensor]:
        """Returns recalibrated {'U_epi','U_ale'} maps with the SAME shape as the
        inputs (so they are drop-in for daep_pseudo_label_loss). Also updates the
        per-domain histograms and refits maps on schedule."""
        # normalize shapes to [B,H,W] internally
        epi = teacher_U_epi.squeeze(1) if teacher_U_epi.dim() == 4 else teacher_U_epi
        ale = teacher_U_ale.squeeze(1) if teacher_U_ale.dim() == 4 else teacher_U_ale

        # Component B: domain cluster ids
        dom_ids = self.cluster.assign(teacher_bottleneck)        # [B]
        if not self.per_domain:
            dom_ids = torch.zeros_like(dom_ids)

        # Component A signal: label-free stability under style swap
        stable = stability_proxy(teacher_prob, teacher_prob_styled)   # [B,H,W]

        # update + (scheduled) refit
        self.cal_epi.update(dom_ids, epi.clamp(0, 1), stable)
        if self.cal_ale is not None:
            self.cal_ale.update(dom_ids, ale.clamp(0, 1), stable)
        if iter_num > 0 and iter_num % self.refit_every == 0:
            self.cal_epi.refit()
            if self.cal_ale is not None:
                self.cal_ale.refit()

        # apply
        epi_recal = self.cal_epi.apply(dom_ids, epi.clamp(0, 1))
        ale_recal = (self.cal_ale.apply(dom_ids, ale.clamp(0, 1))
                     if self.cal_ale is not None else ale)

        self._last = {
            "cadg_u_epi_raw_mean": float(epi.mean().item()),
            "cadg_u_epi_recal_mean": float(epi_recal.mean().item()),
            "cadg_stable_frac": float(stable.mean().item()),
            "cadg_n_active_clusters": int((self.cluster.counts > 0).sum().item()),
        }

        # restore original [B,1,H,W] shape if that's what came in
        if teacher_U_epi.dim() == 4:
            epi_recal = epi_recal.unsqueeze(1)
            ale_recal = ale_recal.unsqueeze(1) if self.cal_ale is not None else teacher_U_ale
        return {"U_epi": epi_recal, "U_ale": ale_recal, "domain_ids": dom_ids}

    def last_stats(self) -> Dict[str, float]:
        return dict(self._last)
