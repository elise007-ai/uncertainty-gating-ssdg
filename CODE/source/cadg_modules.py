"""
cadg_modules.py
===============
Calibration-Aware Domain-Generalized pseudo-labeling (CADG) — core mechanism.

This file implements the THREE components of the Option-B design
(see docs/DESIGN.md). The novel part is `PerDomainReliabilityCalibrator`
(Component A). The other classes are substrate (evidential head, online domain
clustering) and are deliberately small.

Everything here is plain PyTorch + NumPy and is unit-tested at the bottom
(`python cadg_modules.py` runs the self-test on CPU in a few seconds).

Design intent (short version):
  raw uncertainty u(x)  --(per-domain isotonic map g_d)-->  recalibrated u~(x)
  where g_d is fit, ONLINE and WITHOUT LABELS, so that recalibrated uncertainty
  tracks 1 - (stability of the teacher's prediction under an FDA style swap).

We never use target labels anywhere in this module.
"""

from __future__ import annotations
from typing import Dict, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Component C substrate: Evidential (Dirichlet) head — same as Option A.
# Parameter-free; turns logits into evidence and uncertainties.
# ---------------------------------------------------------------------------
class EvidentialHead(nn.Module):
    """Softplus evidence -> Dirichlet. Returns prob, vacuity (U_epi), aleatoric.

    u used by the calibrator is the Dirichlet vacuity U_epi = K / S, which is
    in (0, 1] and is the natural 'how much evidence do I lack' signal.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @staticmethod
    def logits_to_evidence(logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits)

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        e = self.logits_to_evidence(logits)                 # [B,K,H,W] >= 0
        alpha = e + 1.0
        S = alpha.sum(dim=1, keepdim=True)                  # [B,1,H,W]
        prob = alpha / S
        K = self.num_classes
        u_epi = (K / S).squeeze(1)                          # [B,H,W] in (0,1]
        # aleatoric (dissonance), kept for completeness / ablations
        u_ale = (prob * (1.0 - prob)).sum(dim=1) / (S.squeeze(1) + 1.0)
        return {"prob": prob, "U_epi": u_epi, "U_ale": u_ale, "alpha": alpha}


# ---------------------------------------------------------------------------
# Component B substrate: online domain clustering of bottleneck features.
# No domain IDs are used; we cluster EMA-teacher embeddings into n_domains
# soft clusters and use the hard assignment to index reliability histograms.
# ---------------------------------------------------------------------------
class OnlineDomainCluster(nn.Module):
    """EMA k-means over pooled bottleneck features. Assignment indexes the
    per-domain reliability histograms in the calibrator.

    This is intentionally simple and standard. Its only job is to give a stable
    integer in [0, n_clusters) per image. We over-cluster slightly (n_clusters
    >= n_source_domains) and let empty clusters be harmless.
    """

    def __init__(self, feat_dim: int, n_clusters: int, ema: float = 0.99):
        super().__init__()
        self.n_clusters = n_clusters
        self.ema = ema
        self.register_buffer("centers", torch.zeros(n_clusters, feat_dim))
        self.register_buffer("counts", torch.zeros(n_clusters))

    @staticmethod
    def _pool(feat_map: torch.Tensor) -> torch.Tensor:
        return feat_map.mean(dim=[2, 3])  # [B,C]

    @torch.no_grad()
    def assign(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Return hard cluster id per sample. Initializes empty centers."""
        h = F.normalize(self._pool(feat_map).float(), dim=1)        # [B,C]
        out = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        for b in range(h.shape[0]):
            uninit = (self.counts == 0)
            if uninit.any():
                cid = int(torch.nonzero(uninit)[0].item())
                self.centers[cid] = h[b]
                self.counts[cid] += 1.0
                out[b] = cid
                continue
            cn = F.normalize(self.centers, dim=1)
            sims = cn @ h[b]                                        # [n_clusters]
            cid = int(sims.argmax().item())
            self.centers[cid] = self.ema * self.centers[cid] + (1 - self.ema) * h[b]
            self.counts[cid] += 1.0
            out[b] = cid
        return out


# ---------------------------------------------------------------------------
# Component A (NOVEL): label-free per-domain reliability calibrator.
# ---------------------------------------------------------------------------
class PerDomainReliabilityCalibrator(nn.Module):
    """
    Maintains, per inferred domain d, an EMA 2-D histogram over
        (uncertainty bin) x (stable / unstable)
    where 'stable' is whether the teacher's argmax is unchanged under an FDA
    style swap (label-free). From it we derive, per bin, the empirical
    UNRELIABILITY  q_d(bin) = P(unstable | bin, d).

    We then fit a MONOTONE (isotonic, pool-adjacent-violators) map
        g_d : raw_uncertainty -> recalibrated_uncertainty
    by mapping each bin's center-uncertainty to q_d(bin), enforcing
    non-decreasing output, and linearly interpolating between bin centers.

    Key properties:
      * No labels are ever used.
      * Per-domain: each d gets its own g_d. On a calibrated domain g_d ~ identity;
        on a shifted domain g_d is shifted/steepened.
      * Cheap: histograms updated every step (just scatter-adds); g_d refit
        every `refit_every` steps via PAVA (O(n_bins)).
      * Safe early: until a domain has >= min_count observations, g_d = identity,
        so early training behaves exactly like the un-recalibrated gate.

    Usage in the training loop (per unlabeled batch):
        cal.update(domain_ids, u_raw_pixelwise, stable_pixelwise)   # every step
        if step % refit_every == 0: cal.refit()
        u_tilde = cal.apply(domain_ids, u_raw_pixelwise)            # recalibrated
        weight  = (u_tilde < tau)   OR   exp(-lam * u_tilde)
    """

    def __init__(self,
                 n_domains: int,
                 n_bins: int = 20,
                 ema: float = 0.99,
                 min_count: float = 2000.0,
                 refit_every: int = 200,
                 subsample_pixels: int = 4096):
        super().__init__()
        self.n_domains = n_domains
        self.n_bins = n_bins
        self.ema = ema
        self.min_count = min_count
        self.refit_every = refit_every
        self.subsample_pixels = subsample_pixels

        # hist[d, bin, 0] = soft count of STABLE pixels; [...,1] = UNSTABLE
        self.register_buffer("hist", torch.zeros(n_domains, n_bins, 2))
        # fitted maps: for each domain, the recalibrated value at each bin center
        # initialized to identity (bin center value)
        bin_centers = (torch.arange(n_bins).float() + 0.5) / n_bins
        self.register_buffer("bin_centers", bin_centers)
        self.register_buffer("g_values", bin_centers.unsqueeze(0).repeat(n_domains, 1).clone())
        self.register_buffer("domain_total", torch.zeros(n_domains))

    # -- helpers ----------------------------------------------------------
    def _bin_index(self, u: torch.Tensor) -> torch.Tensor:
        idx = (u.clamp(0, 1 - 1e-6) * self.n_bins).long()
        return idx.clamp(0, self.n_bins - 1)

    # -- online update ----------------------------------------------------
    @torch.no_grad()
    def update(self,
               domain_ids: torch.Tensor,    # [B]
               u_raw: torch.Tensor,         # [B,H,W] raw uncertainty in [0,1]
               stable: torch.Tensor):       # [B,H,W] in {0,1}; 1 = stable
        """EMA-accumulate the per-domain reliability histogram from a batch.

        We optionally subsample pixels per image for speed; the histogram is an
        EMA so subsampling just adds a little variance.
        """
        B = u_raw.shape[0]
        for b in range(B):
            d = int(domain_ids[b].item())
            if d < 0 or d >= self.n_domains:
                continue
            u = u_raw[b].reshape(-1)
            s = stable[b].reshape(-1).float()
            if self.subsample_pixels and u.numel() > self.subsample_pixels:
                sel = torch.randperm(u.numel(), device=u.device)[: self.subsample_pixels]
                u = u[sel]; s = s[sel]
            bins = self._bin_index(u)
            # scatter counts for this image
            stable_counts = torch.zeros(self.n_bins, device=u.device)
            unstable_counts = torch.zeros(self.n_bins, device=u.device)
            stable_counts.scatter_add_(0, bins, s)
            unstable_counts.scatter_add_(0, bins, (1.0 - s))
            # EMA blend into the running histogram
            self.hist[d, :, 0].mul_(self.ema).add_((1 - self.ema) * stable_counts)
            self.hist[d, :, 1].mul_(self.ema).add_((1 - self.ema) * unstable_counts)
            self.domain_total[d] += float(u.numel())

    # -- isotonic fit (pool-adjacent-violators) ---------------------------
    @staticmethod
    def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Weighted isotonic regression (non-decreasing) via PAVA.
        y: target values per bin; w: weights per bin. Returns fitted y_hat.
        """
        n = len(y)
        # blocks: value, weight, count
        vals = list(y.astype(float))
        wts = list(w.astype(float))
        # guard zero weights -> tiny
        wts = [max(x, 1e-8) for x in wts]
        # PAVA
        i = 0
        level_val = []
        level_w = []
        level_n = []
        for j in range(n):
            level_val.append(vals[j]); level_w.append(wts[j]); level_n.append(1)
            while len(level_val) > 1 and level_val[-2] > level_val[-1]:
                v2 = level_val.pop(); w2 = level_w.pop(); n2 = level_n.pop()
                v1 = level_val.pop(); w1 = level_w.pop(); n1 = level_n.pop()
                nv = (v1 * w1 + v2 * w2) / (w1 + w2)
                level_val.append(nv); level_w.append(w1 + w2); level_n.append(n1 + n2)
        out = []
        for v, _, c in zip(level_val, level_w, level_n):
            out.extend([v] * c)
        return np.asarray(out, dtype=float)

    @torch.no_grad()
    def refit(self):
        """Refit g_d for every domain that has enough observations."""
        for d in range(self.n_domains):
            if float(self.domain_total[d]) < self.min_count:
                continue  # keep identity
            stable = self.hist[d, :, 0].cpu().numpy()
            unstable = self.hist[d, :, 1].cpu().numpy()
            total = stable + unstable
            # empirical unreliability per bin; bins with no mass -> NaN -> fill by interp
            with np.errstate(invalid="ignore", divide="ignore"):
                q = unstable / np.maximum(total, 1e-8)
            w = total
            # bins with no mass: set weight ~0 and value to neighbor (handled by PAVA weight)
            # isotonic (non-decreasing) fit of q vs bin index, weighted by mass
            q_hat = self._pava(np.nan_to_num(q, nan=0.0), w)
            # clip to [0,1]
            q_hat = np.clip(q_hat, 0.0, 1.0)
            self.g_values[d] = torch.from_numpy(q_hat).to(self.g_values.device).float()

    # -- apply ------------------------------------------------------------
    @torch.no_grad()
    def apply(self, domain_ids: torch.Tensor, u_raw: torch.Tensor) -> torch.Tensor:
        """Map raw uncertainty -> recalibrated uncertainty per domain via g_d,
        with linear interpolation between bin centers. Shapes preserved.
        """
        out = torch.empty_like(u_raw)
        B = u_raw.shape[0]
        centers = self.bin_centers  # [n_bins]
        for b in range(B):
            d = int(domain_ids[b].item())
            if d < 0 or d >= self.n_domains:
                out[b] = u_raw[b]
                continue
            g = self.g_values[d]                      # [n_bins]
            u = u_raw[b].clamp(0, 1)
            # piecewise-linear interp of (centers -> g) at query points u
            out[b] = self._interp1d(centers, g, u)
        return out

    @staticmethod
    def _interp1d(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """1-D linear interpolation; xp ascending in [0,1]. x any shape in [0,1]."""
        x_flat = x.reshape(-1)
        idx = torch.bucketize(x_flat, xp)
        idx = idx.clamp(1, len(xp) - 1)
        x0 = xp[idx - 1]; x1 = xp[idx]
        y0 = fp[idx - 1]; y1 = fp[idx]
        denom = (x1 - x0).clamp_min(1e-8)
        t = (x_flat - x0) / denom
        y = y0 + t * (y1 - y0)
        return y.reshape(x.shape)


# ---------------------------------------------------------------------------
# Convenience: the label-free stability proxy.
# ---------------------------------------------------------------------------
@torch.no_grad()
def stability_proxy(prob_view1: torch.Tensor, prob_view2: torch.Tensor) -> torch.Tensor:
    """1 if argmax matches across the two style views, else 0. Shapes [B,K,H,W]
    -> [B,H,W]. This is the label-free correctness surrogate."""
    a = prob_view1.argmax(dim=1)
    b = prob_view2.argmax(dim=1)
    return (a == b).float()


# ===========================================================================
# Self-test (CPU). Run: python cadg_modules.py
# The critical test SIMULATES a miscalibrated domain and checks that the
# label-free isotonic recalibration recovers the correct reliability ranking.
# ===========================================================================
def _selftest():
    torch.manual_seed(0); np.random.seed(0)
    print("[selftest] evidential head shapes...")
    head = EvidentialHead(num_classes=2)
    logits = torch.randn(3, 2, 16, 16)
    out = head(logits)
    assert out["prob"].shape == (3, 2, 16, 16)
    assert out["U_epi"].shape == (3, 16, 16)
    assert (out["U_epi"] > 0).all() and (out["U_epi"] <= 1.0 + 1e-5).all()
    print("    ok.")

    print("[selftest] domain clustering returns valid ids...")
    odc = OnlineDomainCluster(feat_dim=8, n_clusters=4)
    feats = torch.randn(5, 8, 4, 4)
    ids = odc.assign(feats)
    assert ids.shape == (5,) and ids.min() >= 0 and ids.max() < 4
    print("    ok.")

    print("[selftest] CRITICAL: label-free isotonic recalibration on a "
          "simulated miscalibrated domain...")
    # Construct a domain where TRUE unreliability increases with u, but the
    # raw u is mis-scaled (e.g., compressed into [0,0.3]). The stability proxy
    # is generated to match true unreliability. We check that after refit,
    # g_d is monot, and that applying g_d SEPARATES reliable from unreliable
    # better than raw u (higher AUROC of recalibrated-u vs unreliability).
    cal = PerDomainReliabilityCalibrator(n_domains=1, n_bins=20, ema=0.0,
                                         min_count=10.0, refit_every=1,
                                         subsample_pixels=0)
    N = 20000
    # raw uncertainty compressed into [0, 0.3]
    u_true_latent = torch.rand(N)                  # the "real" unreliability driver
    u_raw = (u_true_latent * 0.3)                  # mis-scaled raw uncertainty
    # stability: unstable (0) with prob = u_true_latent (true unreliability)
    unstable_prob = u_true_latent
    stable = (torch.rand(N) > unstable_prob).float()   # 1 = stable
    # feed as a single "image"
    dom = torch.zeros(1, dtype=torch.long)
    cal.update(dom, u_raw.view(1, N, 1), stable.view(1, N, 1))
    cal.refit()
    u_tilde = cal.apply(dom, u_raw.view(1, N, 1)).view(N)

    # g_d must be non-decreasing
    g = cal.g_values[0].numpy()
    assert np.all(np.diff(g) >= -1e-6), "isotonic map not monotone!"

    # AUROC of (score) vs (unstable label). Recalibrated should be >= raw,
    # and crucially should be high (~ matches true ranking).
    def auroc(score, label):
        score = score.numpy(); label = label.numpy()
        order = np.argsort(score)
        label = label[order]
        n_pos = label.sum(); n_neg = len(label) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        # rank-sum
        ranks = np.argsort(np.argsort(score)) + 1
        auc = (ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return auc

    unstable = 1.0 - stable
    auc_raw = auroc(u_raw, unstable)
    auc_cal = auroc(u_tilde, unstable)
    print(f"    AUROC raw-u vs unreliability   = {auc_raw:.3f}")
    print(f"    AUROC recal-u vs unreliability = {auc_cal:.3f}")
    # Monotone maps preserve ranking, so AUROC should be ~equal; the POINT of
    # recalibration is not AUROC (ranking) but SCALE: recalibrated u should now
    # span [0,1] tracking true unreliability, so a FIXED threshold works.
    # Check scale recovery: mean recalibrated-u in top-unreliability decile
    # should be high, bottom decile low.
    top = u_tilde[u_true_latent > 0.9].mean().item()
    bot = u_tilde[u_true_latent < 0.1].mean().item()
    print(f"    recal-u mean: bottom-decile={bot:.3f}  top-decile={top:.3f}")
    assert top > 0.6 and bot < 0.4, (
        "recalibration failed to rescale uncertainty to a usable range")
    assert auc_cal >= auc_raw - 0.02, "recalibration hurt ranking"
    print("    ok — label-free recalibration recovered a usable uncertainty scale.")

    print("[selftest] identity behavior before min_count is reached...")
    cal2 = PerDomainReliabilityCalibrator(n_domains=1, n_bins=20, min_count=1e9)
    u = torch.rand(1, 32, 32)
    u_t = cal2.apply(torch.zeros(1, dtype=torch.long), u)
    assert torch.allclose(u, u_t, atol=1e-4), "should be identity before refit"
    print("    ok.")

    print("\n[selftest] ALL PASSED.")


if __name__ == "__main__":
    _selftest()
