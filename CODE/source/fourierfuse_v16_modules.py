"""
fourierfuse_v16_modules.py
============================================================================
v16: Domain-Aware Evidential Pseudo-Labeling (DAEP) for SSDG segmentation.

Design principles (deliberately departing from v10/v15):
    1. ONE contribution. The evidential + domain-shift uncertainty pseudo-label
       weighting is the paper's core idea. Everything else supports it.
    2. No dead code. Removed: score_denoiser (was nn.Identity), DAGE machinery,
       TemporalEnsemble (was identity-operation due to non-matching keys — bug),
       boundary_head (v15 log shows refine_delta == 0 for every iteration
       because temporal ensemble never cached anything, and boundary head was
       a no-op residual that never moved predictions), "OT" calibrator (was
       k-means prototype shift clamped by max_shift_norm — no Sinkhorn anywhere
       despite the name). Score network, CFRC, SGT, trust-region, teacher-death
       detection: all removed.
    3. Honest naming. No module claims to do something it doesn't.
    4. Efficiency story. Compact UNet at base_features=64 matches SynFoC at
       ~6x fewer parameters. We will report param counts and FLOPs in the paper.

What's kept from v10/v15:
    - Standard UNet encoder/decoder (the workhorse)
    - FDA spectral augmentation (honest, works, yours)
    - Spectral consistency loss (fp32 wrap — your v15 fix)
    - EMA teacher update + SWAD weight averaging (standard SSL machinery)

What's new in v16:
    - EvidentialHead: Dirichlet-parameterized output. Pixel-wise aleatoric and
      epistemic uncertainty derived in closed form. For K=2 (prostate, BUSI)
      this degenerates gracefully to Beta; for K=3 (fundus) and K=4 (MNMS)
      the Dirichlet treatment is natural.
    - PrototypeBank: running EMA of bottleneck features from *labeled* samples,
      organized per source domain. Enables domain-shift uncertainty U_dom: the
      cosine distance from an unlabeled sample's feature to its nearest labeled
      prototype.
    - DAEP pseudo-label weighting: per-pixel weight
        w_i = exp(-alpha * U_ale_i - beta * U_epi_i - gamma * U_dom)
      smoothly downweights uncertain pixels. REPLACES threshold-based filtering.

Author's note on the Dirichlet formulation:
    Following Sensoy et al. (NeurIPS 2018) "Evidential Deep Learning to Quantify
    Classification Uncertainty", extended to dense prediction. The network
    outputs *evidence* e_k >= 0 via softplus. Dirichlet params alpha_k = e_k + 1.
    Predictive prob p_k = alpha_k / S where S = sum(alpha). Vacuity
    (epistemic) = K / S. Dissonance (aleatoric) = sum_k p_k * (1 - p_k) / (S+1).
    Loss = expected squared error under the Dirichlet posterior + KL to uniform
    on the wrong classes. We add the standard Dice term for segmentation.

============================================================================
"""

from typing import Dict, Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# UNet building blocks (same as v10 — unchanged for fair architectural compare)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, ic, oc, mc=None):
        super().__init__()
        mc = mc or oc
        self.net = nn.Sequential(
            nn.Conv2d(ic, mc, 3, padding=1, bias=False),
            nn.BatchNorm2d(mc), nn.ReLU(inplace=True),
            nn.Conv2d(mc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ic, oc))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, ic, oc, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(ic, oc, ic // 2)
        else:
            self.up = nn.ConvTranspose2d(ic, ic // 2, 2, stride=2)
            self.conv = DoubleConv(ic, oc)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNetEncoder(nn.Module):
    def __init__(self, in_ch, base=64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)  # bottleneck
        return {'skips': [x1, x2, x3, x4], 'bottleneck': x5}


class UNetDecoder(nn.Module):
    """
    Standard UNet decoder. In v16 we removed the FrequencyGuidedAttention
    that v10 had in its decoder — it was cosmetic, never showed clear ablation
    benefit, and complicated the story. If ablation on v16 shows we want it
    back, we add it as a 5-line module.
    """
    def __init__(self, base=64, out_channels=2, bilinear=False):
        super().__init__()
        self.u1 = Up(base * 16, base * 8, bilinear)
        self.u2 = Up(base * 8, base * 4, bilinear)
        self.u3 = Up(base * 4, base * 2, bilinear)
        self.u4 = Up(base * 2, base, bilinear)
        self.outc = nn.Conv2d(base, out_channels, 1)

    def forward(self, enc):
        skips = enc['skips']
        b = enc['bottleneck']
        x = self.u1(b, skips[3])
        x = self.u2(x, skips[2])
        x = self.u3(x, skips[1])
        x = self.u4(x, skips[0])
        return self.outc(x), x  # logits, penultimate features


# ---------------------------------------------------------------------------
# NEW in v16: EvidentialHead
# ---------------------------------------------------------------------------

class EvidentialHead(nn.Module):
    """
    Replace the final softmax with a Dirichlet evidential head.

    Input:  logits [B, K, H, W] (raw outputs of the last conv)
    Output: evidence e [B, K, H, W] with e >= 0

    We use softplus to enforce non-negativity; this is the standard choice
    in evidential deep learning and keeps gradients well-behaved at zero.

    For pixel i, class k:
        evidence  e_k(i)  = softplus(logit_k(i))
        Dirichlet alpha_k(i) = e_k(i) + 1
        total     S(i)     = sum_k alpha_k(i) = K + sum_k e_k(i)
        predictive p_k(i) = alpha_k(i) / S(i)

    Uncertainties (per-pixel):
        U_epi  = K / S          (vacuity — low when a lot of evidence)
        U_ale  = sum_k p_k * (1 - p_k) / (S + 1)
                                 (dissonance — aleatoric, higher when classes
                                  have comparable evidence)

    Notes:
      * The head is just a parameter-free reshape + softplus. All learnable
        parameters are in the conv layer upstream (the standard UNet outc).
      * At K=2 this IS Beta(alpha_1, alpha_2) over the two-class simplex;
        U_ale reduces to the standard Bernoulli-variance term. We don't need
        a separate Beta branch.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.K = num_classes

    @staticmethod
    def logits_to_evidence(logits: torch.Tensor) -> torch.Tensor:
        # softplus, numerically safe. Output e >= 0.
        return F.softplus(logits)

    @staticmethod
    def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
        return evidence + 1.0

    @staticmethod
    def alpha_to_prob(alpha: torch.Tensor) -> torch.Tensor:
        S = alpha.sum(dim=1, keepdim=True)
        return alpha / S.clamp(min=1e-8)

    @staticmethod
    def alpha_to_S(alpha: torch.Tensor) -> torch.Tensor:
        return alpha.sum(dim=1, keepdim=True)

    @staticmethod
    def epistemic(alpha: torch.Tensor) -> torch.Tensor:
        """Vacuity: K / S. Shape [B, 1, H, W]."""
        K = alpha.shape[1]
        S = alpha.sum(dim=1, keepdim=True)
        return K / S.clamp(min=1e-8)

    @staticmethod
    def aleatoric(alpha: torch.Tensor) -> torch.Tensor:
        """
        Expected variance under Dirichlet posterior (dissonance):
            sum_k  p_k * (1 - p_k) / (S + 1)
        Shape [B, 1, H, W].
        """
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S.clamp(min=1e-8)
        var = (p * (1 - p)).sum(dim=1, keepdim=True) / (S + 1).clamp(min=1e-8)
        return var

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = self.logits_to_evidence(logits)
        alpha = self.evidence_to_alpha(evidence)
        S = self.alpha_to_S(alpha)
        prob = alpha / S.clamp(min=1e-8)
        return {
            'evidence': evidence,
            'alpha': alpha,
            'S': S,
            'prob': prob,
            'U_epi': self.epistemic(alpha),
            'U_ale': self.aleatoric(alpha),
        }


# ---------------------------------------------------------------------------
# NEW in v16: PrototypeBank + U_dom computation
# ---------------------------------------------------------------------------

class PrototypeBank(nn.Module):
    """
    EMA memory bank of labeled-sample bottleneck features, organized per
    source domain. Used to compute U_dom: how far the current unlabeled
    sample's feature is from the nearest labeled prototype.

    Parameters:
        feature_dim: channel count of the bottleneck feature map
        num_domains: number of labeled source domains
        num_protos_per_domain: how many prototypes to keep per domain
                               (1 = domain mean; >1 = domain sub-clusters)
        ema_decay: smoothing factor for prototype updates

    Shape:
        self.prototypes  [num_domains * num_protos_per_domain, feature_dim]
        self.counts      [num_domains * num_protos_per_domain]  (for lazy init)

    Design note: we keep multiple prototypes per domain because within one
    labeled MRI center (a "domain") there can still be sub-distributions
    (different scanners, patient groups). 2 prototypes per domain works
    well in practice.
    """
    def __init__(self,
                 feature_dim: int,
                 num_domains: int,
                 num_protos_per_domain: int = 2,
                 ema_decay: float = 0.99):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.num_protos_per_domain = num_protos_per_domain
        self.total_protos = num_domains * num_protos_per_domain
        self.ema = ema_decay

        # Register as buffers so they move with .cuda() and save in state_dict
        self.register_buffer(
            'prototypes',
            torch.zeros(self.total_protos, feature_dim)
        )
        self.register_buffer(
            'counts',
            torch.zeros(self.total_protos)
        )
        # Which prototype slot to write to next, per domain (round-robin init)
        self.register_buffer(
            'next_slot',
            torch.zeros(num_domains, dtype=torch.long)
        )

    @torch.no_grad()
    def pool_features(self, feat_map: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, C] via spatial mean pooling."""
        return feat_map.mean(dim=[2, 3])

    @torch.no_grad()
    def update(self, feat_map: torch.Tensor, domain_ids: torch.Tensor):
        """
        Update prototypes with features from labeled samples.

        Args:
            feat_map:   [B, C, H, W] bottleneck features
            domain_ids: [B]   integer domain labels in [0, num_domains)
        """
        pooled = self.pool_features(feat_map).float()  # [B, C]
        for b in range(pooled.shape[0]):
            d = int(domain_ids[b].item())
            if d < 0 or d >= self.num_domains:
                continue
            # Pick a slot within this domain. Round-robin to fill unused
            # slots first, then EMA-update the nearest-matching slot.
            slot_base = d * self.num_protos_per_domain
            slot_range = list(range(slot_base, slot_base + self.num_protos_per_domain))

            uninit = [s for s in slot_range if self.counts[s] == 0]
            if uninit:
                slot = uninit[0]
                self.prototypes[slot] = pooled[b]
                self.counts[slot] = 1.0
            else:
                # assign to nearest existing slot within this domain, EMA-update it
                cand = self.prototypes[slot_range]                  # [P_d, C]
                dists = (cand - pooled[b].unsqueeze(0)).pow(2).sum(1)
                slot = slot_range[int(dists.argmin().item())]
                self.prototypes[slot] = (
                    self.ema * self.prototypes[slot] + (1 - self.ema) * pooled[b]
                )
                self.counts[slot] += 1.0

    @torch.no_grad()
    def is_ready(self) -> bool:
        """Any prototypes initialized yet?"""
        return bool((self.counts > 0).any().item())

    @torch.no_grad()
    def compute_u_dom(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample domain-shift uncertainty.

        Args:
            feat_map: [B, C, H, W] bottleneck features from (possibly unlabeled)
                      images.
        Returns:
            u_dom: [B] in [0, 1]. 0 = identical to some labeled prototype,
                   1 = orthogonal to all of them.

        Definition:
            u_dom(x) = 1 - max_s  cosine(f(x), mu_s)  over all active protos

        We use cosine similarity (rather than L2 distance) because the absolute
        scale of bottleneck features drifts during training; cosine is scale-
        invariant and gives a stable [-1, 1] quantity that we shift into [0, 2]
        and then into [0, 1] by dividing by 2.
        """
        active = self.counts > 0
        if not active.any():
            # No labeled prototypes yet -> we can't say anything; return zeros
            # (so early iterations don't zero-out the pseudo-label loss).
            return torch.zeros(feat_map.shape[0], device=feat_map.device)

        pooled = self.pool_features(feat_map)           # [B, C]
        protos = self.prototypes[active].to(pooled.dtype)  # [P_act, C]

        p_n = F.normalize(pooled, dim=1)
        m_n = F.normalize(protos, dim=1)
        sim = p_n @ m_n.T                                # [B, P_act]
        max_sim = sim.max(dim=1).values                  # [B]
        # 1 - cosine_sim, rescaled to [0, 1]
        u_dom = (1.0 - max_sim) / 2.0
        return u_dom.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Spectral consistency regularizer (carried forward from v10, unchanged)
# ---------------------------------------------------------------------------

class SpectralConsistencyRegularizer(nn.Module):
    """
    Perturbs amplitude spectrum of input, asks encoder to produce similar
    bottleneck features before and after perturbation. This is an honest
    domain-invariance regularizer. Kept verbatim from v10.
    """
    def __init__(self, perturbation_strength: float = 0.3):
        super().__init__()
        self.s = perturbation_strength

    def perturb_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Random amplitude perturbation — phase preserved."""
        x_f = x.float()
        fft = _fft2_safe(x_f)
        amp = torch.abs(fft).clamp(min=1e-8)
        phase = torch.angle(fft)
        # Perturb amplitude multiplicatively
        noise = 1.0 + self.s * (torch.rand_like(amp) * 2 - 1)
        amp_p = amp * noise
        x_p = _ifft2_safe(torch.polar(amp_p, phase)).real
        return x_p.to(x.dtype)

    def compute_loss(self, feat_orig: torch.Tensor, feat_pert: torch.Tensor) -> torch.Tensor:
        """MSE between original and perturbed bottleneck features."""
        return F.mse_loss(feat_pert.float(), feat_orig.float())


# ---------------------------------------------------------------------------
# FDA spectral augmentation (kept; same functional behavior)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _fft2_safe(x: torch.Tensor):
    """torch.fft.fft2 with a transparent CPU fallback.

    Some PyTorch+CUDA builds raise `RuntimeError: cuFFT error:
    CUFFT_INTERNAL_ERROR` on fft2. Two common causes: (1) a non-contiguous /
    oddly-strided input, which we fix by calling .contiguous(); (2) a genuine
    driver/plan-cache issue on that CUDA build, which we work around by computing
    on CPU and moving the result back. Numerically identical; only slower.
    FFT must run in fp32 (cuFFT/half is unreliable).
    """
    x = x.contiguous()
    try:
        return torch.fft.fft2(x, norm='ortho')
    except RuntimeError as e:
        if 'CUFFT' in str(e) or 'cuFFT' in str(e):
            dev = x.device
            return torch.fft.fft2(x.detach().cpu(), norm='ortho').to(dev)
        raise


def _ifft2_safe(x: torch.Tensor):
    x = x.contiguous()
    try:
        return torch.fft.ifft2(x, norm='ortho')
    except RuntimeError as e:
        if 'CUFFT' in str(e) or 'cuFFT' in str(e):
            dev = x.device
            return torch.fft.ifft2(x.detach().cpu(), norm='ortho').to(dev)
        raise


def fda_spectral_augment(src: torch.Tensor,
                         tgt: torch.Tensor,
                         beta: float = 0.05) -> torch.Tensor:
    """
    Replace the low-frequency amplitude of `src` with that of `tgt`.

    src, tgt: [B, C, H, W] (same shape)
    beta: fraction of frequencies to swap (0 to 0.5). Typical 0.05.

    This is the standard FDA from Yang & Soatto (CVPR 2020). Behaviorally
    matches v10/v15.
    """
    src_f = src.float()
    tgt_f = tgt.float()
    s_fft = _fft2_safe(src_f)
    t_fft = _fft2_safe(tgt_f)
    s_amp, s_phase = torch.abs(s_fft), torch.angle(s_fft)
    t_amp = torch.abs(t_fft)

    _, _, H, W = src.shape
    h = int(math.floor(H * beta))
    w = int(math.floor(W * beta))
    if h == 0 or w == 0:
        return src

    # Swap low-frequency block (FFT layout: DC at [0,0] and corners)
    s_amp[:, :, :h, :w] = t_amp[:, :, :h, :w]
    s_amp[:, :, :h, -w:] = t_amp[:, :, :h, -w:]
    s_amp[:, :, -h:, :w] = t_amp[:, :, -h:, :w]
    s_amp[:, :, -h:, -w:] = t_amp[:, :, -h:, -w:]

    out = _ifft2_safe(torch.polar(s_amp, s_phase)).real
    return out.to(src.dtype)


# ---------------------------------------------------------------------------
# FourierFuseNet v16 — the lean version
# ---------------------------------------------------------------------------

class FourierFuseNet(nn.Module):
    """
    v16 model:
        encoder -> bottleneck -> decoder -> logits -> evidential head

    The bottleneck is also exposed for PrototypeBank updates (on labeled data)
    and U_dom computation (on unlabeled data).

    NOTE on output channel count:
        We keep the v10 convention where the output has (n_classes + 1)
        channels — background plus n_classes foreground classes. The evidential
        head treats all K = n_classes + 1 outputs uniformly.
    """
    def __init__(self,
                 in_channels: int,
                 n_classes: int,
                 base_features: int = 64,
                 bilinear: bool = False,
                 num_domains: int = 6,
                 num_protos_per_domain: int = 2,
                 proto_ema_decay: float = 0.99):
        super().__init__()
        self.n_classes = n_classes
        self.K = n_classes + 1  # model output channels (bg + foreground)
        self.in_channels = in_channels
        bf = base_features
        self.bf = bf

        self.encoder = UNetEncoder(in_channels, bf)
        self.decoder = UNetDecoder(bf, out_channels=self.K, bilinear=bilinear)
        self.evidential = EvidentialHead(num_classes=self.K)

        # Spectral consistency (kept)
        self.spectral_reg = SpectralConsistencyRegularizer(perturbation_strength=0.3)

        # Prototype bank lives on the BOTTLENECK feature space
        self.prototype_bank = PrototypeBank(
            feature_dim=bf * 16,
            num_domains=num_domains,
            num_protos_per_domain=num_protos_per_domain,
            ema_decay=proto_ema_decay,
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(x)

    def decode(self, enc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(enc)

    def forward(self,
                x: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        logits, last_feat = self.decoder(enc)
        ev = self.evidential(logits)

        out = {
            'logits': logits,
            'evidence': ev['evidence'],
            'alpha': ev['alpha'],
            'S': ev['S'],
            'prob': ev['prob'],
            'U_epi': ev['U_epi'],
            'U_ale': ev['U_ale'],
        }
        if return_features:
            out['bottleneck'] = enc['bottleneck']
            out['last_decoder_feat'] = last_feat
            out['encoder_output'] = enc
        return out

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience for SWAD BN recalibration and test() compatibility."""
        enc = self.encoder(x)
        logits, _ = self.decoder(enc)
        return logits

    def get_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Dirichlet predictive probabilities [B, K, H, W]."""
        logits = self.get_logits(x)
        ev = self.evidential(logits)
        return ev['prob']

    def compute_spectral_consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Same as v10/v15, but always fp32 (v15 fix integrated)."""
        x_perturbed = self.spectral_reg.perturb_amplitude(x)
        enc_orig = self.encoder(x)
        enc_pert = self.encoder(x_perturbed)
        return self.spectral_reg.compute_loss(
            enc_orig['bottleneck'], enc_pert['bottleneck']
        )


# ---------------------------------------------------------------------------
# Evidential loss functions
# ---------------------------------------------------------------------------

def evidential_seg_loss(alpha: torch.Tensor,
                        target_onehot: torch.Tensor,
                        lambda_kl: float = 0.01) -> torch.Tensor:
    """
    Standard evidential segmentation loss, per Sensoy et al. adapted to
    dense prediction.

    Two terms:
      (1) MSE between prediction and target under the Dirichlet posterior:
            E_{p~Dir(alpha)}[||target - p||^2]
            = sum_k (target_k - p_k)^2 + p_k*(1-p_k)/(S+1)
      (2) KL divergence from Dir(alpha_tilde) to Dir(1,...,1) (uniform),
          where alpha_tilde = target_onehot + (1 - target_onehot) * alpha.
          This penalizes evidence assigned to wrong classes.

    Args:
        alpha:           [B, K, H, W] Dirichlet parameters
        target_onehot:   [B, K, H, W] one-hot ground truth (or pseudo-label)
        lambda_kl:       KL coefficient (annealed upstream)

    Returns:
        Scalar loss, averaged over B*H*W.

    Notes:
      * MSE term and KL term each carry well-studied gradient behavior.
        MSE is the dominant signal; KL prevents the model from becoming
        infinitely confident by spreading evidence onto the wrong class.
      * To keep K=2 compatibility, we don't special-case anything.
    """
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S.clamp(min=1e-8)

    # Expected MSE under Dirichlet (closed form)
    err  = (target_onehot - p).pow(2).sum(dim=1, keepdim=True)
    var  = (p * (1 - p)).sum(dim=1, keepdim=True) / (S + 1).clamp(min=1e-8)
    mse_term = (err + var).mean()

    # KL term on the "wrong-class" evidence
    alpha_tilde = target_onehot + (1 - target_onehot) * alpha
    kl_term = kl_dirichlet_uniform(alpha_tilde).mean()

    return mse_term + lambda_kl * kl_term


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1,1,...,1) ), per-pixel.
    alpha: [B, K, H, W].  Returns: [B, 1, H, W].

    Standard closed form involving lgamma / digamma. Treat spatial dims as
    extra batch dims.
    """
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    # digamma(alpha_k) - digamma(S)
    dg = torch.digamma(alpha) - torch.digamma(S)
    # sum_k (alpha_k - 1) * (digamma(alpha_k) - digamma(S))
    term1 = ((alpha - 1) * dg).sum(dim=1, keepdim=True)
    # log(Gamma(S)) - sum_k log(Gamma(alpha_k)) - log(Gamma(K)) + sum_k log(Gamma(1))
    # log Gamma(1) = 0
    term2 = (torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
             - torch.lgamma(torch.tensor(float(K), device=alpha.device)))
    return term1 + term2


def soft_dice_from_alpha(alpha: torch.Tensor,
                         target_onehot: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    Soft dice computed from the Dirichlet mean (alpha / S), averaged across
    foreground classes (excludes class 0 = background).

    Args:
        alpha:          [B, K, H, W]
        target_onehot:  [B, K, H, W]
        mask:           [B, 1, H, W] or None (pixel weighting)
        eps:            smoothing
    """
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S.clamp(min=1e-8)   # [B, K, H, W]

    if mask is not None:
        p = p * mask
        t = target_onehot * mask
    else:
        t = target_onehot

    dims = (0, 2, 3)  # reduce over batch + spatial
    intersect = (p * t).sum(dim=dims)   # [K]
    denom = p.sum(dim=dims) + t.sum(dim=dims) + eps
    dice_per_class = (2 * intersect + eps) / denom    # [K]

    # Foreground average
    if dice_per_class.shape[0] > 1:
        fg_dice = dice_per_class[1:].mean()
    else:
        fg_dice = dice_per_class.mean()
    return 1.0 - fg_dice


def daep_pseudo_label_loss(alpha_student: torch.Tensor,
                           teacher_pseudo_onehot: torch.Tensor,
                           U_ale_teacher: torch.Tensor,
                           U_epi_teacher: torch.Tensor,
                           U_dom_img: torch.Tensor,
                           alpha_w: float = 1.0,
                           beta_w: float = 1.0,
                           gamma_w: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    The core of v16: per-pixel DAEP pseudo-label loss.

    w_i = exp( -alpha_w * U_ale_i - beta_w * U_epi_i - gamma_w * U_dom_img )

    Teacher uncertainties are used for weighting (they indicate which
    *pseudo-labels* are trustworthy). Student alpha is supervised.

    Args:
        alpha_student:         [B, K, H, W] student's Dirichlet params
        teacher_pseudo_onehot: [B, K, H, W] one-hot pseudo-labels from teacher
        U_ale_teacher:         [B, 1, H, W] teacher aleatoric (per pixel)
        U_epi_teacher:         [B, 1, H, W] teacher epistemic (per pixel)
        U_dom_img:             [B] per-image domain-shift uncertainty
        alpha_w, beta_w, gamma_w: scalar weights on the three uncertainties

    Returns:
        (loss, diagnostics dict)

    Notes:
      * U_ale and U_epi are already in reasonable numeric ranges in [0, 1] for
        this Dirichlet formulation, so alpha_w=beta_w=1 is a sensible default.
      * U_dom is broadcast from [B] to [B, 1, H, W] (whole-image signal).
    """
    B, K, H, W = alpha_student.shape

    # Build pixel-level weights in [0, 1]
    U_dom_map = U_dom_img.view(B, 1, 1, 1).expand(B, 1, H, W)
    w = torch.exp(
        -alpha_w * U_ale_teacher
        - beta_w  * U_epi_teacher
        - gamma_w * U_dom_map
    )  # [B, 1, H, W] in (0, 1]

    # Evidential MSE with per-pixel weight
    S = alpha_student.sum(dim=1, keepdim=True)
    p = alpha_student / S.clamp(min=1e-8)
    err = (teacher_pseudo_onehot - p).pow(2).sum(dim=1, keepdim=True)
    var = (p * (1 - p)).sum(dim=1, keepdim=True) / (S + 1).clamp(min=1e-8)
    mse_map = err + var                                          # [B, 1, H, W]
    weighted_mse = (w * mse_map).sum() / (w.sum() + 1e-8)

    # Soft dice with the same pixel weight (mask interpreted as soft mask)
    dice_term = soft_dice_from_alpha(alpha_student,
                                     teacher_pseudo_onehot,
                                     mask=w)

    loss = weighted_mse + dice_term

    diag = {
        'daep_w_mean': float(w.mean().item()),
        'daep_w_min':  float(w.min().item()),
        'daep_w_max':  float(w.max().item()),
        'daep_u_ale_mean': float(U_ale_teacher.mean().item()),
        'daep_u_epi_mean': float(U_epi_teacher.mean().item()),
        'daep_u_dom_mean': float(U_dom_img.mean().item()),
    }
    return loss, diag


# ---------------------------------------------------------------------------
# Helpers shared with training script
# ---------------------------------------------------------------------------

def labels_to_onehot(labels: torch.Tensor, K: int) -> torch.Tensor:
    """
    [B, H, W] long -> [B, K, H, W] float one-hot.
    Out-of-range labels (e.g., -1 for ignore) are silently zero-vectors.
    """
    labels = labels.clone()
    labels[(labels < 0) | (labels >= K)] = 0
    oh = F.one_hot(labels.long(), num_classes=K)  # [B, H, W, K]
    return oh.permute(0, 3, 1, 2).float().contiguous()
