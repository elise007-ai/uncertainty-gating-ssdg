"""
revision_modules.py
============================================================================
BSPC-D-26-12287 major-revision additions (session S01, 2026-09-25).

Everything the reviewers asked for that needs NEW model/loss code lives here,
so that `fourierfuse_v16_modules.py` (the code that produced every number in
the submitted manuscript) stays byte-identical.

Contents
--------
1. EvidentialHeadV18      configurable evidence activation (softplus / relu /
                          exp), logit temperature, Dirichlet base-rate weight w
                          (alpha = e + w, vacuity = K*w/S), optional evidence
                          cap. With defaults (softplus, T=1, w=1, no cap) it is
                          numerically identical to v16's EvidentialHead.
2. SoftmaxHead            plain softmax head (true FixMatch baseline). Exposes
                          the same dict keys as the evidential head so the
                          training loop and diagnostics do not branch on it.
3. UNetEncoderV18 / UNetDecoderV18 / ResUNetEncoderV18
                          v16 UNet with optional UA-MT-style Dropout2d (bottleneck +
                          last up block; dropout_place='v18' keeps the S06
                          bottleneck + first-up-block placement, E-20) and a
                          residual-block encoder variant
                          used as the second (heterogeneous) MEDL network.
4. FourierFuseNetV18      same attribute names as v16's FourierFuseNet
                          (encoder/decoder/evidential/spectral_reg/prototype_bank)
                          so v16 checkpoints load with strict=True when the
                          config is the v16 default.
5. Evidential losses      Sensoy's three Bayes-risk forms (mse / ce / log) with
                          the wrong-class KL to Dir(w,...,w); DEviS CUP loss;
                          I-EDL (Fisher) loss used by MEDL.
6. MC-dropout helpers     UA-MT uncertainty gate (predictive entropy of the
                          mean over T stochastic passes; threshold ramp).
7. MEDL helpers           class-aware evidential fusion (CAEF, MEDL eq.3),
                          reliability (eq.4), asymptotic rank weight (eq.6).

Every formula is annotated with the equation number of the paper it comes
from. Anything the source paper leaves unspecified is marked [UNVERIFIED] in
the docstring and exposed as a CLI flag in train_fourierfuse_v18_revision.py.

Self-test:  python revision_modules.py --selftest   (CPU, < 1 min)
============================================================================
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the UNet building blocks, spectral regularizer, FDA op, prototype bank
# and soft-dice from the frozen v16 module so the architecture is provably the
# same object.
from fourierfuse_v16_modules import (
    DoubleConv, Down, Up,
    PrototypeBank, SpectralConsistencyRegularizer,
    soft_dice_from_alpha, labels_to_onehot,
)


# ===========================================================================
# 1. Evidential head with configurable parameterisation
# ===========================================================================

class EvidentialHeadV18(nn.Module):
    """Dirichlet head: logits -> evidence -> alpha -> (prob, U_epi, U_ale).

    activation : 'softplus' (v16 default; DEviS, TBraTS),
                 'relu'     (Sensoy et al. 2018, the original EDL),
                 'exp'      (clamped exp, used by several EDL follow-ups).
    temperature: logits are divided by T before the activation (T=1: off).
    prior_w    : Dirichlet base-rate weight W (subjective logic, DEviS eq.6):
                 alpha_k = e_k + W. Vacuity generalises to U_epi = K*W / S so
                 that sum_k b_k + u = 1 still holds (b_k = e_k / S).
    max_evidence: optional cap e_k <= max_evidence (0 = off). A cap bounds S
                 from above and therefore bounds the vacuity from BELOW, i.e.
                 it is the most direct intervention against range collapse.

    With (softplus, T=1, W=1, cap=0) the outputs are identical to v16's
    EvidentialHead (checked in selftest).
    """

    def __init__(self, num_classes: int, activation: str = 'softplus',
                 temperature: float = 1.0, prior_w: float = 1.0,
                 max_evidence: float = 0.0):
        super().__init__()
        assert activation in ('softplus', 'relu', 'exp'), activation
        assert temperature > 0 and prior_w > 0
        self.K = num_classes
        self.activation = activation
        self.temperature = float(temperature)
        self.prior_w = float(prior_w)
        self.max_evidence = float(max_evidence)

    def logits_to_evidence(self, logits: torch.Tensor) -> torch.Tensor:
        z = logits / self.temperature if self.temperature != 1.0 else logits
        if self.activation == 'softplus':
            e = F.softplus(z)
        elif self.activation == 'relu':
            e = F.relu(z)
        else:  # exp, clamped for fp16 safety
            e = torch.exp(torch.clamp(z, max=10.0))
        if self.max_evidence > 0:
            e = torch.clamp(e, max=self.max_evidence)
        return e

    def evidence_to_alpha(self, evidence: torch.Tensor) -> torch.Tensor:
        return evidence + self.prior_w

    def epistemic(self, alpha: torch.Tensor) -> torch.Tensor:
        """Vacuity K*W/S, shape [B,1,H,W]."""
        S = alpha.sum(dim=1, keepdim=True)
        return (self.K * self.prior_w) / S.clamp(min=1e-8)

    @staticmethod
    def aleatoric(alpha: torch.Tensor) -> torch.Tensor:
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S.clamp(min=1e-8)
        return (p * (1 - p)).sum(dim=1, keepdim=True) / (S + 1).clamp(min=1e-8)

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = self.logits_to_evidence(logits)
        alpha = self.evidence_to_alpha(evidence)
        S = alpha.sum(dim=1, keepdim=True)
        prob = alpha / S.clamp(min=1e-8)
        return {'evidence': evidence, 'alpha': alpha, 'S': S, 'prob': prob,
                'U_epi': self.epistemic(alpha), 'U_ale': self.aleatoric(alpha),
                'belief': evidence / S.clamp(min=1e-8)}


# ===========================================================================
# 2. Softmax head (true FixMatch baseline)
# ===========================================================================

class SoftmaxHead(nn.Module):
    """Plain softmax. For interface parity we also emit
    U_epi := 1 - max_k p_k  (confidence deficit)  and
    U_ale := normalised predictive entropy H(p)/log K.
    'alpha' is set to prob so that soft_dice_from_alpha(alpha, ...) works
    unchanged (it only uses alpha / alpha.sum()).
    """

    def __init__(self, num_classes: int, temperature: float = 1.0):
        super().__init__()
        self.K = num_classes
        self.temperature = float(temperature)

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = logits / self.temperature if self.temperature != 1.0 else logits
        prob = F.softmax(z.float(), dim=1)
        maxp = prob.max(dim=1, keepdim=True).values
        ent = -(prob * torch.log(prob.clamp(min=1e-10))).sum(dim=1, keepdim=True)
        return {'evidence': None, 'alpha': prob, 'S': torch.ones_like(maxp),
                'prob': prob, 'U_epi': 1.0 - maxp,
                'U_ale': ent / math.log(self.K), 'belief': prob}


# ===========================================================================
# 3. Backbones: v16 UNet (+ optional dropout) and a residual encoder variant
# ===========================================================================

class UNetEncoderV18(nn.Module):
    """Identical to v16 UNetEncoder; `dropout_p` adds nn.Dropout2d on the
    bottleneck (UA-MT placement). Module names are kept so v16 checkpoints load."""

    def __init__(self, in_ch, base=64, dropout_p: float = 0.0):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.dropout_p = float(dropout_p)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.drop(self.d4(x4))
        return {'skips': [x1, x2, x3, x4], 'bottleneck': x5}


class ResDoubleConv(nn.Module):
    """DoubleConv with a 1x1 projection shortcut (residual block)."""

    def __init__(self, ic, oc):
        super().__init__()
        self.body = DoubleConv(ic, oc)
        self.short = (nn.Identity() if ic == oc else
                      nn.Sequential(nn.Conv2d(ic, oc, 1, bias=False), nn.BatchNorm2d(oc)))

    def forward(self, x):
        return F.relu(self.body(x) + self.short(x))


class ResUNetEncoderV18(nn.Module):
    """Residual-block encoder (MEDL's 'second architecture with a different
    inductive bias' analogue in 2D; MEDL used VNet + 3D-ResVNet)."""

    def __init__(self, in_ch, base=64, dropout_p: float = 0.0):
        super().__init__()
        self.inc = ResDoubleConv(in_ch, base)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(base, base * 2))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(base * 2, base * 4))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(base * 4, base * 8))
        self.d4 = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(base * 8, base * 16))
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.drop(self.d4(x4))
        return {'skips': [x1, x2, x3, x4], 'bottleneck': x5}


class UNetDecoderV18(nn.Module):
    """Identical to v16 UNetDecoder; optional Dropout2d whose position is set by
    `dropout_place`:
      'uamt' - after the LAST up block, i.e. on the full-resolution features that
               feed the output conv (UA-MT's VNet applies its second Dropout3d to
               block_nine, right before out_conv; the first one sits on the
               bottleneck, see the encoders);
      'v18'  - after the FIRST up block (the S06 placement, E-20; kept so the
               S06 `v18B_*_mcdropout_s*` checkpoints can still be rebuilt)."""

    def __init__(self, base=64, out_channels=2, bilinear=False, dropout_p: float = 0.0,
                 dropout_place: str = 'v18'):
        super().__init__()
        assert dropout_place in ('uamt', 'v18'), dropout_place
        self.u1 = Up(base * 16, base * 8, bilinear)
        self.u2 = Up(base * 8, base * 4, bilinear)
        self.u3 = Up(base * 4, base * 2, bilinear)
        self.u4 = Up(base * 2, base, bilinear)
        self.outc = nn.Conv2d(base, out_channels, 1)
        self.dropout_place = dropout_place
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, enc):
        skips = enc['skips']
        x = self.u1(enc['bottleneck'], skips[3])
        if self.dropout_place == 'v18':
            x = self.drop(x)
        x = self.u2(x, skips[2])
        x = self.u3(x, skips[1])
        x = self.u4(x, skips[0])
        if self.dropout_place == 'uamt':
            x = self.drop(x)
        return self.outc(x), x


# ===========================================================================
# 4. The network
# ===========================================================================

class FourierFuseNetV18(nn.Module):
    """v16 FourierFuseNet with (a) configurable head, (b) optional dropout,
    (c) optional residual encoder. Attribute names match v16 exactly."""

    def __init__(self, in_channels: int, n_classes: int, base_features: int = 64,
                 bilinear: bool = False, num_domains: int = 6,
                 num_protos_per_domain: int = 2, proto_ema_decay: float = 0.99,
                 head: str = 'evidential', evid_activation: str = 'softplus',
                 evid_temperature: float = 1.0, evid_prior_w: float = 1.0,
                 evid_max_evidence: float = 0.0, dropout_p: float = 0.0,
                 encoder_arch: str = 'unet', dropout_place: str = 'v18'):
        super().__init__()
        assert head in ('evidential', 'softmax'), head
        assert encoder_arch in ('unet', 'resunet'), encoder_arch
        assert dropout_place in ('uamt', 'v18'), dropout_place
        self.dropout_place = dropout_place
        self.n_classes = n_classes
        self.K = n_classes + 1
        self.in_channels = in_channels
        self.bf = base_features
        self.head_type = head
        self.dropout_p = float(dropout_p)
        self.encoder_arch = encoder_arch
        bf = base_features
        Enc = UNetEncoderV18 if encoder_arch == 'unet' else ResUNetEncoderV18
        self.encoder = Enc(in_channels, bf, dropout_p=dropout_p)
        self.decoder = UNetDecoderV18(bf, out_channels=self.K, bilinear=bilinear,
                                      dropout_p=dropout_p, dropout_place=dropout_place)
        if head == 'evidential':
            self.evidential = EvidentialHeadV18(self.K, activation=evid_activation,
                                                temperature=evid_temperature,
                                                prior_w=evid_prior_w,
                                                max_evidence=evid_max_evidence)
        else:
            self.evidential = SoftmaxHead(self.K, temperature=evid_temperature)
        self.spectral_reg = SpectralConsistencyRegularizer(perturbation_strength=0.3)
        self.prototype_bank = PrototypeBank(feature_dim=bf * 16, num_domains=num_domains,
                                            num_protos_per_domain=num_protos_per_domain,
                                            ema_decay=proto_ema_decay)

    # --- dropout control (MC-dropout keeps BN in eval, dropout in train) ----
    def set_dropout_active(self, active: bool):
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train(active)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        logits, last_feat = self.decoder(enc)
        ev = self.evidential(logits)
        out = {'logits': logits, 'evidence': ev['evidence'], 'alpha': ev['alpha'],
               'S': ev['S'], 'prob': ev['prob'], 'U_epi': ev['U_epi'],
               'U_ale': ev['U_ale'], 'belief': ev['belief']}
        if return_features:
            out['bottleneck'] = enc['bottleneck']
            out['last_decoder_feat'] = last_feat
            out['encoder_output'] = enc
        return out

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        logits, _ = self.decoder(enc)
        return logits

    def get_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.evidential(self.get_logits(x))['prob']

    def compute_spectral_consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_perturbed = self.spectral_reg.perturb_amplitude(x)
        enc_orig = self.encoder(x)
        enc_pert = self.encoder(x_perturbed)
        return self.spectral_reg.compute_loss(enc_orig['bottleneck'], enc_pert['bottleneck'])


# ===========================================================================
# 5. Losses
# ===========================================================================

def kl_dirichlet_to_prior(alpha: torch.Tensor, prior_w: float = 1.0) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(w,...,w) ), per pixel, [B,1,H,W].
    For w = 1 this is exactly v16's kl_dirichlet_uniform (Sensoy 2018, eq.8).
    General w: KL = lnGamma(S) - sum lnGamma(a_k) - lnGamma(Kw) + K lnGamma(w)
                    + sum (a_k - w)(psi(a_k) - psi(S)).
    """
    alpha = alpha.float()
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    dg = torch.digamma(alpha) - torch.digamma(S)
    term1 = ((alpha - prior_w) * dg).sum(dim=1, keepdim=True)
    w = torch.tensor(float(prior_w), device=alpha.device, dtype=alpha.dtype)
    term2 = (torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
             - torch.lgamma(K * w) + K * torch.lgamma(w))
    return term1 + term2


def evidential_bayes_risk(alpha: torch.Tensor, target_onehot: torch.Tensor,
                          loss_type: str = 'mse') -> torch.Tensor:
    """Per-pixel Bayes risk of Sensoy et al. 2018, one of three forms:
      'mse' : eq.5  sum_k (y_k - p_k)^2 + p_k(1-p_k)/(S+1)     (v16 default)
      'ce'  : eq.4  sum_k y_k (psi(S) - psi(alpha_k))          (DEviS eq.15)
      'log' : eq.3  sum_k y_k (log S - log alpha_k)
    Returns [B,1,H,W]."""
    alpha = alpha.float(); target_onehot = target_onehot.float()
    S = alpha.sum(dim=1, keepdim=True)
    if loss_type == 'mse':
        p = alpha / S.clamp(min=1e-8)
        err = (target_onehot - p).pow(2).sum(dim=1, keepdim=True)
        var = (p * (1 - p)).sum(dim=1, keepdim=True) / (S + 1).clamp(min=1e-8)
        return err + var
    if loss_type == 'ce':
        return (target_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1, keepdim=True)
    if loss_type == 'log':
        return (target_onehot * (torch.log(S) - torch.log(alpha))).sum(dim=1, keepdim=True)
    raise ValueError(loss_type)


def evidential_seg_loss_v18(alpha: torch.Tensor, target_onehot: torch.Tensor,
                            lambda_kl: float, loss_type: str = 'mse',
                            prior_w: float = 1.0,
                            pixel_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bayes risk + lambda_kl * KL(Dir(alpha~) || Dir(w1)) with
    alpha~ = y*w + (1-y)*alpha (correct-class concentration reset to the prior).
    With (mse, w=1, pixel_weight=None) equals v16's evidential_seg_loss."""
    alpha = alpha.float(); target_onehot = target_onehot.float()
    risk = evidential_bayes_risk(alpha, target_onehot, loss_type)
    alpha_tilde = target_onehot * prior_w + (1 - target_onehot) * alpha
    kl = kl_dirichlet_to_prior(alpha_tilde, prior_w)
    per_pix = risk + lambda_kl * kl
    if pixel_weight is None:
        return per_pix.mean()
    return (pixel_weight * per_pix).sum() / (pixel_weight.sum() + 1e-8)


def kl_schedule(iter_num: int, lam_max: float, anneal_iters: int, schedule: str) -> float:
    """'linear' (v16): lam_max * min(1, t/anneal); 'const': lam_max; 'none': 0."""
    if schedule == 'none':
        return 0.0
    if schedule == 'const':
        return float(lam_max)
    return float(lam_max) * min(1.0, iter_num / max(anneal_iters, 1))


def softmax_seg_loss(prob: torch.Tensor, target_onehot: torch.Tensor,
                     pixel_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Conventional CE + soft Dice on softmax probabilities (true FixMatch)."""
    prob = prob.float(); target_onehot = target_onehot.float()
    ce = -(target_onehot * torch.log(prob.clamp(min=1e-8))).sum(dim=1, keepdim=True)
    if pixel_weight is None:
        ce_term = ce.mean()
    else:
        ce_term = (pixel_weight * ce).sum() / (pixel_weight.sum() + 1e-8)
    dice_term = soft_dice_from_alpha(prob, target_onehot, mask=pixel_weight)
    return ce_term + dice_term


# --- DEviS (Zou et al., IEEE TCYB 2025; arXiv:2301.00349) -------------------

def devis_beta_t(iter_num: int, max_iter: int, beta0: float = 0.01) -> float:
    """DEviS eq.20 annealing: beta_t = beta0 * exp(-(ln beta0) * t / T).
    Rises monotonically from beta0 (t=0) to 1 (t=T)."""
    t = min(max(iter_num, 0), max_iter)
    return float(beta0 * math.exp(-math.log(beta0) * t / max(max_iter, 1)))


def devis_cup_loss(belief: torch.Tensor, u: torch.Tensor, pred: torch.Tensor,
                   target: torch.Tensor, beta_t: float,
                   pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """DEviS calibrated uncertainty penalty, eq.20:
       L_CUP = - beta_t     * mean_{correct}   sum_c b_c     log(1 - u)
               - (1-beta_t) * mean_{incorrect} sum_c (1-b_c) log(u)
    belief [B,K,H,W], u [B,1,H,W], pred/target [B,H,W] (hard labels).
    Sums in the paper are converted to means over the respective pixel sets
    so the term is batch-size independent. [pixel_mask: optional valid mask]."""
    eps = 1e-6
    correct = (pred == target).float().unsqueeze(1)
    if pixel_mask is not None:
        correct = correct * pixel_mask
        incorrect = (1 - (pred == target).float().unsqueeze(1)) * pixel_mask
    else:
        incorrect = 1 - correct
    u = u.clamp(eps, 1 - eps)
    ac = -(belief.sum(dim=1, keepdim=True) * torch.log(1 - u))          # A&U penalty
    iu = -((1 - belief).sum(dim=1, keepdim=True) * torch.log(u))        # I&C penalty
    term_c = (correct * ac).sum() / (correct.sum() + 1.0)
    term_i = (incorrect * iu).sum() / (incorrect.sum() + 1.0)
    return beta_t * term_c + (1 - beta_t) * term_i


# --- I-EDL (Deng et al., ICML 2023) used by MEDL eq.7 ----------------------

def dirichlet_log_fisher_det(alpha: torch.Tensor) -> torch.Tensor:
    """log |I(alpha)| for a Dirichlet, per pixel [B,1,H,W]:
       |I| = prod_k psi1(alpha_k) * (1 - psi1(S) * sum_k 1/psi1(alpha_k))."""
    S = alpha.sum(dim=1, keepdim=True)
    tg = torch.polygamma(1, alpha)
    tgS = torch.polygamma(1, S)
    inner = (1 - tgS * (1.0 / tg.clamp(min=1e-12)).sum(dim=1, keepdim=True)).clamp(min=1e-8)
    return torch.log(tg.clamp(min=1e-12)).sum(dim=1, keepdim=True) + torch.log(inner)


def iedl_pixel_loss(alpha: torch.Tensor, target_onehot: torch.Tensor,
                    lambda_fisher: float) -> torch.Tensor:
    """MEDL eq.7 / I-EDL: sum_k [(y_k - a_k/S)^2 + a_k(S-a_k)/(S^2(S+1))] psi1(a_k)
    - lambda * log|I(alpha)|.  Returns per-pixel [B,1,H,W] (unweighted)."""
    alpha = alpha.float(); target_onehot = target_onehot.float()
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S.clamp(min=1e-8)
    tg = torch.polygamma(1, alpha)
    term = ((target_onehot - p).pow(2) + alpha * (S - alpha) / (S.pow(2) * (S + 1)).clamp(min=1e-8)) * tg
    return term.sum(dim=1, keepdim=True) - lambda_fisher * dirichlet_log_fisher_det(alpha)


# ===========================================================================
# 6. MC-dropout (UA-MT, Yu et al. MICCAI 2019) helpers
# ===========================================================================

@torch.no_grad()
def mc_dropout_predict(model: FourierFuseNetV18, x: torch.Tensor, T: int = 8
                       ) -> Dict[str, torch.Tensor]:
    """T stochastic forward passes with dropout ON and BN in eval mode.
    Returns mean prob [B,K,H,W], predictive entropy H [B,1,H,W] (nats),
    mutual information MI = H(mean) - mean(H) [B,1,H,W], and the sample
    std of the max-class prob. The model's mode is restored afterwards."""
    was_training = model.training
    model.eval()
    model.set_dropout_active(True)
    probs = []
    for _ in range(T):
        probs.append(model.forward(x, return_features=False)['prob'].float())
    model.set_dropout_active(False)
    if was_training:
        model.train()
    P = torch.stack(probs, dim=0)                       # [T,B,K,H,W]
    mean = P.mean(dim=0)
    ent_mean = -(mean * torch.log(mean.clamp(min=1e-10))).sum(dim=1, keepdim=True)
    ent_each = -(P * torch.log(P.clamp(min=1e-10))).sum(dim=2, keepdim=True).mean(dim=0)
    return {'prob': mean, 'entropy': ent_mean, 'mi': ent_mean - ent_each,
            'std': P.max(dim=2).values.std(dim=0, unbiased=False).unsqueeze(1)}


def uamt_entropy_threshold(iter_num: int, max_iter: int, K: int) -> float:
    """UA-MT schedule: H_thr = (0.75 + 0.25 * sigmoid_rampup(t, T)) * ln(K)."""
    if max_iter <= 0:
        r = 1.0
    else:
        phase = 1.0 - min(max(iter_num, 0), max_iter) / max_iter
        r = math.exp(-5.0 * phase * phase)
    return (0.75 + 0.25 * r) * math.log(K)


# ===========================================================================
# 7. MEDL (He et al., BIBM 2024 / arXiv:2505.12418) helpers
# ===========================================================================

def caef_fuse(b1: torch.Tensor, u1: torch.Tensor, b2: torch.Tensor, u2: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Class-aware evidential fusion, MEDL eq.3.
    b1,b2: belief masses [B,K,H,W] (sum_k b + u = 1); u1,u2: [B,1,H,W].
      b_n = b1_n b2_n + |C_n|/(|C_n|+|C_K|) (b1_n u2 + b2_n u1),  |C_n|=1, |C_K|=K
      b_K = u1 u2
    then all K+1 masses are normalised to sum to one.
    Returns (fused belief [B,K,H,W], fused uncertainty [B,1,H,W])."""
    K = b1.shape[1]
    coef = 1.0 / (1.0 + K)
    b = b1 * b2 + coef * (b1 * u2 + b2 * u1)
    u = u1 * u2
    Z = b.sum(dim=1, keepdim=True) + u
    return b / Z.clamp(min=1e-8), u / Z.clamp(min=1e-8)


def medl_reliability(b_fused: torch.Tensor, u_fused: torch.Tensor) -> torch.Tensor:
    """MEDL eq.4: R = exp( u * sum_n zeta_n log2 zeta_n ), zeta = fused belief.
    R in (0,1]; R -> 1 when u -> 0 (the entropy term is <= 0)."""
    ent = (b_fused * torch.log2(b_fused.clamp(min=1e-10))).sum(dim=1, keepdim=True)
    return torch.exp(u_fused * ent)


def medl_asymptotic_weight(uncert: torch.Tensor, progress: float, xi: float = 1.0
                           ) -> torch.Tensor:
    """MEDL eq.6 voxel weight omega = Xi * tanh(psi * zeta) + 1, implemented per
    the stated INTENT (certain-to-uncertain curriculum):
      psi  = 1 - 2 h / V with h = ascending rank of uncertainty (most certain
             pixel -> psi = +1),
      zeta = 1 - 2 q / Q (start of training -> +1, end -> -1).
    Early training up-weights certain pixels, late training up-weights
    uncertain ones. [UNVERIFIED sign convention: the paper's text defines h as a
    descending rank, which read literally inverts the stated intent; we follow
    the intent and expose Xi as --medl_xi.]  uncert: [B,1,H,W]; ranks are
    computed per image."""
    B = uncert.shape[0]
    flat = uncert.view(B, -1)
    V = flat.shape[1]
    order = torch.argsort(flat, dim=1)                       # ascending
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, V + 1, device=uncert.device).unsqueeze(0).expand(B, -1))
    psi = 1.0 - 2.0 * ranks.float() / V
    zeta = 1.0 - 2.0 * float(progress)
    return (xi * torch.tanh(psi * zeta) + 1.0).view_as(uncert)


# ===========================================================================
# Self-test (CPU)
# ===========================================================================

def _selftest():
    torch.manual_seed(0)
    from fourierfuse_v16_modules import (FourierFuseNet, evidential_seg_loss,
                                         EvidentialHead)
    # 1. Head equivalence with v16 defaults
    logits = torch.randn(2, 3, 16, 16) * 4
    h16 = EvidentialHead(3)(logits)
    h18 = EvidentialHeadV18(3)(logits)
    for k in ('alpha', 'prob', 'U_epi', 'U_ale'):
        assert torch.allclose(h16[k], h18[k], atol=1e-6), k
    # prior_w generalisation keeps sum b + u = 1
    hw = EvidentialHeadV18(3, prior_w=0.5)(logits)
    assert torch.allclose(hw['belief'].sum(1, keepdim=True) + hw['U_epi'],
                          torch.ones_like(hw['U_epi']), atol=1e-5)
    # cap bounds vacuity from below
    hc = EvidentialHeadV18(3, max_evidence=10.0)(logits * 100)
    assert (hc['U_epi'] >= 3.0 / (3 + 30) - 1e-6).all()
    # 2. Loss equivalence with v16 defaults
    y = labels_to_onehot(torch.randint(0, 3, (2, 16, 16)), K=3)
    l16 = evidential_seg_loss(h16['alpha'], y, lambda_kl=0.02)
    l18 = evidential_seg_loss_v18(h18['alpha'], y, lambda_kl=0.02, loss_type='mse', prior_w=1.0)
    assert torch.allclose(l16, l18, atol=1e-5), (l16, l18)
    for lt in ('ce', 'log'):
        v = evidential_seg_loss_v18(h18['alpha'], y, 0.02, loss_type=lt)
        assert torch.isfinite(v), lt
    # 3. Network: v16 checkpoint loads strictly into the v18 net with defaults
    n16 = FourierFuseNet(1, 1, base_features=8, num_domains=6)
    n18 = FourierFuseNetV18(1, 1, base_features=8, num_domains=6)
    n18.load_state_dict(n16.state_dict(), strict=True)
    x = torch.randn(1, 1, 32, 32)
    n16.eval(); n18.eval()
    o16 = n16(x, return_features=True); o18 = n18(x, return_features=True)
    assert torch.allclose(o16['prob'], o18['prob'], atol=1e-6)
    assert torch.allclose(o16['U_epi'], o18['U_epi'], atol=1e-6)
    # softmax head + resunet + dropout construct and run
    ns = FourierFuseNetV18(1, 1, base_features=8, head='softmax', dropout_p=0.5, encoder_arch='resunet')
    ns.eval(); os_ = ns(x)
    assert torch.allclose(os_['prob'].sum(1), torch.ones(1, 32, 32), atol=1e-5)
    mc = mc_dropout_predict(ns, x, T=4)
    assert mc['entropy'].shape == (1, 1, 32, 32) and (mc['mi'] >= -1e-5).all()
    assert not ns.training  # mode restored
    # S07b (E-20/E-22): both placements are checked AT THE DROPOUT MODULES via forward hooks
    # (position = input shape of the decoder dropout; activity = output mask), not at the
    # network output: on this 8-feature 32x32 test net the 2x2 bottleneck / 4x4 first-up-block
    # dropouts leave `prob` unchanged within allclose (E-22), although the real 384x384
    # checkpoints do show MC variation with the 'v18' placement (S06 smoke harness, mc_std ~1e-3).
    expect_dec_in = {'uamt': (8, 32, 32), 'v18': (64, 4, 4)}   # (C, H, W) entering the decoder dropout
    for place in ('uamt', 'v18'):
        nd = FourierFuseNetV18(1, 1, base_features=8, head='evidential', dropout_p=0.5,
                               dropout_place=place)
        assert nd.decoder.dropout_place == place and isinstance(nd.decoder.drop, nn.Dropout2d)
        assert isinstance(nd.encoder.drop, nn.Dropout2d)
        seen = {}
        def mk(k):
            return lambda mod, inp, out: seen.setdefault(k, []).append((inp[0].detach().clone(), out.detach().clone()))
        hooks = [nd.encoder.drop.register_forward_hook(mk('enc')), nd.decoder.drop.register_forward_hook(mk('dec'))]
        nd.eval(); a = nd(x)['prob']; b = nd(x)['prob']
        assert torch.allclose(a, b), place                                   # eval: deterministic output
        assert tuple(seen['dec'][0][0].shape[1:]) == expect_dec_in[place], (place, seen['dec'][0][0].shape)  # placement
        assert tuple(seen['enc'][0][0].shape[1:]) == (128, 2, 2), seen['enc'][0][0].shape                  # bottleneck
        for k in ('enc', 'dec'):
            (i1, o1), (i2, o2) = seen[k]
            assert torch.equal(i1, o1) and torch.equal(i2, o2), (place, k)   # inactive: identity
        seen.clear()
        nd.set_dropout_active(True)
        outs = [nd(x)['prob'] for _ in range(3)]; c, d = outs[0], outs[1]
        nd.set_dropout_active(False)
        for k in ('enc', 'dec'):
            trip = seen[k]; assert len(trip) == 3, (place, k, len(trip))
            i1 = trip[0][0]; alive = (i1.flatten(2).abs().sum(-1) > 0)[0]     # channels carrying signal
            if alive.sum() >= 2:
                os_ = [o for _, o in trip]
                assert not (torch.equal(os_[0], os_[1]) and torch.equal(os_[1], os_[2])), (place, k)   # stochastic mask
                fr = [(((o.flatten(2).abs().sum(-1) == 0)[0] & alive).float().sum() / alive.float().sum()).item() for o in os_]
                assert any(0.0 < f < 1.0 for f in fr), (place, k, fr)          # some, not all, live channels dropped
            else:                                                             # dead deep path at init: test the module directly
                mod = nd.encoder.drop if k == 'enc' else nd.decoder.drop
                mod.train(True); t = torch.ones(1, i1.shape[1], 4, 4); us = [mod(t) for _ in range(3)]; mod.train(False)
                assert not (torch.equal(us[0], us[1]) and torch.equal(us[1], us[2])), (place, k)
                assert any(0.0 < (u.flatten(2).sum(-1) == 0).float().mean().item() < 1.0 for u in us), (place, k)
        for h in hooks:
            h.remove()
        print(f"[selftest] dropout_place={place}: placement + mask OK; max|dprob| between two MC passes "
              f"on the test net = {(c - d).abs().max().item():.2e} (informative only)")
        assert not nd.training and not nd.decoder.drop.training
    t0 = uamt_entropy_threshold(0, 100, 2)
    assert 0.75 * math.log(2) <= t0 <= 0.76 * math.log(2), t0
    assert abs(uamt_entropy_threshold(100, 100, 2) - math.log(2)) < 1e-6
    # 4. DEviS pieces
    assert abs(devis_beta_t(0, 100) - 0.01) < 1e-9 and abs(devis_beta_t(100, 100) - 1.0) < 1e-9
    b = h18['belief']; u = h18['U_epi']
    pred = h18['prob'].argmax(1); tgt = torch.randint(0, 3, (2, 16, 16))
    cup = devis_cup_loss(b, u, pred, tgt, beta_t=0.3)
    assert torch.isfinite(cup) and cup >= 0
    # 5. I-EDL
    li = iedl_pixel_loss(h18['alpha'], y, lambda_fisher=0.1)
    assert torch.isfinite(li).all()
    # 6. MEDL fusion
    b2 = torch.rand(2, 3, 16, 16); u2 = torch.rand(2, 1, 16, 16)
    Z2 = b2.sum(1, keepdim=True) + u2; b2 /= Z2; u2 /= Z2
    bf, uf = caef_fuse(b, u, b2, u2)
    assert torch.allclose(bf.sum(1, keepdim=True) + uf, torch.ones_like(uf), atol=1e-5)
    R = medl_reliability(bf, uf)
    assert (R > 0).all() and (R <= 1 + 1e-6).all()
    w = medl_asymptotic_weight(uf, progress=0.0, xi=1.0)
    # early: most certain pixel gets the largest weight
    flat_u = uf.view(2, -1); flat_w = w.view(2, -1)
    assert flat_w[0, flat_u[0].argmin()] >= flat_w[0, flat_u[0].argmax()]
    w_late = medl_asymptotic_weight(uf, progress=1.0, xi=1.0)
    flat_wl = w_late.view(2, -1)
    assert flat_wl[0, flat_u[0].argmax()] >= flat_wl[0, flat_u[0].argmin()]
    print('[revision_modules] self-test PASSED')


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
