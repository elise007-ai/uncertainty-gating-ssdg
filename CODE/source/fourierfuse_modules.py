"""
fourierfuse_modules.py — FourierFuse v7: Spectral-Augmented Spatial Network

ARCHITECTURAL INVERSION (v7):
  v1-v6: image → FFT → phase encoder (primary) + spatial branch (auxiliary)
  v7:    image → UNet encoder (primary) + FFT spectral regularization (auxiliary)

Why: The FFT phase-only reconstruction is a GLOBAL transform that destroys
local spatial specificity. v5 and v6 both plateau at 0.64 Dice because the
phase encoder cannot capture fine-grained boundary features. A plain UNet
on the same data achieves 0.70-0.75. The spectral insight (phase=domain-
invariant) is valuable but should be a REGULARIZER, not the backbone.

Components:
  1. UNet encoder-decoder (PRIMARY) — full spatial resolution, no information loss
  2. Spectral Consistency Loss — forces encoder features to be invariant to
     amplitude perturbations (achieves domain invariance by training, not
     by architectural constraint that destroys spatial info)
  3. FDA Spectral Augmentation — swaps amplitude spectra between images during
     training to force domain-invariant feature learning
  4. Frequency-Guided Attention — lightweight FFT analysis tells the decoder
     WHERE domain-specific features are concentrated, guiding robustness
  5. Boundary Refinement Head — Sobel + decoder features for edge sharpening
  6. OT Amplitude Calibrator — style alignment (retained from v3)

Backward compatible: same FourierFuseNet API as v3-v6.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


def safe_log(x, eps=1e-6):
    return torch.log(x.clamp(min=eps))


# ===========================================================================
# UNet building blocks
# ===========================================================================

class DoubleConv(nn.Module):
    def __init__(self, ic, oc, mc=None):
        super().__init__()
        if not mc: mc = oc
        self.net = nn.Sequential(
            nn.Conv2d(ic, mc, 3, padding=1, bias=False), nn.BatchNorm2d(mc), nn.ReLU(True),
            nn.Conv2d(mc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ic, oc))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, ic, oc, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(ic, oc, ic // 2)
        else:
            self.up = nn.ConvTranspose2d(ic, ic // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(ic, oc)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2); dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2, x1], 1))


# ===========================================================================
# 1. PRIMARY BACKBONE: Full UNet Encoder-Decoder
# ===========================================================================

class UNetEncoder(nn.Module):
    """Standard UNet encoder — preserves full spatial resolution at every scale."""
    def __init__(self, in_ch, base=64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*16)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return {'skips': [x1, x2, x3, x4], 'bottleneck': x5}


class UNetDecoder(nn.Module):
    """UNet decoder with optional frequency-guided attention at each scale."""
    def __init__(self, base=64, n_classes=2, bilinear=False, use_freq_attn=True):
        super().__init__()
        f = 2 if bilinear else 1
        self.up1 = Up(base*16, base*8//f, bilinear)
        self.up2 = Up(base*8, base*4//f, bilinear)
        self.up3 = Up(base*4, base*2//f, bilinear)
        self.up4 = Up(base*2, base, bilinear)

        self.use_freq_attn = use_freq_attn
        if use_freq_attn:
            # Frequency-guided attention at each scale (component 4)
            self.fga1 = FrequencyGuidedAttention(base*8//f)
            self.fga2 = FrequencyGuidedAttention(base*4//f)
            self.fga3 = FrequencyGuidedAttention(base*2//f)
            self.fga4 = FrequencyGuidedAttention(base)

        self.outc = nn.Conv2d(base, n_classes + 1, 1)

    def forward(self, enc, freq_maps=None):
        """
        Args:
            enc: dict with 'skips' and 'bottleneck'
            freq_maps: list of [B,1,H,W] frequency domain-sensitivity maps per scale,
                       or None to disable frequency attention
        Returns:
            logits, last_feat
        """
        s = enc['skips']
        x = self.up1(enc['bottleneck'], s[3])
        if self.use_freq_attn and freq_maps is not None:
            x = self.fga1(x, freq_maps[3] if len(freq_maps) > 3 else None)
        x = self.up2(x, s[2])
        if self.use_freq_attn and freq_maps is not None:
            x = self.fga2(x, freq_maps[2] if len(freq_maps) > 2 else None)
        x = self.up3(x, s[1])
        if self.use_freq_attn and freq_maps is not None:
            x = self.fga3(x, freq_maps[1] if len(freq_maps) > 1 else None)
        x = self.up4(x, s[0])
        if self.use_freq_attn and freq_maps is not None:
            x = self.fga4(x, freq_maps[0] if len(freq_maps) > 0 else None)
        return self.outc(x), x


# ===========================================================================
# 2. SPECTRAL CONSISTENCY LOSS
# ===========================================================================

class SpectralConsistencyRegularizer(nn.Module):
    """
    Forces encoder features to be invariant to amplitude perturbations.

    Given an image x, creates an amplitude-perturbed version x' (same phase,
    different amplitude) and penalizes the L2 distance between their encoder
    features. This achieves domain invariance THROUGH TRAINING rather than
    by destroying spatial information in the architecture.

    This is the key theoretical contribution: the Fourier phase-invariance
    property is enforced as a LOSS, not as an architectural constraint.
    """
    def __init__(self, perturbation_strength=0.3):
        super().__init__()
        self.strength = perturbation_strength

    @torch.no_grad()
    def perturb_amplitude(self, x):
        """Create amplitude-perturbed version of input (same phase, random amplitude mix)."""
        x_f = x.float()
        fft = torch.fft.fft2(x_f, norm='ortho')
        phase = torch.angle(fft)
        amp = torch.abs(fft).clamp(min=1e-8)

        # Random amplitude perturbation: mix with shuffled batch member
        B = x.shape[0]
        perm = torch.randperm(B, device=x.device)
        amp_other = torch.abs(torch.fft.fft2(x_f[perm], norm='ortho')).clamp(min=1e-8)

        # Low-frequency-focused mixing (domain shift is mostly low-freq)
        # Create a low-pass mask centered on DC
        H, W = x.shape[-2], x.shape[-1]
        cy, cx = H // 2, W // 2
        radius = int(min(H, W) * 0.15)  # 15% of image size
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                                 torch.arange(W, device=x.device), indexing='ij')
        # In fftshift coordinates
        mask_lf = ((yy - cy)**2 + (xx - cx)**2 <= radius**2).float()
        mask_lf = torch.fft.ifftshift(mask_lf)  # un-shift to match fft layout
        mask_lf = mask_lf[None, None]  # [1,1,H,W]

        # Mix amplitudes: low-freq from other, high-freq from self
        lam = self.strength
        amp_mixed = amp * (1 - lam * mask_lf) + amp_other * (lam * mask_lf)

        # Reconstruct with original phase + mixed amplitude
        x_perturbed = torch.fft.ifft2(
            torch.polar(amp_mixed, phase), norm='ortho'
        ).real.to(x.dtype)

        return x_perturbed

    def compute_loss(self, features_orig, features_perturbed):
        """
        L2 distance between encoder features of original and amplitude-perturbed images.
        Normalized by feature magnitude to be scale-invariant.
        """
        # Compare at bottleneck level (most semantically rich)
        diff = features_orig - features_perturbed
        loss = (diff ** 2).mean() / (features_orig.detach() ** 2).mean().clamp(min=1e-6)
        return loss


# ===========================================================================
# 3. FDA SPECTRAL AUGMENTATION
# ===========================================================================

@torch.no_grad()
def fda_spectral_augment(image, reference, beta=0.05):
    """
    Fourier Domain Adaptation augmentation (Yang & Soatto, CVPR 2020).

    Replaces low-frequency amplitude of `image` with that of `reference`,
    simulating a domain shift while preserving semantic content (phase).

    Args:
        image: [B, C, H, W] — image to augment
        reference: [B, C, H, W] — image whose style to borrow
        beta: float — fraction of low frequencies to swap (0.01-0.1 typical)

    Returns:
        augmented: [B, C, H, W] — image with reference's low-freq amplitude
    """
    img_f = image.float()
    ref_f = reference.float()

    fft_img = torch.fft.fft2(img_f, norm='ortho')
    fft_ref = torch.fft.fft2(ref_f, norm='ortho')

    amp_img = torch.abs(fft_img)
    amp_ref = torch.abs(fft_ref)
    phase_img = torch.angle(fft_img)

    H, W = image.shape[-2], image.shape[-1]
    b_h = int(H * beta)
    b_w = int(W * beta)

    # Create low-frequency mask (centered at DC in fftshift coords, then unshift)
    mask = torch.zeros(H, W, device=image.device)
    mask[:b_h, :b_w] = 1
    mask[:b_h, W-b_w:] = 1
    mask[H-b_h:, :b_w] = 1
    mask[H-b_h:, W-b_w:] = 1
    mask = mask[None, None]

    # Swap low-freq amplitude
    amp_mixed = amp_img * (1 - mask) + amp_ref * mask
    result = torch.fft.ifft2(torch.polar(amp_mixed, phase_img), norm='ortho').real

    return result.to(image.dtype)


# ===========================================================================
# 4. FREQUENCY-GUIDED ATTENTION
# ===========================================================================

class FrequencyGuidedAttention(nn.Module):
    """
    Uses frequency-domain analysis to guide spatial attention.

    Computes a spatial map indicating where high-frequency (boundary) or
    domain-specific (low-frequency amplitude variation) features are
    concentrated, and uses this to modulate decoder features.

    This is lightweight — just a 1x1 conv + sigmoid gate.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feat, freq_map=None):
        """
        Args:
            feat: [B, C, H, W] decoder features
            freq_map: [B, 1, H, W] frequency sensitivity map, or None
        """
        if freq_map is None:
            return feat
        if freq_map.shape[-2:] != feat.shape[-2:]:
            freq_map = F.interpolate(freq_map, feat.shape[-2:], mode='bilinear', align_corners=False)
        gate = self.gate(torch.cat([feat, freq_map], 1))
        return feat * (1 + self.scale * gate)


# ===========================================================================
# 5. BOUNDARY REFINEMENT HEAD
# ===========================================================================

class BoundaryRefinementHead(nn.Module):
    """Edge-sharpening post-processor using Sobel + decoder features."""
    def __init__(self, n_classes, in_channels, dec_ch=64):
        super().__init__()
        self.n_out = n_classes + 1
        self.register_buffer('sx', torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1.]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sy', torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1.]], dtype=torch.float32).view(1,1,3,3))
        self.refine = nn.Sequential(
            nn.Conv2d(self.n_out + 1 + dec_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, self.n_out, 1))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)
        self.blend = nn.Parameter(torch.tensor(-1.0))

    def edge(self, img):
        g = img.mean(1, keepdim=True).float() if img.shape[1] > 1 else img.float()
        gx = F.conv2d(F.pad(g, (1,1,1,1), mode='reflect'), self.sx)
        gy = F.conv2d(F.pad(g, (1,1,1,1), mode='reflect'), self.sy)
        e = (gx**2 + gy**2 + 1e-8).sqrt(); ef = e.flatten(1)
        return ((e - ef.min(1)[0].view(-1,1,1,1)) / (ef.max(1)[0].view(-1,1,1,1) - ef.min(1)[0].view(-1,1,1,1) + 1e-8)).to(img.dtype)

    def forward(self, logits, img, dec_feat):
        e = self.edge(img)
        if dec_feat.shape[-2:] != logits.shape[-2:]:
            dec_feat = F.interpolate(dec_feat, logits.shape[-2:], mode='bilinear', align_corners=False)
        if e.shape[-2:] != logits.shape[-2:]:
            e = F.interpolate(e, logits.shape[-2:], mode='bilinear', align_corners=False)
        return logits + torch.sigmoid(self.blend) * self.refine(
            torch.cat([logits.detach(), e, dec_feat.detach()], 1))


# ===========================================================================
# 6. OT AMPLITUDE CALIBRATOR (unchanged from v6)
# ===========================================================================

class OTAmplitudeCalibrator(nn.Module):
    def __init__(self, afd, num_clusters=6, ema_decay=0.99, mixing_coeff=0.15, max_shift_norm=2.0, **kw):
        super().__init__()
        self.nc = num_clusters; self.ed = ema_decay; self.msn = max_shift_norm; self._init = False
        self.ml = nn.Parameter(torch.tensor(math.log(mixing_coeff / (1 - mixing_coeff + 1e-8))))
        self.register_buffer('c', torch.zeros(num_clusters, afd))
        self.register_buffer('cc', torch.zeros(num_clusters))
        self.register_buffer('b', torch.zeros(afd))

    @torch.no_grad()
    def _upd(self, f):
        f = f.float()
        if not self._init:
            B = f.shape[0]
            K = min(self.nc, B)
            if K < 1:
                return
            # k-means++ init with safe multinomial
            idx = torch.randint(B, (1,))
            sel = [f[idx].squeeze(0)]
            for _ in range(K - 1):
                stacked = torch.stack(sel)
                d = torch.cdist(f, stacked).min(1)[0]  # [B]
                # Guard: if all distances are 0 (identical features), use uniform
                d_sum = d.sum()
                if d_sum < 1e-6 or torch.isnan(d_sum) or torch.isinf(d_sum):
                    prob = torch.ones(B, device=f.device) / B
                else:
                    prob = d / d_sum
                # Guard: clamp negatives, ensure valid distribution
                prob = prob.clamp(min=0.0)
                prob_sum = prob.sum()
                if prob_sum < 1e-8:
                    prob = torch.ones(B, device=f.device) / B
                else:
                    prob = prob / prob_sum
                try:
                    next_idx = torch.multinomial(prob, 1)
                except RuntimeError:
                    # Fallback: random selection
                    next_idx = torch.randint(B, (1,), device=f.device)
                sel.append(f[next_idx].squeeze(0))
            for k in range(K):
                self.c[k] = sel[k]
                self.cc[k] = 1.0
            self.b = torch.stack(sel).mean(0)
            self._init = True
            return
        act = self.cc > 0
        if not act.any():
            return
        d = torch.cdist(f, self.c[act].float())
        ai = torch.where(act)[0]
        assigns = ai[d.argmin(1)]
        for k in range(self.nc):
            m = assigns == k
            if m.any():
                cm = f[m].mean(0)
                self.c[k] = (self.ed * self.c[k].float() + (1 - self.ed) * cm) if self.cc[k] > 0 else cm
                self.cc[k] += m.sum().float()
        a2 = self.cc > 0
        if a2.any():
            self.b = self.c[a2].float().mean(0)

    def calibrate(self, feat_map):
        """Calibrate bottleneck features."""
        B, C, H, W = feat_map.shape
        mix = torch.sigmoid(self.ml).clamp(max=0.3)
        p = feat_map.flatten(2).mean(2)
        self._upd(p.detach())
        if not self._init or (self.cc > 0).sum() < 2:
            return feat_map, {'ot_mix_coeff': mix.item(), 'ot_shift_norm': 0.0, 'ot_active_clusters': (self.cc > 0).sum().item()}
        act = self.cc > 0; d = torch.cdist(p.float(), self.c[act].float()); n = torch.where(act)[0][d.argmin(1)]
        sh = (self.b.unsqueeze(0) - self.c[n]).to(feat_map.dtype)
        sn = sh.float().norm(1, keepdim=True).clamp(min=1e-8)
        sh = (sh.float() * (sn.clamp(max=self.msn) / sn)).to(feat_map.dtype)
        return feat_map + mix * sh.unsqueeze(-1).unsqueeze(-1), {
            'ot_mix_coeff': mix.item(), 'ot_shift_norm': sh.norm(1).mean().item(),
            'ot_active_clusters': act.sum().item()}


# ===========================================================================
# TEMPORAL ENSEMBLE (retained from v6)
# ===========================================================================

class TemporalEnsemble:
    def __init__(self, momentum=0.9, max_cache=4096):
        self.mom = momentum; self.mx = max_cache; self.cache = {}; self.cnt = {}

    @torch.no_grad()
    def update_and_get(self, keys, probs):
        B = probs.shape[0]; result = probs.clone()
        for i in range(B):
            k = keys[i]; p = probs[i]
            if k in self.cache and self.cache[k].shape == p.shape:
                new = self.mom * self.cache[k] + (1-self.mom) * p
                self.cache[k] = new; self.cnt[k] += 1
                c = new / (1 - self.mom ** self.cnt[k])
                c = c.clamp(min=1e-8); result[i] = c / c.sum(0, keepdim=True)
            else:
                self.cache[k] = p.clone(); self.cnt[k] = 1
            if len(self.cache) > self.mx:
                old = min(self.cnt, key=self.cnt.get)
                del self.cache[old]; del self.cnt[old]
        return result


# ===========================================================================
# FREQUENCY MAP COMPUTATION
# ===========================================================================

@torch.no_grad()
def compute_frequency_sensitivity_maps(x, scales=4):
    """
    Compute per-pixel maps indicating where the image has high frequency
    content (boundaries) and where amplitude varies (domain-sensitive regions).

    Returns list of [B, 1, H_s, W_s] maps at progressively lower resolutions.
    """
    x_f = x.float()
    fft = torch.fft.fft2(x_f, norm='ortho')
    amp = torch.abs(fft).clamp(min=1e-8)
    log_amp = torch.log(amp)

    # High-frequency energy map (indicates boundaries)
    H, W = x.shape[-2], x.shape[-1]
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                             torch.arange(W, device=x.device), indexing='ij')
    # Distance from DC (in fft layout, DC is at [0,0])
    dist = torch.sqrt(yy.float()**2 + xx.float()**2)
    dist = torch.minimum(dist, torch.sqrt((H - yy.float())**2 + (W - xx.float())**2))
    dist = torch.minimum(dist, torch.sqrt(yy.float()**2 + (W - xx.float())**2))
    dist = torch.minimum(dist, torch.sqrt((H - yy.float())**2 + xx.float()**2))
    high_freq_weight = (dist / dist.max()).clamp(0, 1)[None, None]

    # High-frequency energy per pixel (inverse FFT of weighted amplitude)
    hf_energy = torch.fft.ifft2(
        torch.polar(amp * high_freq_weight, torch.angle(fft)), norm='ortho'
    ).real.abs()

    # Average across channels → single map
    hf_map = hf_energy.mean(dim=1, keepdim=True)
    # Normalize per sample
    hf_flat = hf_map.flatten(1)
    hf_map = (hf_map - hf_flat.min(1)[0].view(-1,1,1,1)) / \
             (hf_flat.max(1)[0].view(-1,1,1,1) - hf_flat.min(1)[0].view(-1,1,1,1) + 1e-8)

    # Create multi-scale maps
    maps = []
    cur = hf_map.to(x.dtype)
    for _ in range(scales):
        maps.append(cur)
        cur = F.avg_pool2d(cur, 2) if cur.shape[-1] > 4 else cur

    return maps


# ===========================================================================
# FULL FourierFuseNet v7
# ===========================================================================

class FourierFuseNet(nn.Module):
    """
    v7: Spatial-primary architecture with spectral regularization.

    Same external API as v3-v6 for full backward compatibility.
    """
    def __init__(self, in_channels, n_classes, base_features=64,
                 apply_window=True, speckle_suppress=True,
                 score_hidden=64, num_ot_clusters=6,
                 langevin_steps=3, bilinear=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_out = n_classes + 1
        self.in_channels = in_channels
        bf = base_features

        # PRIMARY: Full UNet encoder-decoder
        self.encoder = UNetEncoder(in_channels, bf)
        self.decoder = UNetDecoder(bf, n_classes, bilinear, use_freq_attn=True)

        # Boundary refinement head
        self.boundary_head = BoundaryRefinementHead(n_classes, in_channels, bf)

        # OT calibrator on encoder bottleneck
        self.ot_calibrator = OTAmplitudeCalibrator(bf * 16, num_ot_clusters)

        # Spectral consistency regularizer
        self.spectral_reg = SpectralConsistencyRegularizer(perturbation_strength=0.3)

        # Compatibility stubs
        self.score_denoiser = nn.Identity()
        self.spatial_encoder = nn.Identity()  # spatial IS the primary now

        self._cached_entropy = None

    def forward(self, x, return_features=False):
        # Compute frequency sensitivity maps for attention guidance
        freq_maps = compute_frequency_sensitivity_maps(x, scales=4)

        # PRIMARY: UNet encode
        enc = self.encoder(x)

        # OT calibrate bottleneck
        bn_cal, ot_d = self.ot_calibrator.calibrate(enc['bottleneck'])
        enc_cal = {'skips': enc['skips'], 'bottleneck': bn_cal}

        # Decode with frequency-guided attention
        logits, last_feat = self.decoder(enc_cal, freq_maps)

        # Boundary refinement
        logits = self.boundary_head(logits, x, last_feat)

        # Entropy
        pr = torch.softmax(logits.float(), 1)
        ent = -(pr * torch.log(pr.clamp(min=1e-8))).sum(1, keepdim=True).to(logits.dtype)
        if torch.isfinite(ent).all():
            self._cached_entropy = ent.detach()
        else:
            self._cached_entropy = None

        out = {'logits': logits, 'entropy': ent, 'ot_diagnostics': ot_d}
        if return_features:
            out['phase_bottleneck'] = enc['bottleneck']  # compat naming
            out['amp_bottleneck'] = enc['bottleneck']
            out['last_decoder_feat'] = last_feat
            out['encoder_output'] = enc
        return out

    def get_logits(self, x):
        self._cached_entropy = None
        return self.forward(x)['logits']

    def compute_spectral_consistency_loss(self, x):
        """
        Compute spectral consistency loss for domain invariance.
        Call during training with labeled or unlabeled images.
        """
        x_perturbed = self.spectral_reg.perturb_amplitude(x)
        enc_orig = self.encoder(x)
        enc_pert = self.encoder(x_perturbed)
        return self.spectral_reg.compute_loss(
            enc_orig['bottleneck'], enc_pert['bottleneck'])

    # Backward compat stubs
    def get_score_loss(self, cm, pf):
        return torch.tensor(0.0, device=cm.device)

    @torch.no_grad()
    def refine_pseudo_labels(self, pp, pf):
        return pp
