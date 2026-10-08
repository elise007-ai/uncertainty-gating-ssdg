"""
train_fourierfuse_v18_revision.py
============================================================================
BSPC-D-26-12287 MAJOR-REVISION training script (session S01, 2026-09-25).

A strict SUPERSET of the scripts that produced the submitted numbers:
  * train_fourierfuse_v16.py           (daep / fixmatch / cadg)      [Tables 2-4]
  * train_fourierfuse_v16_adaptive.py  (+ --method adaptive)         [Table 5]
  * train_fourierfuse_v17_consistency.py (+ percentile / consistency) [Table 5]

INVARIANT: with every new flag at its default, `--method fixmatch` and
`--method daep` execute the SAME code path as v16/v17 (evidential softplus head,
MSE+KL(0.02, linear 10k)+Dice supervised loss, evidential-MSE+Dice pseudo-label
loss, FDA beta=0.05 p=0.5, CutMix p=1, spectral consistency 0.5, entropy term
0.1, EMA 0.995, SWAD). The new machinery is only reachable through new flags.

NEW (all default-off / default-v16):
  --split_seed N          labelled-subset seed decoupled from --seed (deep
                          ensembles at a FIXED split; default -1 => use --seed)
  --head softmax          true FixMatch baseline: softmax + CE + Dice, hard gate
  --evid_activation {softplus,relu,exp}   evidence function          (R1 #4,#10)
  --evid_loss {mse,ce,log}                Sensoy Bayes-risk form     (R1 #4,#10)
  --evid_kl_schedule {linear,const,none}  KL annealing               (R1 #6,#10)
  --evid_prior_w W        Dirichlet base-rate weight (alpha = e + W) (R1 #4,#10)
  --evid_temperature T    logit temperature before the activation  (R1 #6,#10)
  --evid_max_evidence C   evidence cap (bounds vacuity from below)  (R1 #10)
  --method mcdropout      UA-MT gate: MC-dropout predictive entropy (R1 #8)
     --mc_dropout_p P --mc_T T
  --method devis          DEviS objective (Bayes-risk CE + KL + Dice(1-beta_t)
                          + CUP) and belief-mass gate                (R1 #5)
  --method medl           MEDL: 2nd heterogeneous evidential net, CAEF fusion,
                          reliability mask, asymptotic weights, I-EDL (R1 #5)
  --ff_fda_beta / --ff_fda_prob (already existed) for the FDA sweep (R1 #9)
  --consistency_ramp_frac F   (S10, D-13 control for E-28) per-iteration sigmoid
                          ramp over the first F*max_iterations iterations, same
                          cap; default -1 = the v16 schedule (150 buckets over
                          the WHOLE run).  --selftest_ramp prints both schedules
                          and exits (no data, no GPU).

Every run writes `run_config.json` (all args + git-free provenance: script
sha256, module sha256, hostname, torch version) next to log.txt so that the
diagnostics harness can rebuild the exact model without guessing.

USAGE (smoke test, ~2 min on one GPU):
    python train_fourierfuse_v18_revision.py --dataset prostate --lb_num 20 \
      --lb_domain 5 --domain_num 6 --max_iterations 20 --num_eval_iter 10 \
      --gpu 0 --seed 1337 --save_name v18_smoke_prostate --overwrite
============================================================================
"""

import argparse
import contextlib
import logging
import math
import os
import random
import shutil
import sys
import time
import socket
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataloaders.dataloader import (FundusSegmentation, ProstateSegmentation,
                                     MNMSSegmentation, BUSISegmentation,
                                     ISICSegmentation)
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps  # note: `util` is NOT imported — we
                                           # use `metrics.dice_coeff_*` only.

from medpy.metric import binary
from scipy.ndimage import zoom

# v16 modules
from fourierfuse_v16_modules import (
    FourierFuseNet,
    fda_spectral_augment,
    evidential_seg_loss,
    soft_dice_from_alpha,
    daep_pseudo_label_loss,
    labels_to_onehot,
)
# v18 revision additions (see revision_modules.py; v16 modules stay untouched)
import hashlib, json, socket
from revision_modules import (
    FourierFuseNetV18,
    evidential_seg_loss_v18, kl_schedule, softmax_seg_loss,
    devis_beta_t, devis_cup_loss, iedl_pixel_loss,
    mc_dropout_predict, uamt_entropy_threshold,
    caef_fuse, medl_reliability, medl_asymptotic_weight,
)

# Option B: CADG controller (label-free per-domain uncertainty recalibration).
from cadg_controller import CADGController
from dataloaders.seed_split import select_labeled_idxs


# ===========================================================================
# Args
# ===========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate',
                    choices=['fundus', 'prostate', 'MNMS', 'BUSI', 'ISIC'])
parser.add_argument("--save_name", type=str, default="")
parser.add_argument("--overwrite", action='store_true')

parser.add_argument("--max_iterations", type=int, default=60000)
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.0005)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--amp', type=int, default=1)
parser.add_argument("--label_bs", type=int, default=4)
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=40)
parser.add_argument('--lb_ratio', type=float, default=0)
parser.add_argument("--ema_decay", type=float, default=0.995)

# Consistency weighting (mirrors v15 knobs exactly)
parser.add_argument("--consistency", type=float, default=1.0)
parser.add_argument("--consistency_rampup", type=float, default=150.0)
parser.add_argument('--consistency_weight_cap', type=float, default=0.5,
                    help="Upper bound on consistency weight. Prevents DAEP "
                         "unsupervised term from dominating when teacher "
                         "pseudo-labels are still unreliable. Default 0.5 "
                         "(matches v15).")
parser.add_argument('--consistency_ramp_frac', type=float, default=-1.0,
                    help="S10 (D-13, E-28 control): if > 0, the consistency weight is the "
                         "per-iteration sigmoid ramp of Laine & Aila over the first "
                         "consistency_ramp_frac * max_iterations iterations, then held at "
                         "min(consistency, consistency_weight_cap). Default -1 keeps the v16 "
                         "schedule: 150 buckets of max_iterations/consistency_rampup iterations "
                         "spanning the whole run (cw 0.04 at 20 %%, cap from 63 %%).")
parser.add_argument('--selftest_ramp', action='store_true',
                    help='Print the v16 schedule and the --consistency_ramp_frac schedule at '
                         'selected iterations and exit 0 (R-35: tests the schedule function itself).')

parser.add_argument('--save_model', action='store_true')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--AdamW', action='store_true')
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--warmup_period', type=int, default=150)

# Architecture
parser.add_argument('--ff_base_features', type=int, default=64)
parser.add_argument('--ff_bilinear', action='store_true')

# Spectral regularization (kept from v10, same defaults)
parser.add_argument('--ff_spec_consistency_weight', type=float, default=0.5)
parser.add_argument('--ff_fda_beta', type=float, default=0.05)
parser.add_argument('--ff_fda_prob', type=float, default=0.5)
parser.add_argument('--ff_no_fda', action='store_true')
parser.add_argument('--ff_no_spec_consistency', action='store_true')

# SWAD (kept from v10, same defaults)
parser.add_argument('--swad_enable', action='store_true', default=True)
parser.add_argument('--swad_disable', dest='swad_enable', action='store_false')
parser.add_argument('--swad_start_iter', type=int, default=10000)
parser.add_argument('--swad_update_every', type=int, default=100)
parser.add_argument('--swad_bn_recalib_batches', type=int, default=20)

# ---------------------------------------------------------------------
# v16 DAEP hyperparameters
# ---------------------------------------------------------------------
parser.add_argument('--daep_alpha', type=float, default=1.0,
                    help='Weight on aleatoric uncertainty in DAEP pixel weight.')
parser.add_argument('--daep_beta',  type=float, default=1.0,
                    help='Weight on epistemic uncertainty in DAEP pixel weight.')
parser.add_argument('--daep_gamma', type=float, default=1.0,
                    help='Weight on domain-shift uncertainty in DAEP pixel weight.')
parser.add_argument('--daep_no_ale', action='store_true',
                    help='Ablation: disable aleatoric term in DAEP weighting.')
parser.add_argument('--daep_no_epi', action='store_true',
                    help='Ablation: disable epistemic term in DAEP weighting.')
parser.add_argument('--daep_no_dom', action='store_true',
                    help='Ablation: disable domain-shift term in DAEP weighting.')
parser.add_argument('--daep_threshold_baseline', action='store_true',
                    help='Ablation: replace DAEP weighting with hard threshold '
                         'on teacher max-prob (>0.8). For comparison.')

# Prototype bank
parser.add_argument('--method', type=str, default='daep',
                    choices=['cadg', 'cadg_global', 'daep', 'fixmatch',
                             'consistency_gate', 'percentile_gate',
                             'adaptive', 'mcdropout', 'devis', 'medl'],
                    help='Option-B method selector. daep == Option A (no '
                         'recalibration); cadg == full Option B; cadg_global == '
                         'single shared calibration map (ablation); fixmatch == '
                         'fixed-threshold baseline.')
parser.add_argument('--cadg_n_bins', type=int, default=20)
parser.add_argument('--cadg_refit_every', type=int, default=200)
parser.add_argument('--cadg_min_count', type=float, default=50000)
parser.add_argument('--cadg_recalibrate_ale', action='store_true',
                    help='also recalibrate U_ale (default: only U_epi).')
parser.add_argument('--cadg_seed_split', action='store_true', default=True,
                    help='use seed-dependent labeled split (mandatory for valid '
                         'multi-seed std; see docs).')
parser.add_argument('--cadg_no_seed_split', dest='cadg_seed_split',
                    action='store_false')

# --- v17: scale-free pseudo-label gates (second-paper sidequest) -------------
# Motivated by the paper-1 finding that the teacher's absolute uncertainty
# (Dirichlet vacuity K/S) range-collapses to ~1e-4 and is unusable by a fixed
# gate. These two gates avoid depending on the absolute uncertainty scale.
parser.add_argument('--cg_n_views', type=int, default=3,
                    help='consistency_gate: number of FDA style views (incl. '
                         'identity) whose teacher predictions must agree.')
parser.add_argument('--cg_agree_frac', type=float, default=1.0,
                    help='consistency_gate: fraction of views that must match '
                         'the base pseudo-label for a pixel to be trusted '
                         '(1.0 = unanimous).')
parser.add_argument('--cg_conf_floor', type=float, default=0.0,
                    help='consistency_gate: optional extra confidence floor on '
                         'the base prediction (0 = pure agreement gate).')
parser.add_argument('--pg_keep_frac', type=float, default=0.80,
                    help='percentile_gate: keep the most-confident this '
                         'fraction of pixels per image (rank-based, scale-free).')
parser.add_argument('--proto_per_domain', type=int, default=2,
                    help='Number of prototypes per labeled source domain.')
parser.add_argument('--proto_ema', type=float, default=0.99,
                    help='EMA decay for prototype bank updates.')

# Evidential loss
parser.add_argument('--evid_lambda_kl_max', type=float, default=0.02,
                    help='Max KL coefficient for evidential loss (annealed).')
parser.add_argument('--evid_anneal_iters', type=int, default=10000,
                    help='KL coefficient ramps up linearly over this many iters.')

# ---------------------------------------------------------------------
# v18 REVISION flags (all defaults reproduce v16/v17 behaviour exactly)
# ---------------------------------------------------------------------
parser.add_argument('--split_seed', type=int, default=-1,
                    help='Seed for the labelled-subset selection ONLY. -1 (default) '
                         'uses --seed (v16 behaviour). Set explicitly to train several '
                         'init seeds on ONE fixed split (deep-ensemble measurement).')
parser.add_argument('--head', type=str, default='evidential',
                    choices=['evidential', 'softmax'],
                    help='softmax = conventional FixMatch (softmax + CE + Dice); only '
                         'valid with --method fixmatch/adaptive/percentile_gate/'
                         'consistency_gate/mcdropout.')
parser.add_argument('--evid_activation', type=str, default='softplus',
                    choices=['softplus', 'relu', 'exp'])
parser.add_argument('--evid_loss', type=str, default='mse',
                    choices=['mse', 'ce', 'log'],
                    help='Sensoy Bayes-risk form for supervised + pseudo-label '
                         'evidential losses (ce = digamma form used by DEviS).')
parser.add_argument('--evid_kl_schedule', type=str, default='linear',
                    choices=['linear', 'const', 'none'])
parser.add_argument('--evid_prior_w', type=float, default=1.0,
                    help='Dirichlet base-rate weight W: alpha = e + W, U_epi = K*W/S.')
parser.add_argument('--evid_temperature', type=float, default=1.0)
parser.add_argument('--evid_max_evidence', type=float, default=0.0,
                    help='Cap on per-class evidence (0 = off).')
parser.add_argument('--mc_dropout_p', type=float, default=0.0,
                    help='Dropout2d prob (UA-MT: 0.5). Must be > 0 for --method mcdropout; '
                         'also used by the D-08 dropout-only control (fixmatch + dropout).')
parser.add_argument('--mc_dropout_place', type=str, default='uamt', choices=['uamt', 'v18'],
                    help="Where the Dropout2d sits: 'uamt' = bottleneck + LAST up block "
                         "(UA-MT VNet: block_five and block_nine); 'v18' = bottleneck + "
                         "FIRST up block (the S06 runs, E-20). Recorded in run_config.")
parser.add_argument('--mc_target', type=str, default='teacher', choices=['teacher', 'mcmean'],
                    help="--method mcdropout only. 'teacher': the pseudo-label is the "
                         "deterministic EMA teacher's prediction exactly as for every other "
                         "gate, and the T MC passes provide ONLY the entropy used by the "
                         "UA-MT mask (gate-only variant, S07 default). 'mcmean': the "
                         "pseudo-label is the argmax of the MC mean (the S06 "
                         "v18B_*_mcdropout_s* runs; quarantined, E-19).")
parser.add_argument('--mc_T', type=int, default=8, help='MC-dropout passes (UA-MT: 8).')
parser.add_argument('--devis_beta0', type=float, default=0.01,
                    help='DEviS CUP annealing start (paper: 0.01).')
parser.add_argument('--devis_cup_weight', type=float, default=1.0,
                    help='Weight on the CUP term (paper eq.21: 1).')
parser.add_argument('--devis_gate', type=str, default='soft',
                    choices=['soft', 'hard', 'none', 'threshold'],
                    help='soft: w = 1-u (belief mass); hard: 1[u < devis_u_thr]; none: no gate; '
                         'threshold: the testbed default gate 1[max prob > 0.80] (isolates the '
                         'DEviS objective from the DEviS gate).')
parser.add_argument('--devis_u_thr', type=float, default=0.2)
parser.add_argument('--medl_arch2', type=str, default='resunet', choices=['resunet', 'unet'])
parser.add_argument('--medl_xi', type=float, default=1.0,
                    help='Xi in MEDL eq.6 [UNVERIFIED: not stated in the paper].')
parser.add_argument('--medl_rel_thr', type=float, default=0.5,
                    help='Reliability-mask threshold on R (MEDL eq.4) [UNVERIFIED].')
parser.add_argument('--medl_lambda_fisher', type=float, default=0.1,
                    help='lambda_2 on log|I(alpha)| in MEDL eq.7 [UNVERIFIED].')
parser.add_argument('--medl_lambda1', type=float, default=0.5,
                    help='lambda_1 in MEDL eq.5 (paper Table VII best: 0.5).')
parser.add_argument('--medl_report', type=str, default='net1',
                    choices=['net1', 'net2', 'ensemble'],
                    help='Which prediction the test() loop scores as the headline.')

parser.add_argument('--resume', type=str, default='auto', choices=['auto', 'off'],
                    help='auto: if <snapshot>/resume_state.pth exists and the run is not complete, '
                         'continue from it instead of deleting the directory (--overwrite is then '
                         'ignored). off: v16 behaviour (--overwrite deletes and restarts).')
parser.add_argument('--resume_every_epochs', type=int, default=2,
                    help='write resume_state.pth every N eval-epochs (N x num_eval_iter iterations).')
parser.add_argument('--die_after_epoch', type=int, default=-1,
                    help='TEST ONLY: exit(3) right after the resume state of this (0-based) epoch is '
                         'written, to exercise the resume path on the server.')

# Numerical safety
parser.add_argument('--grad_clip', type=float, default=5.0)

args = parser.parse_args()

# True iff the evidential LOSS configuration is the v16 one; then the original
# v16 loss function is called so the default path stays byte-identical.
_EVID_DEFAULT_CFG = (args.evid_loss == 'mse' and args.evid_prior_w == 1.0
                     and args.evid_kl_schedule == 'linear')


# ===========================================================================
# Utilities (mirrored from v15 so behavior is identical where it matters)
# ===========================================================================

def get_current_consistency_weight(epoch):
    """
    Matches v15's formula. `epoch` is an integer step counter produced by the
    caller; we apply a sigmoid rampup and cap at consistency_weight_cap.
    """
    raw = args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    return min(raw, args.consistency_weight_cap)


def compute_consistency_weight(iter_num, max_iterations):
    """
    The consistency weight cw at iteration `iter_num` (S10, factored out of the
    training loop so it can be tested on its own; R-35).

    Default (consistency_ramp_frac <= 0): the v15/v16 schedule, byte-identical
    to the previous inline code (R-27):
        epoch_proxy = iter_num // max(1, int(max_iterations / consistency_rampup))
        cw = min(consistency * sigmoid_rampup(epoch_proxy, consistency_rampup), cap)
    i.e. with rampup=150 the ramp spans the WHOLE run (E-28).

    consistency_ramp_frac = F > 0 (D-13 control): the conventional short ramp,
        L  = max(1, int(F * max_iterations))
        cw = min(consistency * sigmoid_rampup(iter_num, L), cap)
    evaluated per iteration; at the default cap 0.5 the cap is reached at
    0.628 * L (exp(-5 (1 - t/L)^2) = 0.5).
    """
    if args.consistency_ramp_frac > 0:
        ramp_len = max(1, int(args.consistency_ramp_frac * max_iterations))
        raw = args.consistency * ramps.sigmoid_rampup(iter_num, ramp_len)
        return min(raw, args.consistency_weight_cap)
    epoch_proxy = iter_num // max(1, int(max_iterations / args.consistency_rampup))
    return get_current_consistency_weight(epoch_proxy)


def _selftest_ramp():
    """--selftest_ramp: print both schedules at 0/5/10/20/50/63/100 % of a
    60 000-iteration run and check the documented values (E-28, D-13)."""
    T = 60000
    pts = [0, 3000, 6000, 12000, 30000, 38000, 60000]
    saved = args.consistency_ramp_frac
    args.consistency_ramp_frac = -1.0
    v16 = [compute_consistency_weight(t, T) for t in pts]
    args.consistency_ramp_frac = 0.10
    short = [compute_consistency_weight(t, T) for t in pts]
    args.consistency_ramp_frac = saved
    print('[selftest_ramp] iter      :', ' '.join(f'{t:>7d}' for t in pts))
    print('[selftest_ramp] v16 (E-28):', ' '.join(f'{v:7.4f}' for v in v16))
    print('[selftest_ramp] frac=0.10 :', ' '.join(f'{v:7.4f}' for v in short))
    ok = (abs(v16[3] - 0.0408) < 0.002 and abs(v16[4] - 0.2865) < 0.002
          and v16[5] == 0.5 and v16[6] == 0.5
          and short[0] < 0.01 and abs(short[1] - 0.2865) < 0.002 and short[2] == 0.5 and short[6] == 0.5)
    print('[selftest_ramp] PASSED' if ok else '[selftest_ramp] FAILED')
    return 0 if ok else 5


if args.selftest_ramp:
    sys.exit(_selftest_ramp())


def cycle(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


class statistics:
    def __init__(self):
        self.record = []
        self.num = 0
        self.avg = 0.0
        self.val = 0.0

    def update(self, val):
        self.val = val
        self.record.append(val)
        self.num += 1
        self.avg = sum(self.record) / self.num


def to_2d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    temp_prob = input_tensor == torch.ones_like(input_tensor)
    tensor_list.append(temp_prob)
    temp_prob2 = input_tensor > torch.zeros_like(input_tensor)
    tensor_list.append(temp_prob2)
    return torch.cat(tensor_list, dim=1).float()


def to_3d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(1, 4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    return torch.cat(tensor_list, dim=1).float()


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4,
                      ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break
    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    return mask


def update_ema(model, ema_model, decay, step):
    alpha = min(1 - 1 / (step + 1), decay)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.data.copy_(buf.data)


def compute_dsdv(domain_dices: list) -> float:
    """Domain-Stratified Dice Variance — paper metric. Same as v15."""
    if len(domain_dices) < 2:
        return 0.0
    arr = np.array(domain_dices)
    return float(np.var(arr))


class EnsembleProb(nn.Module):
    """MEDL reports the mean of the two sub-networks' outputs at test time.
    test() only calls .get_prob() / .eval() / .train(), so this thin wrapper
    is enough."""
    def __init__(self, m1: nn.Module, m2: nn.Module):
        super().__init__()
        self.m1, self.m2 = m1, m2

    def get_prob(self, x):
        return 0.5 * (self.m1.get_prob(x) + self.m2.get_prob(x))


# ===========================================================================
# SWAD (dense weight averaging) — mirrored from v15
# ===========================================================================

class SWADAverager:
    def __init__(self):
        self.state = None
        self.count = 0
        self.first_iter = -1
        self.last_iter = -1

    def update(self, model: nn.Module, iter_num: int):
        with torch.no_grad():
            cur = {k: v.detach().clone().float()
                   for k, v in model.state_dict().items()}
            if self.state is None:
                self.state = cur
                self.count = 1
                self.first_iter = iter_num
            else:
                c = self.count + 1
                for k in self.state:
                    if cur[k].is_floating_point():
                        self.state[k].mul_(self.count / c).add_(cur[k], alpha=1.0 / c)
                    else:
                        self.state[k] = cur[k]
                self.count = c
            self.last_iter = iter_num

    def load_into(self, target_model: nn.Module):
        if self.state is None:
            return False
        target_sd = target_model.state_dict()
        new_sd = {}
        for k, v in self.state.items():
            if k in target_sd:
                new_sd[k] = v.to(target_sd[k].dtype).to(target_sd[k].device)
            else:
                new_sd[k] = v
        target_model.load_state_dict(new_sd, strict=True)
        return True


@torch.no_grad()
def recalibrate_bn(model: nn.Module, dataloader_cycle, num_batches: int,
                   data_key: str = 'unet_size_img'):
    """Re-estimate BN running statistics for averaged model. Same as v15."""
    bn_modules = [m for m in model.modules()
                  if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_modules:
        return

    saved_momentum = []
    for m in bn_modules:
        saved_momentum.append(m.momentum)
        m.reset_running_stats()
        m.momentum = None  # use cumulative moving average

    model.train()
    for _ in range(num_batches):
        sample = next(dataloader_cycle)
        x = sample[data_key].cuda()
        try:
            _ = model.get_logits(x)
        except Exception as e:
            logging.warning(f'BN recalib forward failed: {e}')
            break
    model.eval()

    for m, mom in zip(bn_modules, saved_momentum):
        m.momentum = mom


# ===========================================================================
# Evaluation (mirrored from v15; uses model.get_prob for Dirichlet mean)
# ===========================================================================

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, model_name,
         patch_size, n_part, part, dice_calcu_fn):
    model.eval()
    val_dice = [0.0] * n_part
    val_dc, val_jc, val_hd, val_asd = ([0.0]*n_part, [0.0]*n_part,
                                        [0.0]*n_part, [0.0]*n_part)
    domain_num = len(test_dataloader)
    domain_avg_dices = []
    per_domain_metrics = {}

    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_dice = [0.0] * n_part
        domain_val_dc = [0.0] * n_part
        domain_val_jc = [0.0] * n_part
        domain_val_hd = [0.0] * n_part
        domain_val_asd = [0.0] * n_part
        domain_code = i + 1

        for batch_num, sample in enumerate(cur_dataloader):
            mask = sample['label']
            if args.dataset == 'fundus':
                lb_mask = (mask <= 128) * 2
                lb_mask[mask == 0] = 1
                mask = lb_mask
            elif args.dataset == 'prostate':
                mask = mask.eq(0).long()
            elif args.dataset == 'MNMS':
                mask = mask.long()
            elif args.dataset in ('BUSI', 'ISIC'):
                mask = mask.eq(255).long()

            data = sample['unet_size_img'].cuda()
            # v16: use Dirichlet predictive mean. Equivalent to softmax of logits
            # in v15 only in the sense that both are the argmax-ready probability.
            prob = model.get_prob(data)
            pred_label = torch.argmax(prob, dim=1)
            pred_label = torch.from_numpy(
                zoom(pred_label.cpu(),
                     (1, patch_size / data.shape[-2],
                      patch_size / data.shape[-1]), order=0)
            )

            if args.dataset == 'fundus':
                pred_label = to_2d(pred_label); mask = to_2d(mask)
                pred_onehot = pred_label.clone(); mask_onehot = mask.clone()
            elif args.dataset in ('prostate', 'BUSI', 'ISIC'):
                pred_onehot = pred_label.clone().unsqueeze(1)
                mask_onehot = mask.clone().unsqueeze(1)
            elif args.dataset == 'MNMS':
                pred_onehot = to_3d(pred_label); mask_onehot = to_3d(mask)

            dice = dice_calcu_fn(np.asarray(pred_label.cpu()), mask.cpu())
            dc = [0.0]*n_part; jc = [0.0]*n_part
            hd = [0.0]*n_part; asd = [0.0]*n_part
            for j in range(len(data)):
                for ii, p in enumerate(part):
                    dc[ii] += binary.dc(np.asarray(pred_onehot[j, ii], dtype=bool),
                                         np.asarray(mask_onehot[j, ii], dtype=bool))
                    jc[ii] += binary.jc(np.asarray(pred_onehot[j, ii], dtype=bool),
                                         np.asarray(mask_onehot[j, ii], dtype=bool))
                    if pred_onehot[j, ii].float().sum() < 1e-4:
                        hd[ii] += 100; asd[ii] += 100
                    else:
                        hd[ii] += binary.hd95(np.asarray(pred_onehot[j, ii], dtype=bool),
                                               np.asarray(mask_onehot[j, ii], dtype=bool))
                        asd[ii] += binary.asd(np.asarray(pred_onehot[j, ii], dtype=bool),
                                               np.asarray(mask_onehot[j, ii], dtype=bool))
            for ii in range(n_part):
                dc[ii] /= len(data); jc[ii] /= len(data)
                hd[ii] /= len(data); asd[ii] /= len(data)
            for ii in range(n_part):
                domain_val_dice[ii] += dice[ii]
                domain_val_dc[ii] += dc[ii]
                domain_val_jc[ii] += jc[ii]
                domain_val_hd[ii] += hd[ii]
                domain_val_asd[ii] += asd[ii]

        for ii in range(n_part):
            domain_val_dice[ii] /= len(cur_dataloader)
            domain_val_dc[ii] /= len(cur_dataloader)
            domain_val_jc[ii] /= len(cur_dataloader)
            domain_val_hd[ii] /= len(cur_dataloader)
            domain_val_asd[ii] /= len(cur_dataloader)
            val_dice[ii] += domain_val_dice[ii]
            val_dc[ii] += domain_val_dc[ii]
            val_jc[ii] += domain_val_jc[ii]
            val_hd[ii] += domain_val_hd[ii]
            val_asd[ii] += domain_val_asd[ii]

        domain_avg_dices.append(np.mean(domain_val_dice))
        per_domain_metrics[domain_code] = {
            'dice': list(domain_val_dice), 'dc': list(domain_val_dc),
            'hd95': list(domain_val_hd), 'asd': list(domain_val_asd),
        }
        for n, p in enumerate(part):
            writer.add_scalar(f'{model_name}_val/domain{domain_code}/val_{p}_dice',
                              domain_val_dice[n], epoch)

        text = f'domain{domain_code} epoch {epoch}:'
        for n, p in enumerate(part):
            text += f' val_{p}_dice={domain_val_dice[n]:.4f}'
        for n, p in enumerate(part):
            text += f' hd={domain_val_hd[n]:.2f}'
        logging.info(text)

    model.train()

    for ii in range(n_part):
        val_dice[ii] /= domain_num
        val_dc[ii] /= domain_num
        val_jc[ii] /= domain_num
        val_hd[ii] /= domain_num
        val_asd[ii] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar(f'{model_name}_val/val_{p}_dice', val_dice[n], epoch)

    dsdv = compute_dsdv(domain_avg_dices)
    writer.add_scalar(f'{model_name}_val/DSDV', dsdv, epoch)

    text = f'epoch {epoch}: avg'
    for n, p in enumerate(part):
        text += f' val_{p}_dice={val_dice[n]:.4f}'
    text += f' DSDV={dsdv:.4f}'
    logging.info(text)

    return val_dice, dsdv, per_domain_metrics


# ===========================================================================
# Main training function
# ===========================================================================

def normalise_method_args(args):
    """v10 (E-38, R-44): every method-implied override of a CLI flag happens HERE, before
    run_config.json is written and before training starts, so the recorded provenance equals
    the consumed configuration. S07-S10 DEviS runs recorded evid_loss=mse / evid_kl_schedule=linear
    while training used ce / const (the forcing lived inside train(); the log.txt warnings prove it)."""
    if args.method == 'devis' and args.evid_loss != 'ce':
        logging.warning("[v18] DEviS uses the digamma (ce) Bayes risk; you passed "
                        f"--evid_loss {args.evid_loss}. Forcing 'ce' to stay faithful.")
        args.evid_loss = 'ce'
    if args.method == 'devis' and args.evid_kl_schedule == 'linear':
        logging.warning("[v18] DEviS keeps lambda_KL fixed at 0.02 (no annealing); "
                        "forcing --evid_kl_schedule const.")
        args.evid_kl_schedule = 'const'
    return args


def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr

    # -----------------------------------------------------------------------
    # Dataset setup — MIRRORS v15 EXACTLY. All paths, num_channels, patch
    # sizes, augmentation ranges, and fillcolors are the v15 values that
    # produced working runs. Do not change without measuring.
    # -----------------------------------------------------------------------
    if args.dataset == 'fundus':
        dataset = FundusSegmentation
        train_data_path = '../../data/Fundus'
        part = ['cup', 'disc']
        num_channels = 3
        patch_size = 256
        num_classes = 2
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        args.max_iterations = args.max_iterations if args.max_iterations else 30000
        if args.domain_num >= 4: args.domain_num = 4
        dice_calcu_fn = metrics.dice_coeff_2label
    elif args.dataset == 'prostate':
        dataset = ProstateSegmentation
        train_data_path = '../../data/ProstateSlice'
        part = ['base']
        num_channels = 1
        patch_size = 384
        num_classes = 1
        min_v, max_v = 0.1, 2
        fillcolor = 255
        args.max_iterations = args.max_iterations if args.max_iterations else 60000
        if args.domain_num >= 6: args.domain_num = 6
        dice_calcu_fn = metrics.dice_coeff
    elif args.dataset == 'MNMS':
        dataset = MNMSSegmentation
        train_data_path = '../../data/mnms'
        part = ['lv', 'myo', 'rv']
        num_channels = 1
        patch_size = 288
        num_classes = 3
        min_v, max_v = 0.1, 2
        fillcolor = 0
        args.max_iterations = args.max_iterations if args.max_iterations else 60000
        if args.domain_num >= 4: args.domain_num = 4
        dice_calcu_fn = metrics.dice_coeff_3label
    elif args.dataset == 'BUSI':
        dataset = BUSISegmentation
        train_data_path = '../../data/Dataset_BUSI_with_GT'
        part = ['base']
        num_channels = 1
        patch_size = 256
        num_classes = 1
        min_v, max_v = 0.1, 2
        fillcolor = 0
        args.max_iterations = args.max_iterations if args.max_iterations else 30000
        if args.domain_num >= 2: args.domain_num = 2
        dice_calcu_fn = metrics.dice_coeff
    elif args.dataset == 'ISIC':
        dataset = ISICSegmentation
        train_data_path = '../../data/ISIC2017'
        part = ['base']
        num_channels = 3
        patch_size = 256
        num_classes = 1
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        args.max_iterations = args.max_iterations if args.max_iterations else 30000
        if args.domain_num >= 3: args.domain_num = 3
        dice_calcu_fn = metrics.dice_coeff

    n_part = len(part)
    K = num_classes + 1  # model output channel count (bg + fg)

    # v15 batch-size logic
    if args.lb_num < 8:
        args.label_bs, args.unlabel_bs = 2, 2
    else:
        args.label_bs, args.unlabel_bs = 4, 4

    logging.info(
        f"[v16 DATASET] {args.dataset}: num_classes={num_classes}, K={K}, "
        f"num_channels={num_channels}, patch_size={patch_size}, "
        f"min_v/max_v={min_v}/{max_v}, fillcolor={fillcolor}, "
        f"lb_bs={args.label_bs}, ulb_bs={args.unlabel_bs}"
    )

    # -----------------------------------------------------------------------
    # Model creation (v18: FourierFuseNetV18 == v16 FourierFuseNet at defaults)
    # -----------------------------------------------------------------------
    # Method/flag validation — fail fast, before any data is loaded.
    if args.method in ('daep', 'devis', 'medl') and args.head != 'evidential':
        raise SystemExit(f"--method {args.method} requires --head evidential")
    if args.method == 'mcdropout' and args.mc_dropout_p <= 0:
        raise SystemExit("--method mcdropout requires --mc_dropout_p > 0 (UA-MT used 0.5)")
    normalise_method_args(args)   # v10 (E-38): same normalisation main() applied before run_config.json

    def build_net(arch: str = 'unet'):
        return FourierFuseNetV18(
            in_channels=num_channels,
            n_classes=num_classes,
            base_features=args.ff_base_features,
            bilinear=args.ff_bilinear,
            num_domains=args.domain_num,
            num_protos_per_domain=args.proto_per_domain,
            proto_ema_decay=args.proto_ema,
            head=args.head,
            evid_activation=args.evid_activation,
            evid_temperature=args.evid_temperature,
            evid_prior_w=args.evid_prior_w,
            evid_max_evidence=args.evid_max_evidence,
            dropout_p=args.mc_dropout_p,
            encoder_arch=arch,
            dropout_place=args.mc_dropout_place,
        ).cuda()

    model = build_net('unet')
    ema_model = build_net('unet')
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    # MEDL: second, architecturally different evidential network (+ its EMA
    # teacher). Pseudo-labels for BOTH students come from CAEF-fusing the two
    # EMA teachers' opinions (MEDL eq.3). Only net1 is SWAD-averaged/reported by
    # default (--medl_report) so its capacity matches the single-UNet rows.
    model2, ema_model2 = None, None
    if args.method == 'medl':
        model2 = build_net(args.medl_arch2)
        ema_model2 = build_net(args.medl_arch2)
        ema_model2.load_state_dict(model2.state_dict())
        for p in ema_model2.parameters():
            p.requires_grad = False
        ema_model2.eval()
        n2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
        logging.info(f"[v18 MEDL] net2 arch={args.medl_arch2} params={n2:,}")

    # SWAD eval-only model
    swad_averager = SWADAverager()
    swad_model = None
    if args.swad_enable:
        swad_model = build_net('unet')
        for p in swad_model.parameters():
            p.requires_grad = False
        swad_model.eval()

    logging.info(f"[v18 CFG] head={args.head} act={args.evid_activation} "
                 f"loss={args.evid_loss} kl={args.evid_kl_schedule}/{args.evid_lambda_kl_max}"
                 f"/{args.evid_anneal_iters} W={args.evid_prior_w} T={args.evid_temperature} "
                 f"cap={args.evid_max_evidence} dropout={args.mc_dropout_p} place={args.mc_dropout_place} mc_target={args.mc_target} "
                 f"method={args.method} split_seed={args.split_seed}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"[v16 MODEL] trainable params = {n_params:,} "
                 f"(SynFoC foundation-backed ~ 6x this)")
    logging.info(
        f"[v16 DAEP] alpha={args.daep_alpha}, beta={args.daep_beta}, "
        f"gamma={args.daep_gamma} | ablations: no_ale={args.daep_no_ale}, "
        f"no_epi={args.daep_no_epi}, no_dom={args.daep_no_dom}, "
        f"threshold_baseline={args.daep_threshold_baseline}"
    )
    logging.info(
        f"[v16 OPT] base_lr={base_lr}, AdamW={args.AdamW}, warmup={args.warmup}, "
        f"warmup_period={args.warmup_period}, amp={args.amp}, "
        f"grad_clip={args.grad_clip}"
    )
    logging.info(
        f"[v16 CW] consistency={args.consistency}, "
        f"rampup={args.consistency_rampup}, cap={args.consistency_weight_cap}, "
        f"ramp_frac={args.consistency_ramp_frac} "
        f"({'short per-iteration ramp (D-13)' if args.consistency_ramp_frac > 0 else 'v16 whole-run schedule (E-28)'})"
    )

    # -----------------------------------------------------------------------
    # Option B: CADG controller (label-free per-domain recalibration).
    # Active only for method in {cadg, cadg_global}. For daep/fixmatch it stays
    # None and the loop behaves exactly as Option A / the threshold baseline.
    # -----------------------------------------------------------------------
    cadg = None
    if args.method in ('cadg', 'cadg_global'):
        bottleneck_dim = args.ff_base_features * 16   # base*16 (1024 @ base=64)
        cadg = CADGController(
            feat_dim=bottleneck_dim,
            n_domain_clusters=max(args.domain_num, 2),
            n_bins=args.cadg_n_bins,
            refit_every=args.cadg_refit_every,
            min_count=args.cadg_min_count,
            recalibrate_ale=args.cadg_recalibrate_ale,
            per_domain=(args.method == 'cadg'),
            device='cuda',
        )
        logging.info(
            f"[CADG] method={args.method} per_domain={args.method=='cadg'} "
            f"feat_dim={bottleneck_dim} n_bins={args.cadg_n_bins} "
            f"refit_every={args.cadg_refit_every} min_count={args.cadg_min_count} "
            f"recalibrate_ale={args.cadg_recalibrate_ale}"
        )
    else:
        logging.info(f"[CADG] disabled (method={args.method}).")

    # -----------------------------------------------------------------------
    # Optimizer (single optimizer — no score net in v16)
    # -----------------------------------------------------------------------
    seg_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    if model2 is not None:
        seg_params += list(model2.encoder.parameters()) + list(model2.decoder.parameters())
    # EvidentialHead has no learnable parameters; PrototypeBank has only buffers.

    if args.AdamW:
        seg_optimizer = optim.AdamW(seg_params, lr=base_lr, weight_decay=0.01)
    else:
        seg_optimizer = optim.SGD(seg_params, lr=base_lr, momentum=0.9,
                                   weight_decay=0.0001, nesterov=True)

    if args.warmup:
        def lr_lambda(step):
            if step < args.warmup_period:
                return step / args.warmup_period
            return 0.5 * (1 + math.cos(math.pi * step / args.max_iterations))
        seg_scheduler = LambdaLR(seg_optimizer, lr_lambda)
    else:
        seg_scheduler = None

    # -----------------------------------------------------------------------
    # Data pipeline (v15's exact transform stack)
    # -----------------------------------------------------------------------
    max_iterations = args.max_iterations
    low_res = patch_size

    weak = transforms.Compose([
        tr.RandomScaleCrop(args.img_size), tr.RandomScaleRotate(fillcolor=fillcolor),
        tr.RandomHorizontalFlip(), tr.elastic_transform()
    ])
    strong = transforms.Compose([
        tr.Brightness(min_v, max_v), tr.Contrast(min_v, max_v),
        tr.GaussianBlur(kernel_size=int(0.1 * args.img_size),
                         num_channels=num_channels)
    ])
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0, 1]),
        tr.ToTensor(low_res=low_res, unet_size=patch_size)
    ])

    domain_num = args.domain_num
    domain_list = list(range(1, domain_num + 1))
    if args.dataset == 'fundus':       domain_len = [50, 99, 320, 320]
    elif args.dataset == 'prostate':   domain_len = [225, 305, 136, 373, 338, 133]
    elif args.dataset == 'MNMS':       domain_len = [1030, 1342, 525, 550]
    elif args.dataset == 'BUSI':       domain_len = [350, 168]
    elif args.dataset == 'ISIC':       domain_len = [300, 204, 1098]

    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain - 1]
    lb_num = int(sum(domain_len) * args.lb_ratio) if args.lb_ratio > 0 else args.lb_num
    split_seed = args.seed if args.split_seed < 0 else args.split_seed
    if getattr(args, 'cadg_seed_split', True):
        lb_idxs, unlabeled_idxs = select_labeled_idxs(
            data_num=data_num, lb_num=lb_num, seed=split_seed, log=logging)
    else:
        lb_idxs = list(range(lb_num))
        unlabeled_idxs = list(range(lb_num, data_num))

    with open(os.path.join(snapshot_path, 'split.json'), 'w') as f:
        json.dump({'split_seed': split_seed, 'lb_domain': lb_domain, 'lb_num': lb_num,
                   'data_num': data_num, 'lb_idxs': list(map(int, lb_idxs))}, f, indent=1)

    test_dataset, test_dataloader = [], []
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain,
                          domain=[lb_domain], selected_idxs=lb_idxs,
                          weak_transform=weak, normal_toTensor=normal_toTensor,
                          img_size=args.img_size)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain,
                           domain=domain_list, selected_idxs=unlabeled_idxs,
                           weak_transform=weak, strong_tranform=strong,
                           normal_toTensor=normal_toTensor, img_size=args.img_size)
    for i in range(1, domain_num + 1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1,
                               domain=[i], normal_toTensor=normal_toTensor,
                               img_size=args.img_size)
        test_dataset.append(cur_dataset)

    lb_dataloader = cycle(DataLoader(lb_dataset, batch_size=args.label_bs,
                                      shuffle=True, num_workers=2,
                                      pin_memory=True, drop_last=True))
    ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size=args.unlabel_bs,
                                       shuffle=True, num_workers=2,
                                       pin_memory=True, drop_last=True))
    for i in range(domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size=args.test_bs,
                                     shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    # -----------------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------------
    iter_num = 0
    max_epoch = max_iterations // args.num_eval_iter
    scaler = GradScaler('cuda')
    amp_cm = lambda: autocast('cuda') if args.amp else contextlib.nullcontext()

    best_dice = [0.0] * n_part
    best_avg_dice, best_ema_avg_dice, best_avg_dice_iter = 0.0, 0.0, -1
    dice_of_best_avg = [0.0] * n_part
    best_dsdv = float('inf')
    best_per_domain_metrics = {}

    best_swad_dice = 0.0
    best_swad_iter = -1

    logging.info(f"{args.num_eval_iter} iterations per epoch, "
                 f"{max_epoch} epochs total, max_iterations={max_iterations}")

    # -----------------------------------------------------------------------
    # v18 resume support (E-14). State is written at eval-epoch boundaries, so a
    # resumed run continues from the last saved epoch (<= resume_every_epochs x
    # num_eval_iter iterations are lost). Data-loader order after a resume is a
    # fresh draw; RNG states are restored so augmentations stay random.
    # -----------------------------------------------------------------------
    rs_path = os.path.join(snapshot_path, 'resume_state.pth')

    def _cpu(sd):
        return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

    def save_resume_state(epoch_num_done):
        st = {'iter_num': iter_num, 'epoch_num': epoch_num_done,
              'model': model.state_dict(), 'ema_model': ema_model.state_dict(),
              'optimizer': seg_optimizer.state_dict(), 'scaler': scaler.state_dict(),
              'swad': {'state': _cpu(swad_averager.state) if swad_averager.state is not None else None,
                       'count': swad_averager.count, 'first_iter': swad_averager.first_iter,
                       'last_iter': swad_averager.last_iter},
              'trackers': {'best_dice': list(best_dice), 'best_avg_dice': best_avg_dice,
                           'best_ema_avg_dice': best_ema_avg_dice, 'best_avg_dice_iter': best_avg_dice_iter,
                           'dice_of_best_avg': list(dice_of_best_avg), 'best_dsdv': best_dsdv,
                           'best_per_domain_metrics': best_per_domain_metrics,
                           'best_swad_dice': best_swad_dice, 'best_swad_iter': best_swad_iter},
              'freematch_state': (_cpu(train._freematch_state)
                                  if getattr(train, '_freematch_state', None) else None),
              'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                      'numpy': np.random.get_state(), 'python': random.getstate()},
              'v18_config': vars(args)}
        if model2 is not None:
            st['model2'] = model2.state_dict(); st['ema_model2'] = ema_model2.state_dict()
        tmp = rs_path + '.tmp'
        torch.save(st, tmp)
        os.replace(tmp, rs_path)                      # atomic: never a half-written state
        logging.info(f'[resume] state saved at iter {iter_num} (epoch {epoch_num_done + 1})')

    start_epoch = 0
    if getattr(args, '_resume_from', None):
        st = torch.load(args._resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model']); ema_model.load_state_dict(st['ema_model'])
        if model2 is not None and 'model2' in st:
            model2.load_state_dict(st['model2']); ema_model2.load_state_dict(st['ema_model2'])
        seg_optimizer.load_state_dict(st['optimizer']); scaler.load_state_dict(st['scaler'])
        sw = st['swad']
        swad_averager.state = ({k: v.cuda() if torch.is_tensor(v) else v for k, v in sw['state'].items()}
                               if sw['state'] is not None else None)
        swad_averager.count, swad_averager.first_iter, swad_averager.last_iter = sw['count'], sw['first_iter'], sw['last_iter']
        if swad_model is not None and swad_averager.state is not None:
            swad_averager.load_into(swad_model)
        t = st['trackers']
        best_dice = list(t['best_dice']); best_avg_dice = t['best_avg_dice']
        best_ema_avg_dice = t['best_ema_avg_dice']; best_avg_dice_iter = t['best_avg_dice_iter']
        dice_of_best_avg = list(t['dice_of_best_avg']); best_dsdv = t['best_dsdv']
        best_per_domain_metrics = t['best_per_domain_metrics']
        best_swad_dice = t['best_swad_dice']; best_swad_iter = t['best_swad_iter']
        if st.get('freematch_state'):
            train._freematch_state = {k: v.cuda() for k, v in st['freematch_state'].items()}
        torch.set_rng_state(st['rng']['torch']); torch.cuda.set_rng_state_all(st['rng']['cuda'])
        np.random.set_state(st['rng']['numpy']); random.setstate(st['rng']['python'])
        iter_num = int(st['iter_num']); start_epoch = int(st['epoch_num']) + 1
        logging.info(f"[resume] RESUMED from iter {iter_num} (starting epoch {start_epoch + 1}/{max_epoch}) "
                     f"best_avg so far {best_avg_dice:.4f} @ {best_avg_dice_iter}")
        try:                                           # provenance: every resume is recorded
            rc_path = os.path.join(snapshot_path, 'run_config.json')
            rc = json.load(open(rc_path)) if os.path.exists(rc_path) else {}
            rc.setdefault('resume_events', []).append({'iter': iter_num, 'epoch': start_epoch,
                                                       'time': time.strftime('%Y-%m-%d %H:%M:%S')})
            json.dump(rc, open(rc_path, 'w'), indent=1, default=str)
        except Exception as e:
            logging.warning(f'[resume] could not record resume event: {e}')

    # =======================================================================
    # Training loop
    # =======================================================================
    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'Epoch {epoch_num+1}/{max_epoch} [v16]')

        stats = {
            'loss_total': statistics(),
            'loss_sup': statistics(),
            'loss_unsup': statistics(),
            'loss_spec': statistics(),
            'loss_cons': statistics(),
            'cw': statistics(),
            # DAEP diagnostics
            'daep_w_mean': statistics(),
            'daep_w_min':  statistics(),
            'daep_u_ale_mean': statistics(),
            'daep_u_epi_mean': statistics(),
            'daep_u_dom_mean': statistics(),
            # Evidential diagnostics
            'alpha_S_mean': statistics(),
        }

        for i_batch in range(1, args.num_eval_iter + 1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)

            lb_x_w = lb_sample['unet_size_img']
            ulb_x_w = ulb_sample['unet_size_img']
            ulb_x_s = ulb_sample['unet_size_strong_aug']
            lb_A_y = lb_sample['unet_size_label']

            # Sanity: CutMix requires lb/ulb batch sizes to match (elementwise mix).
            assert lb_x_w.shape[0] == ulb_x_s.shape[0], (
                f"label_bs ({lb_x_w.shape[0]}) must equal unlabel_bs "
                f"({ulb_x_s.shape[0]}) for CutMix copy-paste."
            )

            # Label processing (identical to v15)
            if args.dataset == 'fundus':
                lb_A_mask = (lb_A_y <= 128) * 2
                lb_A_mask[lb_A_y == 0] = 1
            elif args.dataset == 'prostate':
                lb_A_mask = lb_A_y.eq(0).long()
            elif args.dataset == 'MNMS':
                lb_A_mask = lb_A_y.long()
            elif args.dataset in ('BUSI', 'ISIC'):
                lb_A_mask = lb_A_y.eq(255).long()

            lb_x_w = lb_x_w.cuda(); lb_A_mask = lb_A_mask.cuda()
            ulb_x_w = ulb_x_w.cuda(); ulb_x_s = ulb_x_s.cuda()

            # Domain IDs for labeled batch. In SSDG-style setup all labeled
            # samples come from one source domain (lb_domain), 1-indexed in
            # args; the prototype bank uses 0-indexed domain IDs.
            lb_domain_ids = torch.full(
                (lb_x_w.shape[0],),
                fill_value=lb_domain - 1,
                device=lb_x_w.device,
                dtype=torch.long,
            )

            with amp_cm():
                # ===========================================================
                # Step 1: Teacher forward on WEAK unlabeled -> pseudo-labels
                # ===========================================================
                with torch.no_grad():
                    teacher_out = ema_model.forward(ulb_x_w, return_features=True)
                    teacher_prob = teacher_out['prob']
                    teacher_U_ale = teacher_out['U_ale'].detach()
                    teacher_U_epi = teacher_out['U_epi'].detach()
                    teacher_bottleneck = teacher_out['bottleneck'].detach()
                    teacher_pseudo_label = torch.argmax(teacher_prob, dim=1)
                    teacher_max_prob = teacher_prob.max(dim=1).values

                    # ---------------------------------------------------------
                    # Option B (CADG): recalibrate teacher uncertainty per
                    # inferred domain, using a LABEL-FREE stability proxy
                    # (teacher argmax agreement under an FDA style swap). This
                    # rescales U_epi so the downstream DAEP weight operates at a
                    # per-domain-correct threshold. No labels used. For
                    # daep/fixmatch, cadg is None and this is skipped.
                    # ---------------------------------------------------------
                    if cadg is not None:
                        # Build a content-preserving style view of the weak
                        # unlabeled batch using the project's own FDA op, with
                        # the labeled batch as the style reference.
                        styled_ulb_x_w = fda_spectral_augment(
                            ulb_x_w, lb_x_w, beta=args.ff_fda_beta)
                        styled_out = ema_model.forward(styled_ulb_x_w,
                                                       return_features=False)
                        teacher_prob_styled = styled_out['prob']
                        recal = cadg.recalibrate(
                            teacher_prob=teacher_prob,
                            teacher_prob_styled=teacher_prob_styled,
                            teacher_U_epi=teacher_U_epi,
                            teacher_U_ale=teacher_U_ale,
                            teacher_bottleneck=teacher_bottleneck,
                            iter_num=iter_num,
                        )
                        teacher_U_epi = recal['U_epi']
                        teacher_U_ale = recal['U_ale']

                    # ---- v18: alternative teachers / uncertainty signals ----
                    mc_entropy = None
                    medl_R, medl_u = None, None
                    if args.method == 'mcdropout':
                        # UA-MT gate: T stochastic passes give the predictive entropy that
                        # masks the unsupervised loss (H_thr ramp, see below).
                        mc = mc_dropout_predict(ema_model, ulb_x_w, T=args.mc_T)
                        mc_entropy = mc['entropy']                       # [B,1,H,W]
                        if args.mc_target == 'mcmean':
                            # S06 behaviour (E-19): the TARGET was also replaced by the
                            # MC mean, so the run changed model + target + gate at once.
                            teacher_prob = mc['prob']
                            teacher_pseudo_label = torch.argmax(teacher_prob, dim=1)
                            teacher_max_prob = teacher_prob.max(dim=1).values
                        # 'teacher' (default): keep the deterministic EMA-teacher
                        # pseudo-label computed above, exactly as for every other gate.
                    elif args.method == 'medl':
                        # MEDL eq.3: class-aware fusion of the two teachers' opinions.
                        t2 = ema_model2.forward(ulb_x_w, return_features=False)
                        b_f, u_f = caef_fuse(teacher_out['belief'].float(), teacher_U_epi.float(),
                                             t2['belief'].float(), t2['U_epi'].float())
                        teacher_prob = b_f + u_f / float(K)               # projected probability
                        teacher_pseudo_label = torch.argmax(b_f, dim=1)
                        teacher_max_prob = teacher_prob.max(dim=1).values
                        medl_R = medl_reliability(b_f, u_f)              # eq.4, [B,1,H,W]
                        medl_u = u_f

                # Per-image domain-shift uncertainty from PrototypeBank.
                # Note: buffers are shared between student and ema via
                # update_ema's buffer copy; we call from the student.
                if model.prototype_bank.is_ready():
                    U_dom_img = model.prototype_bank.compute_u_dom(teacher_bottleneck)
                else:
                    U_dom_img = torch.zeros(teacher_bottleneck.shape[0],
                                            device=teacher_bottleneck.device)

                # ===========================================================
                # Step 2: Supervised forward on labeled
                # ===========================================================
                lb_out = model.forward(lb_x_w, return_features=True)
                lb_alpha = lb_out['alpha']

                # Update prototype bank with LABELED bottleneck (no-grad inside)
                model.prototype_bank.update(
                    lb_out['bottleneck'].detach(),
                    lb_domain_ids,
                )

                # Target one-hot
                lb_onehot = labels_to_onehot(lb_A_mask, K=K)

                # KL annealing — prevents over-pulling evidence toward uniform early.
                # v18: schedule selectable; 'linear' == v16 formula exactly.
                lambda_kl = kl_schedule(iter_num, args.evid_lambda_kl_max,
                                        args.evid_anneal_iters, args.evid_kl_schedule)
                extra_diag = {}
                if args.head == 'softmax':
                    # True FixMatch baseline: CE + soft Dice on softmax probs.
                    sup_mse_kl = softmax_seg_loss(lb_out['prob'], lb_onehot)
                    sup_dice = torch.zeros((), device=lb_x_w.device)
                    sup_loss = sup_mse_kl
                elif args.method == 'devis':
                    # DEviS (Zou et al. 2025) eq.21: L_ice + lambda*KL + (1-beta_t)*Dice + CUP
                    beta_t = devis_beta_t(iter_num, max_iterations, args.devis_beta0)
                    sup_mse_kl = evidential_seg_loss_v18(lb_alpha, lb_onehot, lambda_kl=lambda_kl,
                                                         loss_type='ce', prior_w=args.evid_prior_w)
                    sup_dice = (1.0 - beta_t) * soft_dice_from_alpha(lb_alpha, lb_onehot)
                    with autocast('cuda', enabled=False):
                        cup = devis_cup_loss(lb_out['belief'].float(), lb_out['U_epi'].float(),
                                             lb_out['prob'].argmax(dim=1), lb_A_mask, beta_t)
                    sup_loss = sup_mse_kl + sup_dice + args.devis_cup_weight * cup
                    extra_diag['devis_beta_t'] = beta_t
                    extra_diag['devis_cup'] = float(cup.item())
                elif _EVID_DEFAULT_CFG:
                    # v16/v17 path, byte-identical (drives the original loss).
                    sup_mse_kl = evidential_seg_loss(lb_alpha, lb_onehot,
                                                      lambda_kl=lambda_kl)
                    sup_dice = soft_dice_from_alpha(lb_alpha, lb_onehot)
                    sup_loss = sup_mse_kl + sup_dice
                else:
                    sup_mse_kl = evidential_seg_loss_v18(lb_alpha, lb_onehot, lambda_kl=lambda_kl,
                                                         loss_type=args.evid_loss,
                                                         prior_w=args.evid_prior_w)
                    sup_dice = soft_dice_from_alpha(lb_alpha, lb_onehot)
                    sup_loss = sup_mse_kl + sup_dice

                # MEDL: net2 supervised loss + uncertainty-based asymptotic weighting on
                # labelled pixels (MEDL eq.5/8, u_final = l1*u_own + l2*u_fused).
                sup_loss2 = torch.zeros((), device=lb_x_w.device)
                if args.method == 'medl':
                    lb_out2 = model2.forward(lb_x_w, return_features=False)
                    with torch.no_grad():
                        bf_l, uf_l = caef_fuse(lb_out['belief'].float(), lb_out['U_epi'].float(),
                                               lb_out2['belief'].float(), lb_out2['U_epi'].float())
                        progress = iter_num / max(max_iterations, 1)
                        u1_final = args.medl_lambda1 * lb_out['U_epi'].float() + (1 - args.medl_lambda1) * uf_l
                        u2_final = args.medl_lambda1 * lb_out2['U_epi'].float() + (1 - args.medl_lambda1) * uf_l
                        om1 = medl_asymptotic_weight(u1_final, progress, xi=args.medl_xi)
                        om2 = medl_asymptotic_weight(u2_final, progress, xi=args.medl_xi)
                    # L_l (unweighted risk+KL+dice) + L_w,l (asymptotically weighted risk)
                    sup_loss = (sup_loss
                                + evidential_seg_loss_v18(lb_alpha, lb_onehot, lambda_kl=0.0,
                                                          loss_type=args.evid_loss,
                                                          prior_w=args.evid_prior_w, pixel_weight=om1))
                    sup_loss2 = (evidential_seg_loss_v18(lb_out2['alpha'], lb_onehot, lambda_kl=lambda_kl,
                                                         loss_type=args.evid_loss, prior_w=args.evid_prior_w)
                                 + soft_dice_from_alpha(lb_out2['alpha'], lb_onehot)
                                 + evidential_seg_loss_v18(lb_out2['alpha'], lb_onehot, lambda_kl=0.0,
                                                           loss_type=args.evid_loss,
                                                           prior_w=args.evid_prior_w, pixel_weight=om2))

                # ===========================================================
                # Step 3: Spectral consistency (fp32; v15's fix carried over)
                # ===========================================================
                spec_loss = torch.tensor(0.0, device=lb_x_w.device)
                if not args.ff_no_spec_consistency:
                    # Use the new AMP API to avoid deprecation warnings.
                    with autocast('cuda', enabled=False):
                        spec_loss = model.compute_spectral_consistency_loss(
                            ulb_x_w.float()
                        ).float()

                # ===========================================================
                # Step 4: FDA augmentation on strong unlabeled
                # ===========================================================
                if not args.ff_no_fda:
                    with torch.no_grad():
                        fda_ulb = fda_spectral_augment(ulb_x_s, lb_x_w,
                                                        beta=args.ff_fda_beta)
                        fda_mask = (torch.rand(ulb_x_s.shape[0], 1, 1, 1,
                                                device=ulb_x_s.device)
                                    > (1 - args.ff_fda_prob)).float()
                        ulb_x_s = ulb_x_s * (1 - fda_mask) + fda_ulb * fda_mask

                # ===========================================================
                # Step 5: CutMix copy-paste (labeled into unlabeled)
                # ===========================================================
                label_box = torch.stack(
                    [obtain_cutmix_box(img_size=patch_size, p=1.0)
                     for _ in range(ulb_x_s.shape[0])], dim=0
                )
                img_box = label_box.unsqueeze(1)
                ulb_x_s_cp = ulb_x_s * (1 - img_box) + lb_x_w * img_box

                # Copy-paste the pseudo-labels and the uncertainties.
                # Where we pasted labeled pixels, pseudo-label is the true
                # label and uncertainties get driven down to zero.
                pseudo_label_cp = (teacher_pseudo_label * (1 - label_box.long())
                                   + lb_A_mask * label_box.long()).long()
                U_ale_cp = teacher_U_ale * (1 - img_box)   # zero inside pasted box
                U_epi_cp = teacher_U_epi * (1 - img_box)

                pseudo_onehot_cp = labels_to_onehot(pseudo_label_cp, K=K)

                # ===========================================================
                # Step 6: Student forward on strong+cutmix
                # ===========================================================
                student_out = model.forward(ulb_x_s_cp, return_features=True)
                student_alpha = student_out['alpha']

                # ===========================================================
                # Step 7: DAEP unsupervised loss (or threshold baseline)
                # ===========================================================
                unsup_loss2 = torch.zeros((), device=lb_x_w.device)
                if args.method in ('adaptive', 'mcdropout', 'devis', 'medl') or args.head == 'softmax':
                    # =========================================================
                    # v18 gates. Each produces a per-pixel weight w in [0,1]
                    # ([B,1,H,W]); the pasted labelled box is always trusted.
                    # =========================================================
                    with torch.no_grad():
                        if args.method == 'adaptive':
                            # FreeMatch-style self-adaptive threshold (ported verbatim from
                            # train_fourierfuse_v16_adaptive.py; no labels used).
                            FM_M = 0.999
                            batch_tau = teacher_max_prob.mean()
                            batch_p = teacher_prob.mean(dim=(0, 2, 3))   # [K]
                            st = getattr(train, '_freematch_state', None)
                            if st is None:
                                st = {'tau': batch_tau.clone(), 'p': batch_p.clone()}
                            else:
                                st['tau'] = FM_M * st['tau'] + (1 - FM_M) * batch_tau
                                st['p'] = FM_M * st['p'] + (1 - FM_M) * batch_p
                            train._freematch_state = st
                            p_norm = st['p'] / st['p'].max().clamp(min=1e-8)
                            tau_map = st['tau'] * p_norm[teacher_pseudo_label]
                            w_gate = (teacher_max_prob > tau_map).float().unsqueeze(1)
                        elif args.method == 'mcdropout':
                            # UA-MT: keep pixels whose MC predictive entropy is below the
                            # ramped threshold H_thr(t) = (0.75 + 0.25 r(t)) ln K.
                            h_thr = uamt_entropy_threshold(iter_num, max_iterations, K)
                            w_gate = (mc_entropy < h_thr).float()
                            extra_diag['mc_h_thr'] = h_thr
                            extra_diag['mc_entropy_mean'] = float(mc_entropy.mean().item())
                        elif args.method == 'devis':
                            u_t = teacher_U_epi.float()
                            if args.devis_gate == 'soft':
                                w_gate = (1.0 - u_t)                    # belief mass sum
                            elif args.devis_gate == 'hard':
                                w_gate = (u_t < args.devis_u_thr).float()
                            elif args.devis_gate == 'threshold':
                                w_gate = (teacher_max_prob > 0.80).float().unsqueeze(1)
                            else:
                                w_gate = torch.ones_like(u_t)
                        elif args.method == 'medl':
                            # reliability mask (eq.4) x asymptotic weight (eq.6)
                            progress = iter_num / max(max_iterations, 1)
                            om_u = medl_asymptotic_weight(medl_u, progress, xi=args.medl_xi)
                            w_gate = (medl_R > args.medl_rel_thr).float()
                            extra_diag['medl_R_mean'] = float(medl_R.mean().item())
                            extra_diag['medl_u_mean'] = float(medl_u.mean().item())
                        else:  # softmax head + fixed threshold (true FixMatch)
                            w_gate = (teacher_max_prob > 0.80).float().unsqueeze(1)
                        w_gate = torch.maximum(w_gate, img_box)
                        w_gate = w_gate * (1 - img_box) + img_box   # pasted box weight = 1

                    if args.head == 'softmax':
                        unsup_loss = softmax_seg_loss(student_out['prob'], pseudo_onehot_cp,
                                                      pixel_weight=w_gate)
                    else:
                        lt = 'ce' if args.method == 'devis' else args.evid_loss
                        unsup_loss = (evidential_seg_loss_v18(student_alpha, pseudo_onehot_cp,
                                                              lambda_kl=0.0, loss_type=lt,
                                                              prior_w=args.evid_prior_w,
                                                              pixel_weight=w_gate)
                                      + soft_dice_from_alpha(student_alpha, pseudo_onehot_cp,
                                                             mask=w_gate))
                    if args.method == 'medl':
                        # + lambda_GWU * omega-weighted I-EDL term (eq.7) for net1 ...
                        with autocast('cuda', enabled=False):
                            ie1 = iedl_pixel_loss(student_alpha.float(), pseudo_onehot_cp,
                                                  args.medl_lambda_fisher)
                            unsup_loss = unsup_loss + ((w_gate * om_u * ie1).sum()
                                                       / ((w_gate * om_u).sum() + 1e-8))
                        # ... and the full unsupervised objective for net2 on the same
                        # strong+cutmix batch (eq.9).
                        student_out2 = model2.forward(ulb_x_s_cp, return_features=False)
                        a2 = student_out2['alpha']
                        unsup_loss2 = (evidential_seg_loss_v18(a2, pseudo_onehot_cp, lambda_kl=0.0,
                                                               loss_type=args.evid_loss,
                                                               prior_w=args.evid_prior_w,
                                                               pixel_weight=w_gate)
                                       + soft_dice_from_alpha(a2, pseudo_onehot_cp, mask=w_gate))
                        with autocast('cuda', enabled=False):
                            ie2 = iedl_pixel_loss(a2.float(), pseudo_onehot_cp, args.medl_lambda_fisher)
                            unsup_loss2 = unsup_loss2 + ((w_gate * om_u * ie2).sum()
                                                         / ((w_gate * om_u).sum() + 1e-8))
                    daep_diag = {
                        'daep_w_mean': float(w_gate.mean().item()),
                        'daep_w_min':  float(w_gate.min().item()),
                        'daep_u_ale_mean': float(teacher_U_ale.mean().item()),
                        'daep_u_epi_mean': float(teacher_U_epi.mean().item()),
                        'daep_u_dom_mean': float(U_dom_img.mean().item()),
                    }
                elif (args.daep_threshold_baseline or args.method == 'fixmatch'
                        or args.method in ('consistency_gate', 'percentile_gate')):
                    if args.method == 'consistency_gate':
                        # -----------------------------------------------------
                        # SCALE-FREE GATE (v17, second-paper idea).
                        # Trust a pseudo-label pixel only where the teacher's
                        # hard prediction is STABLE across FDA style views of
                        # the same image. Agreement is an intersection of label
                        # maps -> independent of the absolute uncertainty scale,
                        # so it is immune to the K/S range-collapse documented in
                        # paper 1. The style views simulate the domain shift, so
                        # the gate is domain-aware by construction. No labels are
                        # used to build the mask.
                        # -----------------------------------------------------
                        with torch.no_grad():
                            agree_count = torch.zeros_like(teacher_max_prob)
                            n_extra = max(1, args.cg_n_views - 1)
                            B = ulb_x_w.shape[0]
                            for v in range(n_extra):
                                # Each view borrows a DIFFERENT style reference so
                                # the views actually differ: roll the labeled
                                # batch by (v+1) positions. fda_spectral_augment is
                                # deterministic given (src, tgt), so varying tgt is
                                # what produces distinct style views here.
                                roll = (v + 1) % B if B > 1 else 0
                                style_ref = torch.roll(lb_x_w, shifts=roll, dims=0)
                                styled = fda_spectral_augment(
                                    ulb_x_w, style_ref, beta=args.ff_fda_beta)
                                styled_pred = torch.argmax(
                                    ema_model.forward(
                                        styled, return_features=False)['prob'],
                                    dim=1)
                                agree_count += (styled_pred ==
                                                teacher_pseudo_label).float()
                            agree_frac = agree_count / float(n_extra)
                            cg_mask = (agree_frac >= args.cg_agree_frac).float()
                            if args.cg_conf_floor > 0.0:
                                cg_mask = cg_mask * (
                                    teacher_max_prob > args.cg_conf_floor).float()
                            pl_mask = cg_mask.unsqueeze(1)
                    elif args.method == 'percentile_gate':
                        # -----------------------------------------------------
                        # RANK-BASED GATE (v17 baseline). Keep the most-confident
                        # cg fraction of pixels PER IMAGE. Uses only the ranking
                        # of confidence (scale-free), directly probing whether the
                        # paper-1 'good ranking, bad range' signal becomes useful
                        # once the broken absolute scale is removed.
                        # -----------------------------------------------------
                        with torch.no_grad():
                            B = teacher_max_prob.shape[0]
                            flat = teacher_max_prob.view(B, -1)
                            k = max(1, int((1.0 - args.pg_keep_frac)
                                           * flat.shape[1]))
                            # threshold = k-th smallest conf per image; keep above
                            thresh = flat.kthvalue(k, dim=1).values.view(B, 1, 1)
                            pl_mask = (teacher_max_prob >=
                                       thresh).float().unsqueeze(1)
                    else:
                        # Hard fixed threshold (>0.8) — strong baseline
                        pl_mask = (teacher_max_prob > 0.80).float().unsqueeze(1)
                    # Pasted labeled region always trusted
                    pl_mask = torch.maximum(pl_mask, img_box)
                    # Evidential MSE + dice on confident pixels
                    S_s = student_alpha.sum(dim=1, keepdim=True)
                    p_s = student_alpha / S_s.clamp(min=1e-8)
                    err = (pseudo_onehot_cp - p_s).pow(2).sum(dim=1, keepdim=True)
                    var = ((p_s * (1 - p_s)).sum(dim=1, keepdim=True)
                           / (S_s + 1).clamp(min=1e-8))
                    mse_term = (pl_mask * (err + var)).sum() / (pl_mask.sum() + 1e-8)
                    dice_term = soft_dice_from_alpha(student_alpha, pseudo_onehot_cp,
                                                      mask=pl_mask)
                    unsup_loss = mse_term + dice_term
                    daep_diag = {
                        'daep_w_mean': float(pl_mask.mean().item()),
                        'daep_w_min':  0.0,
                        'daep_u_ale_mean': float(teacher_U_ale.mean().item()),
                        'daep_u_epi_mean': float(teacher_U_epi.mean().item()),
                        'daep_u_dom_mean': float(U_dom_img.mean().item()),
                    }
                else:
                    # Apply ablation flags by zeroing specific weights
                    aw = 0.0 if args.daep_no_ale else args.daep_alpha
                    bw = 0.0 if args.daep_no_epi else args.daep_beta
                    gw = 0.0 if args.daep_no_dom else args.daep_gamma
                    unsup_loss, daep_diag = daep_pseudo_label_loss(
                        alpha_student=student_alpha,
                        teacher_pseudo_onehot=pseudo_onehot_cp,
                        U_ale_teacher=U_ale_cp,
                        U_epi_teacher=U_epi_cp,
                        U_dom_img=U_dom_img,
                        alpha_w=aw, beta_w=bw, gamma_w=gw,
                    )

                # ===========================================================
                # Step 8: Entropy minimization (mild consistency push)
                # ===========================================================
                student_prob = student_out['prob']
                consistency_loss = -(student_prob *
                                     torch.log(student_prob.clamp(min=1e-10))
                                    ).sum(dim=1).mean()

                # ===========================================================
                # Step 9: Total — v15-style consistency-weight schedule
                # ===========================================================
                # v15 used:
                #   epoch_proxy = iter_num // (max_iterations / consistency_rampup)
                #   cw = min(consistency * sigmoid_rampup(epoch_proxy, rampup),
                #            consistency_weight_cap)
                cw = compute_consistency_weight(iter_num, max_iterations)

                total_loss = (
                    sup_loss
                    + cw * (unsup_loss + 0.1 * consistency_loss)
                    + args.ff_spec_consistency_weight * spec_loss
                )
                if args.method == 'medl':
                    total_loss = total_loss + sup_loss2 + cw * unsup_loss2

            # NaN/Inf guard (v15 style)
            if not torch.isfinite(total_loss):
                logging.warning(
                    f'[iter {iter_num}] non-finite total loss, skipping. '
                    f'sup={sup_loss.item():.3f} unsup={unsup_loss.item():.3f} '
                    f'spec={spec_loss.item():.3f}'
                )
                seg_optimizer.zero_grad()
                iter_num += 1
                if p_bar is not None:
                    p_bar.update()
                continue

            seg_optimizer.zero_grad()
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(seg_optimizer)
                torch.nn.utils.clip_grad_norm_(seg_params, max_norm=args.grad_clip)
                scaler.step(seg_optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(seg_params, max_norm=args.grad_clip)
                seg_optimizer.step()

            if seg_scheduler is not None:
                seg_scheduler.step()

            # EMA update (also copies buffers — shares prototype bank state)
            update_ema(model, ema_model, args.ema_decay, iter_num)
            if model2 is not None:
                update_ema(model2, ema_model2, args.ema_decay, iter_num)
            iter_num += 1

            # SWAD update
            if (args.swad_enable
                    and iter_num >= args.swad_start_iter
                    and iter_num % args.swad_update_every == 0):
                swad_averager.update(model, iter_num)

            # Accumulate stats
            stats['loss_total'].update(total_loss.item())
            stats['loss_sup'].update(sup_loss.item())
            stats['loss_unsup'].update(unsup_loss.item())
            stats['loss_spec'].update(spec_loss.item())
            stats['loss_cons'].update(consistency_loss.item())
            stats['cw'].update(float(cw))
            for k, v in daep_diag.items():
                if k in stats:
                    stats[k].update(v)
            for k, v in extra_diag.items():
                if k not in stats:
                    stats[k] = statistics()
                stats[k].update(float(v))
            with torch.no_grad():
                stats['alpha_S_mean'].update(
                    float(student_alpha.sum(dim=1).mean().item())
                )

            # Tensorboard
            writer.add_scalar('train/loss_total', total_loss.item(), iter_num)
            writer.add_scalar('train/loss_sup',   sup_loss.item(),   iter_num)
            writer.add_scalar('train/loss_unsup', unsup_loss.item(), iter_num)
            writer.add_scalar('train/loss_spec',  spec_loss.item(),  iter_num)
            writer.add_scalar('train/cw',          float(cw),         iter_num)
            writer.add_scalar('daep/w_mean',       daep_diag['daep_w_mean'], iter_num)
            writer.add_scalar('daep/u_ale_mean',   daep_diag['daep_u_ale_mean'], iter_num)
            writer.add_scalar('daep/u_epi_mean',   daep_diag['daep_u_epi_mean'], iter_num)
            writer.add_scalar('daep/u_dom_mean',   daep_diag['daep_u_dom_mean'], iter_num)

            if cadg is not None:
                for k, v in cadg.last_stats().items():
                    writer.add_scalar(f'cadg/{k}', v, iter_num)

            if p_bar is not None:
                p_bar.update()
                p_bar.set_description(
                    f'E{epoch_num+1} [v16] '
                    f'L:{total_loss.item():.3f} '
                    f'sup:{sup_loss.item():.3f} '
                    f'uns:{unsup_loss.item():.3f} '
                    f'w:{daep_diag["daep_w_mean"]:.2f}'
                )

            if iter_num % args.num_eval_iter == 0:
                logging.info(
                    f'iter {iter_num}: '
                    f'loss={stats["loss_total"].avg:.4f}, '
                    f'sup={stats["loss_sup"].avg:.4f}, '
                    f'unsup={stats["loss_unsup"].avg:.4f}, '
                    f'spec={stats["loss_spec"].avg:.4f}, '
                    f'cw={stats["cw"].avg:.3f}, '
                    f'w_mean={stats["daep_w_mean"].avg:.3f}, '
                    f'w_min={stats["daep_w_min"].avg:.3f}, '
                    f'U_ale={stats["daep_u_ale_mean"].avg:.3f}, '
                    f'U_epi={stats["daep_u_epi_mean"].avg:.3f}, '
                    f'U_dom={stats["daep_u_dom_mean"].avg:.3f}, '
                    f'S_mean={stats["alpha_S_mean"].avg:.2f}'
                    + ''.join(f', {k}={stats[k].avg:.4g}' for k in sorted(stats)
                              if k.startswith(('devis_', 'mc_', 'medl_')))
                )

        if p_bar is not None:
            p_bar.close()

        # =====================================================================
        # Evaluation
        # =====================================================================
        logging.info(f'Testing student...')
        if args.method == 'medl':
            # net2 and the ensemble are logged every epoch; the headline
            # (best_avg / checkpoint) follows --medl_report.
            d2, _, _ = test(args, model2, test_dataloader, epoch_num + 1, writer,
                            model_name='v18_medl_net2', patch_size=patch_size,
                            n_part=n_part, part=part, dice_calcu_fn=dice_calcu_fn)
            ens = EnsembleProb(model, model2)
            de, _, _ = test(args, ens, test_dataloader, epoch_num + 1, writer,
                            model_name='v18_medl_ensemble', patch_size=patch_size,
                            n_part=n_part, part=part, dice_calcu_fn=dice_calcu_fn)
            logging.info(f'[medl] net2_avg={float(np.mean(d2)):.4f} '
                         f'ensemble_avg={float(np.mean(de)):.4f}')
            headline_model = {'net1': model, 'net2': model2, 'ensemble': ens}[args.medl_report]
        else:
            headline_model = model
        val_dice, val_dsdv, val_pdm = test(
            args, headline_model, test_dataloader, epoch_num + 1, writer,
            model_name='v16_student',
            patch_size=patch_size, n_part=n_part, part=part,
            dice_calcu_fn=dice_calcu_fn,
        )
        logging.info(f'Testing EMA teacher...')
        ema_val_dice, ema_dsdv, ema_pdm = test(
            args, ema_model, test_dataloader, epoch_num + 1, writer,
            model_name='v16_ema',
            patch_size=patch_size, n_part=n_part, part=part,
            dice_calcu_fn=dice_calcu_fn,
        )

        # Track best (student)
        avg_student = float(np.mean(val_dice))
        avg_ema = float(np.mean(ema_val_dice))
        for n in range(n_part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]

        if avg_student > best_avg_dice:
            best_avg_dice = avg_student
            best_avg_dice_iter = iter_num
            dice_of_best_avg = list(val_dice)
            best_dsdv = val_dsdv
            best_per_domain_metrics = val_pdm
            if args.save_model:
                ckpt = {
                    'iter': iter_num,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_avg_dice': best_avg_dice,
                    'v18_config': vars(args),
                }
                if model2 is not None:
                    ckpt['state_dict2'] = model2.state_dict()
                    ckpt['ema_state_dict2'] = ema_model2.state_dict()
                torch.save(ckpt, os.path.join(snapshot_path, 'best_student.pth'))
        if avg_ema > best_ema_avg_dice:
            best_ema_avg_dice = avg_ema
        if val_dsdv < best_dsdv:
            best_dsdv = val_dsdv

        # SWAD eval (when enough snapshots exist)
        if (args.swad_enable and swad_model is not None
                and swad_averager.count >= 5):
            swad_averager.load_into(swad_model)
            recalibrate_bn(swad_model, lb_dataloader,
                           args.swad_bn_recalib_batches)
            logging.info(f'Testing SWAD (snapshots={swad_averager.count})...')
            swad_val_dice, swad_dsdv, swad_pdm = test(
                args, swad_model, test_dataloader, epoch_num + 1, writer,
                model_name='v16_swad',
                patch_size=patch_size, n_part=n_part, part=part,
                dice_calcu_fn=dice_calcu_fn,
            )
            avg_swad = float(np.mean(swad_val_dice))
            if avg_swad > best_swad_dice:
                best_swad_dice = avg_swad
                best_swad_iter = iter_num
                if args.save_model:
                    torch.save({
                        'iter': iter_num,
                        'state_dict': swad_model.state_dict(),
                        'best_swad_dice': best_swad_dice,
                    }, os.path.join(snapshot_path, 'best_swad.pth'))

        logging.info(
            f'[epoch {epoch_num+1}] '
            f'student_avg={avg_student:.4f}  ema_avg={avg_ema:.4f}  '
            f'best_avg={best_avg_dice:.4f} @ iter {best_avg_dice_iter}  '
            f'swad_best={best_swad_dice:.4f} @ iter {best_swad_iter}  '
            f'best_DSDV={best_dsdv:.4f}'
        )
        if (args.resume != 'off' and (epoch_num + 1) % max(args.resume_every_epochs, 1) == 0
                and epoch_num + 1 < max_epoch):
            save_resume_state(epoch_num)
            if args.die_after_epoch == epoch_num:
                logging.info(f'[resume-test] dying on purpose after epoch {epoch_num + 1}')
                writer.close(); sys.exit(3)

    # --------------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------------
    if os.path.exists(rs_path):                        # complete runs keep no resume state (disk)
        os.remove(rs_path)
    logging.info("=" * 60)
    logging.info("TRAINING COMPLETE — FourierFuse v16 summary")
    logging.info(f"  Best avg dice (student): {best_avg_dice:.6f} "
                 f"at iter {best_avg_dice_iter}")
    for n, p in enumerate(part):
        logging.info(f"  Best {p} dice: {best_dice[n]:.6f}")
    logging.info(f"  Best DSDV: {best_dsdv:.6f}")
    if args.swad_enable and best_swad_dice > 0:
        logging.info(f"  Best SWAD dice: {best_swad_dice:.6f} "
                     f"at iter {best_swad_iter}")
        logging.info(f"  SWAD - student best delta: "
                     f"{best_swad_dice - best_avg_dice:+.6f}")
    if best_per_domain_metrics:
        logging.info(f"  --- Per-domain metrics at best avg dice "
                     f"(iter {best_avg_dice_iter}) ---")
        for dc in sorted(best_per_domain_metrics.keys()):
            dm = best_per_domain_metrics[dc]
            for n, p in enumerate(part):
                d_v = dm['dice'][n] if n < len(dm['dice']) else 0.0
                h_v = dm['hd95'][n] if n < len(dm['hd95']) else 0.0
                a_v = dm['asd'][n]  if n < len(dm['asd'])  else 0.0
                logging.info(f"    Domain {dc} — {p}: "
                             f"Dice={d_v:.6f}, HD95={h_v:.4f}, ASD={a_v:.4f}")
    logging.info("=" * 60)

    writer.close()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    # Snapshot path
    if len(args.save_name) == 0:
        args.save_name = (
            f'v16_{args.dataset}_bf{args.ff_base_features}'
            f'_lb{args.lb_num}_dm{args.lb_domain}_s{args.seed}'
        )
    snapshot_path = f"../model/{args.dataset}/train_fourierfuse_v16/{args.save_name}/"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    resume_from = None
    run_lock = os.path.join(snapshot_path, 'RUNNING.lock')      # E-23: a run directory in use is never touched
    if os.path.exists(run_lock):
        try:
            other = open(run_lock).read().split(); opid = int(other[0])
        except Exception:
            opid = -1
        alive = False
        if opid > 0 and opid != os.getpid():
            try:
                os.kill(opid, 0); alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        if alive:
            print(f"[v18] {snapshot_path} is being trained by pid {opid} (started {' '.join(other[1:3])}); "
                  f"refusing to resume, wipe or share it. Exit 6.")
            sys.exit(6)
        print(f"[v18] stale RUNNING.lock from pid {opid} (not alive) -> ignored")
    if os.path.exists(snapshot_path):
        rs_path = os.path.join(snapshot_path, 'resume_state.pth')
        log_path = os.path.join(snapshot_path, 'log.txt')
        already_complete = (os.path.exists(log_path)
                            and 'TRAINING COMPLETE' in open(log_path, errors='replace').read())
        if args.resume == 'auto' and os.path.exists(rs_path) and not already_complete:
            resume_from = rs_path                      # E-14: continue, do not wipe
        elif args.resume == 'auto' and already_complete:
            print(f"[v18] {snapshot_path} is already COMPLETE (TRAINING COMPLETE in log.txt); "
                  f"nothing to do. Pass --resume off to wipe and redo it.")
            sys.exit(0)                                # never destroy a finished experiment by accident
        elif args.overwrite:
            shutil.rmtree(snapshot_path)
        else:
            raise Exception(
                f'snapshot_path {snapshot_path} exists. Use --overwrite.'
            )
    os.makedirs(snapshot_path, exist_ok=True)
    args._resume_from = resume_from
    with open(run_lock, 'w') as _f:                               # E-23: mark the directory as in use
        _f.write(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')} {socket.gethostname()}\n")
    import atexit as _atexit
    _atexit.register(lambda: os.path.exists(run_lock) and open(run_lock).read().split()[0] == str(os.getpid())
                     and os.remove(run_lock))

    # Copy source files for reproducibility
    code_dir = os.path.join(snapshot_path, 'code')
    os.makedirs(code_dir, exist_ok=True)
    try:
        shutil.copy(f'./{sys.argv[0]}', os.path.join(code_dir, os.path.basename(sys.argv[0])))
        if os.path.exists('./fourierfuse_v16_modules.py'):
            shutil.copy('./fourierfuse_v16_modules.py',
                        os.path.join(code_dir, 'fourierfuse_v16_modules.py'))
    except Exception as e:
        logging.warning(f"Could not copy source files: {e}")

    # Logging
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(" ".join(["python"] + sys.argv))
    logging.info(str(args))

    # v18 provenance recorded AT LAUNCH (R-09): args, code hashes, environment.
    def _sha(path):
        try:
            return hashlib.sha256(open(path, 'rb').read()).hexdigest()
        except Exception:
            return None
    normalise_method_args(args)   # v10 (E-38): provenance records what train() will consume
    provenance = {
        'argv': sys.argv, 'args': vars(args), 'snapshot_path': snapshot_path,
        'script_sha256': _sha(sys.argv[0]),
        'fourierfuse_v16_modules_sha256': _sha('./fourierfuse_v16_modules.py'),
        'revision_modules_sha256': _sha('./revision_modules.py'),
        'hostname': socket.gethostname(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'evid_default_cfg': _EVID_DEFAULT_CFG,
    }
    rc_path = os.path.join(snapshot_path, 'run_config.json')
    if resume_from is None or not os.path.exists(rc_path):
        with open(rc_path, 'w') as f:
            json.dump(provenance, f, indent=1, default=str)
    try:
        shutil.copy('./revision_modules.py', os.path.join(code_dir, 'revision_modules.py'))
    except Exception as e:
        logging.warning(f"Could not copy revision_modules.py: {e}")

    train(args, snapshot_path)


if __name__ == '__main__':
    main()
