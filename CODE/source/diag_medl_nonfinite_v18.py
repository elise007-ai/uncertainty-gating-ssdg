#!/usr/bin/env python3
"""
diag_medl_nonfinite_v18.py  (tools v10, S12 / N-24)  —  O-02 root-cause diagnostic for MEDL.

Question (HANDOFF v1.11 D-15 / O-02): the Prostate MEDL runs skipped 42 % / 21 % of their
iterations for a non-finite loss (seeds 1337 / 2024) and seed 42 ran out degenerate. Is the
non-finite value produced by half precision (fp16 autocast) in my implementation, or is it
present in full precision (i.e. in the objective / the weights the objective produced)?

What it does (forward only, no training, ~10 min on one GPU for 3 checkpoints x 6 domains):
  for every checkpoint given:
    1. loads student (net1, net2) and EMA (net1, net2) with the harness's own load_model()
    2. weight statistics per network: max |w|, global L2 norm, number of non-finite parameters
    3. for the first --n_images test images of every domain, for each of the four networks,
       in fp32 AND under torch.autocast(fp16): logit min / max / #non-finite, #|logit| > 65504
       (the fp16 limit — a finite fp32 logit above it is the signature of a conv-stack overflow),
       evidence max, S min / max, #non-finite U_epi / U_ale
    4. reproduces every MEDL loss term the training loop computes, in both precisions, from the
       same tensors the loop would see (EMA teachers -> CAEF fusion -> pseudo-label; students ->
       mse risk, KL, soft Dice, I-EDL term, log|I(alpha)|, reliability, asymptotic weight), and
       reports the value and finiteness of EACH term (E-35: the training guard printed only net-1's
       sup/unsup/spec)
    5. verdict per checkpoint (see verdict()):
         WEIGHTS_NONFINITE       a parameter is non-finite: the checkpoint itself is broken
         NONFINITE_FP16_ONLY     non-finite values appear only under fp16 autocast -> precision cause
         NONFINITE_FP32_TOO      non-finite values appear in full precision -> not (only) precision
         FP16_OVERFLOW_PREDICTED no non-finite value reproduced, but fp32 logits exceed 65504 ->
                                 the fp16 forward of the TRAINING inputs (strong view + CutMix)
                                 will overflow; precision cause, weights already extreme
         NOT_REPRODUCED          everything finite and bounded on these inputs
Outputs: diag_out/medl_nonfinite_<run>.json (all numbers) and [medl-diag] console lines.

Usage (RUNBOOK_S12 step 6):
  python diag_medl_nonfinite_v18.py --dataset prostate --gpu 1 \
      --ckpt ../model/prostate/train_fourierfuse_v16/v18B_prostate_medl_s1337/best_student.pth \
      --ckpt ../model/prostate/train_fourierfuse_v16/v18B_prostate_medl_s2024/best_student.pth \
      --ckpt ../model/prostate/train_fourierfuse_v16/v18B_prostate_medl_s42/best_student.pth
  python diag_medl_nonfinite_v18.py --selftest          (CPU, needs torch, < 1 min, synthetic)
  python diag_medl_nonfinite_v18.py --selftest_logic    (no torch: verdict logic only)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

FP16_MAX = 65504.0
TAG = '[medl-diag]'


def log(msg: str):
    print(f'{TAG} {msg}', flush=True)


# --------------------------------------------------------------------------- verdict (torch-free)
def verdict(weights: dict, fwd: dict, terms: dict) -> dict:
    """weights: {net: {'n_nonfinite': int, ...}}; fwd: {net: {'fp32': {...}, 'fp16': {...}}} with keys
    n_nonfinite_logits, n_over_fp16max, n_nonfinite_u; terms: {'fp32': {term: {'value', 'finite'}}, 'fp16': {...}}"""
    if any(w['n_nonfinite'] > 0 for w in weights.values()):
        return {'verdict': 'WEIGHTS_NONFINITE', 'nets': [n for n, w in weights.items() if w['n_nonfinite'] > 0]}
    nf32 = [f'{n}:logits' for n, s in fwd.items() if s['fp32']['n_nonfinite_logits'] > 0] + \
           [f'{n}:U' for n, s in fwd.items() if s['fp32']['n_nonfinite_u'] > 0] + \
           [f'term:{t}' for t, v in terms.get('fp32', {}).items() if not v['finite']]
    nf16 = [f'{n}:logits' for n, s in fwd.items() if s['fp16']['n_nonfinite_logits'] > 0] + \
           [f'{n}:U' for n, s in fwd.items() if s['fp16']['n_nonfinite_u'] > 0] + \
           [f'term:{t}' for t, v in terms.get('fp16', {}).items() if not v['finite']]
    if nf32:
        return {'verdict': 'NONFINITE_FP32_TOO', 'fp32': nf32, 'fp16': nf16}
    if nf16:
        return {'verdict': 'NONFINITE_FP16_ONLY', 'fp16': nf16}
    over = {n: s['fp32']['n_over_fp16max'] for n, s in fwd.items() if s['fp32']['n_over_fp16max'] > 0}
    if over:
        return {'verdict': 'FP16_OVERFLOW_PREDICTED', 'n_over_fp16max': over}
    return {'verdict': 'NOT_REPRODUCED'}


# --------------------------------------------------------------------------- torch part
def _unpack(batch):
    import torch
    if isinstance(batch, dict):
        for k in ('image', 'img', 'x'):
            if k in batch and torch.is_tensor(batch[k]):
                return batch[k]
        for v in batch.values():
            if torch.is_tensor(v) and v.ndim == 4:
                return v
        raise KeyError(f'no 4-D image tensor in batch keys {list(batch.keys())}')
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def weight_stats(model) -> dict:
    import torch
    mx, sq, nnf, n = 0.0, 0.0, 0, 0
    for p in model.parameters():
        pf = p.detach().float()
        fin = torch.isfinite(pf)
        nnf += int((~fin).sum().item()); n += pf.numel()
        if fin.any():
            mx = max(mx, float(pf[fin].abs().max().item())); sq += float((pf[fin] ** 2).sum().item())
    return {'max_abs': mx, 'l2_norm': math.sqrt(sq), 'n_nonfinite': nnf, 'n_params': n}


def _head_outputs(model, x):
    """logits + head dict, using the module's own head so the arithmetic is the training loop's."""
    logits = model.get_logits(x)
    head = getattr(model, 'evidential', None)
    out = head(logits) if head is not None else model.forward(x, return_features=False)
    return logits, out


def forward_stats(model, x, use_fp16: bool, device: str) -> dict:
    import torch
    dev_type = 'cuda' if device.startswith('cuda') else 'cpu'
    with torch.no_grad():
        with torch.autocast(dev_type, dtype=torch.float16, enabled=use_fp16 and dev_type == 'cuda'):
            logits, out = _head_outputs(model, x)
    lg = logits.detach().float()
    fin = torch.isfinite(lg)
    st = {'dtype_logits': str(logits.dtype),
          'n_nonfinite_logits': int((~fin).sum().item()),
          'logit_min': float(lg[fin].min().item()) if fin.any() else float('nan'),
          'logit_max': float(lg[fin].max().item()) if fin.any() else float('nan'),
          'n_over_fp16max': int((lg[fin].abs() > FP16_MAX).sum().item()) if fin.any() else 0}
    S = out['S'].detach().float(); e = out['evidence'].detach().float()
    st.update({'evidence_max': float(e[torch.isfinite(e)].max().item()) if torch.isfinite(e).any() else float('nan'),
               'S_min': float(S[torch.isfinite(S)].min().item()) if torch.isfinite(S).any() else float('nan'),
               'S_max': float(S[torch.isfinite(S)].max().item()) if torch.isfinite(S).any() else float('nan'),
               'n_nonfinite_u': int((~torch.isfinite(out['U_epi'])).sum().item() + (~torch.isfinite(out['U_ale'])).sum().item())})
    return st, out


def loss_terms(st1, st2, ema1, ema2, x, use_fp16: bool, device: str, lam_fisher: float, xi: float,
               prior_w: float, evid_loss: str) -> dict:
    """Every term of the MEDL objective the loop builds (train_fourierfuse_v18_revision.py L1198-1431),
    forward-only on x (weak view, no CutMix): value + finiteness. Uses the project's own functions."""
    import torch
    from revision_modules import (caef_fuse, medl_reliability, medl_asymptotic_weight, iedl_pixel_loss,
                                  dirichlet_log_fisher_det, evidential_seg_loss_v18, kl_dirichlet_to_prior)
    from fourierfuse_v16_modules import soft_dice_from_alpha, labels_to_onehot
    dev_type = 'cuda' if device.startswith('cuda') else 'cpu'
    terms = {}

    def rec(name, t):
        v = float(t.detach().float().mean().item()) if torch.is_tensor(t) else float(t)
        terms[name] = {'value': v, 'finite': bool(math.isfinite(v))}

    with torch.no_grad():
        with torch.autocast(dev_type, dtype=torch.float16, enabled=use_fp16 and dev_type == 'cuda'):
            _, t1 = _head_outputs(ema1, x); _, t2 = _head_outputs(ema2, x)
            _, s1 = _head_outputs(st1, x); _, s2 = _head_outputs(st2, x)
            K = t1['belief'].shape[1]
            b1, u1, b2, u2 = t1['belief'].float(), t1['U_epi'].float(), t2['belief'].float(), t2['U_epi'].float()
            Z = (b1 * b2 + (1.0 / (1.0 + K)) * (b1 * u2 + b2 * u1)).sum(dim=1, keepdim=True) + u1 * u2
            rec('caef_Z_min', Z.min())                       # the normaliser BEFORE clamp(min=1e-8)
            b_f, u_f = caef_fuse(b1, u1, b2, u2)
            rec('caef_b_fused_max', b_f.max()); rec('caef_u_fused_mean', u_f.mean())
            R = medl_reliability(b_f, u_f); rec('reliability_mean', R.mean()); rec('reliability_min', R.min())
            om = medl_asymptotic_weight(u_f, 0.5, xi=xi); rec('omega_mean', om.mean())
            pseudo = labels_to_onehot(torch.argmax(b_f, dim=1), K=K)
            for tag, s in (('net1', s1), ('net2', s2)):
                a = s['alpha'].float()
                rec(f'{tag}_risk_{evid_loss}', evidential_seg_loss_v18(a, pseudo, lambda_kl=0.0, loss_type=evid_loss, prior_w=prior_w))
                rec(f'{tag}_kl', kl_dirichlet_to_prior(pseudo * prior_w + (1 - pseudo) * a, prior_w).mean())
                rec(f'{tag}_dice', soft_dice_from_alpha(a, pseudo))
                rec(f'{tag}_logfisher', dirichlet_log_fisher_det(a).mean())
                rec(f'{tag}_iedl', iedl_pixel_loss(a, pseudo, lam_fisher).mean())
                rec(f'{tag}_omega_iedl', (om * iedl_pixel_loss(a, pseudo, lam_fisher)).sum() / (om.sum() + 1e-8))
    return terms


def run_checkpoint(ckpt: str, dataset: str, device: str, n_images: int, out_dir: str, lam_fisher: float,
                   xi: float, prior_w: float, evid_loss: str) -> dict:
    import torch
    import diag_uncertainty_harness_v18 as H
    name = os.path.basename(os.path.dirname(ckpt)) or os.path.basename(ckpt)
    log(f'checkpoint {ckpt}')
    students, meta = H.load_model(dataset, ckpt, 'student', device)
    emas, _ = H.load_model(dataset, ckpt, 'ema', device)
    if len(students) != 2 or len(emas) != 2:
        raise SystemExit(f'{ckpt}: expected a MEDL checkpoint with two networks, got {len(students)}/{len(emas)}')
    nets = {'student_net1': students[0], 'student_net2': students[1], 'ema_net1': emas[0], 'ema_net2': emas[1]}
    weights = {n: weight_stats(m) for n, m in nets.items()}
    for n, w in weights.items():
        log(f'  weights {n}: max|w|={w["max_abs"]:.3e} l2={w["l2_norm"]:.3e} nonfinite={w["n_nonfinite"]}/{w["n_params"]}')
    cfg = H.DATASET_CFG[dataset]
    fwd = {n: {'fp32': None, 'fp16': None} for n in nets}
    agg = {n: {p: {'n_nonfinite_logits': 0, 'n_over_fp16max': 0, 'n_nonfinite_u': 0, 'logit_min': float('inf'),
                   'logit_max': float('-inf'), 'evidence_max': 0.0, 'S_min': float('inf'), 'S_max': 0.0, 'n_images': 0}
               for p in ('fp32', 'fp16')} for n in nets}
    terms_agg = {'fp32': {}, 'fp16': {}}
    n_term_imgs = 0
    for dom in range(1, cfg['domain_num'] + 1):
        loader = H.build_test_loader(dataset, dom)
        seen = 0
        for batch in loader:
            if seen >= n_images:
                break
            x = _unpack(batch).to(device).float()
            for prec, f16 in (('fp32', False), ('fp16', True)):
                for n, m in nets.items():
                    st, _ = forward_stats(m, x, f16, device)
                    a = agg[n][prec]
                    for k in ('n_nonfinite_logits', 'n_over_fp16max', 'n_nonfinite_u'):
                        a[k] += st[k]
                    a['logit_min'] = min(a['logit_min'], st['logit_min']); a['logit_max'] = max(a['logit_max'], st['logit_max'])
                    a['evidence_max'] = max(a['evidence_max'], st['evidence_max'] if math.isfinite(st['evidence_max']) else 0.0)
                    a['S_min'] = min(a['S_min'], st['S_min']); a['S_max'] = max(a['S_max'], st['S_max']); a['n_images'] += 1
                t = loss_terms(nets['student_net1'], nets['student_net2'], nets['ema_net1'], nets['ema_net2'], x, f16,
                               device, lam_fisher, xi, prior_w, evid_loss)
                for k, v in t.items():
                    d = terms_agg[prec].setdefault(k, {'value_sum': 0.0, 'n': 0, 'finite': True, 'first_nonfinite_domain': None})
                    d['value_sum'] += v['value'] if v['finite'] else 0.0; d['n'] += 1
                    if not v['finite']:
                        d['finite'] = False
                        d['first_nonfinite_domain'] = d['first_nonfinite_domain'] or dom
            seen += 1; n_term_imgs += 1
        log(f'  domain {dom}: {seen} images')
    for n in nets:
        for p in ('fp32', 'fp16'):
            fwd[n][p] = agg[n][p]
            a = agg[n][p]
            log(f'  fwd {n:12s} {p}: logits [{a["logit_min"]:.3e}, {a["logit_max"]:.3e}] nonfinite={a["n_nonfinite_logits"]} '
                f'over65504={a["n_over_fp16max"]} S=[{a["S_min"]:.3e}, {a["S_max"]:.3e}] nonfiniteU={a["n_nonfinite_u"]}')
    terms = {p: {k: {'value': (d['value_sum'] / max(d['n'], 1)), 'finite': d['finite'],
                     'first_nonfinite_domain': d['first_nonfinite_domain']} for k, d in terms_agg[p].items()}
             for p in ('fp32', 'fp16')}
    for p in ('fp32', 'fp16'):
        bad = [k for k, v in terms[p].items() if not v['finite']]
        log(f'  terms {p}: {len(terms[p])} terms, non-finite: {bad if bad else "none"}')
    v = verdict(weights, fwd, terms)
    log(f'  VERDICT {name}: {json.dumps(v)}')
    res = {'run': name, 'ckpt': ckpt, 'ckpt_sha256': meta.get('ckpt_sha256'), 'iter': meta.get('iter'),
           'best_avg_dice': meta.get('best_avg_dice'), 'n_images_per_domain': n_images, 'n_images_total': n_term_imgs,
           'weights': weights, 'forward': fwd, 'terms': terms, 'verdict': v, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
           'note': 'forward-only on weak test images; the training inputs (strong view + CutMix) can overflow earlier'}
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'medl_nonfinite_{name}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    log(f'  [OK] {out}')
    return res


# --------------------------------------------------------------------------- self-tests
def _selftest_logic():
    W = {'a': {'n_nonfinite': 0}, 'b': {'n_nonfinite': 0}}
    F = {'a': {'fp32': {'n_nonfinite_logits': 0, 'n_over_fp16max': 0, 'n_nonfinite_u': 0},
               'fp16': {'n_nonfinite_logits': 0, 'n_over_fp16max': 0, 'n_nonfinite_u': 0}}}
    T = {'fp32': {'t': {'value': 1.0, 'finite': True}}, 'fp16': {'t': {'value': 1.0, 'finite': True}}}
    assert verdict(W, F, T)['verdict'] == 'NOT_REPRODUCED'
    F2 = json.loads(json.dumps(F)); F2['a']['fp32']['n_over_fp16max'] = 3
    assert verdict(W, F2, T)['verdict'] == 'FP16_OVERFLOW_PREDICTED'
    F3 = json.loads(json.dumps(F)); F3['a']['fp16']['n_nonfinite_logits'] = 1
    assert verdict(W, F3, T)['verdict'] == 'NONFINITE_FP16_ONLY'
    T4 = json.loads(json.dumps(T)); T4['fp32']['t']['finite'] = False
    assert verdict(W, F3, T4)['verdict'] == 'NONFINITE_FP32_TOO'
    W5 = {'a': {'n_nonfinite': 2}, 'b': {'n_nonfinite': 0}}
    assert verdict(W5, F3, T4) == {'verdict': 'WEIGHTS_NONFINITE', 'nets': ['a']}
    log('selftest_logic PASSED (5 verdict branches)')


def _selftest():
    """CPU, synthetic: two tiny evidential nets, one healthy, one with weights that overflow fp16."""
    import torch, torch.nn as nn
    from revision_modules import EvidentialHeadV18
    _selftest_logic()
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1); self.evidential = EvidentialHeadV18(2)
            with torch.no_grad():
                self.conv.weight.mul_(scale); self.conv.bias.zero_()
        def get_logits(self, x):
            return self.conv(x)
        def forward(self, x, return_features=False):
            return self.evidential(self.get_logits(x))
    x = torch.rand(1, 1, 16, 16) * 50.0
    healthy, hot = Tiny(1.0), Tiny(1e5)
    st_h, _ = forward_stats(healthy, x, False, 'cpu'); st_x, _ = forward_stats(hot, x, False, 'cpu')
    assert st_h['n_nonfinite_logits'] == 0 and st_h['n_over_fp16max'] == 0, st_h
    assert st_x['n_over_fp16max'] > 0, st_x                      # fp32 finite but above the fp16 limit
    with torch.no_grad():                                        # the same logits in fp16 overflow to inf
        lg16 = hot.get_logits(x).half()
    assert (~torch.isfinite(lg16)).any(), 'fp16 cast did not overflow — synthetic scale too small'
    ws = weight_stats(hot); assert ws['n_nonfinite'] == 0 and ws['max_abs'] > 1e3
    t = loss_terms(healthy, healthy, healthy, healthy, x, False, 'cpu', 0.1, 1.0, 1.0, 'mse')
    for k, v in t.items():
        assert v['finite'], (k, v)
    assert 'net1_logfisher' in t and 'caef_Z_min' in t and len(t) >= 18, list(t)
    W = {'student_net1': ws, 'student_net2': ws, 'ema_net1': ws, 'ema_net2': ws}
    F = {n: {'fp32': st_x, 'fp16': st_x} for n in W}
    assert verdict(W, F, {'fp32': t, 'fp16': t})['verdict'] == 'FP16_OVERFLOW_PREDICTED'
    log('selftest PASSED (forward_stats, weight_stats, loss_terms (%d terms), verdict)' % len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='prostate'); ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--gpu', default='0'); ap.add_argument('--n_images', type=int, default=8)
    ap.add_argument('--out_dir', default='diag_out')
    ap.add_argument('--medl_lambda_fisher', type=float, default=0.1); ap.add_argument('--medl_xi', type=float, default=1.0)
    ap.add_argument('--evid_prior_w', type=float, default=1.0); ap.add_argument('--evid_loss', default='mse')
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--selftest_logic', action='store_true')
    a = ap.parse_args()
    if a.selftest_logic:
        _selftest_logic(); return
    if a.selftest:
        _selftest(); return
    if not a.ckpt:
        raise SystemExit('give at least one --ckpt (a MEDL best_student.pth) or --selftest')
    import torch
    device = f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu'   # --gpu alone selects (E-26)
    log(f'device {device}; torch {torch.__version__}; checkpoints {len(a.ckpt)}; {a.n_images} images/domain')
    results = [run_checkpoint(c, a.dataset, device, a.n_images, a.out_dir, a.medl_lambda_fisher, a.medl_xi,
                              a.evid_prior_w, a.evid_loss) for c in a.ckpt]
    log('SUMMARY ' + '; '.join(f'{r["run"]}={r["verdict"]["verdict"]}' for r in results))


if __name__ == '__main__':
    main()
