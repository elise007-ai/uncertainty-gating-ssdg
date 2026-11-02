#!/usr/bin/env python3
"""test_devis_released_s23.py -- parity tests for the released-form DEviS objective (tools v11, S23, D-28).

  python test_devis_released_s23.py
      torch-free (analyst's dry run, also runs on the server): (1) a numpy reference that follows the released code line by line
      (models/criterions.py: DiceLoss, KL, dce_evidence_u_loss; models/TrustworthySeg.py: torch.mean); (2) the three new functions of
      revision_modules.py, taken from the file by name and executed through a small numpy stand-in for torch, compared with (1).
  python test_devis_released_s23.py --torch --devis_repo ../third_party/DEviS-main --gpu 0
      server: (3) the real functions of revision_modules.py against the numpy reference (float32 tolerance); (4) against the RELEASED
      functions themselves, imported from the unpacked DEviS-main.zip (values and gradients). (4) needs a GPU: the released KL() calls .cuda().
Last line on success: "SELFTEST DEVIS-RELEASED PASSED ...". Any other ending: do not launch; send the console."""
import argparse, ast, importlib.util, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = (0, 1, 25, 49, 50, 75, 98)


def _digamma(x):
    x = np.asarray(x, dtype=np.float64).copy(); r = np.zeros_like(x)
    for _ in range(8):                      # psi(x) = psi(x + 1) - 1 / x, until x >= 8
        m = x < 8.0; r = r - np.where(m, 1.0 / np.where(m, x, 1.0), 0.0); x = np.where(m, x + 1.0, x)
    i = 1.0 / x; i2 = i * i
    return r + np.log(x) - 0.5 * i - i2 * (1.0 / 12 - i2 * (1.0 / 120 - i2 * (1.0 / 252 - i2 * (1.0 / 240 - i2 / 132.0))))


_lgamma = np.vectorize(math.lgamma, otypes=[np.float64])


def fixture(K, seed, n=2, hw=12, scale=2.0):
    rs = np.random.RandomState(seed); z = rs.normal(0.0, scale, size=(n, K, hw, hw))
    E = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)          # softplus
    return E, rs.randint(0, K, size=(n, hw, hw))


def ref_loss(E, y, K, epoch, lam=50, total=100, eps=1e-10):
    """The released objective in numpy (float64), flattened as in the released code."""
    preds = E.transpose(0, 2, 3, 1).reshape(-1, K); t = y.reshape(-1); M = len(t)
    lp = preds - preds.max(1, keepdims=True); P = np.exp(lp - np.log(np.exp(lp).sum(1, keepdims=True)))
    cm = np.zeros_like(preds) + 1e-8; cm[np.arange(M), t] = 1.0
    TP = P * cm; FP = P * (1.0 - cm); FN = (1.0 - P) * cm
    a = FP.sum(0) / ((FP.sum(0) + FN.sum(0)) + 1e-5); a = np.clip(a, 0.2, 0.8); b = 1.0 - a
    num = TP.sum(0); dice = num / (num + a * FP.sum(0) + b * FN.sum(0) + 1e-5); L_dice = (1.0 - dice).sum() / K
    al = (E + 1.0).transpose(0, 2, 3, 1).reshape(-1, K); S = al.sum(1, keepdims=True); Ev = al - 1.0; lab = np.eye(K)[t]
    L_ace = (lab * (_digamma(S) - _digamma(al))).sum(1, keepdims=True)
    ann = min(1.0, epoch / lam); bAU = 0.01 * math.exp(-math.log(0.01) / total * epoch)
    alp = Ev * (1.0 - lab) + 1.0; Sa = alp.sum(1, keepdims=True)
    kl = ((alp - 1.0) * (_digamma(alp) - _digamma(Sa))).sum(1, keepdims=True) + _lgamma(Sa) - _lgamma(alp).sum(1, keepdims=True) - math.lgamma(K)
    p = al / S; ps = p.max(1, keepdims=True); pc = p.argmax(1)[:, None]; unc = K / S; acc = (pc == t[:, None]).astype(np.float64)
    L_AU = bAU * acc * (-ps * np.log(1.0 - unc + eps)) + (1.0 - bAU) * (1.0 - acc) * (-(1.0 - ps) * np.log(unc + eps))
    return {'total': float((L_ace + ann * kl + (1.0 - bAU) * L_dice + L_AU).mean()), 'ice_kl': float((L_ace + ann * kl).mean()),
            'au': float(L_AU.mean()), 'dice_w': float((1.0 - bAU) * L_dice), 'kl_weight': ann, 'beta_t': bAU}


# ---------------- a numpy stand-in for the few torch operations the three functions use ----------------
class T:
    __array_priority__ = 1000
    def __init__(s, a): s.a = np.asarray(a)
    shape = property(lambda s: s.a.shape); dtype = property(lambda s: s.a.dtype); device = 'cpu'
    def float(s): return T(s.a.astype(np.float64))
    def long(s): return T(s.a.astype(np.int64))
    def permute(s, *d): return T(np.transpose(s.a, d))
    def contiguous(s): return s
    def view(s, *sh): return T(s.a.reshape(sh))
    def sum(s, dim=None, keepdim=False): return T(s.a.sum(axis=dim, keepdims=keepdim))
    def mean(s): return T(s.a.mean())
    def unsqueeze(s, d): return T(np.expand_dims(s.a, d))
    def item(s): return float(s.a)
    def __float__(s): return float(s.a)
    def scatter_(s, dim, idx, val):
        assert dim == 1; s.a = s.a.copy(); s.a[np.arange(s.a.shape[0]), idx.a[:, 0]] = val; return s
    def __neg__(s): return T(-s.a)
    def __add__(s, o): return T(s.a + _u(o))
    __radd__ = __add__
    def __sub__(s, o): return T(s.a - _u(o))
    def __rsub__(s, o): return T(_u(o) - s.a)
    def __mul__(s, o): return T(s.a * _u(o))
    __rmul__ = __mul__
    def __truediv__(s, o): return T(s.a / _u(o))
    def __rtruediv__(s, o): return T(_u(o) / s.a)


def _u(o): return o.a if isinstance(o, T) else o


class _FakeTorch:
    Tensor = T
    exp = staticmethod(lambda x: T(np.exp(x.a))); log = staticmethod(lambda x: T(np.log(x.a)))
    digamma = staticmethod(lambda x: T(_digamma(x.a))); lgamma = staticmethod(lambda x: T(_lgamma(x.a)))
    zeros_like = staticmethod(lambda x: T(np.zeros_like(x.a))); eq = staticmethod(lambda a, b: T(a.a == b.a))
    clamp = staticmethod(lambda x, min=None, max=None: T(np.clip(x.a, min, max)))
    tensor = staticmethod(lambda v, device=None, dtype=None: T(np.asarray(v, dtype=np.float64)))
    @staticmethod
    def max(x, dim=None, keepdim=False): return T(x.a.max(axis=dim, keepdims=keepdim)), T(np.expand_dims(x.a.argmax(axis=dim), dim) if keepdim else x.a.argmax(axis=dim))


class _FakeF:
    one_hot = staticmethod(lambda t, num_classes: T(np.eye(num_classes)[t.a]))
    @staticmethod
    def log_softmax(x, dim=1):
        m = x.a - x.a.max(axis=dim, keepdims=True); return T(m - np.log(np.exp(m).sum(axis=dim, keepdims=True)))


def port_through_shim():
    src = open(os.path.join(HERE, 'revision_modules.py')).read(); tree = ast.parse(src)
    want = ('kl_dirichlet_to_prior', 'devis_released_schedule', 'devis_released_dice', 'devis_released_sup_loss')
    got = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want}
    assert set(got) == set(want), sorted(set(want) - set(got))
    ns = {'torch': _FakeTorch, 'F': _FakeF, 'math': math}
    exec(compile('from __future__ import annotations\n' + '\n\n'.join(got[k] for k in want), 'revision_modules(shim)', 'exec'), ns)
    return ns['devis_released_sup_loss']


def close(a, b, tol_abs, tol_rel): return abs(a - b) <= tol_abs + tol_rel * abs(b)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--torch', action='store_true'); ap.add_argument('--devis_repo', default=''); ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args(); n1 = n2 = n3 = n4 = 0
    # (1) the reference against scipy-free identities
    g = 0.5772156649015329                      # psi(1) = -gamma; psi(1/2) = -gamma - 2 ln 2; psi(2) = 1 - gamma; psi(x + 1) - psi(x) = 1 / x
    assert abs(float(_digamma(1.0)) + g) < 1e-10 and abs(float(_digamma(0.5)) + g + 2 * math.log(2)) < 1e-10 and abs(float(_digamma(2.0)) - (1 - g)) < 1e-10
    assert abs(float(_digamma(31.0)) - float(_digamma(30.0)) - 1 / 30.0) < 1e-12 and abs(float(_digamma(3000.5)) - float(_digamma(2999.5)) - 1 / 2999.5) < 1e-12
    r0 = ref_loss(*fixture(2, 0), 2, 0); r98 = ref_loss(*fixture(2, 0), 2, 98)
    assert r0['kl_weight'] == 0.0 and abs(r0['beta_t'] - 0.01) < 1e-15 and r98['kl_weight'] == 1.0 and abs(r98['beta_t'] - 0.01 ** 0.02) < 1e-12
    n1 = 5
    # (2) the port, executed through the stand-in
    fn = port_through_shim()
    for K in (2, 3):
        for seed in (0, 1):
            E, y = fixture(K, seed)
            for ep in EPOCHS:
                ref = ref_loss(E, y, K, ep); out = fn(T(E + 1.0), T(y), ep, 99)
                for k in ('total', 'ice_kl', 'au', 'dice_w', 'kl_weight', 'beta_t'):
                    assert close(float(out[k]), ref[k], 1e-10, 1e-9), ('shim', K, seed, ep, k, float(out[k]), ref[k]); n2 += 1
    print(f'[1] numpy reference: {n1} identities OK   [2] port through the numpy stand-in: {n2} values equal to the reference (1e-9)')
    if not a.torch:
        print(f'SELFTEST DEVIS-RELEASED PASSED (torch-free part only: {n1} + {n2} checks). On the server run it with --torch --devis_repo <DEviS-main>.'); return
    import torch
    sys.path.insert(0, HERE); import revision_modules as rm
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    for K in (2, 3):
        for seed in (0, 1):
            E, y = fixture(K, seed); al = torch.tensor(E + 1.0, dtype=torch.float32, device=dev); yt = torch.tensor(y, dtype=torch.int64, device=dev)
            for ep in EPOCHS:
                ref = ref_loss(E, y, K, ep); out = rm.devis_released_sup_loss(al, yt, ep, 99)
                for k in ('total', 'ice_kl', 'au', 'dice_w'):
                    assert close(float(out[k]), ref[k], 2e-4, 2e-4), ('torch', K, seed, ep, k, float(out[k]), ref[k]); n3 += 1
    print(f'[3] revision_modules.devis_released_sup_loss on {dev}: {n3} values equal to the reference (2e-4)')
    assert a.devis_repo, '--devis_repo <path to the unpacked DEviS-main> is required with --torch'
    crit = os.path.join(a.devis_repo, 'models', 'criterions.py'); assert os.path.exists(crit), crit
    assert torch.cuda.is_available(), 'the released KL() calls .cuda(): run this step on the GPU server'
    torch.cuda.set_device(a.gpu)
    spec = importlib.util.spec_from_file_location('devis_released_criterions', crit); off = importlib.util.module_from_spec(spec); spec.loader.exec_module(off)
    worst_v = worst_g = 0.0
    for K in (2, 3):
        for seed in (0, 1, 2):
            for scale in (2.0, 6.0):                       # ordinary and saturated evidence
                E, y = fixture(K, seed, scale=scale); yt = torch.tensor(y, dtype=torch.int64, device=dev)
                for ep in EPOCHS:
                    e1 = torch.tensor(E, dtype=torch.float32, device=dev, requires_grad=True); e2 = torch.tensor(E, dtype=torch.float32, device=dev, requires_grad=True)
                    lo = torch.mean(off.dce_evidence_u_loss(yt, e1 + 1.0, K, ep, 50, 100, 1e-10, False, e1, None)); lo.backward()
                    lm = rm.devis_released_sup_loss(e2 + 1.0, yt, ep, 99)['total']; lm.backward()
                    dv = abs(float(lo) - float(lm)); dg = float((e1.grad - e2.grad).abs().max()); gs = float(e1.grad.abs().max())
                    assert dv <= 1e-5 + 1e-5 * abs(float(lo)), ('released code, value', K, seed, scale, ep, float(lo), float(lm))
                    assert dg <= 1e-7 + 1e-4 * gs, ('released code, gradient', K, seed, scale, ep, dg, gs)
                    worst_v = max(worst_v, dv); worst_g = max(worst_g, dg / max(gs, 1e-12)); n4 += 1
    print(f'[4] against the released functions ({crit}): {n4} cases, largest value difference {worst_v:.2e}, largest relative gradient difference {worst_g:.2e}')
    print(f'SELFTEST DEVIS-RELEASED PASSED ({n1} + {n2} + {n3} + {n4} checks; released code imported from {a.devis_repo})')


if __name__ == '__main__':
    main()
