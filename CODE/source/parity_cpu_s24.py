#!/usr/bin/env python3
"""parity_cpu_s24.py (S24) -- parts [3] and [4] of test_devis_released_s23.py on a CPU.

The S23 test needs a GPU for part [4] only because the released KL() calls .cuda(). Here .cuda() is made a no-op
(it moves a tensor and does no arithmetic), the device is the CPU, and everything else is taken from the S23 test
unchanged: the same fixtures, epochs, loops, tolerances and assertions. Loads
  - revision_modules.py from --tools (the file whose sha256 the six runs recorded), and
  - models/criterions.py from --devis_repo (the authors' released code, unmodified on disk).
Prints the two result lines in the format of the S23 test and the hashes of the three files it compared.
"""
import argparse, hashlib, importlib.util, os, sys
import torch

def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tools', required=True); ap.add_argument('--devis_repo', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.tools)
    import test_devis_released_s23 as t23
    import revision_modules as rm
    crit = os.path.join(a.devis_repo, 'models', 'criterions.py'); assert os.path.exists(crit), crit
    torch.Tensor.cuda = lambda self, *x, **k: self          # the only change: .cuda() is a no-op on this machine
    dev = torch.device('cpu'); n3 = n4 = 0
    for K in (2, 3):
        for seed in (0, 1):
            E, y = t23.fixture(K, seed); al = torch.tensor(E + 1.0, dtype=torch.float32, device=dev); yt = torch.tensor(y, dtype=torch.int64, device=dev)
            for ep in t23.EPOCHS:
                ref = t23.ref_loss(E, y, K, ep); out = rm.devis_released_sup_loss(al, yt, ep, 99)
                for k in ('total', 'ice_kl', 'au', 'dice_w'):
                    assert t23.close(float(out[k]), ref[k], 2e-4, 2e-4), ('torch', K, seed, ep, k, float(out[k]), ref[k]); n3 += 1
    print(f'[3] revision_modules.devis_released_sup_loss on {dev}: {n3} values equal to the reference (2e-4)')
    spec = importlib.util.spec_from_file_location('devis_released_criterions', crit); off = importlib.util.module_from_spec(spec); spec.loader.exec_module(off)
    worst_v = worst_g = 0.0
    for K in (2, 3):
        for seed in (0, 1, 2):
            for scale in (2.0, 6.0):
                E, y = t23.fixture(K, seed, scale=scale); yt = torch.tensor(y, dtype=torch.int64, device=dev)
                for ep in t23.EPOCHS:
                    e1 = torch.tensor(E, dtype=torch.float32, device=dev, requires_grad=True); e2 = torch.tensor(E, dtype=torch.float32, device=dev, requires_grad=True)
                    lo = torch.mean(off.dce_evidence_u_loss(yt, e1 + 1.0, K, ep, 50, 100, 1e-10, False, e1, None)); lo.backward()
                    lm = rm.devis_released_sup_loss(e2 + 1.0, yt, ep, 99)['total']; lm.backward()
                    dv = abs(float(lo) - float(lm)); dg = float((e1.grad - e2.grad).abs().max()); gs = float(e1.grad.abs().max())
                    assert dv <= 1e-5 + 1e-5 * abs(float(lo)), ('released code, value', K, seed, scale, ep, float(lo), float(lm))
                    assert dg <= 1e-7 + 1e-4 * gs, ('released code, gradient', K, seed, scale, ep, dg, gs)
                    worst_v = max(worst_v, dv); worst_g = max(worst_g, dg / max(gs, 1e-12)); n4 += 1
    print(f'[4] against the released functions ({crit}): {n4} cases, largest value difference {worst_v:.2e}, largest relative gradient difference {worst_g:.2e}')
    print(f'EPOCHS tested: {list(t23.EPOCHS)}')
    print(f'sha256 revision_modules.py {sha(os.path.join(a.tools, "revision_modules.py"))[:16]}  test_devis_released_s23.py {sha(os.path.join(a.tools, "test_devis_released_s23.py"))[:16]}  criterions.py {sha(crit)[:16]}')
    print(f'PARITY ON CPU PASSED ({n3} + {n4} checks; torch {torch.__version__}; released code imported from {a.devis_repo})')

if __name__ == '__main__':
    main()
