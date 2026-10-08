#!/usr/bin/env python3
"""
test_ramp_schedule_s10.py — torch-free test of the S10 consistency-schedule change (D-13, E-28).

Extracts the REAL source of `get_current_consistency_weight`, `compute_consistency_weight` and
`_selftest_ramp` from train_fourierfuse_v18_revision.py with `ast` (no import of torch), binds the
real `utils.ramps`, and asserts:
  1. R-27: with --consistency_ramp_frac at its default (-1) the new function reproduces the pre-S10
     inline formula  cw = min(consistency * sigmoid_rampup(iter // max(1, int(T/150)), 150), cap)
     at EVERY iteration of 60 000- and 30 000-iteration runs (exact float equality).
  2. D-13: with --consistency_ramp_frac 0.10 the schedule is the per-iteration Laine-Aila sigmoid
     over L = int(0.1*T) iterations, cap reached at ceil(0.628 L), constant 0.5 afterwards.
  3. The in-script --selftest_ramp routine passes.
Run from CODE/source (needs utils/ramps.py):  python test_ramp_schedule_s10.py
"""
import ast, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ramps

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_fourierfuse_v18_revision.py')
tree = ast.parse(open(SRC, encoding='utf-8').read())
wanted = {'get_current_consistency_weight', 'compute_consistency_weight', '_selftest_ramp'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
assert {n.name for n in nodes} == wanted, 'functions not found in the script'
ns = {'ramps': ramps, 'args': types.SimpleNamespace(consistency=1.0, consistency_rampup=150.0,
                                                   consistency_weight_cap=0.5, consistency_ramp_frac=-1.0)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), SRC, 'exec'), ns)
args = ns['args']; cw_new = ns['compute_consistency_weight']


def cw_old(iter_num, T):   # the pre-S10 inline code, verbatim
    epoch_proxy = iter_num // max(1, int(T / args.consistency_rampup))
    raw = args.consistency * ramps.sigmoid_rampup(epoch_proxy, args.consistency_rampup)
    return min(raw, args.consistency_weight_cap)


# 1. R-27 identity on the default path
for T in (60000, 30000, 20):
    args.consistency_ramp_frac = -1.0
    for t in range(T + 1):
        a, b = cw_old(t, T), cw_new(t, T)
        assert a == b, (T, t, a, b)
print('[test] R-27: default path identical to the pre-S10 formula at every iteration of T=60000/30000/20')
args.consistency_ramp_frac = -1.0
print('[test] E-28 values (T=60000): 20%%=%.4f 50%%=%.4f 63.3%%=%.4f' % (cw_new(12000, 60000), cw_new(30000, 60000), cw_new(38000, 60000)))
assert abs(cw_new(12000, 60000) - 0.0408) < 0.001 and abs(cw_new(30000, 60000) - 0.2865) < 0.001 and cw_new(38000, 60000) == 0.5

# 2. D-13 short ramp
args.consistency_ramp_frac = 0.10
for T in (60000, 30000):
    L = int(0.1 * T)
    for t in range(T + 1):
        expect = min(0.5, ramps.sigmoid_rampup(t, L))
        assert cw_new(t, T) == expect, (T, t)
    first_cap = next(t for t in range(T + 1) if cw_new(t, T) == 0.5)
    print(f'[test] D-13 ramp_frac=0.10, T={T}: L={L} it., cw(0)={cw_new(0,T):.4f}, cap 0.5 first reached at iter {first_cap} = {100*first_cap/T:.2f} % of training')
    assert abs(first_cap / L - 0.628) < 0.01
    assert all(cw_new(t, T) == 0.5 for t in range(first_cap, T + 1))

# 3. in-script self-test
args.consistency_ramp_frac = -1.0
rc = ns['_selftest_ramp']()
assert rc == 0, 'in-script --selftest_ramp failed'
print('[test] ALL PASSED')
