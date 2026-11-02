#!/usr/bin/env python3
r"""make_power_s29.py -- S29 (audit F-1, proposal C-2): what three seeds can and cannot exclude in the headline gate comparison.
Reader, assertion-first (R-48): reads the per-seed values of the four paired tests of Table main (table_main_gate_tests_s17.csv,
the file the S17 layer pinned the printed means and p on), re-derives the printed numbers, then computes for each benchmark
  - the paired t-test (re-derived; asserted equal to the stored p to 4 decimals),
  - the 90 % confidence interval of the mean difference (the interval a TOST at alpha 0.05 uses),
  - the TOST at the registered margin of +-0.5 Dice points (Phase-B criterion), p_TOST = max of the two one-sided p,
  - the minimum detectable effect (MDE): the smallest |Delta| a two-sided paired t at alpha 0.05 detects with power 0.8, n = 3,
    given the observed sd of the paired differences (noncentral t, scipy).
No sentence is written from this file until its facts are adjudicated (N-62). CPU only; no training.
  python make_power_s29.py --csv <derived>/extra_s17/table_main_gate_tests_s17.csv --out <derived>/extra_s29"""
import argparse, csv, json, math, os, sys
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

MARGIN, ALPHA, POWER = 0.5, 0.05, 0.8


def mde(sd_d, n):
    # power of the two-sided paired t, written as an integral over V ~ chi2(df): P(|Z + nc| > tc sqrt(V/df)), Z ~ N(0,1).
    # (scipy.stats.nct returns NaN at df = 2 for some noncentralities, so the noncentral t is integrated directly.)
    se = sd_d / math.sqrt(n); df = n - 1; tc = stats.t.ppf(1 - ALPHA / 2, df)

    def pw(d):
        nc = d / se; f = lambda v: (stats.norm.sf(tc * math.sqrt(v / df) - nc) + stats.norm.cdf(-tc * math.sqrt(v / df) - nc)) * stats.chi2.pdf(v, df)
        return quad(f, 0, math.inf, limit=200)[0]
    assert abs(pw(1e-9 * se) - ALPHA) < 1e-4, 'power at zero effect must equal alpha'
    return brentq(lambda d: pw(d) - POWER, 1e-6 * se, 60 * se)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, encoding='utf-8'))); out = {}; n_ass = 0

    def ok(c, m):
        nonlocal n_ass
        n_ass += 1
        assert c, m
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        per = [r for r in rows if r['dataset'] == ds and r['seed'] != 'paired']
        pr = [r for r in rows if r['dataset'] == ds and r['seed'] == 'paired']
        ok(len(per) == 3 and len(pr) == 1, f'{ds}: three seeds and one paired row')
        d = [100 * (float(r['daep']) - float(r['threshold'])) for r in per]; n = len(d)
        m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)); se = sd / math.sqrt(n)
        t = m / se; p = 2 * stats.t.sf(abs(t), n - 1)
        stored = pr[0]['threshold_source']                                    # "sd ..; diff +0.015; t 0.089; p 0.9369"
        sp = float(stored.split('p ')[-1]); sdiff = float(stored.split('diff ')[1].split(';')[0])
        ok(abs(p - sp) < 5e-5, f'{ds}: paired p re-derived {p:.4f} vs stored {sp}')
        ok(abs(m - sdiff) < 5e-4, f'{ds}: mean difference re-derived {m:.3f} vs stored {sdiff}')
        ok(abs(sum(float(r["daep"]) for r in per) / 3 * 100 - float(pr[0]['daep'])) < 1e-3, f'{ds}: DAEP mean')
        t90 = stats.t.ppf(0.95, n - 1); lo, hi = m - t90 * se, m + t90 * se
        p_lo = stats.t.sf((m + MARGIN) / se, n - 1); p_hi = stats.t.cdf((m - MARGIN) / se, n - 1); p_tost = max(p_lo, p_hi)
        ok((p_tost < ALPHA) == (lo > -MARGIN and hi < MARGIN), f'{ds}: TOST verdict = 90 % CI inside the margin')
        e = mde(sd, n)
        out[ds] = {'n': n, 'diffs_points': [round(x, 4) for x in d], 'mean_diff': round(m, 4), 'sd_diff': round(sd, 4),
                   'p_paired': round(p, 4), 'ci90': [round(lo, 3), round(hi, 3)], 'p_tost_0.5': round(p_tost, 4),
                   'equivalent_at_0.5': bool(p_tost < ALPHA), 'mde_alpha05_power80': round(e, 3)}
    ok(all(out[k]['p_paired'] > 0.1 for k in out), 'no headline paired test is below 0.1 (Table main)')
    ok(min(out[k]['p_paired'] for k in out) >= 0.18 - 0.005 and max(out[k]['p_paired'] for k in out) <= 0.94 + 0.005,
       'paired p within the 0.18-0.94 that Section 1.2 prints')
    os.makedirs(a.out, exist_ok=True)
    json.dump({'margin_points': MARGIN, 'alpha': ALPHA, 'power': POWER, 'source': os.path.basename(a.csv), 'benchmarks': out,
               'assertions': n_ass}, open(os.path.join(a.out, 'facts_s29_power.json'), 'w', encoding='utf-8'), indent=1)
    for k, v in out.items():
        print(f"{k:9s} diff {v['mean_diff']:+.2f}  sd {v['sd_diff']:.2f}  p {v['p_paired']:.3f}  90%CI [{v['ci90'][0]:+.2f}, {v['ci90'][1]:+.2f}]"
              f"  TOST(0.5) p {v['p_tost_0.5']:.3f} {'EQUIV' if v['equivalent_at_0.5'] else 'not shown'}  MDE {v['mde_alpha05_power80']:.2f}")
    print(f'[power] {n_ass} assertions PASSED; facts -> facts_s29_power.json')


if __name__ == '__main__':
    main()
