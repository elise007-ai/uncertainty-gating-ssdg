#!/usr/bin/env python3
"""
make_extra_s30.py (S30, 2026-10-21) -- N-66 reader: the counts of the text that the six D-31 runs change, assertion-first (R-48, R-69).
It first reproduces the S25 numbers from the derived tables of record, then adds the D-31 runs from facts_s29.json.
  python make_extra_s30.py --derived <derived_v04_S12> --out <derived_v04_S12/extra_s29>
Writes facts_s30_counts.json. No training, no GPU; reads csv/json only.
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_extra_s23 import need, PASSED          # noqa: E402
import make_selection_s24 as s24                  # noqa: E402  (the selection instrument: load(), paired())

DS4 = ('prostate', 'fundus', 'MNMS', 'BUSI'); SEEDS = ('1337', '42', '2024')


def cls_run(run):                                 # the classification of make_selection_s25.py (S9 counts)
    return 'ensemble' if 'ENSEMBLE' in run else 'devis' if 'devis' in run else 'medl' if 'medl' in run else 'exp' if 'act_exp' in run else 'testbed'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--returns', required=True); ap.add_argument('--derived', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    D = a.derived; F25 = json.load(open(os.path.join(D, 'extra_s25_selection', 'facts_s25.json'))); F29 = json.load(open(os.path.join(D, 'extra_s29', 'facts_s29.json')))
    O = {'session': 'S30', 'reader': 'make_extra_s30.py', 'sources': ['extra_s25_selection/table_selection_runs_s25.csv', 'extra_s25_selection/facts_s25.json', 'extra_s29/facts_s29.json',
         'table_signal_identity.csv', 'extra_s25_selection/table_signal_identity_s25.csv', 'extra_s29/table_signal_identity_s29.csv']}
    # ------------------------------------------------ checkpoint selection: every complete log under RETURNS, read by the S24 instrument (exact values, not
    # the two-decimal csv); the S25 numbers are reproduced first from the logs without D-31, then the six D-31 runs are added (R-69)
    R = s24.load(a.returns)
    grp = lambda run, r: 'failure' if (r['flag'] != 'ok' or r['nf'] > 0) else 'short_ramp' if 'ramp10' in run else 'counted'
    old = {k: r for k, r in R.items() if not k[1].startswith('v18F_')}; new = {k: r for k, r in R.items() if k[1].startswith('v18F_')}
    need(len(old) == F25['n_logs'] == 115 and len(new) == 6, 'complete logs: the 115 that facts_s25.json counts, plus the six of D-31 (smoke runs excluded)')
    for (ds, run), r in new.items():
        p = F29[f'{ds}_per_seed'][run.rsplit('_s', 1)[1]]
        need(abs(r['best'] - p['best']) < 1e-9 and abs(r['last'] - p['last']) < 1e-9 and grp(run, r) == 'counted', f'{run}: the S24 instrument reads the D-31 reader\'s best and last values; counted')
    sel = {}
    for ds in DS4:
        b0 = [100 * (r['best'] - r['last']) for (d, run), r in old.items() if d == ds and grp(run, r) == 'counted']; S = F25['size'][ds]
        need(len(b0) == S['n'] and abs(float(np.median(b0)) - S['bl_median']) < 1e-9 and sum(x > 1 for x in b0) == S['bl_n_gt1'] and sum(x > 5 for x in b0) == S['bl_n_gt5'],
             f'{ds}: the logs reproduce the S25 counts exactly (n {S["n"]}, median {S["bl_median"]:.4f}, >1 {S["bl_n_gt1"]}, >5 {S["bl_n_gt5"]})')
        b1 = [100 * (r['best'] - r['last']) for (d, run), r in new.items() if d == ds]; allv = b0 + b1
        sel[ds] = {'n_s25': len(b0), 'n': len(allv), 'median_s25': float(np.median(b0)), 'median': float(np.median(allv)), 'n_gt1_s25': sum(x > 1 for x in b0),
                   'n_gt1': sum(x > 1 for x in allv), 'n_gt5_s25': sum(x > 5 for x in b0), 'n_gt5': sum(x > 5 for x in allv), 'd31_best_minus_last': sorted(b1)}
    need(sel['prostate']['n'] == 55 and sel['fundus']['n'] == 37 and sel['MNMS']['n'] == 11 and sel['BUSI']['n'] == 11, 'counted runs after D-31: Prostate 55, Fundus 37, M&Ms 11, BUSI 11')
    O['selection'] = sel; O['n_logs'] = len(R)
    # ------------------------------------------------ the percentile gate at the three checkpoints (Section 4.2: "no alternative meets its criterion at either checkpoint")
    ref = lambda ds, s: f'v18B_{ds}_fixmatch_split1337_init1337' if s == '1337' else f'v18B_{ds}_fixmatch_s{s}'
    O['percentile_checkpoints'] = {}
    for ds in ('prostate', 'fundus'):
        for ck in ('best', 'last', 'swad'):
            x = s24.paired([R[(ds, f'v18F_{ds}_percentile_s{s}')][ck] for s in SEEDS], [R[(ds, ref(ds, s))][ck] for s in SEEDS])
            O['percentile_checkpoints'][f'{ds}_{ck}'] = {'delta_points': 100 * x['delta'], 'p': x['p'], 'per_seed_points': [100 * v for v in x['per_seed']]}
            need(not (x['p'] < 0.05 and x['delta'] >= 0.005), f'{ds} {ck}: the percentile gate does not meet the criterion of Section 4.4 against the threshold reference')
        need(abs(O['percentile_checkpoints'][f'{ds}_best']['delta_points'] - 100 * F29[f'{ds}_rule']['delta']) < 1e-9, f'{ds}: the best-checkpoint line equals ADJUDICATION_S29')
    # ------------------------------------------------ S9: the vacuity-confidence identity, reproduce S25, then add D-31
    old = list(csv.DictReader(open(os.path.join(D, 'table_signal_identity.csv'))))
    s25 = list(csv.DictReader(open(os.path.join(D, 'extra_s25_selection', 'table_signal_identity_s25.csv'))))
    s29 = list(csv.DictReader(open(os.path.join(D, 'extra_s29', 'table_signal_identity_s29.csv'))))
    C = F25['s9_counts']
    tb = [r for r in old if cls_run(r['run']) == 'testbed'] + s25
    need(len(old) == C['n_all_s12'] == 83 and len(tb) == C['n_testbed_objective_with_d30'] == 67, 'the identity tables reproduce the S25 counts (83 + 17 = 100 checkpoints, 67 testbed objective)')
    need(sum(r['identity_holds'] != '1' for r in tb) == C['n_testbed_identity_fails_with_d30'] == 1, 'the identity fails on 1 of the 67 testbed-objective checkpoints (S25)')
    per = {ds: sum(r['dataset'] == ds for r in tb) for ds in DS4}
    need(per == {'prostate': 38, 'fundus': 15, 'MNMS': 7, 'BUSI': 7}, f'per benchmark the 67 are 38 / 15 / 7 / 7 (Supplementary S9): {per}')
    need(len(s29) == 6 and all(cls_run(r['run']) == 'testbed' for r in s29), 'the six D-31 teachers are evidential-head teachers trained with the testbed objective')
    need(all(abs(float(r['median_ratio_u_epi_over_conf']) / float(r['expected_ratio_K_over_Km1']) - 1) < 0.01 for r in s29),
         'Supplementary S9 statement holds on the six D-31 teachers: the medians of u_epi and 1 - max p stand in the ratio K/(K-1) to within 1 %')
    tb2 = tb + s29; per2 = {ds: sum(r['dataset'] == ds for r in tb2) for ds in DS4}
    band = [v['u_epi_band'] for v in F29['signal'].values()]; lo, hi = C['testbed_band_min_max_with_d30']
    O['s9'] = {'n_all': C['n_all_with_d28_d29_d30'] + 6, 'n_testbed': len(tb2), 'n_testbed_identity_holds': sum(r['identity_holds'] == '1' for r in tb2), 'per_benchmark': per2,
               'testbed_band_min_max_s25': [lo, hi], 'testbed_band_min_max': [min(lo, min(band)), max(hi, max(band))], 'n_devis_medl': C['n_devis_medl_with_d28_d29'],
               'n_devis_medl_identity_fails': C['n_devis_medl_identity_fails_with_d28_d29']}
    need(O['s9']['n_all'] == 106 and O['s9']['n_testbed'] == 73 and O['s9']['n_testbed_identity_holds'] == 72, 'with D-31: 106 checkpoints, 73 with the testbed objective, the identity holds on 72')
    need(round(O['s9']['testbed_band_min_max'][0], 2) == 0.56 and round(O['s9']['testbed_band_min_max'][1], 2) == 0.73, 'the testbed-objective band range stays 0.56-0.73 with D-31 (its six teachers: 0.61-0.68)')
    # ------------------------------------------------ run count of the letter
    need(F29['runner']['jobs'] == 6 and F29['runner']['final_pass'] == 6, 'D-31 adds six trainings')
    O['runs_trained'] = {'before_d31': 101, 'd31': 6, 'total': 107}
    O['assertions_passed'] = PASSED
    json.dump(O, open(os.path.join(a.out, 'facts_s30_counts.json'), 'w'), indent=1, default=float)
    for ds in DS4:
        s = sel[ds]; print(f"[counts] {ds}: n {s['n_s25']} -> {s['n']}; median {s['median_s25']:.4f} -> {s['median']:.4f}; >1 {s['n_gt1_s25']} -> {s['n_gt1']}; >5 {s['n_gt5_s25']} -> {s['n_gt5']}")
    for k, v in O['percentile_checkpoints'].items():
        print(f"[percentile] {k}: delta {v['delta_points']:+.2f} p {v['p']:.3f} per seed {', '.join(f'{x:+.2f}' for x in v['per_seed_points'])}")
    print(f"[counts] S9: {O['s9']['n_all']} checkpoints; testbed {O['s9']['n_testbed']} ({per2}), identity holds {O['s9']['n_testbed_identity_holds']}; band {O['s9']['testbed_band_min_max']}")
    print(f'[make_extra_s30] {len(PASSED)} assertions PASSED -> facts_s30_counts.json')


if __name__ == '__main__':
    main()
