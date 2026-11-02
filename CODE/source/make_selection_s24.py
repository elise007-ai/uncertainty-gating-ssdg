#!/usr/bin/env python3
"""
make_selection_s24.py (S24 turn 3, 2026-10-11) -- the checkpoint-selection rule and what depends on it (E-69, N-59, N-60).
Assertion-first (R-48); nothing typed by hand (R-14).

  python make_selection_s24.py --returns <BSPC_results/RETURNS> --release <release_s12> --supp_pre <supplement.tex before the S24 layer> --out <derived_v04_S12/extra_s24_selection>

Every Dice value of the paper is the highest of the evaluations that the training loop makes on the test images every
500 iterations ("best checkpoint"). This reader measures, on every complete run whose training log is held:
  best      the reported value (max over the evaluations of the student's test-average Dice)
  last      the student at the last iteration
  swad      the SWAD average at the last iteration (an average of weights from T_swad on; no checkpoint is chosen)
and recomputes with `last` and `swad` (a) the comparison of the two gates where both logs are held, (b) every row of the
baseline table against the code-matched threshold-gate reference, with the registered criterion (paired t p < 0.05 and
delta >= 0.5 Dice).

Writes  table_selection_runs_s24.csv      one row per run
        table_selection_summary_s24.csv   best-minus-last and best-minus-swad per dataset
        table_selection_gates_s24.csv     DAEP minus threshold at the three checkpoints (M&Ms, BUSI: paired over seeds; Prostate: three trainings of each gate on split 1337)
        table_selection_baselines_s24.csv the rows of the baseline table at the three checkpoints
        facts_s24_selection.json          every number the text quotes; the assertions that passed
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, re, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_derived_v02 as v02          # noqa: E402
import make_derived_v04 as v04          # noqa: E402

SEEDS = (1337, 42, 2024)
CK = ('best', 'last', 'swad')
RULE_P, RULE_DELTA = 0.05, 0.005
PASSED = []


def need(cond, what):
    if not cond:
        raise AssertionError('ASSERTION FAILED: ' + what)
    PASSED.append(what)


def wcsv(path, rows, header):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def parse_all(txt):
    """Evaluations in order: student average (as logged), EMA average, and the SWAD block average when a SWAD block exists."""
    evals, cur, phase, pending = [], None, None, {}
    for line in txt.splitlines():
        if 'Testing student...' in line:
            cur, phase, pending = {'student': [], 'ema': [], 'swad': []}, 'student', {}
        elif 'Testing EMA teacher...' in line:
            phase, pending = 'ema', {}
        elif 'Testing SWAD' in line:
            phase, pending = 'swad', {}
        elif cur is not None and phase:
            m = v04.DOM_RX.search(line)
            if m:
                pending[int(m.group(1))] = ([float(x) for x in re.findall(r'dice=([\d.]+)', m.group(3))], [float(x) for x in re.findall(r'hd=([\d.]+)', m.group(4))])
            elif re.search(r'\] epoch \d+: avg ', line) or re.match(r'epoch \d+: avg ', line):
                if pending:
                    cur[phase].append(pending)
                pending = {}
        s = v04.SUM_RX.search(line)
        if s and cur is not None:
            evals.append({'epoch': int(s.group(1)), 'student_avg': float(s.group(2)), 'best_avg': float(s.group(4)),
                          'swad': v04.block_avg(cur['swad'][-1]) if cur['swad'] else None})
            cur, phase = None, None
    return evals


def load(returns):
    runs = {}
    for p in sorted(glob.glob(os.path.join(returns, '*', 'runs', '*', '*', 'log.txt'))):
        run = p.split(os.sep)[-2]; ds = p.split(os.sep)[-3]
        if 'smoke' in run:
            continue
        txt = open(p, errors='replace').read()
        if 'TRAINING COMPLETE' not in txt:
            continue
        ev = parse_all(txt); q = v02.parse_log(p); t = v04.screen_run(p)
        if not ev or q is None:
            continue
        sw = [e['swad'] for e in ev if e['swad'] is not None]
        n_eval = int(re.search(r'num_eval_iter=(\d+)', txt).group(1))
        r = {'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': ev[-1]['student_avg'], 'swad': sw[-1] if sw else None, 'max_evals': max(e['student_avg'] for e in ev),
             'n_evals': len(ev), 'n_eval_iter': n_eval, 'max_it': t['max_iterations'], 'flag': v04.flag_of(t), 'nf': t['nf_n'], 'ret': p.split(os.sep)[-5],
             'resumed': bool(t['resumed_iters'])}
        k = (ds, run)
        if k in runs:      # a run may sit in two returns (an early partial copy and the complete one): the complete copies must agree
            assert abs(runs[k]['best'] - r['best']) < 1e-9, k
        runs[k] = r
    return runs


def paired(a, b):
    a, b = np.array(a), np.array(b); d = a - b; t, p = stats.ttest_rel(a, b)
    return {'a_mean': float(a.mean()), 'a_std': float(a.std(ddof=1)), 'b_mean': float(b.mean()), 'b_std': float(b.std(ddof=1)), 'delta': float(d.mean()),
            'per_seed': [float(x) for x in d], 't': float(t), 'p': float(p), 'meets': bool(p < RULE_P and d.mean() >= RULE_DELTA)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--returns', required=True); ap.add_argument('--release', required=True)
    ap.add_argument('--supp_pre', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--not_yet_in_paper', default='v18D_', help='comma-separated run-name prefixes of runs that the manuscript does not describe yet; they are listed and not counted (dropped at the layer that writes them up, N-61)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = load(a.returns); F = {'session': 'S24 turn 3', 'reader': 'make_selection_s24.py', 'not_yet_in_paper': a.not_yet_in_paper}
    nyp = [x for x in a.not_yet_in_paper.split(',') if x]

    # ---------------------------------------------------------------- the rule, read off the logs
    for k, r in R.items():
        assert r['n_eval_iter'] == 500, k
        if not r['resumed']:
            assert r['n_evals'] == r['max_it'] // 500, (k, r['n_evals'])
        assert abs(r['best'] - r['max_evals']) < 6e-5, (k, r['best'], r['max_evals'])
    need(True, f'in each of the {len(R)} complete logs held, the reported Dice equals the maximum of the evaluations made on the test images every 500 iterations')
    F['n_logs'] = len(R)

    # ---------------------------------------------------------------- per run and per dataset
    rows, S = [], {}
    for (ds, run), r in sorted(R.items()):
        grp = ('not_yet_in_paper' if any(run.startswith(x) for x in nyp) else 'failure' if (r['flag'] != 'ok' or r['nf'] > 0) else 'short_ramp' if 'ramp10' in run else 'counted')
        rows.append([ds, run, r['ret'], grp, f'{r["best"]:.6f}', r['best_iter'], f'{r["last"]:.4f}', '' if r['swad'] is None else f'{r["swad"]:.4f}',
                     f'{100 * (r["best"] - r["last"]):.2f}', '' if r['swad'] is None else f'{100 * (r["best"] - r["swad"]):.2f}', r['flag'], r['nf']])
        r['grp'] = grp
    wcsv(os.path.join(a.out, 'table_selection_runs_s24.csv'), rows,
         ['dataset', 'run', 'return', 'group', 'best', 'best_iter', 'last', 'swad_final', 'best_minus_last_points', 'best_minus_swad_points', 'r34_flag', 'n_nonfinite_skips'])
    srow = []
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        c = {run: r for (d, run), r in R.items() if d == ds and r['grp'] == 'counted'}
        assert all(r['swad'] is not None for r in c.values()), ds
        bl = np.array([100 * (r['best'] - r['last']) for r in c.values()]); bs = np.array([100 * (r['best'] - r['swad']) for r in c.values()])
        S[ds] = {'n': len(c), 'n_all_complete': sum(d == ds for d, _ in R), 'n_failures': sum(d == ds and r['grp'] == 'failure' for (d, _), r in R.items()),
                 'n_short_ramp': sum(d == ds and r['grp'] == 'short_ramp' for (d, _), r in R.items()),
                 'n_not_yet_in_paper': sum(d == ds and r['grp'] == 'not_yet_in_paper' for (d, _), r in R.items()),
                 'bl_median': float(np.median(bl)), 'bl_max': float(bl.max()), 'bl_n_gt1': int((bl > 1).sum()), 'bl_n_gt5': int((bl > 5).sum()),
                 'bs_median': float(np.median(bs)), 'bs_max': float(bs.max()), 'bs_min': float(bs.min()),
                 'bl_gt5': {run: round(100 * (r['best'] - r['last']), 2) for run, r in sorted(c.items()) if 100 * (r['best'] - r['last']) > 5}}
        srow.append([ds, S[ds]['n_all_complete'], S[ds]['n_failures'], S[ds]['n_short_ramp'], S[ds]['n_not_yet_in_paper'], S[ds]['n'], f'{S[ds]["bl_median"]:.2f}', f'{S[ds]["bl_max"]:.2f}', S[ds]['bl_n_gt1'], S[ds]['bl_n_gt5'],
                     f'{S[ds]["bs_median"]:.2f}', f'{S[ds]["bs_min"]:.2f}', f'{S[ds]["bs_max"]:.2f}'])
    wcsv(os.path.join(a.out, 'table_selection_summary_s24.csv'), srow,
         ['dataset', 'n_complete_logs', 'n_training_failures', 'n_short_ramp', 'n_not_yet_in_paper', 'n_counted', 'best_minus_last_median', 'best_minus_last_max', 'n_above_1_point', 'n_above_5_points',
          'best_minus_swad_median', 'best_minus_swad_min', 'best_minus_swad_max'])
    F['size'] = S
    need(S['prostate']['n'] == 44 and S['fundus']['n'] == 25 and S['MNMS']['n'] == 11 and S['BUSI']['n'] == 11, 'counted runs (complete, no training failure, whole-run ramp, described in the manuscript): Prostate 44, Fundus 25, M&Ms 11, BUSI 11')
    need(abs(S['prostate']['bl_median'] - 1.2) < 0.05 and abs(S['fundus']['bl_median'] - 0.3) < 0.05 and abs(S['MNMS']['bl_median'] - 0.3) < 0.05 and abs(S['BUSI']['bl_median'] - 0.9) < 0.06,
         'median best-minus-last: 1.2 points on Prostate, 0.3 on Fundus and on M&Ms, 0.9 on BUSI (to one decimal)')
    need(S['prostate']['bl_n_gt5'] == 6 and S['fundus']['bl_n_gt5'] == 0, 'best-minus-last exceeds 5 points in 6 of the 44 Prostate runs and in no Fundus run')
    need('v18B_prostate_fixmatch_s2024' in S['prostate']['bl_gt5'] and abs(S['prostate']['bl_gt5']['v18B_prostate_fixmatch_s2024'] - 10.4) < 0.05,
         'one of the three threshold-gate reference runs on Prostate (seed 2024) ends 10.4 points below its best checkpoint')

    def ref(ds):
        return [f'v18B_{ds}_fixmatch_split1337_init1337' if (s == 1337 and ds in ('prostate', 'fundus')) else f'v18B_{ds}_fixmatch_s{s}' for s in SEEDS]

    def tri(ds, names, key):
        return [R[(ds, n)][key] for n in names]

    # ---------------------------------------------------------------- threshold-gate references: SWAD against best (N-60)
    F['threshold_refs'] = {}
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        b, l, s = (np.array(tri(ds, ref(ds), k)) for k in CK)
        F['threshold_refs'][ds] = {'best_mean': float(b.mean()), 'best_std': float(b.std(ddof=1)), 'last_mean': float(l.mean()), 'last_std': float(l.std(ddof=1)),
                                   'swad_mean': float(s.mean()), 'swad_std': float(s.std(ddof=1)), 'best_minus_swad': float(100 * (b.mean() - s.mean())),
                                   'best_minus_last': float(100 * (b.mean() - l.mean())), 'best_minus_last_per_seed': [float(100 * x) for x in (b - l)]}
    F['refs_last'] = {ds: [float(v) for v in tri(ds, ref(ds), 'last')] for ds in ('prostate', 'fundus', 'MNMS', 'BUSI')}
    bms = [F['threshold_refs'][ds]['best_minus_swad'] for ds in ('prostate', 'fundus', 'MNMS', 'BUSI')]
    F['threshold_refs']['best_minus_swad_min_max'] = [min(bms), max(bms)]
    need(0.55 <= min(bms) <= 0.65 and 2.4 <= max(bms) <= 2.5, 'threshold-gate runs, three seeds per benchmark: the final SWAD average is 0.6 to 2.5 points below the best-checkpoint mean')

    # ---------------------------------------------------------------- the two gates at the three checkpoints
    G, grow = {}, []
    for ds, dn in (('MNMS', 'mnms'), ('BUSI', 'busi')):
        d = [f'adaptive_cmp_{dn}_daep_s{s}' for s in SEEDS]; G[ds] = {}
        for k in CK:
            G[ds][k] = paired(tri(ds, d, k), tri(ds, ref(ds), k)); x = G[ds][k]
            grow.append([ds, 'paired over seeds 1337/42/2024', k, f'{100 * x["a_mean"]:.2f}', f'{100 * x["a_std"]:.2f}', f'{100 * x["b_mean"]:.2f}', f'{100 * x["b_std"]:.2f}',
                         f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}', ';'.join(f'{100 * v:+.2f}' for v in x['per_seed'])])
    dP = ['v18A_prostate_daep_default_s1337', 'v18A_prostate_daep_default_s1337_rep2', 'v18A_prostate_daep_default_s1337_rep3']
    tP = ['v18B_prostate_fixmatch_split1337_init1337', 'v18B_prostate_fixmatch_split1337_init43', 'v18B_prostate_fixmatch_split1337_init44']
    G['prostate_split1337'] = {}
    for k in CK:
        x, y = np.array(tri('prostate', dP, k)), np.array(tri('prostate', tP, k)); t, p = stats.ttest_ind(x, y, equal_var=False)
        G['prostate_split1337'][k] = {'a_mean': float(x.mean()), 'b_mean': float(y.mean()), 'delta': float(x.mean() - y.mean()), 'p': float(p), 'a': [float(v) for v in x], 'b': [float(v) for v in y]}
        grow.append(['prostate', 'three trainings of each gate, labeled split 1337 (Welch)', k, f'{100 * x.mean():.2f}', f'{100 * x.std(ddof=1):.2f}', f'{100 * y.mean():.2f}', f'{100 * y.std(ddof=1):.2f}',
                     f'{100 * (x.mean() - y.mean()):+.2f}', f'{p:.3f}', ''])
    wcsv(os.path.join(a.out, 'table_selection_gates_s24.csv'), grow,
         ['dataset', 'design', 'checkpoint', 'daep_mean', 'daep_std', 'threshold_mean', 'threshold_std', 'daep_minus_threshold', 'p', 'per_seed'])
    F['gates'] = G
    for ds in ('MNMS', 'BUSI', 'prostate_split1337'):
        need(all(abs(G[ds][k]['delta']) < 0.011 and G[ds][k]['p'] > 0.25 for k in CK),
             f'{ds}: DAEP minus threshold is below 1.1 points in absolute value with p > 0.25 at the best checkpoint, at the last iteration and with the final SWAD average')
    need(('fundus', 'v18A_fundus_daep_default_s1337') not in R and ('prostate', 'v18A_prostate_daep_default_s42') not in R,
         'no complete DAEP log is held for Fundus, nor for Prostate seeds 42 and 2024: the gate check does not cover them')

    # ---------------------------------------------------------------- the rows of the baseline table at the three checkpoints
    B, brow = {}, []
    names = {'softmax_fixmatch': 'softmax head, threshold gate (control)', 'mcdropout2': 'MC-dropout gate', 'dropoutctl2_fixmatch': 'dropout only (control)',
             'devisobj_threshold': 'DEviS objective, threshold gate', 'devis': 'DEviS objective, belief-mass gate', 'medl': 'MEDL (headline student)'}
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        for m in names:
            runs = [f'v18B_{ds}_{m}_s{s}' for s in SEEDS]
            if not all((ds, r) in R for r in runs):
                continue
            B[(ds, m)] = {}
            for k in CK:
                x = paired(tri(ds, runs, k), tri(ds, ref(ds), k)); B[(ds, m)][k] = x
                brow.append([ds, m, names[m], k, f'{100 * x["a_mean"]:.2f}', f'{100 * x["b_mean"]:.2f}', f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}', int(x['meets']),
                             ';'.join(f'{100 * v:+.2f}' for v in x['per_seed'])])
    wcsv(os.path.join(a.out, 'table_selection_baselines_s24.csv'), brow,
         ['dataset', 'method', 'row', 'checkpoint', 'mean', 'threshold_reference_mean', 'delta', 'p_paired', 'meets_criterion', 'per_seed_delta_1337_42_2024'])
    F['baselines'] = {f'{ds}:{m}': v for (ds, m), v in B.items()}
    base = {(r['dataset'], r['method']): r for r in csv.DictReader(open(os.path.join(a.returns, '..', 'derived', 'derived_v04_S12', 'table_baselines.csv')))}
    for (ds, m), v in B.items():
        if m == 'medl':
            continue          # the published MEDL value of the baseline table is the mean of its two networks (D-05); this reader takes the headline student of the log
        b = base[(ds, m)]
        assert abs(v['best']['delta'] - float(b['delta_vs_threshold'])) < 6e-5 and abs(v['best']['p'] - float(b['p_paired'])) < 6e-5, (ds, m)
    need(True, 'at the best checkpoint this reader reproduces delta and p of every row of table_baselines.csv except MEDL (whose table value is the two-network mean)')
    ua = ('mcdropout2', 'devisobj_threshold', 'devis', 'medl')
    need(all(not B[(ds, m)][k]['meets'] for ds in ('prostate', 'fundus') for m in ua for k in CK),
         'no uncertainty-aware alternative (MC-dropout gate, DEviS objective under either gate, MEDL) meets the criterion on Prostate or Fundus at any of the three checkpoints')
    sm = {ds: B[(ds, 'softmax_fixmatch')] for ds in ('prostate', 'fundus', 'MNMS', 'BUSI')}
    meets = sorted((ds, k) for ds in sm for k in CK if sm[ds][k]['meets'])
    F['softmax_meets'] = meets
    need(meets == [('prostate', 'swad')], 'the softmax-head control meets the numerical criterion in exactly one place: Prostate with the final SWAD average')
    x = sm['prostate']
    need(abs(100 * x['swad']['delta'] - 1.73) < 0.006 and abs(x['swad']['p'] - 0.028) < 0.0006 and abs(100 * x['best']['delta'] - 1.12) < 0.006 and abs(100 * x['last']['delta'] - 3.06) < 0.006,
         'softmax control on Prostate: +1.12 (best), +3.06 (last), +1.73 (SWAD; p = 0.028)')
    dv = B[('prostate', 'devisobj_threshold')], B[('fundus', 'devisobj_threshold')]
    need(all(d['swad']['delta'] < -0.07 for d in dv) and all(abs(d['last']['delta'] - d['best']['delta']) < 0.045 for d in dv),
         'DEviS objective (threshold gate): more than 7 points below the threshold gate with the SWAD average on both datasets')

    # ---------------------------------------------------------------- N-60: the SWAD sentence of the supplement as it stood
    sp = open(a.supp_pre).read()
    m = re.search(r'trails by only\s*\$0\.7\$--\$1\.4\$ Dice across the four benchmarks \(Prostate \$([\d.]+)\\pm([\d.]+)\$, Fundus\s*\$([\d.]+)\\pm([\d.]+)\$, M\\&Ms \$([\d.]+)\\pm([\d.]+)\$, BUSI \$([\d.]+)\\pm([\d.]+)\$\)', sp)
    assert m, 'the S7 SWAD sentence was not found in the supplement pre-text'
    printed = {'prostate': float(m.group(1)), 'fundus': float(m.group(3)), 'mnms': float(m.group(5)), 'busi': float(m.group(7))}
    t2 = {r['dataset']: float(r['best_avg_dice_3seed_mean']) for r in csv.DictReader(open(os.path.join(a.release, 'RESULTS', 'table2_main_results.csv'))) if r['variant'] == 'FF-DAEP'}
    d_sub = {k: t2[k] - printed[k] for k in printed}
    t3 = {'prostate': 85.38, 'fundus': 88.54, 'mnms': 80.85, 'busi': 70.83}          # Table 3 of the revised article, DAEP row (pinned in S17)
    d_now = {k: t3[k] - printed[k] for k in printed}
    F['n60'] = {'printed_swad': printed, 'submitted_table2_daep': t2, 'difference_to_submitted_table2': d_sub, 'difference_to_revised_table3': d_now}
    need(abs(min(d_sub.values()) - 0.73) < 0.006 and abs(max(d_sub.values()) - 1.44) < 0.006,
         'N-60: the printed "0.7-1.4" is the difference between the SWAD values of the fixed-split DAEP runs and the best-checkpoint values of the same runs (Table 2 of the submitted version: 84.28, 88.23, 81.53, 68.78)')
    need(max(d_now.values()) > 3.4 and min(d_now.values()) < 0.6,
         'N-60: against the seed-split values that the revised Table 3 prints, the same SWAD values trail by 0.5 to 3.5 points, so the sentence as printed cannot be checked by a reader and mixes two protocols')
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s24_selection.json'), 'w'), indent=1, default=float)
    print(f'[make_selection_s24] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


if __name__ == '__main__':
    main()
