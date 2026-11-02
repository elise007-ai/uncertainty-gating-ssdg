#!/usr/bin/env python3
"""
make_extra_s30_d32.py (S31, 2026-10-22) -- reader of the S30 return: D-32, six trainings without a pseudo-label gate (`v18G_<ds>_ungated_s<seed>`,
the percentile rule with keep fraction 1.0), assertion-first (R-48); nothing typed by hand (R-14). Rules: DOCS/D32_PREREGISTRATION_S30.md (2026-10-21,
before any run) and its Amendment 1 (rule 0 reads w_mean only; E-85). Helpers are imported from the D-28, D-30, D-31 and C-2 readers.
  python make_extra_s30_d32.py --returns <RETURNS> --derived <derived_v04_S12> --release <release_s12> --out <derived_v04_S12/extra_s30>
Writes table_d32_{integrity,per_seed,rule,checkpoints,signal}_s30.csv, table_{band_generality,band_summary,signal_identity}_s30.csv, facts_s30.json.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, shutil, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_derived_v02 as v02                      # noqa: E402
import make_derived_v04 as v04                      # noqa: E402
import make_selection_s24 as s24                    # noqa: E402
from make_extra_s23 import (need, sha, wcsv, f4, find_log, split_of, rule,   # noqa: E402
                            SEEDS, DATASETS, DOM, MAXIT, LBDOM, RULE_P, RULE_DELTA, RULE_BAND, RULE_RANGE, PASSED)
from make_selection_s25 import cmd_pairs, ref_name  # noqa: E402
from make_extra_s29 import tost, wfrac              # noqa: E402

WIN_RX = re.compile(r'\] iter (\d+): .*?w_mean=([\d.]+), w_min=([\d.]+)')
HASHES = ('a6fe4357a39d0020', '20d39982c157ff0e', '3e214990fca90428', 'dccfbdaaf6d9fc1b', '91d54698dca89a1b', 'eaac7fbba4ebe217')


def verdict(r):
    # rule 1, both directions (the criterion of Section 4.4): below / above iff paired p < 0.05 and |delta| >= 0.5 Dice
    return 'above the threshold gate' if (r['p'] < RULE_P and r['delta'] >= RULE_DELTA) else 'below the threshold gate' if (r['p'] < RULE_P and r['delta'] <= -RULE_DELTA) else 'within seed noise'


def main():
    ap = argparse.ArgumentParser()
    for k in ('returns', 'derived', 'release', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S30 = os.path.join(a.returns, 'S30'); S29 = os.path.join(a.returns, 'S29'); CL = os.path.join(S30, 'console_logs', 'S30'); src = os.path.join(a.release, 'CODE', 'source')
    F = {'session': 'S31', 'reader': 'make_extra_s30_d32.py', 'preregistration': 'DOCS/D32_PREREGISTRATION_S30.md (+ Amendment 1)'}
    code_sha = {n: sha(os.path.join(src, n)) for n in ('train_fourierfuse_v18_revision.py', 'revision_modules.py', 'fourierfuse_v16_modules.py', 'diag_uncertainty_harness_v18.py')}
    F['code_sha256'] = {k: v[:16] for k, v in code_sha.items()}
    F29 = json.load(open(os.path.join(a.derived, 'extra_s29', 'facts_s29.json'))); C30 = json.load(open(os.path.join(a.derived, 'extra_s29', 'facts_s30_counts.json')))
    # ------------------------------------------------ the sendback, the runner, the step files (V-25 closed on record)
    n_ok, listed = 0, []
    for mf in sorted(glob.glob(os.path.join(S30, '_manifests', 'p*', 'MANIFEST.txt'))):
        for line in open(mf, encoding='utf-8'):
            if line.strip():
                p, size, h = line.rstrip('\n').split('\t'); q = os.path.join(S30, *p.split('/')); listed.append(p)
                assert os.path.exists(q) and os.path.getsize(q) == int(size) and sha(q) == h, f'manifest mismatch {p}'; n_ok += 1
    F['manifest_files_verified'] = n_ok
    need(n_ok == 65 and len(set(listed)) == 65, 'every file of the S30 sendback (65, four parts) is in the results tier and matches its part manifest (size and sha256)')
    need(not any(re.search(r'v18[ABDEF]_', p) for p in listed), 'no file of an earlier training was sent again (the collector used HELD_MANIFEST_S30.txt)')
    need(sha(os.path.join(S30, 'jobs', 'jobs_D32.txt'))[:16] == HASHES[3] and sha(os.path.join(S30, 'jobs', 'jobs_D32smoke.txt'))[:16] == HASHES[4], 'the job lists that ran are the registered ones (tools v16)')
    pf = open(os.path.join(CL, 'preflight.txt'), errors='replace').read()
    need(all(h in pf for h in HASHES) and '1269 HELD_MANIFEST_S30.txt' in pf and 'No such file or directory' in pf and re.search(r'\n0\s*$', pf),
         'step 0 on record: the six hashes, the 1,269-line held manifest, no runner lock, no D-32 run before the launch')
    need('self-test PASSED' in open(os.path.join(CL, 'selftest_revision_modules.txt'), errors='replace').read(), 'step 1 on record: the server self-test passed and was written to a file')
    ev = os.path.join(a.release, 'DOCS', 'evidence', 'S30_smoke')
    for f, p in (('log.txt', 'runs/fundus/v18smoke_fundus_ungated/log.txt'), ('status_jobs_D32smoke.csv', 'console_logs/S30/status_jobs_D32smoke.csv')):
        need(open(os.path.join(S30, *p.split('/')), 'rb').read() == open(os.path.join(ev, f), 'rb').read(), f'the smoke file {f} equals the bytes filed in S30 turn 4')
    hist = {}
    for r in csv.DictReader(open(os.path.join(CL, 'status_jobs_D32.csv'))):
        hist.setdefault(r['job'], []).append(r)
    hrs = lambda s, e: round(float((np.datetime64(e.replace(' ', 'T')) - np.datetime64(s.replace(' ', 'T'))) / np.timedelta64(1, 'h')), 2)
    R = F['runner'] = {'jobs': len(hist), 'final_pass': sum(rs[-1]['postflight'] == 'PASS' for rs in hist.values()), 'first_start': min(rs[0]['start'] for rs in hist.values()),
                       'last_end': max(rs[-1]['end'] for rs in hist.values()), 'hours_per_job': {j: hrs(rs[-1]['start'], rs[-1]['end']) for j, rs in hist.items()}}
    R['wall_clock_hours'] = hrs(R['first_start'], R['last_end'])
    need(len(hist) == 6 and R['final_pass'] == 6 and all(len(rs) == 1 for rs in hist.values()), 'runner: six jobs, one attempt each, each PASS (no failure branch, no relaunch)')
    need(not os.path.isdir(os.path.join(S30, 'console_logs', 'S30_first_attempt')) and '[preflight] GREEN' in open(os.path.join(CL, 'runner_jobs_D32.log'), errors='replace').read(),
         'no first-attempt folder; the runner preflight is GREEN')
    # ------------------------------------------------ per run: integrity, rule 0 (every window keeps every pixel), screens (rule 4)
    jobs = {l.split('\t')[0]: l.split('\t')[2] for l in open(os.path.join(S30, 'jobs', 'jobs_D32.txt'), encoding='utf-8').read().splitlines() if l.strip()}
    RUN, rows_i, rows_p = {}, [], []
    for ds in DATASETS:
        dm = lambda pd: {DOM[ds][k - 1]: float(np.mean(list(v.values()))) for k, v in sorted(pd.items())}
        for sd in SEEDS:
            run = f'v18G_{ds}_ungated_s{sd}'; d = os.path.join(S30, 'runs', ds, run); log = os.path.join(d, 'log.txt')
            txt = open(log, errors='replace').read(); t = v04.screen_run(log); q = v02.parse_log(log)
            cfg = json.load(open(os.path.join(d, 'run_config.json'))); ar = cfg['args']
            sa = np.array([e['student_avg'] for e in v04.parse_evaluations(txt)])
            ok_cfg = (ar['method'] == 'percentile_gate' and ar['pg_keep_frac'] == 1.0 and ar['seed'] == sd and ar['max_iterations'] == MAXIT[ds]
                      and ar['lb_domain'] == LBDOM[ds] and ar['lb_num'] == 20 and ar['head'] == 'evidential' and ar['consistency_ramp_frac'] == -1.0 and ar['amp'] == 1)
            ok_sha = all(cfg[k] == code_sha[n] == sha(os.path.join(d, 'code', n)) for k, n in (('script_sha256', 'train_fourierfuse_v18_revision.py'),
                         ('revision_modules_sha256', 'revision_modules.py'), ('fourierfuse_v16_modules_sha256', 'fourierfuse_v16_modules.py')))
            cp, jp = cmd_pairs(txt), cmd_pairs('python train_fourierfuse_v18_revision.py ' + jobs[run]); cmd_diff = sorted(k for k in set(cp) | set(jp) if cp.get(k) != jp.get(k))
            c31 = json.load(open(os.path.join(S29, 'runs', ds, f'v18F_{ds}_percentile_s{sd}', 'run_config.json')))['args']
            arg_diff = sorted(k for k in set(ar) | set(c31) if ar.get(k) != c31.get(k))
            refname = ref_name(ds, sd); rlog, rq = find_log(a.returns, ds, refname); tr = v04.screen_run(rlog)
            sp, spr = split_of(log), split_of(rlog); sj = json.load(open(os.path.join(d, 'split.json')))
            J = json.load(open(os.path.join(S30, 'diag_out', f'{ds}__{run}__ema.json'))); me = J['meta']
            ok_h = (me['which'] == 'ema' and me['iter'] == q['best_iter'] and abs(me['best_avg_dice'] - q['best_avg']) < 1e-6 and run in me['ckpt']
                    and me['harness_sha256'] == code_sha['diag_uncertainty_harness_v18.py'] and me['n_models'] == 1)
            win = WIN_RX.findall(txt); w_ok = len(win) == MAXIT[ds] // 500 and all(m == '1.000' for _, m, _ in win)
            p31 = F29[f'{ds}_per_seed'][str(sd)]
            r = RUN[(ds, sd)] = {'run': run, 'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': t['student_avg_last'], 'last10': float(sa[int(len(sa) * 0.9):].mean()),
                                 'swad_best': q['swad_best'], 'ref': rq['best_avg'], 'refname': refname, 'ref_last': tr['student_avg_last'], 'flag': v04.flag_of(t), 'nf': t['nf_n'],
                                 'nf_frac': t['nf_n'] / MAXIT[ds], 'n_windows': len(win), 'pct': p31['best'], 'pct_last': p31['last'], 'at_best': dm(q['per_domain']),
                                 'ref_at_best': dm(rq['per_domain'])}
            need(t['complete'] and t['parse_ok'] and t['n_evals'] == MAXIT[ds] // 500 and t['n_headers'] == 1 and not t['resumed_iters'] and not cfg.get('resume_events'),
                 f'{run}: one training from iteration 0 to the last iteration, no resume event')
            need(ok_cfg and 'method=percentile_gate' in txt, f'{run}: configuration = percentile rule with keep fraction 1.0 (no gate), headline settings')
            need(set(cmd_diff) <= {'--gpu'}, f'{run}: the command in its log equals its line of jobs_D32.txt (the runner adds only the GPU index)')
            need(set(arg_diff) <= {'pg_keep_frac', 'save_name', 'gpu'} and 'pg_keep_frac' in arg_diff, f'{run}: recorded arguments = those of the D-31 run of the same seed except the keep fraction (and name, GPU)')
            need(ok_sha, f'{run}: code snapshot, recorded hashes and the release code are the same bytes')
            need(bool(sp[1]) and sp[0] == sd and sp[1] == spr[1] == sj['lb_idxs'], f'{run}: labeled slices identical to those of its threshold reference ({refname})')
            need(ok_h, f'{run}: the measured teacher is the EMA weights of the best checkpoint of this log, measured with the release harness')
            need(w_ok, f'{run}: rule 0 (Amendment 1): all {MAXIT[ds] // 500} logged windows show w_mean=1.000, so every pixel enters the pseudo-label term')
            rows_i.append([ds, run, sd, int(t['complete']), t['n_evals'], len(t['resumed_iters']), t['nf_n'], r['flag'], int(ok_cfg), ';'.join(cmd_diff), ';'.join(arg_diff),
                           int(ok_sha), int(ok_h), len(win), int(w_ok), cfg['torch'], cfg['hostname'], hist[run][-1]['start'], hist[run][-1]['end']])
            rows_p.append([ds, sd, run, f'{r["best"]:.6f}', r['best_iter'], f4(r['last']), f4(r['swad_best']), refname, f'{r["ref"]:.6f}', f4(r['ref_last']),
                           f'{r["best"] - r["ref"]:+.6f}', f'{r["pct"]:.6f}', f'{r["pct"] - r["best"]:+.6f}', r['flag'], r['nf']])
    wcsv(os.path.join(a.out, 'table_d32_integrity_s30.csv'), rows_i, ['dataset', 'run', 'seed', 'complete', 'n_evals', 'n_resumes', 'n_nonfinite_skips', 'r34_flag', 'config_ok',
         'cmd_keys_differing_from_job_line', 'arg_keys_differing_from_d31', 'hashes_ok', 'harness_meta_ok', 'n_gate_windows', 'every_window_w_mean_1', 'torch', 'hostname', 'start', 'end'])
    wcsv(os.path.join(a.out, 'table_d32_per_seed_s30.csv'), rows_p, ['dataset', 'seed', 'run', 'best_avg', 'best_iter', 'student_avg_last', 'swad_best', 'threshold_reference',
         'ref_best_avg', 'ref_last', 'ungated_minus_threshold', 'percentile_d31_best_avg', 'percentile_minus_ungated', 'r34_flag', 'n_nonfinite'])
    # ------------------------------------------------ rules 1-3
    rows_r = []
    for ds in DATASETS:
        ug = [RUN[(ds, s)]['best'] for s in SEEDS]; ref = [RUN[(ds, s)]['ref'] for s in SEEDS]; pc = [RUN[(ds, s)]['pct'] for s in SEEDS]
        r1 = rule(ug, ref); e1 = tost([100 * (x - y) for x, y in zip(ug, ref)]); r1['verdict'] = verdict(r1)
        r3 = rule(pc, ug); e3 = tost([100 * (x - y) for x, y in zip(pc, ug)]); r3['verdict'] = verdict(r3).replace('the threshold gate', 'the ungated runs')
        need(abs(e1['mean_points'] - 100 * r1['delta']) < 1e-9 and abs(e3['mean_points'] - 100 * r3['delta']) < 1e-9, f'{ds}: TOST and the paired tests use the same differences')
        need(abs(np.mean(ref) - F29[f'{ds}_rule']['ref_mean']) < 1e-12 and abs(np.mean(pc) - F29[f'{ds}_rule']['mean']) < 1e-12, f'{ds}: the references and the D-31 values are those of ADJUDICATION_S29')
        fl = [RUN[(ds, s)]['flag'] for s in SEEDS]
        F[f'{ds}_rule1'] = {**r1, **e1, 'r34_flags': fl, 'unstable': [RUN[(ds, s)]['nf_frac'] > 0.01 for s in SEEDS]}
        F[f'{ds}_rule3_percentile_minus_ungated'] = {**r3, **e3}
        F[f'{ds}_per_seed'] = {str(s): RUN[(ds, s)] for s in SEEDS}
        F[f'{ds}_descriptive'] = {'best_minus_last_points': [100 * (RUN[(ds, s)]['best'] - RUN[(ds, s)]['last']) for s in SEEDS],
                                  'ref_best_minus_last_points': [100 * (RUN[(ds, s)]['ref'] - RUN[(ds, s)]['ref_last']) for s in SEEDS],
                                  'best_frac': [RUN[(ds, s)]['best_iter'] / MAXIT[ds] for s in SEEDS]}
        for lab, r_, e_ in (('ungated_minus_threshold (rule 1)', r1, e1), ('percentile_minus_ungated (rule 3)', r3, e3)):
            rows_r.append([ds, lab, f4(r_['mean']), f4(r_['std']), f4(r_['ref_mean']), f4(r_['ref_std']), f'{100 * r_["delta"]:+.2f}', ';'.join(f'{100 * x:+.2f}' for x in r_['per_seed_delta']),
                           f'{r_["p"]:.4f}', r_['n_pos'], f'{r_["sign_p_two_sided"]:.4f}', f'{e_["ci90_points"][0]:+.2f};{e_["ci90_points"][1]:+.2f}', f'{e_["p_tost_0.5"]:.4f}',
                           f'{e_["mde_points"]:.2f}', r_['verdict']])
    wcsv(os.path.join(a.out, 'table_d32_rule_s30.csv'), rows_r, ['dataset', 'comparison', 'mean_dice', 'std_dice', 'ref_mean', 'ref_std', 'delta_points', 'per_seed_delta_points_1337_42_2024',
         'p_paired', 'n_seeds_positive', 'sign_test_p_two_sided', 'ci90_points', 'p_tost_0.5', 'mde_points', 'verdict'])
    F['expectation'] = {'registered': 'within seed noise of the threshold reference on both benchmarks',
                        'failed_on': [ds for ds in DATASETS if F[f'{ds}_rule1']['verdict'] != 'within seed noise']}
    F['expectation']['holds'] = not F['expectation']['failed_on']
    # ------------------------------------------------ the three checkpoints, and the counts of the text with D-32 added (R-66): every complete log, the S24 instrument
    L = s24.load(a.returns); grp = lambda run, r: 'failure' if (r['flag'] != 'ok' or r['nf'] > 0) else 'short_ramp' if 'ramp10' in run else 'counted'
    old = {k: r for k, r in L.items() if not k[1].startswith('v18G_')}; new = {k: r for k, r in L.items() if k[1].startswith('v18G_')}
    need(len(old) == C30['n_logs'] == 121 and len(new) == 6, 'complete logs: the 121 of facts_s30_counts.json plus the six of D-32 (smoke runs excluded)')
    rows_c = []; F['checkpoints'] = {}
    for ds in DATASETS:
        for ck in ('best', 'last', 'swad'):
            for lab, A_, B_ in (('ungated_minus_threshold', [L[(ds, f'v18G_{ds}_ungated_s{s}')][ck] for s in SEEDS], [L[(ds, ref_name(ds, s))][ck] for s in SEEDS]),
                                ('percentile_minus_ungated', [L[(ds, f'v18F_{ds}_percentile_s{s}')][ck] for s in SEEDS], [L[(ds, f'v18G_{ds}_ungated_s{s}')][ck] for s in SEEDS])):
                x = s24.paired(A_, B_); F['checkpoints'][f'{ds}_{lab}_{ck}'] = {'delta_points': 100 * x['delta'], 'p': x['p'], 'per_seed_points': [100 * v for v in x['per_seed']]}
                rows_c.append([ds, lab, ck, f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}', ';'.join(f'{100 * v:+.2f}' for v in x['per_seed'])])
        need(abs(F['checkpoints'][f'{ds}_ungated_minus_threshold_best']['delta_points'] - 100 * F[f'{ds}_rule1']['delta']) < 1e-9, f'{ds}: the best-checkpoint line equals rule 1')
    wcsv(os.path.join(a.out, 'table_d32_checkpoints_s30.csv'), rows_c, ['dataset', 'comparison', 'checkpoint', 'delta_points', 'p_paired', 'per_seed_points'])
    sel = {}
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        b0 = [100 * (r['best'] - r['last']) for (d, run), r in old.items() if d == ds and grp(run, r) == 'counted']; S = C30['selection'][ds]
        need(len(b0) == S['n'] and abs(float(np.median(b0)) - S['median']) < 1e-9 and sum(x > 5 for x in b0) == S['n_gt5'] and sum(x > 1 for x in b0) == S['n_gt1'],
             f'{ds}: the logs reproduce the S30 selection counts exactly (n {S["n"]}, median {S["median"]:.4f})')
        b1 = [100 * (r['best'] - r['last']) for (d, run), r in new.items() if d == ds and grp(run, r) == 'counted']; v = b0 + b1
        sel[ds] = {'n': len(v), 'median': float(np.median(v)), 'n_gt1': sum(x > 1 for x in v), 'n_gt5': sum(x > 5 for x in v), 'n_s30': len(b0), 'median_s30': S['median'], 'n_gt5_s30': S['n_gt5']}
    need(sel['prostate']['n'] == 58 and sel['fundus']['n'] == 40, 'counted runs after D-32: Prostate 58, Fundus 40')
    F['selection'] = sel; F['n_logs'] = len(L)
    # ------------------------------------------------ rule 5 (descriptive): the six teachers, v02 functions unchanged; the S9 identity counts with D-32
    H = {k: x for k, x in v02.load_harness(S30).items() if k[1].startswith('v18G_')}
    need(len(H) == 6 and all(k[2] == 'ema' for k in H), 'six harness JSONs, EMA teachers')
    tmp = tempfile.mkdtemp(); v02.band_tables(H, tmp); v02.signal_identity_table(H, tmp, {})
    for n in ('table_band_generality', 'table_band_summary', 'table_signal_identity'):
        shutil.copy(os.path.join(tmp, n + '.csv'), os.path.join(a.out, n + '_s30.csv'))
    bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_band_summary_s30.csv')))}
    idn = list(csv.DictReader(open(os.path.join(a.out, 'table_signal_identity_s30.csv'))))
    F['signal'] = {}; rows_g = []
    for ds in DATASETS:
        for sd in SEEDS:
            run = RUN[(ds, sd)]['run']; g = lambda sig, col: float(bs[(ds, run, sig)][col])
            s = {'u_epi_all': g('u_epi', 'auroc_all_mean'), 'u_epi_band': g('u_epi', 'auroc_edt5_mean'), 'u_epi_range80': g('u_epi', 'range80_all_mean'),
                 'conf_all': g('conf', 'auroc_all_mean'), 'conf_band': g('conf', 'auroc_edt5_mean'),
                 'identity_holds': [r for r in idn if r['dataset'] == ds and r['run'] == run][0]['identity_holds'] == '1'}
            s['phaseA_rule_collapse_persists'] = bool(s['u_epi_band'] <= RULE_BAND and s['u_epi_range80'] <= RULE_RANGE); F['signal'][f'{ds}_s{sd}'] = s
            rows_g.append([ds, sd, run] + [f'{x:.4g}' if isinstance(x, float) else int(x) for x in s.values()])
    wcsv(os.path.join(a.out, 'table_d32_signal_s30.csv'), rows_g, ['dataset', 'seed', 'run'] + list(s.keys()))
    I = C30['s9']; band = [v['u_epi_band'] for v in F['signal'].values()]
    F['s9'] = {'n_all': I['n_all'] + 6, 'n_testbed': I['n_testbed'] + 6, 'n_testbed_identity_holds': I['n_testbed_identity_holds'] + sum(v['identity_holds'] for v in F['signal'].values()),
               'per_benchmark': {k: v + (3 if k in ('prostate', 'fundus') else 0) for k, v in I['per_benchmark'].items()},
               'testbed_band_min_max': [min(I['testbed_band_min_max'][0], min(band)), max(I['testbed_band_min_max'][1], max(band))]}
    F['runs_trained'] = {'before_d32': C30['runs_trained']['total'], 'd32': 6, 'total': C30['runs_trained']['total'] + 6}
    # descriptive (no rule): how much of the pseudo-label term each gate keeps, in the first fifth of training and after it (the logged w_mean:
    # the mask mean for the threshold gate, line 1543; the weight mean for DAEP, line 1464; code-matched DAEP trainings, as in S7)
    F['gate_density'] = {}
    for ds in DATASETS:
        for sd in SEEDS:
            Tm = MAXIT[ds]
            for lab, run in (('threshold', ref_name(ds, sd)), ('daep', f'v18A_{ds}_daep_default_s{sd}')):
                lg, _ = find_log(a.returns, ds, run); w = [(int(i), float(m)) for i, m, _ in WIN_RX.findall(open(lg, errors='replace').read())]
                need(len(w) == Tm // 500, f'{run}: w_mean logged at every evaluation interval')
                e = [m for i, m in w if i <= 0.2 * Tm]; l = [m for i, m in w if i > 0.2 * Tm]
                F['gate_density'][f'{ds}_s{sd}_{lab}'] = {'run': run, 'first_window': w[0][1], 'first_fifth_mean': float(np.mean(e)), 'first_fifth_min': float(min(e)),
                                                         'after_fifth_mean': float(np.mean(l)), 'after_fifth_min': float(min(l))}
    # N-71 (S31, review M1): the coverage of the percentile gate (D-31 logs) per phase, next to the threshold gate and DAEP above; no gate keeps 1.000 (rule 0)
    for ds in DATASETS:
        for sd in SEEDS:
            run = f'v18F_{ds}_percentile_s{sd}'; Tm = MAXIT[ds]
            w = [(int(i), float(m)) for i, m, _ in WIN_RX.findall(open(os.path.join(S29, 'runs', ds, run, 'log.txt'), errors='replace').read())]
            need(len(w) == Tm // 500, f'{run}: w_mean logged at every evaluation interval')
            e = [m for i, m in w if i <= 0.2 * Tm]; l = [m for i, m in w if i > 0.2 * Tm]
            F['gate_density'][f'{ds}_s{sd}_percentile'] = {'run': run, 'first_window': w[0][1], 'first_fifth_mean': float(np.mean(e)), 'first_fifth_min': float(min(e)),
                                                            'after_fifth_mean': float(np.mean(l)), 'after_fifth_min': float(min(l))}
    G = F['gate_density']
    F['coverage_summary'] = {lab: {k: [min(G[f'{d}_s{s}_{lab}'][k] for d in DATASETS for s in SEEDS), max(G[f'{d}_s{s}_{lab}'][k] for d in DATASETS for s in SEEDS)]
                                   for k in ('first_window', 'first_fifth_mean', 'after_fifth_mean')} for lab in ('threshold', 'daep', 'percentile')}
    F['coverage_summary']['max_removed_after_fifth_threshold_daep'] = max(1 - G[k]['after_fifth_mean'] for k in G if k.endswith(('_threshold', '_daep')))
    need(F['coverage_summary']['max_removed_after_fifth_threshold_daep'] < 0.015, 'after the first fifth neither the threshold gate nor DAEP removes more than 1.5 % of the pseudo-label term')
    # N-72 (S31, review 4b): training without a gate against the code-matched DAEP trainings (D-30 + S06), paired by seed, at the three checkpoints
    F['nogate_vs_daep'] = {}
    for ds in DATASETS:
        for ck in ('best', 'last', 'swad'):
            x = s24.paired([L[(ds, f'v18G_{ds}_ungated_s{s}')][ck] for s in SEEDS], [L[(ds, f'v18A_{ds}_daep_default_s{s}')][ck] for s in SEEDS])
            F['nogate_vs_daep'][f'{ds}_{ck}'] = {'delta_points': 100 * x['delta'], 'p': x['p'], 'per_seed_points': [100 * v for v in x['per_seed']]}
    need(all(v['p'] >= 0.05 for v in F['nogate_vs_daep'].values()), 'no gate is within seed noise of the code-matched DAEP trainings at the three checkpoints on both benchmarks')
    for k, v in F['gate_density'].items():
        print(f"[density] {k}: first window {v['first_window']:.3f}; first fifth mean {v['first_fifth_mean']:.4f} (min {v['first_fifth_min']:.3f}); after mean {v['after_fifth_mean']:.4f} (min {v['after_fifth_min']:.3f})")
    print('[coverage]', json.dumps(F['coverage_summary']))
    for k, v in F['nogate_vs_daep'].items():
        print(f"[nogate-daep] {k}: {v['delta_points']:+.2f} p {v['p']:.3f} per seed {', '.join(f'{x:+.2f}' for x in v['per_seed_points'])}")
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s30.json'), 'w'), indent=1, default=float)
    for ds in DATASETS:
        r, e = F[f'{ds}_rule1'], F[f'{ds}_rule3_percentile_minus_ungated']
        print(f"[d32] {ds}: ungated {100 * r['mean']:.2f}+-{100 * r['std']:.2f} vs threshold {100 * r['ref_mean']:.2f}+-{100 * r['ref_std']:.2f}; delta {100 * r['delta']:+.2f} "
              f"({', '.join(f'{100 * x:+.2f}' for x in r['per_seed_delta'])}); p {r['p']:.3f}; CI90 [{r['ci90_points'][0]:+.2f}, {r['ci90_points'][1]:+.2f}]; TOST {r['p_tost_0.5']:.3f}; "
              f"MDE {r['mde_points']:.2f} -> {r['verdict']}")
        print(f"[d32] {ds}: percentile - ungated {100 * e['delta']:+.2f} ({', '.join(f'{100 * x:+.2f}' for x in e['per_seed_delta'])}); p {e['p']:.3f}; TOST {e['p_tost_0.5']:.3f} -> {e['verdict']}")
    for k, v in F['checkpoints'].items():
        print(f"[ckpt] {k}: {v['delta_points']:+.2f} p {v['p']:.3f}")
    for ds, s in sel.items():
        print(f"[counts] {ds}: n {s['n_s30']} -> {s['n']}; median {s['median_s30']:.2f} -> {s['median']:.2f}; >5 {s['n_gt5_s30']} -> {s['n_gt5']}")
    print(f"[counts] S9: {F['s9']}; runs {F['runs_trained']}")
    print(f'[make_extra_s30_d32] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


if __name__ == '__main__':
    main()
