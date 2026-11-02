#!/usr/bin/env python3
"""
make_extra_s31_d33.py (S32, 2026-10-23) -- reader of the S31 return: D-33, six trainings with the oracle gate (`v18H_<ds>_oracle_s<seed>`, trainer v12),
a positive control, assertion-first (R-48); nothing typed by hand (R-14). Rules: DOCS/D33_PREREGISTRATION_S31.md (2026-10-22, before any run) and its
Amendment 1. Helpers are imported from the D-28, D-30, D-31, D-32 and C-2 readers, so every reader applies the same functions.
  python make_extra_s31_d33.py --returns <RETURNS> --derived <derived_v04_S12> --release <release_s12> --out <derived_v04_S12/extra_s31>
Writes table_d33_{integrity,per_seed,rule,checkpoints,coverage,signal}_s31.csv, table_{band_generality,band_summary,signal_identity}_s31.csv, facts_s31.json.
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
from make_extra_s29 import tost                     # noqa: E402
from make_extra_s30_d32 import verdict, WIN_RX      # noqa: E402

HASHES = ('20d39982c157ff0e', 'a6fe4357a39d0020', '915347736f4855da', '3e214990fca90428', '06951e18b2178b28', 'd4419572311242c9', '787abffdca2601c9')
BOX_MEAN = 0.21          # the mean area of the pasted box, U(0.02, 0.4) of the image, drawn at every step (asserted in the S31b layer)


def main():
    ap = argparse.ArgumentParser()
    for k in ('returns', 'derived', 'release', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S31 = os.path.join(a.returns, 'S31'); S30 = os.path.join(a.returns, 'S30'); CL = os.path.join(S31, 'console_logs', 'S31'); src = os.path.join(a.release, 'CODE', 'source')
    F = {'session': 'S32', 'reader': 'make_extra_s31_d33.py', 'preregistration': 'DOCS/D33_PREREGISTRATION_S31.md (+ Amendment 1)'}
    code_sha = {'train_fourierfuse_v18_revision.py': sha(os.path.join(src, 'v12', 'train_fourierfuse_v18_revision.py'))}
    for n in ('revision_modules.py', 'fourierfuse_v16_modules.py', 'diag_uncertainty_harness_v18.py'):
        code_sha[n] = sha(os.path.join(src, n))
    F['code_sha256'] = {k: v[:16] for k, v in code_sha.items()}
    need(F['code_sha256']['train_fourierfuse_v18_revision.py'] == HASHES[2], 'the release copy of trainer v12 is 915347736f4855da')
    F30 = json.load(open(os.path.join(a.derived, 'extra_s30', 'facts_s30.json')))
    # ------------------------------------------------ the sendback, the runner, the four step files
    n_ok, listed = 0, []
    for mf in sorted(glob.glob(os.path.join(S31, '_manifests', 'p*', 'MANIFEST.txt'))):
        for line in open(mf, encoding='utf-8'):
            if line.strip():
                p, size, h = line.rstrip('\n').split('\t'); q = os.path.join(S31, *p.split('/')); listed.append(p)
                assert os.path.exists(q) and os.path.getsize(q) == int(size) and sha(q) == h, f'manifest mismatch {p}'; n_ok += 1
    F['manifest_files_verified'] = n_ok
    need(n_ok == 67 and len(set(listed)) == 67, 'every file of the S31 sendback (67, four parts) is in the results tier and matches its part manifest (size and sha256)')
    need(not any(re.search(r'v18[ABDEFG]_', p) for p in listed), 'no file of an earlier training was sent again (the collector used HELD_MANIFEST_S31.txt)')
    need(sha(os.path.join(S31, 'jobs', 'jobs_D33.txt'))[:16] == HASHES[4] and sha(os.path.join(S31, 'jobs', 'jobs_D33smoke.txt'))[:16] == HASHES[5], 'the job lists that ran are the registered ones (tools v17)')
    pf = open(os.path.join(CL, 'preflight.txt'), errors='replace').read()
    need(all(h in pf for h in HASHES) and '1334 HELD_MANIFEST_S31.txt' in pf and 'No such file or directory' in pf and re.search(r'\n0\s*$', pf),
         'step 0 on record: the seven hashes (backed-up trainer v11, trainer v12), the 1,334-line held manifest, no runner lock, no D-33 run before the launch')
    need('self-test PASSED' in open(os.path.join(CL, 'selftest_revision_modules.txt'), errors='replace').read(), 'step 1: the server self-test passed and was written to a file')
    need('COMPILE OK v12' in open(os.path.join(CL, 'compile_trainer_v12.txt'), errors='replace').read(), 'step 1: trainer v12 compiled on the server')
    cen = [l for l in open(os.path.join(CL, 'mask_census.txt'), errors='replace').read().splitlines() if ' values ' in l]
    enc = {ds: {l.split(' values ')[1].strip() for l in cen if key in l} for ds, key in (('prostate', 'ProstateSlice'), ('fundus', 'Fundus'))}
    need(len([l for l in cen if 'ProstateSlice' in l]) == 6 and len([l for l in cen if 'Fundus' in l]) >= 4 and enc == {'prostate': {'[0, 255]'}, 'fundus': {'[0, 128, 255]'}},
         'Amendment 1 rule 0: the mask census shows one encoding per benchmark (Prostate {0, 255} in all six domains; Fundus {0, 128, 255} in every folder)')
    F['mask_census'] = {k: sorted(v) for k, v in enc.items()}
    hist = {}
    for r in csv.DictReader(open(os.path.join(CL, 'status_jobs_D33.csv'))):
        hist.setdefault(r['job'], []).append(r)
    hrs = lambda s, e: round(float((np.datetime64(e.replace(' ', 'T')) - np.datetime64(s.replace(' ', 'T'))) / np.timedelta64(1, 'h')), 2)
    R = F['runner'] = {'jobs': len(hist), 'final_pass': sum(rs[-1]['postflight'] == 'PASS' for rs in hist.values()), 'first_start': min(rs[0]['start'] for rs in hist.values()),
                       'last_end': max(rs[-1]['end'] for rs in hist.values())}
    R['wall_clock_hours'] = hrs(R['first_start'], R['last_end'])
    need(len(hist) == 6 and R['final_pass'] == 6 and all(len(rs) == 1 for rs in hist.values()), 'runner: six jobs, one attempt each, each PASS')
    need(not os.path.isdir(os.path.join(S31, 'console_logs', 'S31_first_attempt')) and '[preflight] GREEN' in open(os.path.join(CL, 'runner_jobs_D33.log'), errors='replace').read(),
         'no first-attempt folder; the runner preflight is GREEN')
    sm = open(os.path.join(S31, 'runs', 'fundus', 'v18smoke_fundus_oracle', 'log.txt'), errors='replace').read(); sw = WIN_RX.findall(sm)
    need('method=oracle_gate' in sm and len(sw) == 2 and not any(m == '1.000' for _, m, _ in sw), 'the smoke run: oracle gate, two logged windows, neither keeps every pixel')
    # ------------------------------------------------ per run: integrity and rule 0 (Amendment 1)
    jobs = {l.split('\t')[0]: l.split('\t')[2] for l in open(os.path.join(S31, 'jobs', 'jobs_D33.txt'), encoding='utf-8').read().splitlines() if l.strip()}
    RUN, rows_i, rows_p, rows_v = {}, [], [], []
    for ds in DATASETS:
        dm = lambda pd: {DOM[ds][k - 1]: float(np.mean(list(v.values()))) for k, v in sorted(pd.items())}
        Tm = MAXIT[ds]
        for sd in SEEDS:
            run = f'v18H_{ds}_oracle_s{sd}'; d = os.path.join(S31, 'runs', ds, run); log = os.path.join(d, 'log.txt')
            txt = open(log, errors='replace').read(); t = v04.screen_run(log); q = v02.parse_log(log)
            cfg = json.load(open(os.path.join(d, 'run_config.json'))); ar = cfg['args']
            ok_cfg = (ar['method'] == 'oracle_gate' and ar['seed'] == sd and ar['max_iterations'] == Tm and ar['lb_domain'] == LBDOM[ds] and ar['lb_num'] == 20
                      and ar['head'] == 'evidential' and ar['consistency_ramp_frac'] == -1.0 and ar['amp'] == 1)
            ok_sha = all(cfg[k] == code_sha[n] == sha(os.path.join(d, 'code', n)) for k, n in (('script_sha256', 'train_fourierfuse_v18_revision.py'),
                         ('revision_modules_sha256', 'revision_modules.py'), ('fourierfuse_v16_modules_sha256', 'fourierfuse_v16_modules.py')))
            cp, jp = cmd_pairs(txt), cmd_pairs('python train_fourierfuse_v18_revision.py ' + jobs[run]); cmd_diff = sorted(k for k in set(cp) | set(jp) if cp.get(k) != jp.get(k))
            c32 = json.load(open(os.path.join(S30, 'runs', ds, f'v18G_{ds}_ungated_s{sd}', 'run_config.json')))['args']
            arg_diff = sorted(k for k in set(ar) | set(c32) if ar.get(k) != c32.get(k))
            refname = ref_name(ds, sd); rlog, rq = find_log(a.returns, ds, refname); tr = v04.screen_run(rlog)
            sp, spr = split_of(log), split_of(rlog); sj = json.load(open(os.path.join(d, 'split.json')))
            J = json.load(open(os.path.join(S31, 'diag_out', f'{ds}__{run}__ema.json'))); me = J['meta']
            ok_h = (me['which'] == 'ema' and me['iter'] == q['best_iter'] and abs(me['best_avg_dice'] - q['best_avg']) < 1e-6 and run in me['ckpt']
                    and me['harness_sha256'] == code_sha['diag_uncertainty_harness_v18.py'] and me['n_models'] == 1)
            win = [(int(i), float(m)) for i, m, _ in WIN_RX.findall(txt)]
            e = [m for i, m in win if i <= 0.2 * Tm]; l = [m for i, m in win if i > 0.2 * Tm]
            g32 = F30[f'{ds}_per_seed'][str(sd)]
            r = RUN[(ds, sd)] = {'run': run, 'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': t['student_avg_last'], 'ref': rq['best_avg'], 'refname': refname,
                                 'ref_last': tr['student_avg_last'], 'nogate': g32['best'], 'flag': v04.flag_of(t), 'nf': t['nf_n'], 'nf_frac': t['nf_n'] / Tm,
                                 'at_best': dm(q['per_domain']), 'ref_at_best': dm(rq['per_domain']), 'w_first': win[0][1], 'w_first_fifth_mean': float(np.mean(e)),
                                 'w_after_mean': float(np.mean(l)), 'w_after_median': float(np.median(l)), 'w_after_min': float(min(l)), 'w_after_max': float(max(l))}
            need(t['complete'] and t['parse_ok'] and t['n_evals'] == Tm // 500 and t['n_headers'] == 1 and not t['resumed_iters'] and not cfg.get('resume_events'),
                 f'{run}: one training from iteration 0 to the last iteration, no resume event')
            need(ok_cfg and 'method=oracle_gate' in txt, f'{run}: configuration = the oracle gate, headline settings')
            need(set(cmd_diff) <= {'--gpu'}, f'{run}: the command in its log equals its line of jobs_D33.txt (the runner adds only the GPU index)')
            need(set(arg_diff) <= {'method', 'pg_keep_frac', 'save_name', 'gpu'} and 'method' in arg_diff, f'{run}: recorded arguments = those of the D-32 run of the same seed except the method (and the dropped keep fraction, name, GPU)')
            need(ok_sha, f'{run}: Amendment 1 rule 0: script_sha256 = trainer v12; code snapshot, recorded hashes and the release code are the same bytes')
            need(bool(sp[1]) and sp[0] == sd and sp[1] == spr[1] == sj['lb_idxs'], f'{run}: Amendment 1 rule 0: labeled slices identical to those of its threshold reference ({refname})')
            need(ok_h, f'{run}: the measured teacher is the EMA weights of the best checkpoint of this log, measured with the release harness')
            need(len(win) == Tm // 500 and any(m < 1.0 for _, m in win), f'{run}: rule 0: the oracle removes pseudo-labels (at least one logged window below 1.000)')
            need(r['w_after_median'] >= 0.90, f'{run}: Amendment 1 rule 0: median w_mean after the first fifth >= 0.90 (no suspected mask fault)')
            rows_i.append([ds, run, sd, int(t['complete']), t['n_evals'], len(t['resumed_iters']), t['nf_n'], r['flag'], int(ok_cfg), ';'.join(cmd_diff), ';'.join(arg_diff),
                           int(ok_sha), int(ok_h), len(win), cfg['torch'], cfg['hostname'], hist[run][-1]['start'], hist[run][-1]['end']])
            rows_p.append([ds, sd, run, f'{r["best"]:.6f}', r['best_iter'], f4(r['last']), refname, f'{r["ref"]:.6f}', f4(r['ref_last']), f'{r["best"] - r["ref"]:+.6f}',
                           f'{r["nogate"]:.6f}', f'{r["best"] - r["nogate"]:+.6f}', r['flag'], r['nf']])
            rows_v.append([ds, sd, run, f'{r["w_first"]:.3f}', f'{r["w_first_fifth_mean"]:.4f}', f'{r["w_after_mean"]:.4f}', f'{r["w_after_median"]:.4f}', f'{r["w_after_min"]:.3f}',
                           f'{(1 - r["w_after_mean"]) / (1 - BOX_MEAN):.4f}'])
    wcsv(os.path.join(a.out, 'table_d33_integrity_s31.csv'), rows_i, ['dataset', 'run', 'seed', 'complete', 'n_evals', 'n_resumes', 'n_nonfinite_skips', 'r34_flag', 'config_ok',
         'cmd_keys_differing_from_job_line', 'arg_keys_differing_from_d32', 'hashes_ok', 'harness_meta_ok', 'n_gate_windows', 'torch', 'hostname', 'start', 'end'])
    wcsv(os.path.join(a.out, 'table_d33_per_seed_s31.csv'), rows_p, ['dataset', 'seed', 'run', 'best_avg', 'best_iter', 'student_avg_last', 'threshold_reference', 'ref_best_avg', 'ref_last',
         'oracle_minus_threshold', 'nogate_d32_best_avg', 'oracle_minus_nogate', 'r34_flag', 'n_nonfinite'])
    wcsv(os.path.join(a.out, 'table_d33_coverage_s31.csv'), rows_v, ['dataset', 'seed', 'run', 'w_mean_first_window', 'w_mean_first_fifth', 'w_mean_after_fifth', 'w_mean_after_fifth_median',
         'w_mean_after_fifth_min', 'removed_share_outside_box_approx'])
    # ------------------------------------------------ rules 1-2
    rows_r = []
    for ds in DATASETS:
        orc = [RUN[(ds, s)]['best'] for s in SEEDS]; ref = [RUN[(ds, s)]['ref'] for s in SEEDS]; ng = [RUN[(ds, s)]['nogate'] for s in SEEDS]
        r1 = rule(orc, ref); e1 = tost([100 * (x - y) for x, y in zip(orc, ref)]); r1['verdict'] = verdict(r1)
        r2 = rule(orc, ng); e2 = tost([100 * (x - y) for x, y in zip(orc, ng)]); r2['verdict'] = verdict(r2).replace('the threshold gate', 'the no-gate runs')
        need(abs(e1['mean_points'] - 100 * r1['delta']) < 1e-9 and abs(e2['mean_points'] - 100 * r2['delta']) < 1e-9, f'{ds}: TOST and the paired tests use the same differences')
        need(abs(np.mean(ref) - F30[f'{ds}_rule1']['ref_mean']) < 1e-12 and abs(np.mean(ng) - F30[f'{ds}_rule1']['mean']) < 1e-12, f'{ds}: the references and the D-32 values are those of ADJUDICATION_S30')
        F[f'{ds}_rule1'] = {**r1, **e1, 'r34_flags': [RUN[(ds, s)]['flag'] for s in SEEDS], 'unstable': [RUN[(ds, s)]['nf_frac'] > 0.01 for s in SEEDS]}
        F[f'{ds}_rule2_oracle_minus_nogate'] = {**r2, **e2}
        F[f'{ds}_per_seed'] = {str(s): RUN[(ds, s)] for s in SEEDS}
        F[f'{ds}_descriptive'] = {'best_minus_last_points': [100 * (RUN[(ds, s)]['best'] - RUN[(ds, s)]['last']) for s in SEEDS],
                                  'ref_best_minus_last_points': [100 * (RUN[(ds, s)]['ref'] - RUN[(ds, s)]['ref_last']) for s in SEEDS],
                                  'best_frac': [RUN[(ds, s)]['best_iter'] / MAXIT[ds] for s in SEEDS],
                                  'per_vendor_delta_points': {k: [100 * (RUN[(ds, s)]['at_best'][k] - RUN[(ds, s)]['ref_at_best'][k]) for s in SEEDS] for k in RUN[(ds, SEEDS[0])]['at_best']}}
        for lab, r_, e_ in (('oracle_minus_threshold (rule 1)', r1, e1), ('oracle_minus_nogate (rule 2)', r2, e2)):
            rows_r.append([ds, lab, f4(r_['mean']), f4(r_['std']), f4(r_['ref_mean']), f4(r_['ref_std']), f'{100 * r_["delta"]:+.2f}', ';'.join(f'{100 * x:+.2f}' for x in r_['per_seed_delta']),
                           f'{r_["p"]:.4f}', r_['n_pos'], f'{r_["sign_p_two_sided"]:.4f}', f'{e_["ci90_points"][0]:+.2f};{e_["ci90_points"][1]:+.2f}', f'{e_["p_tost_0.5"]:.4f}',
                           f'{e_["mde_points"]:.2f}', r_['verdict']])
    wcsv(os.path.join(a.out, 'table_d33_rule_s31.csv'), rows_r, ['dataset', 'comparison', 'mean_dice', 'std_dice', 'ref_mean', 'ref_std', 'delta_points', 'per_seed_delta_points_1337_42_2024',
         'p_paired', 'n_seeds_positive', 'sign_test_p_two_sided', 'ci90_points', 'p_tost_0.5', 'mde_points', 'verdict'])
    F['expectation'] = {'registered': 'the oracle is not below the threshold gate on either benchmark',
                        'failed_on': [ds for ds in DATASETS if F[f'{ds}_rule1']['verdict'] == 'below the threshold gate']}
    F['expectation']['holds'] = not F['expectation']['failed_on']
    # ------------------------------------------------ the three checkpoints, and the counts of the text with D-33 added (R-66)
    L = s24.load(a.returns); grp = lambda run, r: 'failure' if (r['flag'] != 'ok' or r['nf'] > 0) else 'short_ramp' if 'ramp10' in run else 'counted'
    old = {k: r for k, r in L.items() if not k[1].startswith('v18H_')}; new = {k: r for k, r in L.items() if k[1].startswith('v18H_')}
    need(len(old) == F30['n_logs'] and len(new) == 6, f'complete logs: the {F30["n_logs"]} of facts_s30.json plus the six of D-33 (smoke runs excluded)')
    rows_c = []; F['checkpoints'] = {}
    for ds in DATASETS:
        for ck in ('best', 'last', 'swad'):
            for lab, A_, B_ in (('oracle_minus_threshold', [L[(ds, f'v18H_{ds}_oracle_s{s}')][ck] for s in SEEDS], [L[(ds, ref_name(ds, s))][ck] for s in SEEDS]),
                                ('oracle_minus_nogate', [L[(ds, f'v18H_{ds}_oracle_s{s}')][ck] for s in SEEDS], [L[(ds, f'v18G_{ds}_ungated_s{s}')][ck] for s in SEEDS])):
                x = s24.paired(A_, B_); F['checkpoints'][f'{ds}_{lab}_{ck}'] = {'delta_points': 100 * x['delta'], 'p': x['p'], 'per_seed_points': [100 * v for v in x['per_seed']]}
                rows_c.append([ds, lab, ck, f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}', ';'.join(f'{100 * v:+.2f}' for v in x['per_seed'])])
        need(abs(F['checkpoints'][f'{ds}_oracle_minus_threshold_best']['delta_points'] - 100 * F[f'{ds}_rule1']['delta']) < 1e-9, f'{ds}: the best-checkpoint line equals rule 1')
    wcsv(os.path.join(a.out, 'table_d33_checkpoints_s31.csv'), rows_c, ['dataset', 'comparison', 'checkpoint', 'delta_points', 'p_paired', 'per_seed_points'])
    sel = {}
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        b0 = [100 * (r['best'] - r['last']) for (d, run), r in old.items() if d == ds and grp(run, r) == 'counted']; S = F30['selection'][ds]
        need(len(b0) == S['n'] and abs(float(np.median(b0)) - S['median']) < 1e-9 and sum(x > 5 for x in b0) == S['n_gt5'] and sum(x > 1 for x in b0) == S['n_gt1'],
             f'{ds}: the logs reproduce the S31 selection counts exactly (n {S["n"]}, median {S["median"]:.4f})')
        b1 = [100 * (r['best'] - r['last']) for (d, run), r in new.items() if d == ds and grp(run, r) == 'counted']; v = b0 + b1
        sel[ds] = {'n': len(v), 'median': float(np.median(v)), 'n_gt1': sum(x > 1 for x in v), 'n_gt5': sum(x > 5 for x in v), 'n_prev': len(b0), 'median_prev': S['median'], 'n_gt5_prev': S['n_gt5']}
    need(sel['prostate']['n'] == 61 and sel['fundus']['n'] == 43, 'counted runs after D-33: Prostate 61, Fundus 43')
    F['selection'] = sel; F['n_logs'] = len(L)
    # ------------------------------------------------ descriptive: the six teachers (v02 functions unchanged); the S9 identity counts with D-33
    H = {k: x for k, x in v02.load_harness(S31).items() if k[1].startswith('v18H_')}
    need(len(H) == 6 and all(k[2] == 'ema' for k in H), 'six harness JSONs, EMA teachers')
    tmp = tempfile.mkdtemp(); v02.band_tables(H, tmp); v02.signal_identity_table(H, tmp, {})
    for n in ('table_band_generality', 'table_band_summary', 'table_signal_identity'):
        shutil.copy(os.path.join(tmp, n + '.csv'), os.path.join(a.out, n + '_s31.csv'))
    bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_band_summary_s31.csv')))}
    idn = list(csv.DictReader(open(os.path.join(a.out, 'table_signal_identity_s31.csv'))))
    F['signal'] = {}; rows_g = []
    for ds in DATASETS:
        for sd in SEEDS:
            run = RUN[(ds, sd)]['run']; g = lambda sig, col: float(bs[(ds, run, sig)][col])
            s = {'u_epi_all': g('u_epi', 'auroc_all_mean'), 'u_epi_band': g('u_epi', 'auroc_edt5_mean'), 'u_epi_range80': g('u_epi', 'range80_all_mean'),
                 'conf_all': g('conf', 'auroc_all_mean'), 'conf_band': g('conf', 'auroc_edt5_mean'),
                 'identity_holds': [r for r in idn if r['dataset'] == ds and r['run'] == run][0]['identity_holds'] == '1'}
            s['phaseA_rule_collapse_persists'] = bool(s['u_epi_band'] <= RULE_BAND and s['u_epi_range80'] <= RULE_RANGE); F['signal'][f'{ds}_s{sd}'] = s
            rows_g.append([ds, sd, run] + [f'{x:.4g}' if isinstance(x, float) else int(x) for x in s.values()])
    wcsv(os.path.join(a.out, 'table_d33_signal_s31.csv'), rows_g, ['dataset', 'seed', 'run'] + list(s.keys()))
    I = F30['s9']; band = [v['u_epi_band'] for v in F['signal'].values()]
    F['s9'] = {'n_all': I['n_all'] + 6, 'n_testbed': I['n_testbed'] + 6, 'n_testbed_identity_holds': I['n_testbed_identity_holds'] + sum(v['identity_holds'] for v in F['signal'].values()),
               'per_benchmark': {k: v + (3 if k in ('prostate', 'fundus') else 0) for k, v in I['per_benchmark'].items()},
               'testbed_band_min_max': [min(I['testbed_band_min_max'][0], min(band)), max(I['testbed_band_min_max'][1], max(band))]}
    F['runs_trained'] = {'before_d33': F30['runs_trained']['total'], 'd33': 6, 'total': F30['runs_trained']['total'] + 6}
    F['coverage'] = {ds: {k: [min(RUN[(ds, s)][k] for s in SEEDS), max(RUN[(ds, s)][k] for s in SEEDS)] for k in ('w_first', 'w_first_fifth_mean', 'w_after_mean')} for ds in DATASETS}
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s31.json'), 'w'), indent=1, default=float)
    for ds in DATASETS:
        r, e = F[f'{ds}_rule1'], F[f'{ds}_rule2_oracle_minus_nogate']
        print(f"[d33] {ds}: oracle {100 * r['mean']:.2f}+-{100 * r['std']:.2f} vs threshold {100 * r['ref_mean']:.2f}+-{100 * r['ref_std']:.2f}; delta {100 * r['delta']:+.2f} "
              f"({', '.join(f'{100 * x:+.2f}' for x in r['per_seed_delta'])}); p {r['p']:.4f}; CI90 [{r['ci90_points'][0]:+.2f}, {r['ci90_points'][1]:+.2f}]; TOST {r['p_tost_0.5']:.3f}; "
              f"MDE {r['mde_points']:.2f} -> {r['verdict']}")
        print(f"[d33] {ds}: oracle - no gate {100 * e['delta']:+.2f} ({', '.join(f'{100 * x:+.2f}' for x in e['per_seed_delta'])}); p {e['p']:.4f} -> {e['verdict']}")
    for k, v in F['checkpoints'].items():
        print(f"[ckpt] {k}: {v['delta_points']:+.2f} p {v['p']:.3f}")
    print('[coverage]', json.dumps(F['coverage']))
    for ds, s in sel.items():
        print(f"[counts] {ds}: n {s['n_prev']} -> {s['n']}; median {s['median_prev']:.2f} -> {s['median']:.2f}; >5 {s['n_gt5_prev']} -> {s['n_gt5']}")
    print(f"[counts] S9: {F['s9']}; runs {F['runs_trained']}")
    print(f'[make_extra_s31_d33] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


if __name__ == '__main__':
    main()
