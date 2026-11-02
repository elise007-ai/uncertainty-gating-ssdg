#!/usr/bin/env python3
"""
make_extra_s29.py (S30, 2026-10-21) -- reader of the S29 return: D-31, six trainings of the percentile (rank) gate (`v18F_<ds>_percentile_s<seed>`),
assertion-first (R-48); nothing is typed by hand (R-14). Rules: DOCS/D31_PREREGISTRATION_S29.md (fixed 2026-10-18, before any run). Helpers come
from make_extra_s23.py (D-28), make_selection_s25.py (D-30) and make_power_s29.py (TOST, MDE), so the readers apply the same functions.
  python make_extra_s29.py --returns <RETURNS> --derived <derived_v04_S12> --release <release_s12> --tex <main.tex> --out <derived_v04_S12/extra_s29>
Writes table_d31_{integrity,per_seed,rule,signal}_s29.csv, table_{band_generality,band_summary,signal_identity}_s29.csv (v02 functions, unchanged),
facts_s29.json. The adjudication is generated from facts_s29.json by make_adjudication_s29.py.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, re, shutil, sys, tempfile
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_derived_v02 as v02                      # noqa: E402
import make_derived_v04 as v04                      # noqa: E402
from make_extra_s23 import (need, sha, wcsv, f4, find_log, split_of, rule,   # noqa: E402
                            SEEDS, DATASETS, DOM, MAXIT, LBDOM, RULE_P, RULE_DELTA, RULE_BAND, RULE_RANGE, PASSED)
from make_selection_s25 import cmd_pairs, ref_name  # noqa: E402
from make_power_s29 import mde, MARGIN, ALPHA       # noqa: E402

W_RX = re.compile(r'\] iter (\d+): .*?w_mean=([\d.eE+-]+), w_min=([\d.eE+-]+)')
NUM = r'\$([\d.]+)(?: \\pm ([\d.]+))?\$'
T6 = {'threshold': r'Fixed threshold \(\$\\tau=0\.80\$\)', 'percentile_probe': r'Percentile gate\\textsuperscript\{\$\\dagger\$\}', 'daep': r'Evidential \(\\daep\)'}


def tost(dp):
    # rule 2: 90 % CI, TOST at the registered +-0.5 Dice points, MDE (alpha 0.05, power 0.8): the formulas of make_power_s29.py
    d = np.array(dp, float); n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1)); se = sd / math.sqrt(n); t90 = float(stats.t.ppf(0.95, n - 1))
    p = max(float(stats.t.sf((m + MARGIN) / se, n - 1)), float(stats.t.cdf((m - MARGIN) / se, n - 1)))
    return {'mean_points': m, 'sd_points': sd, 'ci90_points': [m - t90 * se, m + t90 * se], 'p_tost_0.5': p, 'equivalent_at_0.5': bool(p < ALPHA), 'mde_points': float(mde(sd, n))}


def wfrac(txt, Tm):
    # the logged mean of the pseudo-label mask (w_mean) after 20 % of training: mean, min, max; and the number of logged lines
    w = [(int(i), float(m)) for i, m, _ in W_RX.findall(txt)]; late = [m for i, m in w if i > 0.2 * Tm]
    return float(np.mean(late)), float(min(late)), float(max(late)), len(w)


def main():
    ap = argparse.ArgumentParser()
    for k in ('returns', 'derived', 'release', 'tex', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S29 = os.path.join(a.returns, 'S29'); CL = os.path.join(S29, 'console_logs', 'S29'); src = os.path.join(a.release, 'CODE', 'source')
    F = {'session': 'S30', 'reader': 'make_extra_s29.py', 'preregistration': 'DOCS/D31_PREREGISTRATION_S29.md'}
    code_sha = {n: sha(os.path.join(src, n)) for n in ('train_fourierfuse_v18_revision.py', 'revision_modules.py', 'fourierfuse_v16_modules.py', 'diag_uncertainty_harness_v18.py')}
    F['code_sha256'] = {k: v[:16] for k, v in code_sha.items()}; F24 = json.load(open(os.path.join(a.derived, 'extra_s24', 'facts_s24.json')))
    # ---------------------------------------------------------------- rule 1: every file as sent, the runner, the smoke run, the step files
    n_ok, listed = 0, []
    for mf in sorted(glob.glob(os.path.join(S29, '_manifests', 'p*', 'MANIFEST.txt'))):
        for line in open(mf, encoding='utf-8'):
            if line.strip():
                p, size, h = line.rstrip('\n').split('\t'); q = os.path.join(S29, *p.split('/')); listed.append(p)
                assert os.path.exists(q) and os.path.getsize(q) == int(size) and sha(q) == h, f'manifest mismatch {p}'; n_ok += 1
    F['manifest_files_verified'] = n_ok
    need(n_ok == 63 and len(set(listed)) == 63, 'every file of the S29 sendback (63, four parts) is in the results tier and matches its part manifest (size and sha256)')
    need(not any(re.search(r'v18[ABDE]_', p) for p in listed), 'the sendback carries no file of an earlier training (the collector used HELD_MANIFEST_S29.txt)')
    need(sha(os.path.join(S29, 'jobs', 'jobs_D31.txt'))[:16] == 'ddbc582de5a2c73e' and sha(os.path.join(S29, 'jobs', 'jobs_D31smoke.txt'))[:16] == 'dc97009349db6a65',
         'the job lists that ran are the registered ones (tools v14)')
    hist = {}
    for r in csv.DictReader(open(os.path.join(CL, 'status_jobs_D31.csv'))):
        hist.setdefault(r['job'], []).append(r)
    hrs = lambda s, e: round(float((np.datetime64(e.replace(' ', 'T')) - np.datetime64(s.replace(' ', 'T'))) / np.timedelta64(1, 'h')), 2)
    R = F['runner'] = {'jobs': len(hist), 'final_pass': sum(rs[-1]['postflight'] == 'PASS' for rs in hist.values()), 'first_start': min(rs[0]['start'] for rs in hist.values()),
                       'last_end': max(rs[-1]['end'] for rs in hist.values()), 'hours_per_job': {j: hrs(rs[-1]['start'], rs[-1]['end']) for j, rs in hist.items()}}
    R['wall_clock_hours'] = hrs(R['first_start'], R['last_end']); R['preflight_green_line'] = '[preflight] GREEN' in open(os.path.join(CL, 'runner_jobs_D31.log'), errors='replace').read()
    need(len(hist) == 6 and R['final_pass'] == 6 and all(len(rs) == 1 for rs in hist.values()), 'runner: six jobs, one attempt each, each PASS (no failure branch, no relaunch)')
    need(not os.path.isdir(os.path.join(S29, 'console_logs', 'S29_first_attempt')), 'no S29_first_attempt folder: the failure branch of RUNBOOK S29 was not used')
    sm = list(csv.DictReader(open(os.path.join(CL, 'status_jobs_D31smoke.csv'))))
    need(len(sm) == 1 and sm[0]['postflight'] == 'PASS', 'the smoke run (Fundus, 20 iterations) passed')
    F['step_files'] = {n: os.path.exists(os.path.join(CL, n)) for n in ('preflight.txt', 'selftest_revision_modules.txt')}
    need(not any(F['step_files'].values()), 'recorded absence (V-25): the step-0 preflight file and the step-1 self-test file of RUNBOOK S29 are not in the sendback')
    # ---------------------------------------------------------------- per run: integrity, screens (rules 4, 5), values
    jobs = {l.split('\t')[0]: l.split('\t')[2] for l in open(os.path.join(S29, 'jobs', 'jobs_D31.txt'), encoding='utf-8').read().splitlines() if l.strip()}
    RUN, rows_i, rows_p = {}, [], []
    for ds in DATASETS:
        d30 = {int(re.search(r'_s(\d+)$', os.path.basename(os.path.dirname(p))).group(1)): json.load(open(p))['args']
               for p in glob.glob(os.path.join(a.returns, 'S25', 'runs', ds, 'v18A_*', 'run_config.json'))}
        dm = lambda pd: {DOM[ds][k - 1]: float(np.mean(list(v.values()))) for k, v in sorted(pd.items())}
        for sd in SEEDS:
            run = f'v18F_{ds}_percentile_s{sd}'; d = os.path.join(S29, 'runs', ds, run); log = os.path.join(d, 'log.txt')
            txt = open(log, errors='replace').read(); t = v04.screen_run(log); q = v02.parse_log(log)
            cfg = json.load(open(os.path.join(d, 'run_config.json'))); ar = cfg['args']; sa = np.array([e['student_avg'] for e in v04.parse_evaluations(txt)])
            ok_cfg = (ar['method'] == 'percentile_gate' and abs(ar['pg_keep_frac'] - 0.80) < 1e-12 and ar['seed'] == sd and ar['max_iterations'] == MAXIT[ds]
                      and ar['lb_domain'] == LBDOM[ds] and ar['lb_num'] == 20 and ar['head'] == 'evidential' and ar['consistency_ramp_frac'] == -1.0 and ar['amp'] == 1)
            ok_sha = all(cfg[k] == code_sha[n] == sha(os.path.join(d, 'code', n)) for k, n in (('script_sha256', 'train_fourierfuse_v18_revision.py'),
                         ('revision_modules_sha256', 'revision_modules.py'), ('fourierfuse_v16_modules_sha256', 'fourierfuse_v16_modules.py')))
            cp, jp = cmd_pairs(txt), cmd_pairs('python train_fourierfuse_v18_revision.py ' + jobs[run]); cmd_diff = sorted(k for k in set(cp) | set(jp) if cp.get(k) != jp.get(k))
            c30 = d30.get(sd, d30[min(d30)]); arg_diff = sorted(k for k in set(ar) | set(c30) if ar.get(k) != c30.get(k))
            refname = ref_name(ds, sd); rlog, rq = find_log(a.returns, ds, refname); tr = v04.screen_run(rlog)
            sp, spr = split_of(log), split_of(rlog); sj = json.load(open(os.path.join(d, 'split.json')))
            J = json.load(open(os.path.join(S29, 'diag_out', f'{ds}__{run}__ema.json'))); me = J['meta']
            ok_h = (me['which'] == 'ema' and me['iter'] == q['best_iter'] and abs(me['best_avg_dice'] - q['best_avg']) < 1e-6 and run in me['ckpt']
                    and me['harness_sha256'] == code_sha['diag_uncertainty_harness_v18.py'] and me['n_models'] == 1)
            pf, pmin, pmax, nw = wfrac(txt, MAXIT[ds]); rpf = wfrac(open(rlog, errors='replace').read(), MAXIT[ds])[0]
            r = RUN[(ds, sd)] = {'run': run, 'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': t['student_avg_last'], 'last10': float(sa[int(len(sa) * 0.9):].mean()),
                                 'swad_best': q['swad_best'], 'ref': rq['best_avg'], 'refname': refname, 'ref_last': tr['student_avg_last'], 'ref_best_iter': rq['best_iter'],
                                 'flag': v04.flag_of(t), 'nf': t['nf_n'], 'nf_frac': t['nf_n'] / MAXIT[ds], 'mask_after_20pct': pf, 'mask_min': pmin, 'mask_max': pmax,
                                 'ref_mask_after_20pct': rpf, 'at_best': dm(q['per_domain']), 'ref_at_best': dm(rq['per_domain'])}
            need(t['complete'] and t['parse_ok'] and t['n_evals'] == MAXIT[ds] // 500 and t['n_headers'] == 1 and not t['resumed_iters'] and not cfg.get('resume_events'),
                 f'{run}: one training from iteration 0 to the last iteration, no resume event')
            need(ok_cfg and 'method=percentile_gate' in txt, f'{run}: configuration = percentile gate, keep fraction 0.80, headline settings; the gate is named in the log')
            need(set(cmd_diff) <= {'--gpu'}, f'{run}: the command in its log equals its line of jobs_D31.txt (the runner adds only the GPU index)')
            need(set(arg_diff) <= {'method', 'save_name', 'gpu', 'seed'} and 'method' in arg_diff, f'{run}: recorded arguments = those of the D-30 runs of the dataset except the method (and seed, name, GPU)')
            need(ok_sha, f'{run}: code snapshot, recorded hashes and the release code are the same bytes')
            need(bool(sp[1]) and sp[0] == sd and sp[1] == spr[1] == sj['lb_idxs'], f'{run}: labeled slices identical to those of its threshold reference ({refname})')
            need(ok_h, f'{run}: the measured teacher is the EMA weights of the best checkpoint of this log, measured with the release harness')
            need(nw == MAXIT[ds] // 500, f'{run}: the mean of the pseudo-label mask is logged at every evaluation interval')
            need(abs(rq['best_avg'] - F24[f'{ds}_per_seed'][str(sd)]['ref']) < 1e-9, f'{run}: its reference value is the one D-29 was read against')
            rows_i.append([ds, run, sd, int(t['complete']), t['n_evals'], len(t['resumed_iters']), t['nf_n'], r['flag'], int(ok_cfg), ';'.join(cmd_diff), ';'.join(arg_diff),
                           int(ok_sha), int(ok_h), cfg['torch'], cfg['hostname'], hist[run][-1]['start'], hist[run][-1]['end']])
            rows_p.append([ds, sd, run, f'{r["best"]:.6f}', r['best_iter'], f4(r['last']), f4(r['last10']), f4(r['swad_best']), refname, f'{r["ref"]:.6f}', f4(r['ref_last']),
                           f'{r["best"] - r["ref"]:+.6f}', r['flag'], r['nf'], f'{pf:.4f}', f'{rpf:.4f}', ';'.join(f'{k}={v:.4f}' for k, v in r['at_best'].items()),
                           ';'.join(f'{k}={v:.4f}' for k, v in r['ref_at_best'].items())])
    wcsv(os.path.join(a.out, 'table_d31_integrity_s29.csv'), rows_i, ['dataset', 'run', 'seed', 'complete', 'n_evals', 'n_resumes', 'n_nonfinite_skips', 'r34_flag', 'config_ok',
         'cmd_keys_differing_from_job_line', 'arg_keys_differing_from_d30', 'hashes_ok', 'harness_meta_ok', 'torch', 'hostname', 'start', 'end'])
    wcsv(os.path.join(a.out, 'table_d31_per_seed_s29.csv'), rows_p, ['dataset', 'seed', 'run', 'best_avg', 'best_iter', 'student_avg_last', 'student_avg_last10pct', 'swad_best',
         'threshold_reference', 'ref_best_avg', 'ref_last', 'delta_vs_reference', 'r34_flag', 'n_nonfinite', 'mask_mean_after_20pct', 'ref_mask_mean_after_20pct',
         'dice_at_best_per_domain', 'ref_dice_at_best_per_domain'])
    # ---------------------------------------------------------------- rules 1-3 (registered comparison) and 2 (TOST, MDE); descriptive yardsticks of Table 6
    tex = open(a.tex, encoding='utf-8').read(); T = {}
    for k, lab in T6.items():
        m = re.search(lab + r'\s*&\s*' + NUM + r'\s*&\s*' + NUM, tex); need(m is not None, f'Table 6 row "{k}" read from main.tex (the body\'s own words)')
        T[k] = {'fundus': float(m.group(1)), 'fundus_sd': m.group(2) and float(m.group(2)), 'prostate': float(m.group(3)), 'prostate_sd': m.group(4) and float(m.group(4))}
    F['table6_main_tex'] = T; rows_r = []
    for ds in DATASETS:
        pg = [RUN[(ds, s)]['best'] for s in SEEDS]; ref = [RUN[(ds, s)]['ref'] for s in SEEDS]
        r = rule(pg, ref); e = tost([100 * (x - y) for x, y in zip(pg, ref)]); below = bool(r['p'] < RULE_P and r['delta'] <= -RULE_DELTA)
        v = 'beats the threshold gate' if r['beats'] else 'below the threshold gate' if below else 'within seed noise'
        need(abs(e['mean_points'] - 100 * r['delta']) < 1e-9 and e['equivalent_at_0.5'] == (e['ci90_points'][0] > -MARGIN and e['ci90_points'][1] < MARGIN),
             f'{ds}: TOST and the paired test use the same differences; the TOST verdict equals "90 % CI inside +-0.5"')
        fl = [RUN[(ds, s)]['flag'] for s in SEEDS]
        F[f'{ds}_rule'] = {**r, **e, 'below': below, 'verdict': v, 'r34_flags': fl, 'n_r34_flagged': sum(f != 'ok' for f in fl), 'unstable': [RUN[(ds, s)]['nf_frac'] > 0.01 for s in SEEDS]}
        F[f'{ds}_per_seed'] = {str(s): RUN[(ds, s)] for s in SEEDS}
        F[f'{ds}_descriptive'] = {'table6_threshold': T['threshold'][ds], 'table6_daep': T['daep'][ds], 'table6_probe_s1337': T['percentile_probe'][ds],
                                  'd31_mean_minus_table6_threshold_points': 100 * r['mean'] - T['threshold'][ds], 'd31_s1337_minus_probe_points': 100 * RUN[(ds, 1337)]['best'] - T['percentile_probe'][ds],
                                  'best_minus_last_points': [100 * (RUN[(ds, s)]['best'] - RUN[(ds, s)]['last']) for s in SEEDS],
                                  'ref_best_minus_last_points': [100 * (RUN[(ds, s)]['ref'] - RUN[(ds, s)]['ref_last']) for s in SEEDS],
                                  'mask_after_20pct': [RUN[(ds, s)]['mask_after_20pct'] for s in SEEDS], 'ref_mask_after_20pct': [RUN[(ds, s)]['ref_mask_after_20pct'] for s in SEEDS]}
        rows_r.append([ds, f4(r['mean']), f4(r['std']), f4(r['ref_mean']), f4(r['ref_std']), f'{100 * r["delta"]:+.2f}', ';'.join(f'{100 * x:+.2f}' for x in r['per_seed_delta']), f'{r["p"]:.4f}',
                       r['n_pos'], f'{r["sign_p_two_sided"]:.4f}', f'{e["ci90_points"][0]:+.2f};{e["ci90_points"][1]:+.2f}', f'{e["p_tost_0.5"]:.4f}', f'{e["mde_points"]:.2f}', v])
    wcsv(os.path.join(a.out, 'table_d31_rule_s29.csv'), rows_r, ['dataset', 'mean_dice', 'std_dice', 'ref_mean', 'ref_std', 'delta_points', 'per_seed_delta_points_1337_42_2024',
         'p_paired', 'n_seeds_positive', 'sign_test_p_two_sided', 'ci90_points', 'p_tost_0.5', 'mde_points', 'verdict'])
    F['expectation'] = {'registered': 'within seed noise of the reference on both benchmarks', 'failed_on': [ds for ds in DATASETS if F[f'{ds}_rule']['verdict'] != 'within seed noise']}
    F['expectation']['holds'] = not F['expectation']['failed_on']
    # ---------------------------------------------------------------- rule 6 (descriptive): the signal of the six teachers, v02 functions unchanged
    H = {k: x for k, x in v02.load_harness(S29).items() if k[1].startswith('v18F_')}
    need(len(H) == 6 and all(k[2] == 'ema' for k in H), 'six harness JSONs, EMA teachers')
    tmp = tempfile.mkdtemp(); v02.band_tables(H, tmp); v02.signal_identity_table(H, tmp, {})
    for n in ('table_band_generality', 'table_band_summary', 'table_signal_identity'):
        shutil.copy(os.path.join(tmp, n + '.csv'), os.path.join(a.out, n + '_s29.csv'))
    bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_band_summary_s29.csv')))}
    idn = {(r['dataset'], r['run']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_signal_identity_s29.csv')))}
    old = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.derived, 'table_band_summary.csv')))}
    rows_g = []; F['signal'] = {}
    for ds in DATASETS:
        g = lambda tab, key, sig, col: float(tab[(ds, key, sig)][col]) if (ds, key, sig) in tab else None
        for sd in SEEDS:
            run, refn = RUN[(ds, sd)]['run'], RUN[(ds, sd)]['refname']
            s = {'u_epi_all': g(bs, run, 'u_epi', 'auroc_all_mean'), 'u_epi_band': g(bs, run, 'u_epi', 'auroc_edt5_mean'), 'u_epi_range80': g(bs, run, 'u_epi', 'range80_all_mean'),
                 'conf_all': g(bs, run, 'conf', 'auroc_all_mean'), 'conf_band': g(bs, run, 'conf', 'auroc_edt5_mean'), 'identity_holds': idn[(ds, run)]['identity_holds'] == '1',
                 'ref_conf_all': g(old, refn, 'conf', 'auroc_all_mean'), 'ref_conf_band': g(old, refn, 'conf', 'auroc_edt5_mean'),
                 'ref_u_epi_band': g(old, refn, 'u_epi', 'auroc_edt5_mean'), 'ref_u_epi_range80': g(old, refn, 'u_epi', 'range80_all_mean')}
            s['phaseA_rule_collapse_persists'] = bool(s['u_epi_band'] <= RULE_BAND and s['u_epi_range80'] <= RULE_RANGE); F['signal'][f'{ds}_s{sd}'] = s
            rows_g.append([ds, sd, run] + ['' if x is None else f'{x:.4g}' if isinstance(x, float) else int(x) for x in s.values()])
    wcsv(os.path.join(a.out, 'table_d31_signal_s29.csv'), rows_g, ['dataset', 'seed', 'run'] + list(s.keys()))
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s29.json'), 'w'), indent=1, default=float)
    for ds in DATASETS:
        r = F[f'{ds}_rule']
        print(f"[d31] {ds}: {100 * r['mean']:.2f}+-{100 * r['std']:.2f} vs {100 * r['ref_mean']:.2f}+-{100 * r['ref_std']:.2f}; delta {100 * r['delta']:+.2f} "
              f"({', '.join(f'{100 * x:+.2f}' for x in r['per_seed_delta'])}); p {r['p']:.4f}; CI90 [{r['ci90_points'][0]:+.2f}, {r['ci90_points'][1]:+.2f}]; "
              f"TOST {r['p_tost_0.5']:.3f}; MDE {r['mde_points']:.2f}; flags {r['r34_flags']}; -> {r['verdict']}")
    print(f'[make_extra_s29] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


if __name__ == '__main__':
    main()
