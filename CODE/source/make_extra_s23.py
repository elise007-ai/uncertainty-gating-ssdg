#!/usr/bin/env python3
"""
make_extra_s23.py (S24, 2026-10-11) -- reader of the S23 return: the six trainings of the DEviS objective in the form
of the authors' released code (D-28; `v18D_<ds>_devisrel_threshold_s<seed>`), assertion-first (R-48).
Nothing is typed by hand (R-14). Reading rules: DOCS/N51_DEVIS_RELEASED_CODE_S22c.md section 3 (fixed on 2026-10-09,
before the runs existed): the Phase-B rule for Dice, the R-34 and non-finite screens, the Phase-A rule for the signal.

  python make_extra_s23.py --returns <BSPC_results/RETURNS> --derived <derived_v04_S12> --release <release_s12> --out <derived_v04_S12/extra_s23>

Reads   RETURNS/S23/{runs,diag_out,console_logs/S23}; the logs of the within-v18 threshold references and of the
        published-form runs (`v18B_*_fixmatch_*`, `v18B_*_devisobj_threshold_s*`) wherever they sit under RETURNS;
        derived tables table_baselines.csv, table_signal_identity.csv, table_band_summary.csv (S12, unchanged).
Writes  table_d28_integrity_s23.csv      one row per run: completeness, resume, skipped iterations, hashes, split, runner history
        table_d28_per_seed_s23.csv       best / last Dice, per-domain Dice at best and at the last evaluation, references, differences
        table_d28_rule_s23.csv           the Phase-B rule per dataset (paired t, exact sign test), and the descriptive comparison with the published form
        table_d28_trajectory_s23.csv     best-minus-last of every complete v18 run of Prostate and Fundus held (base rate, R-43)
        table_d28_schedule_s23.csv       logged KL weight / beta_t against the schedule function (window means)
        table_band_generality_s23.csv, table_band_summary_s23.csv, table_signal_identity_s23.csv   (v02 functions, unchanged, on the six JSONs)
        table_d28_signal_s23.csv         per teacher: the Phase-A rule, the identity check, the reference teachers of the same seeds
        facts_s23.json                   every number the text may quote, and the list of assertions that passed
Every statement that the S24 documents make about these runs is an assertion below; a failed assertion stops the script.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, math, os, re, shutil, sys, tempfile
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_derived_v02 as v02          # noqa: E402
import make_derived_v04 as v04          # noqa: E402

SEEDS = (1337, 42, 2024)                 # the order of table_baselines.csv
DATASETS = ('prostate', 'fundus')
DOM = {'prostate': ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL'], 'fundus': ['Domain1', 'Domain2', 'Domain3', 'Domain4']}
MAXIT = {'prostate': 60000, 'fundus': 30000}
LBDOM = {'prostate': 5, 'fundus': 3}
RULE_P, RULE_DELTA = 0.05, 0.005         # Phase B (pre-registered): beats the threshold iff p < 0.05 and delta >= 0.5 Dice
RULE_BAND, RULE_RANGE = 0.70, 1e-2       # Phase A (pre-registered): collapse persists iff band AUROC <= 0.70 and range <= 1e-2
DEV_RX = re.compile(r'\] iter (\d+): .*?S_mean=([\d.eE+-]+), devis_beta_t=([\d.eE+-]+), devis_cup=([\d.eE+-]+), devis_kl_w=([\d.eE+-]+)')
SPLIT_RX = re.compile(r'\[seed_split\] seed=(\d+) data_num=(\d+) lb_num=(\d+) labeled_idxs=\[([\d, ]+)\]')
PASSED = []


def need(cond, what):
    if not cond:
        raise AssertionError('ASSERTION FAILED: ' + what)
    PASSED.append(what)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def wcsv(path, rows, header):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def f4(x):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.4f}'


def find_log(returns, ds, run):
    """The complete log of a run, wherever it sits under RETURNS; all complete copies must agree on the best value."""
    c = []
    for p in sorted(glob.glob(os.path.join(returns, '*', 'runs', ds, run, 'log.txt'))):
        q = v02.parse_log(p)
        if q and q['complete']:
            c.append((p, q))
    assert c, f'no complete log for {ds}/{run}'
    assert len({round(q['best_avg'], 6) for _, q in c}) == 1, f'complete copies of {run} disagree'
    return c[-1]


def split_of(logpath):
    m = SPLIT_RX.search(open(logpath, errors='replace').read())
    return (int(m.group(1)), [int(x) for x in m.group(4).split(',')]) if m else (None, None)


def sched(t, T, beta0=0.01, E=99, L=50):
    """Two lines of revision_modules.devis_released_schedule (kept here so that the reader needs no torch)."""
    ep = E * min(max(t, 0), T) / T
    return min(1.0, ep / L), beta0 * math.exp(-math.log(beta0) * ep / (E + 1))


def dom_means(block):
    return {d: float(np.mean(v[0])) for d, v in sorted(block.items())}


def rule(vals, ref):
    vals, ref = np.array(vals), np.array(ref); d = vals - ref
    t, p = stats.ttest_rel(vals, ref); npos = int((d > 0).sum())
    return {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)), 'ref_mean': float(ref.mean()), 'ref_std': float(ref.std(ddof=1)),
            'delta': float(d.mean()), 'per_seed_delta': [float(x) for x in d], 't': float(t), 'p': float(p), 'n_pos': npos,
            'sign_p_greater': float(stats.binomtest(npos, 3, 0.5, alternative='greater').pvalue),
            'sign_p_two_sided': float(stats.binomtest(npos, 3, 0.5).pvalue),
            'beats': bool(p < RULE_P and d.mean() >= RULE_DELTA)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--returns', required=True); ap.add_argument('--derived', required=True)
    ap.add_argument('--release', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S23 = os.path.join(a.returns, 'S23'); F = {'session': 'S24', 'reader': 'make_extra_s23.py'}
    src = os.path.join(a.release, 'CODE', 'source')
    code_sha = {n: sha(os.path.join(src, n)) for n in ('train_fourierfuse_v18_revision.py', 'revision_modules.py', 'fourierfuse_v16_modules.py', 'diag_uncertainty_harness_v18.py')}
    F['code_sha256'] = {k: v[:16] for k, v in code_sha.items()}

    # ---------------------------------------------------------------- the collector's manifests: every file as sent
    n_ok = 0
    for mf in sorted(glob.glob(os.path.join(S23, '_manifests', 'p*', 'MANIFEST.txt'))):
        for line in open(mf):
            if line.strip():
                p, size, h = line.rstrip('\n').split('\t'); q = os.path.join(S23, p)
                if os.path.exists(q):
                    assert os.path.getsize(q) == int(size) and sha(q) == h, f'manifest mismatch {p}'; n_ok += 1
                else:   # a file moved to the quarantine zip is named in HELD_IN_QUARANTINE.txt with the same hash
                    held = open(os.path.join(S23, 'HELD_IN_QUARANTINE.txt')).read(); assert p in held and h in held, f'missing {p}'
    F['manifest_files_verified'] = n_ok
    need(n_ok >= 63, 'every file of the S23 sendback present in the results tier matches the collector manifest (sha256)')

    # ---------------------------------------------------------------- runner history
    st = list(csv.DictReader(open(os.path.join(S23, 'console_logs', 'S23', 'status_jobs_D28.csv'))))
    hist = {}
    for r in st:
        hist.setdefault(r['job'], []).append(r)
    first_fail = sorted(j for j, rs in hist.items() if len(rs) == 2 and rs[0]['postflight'] == 'FAIL' and rs[1]['postflight'] == 'PASS')
    F['runner'] = {'jobs': len(hist), 'final_pass': sum(rs[-1]['postflight'] == 'PASS' for rs in hist.values()),
                   'failed_on_first_launch': first_fail,
                   'first_launch_fail_minutes': {j: round((np.datetime64(hist[j][0]['end'].replace(' ', 'T')) - np.datetime64(hist[j][0]['start'].replace(' ', 'T'))) / np.timedelta64(1, 'm'), 1) for j in first_fail},
                   'first_launch_fail_detail': sorted({hist[j][0]['detail'] + ' rc=' + hist[j][0]['exit_code'] for j in first_fail})}
    need(len(hist) == 6 and F['runner']['final_pass'] == 6, 'runner: six jobs, each ends PASS')
    need(len(first_fail) == 3 and all(hist[j][0]['gpu'] == '1' for j in first_fail), 'three jobs failed on the first launch, all in one slot of GPU 1, and passed on the second launch')
    hc = open(os.path.join(S23, 'console_logs', 'S23', 'diag_harness.console.log'), errors='replace').read()
    F['runner']['truncated_checkpoints_seen_by_harness'] = hc.count('failed finding central directory')
    need(F['runner']['truncated_checkpoints_seen_by_harness'] == 3, 'the harness call between the two launches met three unreadable best_student.pth files (the three failed jobs)')

    # ---------------------------------------------------------------- per run: integrity, screens, values
    RUN, rows_i, rows_p = {}, [], []
    for ds in DATASETS:
        for sd in SEEDS:
            run = f'v18D_{ds}_devisrel_threshold_s{sd}'; d = os.path.join(S23, 'runs', ds, run); log = os.path.join(d, 'log.txt')
            txt = open(log, errors='replace').read(); t = v04.screen_run(log); q = v02.parse_log(log)
            cfg = json.load(open(os.path.join(d, 'run_config.json'))); ar = cfg['args']
            ev = v04.parse_evaluations(txt); sa = np.array([e['student_avg'] for e in ev])
            ok_cfg = (ar['method'] == 'devis' and ar['devis_variant'] == 'released' and ar['devis_gate'] == 'threshold' and ar['devis_rel_epochs'] == 99
                      and ar['devis_rel_lambda_epochs'] == 50 and ar['seed'] == sd and ar['max_iterations'] == MAXIT[ds] and ar['lb_domain'] == LBDOM[ds]
                      and ar['lb_num'] == 20 and ar['head'] == 'evidential' and ar['consistency_ramp_frac'] == -1.0 and ar['amp'] == 1)
            ok_sha = (cfg['script_sha256'] == code_sha['train_fourierfuse_v18_revision.py'] == sha(os.path.join(d, 'code', 'train_fourierfuse_v18_revision.py'))
                      and cfg['revision_modules_sha256'] == code_sha['revision_modules.py'] == sha(os.path.join(d, 'code', 'revision_modules.py'))
                      and cfg['fourierfuse_v16_modules_sha256'] == code_sha['fourierfuse_v16_modules.py'] == sha(os.path.join(d, 'code', 'fourierfuse_v16_modules.py')))
            warn = txt.count("DEviS variant 'released' (v11)")
            refname = f'v18B_{ds}_fixmatch_split1337_init1337' if sd == 1337 else f'v18B_{ds}_fixmatch_s{sd}'
            rlog, rq = find_log(a.returns, ds, refname); plog, pq = find_log(a.returns, ds, f'v18B_{ds}_devisobj_threshold_s{sd}')
            sp, spr, spp = split_of(log), split_of(rlog), split_of(plog)
            same_split = bool(sp[1]) and sp[1] == spr[1] == spp[1] and sp[0] == sd
            sj = json.load(open(os.path.join(d, 'split.json'))); same_split = same_split and sj['lb_idxs'] == sp[1]
            J = json.load(open(os.path.join(S23, 'diag_out', f'{ds}__{run}__ema.json'))); me = J['meta']
            ok_h = (me['which'] == 'ema' and me['iter'] == q['best_iter'] and abs(me['best_avg_dice'] - q['best_avg']) < 1e-6 and run in me['ckpt']
                    and me['harness_sha256'] == code_sha['diag_uncertainty_harness_v18.py'] and me['n_models'] == 1)
            n = len(sa); last10 = float(sa[int(n * 0.9):].mean())
            head_last = dom_means(t['head']); at_best = {k: float(np.mean(list(v.values()))) for k, v in sorted(q['per_domain'].items())}
            tr = v04.screen_run(rlog); tp = v04.screen_run(plog)
            RUN[(ds, sd)] = {'run': run, 'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': t['student_avg_last'], 'last10': last10,
                             'swad_best': q['swad_best'], 'at_best': at_best, 'at_last': head_last, 'ref': rq['best_avg'], 'refname': refname,
                             'ref_last': tr['student_avg_last'], 'ref_best_iter': rq['best_iter'], 'pub': pq['best_avg'], 'pub_last': tp['student_avg_last'],
                             'pub_best_iter': pq['best_iter'], 'flag': v04.flag_of(t), 'nf': t['nf_n'], 'J': J, 'log': log, 'txt': txt}
            rows_i.append([ds, run, sd, int(t['complete']), t['n_evals'], t['n_headers'], len(t['resumed_iters']), len(cfg.get('resume_events', [])), t['nf_n'],
                           int(t['parse_ok']), v04.flag_of(t), int(ok_cfg), int(ok_sha), warn, int(same_split), int(ok_h), cfg['torch'], cfg['hostname'],
                           len(hist[run]), hist[run][0]['postflight'], hist[run][-1]['postflight'], hist[run][-1]['start'], hist[run][-1]['end']])
            need(t['complete'] and t['n_evals'] == MAXIT[ds] // 500 and t['n_headers'] == 1 and not t['resumed_iters'] and not cfg.get('resume_events'),
                 f'{run}: one training from iteration 0 to the last iteration, no resume')
            need(t['nf_n'] == 0 and t['parse_ok'] and v04.flag_of(t) == 'ok', f'{run}: no skipped iteration; R-34 screen (registered rule, last evaluation) ok')
            need(ok_cfg and warn >= 1, f'{run}: configuration = released form, threshold gate, headline settings; v11 notice in the log')
            need(ok_sha, f'{run}: code snapshot, recorded hashes and the release code are the same bytes')
            need(same_split, f'{run}: labeled slices identical to those of the threshold reference and of the published-form run of the same seed')
            need(ok_h, f'{run}: the measured teacher is the EMA weights of the best checkpoint of this log, measured with the release harness')
            rows_p.append([ds, sd, run, f'{q["best_avg"]:.6f}', q['best_iter'], f'{q["best_iter"] / MAXIT[ds]:.3f}', f4(t['student_avg_last']), f4(last10), f4(q['swad_best']),
                           ';'.join(f4(at_best[k]) for k in sorted(at_best)), ';'.join(f4(head_last[k]) for k in sorted(head_last)),
                           refname, f'{rq["best_avg"]:.6f}', f4(tr['student_avg_last']), f'{q["best_avg"] - rq["best_avg"]:+.6f}',
                           f'{pq["best_avg"]:.6f}', f4(tp['student_avg_last']), f'{q["best_avg"] - pq["best_avg"]:+.6f}'])
    wcsv(os.path.join(a.out, 'table_d28_integrity_s23.csv'), rows_i,
         ['dataset', 'run', 'seed', 'complete', 'n_evals', 'n_headers', 'n_resumes_log', 'n_resume_events_cfg', 'n_nonfinite_skips', 'parse_ok', 'r34_flag',
          'config_ok', 'hashes_ok', 'n_v11_notice_lines', 'split_equals_reference', 'harness_meta_ok', 'torch', 'hostname',
          'n_runner_attempts', 'first_attempt', 'last_attempt', 'last_start', 'last_end'])
    wcsv(os.path.join(a.out, 'table_d28_per_seed_s23.csv'), rows_p,
         ['dataset', 'seed', 'run', 'best_avg', 'best_iter', 'best_frac', 'student_avg_last', 'student_avg_last10pct', 'swad_best', 'dice_at_best_per_domain',
          'dice_at_last_per_domain', 'threshold_reference', 'ref_best_avg', 'ref_last', 'delta_vs_threshold', 'published_form_best_avg', 'published_form_last', 'delta_vs_published_form'])

    # ---------------------------------------------------------------- the reader reproduces the published-form row first
    base = {(r['dataset'], r['method']): r for r in csv.DictReader(open(os.path.join(a.derived, 'table_baselines.csv')))}
    rows_r = []
    for ds in DATASETS:
        ref = [RUN[(ds, s)]['ref'] for s in SEEDS]; pub = [RUN[(ds, s)]['pub'] for s in SEEDS]; rel = [RUN[(ds, s)]['best'] for s in SEEDS]
        rp = rule(pub, ref); b = base[(ds, 'devisobj_threshold')]
        need(abs(rp['mean'] - float(b['mean_dice'])) < 6e-5 and abs(rp['std'] - float(b['std_dice'])) < 6e-5 and abs(rp['delta'] - float(b['delta_vs_threshold'])) < 6e-5
             and abs(rp['t'] - float(b['t_paired'])) < 6e-4 and abs(rp['p'] - float(b['p_paired'])) < 6e-5 and abs(rp['ref_mean'] - float(b['threshold_ref_mean'])) < 6e-5,
             f'{ds}: this reader reproduces the published-form row of table_baselines.csv (mean, sd, delta, t, p) from the logs')
        rr = rule(rel, ref); rd = rule(rel, pub)
        F[f'{ds}_rule'] = {k: rr[k] for k in rr}; F[f'{ds}_published_form'] = {k: rp[k] for k in ('mean', 'std', 'delta', 'p', 'per_seed_delta')}
        F[f'{ds}_released_minus_published'] = {'delta': rd['delta'], 'per_seed': rd['per_seed_delta'], 't': rd['t'], 'p': rd['p'], 'n_pos': rd['n_pos']}
        F[f'{ds}_per_seed'] = {str(s): {k: RUN[(ds, s)][k] for k in ('best', 'best_iter', 'last', 'last10', 'swad_best', 'ref', 'ref_last', 'pub', 'pub_last')} for s in SEEDS}
        for lab, x, y, r_ in (('released_form_vs_threshold_reference', rel, ref, rr), ('published_form_vs_threshold_reference', pub, ref, rp), ('released_form_vs_published_form', rel, pub, rd)):
            rows_r.append([ds, lab, f4(r_['mean']), f4(r_['std']), f4(r_['ref_mean']), f4(r_['ref_std']), f'{r_["delta"]:+.4f}', ';'.join(f'{v:+.4f}' for v in r_['per_seed_delta']),
                           f'{r_["t"]:.4f}', f'{r_["p"]:.4f}', r_['n_pos'], f'{r_["sign_p_greater"]:.4f}', f'{r_["sign_p_two_sided"]:.4f}',
                           int(r_['beats']) if lab != 'released_form_vs_published_form' else ''])
        need(not rr['beats'], f'{ds}: the released form does not meet the pre-registered criterion (paired t p < 0.05 and delta >= 0.5 Dice) against the threshold gate')
        need(abs(rr['delta']) < 0.005 and rr['p'] > 0.5, f'{ds}: released form within half a Dice point of the threshold gate in the three-seed mean, paired p > 0.5')
        need(rd['n_pos'] == 3 and rd['delta'] > 0.01, f'{ds}: the released form is above the published form in each of the three seeds, by more than one point in the mean')
    wcsv(os.path.join(a.out, 'table_d28_rule_s23.csv'), rows_r,
         ['dataset', 'comparison', 'mean_dice', 'std_dice', 'ref_mean', 'ref_std', 'delta', 'per_seed_delta_1337_42_2024', 't_paired', 'p_paired', 'n_seeds_positive',
          'sign_test_p_one_sided', 'sign_test_p_two_sided', 'meets_phaseB_rule'])

    # ---------------------------------------------------------------- trajectories and the base rate of late degradation (R-43)
    rows_t, T = [], {}
    for ds in DATASETS:
        seen = {}
        for p in sorted(glob.glob(os.path.join(a.returns, '*', 'runs', ds, 'v18*', 'log.txt'))):
            run = p.split(os.sep)[-2]
            if run.startswith('v18smoke'):
                continue
            q = v02.parse_log(p)
            if not (q and q['complete']):
                continue
            t = v04.screen_run(p)
            if t.get('student_avg_last') is None:
                continue
            ev = v04.parse_evaluations(open(p, errors='replace').read()); sa = np.array([e['student_avg'] for e in ev]); n = len(sa)
            seen[run] = {'best': q['best_avg'], 'frac': q['best_iter'] / t['max_iterations'], 'last': t['student_avg_last'], 'last10': float(sa[int(n * 0.9):].mean()),
                         'flag': v04.flag_of(t), 'nf': t['nf_n'], 'ramp': 'ramp10' in run}
        T[ds] = seen
        for run, s in sorted(seen.items()):
            grp = ('released_form' if run.startswith('v18D') else 'threshold_reference' if re.match(r'v18B_\w+?_fixmatch_', run) else
                   'published_form_threshold_gate' if 'devisobj_threshold' in run else 'short_ramp' if s['ramp'] else 'other')
            rows_t.append([ds, run, grp, f4(s['best']), f'{s["frac"]:.3f}', f4(s['last']), f4(s['last10']), f'{100 * (s["best"] - s["last"]):.2f}', s['flag'], s['nf']])
    wcsv(os.path.join(a.out, 'table_d28_trajectory_s23.csv'), rows_t,
         ['dataset', 'run', 'group', 'best_avg', 'best_frac', 'student_avg_last', 'student_avg_last10pct', 'best_minus_last_points', 'r34_flag', 'n_nonfinite_skips'])
    for ds in DATASETS:
        rel = [100 * (RUN[(ds, s)]['best'] - RUN[(ds, s)]['last']) for s in SEEDS]; ref = [100 * (RUN[(ds, s)]['ref'] - RUN[(ds, s)]['ref_last']) for s in SEEDS]
        pub = [100 * (RUN[(ds, s)]['pub'] - RUN[(ds, s)]['pub_last']) for s in SEEDS]
        oth = {r: 100 * (s['best'] - s['last']) for r, s in T[ds].items() if not r.startswith('v18D') and not s['ramp'] and s['flag'] == 'ok' and s['nf'] == 0}
        F[f'{ds}_late'] = {'released_best_minus_last': rel, 'threshold_ref_best_minus_last': ref, 'published_form_best_minus_last': pub,
                           'released_best_frac': [RUN[(ds, s)]['best_iter'] / MAXIT[ds] for s in SEEDS],
                           'released_last': [RUN[(ds, s)]['last'] for s in SEEDS], 'released_last10': [RUN[(ds, s)]['last10'] for s in SEEDS],
                           'other_healthy_runs': {'n': len(oth), 'median': float(np.median(list(oth.values()))), 'max': float(max(oth.values())),
                                                  'n_above_5': sum(v > 5 for v in oth.values()), 'n_above_10': sum(v > 10 for v in oth.values()),
                                                  'above_5': {k: round(v, 2) for k, v in sorted(oth.items()) if v > 5}},
                           'short_ramp': {r: round(100 * (s['best'] - s['last']), 2) for r, s in T[ds].items() if s['ramp']}}
    P = F['prostate_late']; Fu = F['fundus_late']
    need(min(P['released_best_minus_last']) > 4 and sorted(P['released_best_minus_last'])[1] > 15, 'Prostate: each released-form run ends more than 4 points below its best checkpoint, two of them more than 15')
    need(all(0.55 < x < 0.70 for x in P['released_best_frac']), 'Prostate: the best checkpoint of each released-form run lies between 55 % and 70 % of training')
    need(max(Fu['released_best_minus_last']) < 0.5, 'Fundus: each released-form run ends within half a point of its best checkpoint')
    need(max(P['threshold_ref_best_minus_last']) > 10 and min(P['threshold_ref_best_minus_last']) < 0.5, 'Prostate: the three threshold references end 0.3 to 10.4 points below their best checkpoint')
    need(max(P['published_form_best_minus_last']) < 2, 'Prostate: the three published-form runs end less than 2 points below their best checkpoint')
    need(P['other_healthy_runs']['n_above_5'] >= 4, 'Prostate: at least four other complete, unflagged whole-ramp runs end more than 5 points below their best checkpoint (the behaviour is not specific to the released form)')
    drops = {}
    for s in SEEDS:
        r = RUN[('prostate', s)]; drops[s] = {DOM['prostate'][k - 1]: 100 * (r['at_best'][k] - r['at_last'][k]) for k in sorted(r['at_best'])}
    F['prostate_per_vendor_drop_best_to_last'] = {str(s): {k: round(v, 2) for k, v in d.items()} for s, d in drops.items()}
    F['prostate_per_vendor_last'] = {str(s): {DOM['prostate'][k - 1]: round(RUN[('prostate', s)]['at_last'][k], 4) for k in sorted(RUN[('prostate', s)]['at_last'])} for s in SEEDS}
    for s in (1337, 2024):
        top3 = sorted(drops[s], key=drops[s].get, reverse=True)[:3]
        need(set(top3) == {'BIDMC', 'HK', 'BMC'} and max(drops[s][v] for v in ('I2CVB', 'RUNMC', 'UCL')) < 3,
             f'Prostate s{s}: the loss from best to last is carried by BIDMC, HK and BMC; I2CVB, RUNMC and UCL lose less than 3 points')

    # ---------------------------------------------------------------- the schedule the runs logged against the schedule function
    rows_s, worst = [], 0.0
    for ds in DATASETS:
        for sd in SEEDS:
            r = RUN[(ds, sd)]; Tm = MAXIT[ds]; st = [(int(i), float(S), float(b), float(c), float(k)) for i, S, b, c, k in DEV_RX.findall(r['txt'])]
            assert len(st) == Tm // 500, (r['run'], len(st))
            for it, S, b, c, k in st:
                w = [sched(t, Tm) for t in range(it - 500, it)]; km = float(np.mean([x[0] for x in w])); bm = float(np.mean([x[1] for x in w]))
                e = max(abs(k - km) / max(km, 1e-12), abs(b - bm) / bm); worst = max(worst, e)
                if it in (500, Tm // 4, Tm // 2, r['best_iter'], 3 * Tm // 4, Tm):
                    rows_s.append([ds, r['run'], it, f'{k:.6g}', f'{km:.6g}', f'{b:.6g}', f'{bm:.6g}', f'{c:.6g}', f'{S:.6g}'])
            late = [c for it, S, b, c, k in st if it > 0.2 * Tm]
            F.setdefault('schedule', {})[r['run']] = {'cup_mean_after_20pct': float(np.mean(late)), 'cup_max_after_20pct': float(np.max(late)),
                                                     'S_mean_last': st[-1][1], 'beta_t_logged_last_window': st[-1][2], 'beta_t_at_best_iter': sched(r['best_iter'], Tm)[1]}
    wcsv(os.path.join(a.out, 'table_d28_schedule_s23.csv'), rows_s,
         ['dataset', 'run', 'iter', 'kl_weight_logged', 'kl_weight_schedule_window_mean', 'beta_t_logged', 'beta_t_schedule_window_mean', 'calibration_penalty_logged', 'S_mean_logged'])
    F['schedule']['largest_relative_difference_logged_vs_function'] = worst
    F['schedule']['beta_t_last_iteration'] = sched(60000, 60000)[1]; F['schedule']['beta_t_released_code_last_epoch'] = 0.01 * math.exp(-math.log(0.01) / 100 * 98)
    F['schedule']['kl_weight_reaches_1_at_fraction'] = 50 / 99
    F['schedule']['dice_weight_at_fraction'] = {str(f): 1 - sched(f * 60000, 60000)[1] for f in (0.25, 0.5, 0.6, 0.75, 0.9, 1.0)}
    need(worst < 2e-3, 'the KL weight and beta_t logged every 500 iterations equal the window means of the schedule function (relative 2e-3) in all six runs')
    need(abs(F['schedule']['beta_t_last_iteration'] - 0.955) < 5e-4 and abs(F['schedule']['beta_t_released_code_last_epoch'] - 0.912) < 5e-4,
         'beta_t ends at 0.955 in our runs (continuous epoch 99 t / T); the released loop ends at 0.912 (epoch 98)')

    # ---------------------------------------------------------------- the signal of the six teachers (v02 functions, unchanged)
    H = {k: v for k, v in v02.load_harness(S23).items() if k[1].startswith('v18D_')}
    need(len(H) == 6 and all(k[2] == 'ema' for k in H), 'six harness JSONs, EMA teachers')
    tmp = tempfile.mkdtemp(); A = {}
    v02.band_tables(H, tmp); v02.signal_identity_table(H, tmp, A)
    for n in ('table_band_generality', 'table_band_summary', 'table_signal_identity'):
        shutil.copy(os.path.join(tmp, n + '.csv'), os.path.join(a.out, n + '_s23.csv'))
    bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_band_summary_s23.csv')))}
    idn = {(r['dataset'], r['run']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_signal_identity_s23.csv')))}
    old_bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.derived, 'table_band_summary.csv')))}
    old_id = list(csv.DictReader(open(os.path.join(a.derived, 'table_signal_identity.csv'))))
    rows_g, SIG = [], {}
    for ds in DATASETS:
        for sd in SEEDS:
            r = RUN[(ds, sd)]; run = r['run']; u = bs[(ds, run, 'u_epi')]; c = bs[(ds, run, 'conf')]; i = idn[(ds, run)]
            band, rng = float(u['auroc_edt5_mean']), float(u['range80_all_mean'])
            per_dom = [float(x['signals']['u_epi']['band']['edt_gt_5']['auroc']) for _, x in sorted(r['J']['per_domain'].items(), key=lambda kv: int(kv[0]))]
            ou = old_bs[(ds, r['refname'], 'u_epi')]; op = old_bs[(ds, f'v18B_{ds}_devisobj_threshold_s{sd}', 'u_epi')]
            SIG[(ds, sd)] = {'auroc_all': float(u['auroc_all_mean']), 'band': band, 'range': rng, 'range_band': float(u['range80_edt5_mean']),
                             'conf_all': float(c['auroc_all_mean']), 'conf_band': float(c['auroc_edt5_mean']), 'errors_in_band': float(u['frac_errors_in_edt5_mean']),
                             'identity_holds': i['identity_holds'] == '1', 'gap_all': float(i['max_abs_auroc_gap_all']), 'gap_band': float(i['max_abs_auroc_gap_edt5']),
                             'median_ratio': float(i['median_ratio_u_epi_over_conf']), 'K': int(i['K']), 'band_per_domain': per_dom,
                             'ref_all': float(ou['auroc_all_mean']), 'ref_band': float(ou['auroc_edt5_mean']), 'ref_range': float(ou['range80_all_mean']),
                             'pub_all': float(op['auroc_all_mean']), 'pub_band': float(op['auroc_edt5_mean']), 'pub_range': float(op['range80_all_mean'])}
            s = SIG[(ds, sd)]
            rows_g.append([ds, sd, run, f4(s['auroc_all']), f4(band), f'{rng:.3e}', f'{s["range_band"]:.3e}', f4(s['conf_all']), f4(s['conf_band']), f4(s['errors_in_band']),
                           int(band <= RULE_BAND), int(rng <= RULE_RANGE), int(band <= RULE_BAND and rng <= RULE_RANGE), f'{band - RULE_BAND:+.4f}', f'{RULE_RANGE / rng:.0f}',
                           int(s['identity_holds']), f4(s['gap_all']), f4(s['gap_band']), f4(s['median_ratio']), ';'.join(f'{x:.3f}' for x in per_dom),
                           f4(s['ref_all']), f4(s['ref_band']), f'{s["ref_range"]:.3e}', f4(s['pub_all']), f4(s['pub_band']), f'{s["pub_range"]:.3e}'])
    wcsv(os.path.join(a.out, 'table_d28_signal_s23.csv'), rows_g,
         ['dataset', 'seed', 'run', 'u_epi_auroc_all', 'u_epi_auroc_band5', 'u_epi_range80_all', 'u_epi_range80_band5', 'conf_auroc_all', 'conf_auroc_band5', 'frac_errors_in_band5',
          'band_le_0.70', 'range_le_1e-2', 'phaseA_rule_collapse_persists', 'band_margin_to_0.70', 'range_factor_below_1e-2', 'identity_holds', 'max_abs_auroc_gap_all',
          'max_abs_auroc_gap_band5', 'median_ratio_u_epi_over_conf', 'u_epi_band5_per_domain', 'threshold_ref_auroc_all', 'threshold_ref_band5', 'threshold_ref_range',
          'published_form_auroc_all', 'published_form_band5', 'published_form_range'])
    for ds in DATASETS:
        v = [SIG[(ds, s)] for s in SEEDS]
        F[f'{ds}_signal'] = {'per_seed': {str(s): {k: (SIG[(ds, s)][k] if not isinstance(SIG[(ds, s)][k], list) else [round(x, 4) for x in SIG[(ds, s)][k]]) for k in SIG[(ds, s)]} for s in SEEDS},
                             'auroc_all_min_max': [min(x['auroc_all'] for x in v), max(x['auroc_all'] for x in v)], 'band_min_max': [min(x['band'] for x in v), max(x['band'] for x in v)],
                             'band_mean': float(np.mean([x['band'] for x in v])), 'range_min_max': [min(x['range'] for x in v), max(x['range'] for x in v)],
                             'range_band_max': max(x['range_band'] for x in v), 'n_band_le_070': sum(x['band'] <= RULE_BAND for x in v), 'n_rule_persists': sum(x['band'] <= RULE_BAND and x['range'] <= RULE_RANGE for x in v),
                             'n_identity_holds': sum(x['identity_holds'] for x in v), 'gap_band_max': max(x['gap_band'] for x in v), 'gap_all_max': max(x['gap_all'] for x in v),
                             'ref_all_min_max': [min(x['ref_all'] for x in v), max(x['ref_all'] for x in v)], 'ref_band_min_max': [min(x['ref_band'] for x in v), max(x['ref_band'] for x in v)],
                             'ref_range_max': max(x['ref_range'] for x in v), 'pub_all_min_max': [min(x['pub_all'] for x in v), max(x['pub_all'] for x in v)],
                             'pub_band_min_max': [min(x['pub_band'] for x in v), max(x['pub_band'] for x in v)], 'pub_range_min_max': [min(x['pub_range'] for x in v), max(x['pub_range'] for x in v)]}
    allsix = list(SIG.values())
    F['signal_all_six'] = {'range_max': max(x['range'] for x in allsix), 'range_factor_below_bound_min': RULE_RANGE / max(x['range'] for x in allsix),
                           'band_min_max': [min(x['band'] for x in allsix), max(x['band'] for x in allsix)], 'n_band_le_070': sum(x['band'] <= RULE_BAND for x in allsix),
                           'n_band_above_070': sum(x['band'] > RULE_BAND for x in allsix), 'band_excess_max': max(x['band'] for x in allsix) - RULE_BAND,
                           'n_identity_holds': sum(x['identity_holds'] for x in allsix), 'median_ratio_dev_max': max(abs(x['median_ratio'] - x['K'] / (x['K'] - 1)) / (x['K'] / (x['K'] - 1)) for x in allsix)}
    need(F['signal_all_six']['range_max'] <= RULE_RANGE / 2, 'range condition of the Phase-A rule: the central-80 % range of u_epi is at most half of the 1e-2 bound on all six teachers')
    # S9 counts with the six teachers added
    def cls(run):
        # the classification of WRITING/S13_patch_main_tex.py (the one behind the counts printed in S9)
        return 'ensemble' if 'ENSEMBLE' in run else 'devis' if 'devis' in run else 'medl' if 'medl' in run else 'exp' if 'act_exp' in run else 'testbed'
    dm_old = [r for r in old_id if cls(r['run']) in ('devis', 'medl')]; tb_old = [r for r in old_id if cls(r['run']) == 'testbed']
    dm_band_old = [float(old_bs[(r['dataset'], r['run'], 'u_epi')]['auroc_edt5_mean']) for r in dm_old]
    dv_old = [r for r in dm_old if cls(r['run']) == 'devis']
    dv_all_old = [float(old_bs[(r['dataset'], r['run'], 'u_epi')]['auroc_all_mean']) for r in dv_old]
    tb_all = [float(old_bs[(r['dataset'], r['run'], 'u_epi')]['auroc_all_mean']) for r in tb_old if r['dataset'] in DATASETS]
    tb_band = [float(old_bs[(r['dataset'], r['run'], 'u_epi')]['auroc_edt5_mean']) for r in tb_old]
    new_band = [x['band'] for x in allsix]; new_fail = sum(not x['identity_holds'] for x in allsix)
    F['s9_counts'] = {'n_all_old': len(old_id), 'n_all_new': len(old_id) + 6, 'n_testbed': len(tb_old), 'n_devis_medl_old': len(dm_old), 'n_devis_medl_new': len(dm_old) + 6,
                      'n_devis_medl_identity_fails_old': sum(r['identity_holds'] != '1' for r in dm_old), 'n_devis_medl_identity_fails_new': sum(r['identity_holds'] != '1' for r in dm_old) + new_fail,
                      'devis_medl_band_min_max_old': [min(dm_band_old), max(dm_band_old)], 'devis_medl_band_min_max_new': [min(dm_band_old + new_band), max(dm_band_old + new_band)],
                      'testbed_band_min_max': [min(tb_band), max(tb_band)], 'n_devis_old': len(dv_old), 'n_devis_new': len(dv_old) + 6,
                      'devis_gap_band_max_old': max(float(r['max_abs_auroc_gap_edt5']) for r in dm_old if r['identity_holds'] != '1'),
                      'devis_all_min_max_published_form_complete': None}
    need(F['s9_counts']['n_all_old'] == 83 and F['s9_counts']['n_testbed'] == 62 and F['s9_counts']['n_devis_medl_old'] == 18
         and sum(cls(r['run']) == 'exp' for r in old_id) == 1 and sum(cls(r['run']) == 'ensemble' for r in old_id) == 2, 'the S12 identity table holds 83 checkpoints: 62 testbed-objective, 18 DEviS / MEDL, 1 clamped-exponential and 2 ensembles')
    # all-pixel ranking of the DEviS-objective teachers against the testbed-objective teachers of the same two datasets (descriptive)
    for ds in DATASETS:
        dv = {r['run']: float(old_bs[(ds, r['run'], 'u_epi')]['auroc_all_mean']) for r in dv_old if r['dataset'] == ds}
        tb = [float(old_bs[(ds, r['run'], 'u_epi')]['auroc_all_mean']) for r in tb_old if r['dataset'] == ds]
        rel = [SIG[(ds, s)]['auroc_all'] for s in SEEDS]
        F[f'{ds}_all_pixel_ranking'] = {'released_form': [min(rel), max(rel)], 'published_form_threshold_gate': [min(v for k, v in dv.items() if 'devisobj' in k), max(v for k, v in dv.items() if 'devisobj' in k)],
                                        'published_form_all_12': {k: round(v, 4) for k, v in sorted(dv.items())}, 'testbed_objective': {'n': len(tb), 'min': min(tb), 'max': max(tb), 'median': float(np.median(tb))}}
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s23.json'), 'w'), indent=1, default=float)
    print(f'[make_extra_s23] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


if __name__ == '__main__':
    main()
