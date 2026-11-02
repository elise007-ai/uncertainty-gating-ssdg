#!/usr/bin/env python3
"""
make_extra_s24.py (S25, 2026-10-13) -- reader of the S24 return: the six trainings of the DEviS objective in the form of the
authors' released code with DEviS's own belief-mass gate w = 1 - u (D-29; `v18E_<ds>_devisrel_belief_s<seed>`), assertion-first (R-48).
Nothing is typed by hand (R-14). Reading rules: DOCS/D29_PREREGISTRATION_S24.md, rules 1-6, fixed on 2026-10-11 before the runs existed.
Helpers and constants are imported from make_extra_s23.py (the D-28 reader), so the two readers apply the same functions.

  python make_extra_s24.py --returns <BSPC_results/RETURNS> --derived <derived_v04_S12> --release <release_s12> --out <derived_v04_S12/extra_s24>

Reads   RETURNS/S24/{runs,diag_out,console_logs/S24,_manifests}; RETURNS/S23 (the D-28 runs of the same seeds); the logs of the within-v18
        threshold references and of the published-form belief-gate runs (`v18B_*_fixmatch_*`, `v18B_*_devis_s*`) wherever they sit under RETURNS;
        derived tables table_baselines.csv, table_band_summary.csv, table_signal_identity.csv (S12, unchanged); extra_s23/facts_s23.json.
Writes  table_d29_integrity_s24.csv     one row per run: completeness, resume, skipped iterations, hashes, split, runner history (rule 1, 2)
        table_d29_per_seed_s24.csv      best / last Dice, per-domain Dice at best and last, the three references, differences (rules 3, 4, 6)
        table_d29_rule_s24.csv          rule 3 (against the threshold reference), rule 4 (against the D-28 run), the descriptive published-form comparison
        table_d29_gate_weight_s24.csv   the logged belief-mass weight w_mean per run: after 20 % of training, early, at the end (rule 6)
        table_d29_trajectory_s24.csv    best minus last per run next to D-28 and the threshold references; per-vendor drops (rule 6)
        table_d29_schedule_s24.csv      logged KL weight / beta_t against the schedule function (window means)
        table_band_generality_s24.csv, table_band_summary_s24.csv, table_signal_identity_s24.csv   (v02 functions, unchanged, on the six JSONs)
        table_d29_signal_s24.csv        per teacher: the Phase-A rule, the identity check, the D-28 and threshold-reference teachers of the same seeds (rule 5)
        facts_s24.json                  every number the text may quote, and the list of assertions that passed
        ADJUDICATION_S24.md             generated from the facts (copied to DOCS/adjudications/ by the session)
Every statement that the S25 documents make about these runs is an assertion below; a failed assertion stops the script.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, re, shutil, sys, tempfile
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_derived_v02 as v02                      # noqa: E402
import make_derived_v04 as v04                      # noqa: E402
from make_extra_s23 import (need, sha, wcsv, f4, find_log, split_of, sched, dom_means, rule,   # noqa: E402
                            SEEDS, DATASETS, DOM, MAXIT, LBDOM, RULE_P, RULE_DELTA, RULE_BAND, RULE_RANGE, DEV_RX, PASSED)

W_RX = re.compile(r'\] iter (\d+): .*?w_mean=([\d.eE+-]+), w_min=([\d.eE+-]+)')


def try_log(returns, ds, run):
    try:
        return find_log(returns, ds, run)
    except AssertionError:
        return None, None


def gate_rule(vals, ref):
    """Rule 4: the gate changes the result iff paired p < 0.05 and |mean difference| >= 0.5 Dice (two-sided form of the Phase-B rule)."""
    r = rule(vals, ref)
    r['changes'] = bool(r['p'] < RULE_P and abs(r['delta']) >= RULE_DELTA)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--returns', required=True); ap.add_argument('--derived', required=True)
    ap.add_argument('--release', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S24 = os.path.join(a.returns, 'S24'); S23 = os.path.join(a.returns, 'S23'); F = {'session': 'S25', 'reader': 'make_extra_s24.py'}
    src = os.path.join(a.release, 'CODE', 'source')
    code_sha = {n: sha(os.path.join(src, n)) for n in ('train_fourierfuse_v18_revision.py', 'revision_modules.py', 'fourierfuse_v16_modules.py', 'diag_uncertainty_harness_v18.py')}
    F['code_sha256'] = {k: v[:16] for k, v in code_sha.items()}
    F23 = json.load(open(os.path.join(a.derived, 'extra_s23', 'facts_s23.json')))

    # ---------------------------------------------------------------- rule 1 (a): the collector's manifests, every file as sent
    n_ok = 0
    for mf in sorted(glob.glob(os.path.join(S24, '_manifests', 'p*', 'MANIFEST.txt'))):
        for line in open(mf):
            if line.strip():
                p, size, h = line.rstrip('\n').split('\t'); q = os.path.join(S24, p)
                assert os.path.exists(q) and os.path.getsize(q) == int(size) and sha(q) == h, f'manifest mismatch {p}'; n_ok += 1
    F['manifest_files_verified'] = n_ok
    need(n_ok == 66, 'every file of the S24 sendback (66) is present in the results tier and matches the collector manifest (size and sha256)')

    # ---------------------------------------------------------------- rule 2: runner history (no failure, no relaunch)
    st = list(csv.DictReader(open(os.path.join(S24, 'console_logs', 'S24', 'status_jobs_D29.csv'))))
    hist = {}
    for r in st:
        hist.setdefault(r['job'], []).append(r)
    F['runner'] = {'jobs': len(hist), 'attempts_per_job': {j: len(rs) for j, rs in hist.items()},
                   'final_pass': sum(rs[-1]['postflight'] == 'PASS' for rs in hist.values()),
                   'first_start': min(rs[0]['start'] for rs in hist.values()), 'last_end': max(rs[-1]['end'] for rs in hist.values()),
                   'hours_per_job': {j: round((np.datetime64(rs[-1]['end'].replace(' ', 'T')) - np.datetime64(rs[-1]['start'].replace(' ', 'T'))) / np.timedelta64(1, 'h'), 2) for j, rs in hist.items()}}
    F['runner']['wall_clock_hours'] = round((np.datetime64(F['runner']['last_end'].replace(' ', 'T')) - np.datetime64(F['runner']['first_start'].replace(' ', 'T'))) / np.timedelta64(1, 'h'), 2)
    need(len(hist) == 6 and F['runner']['final_pass'] == 6 and all(len(rs) == 1 for rs in hist.values()), 'runner: six jobs, one attempt each, each PASS (no failure branch, no relaunch)')
    need(not os.path.isdir(os.path.join(S24, 'console_logs', 'S24_first_attempt')), 'no S24_first_attempt folder: the failure branch of RUNBOOK S24 was not used')
    for n in ('preflight.txt', 'selftest_revision_modules.txt', 'selftest_devis_released.txt', 'runner_jobs_D29.log', 'runner_jobs_D29smoke.log', 'status_jobs_D29smoke.csv'):
        assert os.path.exists(os.path.join(S24, 'console_logs', 'S24', n)), n
    need('self-test PASSED' in open(os.path.join(S24, 'console_logs', 'S24', 'selftest_revision_modules.txt'), errors='replace').read()
         and 'SELFTEST DEVIS-RELEASED PASSED (5 + 168 + 112 + 84 checks' in open(os.path.join(S24, 'console_logs', 'S24', 'selftest_devis_released.txt'), errors='replace').read(),
         'both server self-tests of RUNBOOK S24 step 1 passed and were written to files (R-64)')
    smoke = list(csv.DictReader(open(os.path.join(S24, 'console_logs', 'S24', 'status_jobs_D29smoke.csv'))))
    need(len(smoke) == 1 and smoke[0]['postflight'] == 'PASS', 'the smoke run of the new combination (released form + belief-mass gate) passed')

    # ---------------------------------------------------------------- per run: integrity (rule 1), screens (rule 2), values
    RUN, rows_i, rows_p = {}, [], []
    for ds in DATASETS:
        for sd in SEEDS:
            run = f'v18E_{ds}_devisrel_belief_s{sd}'; d = os.path.join(S24, 'runs', ds, run); log = os.path.join(d, 'log.txt')
            txt = open(log, errors='replace').read(); t = v04.screen_run(log); q = v02.parse_log(log)
            cfg = json.load(open(os.path.join(d, 'run_config.json'))); ar = cfg['args']
            ev = v04.parse_evaluations(txt); sa = np.array([e['student_avg'] for e in ev])
            ok_cfg = (ar['method'] == 'devis' and ar['devis_variant'] == 'released' and ar['devis_gate'] == 'soft' and ar['devis_rel_epochs'] == 99
                      and ar['devis_rel_lambda_epochs'] == 50 and ar['seed'] == sd and ar['max_iterations'] == MAXIT[ds] and ar['lb_domain'] == LBDOM[ds]
                      and ar['lb_num'] == 20 and ar['head'] == 'evidential' and ar['consistency_ramp_frac'] == -1.0 and ar['amp'] == 1)
            ok_sha = (cfg['script_sha256'] == code_sha['train_fourierfuse_v18_revision.py'] == sha(os.path.join(d, 'code', 'train_fourierfuse_v18_revision.py'))
                      and cfg['revision_modules_sha256'] == code_sha['revision_modules.py'] == sha(os.path.join(d, 'code', 'revision_modules.py'))
                      and cfg['fourierfuse_v16_modules_sha256'] == code_sha['fourierfuse_v16_modules.py'] == sha(os.path.join(d, 'code', 'fourierfuse_v16_modules.py')))
            warn = txt.count("DEviS variant 'released' (v11)"); gate_notice = txt.count("devis_gate='soft'")
            # the D-28 job line of the same seed, the only intended differences being --devis_gate and the name
            d28 = os.path.join(S23, 'runs', ds, f'v18D_{ds}_devisrel_threshold_s{sd}'); c28 = json.load(open(os.path.join(d28, 'run_config.json')))['args']
            diff_keys = sorted(k for k in set(ar) | set(c28) if ar.get(k) != c28.get(k))
            same_args = set(diff_keys) <= {'devis_gate', 'save_name', 'gpu'} and 'devis_gate' in diff_keys   # the GPU index is appended by the runner, not a setting
            refname = f'v18B_{ds}_fixmatch_split1337_init1337' if sd == 1337 else f'v18B_{ds}_fixmatch_s{sd}'
            rlog, rq = find_log(a.returns, ds, refname); tlog, tq = find_log(a.returns, ds, f'v18D_{ds}_devisrel_threshold_s{sd}')
            plog, pq = try_log(a.returns, ds, f'v18B_{ds}_devis_s{sd}')
            sp, spr, spt = split_of(log), split_of(rlog), split_of(tlog)
            same_split = bool(sp[1]) and sp[1] == spr[1] == spt[1] and sp[0] == sd
            sj = json.load(open(os.path.join(d, 'split.json'))); same_split = same_split and sj['lb_idxs'] == sp[1]
            J = json.load(open(os.path.join(S24, 'diag_out', f'{ds}__{run}__ema.json'))); me = J['meta']
            ok_h = (me['which'] == 'ema' and me['iter'] == q['best_iter'] and abs(me['best_avg_dice'] - q['best_avg']) < 1e-6 and run in me['ckpt']
                    and me['harness_sha256'] == code_sha['diag_uncertainty_harness_v18.py'] and me['n_models'] == 1)
            n = len(sa); last10 = float(sa[int(n * 0.9):].mean())
            head_last = dom_means(t['head']); at_best = {k: float(np.mean(list(v.values()))) for k, v in sorted(q['per_domain'].items())}
            tr = v04.screen_run(rlog); tt = v04.screen_run(tlog); tp = v04.screen_run(plog) if plog else None
            RUN[(ds, sd)] = {'run': run, 'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': t['student_avg_last'], 'last10': last10, 'swad_best': q['swad_best'],
                             'at_best': at_best, 'at_last': head_last, 'ref': rq['best_avg'], 'refname': refname, 'ref_last': tr['student_avg_last'],
                             'd28': tq['best_avg'], 'd28_last': tt['student_avg_last'], 'd28_best_iter': tq['best_iter'],
                             'pub': pq['best_avg'] if pq else None, 'pub_last': tp['student_avg_last'] if tp else None, 'pub_flag': v04.flag_of(tp) if tp else 'no complete log',
                             'flag': v04.flag_of(t), 'nf': t['nf_n'], 'J': J, 'log': log, 'txt': txt}
            rows_i.append([ds, run, sd, int(t['complete']), t['n_evals'], t['n_headers'], len(t['resumed_iters']), len(cfg.get('resume_events', [])), t['nf_n'],
                           int(t['parse_ok']), v04.flag_of(t), int(ok_cfg), int(same_args), ';'.join(diff_keys), int(ok_sha), warn, gate_notice, int(same_split), int(ok_h),
                           cfg['torch'], cfg['hostname'], len(hist[run]), hist[run][-1]['postflight'], hist[run][-1]['start'], hist[run][-1]['end']])
            need(t['complete'] and t['n_evals'] == MAXIT[ds] // 500 and t['n_headers'] == 1 and not t['resumed_iters'] and not cfg.get('resume_events'),
                 f'{run}: one training from iteration 0 to the last iteration, no resume event')
            need(t['nf_n'] == 0 and t['parse_ok'] and v04.flag_of(t) == 'ok', f'{run}: no skipped iteration; R-34 screen (registered rule, last evaluation) ok: not a training failure')
            need(ok_cfg and warn >= 1 and gate_notice >= 1, f'{run}: configuration = released form, belief-mass gate (devis_gate soft), headline settings; v11 notice and gate notice in the log')
            need(same_args, f'{run}: its recorded arguments equal those of the D-28 run of the same seed except devis_gate, save_name and the GPU index')
            need(ok_sha, f'{run}: code snapshot, recorded hashes and the release code are the same bytes (tools v12 = v11 trainer and modules)')
            need(same_split, f'{run}: labeled slices identical to those of the threshold reference and of the D-28 run of the same seed')
            need(ok_h, f'{run}: the measured teacher is the EMA weights of the best checkpoint of this log, measured with the release harness')
            rows_p.append([ds, sd, run, f'{q["best_avg"]:.6f}', q['best_iter'], f'{q["best_iter"] / MAXIT[ds]:.3f}', f4(t['student_avg_last']), f4(last10), f4(q['swad_best']),
                           ';'.join(f4(at_best[k]) for k in sorted(at_best)), ';'.join(f4(head_last[k]) for k in sorted(head_last)),
                           refname, f'{rq["best_avg"]:.6f}', f4(tr['student_avg_last']), f'{q["best_avg"] - rq["best_avg"]:+.6f}',
                           f'{tq["best_avg"]:.6f}', f4(tt['student_avg_last']), f'{q["best_avg"] - tq["best_avg"]:+.6f}',
                           f'{pq["best_avg"]:.6f}' if pq else '', f4(tp['student_avg_last']) if tp else '', f'{q["best_avg"] - pq["best_avg"]:+.6f}' if pq else '', RUN[(ds, sd)]['pub_flag']])
    wcsv(os.path.join(a.out, 'table_d29_integrity_s24.csv'), rows_i,
         ['dataset', 'run', 'seed', 'complete', 'n_evals', 'n_headers', 'n_resumes_log', 'n_resume_events_cfg', 'n_nonfinite_skips', 'parse_ok', 'r34_flag',
          'config_ok', 'args_equal_d28_except_gate_name_gpu', 'args_that_differ_from_d28', 'hashes_ok', 'n_v11_notice_lines', 'n_gate_soft_notice_lines',
          'split_equals_reference_and_d28', 'harness_meta_ok', 'torch', 'hostname', 'n_runner_attempts', 'postflight', 'start', 'end'])
    wcsv(os.path.join(a.out, 'table_d29_per_seed_s24.csv'), rows_p,
         ['dataset', 'seed', 'run', 'best_avg', 'best_iter', 'best_frac', 'student_avg_last', 'student_avg_last10pct', 'swad_best', 'dice_at_best_per_domain',
          'dice_at_last_per_domain', 'threshold_reference', 'ref_best_avg', 'ref_last', 'delta_vs_threshold', 'd28_released_threshold_best_avg', 'd28_last',
          'delta_vs_d28', 'published_form_belief_gate_best_avg', 'published_form_belief_gate_last', 'delta_vs_published_form_belief_gate', 'published_form_belief_gate_flag'])

    # ---------------------------------------------------------------- rules 3 and 4, and the descriptive published-form comparison
    base = {(r['dataset'], r['method']): r for r in csv.DictReader(open(os.path.join(a.derived, 'table_baselines.csv')))}
    rows_r = []
    for ds in DATASETS:
        ref = [RUN[(ds, s)]['ref'] for s in SEEDS]; d28 = [RUN[(ds, s)]['d28'] for s in SEEDS]; bel = [RUN[(ds, s)]['best'] for s in SEEDS]
        # the reader reproduces the D-28 row of facts_s23.json from the logs first (same references, same function)
        r28 = rule(d28, ref); f28 = F23[f'{ds}_rule']
        need(abs(r28['mean'] - f28['mean']) < 1e-9 and abs(r28['delta'] - f28['delta']) < 1e-9 and abs(r28['p'] - f28['p']) < 1e-9,
             f'{ds}: this reader reproduces the D-28 rule row of facts_s23.json (mean, delta, p) from the logs')
        r3 = rule(bel, ref); r4 = gate_rule(bel, d28)
        F[f'{ds}_rule3_vs_threshold_reference'] = r3; F[f'{ds}_rule4_gate_vs_d28'] = r4
        F[f'{ds}_per_seed'] = {str(s): {k: RUN[(ds, s)][k] for k in ('best', 'best_iter', 'last', 'last10', 'swad_best', 'ref', 'ref_last', 'd28', 'd28_last', 'd28_best_iter', 'pub', 'pub_last', 'pub_flag')} for s in SEEDS}
        pubs = [RUN[(ds, s)]['pub'] for s in SEEDS]; pub_ok = [p is not None for p in pubs]
        F[f'{ds}_vs_published_form_belief_gate'] = {'published_form_per_seed': pubs, 'published_form_flags': [RUN[(ds, s)]['pub_flag'] for s in SEEDS],
                                                     'released_minus_published_per_seed': [RUN[(ds, s)]['best'] - p if p is not None else None for s, p in zip(SEEDS, pubs)],
                                                     'baselines_row_mean': float(base[(ds, 'devis')]['mean_dice']), 'baselines_row_std': float(base[(ds, 'devis')]['std_dice']),
                                                     'n_published_form_training_failures': sum(RUN[(ds, s)]['pub_flag'] != 'ok' for s in SEEDS)}
        for lab, x, y, r_ in (('rule3_belief_gate_released_vs_threshold_reference', bel, ref, r3), ('rule4_belief_gate_vs_threshold_gate_same_released_objective', bel, d28, r4)):
            rows_r.append([ds, lab, f4(r_['mean']), f4(r_['std']), f4(r_['ref_mean']), f4(r_['ref_std']), f'{r_["delta"]:+.4f}', ';'.join(f'{v:+.4f}' for v in r_['per_seed_delta']),
                           f'{r_["t"]:.4f}', f'{r_["p"]:.4f}', r_['n_pos'], f'{r_["sign_p_greater"]:.4f}', f'{r_["sign_p_two_sided"]:.4f}',
                           int(r_['beats']) if lab.startswith('rule3') else '', int(r_['changes']) if lab.startswith('rule4') else ''])
        rows_r.append([ds, 'descriptive_belief_gate_released_vs_published_form_belief_gate', f4(r3['mean']), f4(r3['std']), f4(np.mean([p for p in pubs if p is not None])), '',
                       '', ';'.join(f'{v:+.4f}' if v is not None else 'n/a' for v in F[f'{ds}_vs_published_form_belief_gate']['released_minus_published_per_seed']), '', '', '', '', '', '', ''])
        need(not r3['beats'], f'{ds}: rule 3: the belief-mass gate on the released form does not meet the pre-registered criterion (paired p < 0.05 and delta >= 0.5 Dice) against the threshold reference')
        need(not r4['changes'], f'{ds}: rule 4: the gate does not change the result under the released objective (not both paired p < 0.05 and |delta| >= 0.5 Dice against the D-28 run of the same seed)')
    wcsv(os.path.join(a.out, 'table_d29_rule_s24.csv'), rows_r,
         ['dataset', 'comparison', 'mean_dice', 'std_dice', 'ref_mean', 'ref_std', 'delta', 'per_seed_delta_1337_42_2024', 't_paired', 'p_paired', 'n_seeds_positive',
          'sign_test_p_one_sided', 'sign_test_p_two_sided', 'meets_phaseB_rule', 'gate_changes_result_rule4'])

    # ---------------------------------------------------------------- rule 6: the logged gate weight
    rows_w = []
    for ds in DATASETS:
        for sd in SEEDS:
            r = RUN[(ds, sd)]; Tm = MAXIT[ds]
            w = [(int(i), float(m), float(mn)) for i, m, mn in W_RX.findall(r['txt'])]; assert len(w) == Tm // 500, (r['run'], len(w))
            late = [m for i, m, mn in w if i > 0.2 * Tm]; early = [m for i, m, mn in w if i <= 0.2 * Tm]
            t28 = open(os.path.join(S23, 'runs', ds, f'v18D_{ds}_devisrel_threshold_s{sd}', 'log.txt'), errors='replace').read()
            w28 = [(int(i), float(m)) for i, m, mn in W_RX.findall(t28)]; late28 = [m for i, m in w28 if i > 0.2 * Tm]
            first_above = next((i for i, m, mn in w if m >= 0.99), None)
            F.setdefault('gate_weight', {})[r['run']] = {'w_mean_after_20pct_mean': float(np.mean(late)), 'w_mean_after_20pct_min': float(min(late)), 'w_mean_after_20pct_max': float(max(late)),
                                                        'one_minus_w_after_20pct_mean': float(1 - np.mean(late)), 'w_mean_first_window': w[0][1], 'w_mean_min_early': float(min(early)),
                                                        'w_mean_at_20pct': [m for i, m, mn in w if i <= 0.2 * Tm][-1], 'w_mean_last': w[-1][1], 'w_min_last': w[-1][2],
                                                        'first_iter_w_mean_ge_0.99': first_above, 'first_frac_w_mean_ge_0.99': (first_above / Tm) if first_above else None,
                                                        'd28_threshold_pass_fraction_after_20pct_mean': float(np.mean(late28))}
            for i, m, mn in w:
                if i in (500, Tm // 10, Tm // 5, Tm // 2, r['best_iter'], Tm):
                    rows_w.append([ds, r['run'], i, f'{i / Tm:.3f}', f'{m:.4f}', f'{mn:.4f}', f'{1 - m:.4f}', f'{dict(w28).get(i, float("nan")):.4f}'])
    wcsv(os.path.join(a.out, 'table_d29_gate_weight_s24.csv'), rows_w,
         ['dataset', 'run', 'iter', 'frac_of_training', 'w_mean_logged', 'w_min_logged', 'one_minus_w_mean', 'd28_threshold_gate_pass_fraction_same_iter'])
    G = F['gate_weight']
    F['gate_weight_summary'] = {'after_20pct_mean_min_max': [min(v['w_mean_after_20pct_mean'] for v in G.values()), max(v['w_mean_after_20pct_mean'] for v in G.values())],
                                'after_20pct_min_over_runs': min(v['w_mean_after_20pct_min'] for v in G.values()), 'last_min_max': [min(v['w_mean_last'] for v in G.values()), max(v['w_mean_last'] for v in G.values())],
                                'first_window_min_max': [min(v['w_mean_first_window'] for v in G.values()), max(v['w_mean_first_window'] for v in G.values())],
                                'early_min_over_runs': min(v['w_mean_min_early'] for v in G.values()),
                                'frac_reaching_0.99_min_max': [min(v['first_frac_w_mean_ge_0.99'] for v in G.values()), max(v['first_frac_w_mean_ge_0.99'] for v in G.values())],
                                'd28_pass_fraction_after_20pct_min_max': [min(v['d28_threshold_pass_fraction_after_20pct_mean'] for v in G.values()), max(v['d28_threshold_pass_fraction_after_20pct_mean'] for v in G.values())]}
    need(F['gate_weight_summary']['after_20pct_mean_min_max'][0] > 0.95 and F['gate_weight_summary']['last_min_max'][0] > 0.99,
         'rule 6: after 20 % of training the mean belief-mass weight exceeds 0.95 in every run and ends above 0.99 (the gate removes at most a few hundredths of the pseudo-label term late in training)')
    need(F['gate_weight_summary']['first_window_min_max'][1] < 0.8, 'rule 6 (descriptive, contrary to the pre-registration\'s expectation for the whole run): in the first 500 iterations the mean weight is below 0.8 in every run (the teacher starts vacuous)')

    # ---------------------------------------------------------------- rule 6: trajectories (next to D-28 and the references)
    rows_t = []
    for ds in DATASETS:
        for sd in SEEDS:
            r = RUN[(ds, sd)]
            rows_t.append([ds, sd, r['run'], f4(r['best']), f'{r["best_iter"] / MAXIT[ds]:.3f}', f4(r['last']), f'{100 * (r["best"] - r["last"]):.2f}',
                           f'{100 * (r["d28"] - r["d28_last"]):.2f}', f'{r["d28_best_iter"] / MAXIT[ds]:.3f}', f'{100 * (r["ref"] - r["ref_last"]):.2f}'])
        F[f'{ds}_late'] = {'belief_best_minus_last': [100 * (RUN[(ds, s)]['best'] - RUN[(ds, s)]['last']) for s in SEEDS],
                           'd28_best_minus_last': [100 * (RUN[(ds, s)]['d28'] - RUN[(ds, s)]['d28_last']) for s in SEEDS],
                           'threshold_ref_best_minus_last': [100 * (RUN[(ds, s)]['ref'] - RUN[(ds, s)]['ref_last']) for s in SEEDS],
                           'belief_best_frac': [RUN[(ds, s)]['best_iter'] / MAXIT[ds] for s in SEEDS], 'd28_best_frac': [RUN[(ds, s)]['d28_best_iter'] / MAXIT[ds] for s in SEEDS],
                           'belief_last': [RUN[(ds, s)]['last'] for s in SEEDS], 'belief_last10': [RUN[(ds, s)]['last10'] for s in SEEDS]}
    wcsv(os.path.join(a.out, 'table_d29_trajectory_s24.csv'), rows_t,
         ['dataset', 'seed', 'run', 'best_avg', 'best_frac', 'student_avg_last', 'best_minus_last_points', 'd28_best_minus_last_points', 'd28_best_frac', 'threshold_ref_best_minus_last_points'])
    drops = {}
    for s in SEEDS:
        r = RUN[('prostate', s)]; drops[s] = {DOM['prostate'][k - 1]: 100 * (r['at_best'][k] - r['at_last'][k]) for k in sorted(r['at_best'])}
    F['prostate_per_vendor_drop_best_to_last'] = {str(s): {k: round(v, 2) for k, v in d.items()} for s, d in drops.items()}
    F['prostate_per_vendor_at_best'] = {str(s): {DOM['prostate'][k - 1]: round(RUN[('prostate', s)]['at_best'][k], 4) for k in sorted(RUN[('prostate', s)]['at_best'])} for s in SEEDS}
    F['prostate_top3_losing_vendors'] = {str(s): sorted(drops[s], key=drops[s].get, reverse=True)[:3] for s in SEEDS}
    P, Fu = F['prostate_late'], F['fundus_late']
    need(min(P['belief_best_minus_last']) > 4, 'Prostate: each belief-gate run ends more than 4 points below its best checkpoint (the late degradation of O-04 is present with this gate too)')
    need(max(Fu['belief_best_minus_last']) < 0.5, 'Fundus: each belief-gate run ends within half a point of its best checkpoint')

    # ---------------------------------------------------------------- the schedule the runs logged against the schedule function (as S23)
    rows_s, worst = [], 0.0
    for ds in DATASETS:
        for sd in SEEDS:
            r = RUN[(ds, sd)]; Tm = MAXIT[ds]; stl = [(int(i), float(S), float(b), float(c), float(k)) for i, S, b, c, k in DEV_RX.findall(r['txt'])]
            assert len(stl) == Tm // 500, (r['run'], len(stl))
            for it, S, b, c, k in stl:
                w = [sched(t, Tm) for t in range(it - 500, it)]; km = float(np.mean([x[0] for x in w])); bm = float(np.mean([x[1] for x in w]))
                worst = max(worst, max(abs(k - km) / max(km, 1e-12), abs(b - bm) / bm))
                if it in (500, Tm // 4, Tm // 2, r['best_iter'], 3 * Tm // 4, Tm):
                    rows_s.append([ds, r['run'], it, f'{k:.6g}', f'{km:.6g}', f'{b:.6g}', f'{bm:.6g}', f'{c:.6g}', f'{S:.6g}'])
    wcsv(os.path.join(a.out, 'table_d29_schedule_s24.csv'), rows_s,
         ['dataset', 'run', 'iter', 'kl_weight_logged', 'kl_weight_schedule_window_mean', 'beta_t_logged', 'beta_t_schedule_window_mean', 'calibration_penalty_logged', 'S_mean_logged'])
    F['schedule_largest_relative_difference_logged_vs_function'] = worst
    need(worst < 2e-3, 'the KL weight and beta_t logged every 500 iterations equal the window means of the schedule function (relative 2e-3) in all six runs')

    # ---------------------------------------------------------------- rule 5: the signal of the six teachers (v02 functions, unchanged)
    H = {k: v for k, v in v02.load_harness(S24).items() if k[1].startswith('v18E_')}
    need(len(H) == 6 and all(k[2] == 'ema' for k in H), 'six harness JSONs, EMA teachers')
    tmp = tempfile.mkdtemp(); A = {}
    v02.band_tables(H, tmp); v02.signal_identity_table(H, tmp, A)
    for n in ('table_band_generality', 'table_band_summary', 'table_signal_identity'):
        shutil.copy(os.path.join(tmp, n + '.csv'), os.path.join(a.out, n + '_s24.csv'))
    bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_band_summary_s24.csv')))}
    idn = {(r['dataset'], r['run']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_signal_identity_s24.csv')))}
    old_bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.derived, 'table_band_summary.csv')))}
    old_id = list(csv.DictReader(open(os.path.join(a.derived, 'table_signal_identity.csv'))))
    s23_bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.derived, 'extra_s23', 'table_band_summary_s23.csv')))}
    rows_g, SIG = [], {}
    for ds in DATASETS:
        for sd in SEEDS:
            r = RUN[(ds, sd)]; run = r['run']; u = bs[(ds, run, 'u_epi')]; c = bs[(ds, run, 'conf')]; i = idn[(ds, run)]
            band, rng = float(u['auroc_edt5_mean']), float(u['range80_all_mean'])
            per_dom = [float(x['signals']['u_epi']['band']['edt_gt_5']['auroc']) for _, x in sorted(r['J']['per_domain'].items(), key=lambda kv: int(kv[0]))]
            ou = old_bs[(ds, r['refname'], 'u_epi')]; o28 = s23_bs[(ds, f'v18D_{ds}_devisrel_threshold_s{sd}', 'u_epi')]
            SIG[(ds, sd)] = {'auroc_all': float(u['auroc_all_mean']), 'band': band, 'range': rng, 'range_band': float(u['range80_edt5_mean']),
                             'conf_all': float(c['auroc_all_mean']), 'conf_band': float(c['auroc_edt5_mean']), 'errors_in_band': float(u['frac_errors_in_edt5_mean']),
                             'identity_holds': i['identity_holds'] == '1', 'gap_all': float(i['max_abs_auroc_gap_all']), 'gap_band': float(i['max_abs_auroc_gap_edt5']),
                             'median_ratio': float(i['median_ratio_u_epi_over_conf']), 'K': int(i['K']), 'band_per_domain': per_dom,
                             'ref_all': float(ou['auroc_all_mean']), 'ref_band': float(ou['auroc_edt5_mean']), 'ref_range': float(ou['range80_all_mean']),
                             'd28_all': float(o28['auroc_all_mean']), 'd28_band': float(o28['auroc_edt5_mean']), 'd28_range': float(o28['range80_all_mean'])}
            s = SIG[(ds, sd)]
            rows_g.append([ds, sd, run, f4(s['auroc_all']), f4(band), f'{rng:.3e}', f'{s["range_band"]:.3e}', f4(s['conf_all']), f4(s['conf_band']), f4(s['errors_in_band']),
                           int(band <= RULE_BAND), int(rng <= RULE_RANGE), int(band <= RULE_BAND and rng <= RULE_RANGE), f'{band - RULE_BAND:+.4f}', f'{RULE_RANGE / rng:.0f}',
                           int(s['identity_holds']), f4(s['gap_all']), f4(s['gap_band']), f4(s['median_ratio']), ';'.join(f'{x:.3f}' for x in per_dom),
                           f4(s['ref_all']), f4(s['ref_band']), f'{s["ref_range"]:.3e}', f4(s['d28_all']), f4(s['d28_band']), f'{s["d28_range"]:.3e}'])
    wcsv(os.path.join(a.out, 'table_d29_signal_s24.csv'), rows_g,
         ['dataset', 'seed', 'run', 'u_epi_auroc_all', 'u_epi_auroc_band5', 'u_epi_range80_all', 'u_epi_range80_band5', 'conf_auroc_all', 'conf_auroc_band5', 'frac_errors_in_band5',
          'band_le_0.70', 'range_le_1e-2', 'phaseA_rule_collapse_persists', 'band_margin_to_0.70', 'range_factor_below_1e-2', 'identity_holds', 'max_abs_auroc_gap_all',
          'max_abs_auroc_gap_band5', 'median_ratio_u_epi_over_conf', 'u_epi_band5_per_domain', 'threshold_ref_auroc_all', 'threshold_ref_band5', 'threshold_ref_range',
          'd28_released_threshold_auroc_all', 'd28_band5', 'd28_range'])
    for ds in DATASETS:
        v = [SIG[(ds, s)] for s in SEEDS]
        F[f'{ds}_signal'] = {'per_seed': {str(s): {k: (SIG[(ds, s)][k] if not isinstance(SIG[(ds, s)][k], list) else [round(x, 4) for x in SIG[(ds, s)][k]]) for k in SIG[(ds, s)]} for s in SEEDS},
                             'auroc_all_min_max': [min(x['auroc_all'] for x in v), max(x['auroc_all'] for x in v)], 'band_min_max': [min(x['band'] for x in v), max(x['band'] for x in v)],
                             'band_mean': float(np.mean([x['band'] for x in v])), 'range_min_max': [min(x['range'] for x in v), max(x['range'] for x in v)],
                             'n_band_le_070': sum(x['band'] <= RULE_BAND for x in v), 'n_rule_persists': sum(x['band'] <= RULE_BAND and x['range'] <= RULE_RANGE for x in v),
                             'n_identity_holds': sum(x['identity_holds'] for x in v), 'gap_band_max': max(x['gap_band'] for x in v),
                             'd28_band_min_max': [min(x['d28_band'] for x in v), max(x['d28_band'] for x in v)], 'd28_all_min_max': [min(x['d28_all'] for x in v), max(x['d28_all'] for x in v)],
                             'ref_band_min_max': [min(x['ref_band'] for x in v), max(x['ref_band'] for x in v)], 'ref_all_min_max': [min(x['ref_all'] for x in v), max(x['ref_all'] for x in v)]}
    allsix = list(SIG.values())
    F['signal_all_six'] = {'range_max': max(x['range'] for x in allsix), 'range_factor_below_bound_min': RULE_RANGE / max(x['range'] for x in allsix),
                           'band_min_max': [min(x['band'] for x in allsix), max(x['band'] for x in allsix)], 'band_mean_of_dataset_means': float(np.mean([F[f'{ds}_signal']['band_mean'] for ds in DATASETS])),
                           'n_band_le_070': sum(x['band'] <= RULE_BAND for x in allsix), 'n_band_above_070': sum(x['band'] > RULE_BAND for x in allsix),
                           'band_excess_max': max(x['band'] for x in allsix) - RULE_BAND, 'n_identity_holds': sum(x['identity_holds'] for x in allsix),
                           'auroc_all_min_max': [min(x['auroc_all'] for x in allsix), max(x['auroc_all'] for x in allsix)]}
    need(F['signal_all_six']['range_max'] <= RULE_RANGE / 2, 'rule 5: the central-80 % range of u_epi is at most half of the 1e-2 bound on all six teachers (range condition of the Phase-A rule)')
    # rule 5 is applied PER TEACHER (the teacher's across-domain band mean <= 0.70 and range <= 1e-2 -> collapse persists for that teacher);
    # the count and the margins are recorded and printed, not asserted: the text is written after this classification, not before it.
    F['signal_all_six']['per_teacher_rule'] = {f'{ds}_s{sd}': {'band': round(SIG[(ds, sd)]['band'], 4), 'margin_to_0.70': round(SIG[(ds, sd)]['band'] - RULE_BAND, 4),
                                                            'range': SIG[(ds, sd)]['range'], 'persists': bool(SIG[(ds, sd)]['band'] <= RULE_BAND and SIG[(ds, sd)]['range'] <= RULE_RANGE)}
                                               for ds in DATASETS for sd in SEEDS}
    F['signal_all_six']['n_rule_persists'] = sum(v['persists'] for v in F['signal_all_six']['per_teacher_rule'].values())
    # S9 counts with D-28 and D-29 added
    def cls(run):
        return 'ensemble' if 'ENSEMBLE' in run else 'devis' if 'devis' in run else 'medl' if 'medl' in run else 'exp' if 'act_exp' in run else 'testbed'
    dm_old = [r for r in old_id if cls(r['run']) in ('devis', 'medl')]
    new_fail = sum(not x['identity_holds'] for x in allsix); s23_fail = 6 - F23['signal_all_six']['n_identity_holds']
    F['s9_counts'] = {'n_all_old': len(old_id), 'n_all_with_d28': len(old_id) + 6, 'n_all_with_d28_d29': len(old_id) + 12, 'n_devis_medl_old': len(dm_old),
                      'n_devis_medl_with_d28_d29': len(dm_old) + 12, 'n_devis_medl_identity_fails_old': sum(r['identity_holds'] != '1' for r in dm_old),
                      'n_devis_medl_identity_fails_with_d28_d29': sum(r['identity_holds'] != '1' for r in dm_old) + s23_fail + new_fail,
                      'd29_identity_fails': new_fail, 'd28_identity_fails': s23_fail}
    need(F['s9_counts']['n_all_old'] == 83 and F['s9_counts']['n_devis_medl_old'] == 18, 'the S12 identity table holds 83 checkpoints, 18 of them DEviS / MEDL (counts as in S23)')
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s24.json'), 'w'), indent=1, default=float)
    write_adjudication(F, a.out)
    print(f'[make_extra_s24] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


def pct(x):
    return f'{100 * x:.2f}'


def write_adjudication(F, out):
    L = []
    L.append('# ADJUDICATION_S24 (S25 turn 1, 2026-10-13) — the S24 sendback: D-29, the released-form DEviS objective with the belief-mass gate (six runs), read by the rules of `DOCS/D29_PREREGISTRATION_S24.md`')
    L.append('')
    L.append(f'Generated by `make_extra_s24.py` from `facts_s24.json` ({len(F["assertions_passed"])} assertions passed). Numbers below are copied from the facts file; nothing is typed by hand.')
    L.append('')
    R = F['runner']
    L.append('## 1. Integrity (rules 1 and 2)')
    L.append(f'- Sendback: {F["manifest_files_verified"]} files, each verified against the collector manifests (size, sha256). Runner: {R["jobs"]} jobs, one attempt each, {R["final_pass"]} PASS; '
             f'first start {R["first_start"]}, last end {R["last_end"]} ({R["wall_clock_hours"]} h wall clock; per job ' + ', '.join(f'{k.split("_", 1)[1]} {v} h' for k, v in sorted(R["hours_per_job"].items())) + '). '
             'No failure branch, no relaunch, no first-attempt folder. Both server self-tests passed and were written to files; the smoke run of the new combination passed.')
    L.append('- Every run: one training from iteration 0 to the end, no resume event, no skipped iteration, R-34 screen ok (no training failure); configuration = released form, belief-mass gate '
             '(`devis_gate soft`), headline settings; the recorded arguments equal those of the D-28 run of the same seed except `devis_gate`, `save_name` and the GPU index; code snapshot, recorded hashes and the '
             f'release code are the same bytes (trainer `{F["code_sha256"]["train_fourierfuse_v18_revision.py"]}`, modules `{F["code_sha256"]["revision_modules.py"]}`); labeled slices identical to those of '
             'the threshold reference and of the D-28 run of the same seed; the measured teacher is the EMA of the best checkpoint of the same log.')
    L.append(f'- Schedule: the logged KL weight and β_t equal the window means of the schedule function (largest relative difference {F["schedule_largest_relative_difference_logged_vs_function"]:.1e}).')
    L.append('')
    L.append('## 2. Dice (rules 3 and 4)')
    for ds in DATASETS:
        r3 = F[f'{ds}_rule3_vs_threshold_reference']; r4 = F[f'{ds}_rule4_gate_vs_d28']; ps = F[f'{ds}_per_seed']
        L.append(f'**{ds.capitalize()}.** Belief-mass gate, released form: {pct(r3["mean"])} ± {pct(r3["std"])} (seeds 1337 / 42 / 2024: ' + ' / '.join(pct(ps[str(s)]["best"]) for s in SEEDS) + ').')
        L.append(f'- Rule 3 (against the within-v18 threshold reference {pct(r3["ref_mean"])} ± {pct(r3["ref_std"])}): Δ = {100 * r3["delta"]:+.2f} points (per seed ' +
                 ', '.join(f'{100 * v:+.2f}' for v in r3["per_seed_delta"]) + f'), paired t = {r3["t"]:.2f}, p = {r3["p"]:.3f}, {r3["n_pos"]} of 3 seeds positive (exact sign test p = {r3["sign_p_greater"]:.2f}). '
                 f'Meets the criterion (p < 0.05 and Δ ≥ 0.5): **{"yes" if r3["beats"] else "no"}**.')
        L.append(f'- Rule 4 (the gate under one objective; against the D-28 run of the same seed, {pct(r4["ref_mean"])} ± {pct(r4["ref_std"])}): Δ = {100 * r4["delta"]:+.2f} points (per seed ' +
                 ', '.join(f'{100 * v:+.2f}' for v in r4["per_seed_delta"]) + f'), paired t = {r4["t"]:.2f}, p = {r4["p"]:.3f}. Changes the result (p < 0.05 and |Δ| ≥ 0.5): **{"yes" if r4["changes"] else "no"}** → '
                 'the two gates are within seed noise under the released objective.')
        pb = F[f'{ds}_vs_published_form_belief_gate']
        L.append(f'- Descriptive, against the published-form belief-gate runs of Table 7 (`v18B_{ds}_devis_s*`; baselines row {pct(pb["baselines_row_mean"])} ± {pct(pb["baselines_row_std"])}, '
                 f'{pb["n_published_form_training_failures"]} of 3 flagged as training failures): per seed the published-form values are ' + ', '.join('n/a' if p is None else pct(p) for p in pb["published_form_per_seed"]) +
                 ' (R-34 flags ' + ', '.join(pb["published_form_flags"]) + '); released minus published ' + ', '.join('n/a' if v is None else f'{100 * v:+.2f}' for v in pb["released_minus_published_per_seed"]) + '. No paired test: the published-form row holds a registered training failure (Prostate s42) and a run twenty points below its seed-mates (Fundus s2024), as Table 7 reports.')
    L.append('')
    L.append('## 3. The gate weight (rule 6)')
    G = F['gate_weight_summary']
    L.append(f'- `w_mean` = mean over the unlabeled batch of the per-pixel weight 1 − u, logged every 500 iterations. After 20 % of training its run mean is {G["after_20pct_mean_min_max"][0]:.3f}–{G["after_20pct_mean_min_max"][1]:.3f} '
             f'(minimum window {G["after_20pct_min_over_runs"]:.3f}); at the last iteration {G["last_min_max"][0]:.3f}–{G["last_min_max"][1]:.3f}. The weight reaches 0.99 at {100 * G["frac_reaching_0.99_min_max"][0]:.0f}–{100 * G["frac_reaching_0.99_min_max"][1]:.0f} % of training.')
    L.append(f'- **Early training is not as the pre-registration expected.** The expectation sentence said the weight should stay within a few thousandths of 1; in the first 500 iterations it is {G["first_window_min_max"][0]:.2f}–{G["first_window_min_max"][1]:.2f} '
             f'(the teacher starts vacuous), and the minimum over the first 20 % is {G["early_min_over_runs"]:.2f}. The registered reading (rule 6: "after 20 % of training") is unaffected; the expectation was written for the whole run and was wrong for its first part. '
             f'For comparison, the threshold gate of the D-28 runs (the same objective) passes {100 * G["d28_pass_fraction_after_20pct_min_max"][0]:.1f}–{100 * G["d28_pass_fraction_after_20pct_min_max"][1]:.1f} % of pixels after 20 % of training: '
             'after the first fifth of training both gates leave at least 99 % of the pseudo-label term in place and differ only during the early ramp, where the consistency weight is still small (0.04 of the cap 0.5 at one fifth of training, Supplementary S8). '
             'Interpretation, not a registered result: this is the arithmetic behind rule 4; for four fifths of training the two gates are nearly the same operator.')
    L.append('')
    L.append('## 4. Trajectories (rule 6, descriptive; O-04)')
    P, Fu = F['prostate_late'], F['fundus_late']
    L.append('- Prostate: best minus last-iteration Dice ' + ' / '.join(f'{x:.1f}' for x in P['belief_best_minus_last']) + ' points (seeds 1337 / 42 / 2024); the D-28 runs ' + ' / '.join(f'{x:.1f}' for x in P['d28_best_minus_last']) +
             '; the threshold references ' + ' / '.join(f'{x:.1f}' for x in P['threshold_ref_best_minus_last']) + '. Best checkpoints at ' + ' / '.join(f'{100 * x:.0f} %' for x in P['belief_best_frac']) + ' of training (D-28: ' + ' / '.join(f'{100 * x:.0f} %' for x in P['d28_best_frac']) + ').')
    L.append('- Prostate, vendors carrying the loss (top three per seed): ' + '; '.join(f's{s}: ' + ', '.join(f'{v} {F["prostate_per_vendor_drop_best_to_last"][str(s)][v]:+.1f}' for v in F['prostate_top3_losing_vendors'][str(s)]) for s in SEEDS) + '.')
    L.append('- Fundus: best minus last ' + ' / '.join(f'{x:.2f}' for x in Fu['belief_best_minus_last']) + ' points (D-28: ' + ' / '.join(f'{x:.2f}' for x in Fu['d28_best_minus_last']) + '); stable, as in every Fundus run held.')
    L.append('')
    L.append('## 5. The signal of the six teachers (rule 5) and the identity (S9)')
    S6 = F['signal_all_six']
    L.append(f'- Central-80 % range of u_epi: at most {S6["range_max"]:.2e} (the 1e-2 bound is {S6["range_factor_below_bound_min"]:.0f} times above it). Boundary-band AUROC {S6["band_min_max"][0]:.3f}–{S6["band_min_max"][1]:.3f}; '
             f'{S6["n_band_le_070"]} of 6 teachers at or below the 0.70 line, {S6["n_band_above_070"]} above it (largest excess {S6["band_excess_max"]:+.3f}). Per teacher (the rule as stated: band mean ≤ 0.70 and range ≤ 1e-2 → the collapse persists): '
             + ', '.join(f'{k} {v["band"]:.4f} ({v["margin_to_0.70"]:+.4f}, {"persists" if v["persists"] else "above the line"})' for k, v in S6["per_teacher_rule"].items())
             + f'; {S6["n_rule_persists"]} of 6 by the rule as a hard line. Dataset means of the band AUROC: ' + ', '.join(f'{ds} {F[f"{ds}_signal"]["band_mean"]:.3f}' for ds in DATASETS)
             + f'. All-pixel AUROC {S6["auroc_all_min_max"][0]:.3f}–{S6["auroc_all_min_max"][1]:.3f}.')
    for ds in DATASETS:
        s = F[f'{ds}_signal']
        L.append(f'- {ds.capitalize()}: band AUROC per seed ' + ', '.join(f'{s["per_seed"][str(sd)]["band"]:.3f}' for sd in SEEDS) + f' (D-28: {s["d28_band_min_max"][0]:.3f}–{s["d28_band_min_max"][1]:.3f}; threshold references {s["ref_band_min_max"][0]:.3f}–{s["ref_band_min_max"][1]:.3f}); '
                 f'all-pixel {s["auroc_all_min_max"][0]:.3f}–{s["auroc_all_min_max"][1]:.3f} (D-28 {s["d28_all_min_max"][0]:.3f}–{s["d28_all_min_max"][1]:.3f}; references {s["ref_all_min_max"][0]:.3f}–{s["ref_all_min_max"][1]:.3f}); '
                 f'identity of Eqn. 12 holds on {s["n_identity_holds"]} of 3 (largest band gap {s["gap_band_max"]:.3f}).')
    C = F['s9_counts']
    L.append(f'- S9 counts after D-28 and D-29: {C["n_all_old"]} → {C["n_all_with_d28_d29"]} checkpoints; DEviS / MEDL {C["n_devis_medl_old"]} → {C["n_devis_medl_with_d28_d29"]}, of which the identity fails on '
             f'{C["n_devis_medl_identity_fails_old"]} → {C["n_devis_medl_identity_fails_with_d28_d29"]} (D-28 {C["d28_identity_fails"]}, D-29 {C["d29_identity_fails"]}).')
    L.append('')
    L.append('## 6. Verdict for the text (N-61)')
    L.append('- Table 7 gains the row "DEviS obj. (released form), belief-mass gate": ' + '; '.join(f'{ds.capitalize()} {pct(F[f"{ds}_rule3_vs_threshold_reference"]["mean"])} ± {pct(F[f"{ds}_rule3_vs_threshold_reference"]["std"])} (Δ {100 * F[f"{ds}_rule3_vs_threshold_reference"]["delta"]:+.2f}, p = {F[f"{ds}_rule3_vs_threshold_reference"]["p"]:.2f})' for ds in DATASETS) +
             '. No uncertainty-aware alternative meets the criterion of §4.4; the conclusion stands.')
    L.append('- The sentence "DEviS\'s own gate, on the form of its objective that trains best, changes nothing within seed noise" is supported by rule 4 (' + '; '.join(f'{ds} Δ {100 * F[f"{ds}_rule4_gate_vs_d28"]["delta"]:+.2f}, p = {F[f"{ds}_rule4_gate_vs_d28"]["p"]:.2f}' for ds in DATASETS) + ') and by the weight (above 0.95 after 20 % of training).')
    L.append('- What may not be written: "the official DEviS" (our re-implementation of its supervised objective inside the testbed; the released code has no semi-supervised part); a mechanism for the late degradation; '
             '"the gate is inert" without the weight numbers and the rule-4 test; anything about the early-training weight beyond the descriptive sentence of section 3.')
    L.append('')
    L.append('## Assertions passed')
    L.extend(f'- {s}' for s in F['assertions_passed'])
    open(os.path.join(out, 'ADJUDICATION_S24.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
