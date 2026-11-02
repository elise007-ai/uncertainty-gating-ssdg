#!/usr/bin/env python3
"""
make_selection_s25.py (S26, 2026-10-14) -- reader of the S25 return: the five code-matched DAEP trainings of D-30
(`v18A_prostate_daep_default_s{42,2024}`, `v18A_fundus_daep_default_s{1337,42,2024}`), read by the seven rules of
DOCS/D30_PREREGISTRATION_S25.md (fixed 2026-10-13, before any run existed). Assertion-first (R-48); nothing typed by hand (R-14).
It extends make_selection_s24.py (whose line-197 assertion "no complete DAEP log is held for Fundus / Prostate 42, 2024" is
inverted here) and regenerates its four selection tables with every described run counted (the N-61 re-run, without
--not_yet_in_paper), so that the released-form DEviS rows (D-28, D-29) and the five D-30 runs enter the same tables.

  python make_selection_s25.py --returns <BSPC_results/RETURNS> --derived <derived_v04_S12> --release <release_s12> --out <derived_v04_S12/extra_s25_selection>

Reads   RETURNS/S25/{runs,diag_out,console_logs/S25,_manifests}; the logs of the within-v18 threshold references and of
        every complete run under RETURNS (as make_selection_s24.py); RETURNS/S06/runs/prostate/v18A_prostate_daep_default_s1337
        (the reference command and the first member of the Prostate triple); extra_s17/table_main_gate_tests_s17.csv (the
        submission digests of the Table 4 cells); extra_s23/facts_s23.json, extra_s24/facts_s24.json (D-28 / D-29 trajectories);
        table_signal_identity.csv (S12), extra_s23/ and extra_s24/ identity tables (the S9 counts).
Writes  table_d30_integrity_s25.csv      rule 1, 2: one row per run
        table_d30_gates_s25.csv          rule 3: DAEP minus threshold at the three checkpoints, Prostate and Fundus, paired over seeds
        table_d30_codematched_s25.csv    rule 4: code-matched best values next to the submission digests and the Table 4 means
        table_d30_cup_s25.csv            rule 5: Fundus cup and disc per (domain, seed) at the three checkpoints; pooled delta, t12, tD
        table_d30_signal_s25.csv         rule 6: the Phase-A rule and the identity on the five new teachers
        table_d30_trajectory_s25.csv     rule 7: best, best iteration, last, SWAD, per-domain Dice at best
        table_band_summary_s25.csv, table_band_generality_s25.csv, table_signal_identity_s25.csv (v02 functions on the five JSONs)
        table_selection_runs_s25.csv, table_selection_summary_s25.csv, table_selection_gates_s25.csv, table_selection_baselines_s25.csv
                                         the S24 selection tables regenerated with every described run counted
        facts_s25.json                   every number the text may quote; the assertions that passed
        ADJUDICATION_S25.md              generated from the facts (copied to DOCS/adjudications/ by the session)
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, re, shutil, sys, tempfile
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_derived_v02 as v02                      # noqa: E402
import make_derived_v04 as v04                      # noqa: E402
import make_selection_s24 as s24                    # noqa: E402
from make_extra_s23 import (need, sha, wcsv, f4, find_log, split_of, rule,   # noqa: E402
                            SEEDS, DATASETS, DOM, MAXIT, LBDOM, RULE_P, RULE_DELTA, RULE_BAND, RULE_RANGE, PASSED)

CK = ('best', 'last', 'swad')
NEW = {'prostate': (42, 2024), 'fundus': (1337, 42, 2024)}
TABLE4_MEAN = {'prostate': 85.38, 'fundus': 88.54}            # Table 4, DAEP row, as printed (pinned in S17; rule 4 keeps them)
TABLE5 = {'cup': {'pooled': 0.03, 'p_t12': 0.885, 'p_tD': 0.873, 'plus_minus': '6/6'},
          'disc': {'pooled': 0.11, 'p_t12': 0.509, 'p_tD': 0.545, 'plus_minus': '8/4'}}   # Table 5 as printed (original runs; kept)
ONE_POINT = 1.0                                                # rule 4's consequence line, in Dice points
SUM_RX = v04.SUM_RX; DOM_RX = v04.DOM_RX
# flags that did not exist in the trainer when the S06 reference run was trained (added in v7c, v9, v11) and their defaults, inert for --method daep
LATER_FLAGS = {'consistency_ramp_frac': -1.0, 'selftest_ramp': False, 'mc_dropout_place': 'uamt', 'mc_target': 'teacher', 'devis_variant': 'published', 'devis_rel_epochs': 99, 'devis_rel_lambda_epochs': 50}


def cmd_pairs(txt):
    """The command of a log's first line as {flag: value} (a flag without a value maps to True)."""
    line = txt.splitlines()[0]; toks = line.split('python train_fourierfuse_v18_revision.py ', 1)[1].split()
    out, i = {}, 0
    while i < len(toks):
        if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
            out[toks[i]] = toks[i + 1]; i += 2
        else:
            out[toks[i]] = True; i += 1
    return out


def ref_name(ds, sd):
    return f'v18B_{ds}_fixmatch_split1337_init1337' if (sd == 1337 and ds in ('prostate', 'fundus')) else f'v18B_{ds}_fixmatch_s{sd}'


def parse_blocks(txt):
    """Every evaluation of a log: epoch, the student's per-domain block {domain: [dice per class]}, the SWAD block (when one exists),
    the student average and the running best (as logged)."""
    evals, cur, phase, pending = [], None, None, {}
    for line in txt.splitlines():
        if 'Testing student...' in line:
            cur, phase, pending = {'student': [], 'ema': [], 'swad': []}, 'student', {}
        elif 'Testing EMA teacher...' in line:
            phase, pending = 'ema', {}
        elif 'Testing SWAD' in line:
            phase, pending = 'swad', {}
        elif cur is not None and phase:
            m = DOM_RX.search(line)
            if m:
                pending[int(m.group(1))] = [float(x) for x in re.findall(r'dice=([\d.]+)', m.group(3))]
            elif re.search(r'\] epoch \d+: avg ', line):
                if pending:
                    cur[phase].append(pending)
                pending = {}
        s = SUM_RX.search(line)
        if s and cur is not None:
            evals.append({'epoch': int(s.group(1)), 'student_avg': float(s.group(2)), 'best_avg': float(s.group(4)), 'best_iter': int(s.group(5)),
                          'student': cur['student'][-1] if cur['student'] else None, 'swad': cur['swad'][-1] if cur['swad'] else None})
            cur, phase = None, None
    return evals


def blocks_at_checkpoints(logpath):
    """Per-domain, per-class Dice of a run at (a) the best checkpoint (the student's evaluation at best_iter), (b) the last iteration
    (the student's last evaluation), (c) the final SWAD average (the SWAD block of the last evaluation)."""
    txt = open(logpath, errors='replace').read(); ev = parse_blocks(txt); q = v02.parse_log(logpath)
    n_eval = int(re.search(r'num_eval_iter=(\d+)', txt).group(1)); assert n_eval == 500
    best = ev[q['best_iter'] // n_eval - 1]; assert best['epoch'] * n_eval == q['best_iter'], (logpath, best['epoch'], q['best_iter'])
    assert abs(best['student_avg'] - q['best_avg']) < 6e-5, (logpath, best['student_avg'], q['best_avg'])
    last = ev[-1]; assert last['swad'] is not None, logpath
    return {'best': best['student'], 'last': last['student'], 'swad': last['swad'], 'best_iter': q['best_iter'], 'n_evals': len(ev)}


def verdict(a, b):
    """The registered criterion of section 4.4 in both directions: a gate beats the other iff paired t over the three seeds gives p < 0.05
    and |mean difference| >= 0.5 Dice; otherwise 'within seed noise'."""
    x = s24.paired(a, b)
    v = 'DAEP beats threshold' if (x['p'] < RULE_P and x['delta'] >= RULE_DELTA) else 'threshold beats DAEP' if (x['p'] < RULE_P and x['delta'] <= -RULE_DELTA) else 'within seed noise'
    x['verdict'] = v
    return x


def one_sample(d):
    d = np.array(d, dtype=float); t, p = stats.ttest_1samp(d, 0.0)
    return float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--returns', required=True); ap.add_argument('--derived', required=True)
    ap.add_argument('--release', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S25 = os.path.join(a.returns, 'S25'); F = {'session': 'S26', 'reader': 'make_selection_s25.py', 'rules': 'DOCS/D30_PREREGISTRATION_S25.md (fixed 2026-10-13)'}
    src = os.path.join(a.release, 'CODE', 'source')
    code_sha = {n: sha(os.path.join(src, n)) for n in ('train_fourierfuse_v18_revision.py', 'revision_modules.py', 'fourierfuse_v16_modules.py', 'diag_uncertainty_harness_v18.py')}
    F['code_sha256'] = {k: v[:16] for k, v in code_sha.items()}
    F23 = json.load(open(os.path.join(a.derived, 'extra_s23', 'facts_s23.json'))); F24 = json.load(open(os.path.join(a.derived, 'extra_s24', 'facts_s24.json')))

    # ---------------------------------------------------------------- rule 1 (a): the collector's manifests, every file as sent
    n_ok, parts = 0, 0
    for mf in sorted(glob.glob(os.path.join(S25, '_manifests', 'p*', 'MANIFEST.txt'))):
        parts += 1
        for line in open(mf):
            if line.strip() and not line.startswith('#'):
                p, size, h = line.rstrip('\n').split('\t'); q = os.path.join(S25, p)
                assert os.path.exists(q) and os.path.getsize(q) == int(size) and sha(q) == h, f'manifest mismatch {p}'; n_ok += 1
    F['manifest_files_verified'] = n_ok; F['sendback_parts'] = parts
    need(n_ok == 57 and parts == 3, 'every file of the S25 sendback (57, three collector parts) is present in the results tier and matches the collector manifest (size and sha256)')

    # ---------------------------------------------------------------- rule 2: runner history (no failure, no relaunch), R-64 files
    st = list(csv.DictReader(open(os.path.join(S25, 'console_logs', 'S25', 'status_jobs_D30.csv'))))
    hist = {}
    for r in st:
        hist.setdefault(r['job'], []).append(r)
    T = lambda s: np.datetime64(s.strip().replace(' ', 'T'))
    F['runner'] = {'jobs': len(hist), 'attempts_per_job': {j: len(rs) for j, rs in hist.items()}, 'final_pass': sum(rs[-1]['postflight'] == 'PASS' for rs in hist.values()),
                   'first_start': min(rs[0]['start'] for rs in hist.values()), 'last_end': max(rs[-1]['end'] for rs in hist.values()),
                   'hours_per_job': {j: round((T(rs[-1]['end']) - T(rs[-1]['start'])) / np.timedelta64(1, 'h'), 2) for j, rs in hist.items()}}
    F['runner']['wall_clock_hours'] = round((T(F['runner']['last_end']) - T(F['runner']['first_start'])) / np.timedelta64(1, 'h'), 2)
    need(len(hist) == 5 and F['runner']['final_pass'] == 5 and all(len(rs) == 1 for rs in hist.values()), 'runner: five jobs, one attempt each, each PASS (no failure branch, no relaunch)')
    need(not os.path.isdir(os.path.join(S25, 'console_logs', 'S25_first_attempt')), 'no S25_first_attempt folder: the failure branch of RUNBOOK S25 was not used')
    for n in ('preflight.txt', 'selftest_revision_modules.txt', 'runner_jobs_D30.log', 'runner_jobs_D30smoke.log', 'status_jobs_D30smoke.csv', 'diag_harness.stdout.log'):
        assert os.path.exists(os.path.join(S25, 'console_logs', 'S25', n)), n
    need('self-test PASSED' in open(os.path.join(S25, 'console_logs', 'S25', 'selftest_revision_modules.txt'), errors='replace').read(), 'the server self-test of RUNBOOK S25 step 1 passed and was written to a file (R-64)')
    smoke = list(csv.DictReader(open(os.path.join(S25, 'console_logs', 'S25', 'status_jobs_D30smoke.csv'))))
    need(len(smoke) == 1 and smoke[0]['postflight'] == 'PASS' and 'v18smoke_fundus_daep_default' == smoke[0]['job'], 'the Fundus smoke run of the DAEP gate in the v18 code passed')
    need('[harness] ran 5 (failed 0), skipped 0 existing' in open(os.path.join(S25, 'console_logs', 'S25', 'diag_harness.stdout.log'), errors='replace').read(), 'the harness measured the five new teachers (ran 5, failed 0)')

    # ---------------------------------------------------------------- per run: integrity (rule 1), screens (rule 2), the reference command
    refcfg = json.load(open(os.path.join(a.returns, 'S06', 'runs', 'prostate', 'v18A_prostate_daep_default_s1337', 'run_config.json')))['args']
    p1337_log = os.path.join(a.returns, 'S06', 'runs', 'prostate', 'v18A_prostate_daep_default_s1337', 'log.txt')
    DSKEYS = {'prostate': {'seed', 'save_name', 'gpu'}, 'fundus': {'dataset', 'lb_domain', 'domain_num', 'max_iterations', 'swad_start_iter', 'seed', 'save_name', 'gpu', 'img_size'}}
    RUN, rows_i = {}, []
    for ds in DATASETS:
        for sd in NEW[ds]:
            run = f'v18A_{ds}_daep_default_s{sd}'; d = os.path.join(S25, 'runs', ds, run); log = os.path.join(d, 'log.txt')
            txt = open(log, errors='replace').read(); t = v04.screen_run(log); q = v02.parse_log(log)
            cfg = json.load(open(os.path.join(d, 'run_config.json'))); ar = cfg['args']
            ok_cfg = (ar['method'] == 'daep' and ar['seed'] == sd and ar['max_iterations'] == MAXIT[ds] and ar['lb_domain'] == LBDOM[ds] and ar['lb_num'] == 20 and ar['head'] == 'evidential'
                      and ar['consistency_ramp_frac'] == -1.0 and ar['amp'] == 1 and ar['daep_alpha'] == ar['daep_beta'] == ar['daep_gamma'] == 1.0 and ar['evid_lambda_kl_max'] == 0.02
                      and ar['evid_anneal_iters'] == 10000 and ar['evid_activation'] == 'softplus' and ar['evid_loss'] == 'mse' and ar['evid_prior_w'] == 1.0 and ar['evid_temperature'] == 1.0
                      and ar['evid_max_evidence'] == 0.0 and ar['split_seed'] == -1 and ar['mc_dropout_p'] == 0.0)
            ok_sha = (cfg['script_sha256'] == code_sha['train_fourierfuse_v18_revision.py'] == sha(os.path.join(d, 'code', 'train_fourierfuse_v18_revision.py'))
                      and cfg['revision_modules_sha256'] == code_sha['revision_modules.py'] == sha(os.path.join(d, 'code', 'revision_modules.py'))
                      and cfg['fourierfuse_v16_modules_sha256'] == code_sha['fourierfuse_v16_modules.py'] == sha(os.path.join(d, 'code', 'fourierfuse_v16_modules.py')))
            diff_keys = sorted(k for k in set(ar) | set(refcfg) if ar.get(k) != refcfg.get(k))
            later = sorted(k for k in diff_keys if k not in refcfg)              # flags added to the trainer after the S06 reference run was trained
            later_ok = set(later) <= set(LATER_FLAGS) and all(ar[k] == LATER_FLAGS[k] for k in later)
            same_args = (set(diff_keys) - set(later)) <= DSKEYS[ds] and 'save_name' in diff_keys and (('seed' in diff_keys) == (sd != 1337)) and later_ok
            # the exact check of GATES_S25: the command of log line 1 equals the reference command except the dataset block, --seed, --save_name and --gpu
            cmd, cmd0 = cmd_pairs(txt), cmd_pairs(open(p1337_log, errors='replace').read())
            cmd_diff = sorted(k for k in set(cmd) | set(cmd0) if cmd.get(k) != cmd0.get(k))
            same_cmd = set(cmd_diff) <= {'--dataset', '--lb_domain', '--domain_num', '--max_iterations', '--swad_start_iter', '--seed', '--save_name', '--gpu'} and '--save_name' in cmd_diff and (('--seed' in cmd_diff) == (sd != 1337))
            rn = ref_name(ds, sd); rlog, rq = find_log(a.returns, ds, rn)
            sp, spr = split_of(log), split_of(rlog); sj = json.load(open(os.path.join(d, 'split.json')))
            same_split = bool(sp[1]) and sp[1] == spr[1] and sj['lb_idxs'] == sp[1] and sp[0] == sd
            J = json.load(open(os.path.join(S25, 'diag_out', f'{ds}__{run}__ema.json'))); me = J['meta']
            ok_h = (me['which'] == 'ema' and me['iter'] == q['best_iter'] and abs(me['best_avg_dice'] - q['best_avg']) < 1e-6 and run in me['ckpt']
                    and me['harness_sha256'] == code_sha['diag_uncertainty_harness_v18.py'] and me['n_models'] == 1)
            ck = blocks_at_checkpoints(log); rck = blocks_at_checkpoints(rlog); tr = v04.screen_run(rlog)
            RUN[(ds, sd)] = {'run': run, 'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': t['student_avg_last'], 'swad': float(np.mean([np.mean(v) for v in ck['swad'].values()])),
                             'blocks': ck, 'ref_blocks': rck, 'refname': rn, 'ref': rq['best_avg'], 'ref_last': tr['student_avg_last'], 'ref_best_iter': rq['best_iter'],
                             'ref_swad': float(np.mean([np.mean(v) for v in rck['swad'].values()])), 'at_best': {k: float(np.mean(v)) for k, v in sorted(ck['best'].items())},
                             'at_last': {k: float(np.mean(v)) for k, v in sorted(ck['last'].items())}, 'flag': v04.flag_of(t), 'nf': t['nf_n'], 'J': J, 'log': log, 'split_seed_logged': sp[0]}
            rows_i.append([ds, run, sd, int(t['complete']), t['n_evals'], t['n_headers'], len(t['resumed_iters']), len(cfg.get('resume_events', [])), t['nf_n'], int(t['parse_ok']),
                           v04.flag_of(t), int(ok_cfg), int(same_cmd), ';'.join(cmd_diff), int(same_args), ';'.join(diff_keys), ';'.join(later), int(ok_sha), rn, int(same_split), int(ok_h), cfg['torch'], cfg['hostname'],
                           len(hist[run]), hist[run][-1]['postflight'], hist[run][-1]['start'], hist[run][-1]['end']])
            need(t['complete'] and t['n_evals'] == MAXIT[ds] // 500 and t['n_headers'] == 1 and not t['resumed_iters'] and not cfg.get('resume_events'),
                 f'{run}: one training from iteration 0 to the last iteration, no resume event')
            need(t['nf_n'] == 0 and t['parse_ok'] and v04.flag_of(t) == 'ok', f'{run}: no skipped iteration; R-34 screen (registered rule, last evaluation) ok: not a training failure')
            need(ok_cfg, f'{run}: configuration = the DAEP gate with the headline settings (unit weights, KL 0.02 annealed over 10k iterations, softplus, W = 1, T = 1, no cap, whole-run ramp)')
            need(same_cmd, f'{run}: the command of its log (line 1) equals the command that trained v18A_prostate_daep_default_s1337 except the dataset block, --seed, --save_name and --gpu (differing flags: {", ".join(cmd_diff)})')
            need(same_args, f'{run}: its recorded arguments equal those of v18A_prostate_daep_default_s1337 except the dataset block, the seed, the name and the GPU index, and {len(later)} flags added to the trainer after that run, all at their defaults ({", ".join(later)})')
            need(ok_sha, f'{run}: code snapshot, recorded hashes and the release code are the same bytes (tools v13 = v11 trainer and modules)')
            need(same_split, f'{run}: labeled slices identical to those of the threshold reference {rn} (split derived from --seed {sd}; the reference of seed 1337 was trained with --split_seed 1337)')
            need(ok_h, f'{run}: the measured teacher is the EMA weights of the best checkpoint of this log, measured with the release harness')
    wcsv(os.path.join(a.out, 'table_d30_integrity_s25.csv'), rows_i,
         ['dataset', 'run', 'seed', 'complete', 'n_evals', 'n_headers', 'n_resumes_log', 'n_resume_events_cfg', 'n_nonfinite_skips', 'parse_ok', 'r34_flag', 'config_ok',
          'command_equals_reference_except_dataset_seed_name_gpu', 'command_flags_that_differ', 'args_equal_reference_except_dataset_seed_name_gpu_and_later_flags', 'args_that_differ_from_v18A_prostate_daep_default_s1337',
          'later_flags_at_default', 'hashes_ok', 'threshold_reference', 'split_equals_reference',
          'harness_meta_ok', 'torch', 'hostname', 'n_runner_attempts', 'postflight', 'start', 'end'])
    F['split_assertion_fundus_s1337'] = 'holds (the slices of --seed 1337 and of --split_seed 1337 are the same 20 indices)'

    # ---------------------------------------------------------------- the Prostate s1337 member (the first default run, S06) and the full triples
    p1337 = os.path.join(a.returns, 'S06', 'runs', 'prostate', 'v18A_prostate_daep_default_s1337', 'log.txt')
    q0 = v02.parse_log(p1337); t0 = v04.screen_run(p1337); ck0 = blocks_at_checkpoints(p1337); r0log, r0q = find_log(a.returns, 'prostate', ref_name('prostate', 1337)); rck0 = blocks_at_checkpoints(r0log); tr0 = v04.screen_run(r0log)
    RUN[('prostate', 1337)] = {'run': 'v18A_prostate_daep_default_s1337', 'best': q0['best_avg'], 'best_iter': q0['best_iter'], 'last': t0['student_avg_last'],
                               'swad': float(np.mean([np.mean(v) for v in ck0['swad'].values()])), 'blocks': ck0, 'ref_blocks': rck0, 'refname': ref_name('prostate', 1337), 'ref': r0q['best_avg'],
                               'ref_last': tr0['student_avg_last'], 'ref_best_iter': r0q['best_iter'], 'ref_swad': float(np.mean([np.mean(v) for v in rck0['swad'].values()])),
                               'at_best': {k: float(np.mean(v)) for k, v in sorted(ck0['best'].items())}, 'at_last': {k: float(np.mean(v)) for k, v in sorted(ck0['last'].items())},
                               'flag': v04.flag_of(t0), 'nf': t0['nf_n'], 'J': None, 'log': p1337, 'split_seed_logged': split_of(p1337)[0]}
    need(abs(q0['best_avg'] - 0.863008) < 1e-6 and RUN[('prostate', 1337)]['flag'] == 'ok', 'Prostate seed 1337: the DAEP member is the first default run (S06, 0.863008), as every reader since S06 uses it')
    need(split_of(p1337)[1] == split_of(r0log)[1], 'Prostate seed 1337: the DAEP member and the threshold reference share the labeled slices')

    # ---------------------------------------------------------------- rule 3: the gate comparison at the three checkpoints, paired over seeds
    R24 = s24.load(a.returns)      # every complete log under RETURNS, with best / last / final SWAD (the S24 instrument)
    for ds in DATASETS:
        for sd in SEEDS:
            r = RUN[(ds, sd)]; k = (ds, r['run']); kr = (ds, r['refname'])
            assert k in R24 and kr in R24, (k, kr)
            assert abs(R24[k]['best'] - r['best']) < 1e-9 and abs(R24[k]['last'] - r['last']) < 1e-9 and abs(R24[k]['swad'] - r['swad']) < 6e-4, (k, R24[k]['swad'], r['swad'])
            assert abs(R24[kr]['best'] - r['ref']) < 1e-9 and abs(R24[kr]['last'] - r['ref_last']) < 1e-9 and abs(R24[kr]['swad'] - r['ref_swad']) < 6e-4, (kr, R24[kr]['swad'], r['ref_swad'])
            r['swad'], r['ref_swad'] = R24[k]['swad'], R24[kr]['swad']       # the S24 instrument's value (block average of the logged SWAD evaluation)
    need(True, 'best, last-iteration and final-SWAD values of the six DAEP members and their six threshold references agree between this reader and the S24 instrument (make_selection_s24.load)')
    G3, rows_g = {}, []
    for ds in DATASETS:
        G3[ds] = {}
        for k in CK:
            dv = [RUN[(ds, s)][k] for s in SEEDS]; tv = [RUN[(ds, s)]['ref' if k == 'best' else ('ref_last' if k == 'last' else 'ref_swad')] for s in SEEDS]
            x = verdict(dv, tv); x['daep'] = [float(v) for v in dv]; x['threshold'] = [float(v) for v in tv]; G3[ds][k] = x
            rows_g.append([ds, 'paired over seeds 1337/42/2024', k, f'{100 * x["a_mean"]:.2f}', f'{100 * x["a_std"]:.2f}', f'{100 * x["b_mean"]:.2f}', f'{100 * x["b_std"]:.2f}',
                           f'{100 * x["delta"]:+.2f}', ';'.join(f'{100 * v:+.2f}' for v in x['per_seed']), f'{x["t"]:.3f}', f'{x["p"]:.3f}', x['verdict'],
                           ';'.join(f'{100 * v:.2f}' for v in dv), ';'.join(f'{100 * v:.2f}' for v in tv)])
        G3[ds]['survives_selection_rule'] = bool(G3[ds]['last']['verdict'] == G3[ds]['best']['verdict'] == G3[ds]['swad']['verdict'])
        G3[ds]['verdicts'] = {k: G3[ds][k]['verdict'] for k in CK}
    wcsv(os.path.join(a.out, 'table_d30_gates_s25.csv'), rows_g,
         ['dataset', 'design', 'checkpoint', 'daep_mean', 'daep_std', 'threshold_mean', 'threshold_std', 'daep_minus_threshold', 'per_seed_delta_1337_42_2024', 't_paired', 'p_paired',
          'verdict_registered_criterion', 'daep_per_seed', 'threshold_per_seed'])
    F['rule3'] = {ds: {k: {kk: vv for kk, vv in G3[ds][k].items()} for k in CK} | {'survives_selection_rule': G3[ds]['survives_selection_rule'], 'verdicts': G3[ds]['verdicts']} for ds in DATASETS}
    for ds in DATASETS:
        need(G3[ds]['best']['verdict'] == 'within seed noise', f'{ds}: rule 3 at the best checkpoint: within seed noise (delta {100 * G3[ds]["best"]["delta"]:+.2f}, p = {G3[ds]["best"]["p"]:.3f})')
    F['rule3_summary'] = {ds: {'survives': G3[ds]['survives_selection_rule'], 'verdicts': G3[ds]['verdicts'], 'deltas': {k: round(100 * G3[ds][k]['delta'], 2) for k in CK}, 'p': {k: round(G3[ds][k]['p'], 3) for k in CK}} for ds in DATASETS}

    # ---------------------------------------------------------------- rule 4: code-matched values against Table 4 (descriptive; one fixed consequence)
    dig = {(r['dataset'], int(r['seed'])): (float(r['daep']), float(r['threshold'])) for r in csv.DictReader(open(os.path.join(a.derived, 'extra_s17', 'table_main_gate_tests_s17.csv'))) if r['seed'] != 'paired'}
    need(abs(dig[('prostate', 1337)][0] - 0.857467) < 1e-6 and abs(dig[('fundus', 42)][0] - 0.883793) < 1e-6, 'the submission digests of the Table 4 DAEP cells are read from extra_s17/table_main_gate_tests_s17.csv (Prostate s1337 0.857467, Fundus s42 0.883793)')
    rows_c, C4 = [], {}
    for ds in DATASETS:
        per = {}
        for sd in SEEDS:
            cm, dg = RUN[(ds, sd)]['best'], dig[(ds, sd)][0]; per[sd] = {'code_matched': cm, 'digest': dg, 'diff_points': 100 * (cm - dg), 'source': 'S06 (the first default run)' if (ds, sd) == ('prostate', 1337) else 'S25 (D-30)'}
            rows_c.append([ds, sd, RUN[(ds, sd)]['run'], per[sd]['source'], f'{cm:.6f}', f'{dg:.6f}', f'{100 * (cm - dg):+.2f}'])
        m_cm = 100 * float(np.mean([per[s]['code_matched'] for s in SEEDS])); m_dg = 100 * float(np.mean([per[s]['digest'] for s in SEEDS]))
        C4[ds] = {'per_seed': {str(s): per[s] for s in SEEDS}, 'mean_code_matched': m_cm, 'mean_digest': m_dg, 'table4_mean': TABLE4_MEAN[ds], 'mean_minus_table4': m_cm - TABLE4_MEAN[ds],
                  'abs_diff_min_max_points': [min(abs(per[s]['diff_points']) for s in SEEDS), max(abs(per[s]['diff_points']) for s in SEEDS)],
                  'one_point_consequence_triggered': bool(abs(m_cm - TABLE4_MEAN[ds]) >= ONE_POINT)}
        need(abs(m_dg - TABLE4_MEAN[ds]) < 0.006, f'{ds}: the three-seed mean of the digests equals the Table 4 DAEP mean ({TABLE4_MEAN[ds]})')
        rows_c.append([ds, 'mean', '', '', f'{m_cm:.2f}', f'{m_dg:.2f}', f'{m_cm - TABLE4_MEAN[ds]:+.2f} (code-matched mean minus Table 4 mean {TABLE4_MEAN[ds]}); one-point consequence: {"triggered" if C4[ds]["one_point_consequence_triggered"] else "not triggered"}'])
    # the two yardsticks on record (E-58; the s42 threshold reference against its digest)
    rerun = {'prostate_daep': (RUN[('prostate', 1337)]['best'], 0.857467), 'prostate_threshold': (RUN[('prostate', 1337)]['ref'], v02.THRESHOLD_REF['prostate'][1337]),
             'fundus_threshold': (RUN[('fundus', 1337)]['ref'], v02.THRESHOLD_REF['fundus'][1337])}
    for ds2, nm in (('MNMS', 'v18B_MNMS_fixmatch_s1337'), ('BUSI', 'v18B_BUSI_fixmatch_s1337')):
        _, qq = find_log(a.returns, ds2, nm); rerun[f'{ds2}_threshold'] = (qq['best_avg'], v02.THRESHOLD_REF[ds2][1337])
    yard = {k: 100 * (v[0] - v[1]) for k, v in rerun.items()}
    s42 = 100 * (RUN[('prostate', 42)]['ref'] - v02.THRESHOLD_REF['prostate'][42])
    C4['yardsticks'] = {'five_seed1337_reruns_vs_digest_points': {k: round(v, 2) for k, v in yard.items()}, 'abs_min_max': [round(min(abs(v) for v in yard.values()), 2), round(max(abs(v) for v in yard.values()), 2)],
                        'threshold_reference_s42_vs_digest_points': round(s42, 2)}
    need(abs(min(abs(v) for v in yard.values()) - 0.01) < 0.006 and abs(max(abs(v) for v in yard.values()) - 0.56) < 0.006 and abs(s42 - 0.66) < 0.006,
         'the two yardsticks on record are reproduced from the logs: five seed-1337 re-runs within 0.01-0.56 points of their digests; the s42 Prostate threshold reference +0.66 against its digest')
    wcsv(os.path.join(a.out, 'table_d30_codematched_s25.csv'), rows_c, ['dataset', 'seed', 'run', 'source', 'code_matched_best', 'submission_digest', 'code_matched_minus_digest_points'])
    F['rule4'] = C4

    # ---------------------------------------------------------------- rule 5: the Fundus optic cup (and disc) at the three checkpoints
    rows_5, C5 = [], {}
    for cls_i, cls in enumerate(('cup', 'disc')):
        C5[cls] = {}
        for k in CK:
            cells, per_dom = [], {d: [] for d in range(1, 5)}
            for sd in SEEDS:
                b, rb = RUN[('fundus', sd)]['blocks'][k], RUN[('fundus', sd)]['ref_blocks'][k]
                for d in range(1, 5):
                    delta = 100 * (b[d][cls_i] - rb[d][cls_i]); cells.append(delta); per_dom[d].append(delta)
                    rows_5.append(['fundus', cls, k, d, sd, f'{100 * b[d][cls_i]:.2f}', f'{100 * rb[d][cls_i]:.2f}', f'{delta:+.2f}'])
            dm = [float(np.mean(per_dom[d])) for d in range(1, 5)]
            t12, p12 = one_sample(cells); tD, pD = one_sample(dm)
            C5[cls][k] = {'pooled_delta': float(np.mean(cells)), 'std': float(np.std(cells, ddof=1)), 'n_pos': int(sum(c > 0 for c in cells)), 'n_neg': int(sum(c < 0 for c in cells)),
                          't12': t12, 'p_t12': p12, 'tD': tD, 'p_tD': pD, 'per_domain_mean': dm, 'cells': [float(c) for c in cells]}
        C5[cls]['survives'] = bool(all(not (abs(C5[cls][k]['pooled_delta']) >= 0.5 and C5[cls][k]['p_tD'] < RULE_P) for k in ('last', 'swad')))
        C5[cls]['table5_as_printed'] = TABLE5[cls]
        for k in CK:
            x = C5[cls][k]
            rows_5.append(['fundus', cls, k, 'pooled', '12 cells', '', '', f'{x["pooled_delta"]:+.2f} (sd {x["std"]:.2f}; +/- {x["n_pos"]}/{x["n_neg"]}; t12 {x["t12"]:.2f}, p {x["p_t12"]:.3f}; tD {x["tD"]:.2f}, p {x["p_tD"]:.3f})'])
    wcsv(os.path.join(a.out, 'table_d30_cup_s25.csv'), rows_5, ['dataset', 'class', 'checkpoint', 'domain', 'seed', 'daep_dice', 'threshold_dice', 'daep_minus_threshold_points'])
    F['rule5'] = C5
    need(C5['cup']['survives'], 'rule 5: the reading "the cup shows no advantage for evidential gating" survives (at neither alternative checkpoint does the cup give |pooled delta| >= 0.5 with tD p < 0.05)')
    need(all(C5[c][k]['p_tD'] > RULE_P for c in ('cup', 'disc') for k in CK), 'rule 5: no class at any checkpoint reaches tD p < 0.05 (the recomputation on the code-matched runs is a null on cup and disc at all three checkpoints)')

    # ---------------------------------------------------------------- rule 6: the signal of the five new teachers (v02 functions, unchanged) and the identity
    H = {k: v for k, v in v02.load_harness(S25).items() if k[1].startswith('v18A_')}
    need(len(H) == 5 and all(k[2] == 'ema' for k in H), 'five harness JSONs, EMA teachers')
    tmp = tempfile.mkdtemp(); A = {}
    v02.band_tables(H, tmp); v02.signal_identity_table(H, tmp, A)
    for n in ('table_band_generality', 'table_band_summary', 'table_signal_identity'):
        shutil.copy(os.path.join(tmp, n + '.csv'), os.path.join(a.out, n + '_s25.csv'))
    bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_band_summary_s25.csv')))}
    idn = {(r['dataset'], r['run']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_signal_identity_s25.csv')))}
    old_bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.derived, 'table_band_summary.csv')))}
    old_id = list(csv.DictReader(open(os.path.join(a.derived, 'table_signal_identity.csv'))))
    rows_s, SIG = [], {}
    for ds in DATASETS:
        for sd in NEW[ds]:
            r = RUN[(ds, sd)]; run = r['run']; u = bs[(ds, run, 'u_epi')]; c = bs[(ds, run, 'conf')]; i = idn[(ds, run)]
            band, rng = float(u['auroc_edt5_mean']), float(u['range80_all_mean'])
            per_dom = [float(x['signals']['u_epi']['band']['edt_gt_5']['auroc']) for _, x in sorted(r['J']['per_domain'].items(), key=lambda kv: int(kv[0]))]
            ou = old_bs[(ds, r['refname'], 'u_epi')]
            SIG[(ds, sd)] = {'auroc_all': float(u['auroc_all_mean']), 'band': band, 'range': rng, 'range_band': float(u['range80_edt5_mean']), 'conf_all': float(c['auroc_all_mean']),
                             'conf_band': float(c['auroc_edt5_mean']), 'errors_in_band': float(u['frac_errors_in_edt5_mean']), 'identity_holds': i['identity_holds'] == '1',
                             'gap_all': float(i['max_abs_auroc_gap_all']), 'gap_band': float(i['max_abs_auroc_gap_edt5']), 'median_ratio': float(i['median_ratio_u_epi_over_conf']), 'K': int(i['K']),
                             'band_per_domain': per_dom, 'ref_all': float(ou['auroc_all_mean']), 'ref_band': float(ou['auroc_edt5_mean']), 'ref_range': float(ou['range80_all_mean']),
                             'persists': bool(band <= RULE_BAND and rng <= RULE_RANGE), 'margin': band - RULE_BAND}
            s = SIG[(ds, sd)]
            rows_s.append([ds, sd, run, f4(s['auroc_all']), f4(band), f'{rng:.3e}', f'{s["range_band"]:.3e}', f4(s['conf_all']), f4(s['conf_band']), f4(s['errors_in_band']), int(band <= RULE_BAND),
                           int(rng <= RULE_RANGE), int(s['persists']), f'{s["margin"]:+.4f}', f'{RULE_RANGE / rng:.0f}', int(s['identity_holds']), f4(s['gap_all']), f4(s['gap_band']), f4(s['median_ratio']),
                           ';'.join(f'{x:.3f}' for x in per_dom), f4(s['ref_all']), f4(s['ref_band']), f'{s["ref_range"]:.3e}'])
    wcsv(os.path.join(a.out, 'table_d30_signal_s25.csv'), rows_s,
         ['dataset', 'seed', 'run', 'u_epi_auroc_all', 'u_epi_auroc_band5', 'u_epi_range80_all', 'u_epi_range80_band5', 'conf_auroc_all', 'conf_auroc_band5', 'frac_errors_in_band5', 'band_le_0.70',
          'range_le_1e-2', 'phaseA_rule_collapse_persists', 'band_margin_to_0.70', 'range_factor_below_1e-2', 'identity_holds', 'max_abs_auroc_gap_all', 'max_abs_auroc_gap_band5',
          'median_ratio_u_epi_over_conf', 'u_epi_band5_per_domain', 'threshold_ref_auroc_all', 'threshold_ref_band5', 'threshold_ref_range'])
    allfive = list(SIG.values())
    F['rule6'] = {'per_teacher': {f'{ds}_s{sd}': {k: (SIG[(ds, sd)][k] if not isinstance(SIG[(ds, sd)][k], list) else [round(x, 4) for x in SIG[(ds, sd)][k]]) for k in SIG[(ds, sd)]} for ds in DATASETS for sd in NEW[ds]},
                  'range_max': max(x['range'] for x in allfive), 'range_factor_below_bound_min': RULE_RANGE / max(x['range'] for x in allfive),
                  'band_min_max': [min(x['band'] for x in allfive), max(x['band'] for x in allfive)], 'n_persists': sum(x['persists'] for x in allfive), 'n_band_above_070': sum(x['band'] > RULE_BAND for x in allfive),
                  'band_excess_max': max(x['band'] for x in allfive) - RULE_BAND, 'n_identity_holds': sum(x['identity_holds'] for x in allfive), 'gap_band_max': max(x['gap_band'] for x in allfive),
                  'gap_all_max': max(x['gap_all'] for x in allfive), 'auroc_all_min_max': [min(x['auroc_all'] for x in allfive), max(x['auroc_all'] for x in allfive)],
                  'per_dataset': {ds: {'band': [round(SIG[(ds, sd)]['band'], 4) for sd in NEW[ds]], 'all': [round(SIG[(ds, sd)]['auroc_all'], 4) for sd in NEW[ds]], 'range': [SIG[(ds, sd)]['range'] for sd in NEW[ds]],
                                       'identity': [SIG[(ds, sd)]['identity_holds'] for sd in NEW[ds]], 'ref_band_min_max': [min(SIG[(ds, sd)]['ref_band'] for sd in NEW[ds]), max(SIG[(ds, sd)]['ref_band'] for sd in NEW[ds])]} for ds in DATASETS}}
    need(F['rule6']['range_max'] <= RULE_RANGE / 2, 'rule 6: the central-80 % range of u_epi is at most half of the 1e-2 bound on all five teachers (range condition of the Phase-A rule)')
    # the band condition is applied per teacher as registered; count and margins are recorded and printed, not asserted (the text is written after this classification)
    def cls_run(run):
        return 'ensemble' if 'ENSEMBLE' in run else 'devis' if 'devis' in run else 'medl' if 'medl' in run else 'exp' if 'act_exp' in run else 'testbed'
    tb_old = [r for r in old_id if cls_run(r['run']) == 'testbed']; dm_old = [r for r in old_id if cls_run(r['run']) in ('devis', 'medl')]
    s23_fail = 6 - F23['signal_all_six']['n_identity_holds']; s24_fail = 6 - F24['signal_all_six']['n_identity_holds']; new_fail = 5 - F['rule6']['n_identity_holds']
    F['s9_counts'] = {'n_all_s12': len(old_id), 'n_all_with_d28': len(old_id) + 6, 'n_all_with_d28_d29': len(old_id) + 12, 'n_all_with_d28_d29_d30': len(old_id) + 17,
                      'n_testbed_objective_s12': len(tb_old), 'n_testbed_objective_with_d30': len(tb_old) + 5,
                      'n_testbed_identity_fails_s12': sum(r['identity_holds'] != '1' for r in tb_old), 'n_testbed_identity_fails_with_d30': sum(r['identity_holds'] != '1' for r in tb_old) + new_fail,
                      'n_devis_medl_s12': len(dm_old), 'n_devis_medl_with_d28_d29': len(dm_old) + 12,
                      'n_devis_medl_identity_fails_s12': sum(r['identity_holds'] != '1' for r in dm_old), 'n_devis_medl_identity_fails_with_d28_d29': sum(r['identity_holds'] != '1' for r in dm_old) + s23_fail + s24_fail,
                      'd30_identity_fails': new_fail, 'd28_identity_fails': s23_fail, 'd29_identity_fails': s24_fail}
    need(F['s9_counts']['n_all_s12'] == 83 and F['s9_counts']['n_testbed_objective_s12'] == 62 and F['s9_counts']['n_devis_medl_s12'] == 18 and F['s9_counts']['n_testbed_identity_fails_s12'] == 1,
         'the S12 identity table holds 83 checkpoints: 62 with the testbed objective (identity fails on 1), 18 DEviS / MEDL (counts as in S23 and S24)')
    # the testbed-objective band range of S9 ("0.56-0.73") with the five new teachers
    # the S9 range "0.56-0.73" is over the 62 evidential-head testbed-objective checkpoints of the identity table (softmax-head teachers have no evidential head
    # and are in the band table but not in the identity table; including them would give 0.79, which is not what S9 describes)
    in_id = {(r['dataset'], r['run']) for r in old_id}
    tb_bands = [float(r['auroc_edt5_mean']) for r in csv.DictReader(open(os.path.join(a.derived, 'table_band_summary.csv'))) if r['signal'] == 'u_epi' and cls_run(r['run']) == 'testbed' and (r['dataset'], r['run']) in in_id]
    need(len(tb_bands) == 62 and abs(min(tb_bands) - 0.56) < 0.005 and abs(max(tb_bands) - 0.73) < 0.005, 'the S9 range 0.56-0.73 of the testbed objective is the band AUROC over the 62 evidential-head checkpoints of the identity table (softmax-head teachers excluded)')
    F['s9_counts']['testbed_band_min_max_s12'] = [min(tb_bands), max(tb_bands)]
    F['s9_counts']['testbed_band_min_max_with_d30'] = [min(tb_bands + [x['band'] for x in allfive]), max(tb_bands + [x['band'] for x in allfive])]

    # ---------------------------------------------------------------- rule 7: descriptives
    rows_t = []
    for ds in DATASETS:
        for sd in SEEDS:
            r = RUN[(ds, sd)]
            rows_t.append([ds, sd, r['run'], 'S06' if (ds, sd) == ('prostate', 1337) else 'S25', f'{r["best"]:.6f}', r['best_iter'], f'{r["best_iter"] / MAXIT[ds]:.3f}', f4(r['last']), f4(r['swad']),
                           f'{100 * (r["best"] - r["last"]):.2f}', f'{100 * (r["best"] - r["swad"]):.2f}', ';'.join(f'{DOM[ds][k - 1]}={100 * r["at_best"][k]:.2f}' for k in sorted(r['at_best'])),
                           ';'.join(f'{DOM[ds][k - 1]}={100 * r["at_last"][k]:.2f}' for k in sorted(r['at_last'])), r['refname'], f'{r["ref"]:.6f}', r['ref_best_iter'], f4(r['ref_last']), f4(r['ref_swad']),
                           f'{100 * (r["ref"] - r["ref_last"]):.2f}'])
        F[f'{ds}_descriptive'] = {str(sd): {'best': RUN[(ds, sd)]['best'], 'best_iter': RUN[(ds, sd)]['best_iter'], 'best_frac': RUN[(ds, sd)]['best_iter'] / MAXIT[ds], 'last': RUN[(ds, sd)]['last'],
                                            'swad': RUN[(ds, sd)]['swad'], 'best_minus_last_points': 100 * (RUN[(ds, sd)]['best'] - RUN[(ds, sd)]['last']),
                                            'ref_best_minus_last_points': 100 * (RUN[(ds, sd)]['ref'] - RUN[(ds, sd)]['ref_last']),
                                            'at_best': {DOM[ds][k - 1]: round(100 * v, 2) for k, v in RUN[(ds, sd)]['at_best'].items()}, 'lowest_vendor_at_best': DOM[ds][min(RUN[(ds, sd)]['at_best'], key=RUN[(ds, sd)]['at_best'].get) - 1]} for sd in SEEDS}
    F['prostate_descriptive']['d28_best_minus_last'] = F23['prostate_late']['released_best_minus_last']
    F['prostate_descriptive']['d29_best_minus_last'] = F24['prostate_late']['belief_best_minus_last']
    wcsv(os.path.join(a.out, 'table_d30_trajectory_s25.csv'), rows_t,
         ['dataset', 'seed', 'run', 'return', 'best_avg', 'best_iter', 'best_frac', 'student_avg_last', 'swad_final', 'best_minus_last_points', 'best_minus_swad_points', 'dice_at_best_per_domain',
          'dice_at_last_per_domain', 'threshold_reference', 'ref_best_avg', 'ref_best_iter', 'ref_last', 'ref_swad_final', 'ref_best_minus_last_points'])
    need(all(F['prostate_descriptive'][str(s)]['lowest_vendor_at_best'] == 'BIDMC' for s in SEEDS), 'Prostate: BIDMC is the lowest vendor at the best checkpoint of each of the three code-matched DAEP runs (as in every run of Table 12)')

    # ---------------------------------------------------------------- the S24 selection tables regenerated with every described run counted (N-61)
    rows, S = [], {}
    for (ds, run), r in sorted(R24.items()):
        grp = 'failure' if (r['flag'] != 'ok' or r['nf'] > 0) else 'short_ramp' if 'ramp10' in run else 'counted'
        rows.append([ds, run, r['ret'], grp, f'{r["best"]:.6f}', r['best_iter'], f'{r["last"]:.4f}', '' if r['swad'] is None else f'{r["swad"]:.4f}', f'{100 * (r["best"] - r["last"]):.2f}',
                     '' if r['swad'] is None else f'{100 * (r["best"] - r["swad"]):.2f}', r['flag'], r['nf']]); r['grp'] = grp
    wcsv(os.path.join(a.out, 'table_selection_runs_s25.csv'), rows, ['dataset', 'run', 'return', 'group', 'best', 'best_iter', 'last', 'swad_final', 'best_minus_last_points', 'best_minus_swad_points', 'r34_flag', 'n_nonfinite_skips'])
    srow = []
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        c = {run: r for (d, run), r in R24.items() if d == ds and r['grp'] == 'counted'}; assert all(r['swad'] is not None for r in c.values()), ds
        bl = np.array([100 * (r['best'] - r['last']) for r in c.values()]); bs_ = np.array([100 * (r['best'] - r['swad']) for r in c.values()])
        S[ds] = {'n': len(c), 'n_all_complete': sum(d == ds for d, _ in R24), 'n_failures': sum(d == ds and r['grp'] == 'failure' for (d, _), r in R24.items()),
                 'n_short_ramp': sum(d == ds and r['grp'] == 'short_ramp' for (d, _), r in R24.items()), 'bl_median': float(np.median(bl)), 'bl_max': float(bl.max()),
                 'bl_n_gt1': int((bl > 1).sum()), 'bl_n_gt5': int((bl > 5).sum()), 'bs_median': float(np.median(bs_)), 'bs_max': float(bs_.max()), 'bs_min': float(bs_.min()),
                 'bl_gt5': {run: round(100 * (r['best'] - r['last']), 2) for run, r in sorted(c.items()) if 100 * (r['best'] - r['last']) > 5},
                 'n_d28': sum(run.startswith('v18D_') for run in c), 'n_d29': sum(run.startswith('v18E_') for run in c), 'n_d30': sum(run in {RUN[(ds, s)]['run'] for s in NEW.get(ds, ())} for run in c) if ds in NEW else 0}
        srow.append([ds, S[ds]['n_all_complete'], S[ds]['n_failures'], S[ds]['n_short_ramp'], S[ds]['n'], S[ds]['n_d28'], S[ds]['n_d29'], S[ds]['n_d30'], f'{S[ds]["bl_median"]:.2f}', f'{S[ds]["bl_max"]:.2f}',
                     S[ds]['bl_n_gt1'], S[ds]['bl_n_gt5'], f'{S[ds]["bs_median"]:.2f}', f'{S[ds]["bs_min"]:.2f}', f'{S[ds]["bs_max"]:.2f}'])
    wcsv(os.path.join(a.out, 'table_selection_summary_s25.csv'), srow,
         ['dataset', 'n_complete_logs', 'n_training_failures', 'n_short_ramp', 'n_counted', 'n_d28_released_threshold', 'n_d29_released_belief', 'n_d30_daep', 'best_minus_last_median', 'best_minus_last_max',
          'n_above_1_point', 'n_above_5_points', 'best_minus_swad_median', 'best_minus_swad_min', 'best_minus_swad_max'])
    F['size'] = S; F['n_logs'] = len(R24)
    S24F = json.load(open(os.path.join(a.derived, 'extra_s24_selection', 'facts_s24_selection.json')))
    need(S['prostate']['n'] == S24F['size']['prostate']['n'] + 8 and S['fundus']['n'] == S24F['size']['fundus']['n'] + 9 and S['MNMS']['n'] == 11 and S['BUSI']['n'] == 11,
         f'counted runs: Prostate {S["prostate"]["n"]} (S24: 44, + 3 D-28, + 3 D-29, + 2 D-30), Fundus {S["fundus"]["n"]} (25 + 3 + 3 + 3), M&Ms 11, BUSI 11')
    need(len(R24) == S24F['n_logs'] + 11, f'complete logs held: {len(R24)} (the 104 of S24 turn 3, plus the six of D-29 and the five of D-30)')
    F['size_s24'] = {ds: {'n': S24F['size'][ds]['n'], 'bl_median': S24F['size'][ds]['bl_median'], 'bl_n_gt5': S24F['size'][ds]['bl_n_gt5']} for ds in ('prostate', 'fundus', 'MNMS', 'BUSI')}
    # the gates table: M&Ms and BUSI paired (as S24), the Prostate split-1337 Welch line (as S24), and the two new paired lines of rule 3
    G, grow = {}, []
    for ds, dn in (('MNMS', 'mnms'), ('BUSI', 'busi')):
        d = [f'adaptive_cmp_{dn}_daep_s{s}' for s in SEEDS]; G[ds] = {}
        for k in CK:
            G[ds][k] = s24.paired([R24[(ds, n)][k] for n in d], [R24[(ds, ref_name(ds, s))][k] for s in SEEDS]); x = G[ds][k]
            grow.append([ds, 'paired over seeds 1337/42/2024', k, f'{100 * x["a_mean"]:.2f}', f'{100 * x["a_std"]:.2f}', f'{100 * x["b_mean"]:.2f}', f'{100 * x["b_std"]:.2f}', f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}',
                         ';'.join(f'{100 * v:+.2f}' for v in x['per_seed'])])
    dP = ['v18A_prostate_daep_default_s1337', 'v18A_prostate_daep_default_s1337_rep2', 'v18A_prostate_daep_default_s1337_rep3']
    tP = ['v18B_prostate_fixmatch_split1337_init1337', 'v18B_prostate_fixmatch_split1337_init43', 'v18B_prostate_fixmatch_split1337_init44']
    G['prostate_split1337'] = {}
    for k in CK:
        x, y = np.array([R24[('prostate', n)][k] for n in dP]), np.array([R24[('prostate', n)][k] for n in tP]); t, p = stats.ttest_ind(x, y, equal_var=False)
        G['prostate_split1337'][k] = {'a_mean': float(x.mean()), 'b_mean': float(y.mean()), 'delta': float(x.mean() - y.mean()), 'p': float(p)}
        grow.append(['prostate', 'three trainings of each gate, labeled split 1337 (Welch)', k, f'{100 * x.mean():.2f}', f'{100 * x.std(ddof=1):.2f}', f'{100 * y.mean():.2f}', f'{100 * y.std(ddof=1):.2f}', f'{100 * (x.mean() - y.mean()):+.2f}', f'{p:.3f}', ''])
    for ds in DATASETS:
        for k in CK:
            x = G3[ds][k]
            grow.append([ds, 'paired over seeds 1337/42/2024 (DAEP: code-matched trainings, D-30 + S06)', k, f'{100 * x["a_mean"]:.2f}', f'{100 * x["a_std"]:.2f}', f'{100 * x["b_mean"]:.2f}', f'{100 * x["b_std"]:.2f}',
                         f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}', ';'.join(f'{100 * v:+.2f}' for v in x['per_seed'])])
    wcsv(os.path.join(a.out, 'table_selection_gates_s25.csv'), grow, ['dataset', 'design', 'checkpoint', 'daep_mean', 'daep_std', 'threshold_mean', 'threshold_std', 'daep_minus_threshold', 'p', 'per_seed'])
    for ds in ('MNMS', 'BUSI', 'prostate_split1337'):
        for k in CK:
            assert abs(G[ds][k]['delta'] - S24F['gates'][ds][k]['delta']) < 1e-9 and abs(G[ds][k]['p'] - S24F['gates'][ds][k]['p']) < 1e-9, (ds, k)
    need(True, 'the M&Ms, BUSI and Prostate split-1337 gate lines reproduce facts_s24_selection.json (delta and p to 1e-9)')
    F['gates'] = {**{ds: {k: {kk: vv for kk, vv in G[ds][k].items()} for k in CK} for ds in G}, **{ds: F['rule3'][ds] for ds in DATASETS}}
    # the baseline rows at the three checkpoints, now with the released-form DEviS rows (D-28, D-29)
    B, brow = {}, []
    names = [('v18B_{ds}_softmax_fixmatch_s{s}', 'softmax_fixmatch', 'softmax head, threshold gate (control)'), ('v18B_{ds}_mcdropout2_s{s}', 'mcdropout2', 'MC-dropout gate'),
             ('v18B_{ds}_dropoutctl2_fixmatch_s{s}', 'dropoutctl2_fixmatch', 'dropout only (control)'), ('v18B_{ds}_devisobj_threshold_s{s}', 'devisobj_threshold', 'DEviS objective (published form), threshold gate'),
             ('v18B_{ds}_devis_s{s}', 'devis', 'DEviS objective (published form), belief-mass gate'), ('v18B_{ds}_medl_s{s}', 'medl', 'MEDL (headline student)'),
             ('v18D_{ds}_devisrel_threshold_s{s}', 'devisrel_threshold', 'DEviS objective (released form), threshold gate'), ('v18E_{ds}_devisrel_belief_s{s}', 'devisrel_belief', 'DEviS objective (released form), belief-mass gate')]
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        for pat, m, lab in names:
            runs = [pat.format(ds=ds, s=s) for s in SEEDS]
            if not all((ds, r_) in R24 for r_ in runs):
                continue
            B[(ds, m)] = {}
            for k in CK:
                x = s24.paired([R24[(ds, r_)][k] for r_ in runs], [R24[(ds, ref_name(ds, s))][k] for s in SEEDS]); B[(ds, m)][k] = x
                brow.append([ds, m, lab, k, f'{100 * x["a_mean"]:.2f}', f'{100 * x["b_mean"]:.2f}', f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}', int(x['meets']), ';'.join(f'{100 * v:+.2f}' for v in x['per_seed'])])
    wcsv(os.path.join(a.out, 'table_selection_baselines_s25.csv'), brow, ['dataset', 'method', 'row', 'checkpoint', 'mean', 'threshold_reference_mean', 'delta', 'p_paired', 'meets_criterion', 'per_seed_delta_1337_42_2024'])
    F['baselines'] = {f'{ds}:{m}': v for (ds, m), v in B.items()}
    for key, v in S24F['baselines'].items():
        ds, m = key.split(':')
        for k in CK:
            assert abs(B[(ds, m)][k]['delta'] - v[k]['delta']) < 1e-9 and abs(B[(ds, m)][k]['p'] - v[k]['p']) < 1e-9, (key, k)
    need(True, 'every baseline row of facts_s24_selection.json is reproduced at the three checkpoints (delta and p to 1e-9); the released-form rows (D-28, D-29) are added')
    for ds in DATASETS:
        r28 = B[(ds, 'devisrel_threshold')]['best']; f28 = F23[f'{ds}_rule']; r29 = B[(ds, 'devisrel_belief')]['best']; f29 = F24[f'{ds}_rule3_vs_threshold_reference']
        assert abs(r28['delta'] - f28['delta']) < 1e-9 and abs(r28['p'] - f28['p']) < 1e-9 and abs(r29['delta'] - f29['delta']) < 1e-9 and abs(r29['p'] - f29['p']) < 1e-9, ds
    need(True, 'the released-form rows at the best checkpoint reproduce the rule rows of facts_s23.json (D-28) and facts_s24.json (D-29)')
    ua = ('mcdropout2', 'devisobj_threshold', 'devis', 'medl', 'devisrel_threshold', 'devisrel_belief')
    need(all(not B[(ds, m)][k]['meets'] for ds in DATASETS for m in ua for k in CK),
         'no uncertainty-aware alternative (MC-dropout gate, DEviS objective in either form under either gate, MEDL) meets the criterion on Prostate or Fundus at any of the three checkpoints')
    meets = sorted((ds, m, k) for (ds, m) in B for k in CK if B[(ds, m)][k]['meets'])
    F['rows_meeting_criterion_anywhere'] = meets
    need(meets == [('prostate', 'softmax_fixmatch', 'swad')], 'across all baseline rows, datasets and checkpoints the criterion is met in exactly one place: the softmax-head control on Prostate with the final SWAD average')
    F['released_rows_three_checkpoints'] = {f'{ds}:{m}': {k: {'delta_points': round(100 * B[(ds, m)][k]['delta'], 2), 'p': round(B[(ds, m)][k]['p'], 3)} for k in CK} for ds in DATASETS for m in ('devisrel_threshold', 'devisrel_belief')}

    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s25.json'), 'w'), indent=1, default=float)
    write_adjudication(F, a.out)
    print(f'[make_selection_s25] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


def pct(x):
    return f'{100 * x:.2f}'


def write_adjudication(F, out):
    L = []
    L.append('# ADJUDICATION_S25 (S26, 2026-10-14) — the S25 sendback: D-30, the five code-matched DAEP trainings (Prostate seeds 42 / 2024, Fundus seeds 1337 / 42 / 2024), read by the seven rules of `DOCS/D30_PREREGISTRATION_S25.md`')
    L.append('')
    L.append(f'Generated by `make_selection_s25.py` from `facts_s25.json` ({len(F["assertions_passed"])} assertions passed). Numbers below are copied from the facts file; nothing is typed by hand. '
             'D-30 was the last training before resubmission: nothing is added on the strength of what it shows.')
    L.append('')
    R = F['runner']
    L.append('## 1. Integrity (rules 1 and 2)')
    L.append(f'- Sendback: {F["manifest_files_verified"]} files in {F["sendback_parts"]} collector parts, each verified against the collector manifests (size, sha256). Runner: {R["jobs"]} jobs, one attempt each, {R["final_pass"]} PASS; '
             f'first start {R["first_start"]}, last end {R["last_end"]} ({R["wall_clock_hours"]} h wall clock; per job ' + ', '.join(f'{k.split("_", 1)[1]} {v} h' for k, v in sorted(R["hours_per_job"].items())) + '). '
             'No failure branch, no relaunch, no first-attempt folder. The server self-test passed and was written to a file; the Fundus smoke run passed; the harness measured the five teachers (ran 5, failed 0).')
    L.append('- Every run: one training from iteration 0 to the end, no resume event, no skipped iteration, R-34 screen ok (no training failure); configuration = the DAEP gate with the headline settings; '
             'the recorded arguments equal those of `v18A_prostate_daep_default_s1337` except the dataset block, the seed, the name and the GPU index; code snapshot, recorded hashes and the release code are the same bytes '
             f'(trainer `{F["code_sha256"]["train_fourierfuse_v18_revision.py"]}`, modules `{F["code_sha256"]["revision_modules.py"]}`); labeled slices identical to those of the threshold reference of the same seed '
             f'(Fundus seed 1337: {F["split_assertion_fundus_s1337"]}, so rule 3 pairs all three seeds on both datasets); the measured teacher is the EMA of the best checkpoint of the same log.')
    L.append('')
    L.append('## 2. The gate comparison at the three checkpoints (rule 3, the purpose of D-30)')
    for ds in DATASETS:
        g = F['rule3'][ds]
        L.append(f'**{ds.capitalize()}** (DAEP = code-matched trainings: seed 1337 the first default run of S06, seeds 42 / 2024 the D-30 runs on Prostate; all three D-30 runs on Fundus; threshold = the within-v18 references).')
        for k, lab in (('best', 'best checkpoint'), ('last', 'last iteration'), ('swad', 'final SWAD average')):
            x = g[k]
            L.append(f'- {lab}: DAEP {pct(x["a_mean"])} ± {pct(x["a_std"])} (' + ' / '.join(pct(v) for v in x['daep']) + f') against threshold {pct(x["b_mean"])} ± {pct(x["b_std"])} (' + ' / '.join(pct(v) for v in x['threshold']) + f'): '
                     f'Δ = {100 * x["delta"]:+.2f} points (per seed ' + ', '.join(f'{100 * v:+.2f}' for v in x['per_seed']) + f'), paired t = {x["t"]:.2f}, p = {x["p"]:.3f} → **{x["verdict"]}**.')
        L.append(f'- The comparison **{"survives" if g["survives_selection_rule"] else "does not survive"} the selection rule** (verdict at the last iteration and under the final SWAD average {"equals" if g["survives_selection_rule"] else "differs from"} the verdict at the best checkpoint).')
    L.append('')
    L.append('## 3. Code-matched values against Table 4 (rule 4; Table 4 keeps its values)')
    for ds in DATASETS:
        c = F['rule4'][ds]
        L.append(f'- {ds.capitalize()}: code-matched best values ' + ' / '.join(pct(c['per_seed'][str(s)]['code_matched']) for s in SEEDS) + ' against the submission digests ' + ' / '.join(pct(c['per_seed'][str(s)]['digest']) for s in SEEDS) +
                 ' (seeds 1337 / 42 / 2024; differences ' + ', '.join(f'{c["per_seed"][str(s)]["diff_points"]:+.2f}' for s in SEEDS) + f' points); three-seed mean {c["mean_code_matched"]:.2f} against the Table 4 mean {c["table4_mean"]:.2f} '
                 f'({c["mean_minus_table4"]:+.2f} points). One-point consequence: **{"triggered" if c["one_point_consequence_triggered"] else "not triggered"}** → the text says by how much the code-matched runs reproduce the row and draws nothing further.')
    y = F['rule4']['yardsticks']
    L.append(f'- Yardsticks on record, reproduced from the logs: five seed-1337 re-runs within {y["abs_min_max"][0]:.2f}–{y["abs_min_max"][1]:.2f} points of their digests (§4.2); the s42 Prostate threshold reference {y["threshold_reference_s42_vs_digest_points"]:+.2f} against its digest. '
             'The Prostate s42 DAEP run lies ' + f'{F["rule4"]["prostate"]["per_seed"]["42"]["diff_points"]:+.2f}' + ' points from its digest, above both yardsticks; the pre-registration expected about 0.7. No "replication succeeded / failed" verdict is written: three pairs cannot establish equality.')
    L.append('')
    L.append('## 4. The Fundus optic cup (rule 5; Table 5 keeps its values)')
    for cls in ('cup', 'disc'):
        c = F['rule5'][cls]
        L.append(f'- {cls.capitalize()}, DAEP minus threshold over the 12 (domain, seed) cells, code-matched runs: ' + '; '.join(f'{lab} pooled Δ {c[k]["pooled_delta"]:+.2f} (sd {c[k]["std"]:.2f}, +/− {c[k]["n_pos"]}/{c[k]["n_neg"]}, t12 {c[k]["t12"]:.2f} p {c[k]["p_t12"]:.3f}, tD {c[k]["tD"]:.2f} p {c[k]["p_tD"]:.3f})'
                 for k, lab in (('best', 'best checkpoint'), ('last', 'last iteration'), ('swad', 'final SWAD'))) + f'. Table 5 as printed (original runs, best checkpoint): pooled Δ {c["table5_as_printed"]["pooled"]:+.2f}, tD p {c["table5_as_printed"]["p_tD"]}.')
    L.append(f'- The reading "the cup shows no advantage for evidential gating" **{"survives" if F["rule5"]["cup"]["survives"] else "does not survive"}**: at neither alternative checkpoint does the cup give |pooled Δ| ≥ 0.5 with tD p < 0.05.')
    L.append('')
    L.append('## 5. The signal of the five new teachers (rule 6) and the identity (S9)')
    S6 = F['rule6']
    L.append(f'- Central-80 % range of u_epi: at most {S6["range_max"]:.2e} (the 1e-2 bound is {S6["range_factor_below_bound_min"]:.0f} times above it). Boundary-band AUROC {S6["band_min_max"][0]:.3f}–{S6["band_min_max"][1]:.3f}; '
             f'{5 - S6["n_band_above_070"]} of 5 teachers at or below the 0.70 line, {S6["n_band_above_070"]} above it (largest excess {S6["band_excess_max"]:+.3f}). Per teacher (band mean ≤ 0.70 and range ≤ 1e-2 → the collapse persists): '
             + ', '.join(f'{k} {v["band"]:.4f} ({v["margin"]:+.4f}, {"persists" if v["persists"] else "above the line"})' for k, v in S6['per_teacher'].items()) + f'; {S6["n_persists"]} of 5 by the rule as a hard line. '
             f'All-pixel AUROC {S6["auroc_all_min_max"][0]:.3f}–{S6["auroc_all_min_max"][1]:.3f}. Threshold references of the same seeds: band ' + ', '.join(f'{ds} {S6["per_dataset"][ds]["ref_band_min_max"][0]:.3f}–{S6["per_dataset"][ds]["ref_band_min_max"][1]:.3f}' for ds in DATASETS) + '.')
    L.append(f'- Identity of Eqn. 12 (S9 tolerances): holds on {S6["n_identity_holds"]} of 5 (largest all-pixel gap {S6["gap_all_max"]:.4f}, largest band gap {S6["gap_band_max"]:.4f}); per dataset ' +
             '; '.join(f'{ds} ' + ', '.join('holds' if h else 'fails' for h in S6['per_dataset'][ds]['identity']) for ds in DATASETS) + '.')
    C = F['s9_counts']
    L.append(f'- S9 counts after D-28, D-29 and D-30: {C["n_all_s12"]} → {C["n_all_with_d28_d29"]} → {C["n_all_with_d28_d29_d30"]} checkpoints; testbed objective {C["n_testbed_objective_s12"]} → {C["n_testbed_objective_with_d30"]}, '
             f'identity fails on {C["n_testbed_identity_fails_s12"]} → {C["n_testbed_identity_fails_with_d30"]}; its band AUROC {C["testbed_band_min_max_s12"][0]:.2f}–{C["testbed_band_min_max_s12"][1]:.2f} → '
             f'{C["testbed_band_min_max_with_d30"][0]:.2f}–{C["testbed_band_min_max_with_d30"][1]:.2f}; DEviS / MEDL {C["n_devis_medl_s12"]} → {C["n_devis_medl_with_d28_d29"]}, identity fails on {C["n_devis_medl_identity_fails_s12"]} → {C["n_devis_medl_identity_fails_with_d28_d29"]}.')
    L.append('')
    L.append('## 6. Descriptives (rule 7)')
    for ds in DATASETS:
        d = F[f'{ds}_descriptive']
        L.append(f'- {ds.capitalize()}: best minus last-iteration Dice ' + ' / '.join(f'{d[str(s)]["best_minus_last_points"]:.2f}' for s in SEEDS) + ' points (seeds 1337 / 42 / 2024; threshold references ' +
                 ' / '.join(f'{d[str(s)]["ref_best_minus_last_points"]:.2f}' for s in SEEDS) + '); best checkpoints at ' + ' / '.join(f'{100 * d[str(s)]["best_frac"]:.0f} %' for s in SEEDS) + ' of training' +
                 (f'; D-28 {" / ".join(f"{x:.1f}" for x in d["d28_best_minus_last"])}, D-29 {" / ".join(f"{x:.1f}" for x in d["d29_best_minus_last"])}' if ds == 'prostate' else '') + '.')
        L.append(f'- {ds.capitalize()}, per-domain Dice at the best checkpoint: ' + '; '.join(f's{s}: ' + ', '.join(f'{k} {v:.1f}' for k, v in d[str(s)]['at_best'].items()) for s in SEEDS) + '.')
    L.append('')
    L.append('## 7. The selection tables regenerated with every described run counted (the N-61 re-run)')
    S = F['size']; S0 = F['size_s24']
    L.append('- Counted runs (complete, no training failure, whole-run ramp): ' + ', '.join(f'{ds} {S[ds]["n"]} (S24: {S0[ds]["n"]})' for ds in ('prostate', 'fundus', 'MNMS', 'BUSI')) + f'; complete logs held {F["n_logs"]}.')
    L.append('- Best minus last, median / maximum / runs above 5 points: ' + '; '.join(f'{ds} {S[ds]["bl_median"]:.2f} / {S[ds]["bl_max"]:.2f} / {S[ds]["bl_n_gt5"]} (S24: {S0[ds]["bl_median"]:.2f}, {S0[ds]["bl_n_gt5"]})' for ds in ('prostate', 'fundus', 'MNMS', 'BUSI')) + '.')
    L.append('- Released-form DEviS rows at the three checkpoints (Δ against the threshold reference, p): ' + '; '.join(f'{k} ' + ', '.join(f'{ck} {v[ck]["delta_points"]:+.2f} ({v[ck]["p"]:.2f})' for ck in CK) for k, v in F['released_rows_three_checkpoints'].items()) + '.')
    L.append(f'- The criterion of §4.4 is met in exactly one place across all rows, datasets and checkpoints: {F["rows_meeting_criterion_anywhere"]} (the softmax control, as in S24).')
    L.append('')
    L.append('## 8. Verdict for the text (N-61)')
    L.append('- §4.2: the sentence "Where the logs allow a check … on M&Ms, on BUSI and on one Prostate split" becomes a sentence over the four benchmarks: the difference between the two gates stays within seed noise at the best checkpoint, at the last iteration and under the final SWAD average on all four, '
             'with the statement that on Prostate (seeds 42, 2024) and on Fundus the DAEP side of the check is a re-training in the released code, not the run of Table 4. §5.4: the clause "covers M&Ms, BUSI and one Prostate split only" is removed.')
    L.append('- S7 "The two gates": Prostate and Fundus gain their paired three-checkpoint lines (' + '; '.join(f'{ds} ' + ', '.join(f'{100 * F["rule3"][ds][k]["delta"]:+.2f} (p {F["rule3"][ds][k]["p"]:.2f})' for k in CK) for ds in DATASETS) +
             '); the Welch line on split 1337 stays; the code-matched values against Table 4 (rule 4) and the cup recomputation (rule 5) are stated there.')
    L.append('- Table 4 and Table 5 keep their values. §4.4: no change (no verdict changed). What may not be written: that the original runs of Table 4 were checked (they were re-trained); "replication succeeded"; any per-seed difference below half a point read as an effect.')
    L.append('')
    L.append('## Assertions passed')
    L.extend(f'- {s}' for s in F['assertions_passed'])
    open(os.path.join(out, 'ADJUDICATION_S25.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
