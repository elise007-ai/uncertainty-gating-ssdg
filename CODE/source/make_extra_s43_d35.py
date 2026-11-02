#!/usr/bin/env python3
"""
make_extra_s43_d35.py (S45, 2026-11-01) -- reader of D-35, the gate comparison under a short consistency ramp (v18J_<ds>_<arm>_ramp10_s<seed>,
trainer v12), assertion-first (R-48); nothing typed by hand (R-14). Rules: DOCS/D35_PREREGISTRATION_S43.md, read exactly as written.
One collection, RETURNS/S43 (17 collector parts, 213 files). CPU only, no torch: the consistency schedule is the trainer's own function (ast).
  python make_extra_s43_d35.py --returns <RETURNS> --derived <derived_v04_S12> --release <release_s12> --out <derived_v04_S12/extra_s43>
Writes table_d35_{integrity,coverage,rule,arms,checkpoints,signal}_s43.csv, table_{band_generality,band_summary,signal_identity}_s43.csv, facts_s43.json.
"""
from __future__ import annotations
import argparse, ast, csv, glob, json, os, re, shutil, sys, tempfile, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_derived_v02 as v02                      # noqa: E402
import make_derived_v04 as v04                      # noqa: E402
import make_selection_s24 as s24                    # noqa: E402
from utils import ramps                             # noqa: E402
from make_extra_s23 import need, sha, wcsv, find_log, split_of, rule, SEEDS, DATASETS, MAXIT, RULE_BAND, RULE_RANGE, PASSED   # noqa: E402
from make_selection_s25 import cmd_pairs, ref_name  # noqa: E402
from make_extra_s29 import tost                     # noqa: E402
from make_extra_s30_d32 import verdict, WIN_RX      # noqa: E402
from make_extra_s41_d34 import verify_parts, HDR_RX  # noqa: E402

V12 = '915347736f4855da'
HASHES = (V12, 'a6fe4357a39d0020', '3e214990fca90428', '2781aa2b14491838', 'f0658fbf23e2af7e', '3ca1a786349e21bf', '932540340a508d4e')
ARMS = ('fixmatch', 'daep', 'ungated', 'oracle')
METHOD = {'fixmatch': 'fixmatch', 'daep': 'daep', 'ungated': 'percentile_gate', 'oracle': 'oracle_gate'}
WHOLE = {'fixmatch': ref_name, 'daep': lambda d, s: f'v18A_{d}_daep_default_s{s}', 'ungated': lambda d, s: f'v18G_{d}_ungated_s{s}',
         'oracle': lambda d, s: f'v18H_{d}_oracle_s{s}'}             # the whole-run run of each arm, benchmark and seed (pre-registration)
D13 = {'fixmatch': 'v18R_prostate_fixmatch_ramp10_s1337', 'daep': 'v18R_prostate_daep_ramp10_s1337'}
RULES = (('rule1_daep_minus_threshold', 'daep'), ('rule2_oracle_minus_threshold', 'oracle'), ('rule3_nogate_minus_threshold', 'ungated'))
CW_RX, NF_RX = re.compile(r'\] iter (\d+): loss=.*?, cw=([\d.]+),'), re.compile(r'\[iter (\d+)\] non-finite')
SCHED = 'ramp_frac=0.1 (short per-iteration ramp (D-13))'
name = lambda d, arm, s: f'v18J_{d}_{arm}_ramp10_s{s}'
wmean = lambda v: sum(v) / len(v)                   # the trainer's own mean (statistics.avg = sum(record) / num), not numpy's pairwise sum
f2 = lambda x: f'{x:.2f}'
pts = lambda xs: ';'.join(f'{100 * v:+.2f}' for v in xs)


def windows(trainer):
    """{T: {iteration logged: the 500 values of cw averaged into that window}} from the trainer's own functions (as make_jobs_d35_s43.py)"""
    tree = ast.parse(open(trainer, encoding='utf-8').read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('get_current_consistency_weight', 'compute_consistency_weight')]
    need(len(fn) == 2, 'the two schedule functions are found in trainer v12')
    ns = {'ramps': ramps, 'args': types.SimpleNamespace(consistency=1.0, consistency_rampup=150.0, consistency_weight_cap=0.5, consistency_ramp_frac=0.10)}
    exec(compile(ast.Module(body=fn, type_ignores=[]), trainer, 'exec'), ns); cw = ns['compute_consistency_weight']
    return {T: {500 * k: [float(cw(t, T)) for t in range(500 * (k - 1), 500 * k)] for k in range(1, T // 500 + 1)} for T in sorted(set(MAXIT.values()))}


def cap_points(trainer):
    """the first iteration at which the consistency weight reaches its cap, per run length, for the short ramp (0.10) and the whole-run schedule"""
    tree = ast.parse(open(trainer, encoding='utf-8').read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('get_current_consistency_weight', 'compute_consistency_weight')]
    ns = {'ramps': ramps, 'args': types.SimpleNamespace(consistency=1.0, consistency_rampup=150.0, consistency_weight_cap=0.5, consistency_ramp_frac=0.10)}
    exec(compile(ast.Module(body=fn, type_ignores=[]), trainer, 'exec'), ns); out = {}
    for frac in (0.10, -1.0):
        ns['args'].consistency_ramp_frac = frac
        out[frac] = {T: next(t for t in range(T + 1) if ns['compute_consistency_weight'](t, T) == 0.5) for T in sorted(set(MAXIT.values()))}
    return out


def holm(ps):
    o, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
from make_extra_s23 import LBDOM, SPLIT_RX, f4     # noqa: E402
LABEL = {'fixmatch': 'the threshold gate', 'daep': 'the evidential gate', 'ungated': 'no gate', 'oracle': 'the oracle gate'}


def main():
    ap = argparse.ArgumentParser()
    for k in ('returns', 'derived', 'release', 'tools', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S43 = os.path.join(a.returns, 'S43'); src = os.path.join(a.release, 'CODE', 'source'); TR = os.path.join(src, 'v12', 'train_fourierfuse_v18_revision.py')
    rt = lambda *p: open(os.path.join(S43, *p), encoding='utf-8', errors='replace').read()
    F = {'session': 'S45', 'reader': 'make_extra_s43_d35.py', 'preregistration': 'DOCS/D35_PREREGISTRATION_S43.md'}
    code_sha = {'train_fourierfuse_v18_revision.py': sha(TR)}
    for n in ('revision_modules.py', 'fourierfuse_v16_modules.py', 'diag_uncertainty_harness_v18.py'):
        code_sha[n] = sha(os.path.join(src, n))
    F['code_sha256'] = {k: v[:16] for k, v in code_sha.items()}
    need(F['code_sha256']['train_fourierfuse_v18_revision.py'] == V12, 'the release copy of trainer v12 is 915347736f4855da (unchanged for D-35)')
    F41 = json.load(open(os.path.join(a.derived, 'extra_s41', 'facts_s41.json')))
    # ------------------------------------------------ the collection (RETURNS/S43, filed by pins_tmp/in43.py)
    listed = verify_parts(S43, 213, 'S43')
    need(not any(re.search(r'v18[A-I]_', p) for p in listed), 'the collection carries no file of an earlier training: only the D-35 runs, the smoke run and the step files')
    held = {x[0]: x[2] for x in (l.rstrip('\n').split('\t') for l in open(os.path.join(a.tools, 'HELD_MANIFEST_S42.txt'), encoding='utf-8', errors='replace'))
            if len(x) == 3 and len(x[2]) == 64}
    sk = [l.split('\t') for l in rt('SKIPPED_HELD.txt').splitlines() if l.strip() and not l.startswith('#')]
    need(bool(sk) and all(len(x) == 2 and held.get(x[0]) == x[1] for x in sk) and not any('v18J_' in x[0] for x in sk),
         'every file the collector skipped is one I hold, byte for byte (HELD_MANIFEST_S42.txt); none is a D-35 file')
    F['collection'] = {'files': len(listed), 'skipped_held': len(sk), 'held_not_in_tree': len(set(held) - {x[0] for x in sk})}
    need(sha(os.path.join(S43, 'jobs', 'jobs_D35.txt'))[:16] == HASHES[4] and sha(os.path.join(S43, 'jobs', 'jobs_D35smoke.txt'))[:16] == HASHES[5],
         'the job lists that ran are the registered ones (tools S43)')
    # ------------------------------------------------ RUNBOOK S43 steps 0-2, the runner and the harness, on record
    CL = os.path.join(S43, 'console_logs', 'S43'); cl = lambda n: rt('console_logs', 'S43', n)
    pf = cl('preflight.txt')
    need(len(re.findall(r'BSPC_sendback_S41_2026-10-26_p[1-4]of4\.zip', pf)) == 4 and all(h in pf for h in HASHES) and '1467 HELD_MANIFEST_S42.txt' in pf
         and re.search(r'\n48\n', pf) is not None and 'job_runner.lock: No such file or directory' in pf
         and "cannot access '../model/*/train_fourierfuse_v16/v18J_*'" in pf and re.search(r'\n0\s*$', pf) is not None,
         'step 0 on record: the four S41 parts, the seven hashes (trainer v12 unchanged), the 1,467-line held manifest with 48 v18I_ lines, no runner lock, no D-35 run before the launch')
    F['preflight_gpu_mib'] = {int(g): int(m) for g, m in re.findall(r'^(\d), (\d+) MiB$', pf, re.M)}
    need(F['preflight_gpu_mib'] == {0: 0, 1: 0}, 'step 0: both GPUs at 0 MiB')
    need(re.fullmatch(r'per run \d+ KB, free \d+ KB\nDISK OK\n?', cl('disk.txt')) is not None, 'step 0 on record: DISK OK')
    need('self-test PASSED' in cl('selftest_revision_modules.txt') and cl('compile_trainer_v12.txt').strip() == 'COMPILE OK v12'
         and cl('test_ramp_schedule.txt').strip().split('\n')[-1] == '[test] ALL PASSED', 'step 1 on record: the module self-test passed, trainer v12 compiled, the schedule test passed')
    sm = rt('runs', 'fundus', 'v18smoke_fundus_ramp10', 'log.txt'); scw = ' '.join(f'cw={c}' for _, c in CW_RX.findall(sm))
    need('DONE: 1/1 PASS' in cl('runner_jobs_D35smoke.log') and 'method=daep' in sm and sm.count(SCHED) == 1 and scw == 'cw=0.429 cw=0.500',
         f'step 2 on record: the smoke run passed; it names the evidential gate and the short ramp once; its two windows log "{scw}" as registered')
    rl = cl('runner_jobs_D35.log'); st = list(csv.DictReader(open(os.path.join(CL, 'status_jobs_D35.csv'), encoding='utf-8')))
    need(rl.count('[runner] 24 jobs from jobs/jobs_D35.txt') == 1 and '[runner] 24 to run, 0 skipped' in rl and rl.count('[preflight] GREEN') == 1
         and '[preflight] FAILED' not in rl and '[resume]' not in rl and '[restart]' not in rl and '[runner] DONE: 24/24 PASS' in rl,
         'runner: one launch of the 24 jobs, preflight GREEN, nothing resumed or restarted, DONE 24/24 PASS')
    need(len(st) == 24 and {r['job'] for r in st} == {name(d, x, s) for d in DATASETS for x in ARMS for s in SEEDS}
         and all(r['exit_code'] == '0' and r['postflight'] == 'PASS' for r in st), 'status: each of the 24 jobs once, exit code 0, postflight PASS')
    F['runner'] = {'jobs': 24, 'first_start': min(r['start'] for r in st), 'last_end': max(r['end'] for r in st)}
    need('[harness] ran 24 (failed 0), skipped 0 existing' in cl('diag_harness.stdout.log'), 'step 4 on record: the harness measured the 24 teachers')
    fa = sorted(p for p in listed if p.startswith('console_logs/S43_first_attempt/'))
    need(fa == ['console_logs/S43_first_attempt/state_after_fail.txt']
         and "cannot access '../model/*/train_fourierfuse_v16/v18J_*/'" in rt('console_logs', 'S43_first_attempt', 'state_after_fail.txt'),
         'the failure branch left only its state file, written while no D-35 run folder existed; no run was relaunched (rule 0 item 5 does not apply)')
    F['failure_branch_record'] = 'state_after_fail.txt only, written before any D-35 run folder existed; the runner launched the 24 jobs once'
    # ------------------------------------------------ the registered schedule, from the trainer's own function
    W = windows(TR)
    need(f'{wmean(W[60000][500]):.3f}' == '0.010' and f'{wmean(W[30000][500]):.3f}' == '0.016', 'the registered window means at iteration 500: 0.010 (Prostate), 0.016 (Fundus)')
    for T, first in ((60000, 4500), (30000, 2500)):
        need(min(i for i in W[T] if all(f'{wmean(W[T][j]):.3f}' == '0.500' for j in W[T] if j >= i)) == first, f'T = {T}: the window mean is 0.500 from iteration {first:,} on, as registered')
    for x, run in D13.items():
        L13 = {int(i): c for i, c in CW_RX.findall(open(find_log(a.returns, 'prostate', run)[0], errors='replace').read())}
        need(sorted(L13) == sorted(W[60000]) and all(L13[i] == f'{wmean(W[60000][i]):.3f}' for i in L13), f'{run} (D-13): the same computation reproduces its 120 logged cw values exactly')
    CP = cap_points(TR); pct = lambda fr: {f'{100 * CP[fr][T] / T:.1f}' for T in CP[fr]}
    need(pct(0.10) == {'6.3'} and {f'{100 * CP[-1.0][T] / T:.0f}' for T in CP[-1.0]} == {'63'},
         'the cap of the consistency weight is reached at 6.3 % of training under the short ramp and at 63 % under the whole-run schedule, on both run lengths (as registered)')
    F['schedule'] = {'cap_iter_short': CP[0.10], 'cap_iter_whole': CP[-1.0], 'cap_pct_short': pct(0.10).pop(), 'cap_pct_whole': '63'}
    # ------------------------------------------------ per run: rule 0 items 1-2 (provenance: asserted) and items 3-4 (outcomes: recorded, read by the rules)
    jobs = {l.split('\t')[0]: l.split('\t')[2] for l in rt('jobs', 'jobs_D35.txt').splitlines() if l.strip()}
    need(set(jobs) == {name(d, x, s) for d in DATASETS for x in ARMS for s in SEEDS}, 'jobs_D35.txt holds the 24 registered runs')
    RUN, rows_i, rows_v = {}, [], []
    for ds in DATASETS:
        Tm = MAXIT[ds]; Wm = {i: wmean(v) for i, v in W[Tm].items()}
        for sd in SEEDS:
            for arm in ARMS:
                run = name(ds, arm, sd); d = os.path.join(S43, 'runs', ds, run); log = os.path.join(d, 'log.txt')
                txt = open(log, errors='replace').read(); t = v04.screen_run(log); q = v02.parse_log(log)
                cfg = json.load(open(os.path.join(d, 'run_config.json'))); ar = cfg['args']
                ok_cfg = (ar['method'] == METHOD[arm] and ar['seed'] == sd and ar['max_iterations'] == Tm and ar['lb_domain'] == LBDOM[ds] and ar['lb_num'] == 20
                          and ar['head'] == 'evidential' and abs(ar['consistency_ramp_frac'] - 0.10) < 1e-12 and ar['resume'] == 'off' and ar['amp'] == 1
                          and (abs(ar['pg_keep_frac'] - 1.0) < 1e-12 if arm == 'ungated' else True))
                ok_sha = all(cfg[k] == code_sha[n] == sha(os.path.join(d, 'code', n)) for k, n in (('script_sha256', 'train_fourierfuse_v18_revision.py'),
                             ('revision_modules_sha256', 'revision_modules.py'), ('fourierfuse_v16_modules_sha256', 'fourierfuse_v16_modules.py')))
                hd = HDR_RX.findall(txt); jp = cmd_pairs('python train_fourierfuse_v18_revision.py ' + jobs[run])
                cmd_diff = sorted({k for h_ in hd for cp in [cmd_pairs(h_[h_.index('python'):])] for k in set(cp) | set(jp) if cp.get(k) != jp.get(k)})
                need(t['complete'] and t['parse_ok'] and t['n_evals'] == Tm // 500 and t['n_headers'] == 1 and len(hd) == 1 and not t['resumed_iters']
                     and not cfg.get('resume_events') and txt.count('Namespace(') == 1, f'{run}: one training from iteration 0 to the last, one launch, no resume (rule 0 item 2)')
                need(ok_cfg and f'method={METHOD[arm]}' in txt and txt.count(SCHED) == 1,
                     f'{run}: rule 0 item 1: the configuration and the log name {LABEL[arm]} and the short ramp; --resume off; the headline settings')
                need(set(cmd_diff) <= {'--gpu'}, f'{run}: its launch command equals its line of jobs_D35.txt (the runner adds only the GPU index)')
                need(ok_sha, f'{run}: rule 0 item 2: script_sha256 = trainer v12; code snapshot, recorded hashes and release code are the same bytes')
                wn = WHOLE[arm](ds, sd); wlog, wq = find_log(a.returns, ds, wn); spw = split_of(wlog)
                sps = [(int(m.group(1)), [int(x) for x in m.group(4).split(',')]) for m in SPLIT_RX.finditer(txt)]; sj = json.load(open(os.path.join(d, 'split.json')))
                need(len(sps) == 1 and sps[0][0] == sd and sps[0][1] == spw[1] == sj['lb_idxs'], f'{run}: rule 0 item 2: its labeled slices equal those of {wn}')
                me = json.load(open(os.path.join(S43, 'diag_out', f'{ds}__{run}__ema.json')))['meta']
                need(me['which'] == 'ema' and me['iter'] == q['best_iter'] and abs(me['best_avg_dice'] - q['best_avg']) < 1e-6 and run in me['ckpt']
                     and me['harness_sha256'] == code_sha['diag_uncertainty_harness_v18.py'] and me['n_models'] == 1,
                     f'{run}: the measured teacher is the EMA weights of the best checkpoint of this log, measured with the release harness')
                flag = v04.flag_of(t); need(flag == 'ok' or flag.startswith('DEGENERATE'), f'{run}: R-34 screen readable ({flag})')
                cw = {int(i): float(c) for i, c in CW_RX.findall(txt)}; skipw = {500 * (int(i) // 500 + 1) for i in NF_RX.findall(txt)}
                dev = max((abs(cw[i] - Wm[i]) for i in cw if i not in skipw), default=float('inf'))
                sched_ok = set(cw) <= set(Wm) and set(Wm) - set(cw) <= skipw and dev <= 0.001 + 1e-9; nf_frac = t['nf_n'] / Tm   # a skipped last iteration drops its line
                fail = ('DEGENERATE' if flag != 'ok' else 'UNSTABLE' if nf_frac > 0.01 else 'schedule not as registered' if not sched_ok else '')
                win = [(int(i), float(m)) for i, m, _ in WIN_RX.findall(txt)]; wwin = [(int(i), float(m)) for i, m, _ in WIN_RX.findall(open(wlog, errors='replace').read())]
                need(len(win) == Tm // 500 and len(wwin) == Tm // 500, f'{run} and {wn}: {Tm // 500} gate windows logged each')
                late = lambda w: [m for i, m in w if i > 0.2 * Tm]; early = lambda w: [m for i, m in w if i <= 0.2 * Tm]
                RUN[(ds, arm, sd)] = r = {'run': run, 'flag': flag, 'nf': t['nf_n'], 'nf_frac': nf_frac, 'sched_ok': sched_ok, 'cw_max_dev': dev, 'fail': fail,
                    'w_first': win[0][1], 'w_first_fifth_mean': float(np.mean(early(win))), 'w_after_mean': float(np.mean(late(win))),
                    'w_after_median': float(np.median(late(win))), 'whole_w_first': wwin[0][1], 'whole_w_first_fifth_mean': float(np.mean(early(wwin))),
                    'whole_w_after_mean': float(np.mean(late(wwin))),
                    'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': t['student_avg_last'], 'whole': wn, 'whole_best': wq['best_avg']}
                rows_i.append([ds, sd, arm, run, int(t['complete']), t['n_evals'], len(hd), t['nf_n'], flag, int(sched_ok), f'{dev:.4f}', fail or 'ok', ';'.join(cmd_diff), cfg['hostname']])
                rows_v.append([ds, sd, arm, run, f'{r["w_first"]:.3f}', f'{r["w_first_fifth_mean"]:.4f}', f'{r["w_after_mean"]:.4f}', f'{r["w_after_median"]:.4f}',
                               wn, f'{r["whole_w_first"]:.3f}', f'{r["whole_w_first_fifth_mean"]:.4f}', f'{r["whole_w_after_mean"]:.4f}'])
    wcsv(os.path.join(a.out, 'table_d35_integrity_s43.csv'), rows_i, ['dataset', 'seed', 'arm', 'run', 'complete', 'n_evals', 'n_launches', 'n_nonfinite', 'r34_flag',
         'schedule_as_registered', 'cw_max_abs_deviation', 'rule0_outcome', 'cmd_keys_differing_from_job_line', 'hostname'])
    wcsv(os.path.join(a.out, 'table_d35_coverage_s43.csv'), rows_v, ['dataset', 'seed', 'arm', 'run', 'w_mean_first_window', 'w_mean_first_fifth', 'w_mean_after_fifth',
         'w_mean_after_fifth_median', 'whole_run', 'whole_w_mean_first_window', 'whole_w_mean_first_fifth', 'whole_w_mean_after_fifth'])
    # ------------------------------------------------ rules 1-3 on the best checkpoint; Holm over the two benchmarks (reported alongside)
    rows_r = []
    for key, arm in RULES:
        res = {}
        for ds in DATASETS:
            A_ = [RUN[(ds, arm, s)]['best'] for s in SEEDS]; T_ = [RUN[(ds, 'fixmatch', s)]['best'] for s in SEEDS]
            r = rule(A_, T_); e = tost([100 * (x - y) for x, y in zip(A_, T_)]); r['verdict'] = verdict(r)
            need(abs(e['mean_points'] - 100 * r['delta']) < 1e-9, f'{ds} {key}: TOST and the paired test use the same differences')
            need((r['verdict'] != 'within seed noise') == (r['p'] < 0.05 and abs(r['delta']) >= 0.005), f'{ds} {key}: the verdict follows from p and delta as registered')
            bad = sorted({RUN[(ds, x, s)]['run'] + ': ' + RUN[(ds, x, s)]['fail'] for x in (arm, 'fixmatch') for s in SEEDS if RUN[(ds, x, s)]['fail']})
            r['read'] = not bad; r['not_read_because'] = bad; res[ds] = (r, e)
        for ds, p_h in zip(DATASETS, holm([res[ds][0]['p'] for ds in DATASETS])):
            r, e = res[ds]; vh = verdict({**r, 'p': p_h})
            F[f'{ds}_{key}'] = {**r, **e, 'p_holm': p_h, 'verdict_holm': vh}
            rows_r.append([ds, key, f4(r['mean']), f4(r['std']), f4(r['ref_mean']), f4(r['ref_std']), f'{100 * r["delta"]:+.2f}', pts(r['per_seed_delta']), f'{r["p"]:.4f}',
                           f'{p_h:.4f}', r['n_pos'], f'{r["sign_p_two_sided"]:.4f}', f'{e["ci90_points"][0]:+.2f};{e["ci90_points"][1]:+.2f}', f'{e["p_tost_0.5"]:.4f}',
                           f'{e["mde_points"]:.2f}', r['verdict'] if r['read'] else 'not read (rule 0)', vh, '; '.join(bad)])
    wcsv(os.path.join(a.out, 'table_d35_rule_s43.csv'), rows_r, ['dataset', 'rule', 'mean_dice', 'std_dice', 'threshold_mean', 'threshold_std', 'delta_points',
         'per_seed_delta_points_1337_42_2024', 'p_paired', 'p_holm_two_benchmarks', 'n_seeds_positive', 'sign_test_p_two_sided', 'ci90_points', 'p_tost_0.5', 'mde_points',
         'verdict', 'verdict_holm', 'rule0_failures'])
    ex = F['expectation'] = {'registered': 'DAEP within seed noise of the threshold gate on both benchmarks, and the oracle above it on at least one; no expectation for no gate'}
    for ds in DATASETS:
        ex[ds] = {'daep_within_noise': F[f'{ds}_rule1_daep_minus_threshold']['verdict'] == 'within seed noise',
                  'oracle_above': F[f'{ds}_rule2_oracle_minus_threshold']['verdict'] == 'above the threshold gate'}
    ex['holds'] = all(ex[d]['daep_within_noise'] for d in DATASETS) and any(ex[d]['oracle_above'] for d in DATASETS)
    # ------------------------------------------------ descriptive (rule 4): each arm under both schedules; the D-13 quantity; the D-13 pair
    F['arms'], F['d13_quantity_points'], rows_a = {}, {}, []
    for ds in DATASETS:
        for arm in ARMS:
            S_ = [RUN[(ds, arm, s)]['best'] for s in SEEDS]; Wh = [RUN[(ds, arm, s)]['whole_best'] for s in SEEDS]; x = s24.paired(S_, Wh)
            F['arms'][f'{ds}_{arm}'] = v = {'short_mean': float(np.mean(S_)), 'short_std': float(np.std(S_, ddof=1)), 'whole_mean': float(np.mean(Wh)),
                'whole_std': float(np.std(Wh, ddof=1)), 'short_minus_whole_points': 100 * x['delta'], 'p': x['p'], 'per_seed_points': [100 * y for y in x['per_seed']],
                'w_first': [RUN[(ds, arm, s)]['w_first'] for s in SEEDS], 'w_after_mean': [RUN[(ds, arm, s)]['w_after_mean'] for s in SEEDS],
                'whole_w_first': [RUN[(ds, arm, s)]['whole_w_first'] for s in SEEDS], 'whole_w_after_mean': [RUN[(ds, arm, s)]['whole_w_after_mean'] for s in SEEDS]}
            rows_a.append([ds, arm, f'{100 * v["short_mean"]:.2f}', f'{100 * v["short_std"]:.2f}', f'{100 * v["whole_mean"]:.2f}', f'{100 * v["whole_std"]:.2f}',
                           f'{v["short_minus_whole_points"]:+.2f}', f'{v["p"]:.3f}', pts(x['per_seed']), ';'.join(f'{y:.3f}' for y in v['w_first']), ';'.join(f'{y:.4f}' for y in v['w_after_mean'])])
        F['d13_quantity_points'][ds] = [100 * ((RUN[(ds, 'daep', s)]['best'] - RUN[(ds, 'fixmatch', s)]['best']) - (RUN[(ds, 'daep', s)]['whole_best'] - RUN[(ds, 'fixmatch', s)]['whole_best'])) for s in SEEDS]
    wcsv(os.path.join(a.out, 'table_d35_arms_s43.csv'), rows_a, ['dataset', 'arm', 'short_mean', 'short_std', 'whole_mean', 'whole_std', 'short_minus_whole_points', 'p_paired',
         'per_seed_points', 'w_mean_first_window_per_seed', 'w_mean_after_fifth_per_seed'])
    F['d13_pair'] = {x: {'d13_best': find_log(a.returns, 'prostate', run)[1]['best_avg'], 'd35_s1337_best': RUN[('prostate', x, 1337)]['best']} for x, run in D13.items()}
    for x, v in F['d13_pair'].items():
        v['d35_minus_d13_points'] = 100 * (v['d35_s1337_best'] - v['d13_best'])
        c13 = json.load(open(os.path.join(os.path.dirname(find_log(a.returns, 'prostate', D13[x])[0]), 'run_config.json'), encoding='utf-8'))
        v['d13_trainer_sha16'] = c13['script_sha256'][:16]
    need(all(v['d13_trainer_sha16'] != V12 for v in F['d13_pair'].values()), 'the two D-13 runs were trained with an earlier trainer version than v12')
    F['d13_shift_points'] = 100 * ((F['d13_pair']['daep']['d13_best'] - F['d13_pair']['fixmatch']['d13_best'])
                                   - (RUN[('prostate', 'daep', 1337)]['whole_best'] - RUN[('prostate', 'fixmatch', 1337)]['whole_best']))
    need(f2(F['d13_shift_points']) == '-0.67', f'the D-13 shift recomputed from the held logs is the -0.67 of Supplementary S8 ({F["d13_shift_points"]:+.4f})')
    # ------------------------------------------------ descriptive: rules 1-3 at the other two checkpoints; best minus last; the counts of the text (R-66)
    L = s24.load(a.returns); grp = lambda run, r: 'failure' if (r['flag'] != 'ok' or r['nf'] > 0) else 'short_ramp' if 'ramp10' in run else 'counted'
    need(len(L) == F41['n_logs'] + 24 and all((d, name(d, x, s)) in L for d in DATASETS for x in ARMS for s in SEEDS),
         f'complete logs: the {F41["n_logs"]} of facts_s41.json plus the 24 of D-35 (smoke runs excluded)')
    for (d, x, s), r in RUN.items():
        need(abs(L[(d, r['run'])]['best'] - r['best']) < 1e-12 and L[(d, r['run'])]['ret'] == 'S43', f'{r["run"]}: the checkpoint loader reads the same best value from RETURNS/S43')
    rows_c, F['checkpoints'] = [], {}
    for ds in DATASETS:
        for ck in ('best', 'last', 'swad'):
            for key, arm in RULES:
                A_ = [L[(ds, name(ds, arm, s))][ck] for s in SEEDS]; T_ = [L[(ds, name(ds, 'fixmatch', s))][ck] for s in SEEDS]
                need(all(v is not None for v in A_ + T_), f'{ds} {key}: a {ck} value on every run')
                x = s24.paired(A_, T_); F['checkpoints'][f'{ds}_{key}_{ck}'] = {'delta_points': 100 * x['delta'], 'p': x['p'], 'per_seed_points': [100 * v for v in x['per_seed']],
                                                                                   'criterion_met': bool(x['p'] < 0.05 and abs(x['delta']) >= 0.005)}
                rows_c.append([ds, key, ck, f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}', pts(x['per_seed'])])
        need(all(abs(F['checkpoints'][f'{ds}_{k}_best']['delta_points'] - 100 * F[f'{ds}_{k}']['delta']) < 1e-9 for k, _ in RULES), f'{ds}: the best-checkpoint lines equal rules 1-3')
    wcsv(os.path.join(a.out, 'table_d35_checkpoints_s43.csv'), rows_c, ['dataset', 'rule', 'checkpoint', 'delta_points', 'p_paired', 'per_seed_points'])
    sel = {}
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        b = [100 * (r['best'] - r['last']) for (d, run), r in L.items() if d == ds and grp(run, r) == 'counted']; S = F41['selection'][ds]
        need(len(b) == S['n'] and abs(float(np.median(b)) - S['median']) < 1e-9 and sum(v > 5 for v in b) == S['n_gt5'] and sum(v > 1 for v in b) == S['n_gt1'],
             f'{ds}: the selection counts of the text are unchanged (n {S["n"]}, median {S["median"]:.2f}): the D-35 runs are short-ramp runs, left aside like the D-13 pair')
        sel[ds] = dict(S)
    sr = sorted(run for (d, run), r in L.items() if grp(run, r) == 'short_ramp')
    f35 = sorted(r['run'] for r in RUN.values() if grp(r['run'], L[(r['run'].split('_')[1], r['run'])]) == 'failure')
    need(set(D13.values()) <= set(sr) and set(sr) - set(D13.values()) == {r['run'] for r in RUN.values()} - set(f35),
         f'the short-ramp runs left aside by the selection statistics: the two of D-13 and the D-35 runs that are not training failures ({len(sr)}; failures {f35})')
    F['selection'], F['n_short_ramp'], F['n_logs'] = sel, len(sr), len(L)
    F['best_minus_last_points'] = {f'{d}_{x}': [100 * (RUN[(d, x, s)]['best'] - RUN[(d, x, s)]['last']) for s in SEEDS] for d in DATASETS for x in ARMS}
    F['best_frac'] = {f'{d}_{x}': [RUN[(d, x, s)]['best_iter'] / MAXIT[d] for s in SEEDS] for d in DATASETS for x in ARMS}
    F['runs_trained'] = {'before_d35': F41['runs_trained']['total'], 'd35': 24, 'total': F41['runs_trained']['total'] + 24}
    # ------------------------------------------------ descriptive: the 24 teachers (v02 functions unchanged); the S9 counts with them added
    prev_id, prev_band, outp = {}, {}, os.path.abspath(a.out) + os.sep
    for pat, tgt in (('table_signal_identity*.csv', prev_id), ('table_band_summary*.csv', prev_band)):
        for p in sorted(glob.glob(os.path.join(a.derived, '**', pat), recursive=True)):
            if os.path.abspath(p).startswith(outp):
                continue
            for r in csv.DictReader(open(p, encoding='utf-8')):
                if tgt is prev_id:
                    prev_id.setdefault((r['dataset'], r['run']), r['identity_holds'])
                elif r['which'] == 'ema' and r['signal'] == 'u_epi':
                    prev_band.setdefault((r['dataset'], r['run']), (float(r['auroc_edt5_mean']), float(r['range80_all_mean'])))
    tb = [k for k in prev_id if not re.search(r'(?i)devis|medl', k[1]) and not k[1].startswith('ENSEMBLE3_') and k[1] != 'v18A_prostate_daep_act_exp_s1337']
    pb = {d: sum(k[0] == d for k in tb) for d in ('prostate', 'fundus', 'MNMS', 'BUSI')}
    need(len(prev_id) == F41['s9']['n_all'] == 124 and len(tb) == F41['s9']['n_testbed'] == 91 and sum(prev_id[k] == '1' for k in tb) == F41['s9']['n_testbed_identity_holds'] == 90
         and pb == F41['s9']['per_benchmark'], f'S9 before D-35: 124 checkpoints, 91 testbed objective ({pb}), identity on 90 of them')
    need(all(k in prev_band for k in tb) and (f2(min(prev_band[k][0] for k in tb)), f2(max(prev_band[k][0] for k in tb))) == ('0.56', '0.73'),
         'S9 before D-35: the band AUROC of the 91 testbed-objective checkpoints spans 0.56-0.73 (the text holds)')
    need(set(D13.values()) <= {k[1] for k in tb}, 'the two D-13 teachers are among the 91 testbed-objective checkpoints')
    H = {k: x for k, x in v02.load_harness(S43).items() if k[1].startswith('v18J_')}
    need(len(H) == 24 and all(k[2] == 'ema' for k in H), '24 harness JSONs, EMA teachers')
    tmp = tempfile.mkdtemp(); v02.band_tables(H, tmp); v02.signal_identity_table(H, tmp, {})
    for n in ('table_band_generality', 'table_band_summary', 'table_signal_identity'):
        shutil.copy(os.path.join(tmp, n + '.csv'), os.path.join(a.out, n + '_s43.csv'))
    bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_band_summary_s43.csv')))}
    idn = {r['run']: r['identity_holds'] == '1' for r in csv.DictReader(open(os.path.join(a.out, 'table_signal_identity_s43.csv')))}
    F['signal'], rows_g = {}, []
    for (ds, arm, sd), r in RUN.items():
        g = lambda sig, col: float(bs[(ds, r['run'], sig)][col])
        s = {'u_epi_all': g('u_epi', 'auroc_all_mean'), 'u_epi_band': g('u_epi', 'auroc_edt5_mean'), 'u_epi_range80': g('u_epi', 'range80_all_mean'),
             'conf_all': g('conf', 'auroc_all_mean'), 'conf_band': g('conf', 'auroc_edt5_mean'), 'identity_holds': idn[r['run']]}
        s['phaseA_rule_collapse_persists'] = bool(s['u_epi_band'] <= RULE_BAND and s['u_epi_range80'] <= RULE_RANGE); F['signal'][r['run']] = s
        rows_g.append([ds, sd, arm, r['run']] + [f'{v:.4g}' if isinstance(v, float) else int(v) for v in s.values()])
    wcsv(os.path.join(a.out, 'table_d35_signal_s43.csv'), rows_g, ['dataset', 'seed', 'arm', 'run'] + list(next(iter(F['signal'].values())).keys()))
    nb = [v['u_epi_band'] for v in F['signal'].values()]; allb = [prev_band[k][0] for k in tb] + nb
    F['s9'] = {'n_all': 124 + 24, 'n_testbed': 91 + 24, 'n_testbed_identity_holds': 90 + sum(v['identity_holds'] for v in F['signal'].values()),
               'per_benchmark': {d: pb[d] + (12 if d in DATASETS else 0) for d in pb}, 'testbed_band_min_max': [f2(min(allb)), f2(max(allb))],
               'd35_band_min_max': [f2(min(nb)), f2(max(nb))], 'd35_range_max': max(v['u_epi_range80'] for v in F['signal'].values()),
               'd35_phaseA_persists': sum(v['phaseA_rule_collapse_persists'] for v in F['signal'].values())}
    F['per_run'] = {r['run']: r for r in RUN.values()}
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s43.json'), 'w', encoding='utf-8'), indent=1, default=float)
    # ------------------------------------------------ console
    for key, _ in RULES:
        for ds in DATASETS:
            r = F[f'{ds}_{key}']
            print(f"[d35] {ds} {key}: {100 * r['mean']:.2f}+-{100 * r['std']:.2f} vs threshold {100 * r['ref_mean']:.2f}+-{100 * r['ref_std']:.2f}; delta {100 * r['delta']:+.2f} "
                  f"({pts(r['per_seed_delta'])}); p {r['p']:.4f} (Holm {r['p_holm']:.4f}); signs {r['n_pos']}/3; CI90 [{r['ci90_points'][0]:+.2f}, {r['ci90_points'][1]:+.2f}]; "
                  f"TOST {r['p_tost_0.5']:.3f}; MDE {r['mde_points']:.2f} -> {r['verdict']}{'' if r['read'] else ' [NOT READ: ' + '; '.join(r['not_read_because']) + ']'}"
                  f"{'' if r['verdict_holm'] == r['verdict'] else ' | Holm: ' + r['verdict_holm']}")
    for k, v in F['arms'].items():
        print(f"[arm] {k}: short {100 * v['short_mean']:.2f}+-{100 * v['short_std']:.2f} | whole {100 * v['whole_mean']:.2f}+-{100 * v['whole_std']:.2f} | short-whole "
              f"{v['short_minus_whole_points']:+.2f} (p {v['p']:.3f}) | w_first {v['w_first']} vs {v['whole_w_first']} | w_after {[round(x, 4) for x in v['w_after_mean']]} vs "
              f"{[round(x, 4) for x in v['whole_w_after_mean']]}")
    for k, v in F['checkpoints'].items():
        print(f"[ckpt] {k}: {v['delta_points']:+.2f} p {v['p']:.3f}{' (criterion met)' if v['criterion_met'] else ''}")
    print('[d13 quantity, points]', {k: [round(x, 2) for x in v] for k, v in F['d13_quantity_points'].items()}, '| D-13 pair', F['d13_pair'])
    print('[best-last, points]', {k: [round(x, 2) for x in v] for k, v in F['best_minus_last_points'].items()})
    print('[rule 0]', {r['run']: r['fail'] for r in RUN.values() if r['fail']} or 'every run passes', '| cw max deviation', max(r['cw_max_dev'] for r in RUN.values()))
    print('[signal]', {k: (round(v['u_epi_band'], 3), f"{v['u_epi_range80']:.2e}", int(v['identity_holds'])) for k, v in F['signal'].items()})
    print(f"[counts] S9 {F['s9']}; runs {F['runs_trained']}; short-ramp runs {F['n_short_ramp']}; logs {F['n_logs']}; expectation {F['expectation']}")
    print(f'[make_extra_s43_d35] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


if __name__ == '__main__':
    main()
