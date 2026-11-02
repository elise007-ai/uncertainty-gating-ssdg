#!/usr/bin/env python3
"""
make_extra_s41_d34.py (S42, 2026-10-26) -- reader of D-34, the percentile gate at the oracle's removal rate (`v18I_<ds>_pctmatch_s<seed>`, trainer v12,
keep 0.993 Prostate / 0.983 Fundus), assertion-first (R-48); nothing typed by hand (R-14). Rules: DOCS/D34_PREREGISTRATION_S41.md with Amendments 1 and 2.
Two collections of one server tree:
  RETURNS/S41   the first, after the runner's DONE line: five runs complete, Fundus seed 2024 killed at iteration 27,500;
  RETURNS/S41b  the second, after the owner continued that run from its state saved at iteration 27,000 (RUNBOOK S41 failure clause); its other
                60 files are byte-identical to the first.
Every run is read from RETURNS/S41b. The killed attempt is checked in RETURNS/S41 and shown to be the byte prefix of the continued log.
The interim reading of S42 (Prostate only, from RETURNS/S41 before RETURNS/S41b arrived) is archived in quarantine ARCHIVE/code_superseded_s42/ and
ARCHIVE/derived_superseded/extra_s41_interim_s42/.
  python make_extra_s41_d34.py --returns <RETURNS> --derived <derived_v04_S12> --release <release_s12> --out <derived_v04_S12/extra_s41>
Writes table_d34_{integrity,per_seed,rule,checkpoints,coverage,signal}_s41.csv, table_{band_generality,band_summary,signal_identity}_s41.csv, facts_s41.json.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, shutil, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_derived_v02 as v02                      # noqa: E402
import make_derived_v04 as v04                      # noqa: E402
import make_selection_s24 as s24                    # noqa: E402
from make_extra_s23 import (need, sha, wcsv, f4, find_log, split_of, rule, SPLIT_RX,   # noqa: E402
                            SEEDS, DATASETS, DOM, MAXIT, LBDOM, RULE_BAND, RULE_RANGE, PASSED)
from make_selection_s25 import cmd_pairs, ref_name  # noqa: E402
from make_extra_s29 import tost                     # noqa: E402
from make_extra_s30_d32 import verdict, WIN_RX      # noqa: E402

HASHES = ('915347736f4855da', 'a6fe4357a39d0020', '3e214990fca90428', 'ef79f64279586ad1', 'd3bf72f9dc5eb88b', '531fa9f521ea601a')
KEEP = {'prostate': 0.993, 'fundus': 0.983}; WINDOW = {'prostate': (0.990, 0.997), 'fundus': (0.980, 0.990)}
BOX_MEAN = 0.21                                     # the mean area of the pasted box, as in make_extra_s31_d33.py
KILLED = ('fundus', 2024); KNAME = 'v18I_fundus_pctmatch_s2024'; KILLED_LOG_SHA16 = '9f714105e8a48c62'; RESUME_ITER = 27000
CHANGED = {'console_logs/S41/diag_harness.stdout.log', 'console_logs/S41/runner_jobs_D34.log', 'console_logs/S41/status_jobs_D34.csv',
           f'console_logs/S41/{KNAME}.console.log', f'runs/fundus/{KNAME}/log.txt', f'runs/fundus/{KNAME}/run_config.json'}
APPEND_ONLY = ('console_logs/S41/diag_harness.stdout.log', 'console_logs/S41/runner_jobs_D34.log', 'console_logs/S41/status_jobs_D34.csv', f'runs/fundus/{KNAME}/log.txt')
ERR_RX = re.compile(r'Traceback|out of memory|CUDA error|RuntimeError|MemoryError|Segmentation fault|Killed')
HDR_RX = re.compile(r'^\[[\d:.]+\] python train_fourierfuse_v18_revision\.py .*$', re.M)
RES_RX = re.compile(r'RESUMED from iter (\d+) \(starting epoch (\d+)/(\d+)\) best_avg so far ([\d.]+) @ (\d+)')

def verify_parts(R, n_expected, label):
    listed = {}
    for mf in sorted(glob.glob(os.path.join(R, '_manifests', 'p*', 'MANIFEST.txt'))):
        for line in open(mf, encoding='utf-8'):
            if line.strip():
                p, size, h = line.rstrip('\n').split('\t'); q = os.path.join(R, *p.split('/'))
                assert p not in listed, f'{p} in two parts'
                assert os.path.exists(q) and os.path.getsize(q) == int(size) and sha(q) == h, f'manifest mismatch {p}'; listed[p] = h
    need(len(listed) == n_expected, f'every file of the {label} collection ({len(listed)}) is in the results tier and matches its part manifest (size and sha256)')
    need(not any(re.search(r'v18[ABDEFGH]_', p) for p in listed), f'the {label} collection carries no file of an earlier training')
    return listed


def main():
    ap = argparse.ArgumentParser()
    for k in ('returns', 'derived', 'release', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    S41, S41b = os.path.join(a.returns, 'S41'), os.path.join(a.returns, 'S41b'); src = os.path.join(a.release, 'CODE', 'source')
    CL, CLb = os.path.join(S41, 'console_logs', 'S41'), os.path.join(S41b, 'console_logs', 'S41')
    F = {'session': 'S42', 'reader': 'make_extra_s41_d34.py', 'preregistration': 'DOCS/D34_PREREGISTRATION_S41.md (+ Amendments 1 and 2)'}
    code_sha = {'train_fourierfuse_v18_revision.py': sha(os.path.join(src, 'v12', 'train_fourierfuse_v18_revision.py'))}
    for n in ('revision_modules.py', 'fourierfuse_v16_modules.py', 'diag_uncertainty_harness_v18.py'):
        code_sha[n] = sha(os.path.join(src, n))
    F['code_sha256'] = {k: v[:16] for k, v in code_sha.items()}
    need(F['code_sha256']['train_fourierfuse_v18_revision.py'] == HASHES[0], 'the release copy of trainer v12 is 915347736f4855da (unchanged for D-34)')
    F31 = json.load(open(os.path.join(a.derived, 'extra_s31', 'facts_s31.json')))
    rb = lambda R, p: open(os.path.join(R, *p.split('/')), 'rb').read()
    rt = lambda R, p: rb(R, p).decode('utf-8', 'replace')
    # ------------------------------------------------ the two collections of one server tree
    m1, m2 = verify_parts(S41, 66, 'first S41'), verify_parts(S41b, 66, 'second S41 (RETURNS/S41b)')
    F['manifest_files_verified'] = {'S41': len(m1), 'S41b': len(m2)}
    need(set(m1) == set(m2) and {p for p in m1 if m1[p] != m2[p]} == CHANGED,
         'the two collections list the same 66 paths; exactly six files differ: the continued run\'s log, run_config.json and console, the runner log, the status file and the harness log')
    need(all(rb(S41b, p).startswith(rb(S41, p)) and len(rb(S41b, p)) > len(rb(S41, p)) for p in APPEND_ONLY),
         'the run log, the runner log, the status file and the harness log of the second collection extend those of the first byte for byte')
    need(rb(S41, 'SKIPPED_HELD.txt') == rb(S41b, 'SKIPPED_HELD.txt'), 'both collections skipped the same held files')
    need(sha(os.path.join(S41b, 'jobs', 'jobs_D34.txt'))[:16] == HASHES[3] and sha(os.path.join(S41b, 'jobs', 'jobs_D34smoke.txt'))[:16] == HASHES[4],
         'the job lists that ran are the registered ones (tools S41)')
    # ------------------------------------------------ RUNBOOK S41 steps 0-2 on record (first collection; byte-identical in the second)
    pf = open(os.path.join(CL, 'preflight.txt'), errors='replace').read()
    need(all(h in pf for h in HASHES) and '1401 HELD_MANIFEST_S41.txt' in pf and re.search(r'\n48\n', pf) and 'No such file or directory' in pf and re.search(r'\n0\s*$', pf),
         'step 0 on record: the six hashes (trainer v12 unchanged), the 1,401-line held manifest, 48 v18H_ lines, no runner lock, no D-34 run before the launch')
    F['preflight_gpu_mib'] = {int(g): int(m) for g, m in re.findall(r'^(\d), (\d+) MiB$', pf, re.M)}
    need('self-test PASSED' in open(os.path.join(CL, 'selftest_revision_modules.txt'), errors='replace').read() and 'COMPILE OK v12' in open(os.path.join(CL, 'compile_trainer_v12.txt'), errors='replace').read(),
         'step 1 on record: the server self-test passed; trainer v12 compiled')
    sm = rt(S41b, 'runs/fundus/v18smoke_fundus_pctmatch/log.txt'); sw = WIN_RX.findall(sm)
    need('method=percentile_gate' in sm and 'pg_keep_frac=0.983' in sm and len(sw) == 2 and not any(m == '1.000' for _, m, _ in sw), 'the smoke run: percentile gate at keep 0.983, two logged windows, neither keeps every pixel')
    # ------------------------------------------------ the runner: first launch, then the relaunch (RUNBOOK S41 failure clause)
    rl1, rl2 = rt(S41, 'console_logs/S41/runner_jobs_D34.log'), rt(S41b, 'console_logs/S41/runner_jobs_D34.log'); add = rl2[len(rl1):]
    hist = {}
    for r in csv.DictReader(open(os.path.join(CLb, 'status_jobs_D34.csv'))):
        hist.setdefault(r['job'], []).append(r)
    need(len(hist) == 6 and all(len(rs) == 1 and rs[0]['postflight'] == 'PASS' and rs[0]['exit_code'] == '0' for j, rs in hist.items() if j != KNAME)
         and [(r['exit_code'], r['postflight']) for r in hist[KNAME]] == [('-9', 'FAIL'), ('0', 'PASS')] and 'no TRAINING COMPLETE' in hist[KNAME][0]['detail'],
         'runner: six jobs; five passed on the first launch; the Fundus seed-2024 job was killed (exit code -9, no TRAINING COMPLETE), then passed when continued')
    need(rl1.count('[preflight] FAILED') == 2 and rl1.count('[preflight] GREEN') == 1 and "gpus=['1'] per_gpu=2" in rl1, 'first launch: two attempts refused (GPU 0 in use), the third ran on GPU 1 only')
    need(add.count('[preflight] FAILED') == 1 and add.count('[preflight] GREEN') == 1 and add.count('[skip] ') == 10 and '[restart]' not in add
         and add.count(f'[resume] {KNAME} has resume_state.pth (iter {RESUME_ITER}) -> will CONTINUE, not restart') == 2 and add.count('[runner] 1 to run, 5 skipped') == 2
         and add.count(f'START {KNAME}') == 1 and 'DONE: 1/1 PASS' in add,
         'relaunch: the five finished runs skipped; the killed run continued from its state at iteration 27,000 (one attempt refused, GPUs in use; then GREEN); passed')
    hs = rt(S41b, 'console_logs/S41/diag_harness.stdout.log')[len(rt(S41, 'console_logs/S41/diag_harness.stdout.log')):]
    need(hs.strip() == '[harness] ran 0 (failed 0), skipped 6 existing', 'the harness was not re-run after the relaunch: the six JSONs are those of the first collection')
    F['runner'] = {'jobs': 6, 'first_launch_pass': 5, 'killed': KNAME, 'killed_exit_code': -9, 'killed_start': hist[KNAME][0]['start'], 'killed_end': hist[KNAME][0]['end'],
                   'continued_start': hist[KNAME][1]['start'], 'continued_end': hist[KNAME][1]['end'], 'continued_gpu': hist[KNAME][1]['gpu'],
                   'first_start': min(rs[0]['start'] for rs in hist.values()), 'preflight_refusals': 3}
    # ------------------------------------------------ the killed attempt and its continuation (Amendment 2 item 2)
    kp, fp_ = f'runs/fundus/{KNAME}/log.txt', os.path.join(S41b, 'runs', 'fundus', KNAME, 'log.txt')
    kl, fl = rt(S41, kp), rt(S41b, kp); seg = fl[len(kl):]
    kt = v04.screen_run(os.path.join(S41, *kp.split('/'))); kw = [int(i) for i, _, _ in WIN_RX.findall(kl)]
    saved = [int(x) for x in re.findall(r'\[resume\] state saved at iter (\d+)', kl)]; ksum = v04.SUM_RX.findall(kl)
    kc, fc = rt(S41, f'console_logs/S41/{KNAME}.console.log'), rt(S41b, f'console_logs/S41/{KNAME}.console.log')
    need(sha(os.path.join(S41, *kp.split('/')))[:16] == KILLED_LOG_SHA16 and not kt['complete'] and kt['n_evals'] == 55 and max(kw) == 27500 and v04.flag_of(kt) == 'ok'
         and kt['nf_n'] == 0 and not kt['resumed_iters'] and not ERR_RX.search(kc),
         'the killed attempt (first collection): 55 of 60 evaluations, last logged iteration 27,500, R-34 ok, no non-finite iteration, no resume, no error line in its console')
    need(saved and saved[-1] == RESUME_ITER, 'the killed attempt\'s last saved state is that of iteration 27,000')
    ft, fq = v04.screen_run(fp_), v02.parse_log(fp_); heads = HDR_RX.findall(fl); res = RES_RX.findall(seg); evs = [int(e[0]) for e in v04.SUM_RX.findall(seg)]
    c1 = json.loads(rt(S41, f'runs/fundus/{KNAME}/run_config.json')); c2 = json.loads(rt(S41b, f'runs/fundus/{KNAME}/run_config.json'))
    need(fl.startswith(kl) and ft['complete'] and v04.flag_of(ft) == 'ok' and ft['nf_n'] == 0,
         'the continued run: its log begins with the killed attempt\'s log, ends with TRAINING COMPLETE; R-34 ok; no non-finite iteration')
    need(len(heads) == 2 and ft['n_headers'] == 2 and ft['resumed_iters'] == [RESUME_ITER] and len(res) == 1 and [int(x) for x in res[0][:3]] == [RESUME_ITER, 55, 60],
         'two launch headers; exactly one resume, from iteration 27,000 (epoch 55 of 60)')
    need(len(evs) == 6 and evs == list(range(evs[0], evs[0] + 6)) and evs[0] == int(ksum[-1][0]) and ft['n_evals'] == 61,
         'the continued segment makes six evaluations, the first at the epoch of the killed attempt\'s last one (iteration 27,500, re-trained); '
         'the log holds 61 evaluations, among them the killed attempt\'s evaluation of the discarded iterations 27,000-27,500')
    need({k: v for k, v in c2.items() if k != 'resume_events'} == c1 and [e['iter'] for e in c2['resume_events']] == [RESUME_ITER],
         'run_config.json: the launch provenance is unchanged; one resume event, at iteration 27,000')
    need(fq['best_iter'] == int(res[0][4]) == int(ksum[-1][4]) < RESUME_ITER and abs(float(res[0][3]) - fq['best_avg']) < 6e-5,
         'its best checkpoint (the reported value, E-69) is the same before and after the interruption and lies before iteration 27,000: no later evaluation, the discarded one included, was better')
    need(not ERR_RX.search(fc) and 'stale RUNNING.lock' in fc and f'RESUMED from iter {RESUME_ITER}' in fc, 'the relaunch console: no error line; the killed attempt\'s lock was stale; resumed from iteration 27,000')
    fwin = [(int(i), float(m)) for i, m, _ in WIN_RX.findall(kl) if int(i) <= RESUME_ITER] + [(int(i), float(m)) for i, m, _ in WIN_RX.findall(seg)]
    need([i for i, _ in fwin] == list(range(500, MAXIT['fundus'] + 1, 500)),
         'the trained trajectory of the continued run: 60 gate windows, iterations 500 to 30,000 (the killed attempt\'s window at 27,500 belongs to discarded iterations and is not used)')
    F['continued_run'] = {'run': KNAME, 'killed_at_logged_iter': max(kw), 'resumed_from': RESUME_ITER, 'best_iter': fq['best_iter'], 'n_evals_in_log': ft['n_evals'],
                          'killed_log_sha16': KILLED_LOG_SHA16, 'final_log_sha16': sha(fp_)[:16]}
    # ------------------------------------------------ per run: integrity and rule 0 (every run read from RETURNS/S41b)
    jobs = {l.split('\t')[0]: l.split('\t')[2] for l in open(os.path.join(S41b, 'jobs', 'jobs_D34.txt'), encoding='utf-8').read().splitlines() if l.strip()}
    RUN, rows_i, rows_v, rows_p = {}, [], [], []
    for ds in DATASETS:
        Tm = MAXIT[ds]; dm = lambda pd: {DOM[ds][k - 1]: float(np.mean(list(v.values()))) for k, v in sorted(pd.items())}
        for sd in SEEDS:
            run = f'v18I_{ds}_pctmatch_s{sd}'; cont = run == KNAME; d = os.path.join(S41b, 'runs', ds, run); log = os.path.join(d, 'log.txt')
            txt = open(log, errors='replace').read(); t = v04.screen_run(log); q = v02.parse_log(log); cfg = json.load(open(os.path.join(d, 'run_config.json'))); ar = cfg['args']
            ok_cfg = (ar['method'] == 'percentile_gate' and abs(ar['pg_keep_frac'] - KEEP[ds]) < 1e-12 and ar['seed'] == sd and ar['max_iterations'] == Tm and ar['lb_domain'] == LBDOM[ds]
                      and ar['lb_num'] == 20 and ar['head'] == 'evidential' and ar['consistency_ramp_frac'] == -1.0 and ar['amp'] == 1 and ar['resume'] == 'auto')
            ok_sha = all(cfg[k] == code_sha[n] == sha(os.path.join(d, 'code', n)) for k, n in (('script_sha256', 'train_fourierfuse_v18_revision.py'),
                         ('revision_modules_sha256', 'revision_modules.py'), ('fourierfuse_v16_modules_sha256', 'fourierfuse_v16_modules.py')))
            hd = HDR_RX.findall(txt); jp = cmd_pairs('python train_fourierfuse_v18_revision.py ' + jobs[run])
            cmd_diff = sorted({k for h_ in hd for cp in [cmd_pairs(h_[h_.index('python'):])] for k in set(cp) | set(jp) if cp.get(k) != jp.get(k)})
            c31 = json.load(open(os.path.join(a.returns, 'S29', 'runs', ds, f'v18F_{ds}_percentile_s{sd}', 'run_config.json')))['args']
            arg_diff = sorted(k for k in set(ar) | set(c31) if ar.get(k) != c31.get(k))
            refname = ref_name(ds, sd); rlog, rq = find_log(a.returns, ds, refname); tr = v04.screen_run(rlog)
            sps = [(int(m.group(1)), [int(x) for x in m.group(4).split(',')]) for m in SPLIT_RX.finditer(txt)]
            spr = split_of(rlog); sj = json.load(open(os.path.join(d, 'split.json')))
            me = json.load(open(os.path.join(S41b, 'diag_out', f'{ds}__{run}__ema.json')))['meta']
            ok_h = (me['which'] == 'ema' and me['iter'] == q['best_iter'] and abs(me['best_avg_dice'] - q['best_avg']) < 1e-6 and run in me['ckpt']
                    and me['harness_sha256'] == code_sha['diag_uncertainty_harness_v18.py'] and me['n_models'] == 1)
            win = fwin if cont else [(int(i), float(m)) for i, m, _ in WIN_RX.findall(txt)]
            e = [m for i, m in win if i <= 0.2 * Tm]; l = [m for i, m in win if i > 0.2 * Tm]; lo, hi = WINDOW[ds]; med = float(np.median(l))
            if cont:
                need(t['complete'] and t['parse_ok'] and len(hd) == 2, f'{run}: complete; continued once from its own state saved at iteration 27,000 (Amendment 2; checked above)')
            else:
                need(t['complete'] and t['parse_ok'] and t['n_evals'] == Tm // 500 and t['n_headers'] == 1 and len(hd) == 1 and not t['resumed_iters'] and not cfg.get('resume_events'),
                     f'{run}: one training from iteration 0 to the last iteration, no resume event')
            need(ok_cfg and 'method=percentile_gate' in txt, f'{run}: configuration = the percentile gate at the registered keep fraction {KEEP[ds]}, headline settings')
            need(set(cmd_diff) <= {'--gpu'}, f'{run}: every launch command in its log equals its line of jobs_D34.txt (the runner adds only the GPU index)')
            need(set(arg_diff) <= {'pg_keep_frac', 'save_name', 'gpu'} and 'pg_keep_frac' in arg_diff,
                 f'{run}: recorded arguments = those of the D-31 run of the same seed except the keep fraction (and the name, GPU)')
            need(ok_sha, f'{run}: rule 0: script_sha256 = trainer v12; code snapshot, recorded hashes and the release code are the same bytes')
            need(bool(sps) and all(s_ == sps[0] for s_ in sps) and sps[0][0] == sd and sps[0][1] == spr[1] == sj['lb_idxs'],
                 f'{run}: rule 0: labeled slices identical to those of its threshold reference ({refname}){" at both launches" if cont else ""}')
            need(ok_h, f'{run}: the measured teacher is the EMA weights of the best checkpoint of this log, measured with the release harness')
            need(len(win) == Tm // 500 and lo <= med <= hi, f'{run}: rule 0 item 3: the median logged w_mean after the first fifth ({med:.4f}) lies in the registered window [{lo}, {hi}]')
            o31 = F31[f'{ds}_per_seed'][str(sd)]
            need(abs(o31['ref'] - rq['best_avg']) < 1e-12, f'{run}: its threshold reference is the one of ADJUDICATION_S31 (same seed)')
            r = RUN[(ds, sd)] = {'run': run, 'continued_from': RESUME_ITER if cont else None, 'flag': v04.flag_of(t), 'nf': t['nf_n'], 'nf_frac': t['nf_n'] / Tm,
                                 'w_first': win[0][1], 'w_first_fifth_mean': float(np.mean(e)), 'w_after_mean': float(np.mean(l)), 'w_after_median': med,
                                 'w_after_min': float(min(l)), 'w_after_max': float(max(l)), 'refname': refname,
                                 'best': q['best_avg'], 'best_iter': q['best_iter'], 'last': t['student_avg_last'], 'ref': rq['best_avg'], 'ref_last': tr['student_avg_last'],
                                 'oracle': o31['best'], 'nogate': o31['nogate'], 'at_best': dm(q['per_domain']), 'ref_at_best': dm(rq['per_domain'])}
            rows_p.append([ds, sd, run, f'{r["best"]:.6f}', r['best_iter'], f4(r['last']), refname, f'{r["ref"]:.6f}', f4(r['ref_last']), f'{r["best"] - r["ref"]:+.6f}',
                           f'{r["oracle"]:.6f}', f'{r["oracle"] - r["best"]:+.6f}', f'{r["nogate"]:.6f}', f'{r["best"] - r["nogate"]:+.6f}', r['flag'], r['nf'], RESUME_ITER if cont else ''])
            rows_i.append([ds, run, sd, int(t['complete']), t['n_evals'], len(hd), ';'.join(map(str, t['resumed_iters'])), t['nf_n'], r['flag'], int(ok_cfg), ';'.join(cmd_diff),
                           ';'.join(arg_diff), int(ok_sha), int(ok_h), len(win), cfg['torch'], cfg['hostname']])
            rows_v.append([ds, sd, run, f'{r["w_first"]:.3f}', f'{r["w_first_fifth_mean"]:.4f}', f'{r["w_after_mean"]:.4f}', f'{med:.4f}', f'{r["w_after_min"]:.3f}', f'{r["w_after_max"]:.3f}',
                           f'{(1 - r["w_after_mean"]) / (1 - BOX_MEAN):.4f}'])
    wcsv(os.path.join(a.out, 'table_d34_integrity_s41.csv'), rows_i, ['dataset', 'run', 'seed', 'complete', 'n_evals_in_log', 'n_launches', 'resumed_at_iters', 'n_nonfinite_skips', 'r34_flag',
         'config_ok', 'cmd_keys_differing_from_job_line', 'arg_keys_differing_from_d31', 'hashes_ok', 'harness_meta_ok', 'n_gate_windows_trained', 'torch', 'hostname'])
    wcsv(os.path.join(a.out, 'table_d34_coverage_s41.csv'), rows_v, ['dataset', 'seed', 'run', 'w_mean_first_window', 'w_mean_first_fifth', 'w_mean_after_fifth', 'w_mean_after_fifth_median',
         'w_mean_after_fifth_min', 'w_mean_after_fifth_max', 'removed_share_outside_box_approx'])
    wcsv(os.path.join(a.out, 'table_d34_per_seed_s41.csv'), rows_p, ['dataset', 'seed', 'run', 'best_avg', 'best_iter', 'student_avg_last', 'threshold_reference', 'ref_best_avg', 'ref_last',
         'matched_minus_threshold', 'oracle_d33_best_avg', 'oracle_minus_matched', 'nogate_d32_best_avg', 'matched_minus_nogate', 'r34_flag', 'n_nonfinite', 'continued_from_iter'])
    # ------------------------------------------------ rules 1-2 and the descriptive comparison with no gate
    rows_r = []; F['expectation'] = {'registered': 'matched gate within seed noise of the threshold on both benchmarks, and the oracle above the matched gate on at least one'}
    for ds in DATASETS:
        m = [RUN[(ds, s)]['best'] for s in SEEDS]; ref = [RUN[(ds, s)]['ref'] for s in SEEDS]; orc = [RUN[(ds, s)]['oracle'] for s in SEEDS]; ng = [RUN[(ds, s)]['nogate'] for s in SEEDS]
        r1 = rule(m, ref); e1 = tost([100 * (x - y) for x, y in zip(m, ref)]); r1['verdict'] = verdict(r1)
        r2 = rule(orc, m); e2 = tost([100 * (x - y) for x, y in zip(orc, m)]); r2['verdict'] = verdict(r2).replace('the threshold gate', 'the matched gate')
        r3 = rule(m, ng); e3 = tost([100 * (x - y) for x, y in zip(m, ng)])
        need(all(abs(e['mean_points'] - 100 * r['delta']) < 1e-9 for r, e in ((r1, e1), (r2, e2), (r3, e3))), f'{ds}: TOST and the paired tests use the same differences')
        need(abs(np.mean(ref) - F31[f'{ds}_rule1']['ref_mean']) < 1e-12 and abs(np.mean(orc) - F31[f'{ds}_rule1']['mean']) < 1e-12,
             f'{ds}: the threshold references and the oracle values are those of ADJUDICATION_S31')
        for r, lab in ((r1, 'rule 1'), (r2, 'rule 2')):
            need((r['verdict'] != 'within seed noise') == (r['p'] < 0.05 and abs(r['delta']) >= 0.005), f'{ds}: the {lab} verdict follows from p and delta as registered (p < 0.05 and |delta| >= 0.5)')
        F[f'{ds}_rule1_matched_minus_threshold'] = {**r1, **e1, 'r34_flags': [RUN[(ds, s)]['flag'] for s in SEEDS], 'unstable': [RUN[(ds, s)]['nf_frac'] > 0.01 for s in SEEDS]}
        F[f'{ds}_rule2_oracle_minus_matched'] = {**r2, **e2}
        F[f'{ds}_descriptive_matched_minus_nogate'] = {**r3, **e3, 'note': 'descriptive, not a test (D34 pre-registration, rule 3)'}
        F[f'{ds}_per_seed'] = {str(s): RUN[(ds, s)] for s in SEEDS}
        F[f'{ds}_descriptive'] = {'best_minus_last_points': [100 * (RUN[(ds, s)]['best'] - RUN[(ds, s)]['last']) for s in SEEDS],
                                  'ref_best_minus_last_points': [100 * (RUN[(ds, s)]['ref'] - RUN[(ds, s)]['ref_last']) for s in SEEDS],
                                  'best_frac': [RUN[(ds, s)]['best_iter'] / MAXIT[ds] for s in SEEDS],
                                  'per_vendor_delta_points': {k: [100 * (RUN[(ds, s)]['at_best'][k] - RUN[(ds, s)]['ref_at_best'][k]) for s in SEEDS] for k in RUN[(ds, SEEDS[0])]['at_best']}}
        F['expectation'][ds] = {'matched_within_noise': r1['verdict'] == 'within seed noise', 'oracle_above_matched': r2['verdict'] == 'above the matched gate'}
        for lab, r_, e_, vd in (('matched_minus_threshold (rule 1)', r1, e1, r1['verdict']), ('oracle_minus_matched (rule 2)', r2, e2, r2['verdict']),
                                ('matched_minus_nogate (descriptive)', r3, e3, 'descriptive')):
            rows_r.append([ds, lab, f4(r_['mean']), f4(r_['std']), f4(r_['ref_mean']), f4(r_['ref_std']), f'{100 * r_["delta"]:+.2f}', ';'.join(f'{100 * x:+.2f}' for x in r_['per_seed_delta']),
                           f'{r_["p"]:.4f}', r_['n_pos'], f'{r_["sign_p_two_sided"]:.4f}', f'{e_["ci90_points"][0]:+.2f};{e_["ci90_points"][1]:+.2f}', f'{e_["p_tost_0.5"]:.4f}',
                           f'{e_["mde_points"]:.2f}', vd])
    wcsv(os.path.join(a.out, 'table_d34_rule_s41.csv'), rows_r, ['dataset', 'comparison', 'mean_dice', 'std_dice', 'ref_mean', 'ref_std', 'delta_points', 'per_seed_delta_points_1337_42_2024',
         'p_paired', 'n_seeds_positive', 'sign_test_p_two_sided', 'ci90_points', 'p_tost_0.5', 'mde_points', 'verdict'])
    ex = F['expectation']; ex['holds'] = all(ex[d]['matched_within_noise'] for d in DATASETS) and any(ex[d]['oracle_above_matched'] for d in DATASETS)
    # ------------------------------------------------ the three checkpoints (descriptive), and the counts of the text with D-34 added (R-66)
    L = s24.load(a.returns); grp = lambda run, r: 'failure' if (r['flag'] != 'ok' or r['nf'] > 0) else 'short_ramp' if 'ramp10' in run else 'counted'
    old = {k: r for k, r in L.items() if not k[1].startswith('v18I_')}; new = {k: r for k, r in L.items() if k[1].startswith('v18I_')}
    need(len(old) == F31['n_logs'] and len(new) == 6,
         f'complete logs: the {F31["n_logs"]} of facts_s31.json plus the six D-34 runs (the killed attempt\'s log is incomplete and not counted; its continuation is the sixth)')
    lk = L[(KILLED[0], KNAME)]
    need(lk['ret'] == 'S41b' and lk['resumed'] and abs(lk['best'] - RUN[KILLED]['best']) < 1e-12 and abs(lk['last'] - RUN[KILLED]['last']) < 1e-12,
         'the checkpoint loader reads the continued run from RETURNS/S41b, with its resume, and the same best and last values as the integrity pass')
    rows_c = []; F['checkpoints'] = {}
    for ds in DATASETS:
        for ck in ('best', 'last', 'swad'):
            M_ = [L[(ds, f'v18I_{ds}_pctmatch_s{s}')][ck] for s in SEEDS]
            need(all(x is not None for x in M_), f'{ds}: a {ck} value on every D-34 run')
            for lab, A_, B_ in (('matched_minus_threshold', M_, [L[(ds, ref_name(ds, s))][ck] for s in SEEDS]),
                                ('oracle_minus_matched', [L[(ds, f'v18H_{ds}_oracle_s{s}')][ck] for s in SEEDS], M_),
                                ('matched_minus_nogate', M_, [L[(ds, f'v18G_{ds}_ungated_s{s}')][ck] for s in SEEDS])):
                x = s24.paired(A_, B_); F['checkpoints'][f'{ds}_{lab}_{ck}'] = {'delta_points': 100 * x['delta'], 'p': x['p'], 'per_seed_points': [100 * v for v in x['per_seed']]}
                rows_c.append([ds, lab, ck, f'{100 * x["delta"]:+.2f}', f'{x["p"]:.3f}', ';'.join(f'{100 * v:+.2f}' for v in x['per_seed'])])
        need(abs(F['checkpoints'][f'{ds}_matched_minus_threshold_best']['delta_points'] - 100 * F[f'{ds}_rule1_matched_minus_threshold']['delta']) < 1e-9
             and abs(F['checkpoints'][f'{ds}_oracle_minus_matched_best']['delta_points'] - 100 * F[f'{ds}_rule2_oracle_minus_matched']['delta']) < 1e-9, f'{ds}: the best-checkpoint lines equal rules 1 and 2')
    F['checkpoints_note'] = ('descriptive (rule 3); for Fundus seed 2024 the last-iteration and SWAD values come from the segment continued at iteration 27,000 '
                             '(Amendment 2 item 2); its best checkpoint lies before the interruption')
    wcsv(os.path.join(a.out, 'table_d34_checkpoints_s41.csv'), rows_c, ['dataset', 'comparison', 'checkpoint', 'delta_points', 'p_paired', 'per_seed_points'])
    sel = {}
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        b0 = [100 * (r['best'] - r['last']) for (d, run), r in old.items() if d == ds and grp(run, r) == 'counted']; S = F31['selection'][ds]
        need(len(b0) == S['n'] and abs(float(np.median(b0)) - S['median']) < 1e-9 and sum(x > 5 for x in b0) == S['n_gt5'] and sum(x > 1 for x in b0) == S['n_gt1'],
             f'{ds}: the logs reproduce the S32 selection counts exactly (n {S["n"]}, median {S["median"]:.4f})')
        b1 = [100 * (r['best'] - r['last']) for (d, run), r in new.items() if d == ds and grp(run, r) == 'counted']; v = b0 + b1
        sel[ds] = {'n': len(v), 'median': float(np.median(v)), 'n_gt1': sum(x > 1 for x in v), 'n_gt5': sum(x > 5 for x in v), 'n_prev': len(b0), 'median_prev': S['median'], 'n_gt5_prev': S['n_gt5']}
    need(sel['prostate']['n'] == sel['prostate']['n_prev'] + 3 and sel['fundus']['n'] == sel['fundus']['n_prev'] + 3, 'the six D-34 runs are counted, three per benchmark')
    F['selection'] = sel; F['n_logs'] = len(L)
    # ------------------------------------------------ descriptive: the six teachers (v02 functions unchanged)
    H = {k: x for k, x in v02.load_harness(S41b).items() if k[1].startswith('v18I_')}
    need(len(H) == 6 and all(k[2] == 'ema' for k in H), 'six harness JSONs, EMA teachers')
    need(all(rb(S41b, f'diag_out/{k[0]}__{k[1]}__ema.json') == rb(S41, f'diag_out/{k[0]}__{k[1]}__ema.json') for k in H),
         'the six harness JSONs are those of the first collection, byte for byte: measured before the relaunch; for Fundus seed 2024 on the checkpoint that stayed its best (Amendment 2 item 3)')
    tmp = tempfile.mkdtemp(); v02.band_tables(H, tmp); v02.signal_identity_table(H, tmp, {})
    for n in ('table_band_generality', 'table_band_summary', 'table_signal_identity'):
        shutil.copy(os.path.join(tmp, n + '.csv'), os.path.join(a.out, n + '_s41.csv'))
    bs = {(r['dataset'], r['run'], r['signal']): r for r in csv.DictReader(open(os.path.join(a.out, 'table_band_summary_s41.csv')))}
    idn = list(csv.DictReader(open(os.path.join(a.out, 'table_signal_identity_s41.csv'))))
    F['signal'] = {}; rows_g = []
    for ds in DATASETS:
        for sd in SEEDS:
            run = RUN[(ds, sd)]['run']; g = lambda sig, col: float(bs[(ds, run, sig)][col])
            s = {'u_epi_all': g('u_epi', 'auroc_all_mean'), 'u_epi_band': g('u_epi', 'auroc_edt5_mean'), 'u_epi_range80': g('u_epi', 'range80_all_mean'),
                 'conf_all': g('conf', 'auroc_all_mean'), 'conf_band': g('conf', 'auroc_edt5_mean'),
                 'identity_holds': [r for r in idn if r['dataset'] == ds and r['run'] == run][0]['identity_holds'] == '1'}
            s['phaseA_rule_collapse_persists'] = bool(s['u_epi_band'] <= RULE_BAND and s['u_epi_range80'] <= RULE_RANGE); F['signal'][f'{ds}_s{sd}'] = s
            rows_g.append([ds, sd, run] + [f'{x:.4g}' if isinstance(x, float) else int(x) for x in s.values()])
    wcsv(os.path.join(a.out, 'table_d34_signal_s41.csv'), rows_g, ['dataset', 'seed', 'run'] + list(next(iter(F['signal'].values())).keys()))
    I = F31['s9']
    F['s9'] = {'n_all': I['n_all'] + 6, 'n_testbed': I['n_testbed'] + 6, 'n_testbed_identity_holds': I['n_testbed_identity_holds'] + sum(v['identity_holds'] for v in F['signal'].values()),
               'per_benchmark': {k: v + (3 if k in DATASETS else 0) for k, v in I['per_benchmark'].items()}}
    F['runs_trained'] = {'before_d34': F31['runs_trained']['total'], 'd34': 6, 'total': F31['runs_trained']['total'] + 6,
                         'note': 'Fundus seed 2024 is one run: killed at iteration 27,500 and continued from its state at 27,000'}
    # ------------------------------------------------ the realized rate against the oracle's (descriptive; Amendment 1 item 6)
    O = lambda ds: [(1 - F31[f'{ds}_per_seed'][str(s)]['w_after_mean']) / (1 - BOX_MEAN) for s in SEEDS]
    M = lambda ds: [(1 - RUN[(ds, s)]['w_after_mean']) / (1 - BOX_MEAN) for s in SEEDS]
    F['coverage'] = {ds: {'w_first': [min(RUN[(ds, s)]['w_first'] for s in SEEDS), max(RUN[(ds, s)]['w_first'] for s in SEEDS)],
                          'w_after_median': [min(RUN[(ds, s)]['w_after_median'] for s in SEEDS), max(RUN[(ds, s)]['w_after_median'] for s in SEEDS)],
                          'removed_share_matched': [min(M(ds)), max(M(ds))], 'removed_share_oracle': [min(O(ds)), max(O(ds))],
                          'ratio_matched_to_oracle_mean': float(np.mean(M(ds)) / np.mean(O(ds)))} for ds in DATASETS}
    need(all(0.5 < F['coverage'][ds]['ratio_matched_to_oracle_mean'] < 1.0 for ds in DATASETS), 'the matched gate removes less than the oracle and more than half its share, on both benchmarks')
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s41.json'), 'w'), indent=1, default=float)
    for ds in DATASETS:
        r, e = F[f'{ds}_rule1_matched_minus_threshold'], F[f'{ds}_rule2_oracle_minus_matched']; d3 = F[f'{ds}_descriptive_matched_minus_nogate']
        print(f"[d34] {ds}: matched {100 * r['mean']:.2f}+-{100 * r['std']:.2f} vs threshold {100 * r['ref_mean']:.2f}+-{100 * r['ref_std']:.2f}; delta {100 * r['delta']:+.2f} "
              f"({', '.join(f'{100 * x:+.2f}' for x in r['per_seed_delta'])}); p {r['p']:.4f}; CI90 [{r['ci90_points'][0]:+.2f}, {r['ci90_points'][1]:+.2f}]; TOST {r['p_tost_0.5']:.3f}; "
              f"MDE {r['mde_points']:.2f} -> {r['verdict']}")
        print(f"[d34] {ds}: oracle - matched {100 * e['delta']:+.2f} ({', '.join(f'{100 * x:+.2f}' for x in e['per_seed_delta'])}); p {e['p']:.4f}; t {e['t']:.3f}; "
              f"CI90 [{e['ci90_points'][0]:+.2f}, {e['ci90_points'][1]:+.2f}] -> {e['verdict']}")
        print(f"[d34] {ds}: matched - no gate {100 * d3['delta']:+.2f} ({', '.join(f'{100 * x:+.2f}' for x in d3['per_seed_delta'])}); p {d3['p']:.4f} (descriptive)")
    for k, v in F['checkpoints'].items():
        print(f"[ckpt] {k}: {v['delta_points']:+.2f} p {v['p']:.3f}")
    print('[coverage]', json.dumps(F['coverage']))
    print('[expectation]', json.dumps(F['expectation']))
    for ds, s in sel.items():
        print(f"[counts] {ds}: n {s['n_prev']} -> {s['n']}; median {s['median_prev']:.2f} -> {s['median']:.2f}; >5 {s['n_gt5_prev']} -> {s['n_gt5']}")
    print(f"[counts] S9: {F['s9']}; runs {F['runs_trained']}")
    print(f"[make_extra_s41_d34] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}")


if __name__ == '__main__':
    main()
