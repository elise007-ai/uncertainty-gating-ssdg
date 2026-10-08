#!/usr/bin/env python3
"""make_extra_s18.py -- S18 reader (N-27b: adjudication of the rep3 sendback; D-16: replicate spread and the reproduction gate). No GPU.
  python make_extra_s18.py --derived <derived_v04_S12> --returns <BSPC_results/RETURNS> --release <release root> [--out <dir>]
Writes <derived>/extra_s18/: table_replicates_s18.csv (three trainings of the default Prostate DAEP job line), table_reproduction_s18.csv
(five revision-code re-runs against the original runs), facts_s18.json.
Assertions: sendback integrity; rep3 is one complete, unresumed run of the rep2 job line (only save_name and gpu differ) trained with the
released code; its harness JSON measures its best checkpoint, after training ended, with the harness every other JSON used; the values
the S18 text prints."""
import argparse, glob, hashlib, itertools, json, os, re
import numpy as np, pandas as pd

VEND = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
REPS = ['v18A_prostate_daep_default_s1337', 'v18A_prostate_daep_default_s1337_rep2', 'v18A_prostate_daep_default_s1337_rep3']
REPRO = [('Prostate, DAEP', 'prostate', 'v18A_prostate_daep_default_s1337', ('prostate', 'daep')), ('Prostate, threshold', 'prostate', 'v18B_prostate_fixmatch_split1337_init1337', ('prostate', 'fixmatch')),
         ('Fundus, threshold', 'fundus', 'v18B_fundus_fixmatch_split1337_init1337', ('fundus', 'fixmatch')), ('M&Ms, threshold', 'MNMS', 'v18B_MNMS_fixmatch_s1337', ('mnms', 'fixmatch')),
         ('BUSI, threshold', 'BUSI', 'v18B_BUSI_fixmatch_s1337', ('busi', 'fixmatch'))]


def need(c, m):
    if not c: raise AssertionError('[make_extra_s18] ' + m)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def one(pattern):
    g = glob.glob(pattern); need(len(g) == 1, f'{len(g)} matches for {pattern}'); return g[0]


def run_log(returns, ds, run):
    d = os.path.dirname(one(os.path.join(returns, '*', 'runs', ds, run, 'log.txt'))); t = open(os.path.join(d, 'log.txt'), errors='replace').read()
    m = re.findall(r'Best avg dice \(student\): ([\d.]+) at iter (\d+)', t); need(len(m) == 1 and 'TRAINING COMPLETE' in t, f'{run}: not one complete run')
    dom = [float(x) for x in re.findall(r'Domain \d+ — base: Dice=([\d.]+)', t.split('Per-domain metrics at best avg dice')[-1])]
    return d, t, float(m[0][0]), int(m[0][1]), dom


def build(derived, returns, release, out):
    os.makedirs(out, exist_ok=True); F = {}; S = os.path.join(returns, 'S13')
    # ---------------------------------------------------------------- 1. the sendback
    man = [l.rstrip('\n').split('\t') for l in open(os.path.join(S, 'MANIFEST.txt')) if l.strip()]; held = {l.split('\t')[0]: l.split('\t')[2] for l in open(os.path.join(S, 'HELD_IN_QUARANTINE.txt'))}
    for arc, size, h in man:
        if arc in held: need(held[arc] == h, 'quarantined file hash'); continue
        need(sha(os.path.join(S, arc)) == h, f'hash mismatch: {arc}')
    rl = open(os.path.join(S, 'console_logs', 'S13', 'runner_jobs_D16.log')).read(); need('[preflight] GREEN' in rl and rl.strip().endswith('status: console_logs/S13/status_jobs_D16.csv') and 'DONE: 1/1 PASS' in rl, 'runner log')
    d3, t3, b3, i3, dom3 = run_log(returns, 'prostate', REPS[2]); need('RESUMED' not in t3 and len(dom3) == 6, 'rep3 resumed or per-domain lines missing')
    c3 = json.load(open(os.path.join(d3, 'run_config.json'))); d2 = os.path.dirname(one(os.path.join(returns, '*', 'runs', 'prostate', REPS[1], 'log.txt'))); c2 = json.load(open(os.path.join(d2, 'run_config.json')))
    diff = sorted(k for k in set(c2['args']) | set(c3['args']) if c2['args'].get(k) != c3['args'].get(k)); need(diff == ['gpu', 'save_name'], f'rep3 vs rep2 arguments differ in {diff}')
    src = os.path.join(release, 'CODE', 'source')
    for f in ('train_fourierfuse_v18_revision.py', 'revision_modules.py', 'fourierfuse_v16_modules.py'): need(sha(os.path.join(d3, 'code', f)) == sha(os.path.join(src, f)), f'rep3 snapshot of {f} is not the released file')
    need(sha(os.path.join(d3, 'code', 'revision_modules.py')) == sha(os.path.join(d2, 'code', 'revision_modules.py')), 'revision_modules differs between rep2 and rep3')
    a = open(os.path.join(d2, 'code', 'train_fourierfuse_v18_revision.py')).read().split('\n'); b = open(os.path.join(d3, 'code', 'train_fourierfuse_v18_revision.py')).read().split('\n')
    chg = [l for l in set(a) ^ set(b) if l.strip()]; need(all(('devis' in l.lower() or 'normalise_method_args' in l or 'E-38' in l or 'provenance' in l or 'run_config' in l or 'S07-S10' in l or 'evid_' in l or 'forcing' in l or 'return args' in l or 'logging.warning' in l or 'annealing' in l) for l in chg), f'rep2 -> rep3 script change outside the DEviS provenance refactor: {[l for l in chg][:3]}')
    J = {}; hs = json.load(open(one(os.path.join(S, 'diag_out', f'prostate__{REPS[2]}__ema.json'))))['meta']['harness_sha256']
    for r in REPS:      # a run measured in two sessions has two JSONs (the default run: S06 with an older harness, S07 with the final one): keep the final harness
        js = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(returns, '*', 'diag_out', f'prostate__{r}__ema.json')))]; js = [j for j in js if j['meta']['harness_sha256'] == hs]
        need(len(js) == 1, f'{r}: {len(js)} harness JSONs made with the final harness'); J[r] = js[0]
    m3 = J[REPS[2]]['meta']; need(m3['iter'] == i3 and m3['ckpt'].endswith(REPS[2] + '/best_student.pth') and m3['harness_sha256'] == J[REPS[0]]['meta']['harness_sha256'] == J[REPS[1]]['meta']['harness_sha256'], 'rep3 harness JSON provenance')
    end = re.search(r'\[(\d\d:\d\d:\d\d)[.\d]*\] TRAINING COMPLETE', t3).group(1); need(m3['timestamp'][-8:] > end and m3['timestamp'].startswith('2026-10-08'), 'harness ran before training ended (R-46)')
    F['sendback'] = {'files': len(man), 'quarantined': sorted(held), 'rep3_best': b3, 'rep3_best_iter': i3, 'runner': 'DONE: 1/1 PASS', 'args_differing_from_rep2': diff, 'harness_iter': m3['iter'], 'harness_timestamp': m3['timestamp'], 'training_end': end}
    # ---------------------------------------------------------------- 2. three trainings of one job line
    rows = []
    for r in REPS:
        d, t, best, it, dom = run_log(returns, 'prostate', r); pdj = J[r]['per_domain']; ks = sorted(pdj)
        u = [pdj[k]['signals']['u_epi'] for k in ks]
        rows.append(dict(run=r, best_avg=best, best_iter=it, **{f'dice_{v}': x for v, x in zip(VEND, dom)}, auroc_all=float(np.mean([x['auroc_all'] for x in u])),
                         auroc_band5=float(np.mean([x['band']['edt_gt_5']['auroc'] for x in u])), range80=float(np.mean([x['range80_all'] for x in u])), script_sha=json.load(open(os.path.join(d, 'run_config.json')))['script_sha256'][:12]))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(out, 'table_replicates_s18.csv'), index=False, float_format='%.6f')
    bs = pd.read_csv(os.path.join(derived, 'table_band_summary.csv')); x = bs[(bs.signal == 'u_epi') & (bs.run == REPS[0])].iloc[0]
    need(abs(T.auroc_all[0] - x.auroc_all_mean) < 1.5e-4 and abs(T.auroc_band5[0] - x.auroc_edt5_mean) < 1.5e-4, 'my reading of the harness JSON disagrees with derived_v04 for the default run')
    v = T.best_avg.values * 100; pair = {f'{i}-{j}': float(abs(v[i] - v[j])) for i, j in itertools.combinations(range(3), 2)}; pv = T[[f'dice_{k}' for k in VEND]].values * 100
    F['replicates'] = {'best': [float(x) for x in v], 'best_iter': [int(x) for x in T.best_iter], 'pairwise_abs_diff_points': pair, 'max_pair': max(pair.values()), 'first_pair': pair['0-1'], 'sd': float(v.std(ddof=1)), 'range': float(v.max() - v.min()),
                       'per_vendor_range_points': {k: float(pv[:, i].max() - pv[:, i].min()) for i, k in enumerate(VEND)}, 'per_vendor_range_max': float((pv.max(0) - pv.min(0)).max()),
                       'auroc_all': [float(x) for x in T.auroc_all], 'auroc_band5': [float(x) for x in T.auroc_band5], 'range80': [float(x) for x in T.range80], 'script_sha': list(T.script_sha)}
    Rp = F['replicates']
    need([round(x, 2) for x in v] == [86.30, 86.35, 85.91] and round(Rp['first_pair'], 2) == 0.05 and [round(pair['0-2'], 2), round(pair['1-2'], 2)] == [0.39, 0.45], f'replicate values {v} {pair}')
    need(all(x <= 0.705 for x in Rp['auroc_band5']) and all(x <= 1.5e-4 for x in Rp['range80']) and 2.0 <= Rp['per_vendor_range_max'] <= 2.7, f"replicate measurements {Rp['auroc_band5']} {Rp['range80']} {Rp['per_vendor_range_max']}")
    # ---------------------------------------------------------------- 3. reproduction: revision code against the original runs (seed 1337)
    Rr = os.path.join(release, 'RESULTS'); dig = {}
    for ds, fn in (('prostate', 'adaptive_threshold/prostate_logs.txt'), ('fundus', 'adaptive_threshold/fundus_logs.txt')):
        for g, s, a in re.findall(r'^(fixmatch|daep)_s(\d+):\s+avg=([\d.]+)', open(os.path.join(Rr, fn)).read(), flags=re.M): dig[(ds, g, int(s))] = float(a)
    for d_, s, a in re.findall(r'(mnms|busi)_fixmatch_s(\d+):\s*([\d.]+)', open(os.path.join(Rr, 'seed_split_migration', 'mnms_busi_threshold_logs.txt')).read()): dig[(d_, 'fixmatch', int(s))] = float(a)
    rows = []
    for label, ds, run, (dk, g) in REPRO:
        _, _, best, it, _ = run_log(returns, ds, run); ref = dig[(dk, g, 1337)]; rows.append(dict(configuration=label, revision_run=run, revision_best=best, original_best=ref, abs_diff_points=abs(best - ref) * 100))
    P = pd.DataFrame(rows); P['within_first_pair'] = P.abs_diff_points <= Rp['first_pair']; P['within_max_pair'] = P.abs_diff_points <= Rp['max_pair']
    P.to_csv(os.path.join(out, 'table_reproduction_s18.csv'), index=False, float_format='%.6f')
    need([round(x, 2) for x in P.abs_diff_points] == [0.55, 0.10, 0.01, 0.28, 0.56], f'reproduction differences {list(P.abs_diff_points)}')
    need(int(P.within_first_pair.sum()) == 1 and int(P.within_max_pair.sum()) == 3, 'counts within the replicate yardsticks (1 of 5 with the first pair, 3 of 5 with the largest)')
    s01 = dig[('prostate', 'daep', 1337)] * 100
    F['reproduction'] = {'rows': {r.configuration: {'revision': r.revision_best, 'original': r.original_best, 'abs_diff_points': r.abs_diff_points, 'within_first_pair': bool(r.within_first_pair), 'within_max_pair': bool(r.within_max_pair)} for r in P.itertuples()},
                         'min': float(P.abs_diff_points.min()), 'max': float(P.abs_diff_points.max()), 'n_within_first_pair': 1, 'n_within_max_pair': 3,
                         'prostate_daep_original_vs_three_replicates': [float(abs(x - s01)) for x in v], 'prostate_daep_original_below_lowest_replicate_by': float(v.min() - s01)}
    # ---------------------------------------------------------------- 4. the registered ramp comparison (Phase R) against the replicate spread
    _, _, ds_, _, _ = run_log(returns, 'prostate', 'v18R_prostate_daep_ramp10_s1337'); _, _, ts_, _, _ = run_log(returns, 'prostate', 'v18R_prostate_fixmatch_ramp10_s1337')
    whole = (T.best_avg[0] - P.revision_best[1]) * 100; short = (ds_ - ts_) * 100; shift = abs(short - whole)
    F['ramp'] = {'short': float(short), 'whole': float(whole), 'shift_points': float(shift), 'shift_over_max_replicate_diff': float(shift / Rp['max_pair']), 'shift_over_first_pair': float(shift / Rp['first_pair'])}
    need(round(short, 2) == -0.57 and round(whole, 2) == 0.10 and round(shift, 2) == 0.67 and 1.45 <= shift / Rp['max_pair'] <= 1.6, f"ramp facts {F['ramp']}")
    json.dump(F, open(os.path.join(out, 'facts_s18.json'), 'w'), indent=1, sort_keys=True)
    return F


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('derived', 'returns', 'release'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--out', default=''); a = ap.parse_args(); out = a.out or os.path.join(a.derived, 'extra_s18'); F = build(a.derived, a.returns, a.release, out); R = F['replicates']
    print(f"[sendback] {F['sendback']['files']} files, hashes match; quarantined {F['sendback']['quarantined']}; rep3 {F['sendback']['rep3_best']:.6f} at iter {F['sendback']['rep3_best_iter']}; harness at {F['sendback']['harness_timestamp']} (training ended {F['sendback']['training_end']})")
    print(f"[replicates] best {[round(x, 2) for x in R['best']]} at iters {R['best_iter']}; pairwise |diff| {dict((k, round(x, 2)) for k, x in R['pairwise_abs_diff_points'].items())}; sd {R['sd']:.2f}; per-vendor range max {R['per_vendor_range_max']:.2f} {dict((k, round(x, 2)) for k, x in R['per_vendor_range_points'].items())}")
    print(f"[replicates] AUROC all {[round(x, 3) for x in R['auroc_all']]}, band {[round(x, 3) for x in R['auroc_band5']]}, range {[f'{x:.1e}' for x in R['range80']]}; script sha {R['script_sha']}")
    print('[reproduction] ' + '; '.join(f"{k} {v['abs_diff_points']:.2f}" for k, v in F['reproduction']['rows'].items()) + f"; within first pair {F['reproduction']['n_within_first_pair']}/5, within largest pair {F['reproduction']['n_within_max_pair']}/5; original Prostate DAEP below the lowest replicate by {F['reproduction']['prostate_daep_original_below_lowest_replicate_by']:.2f}")
    print(f"[ramp] short {F['ramp']['short']:+.2f}, whole {F['ramp']['whole']:+.2f}, shift {F['ramp']['shift_points']:.2f} = {F['ramp']['shift_over_max_replicate_diff']:.2f} x largest replicate difference")
    print(f'[done] {out}: 2 tables + facts_s18.json; all assertions passed')
