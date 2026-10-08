#!/usr/bin/env python3
"""
make_derived_v04.py (S11; runs v03 -- and through it v02 -- UNCHANGED, then adds / corrects).
Nothing is typed by hand (R-14).

  python make_derived_v04.py --sendback <full server-state tree> --out derived_v04
  python make_derived_v04.py --selftest

Why v04 (first return with all 54 Phase-B runs complete, S10 collector of 2026-10-04):
  E-33 (mine)  v03's R-34 screen kept "the FIRST per-domain block of the last epoch". For MEDL runs the
               first block is net2 (then ensemble, then the headline net1), so the per-domain values it
               tabulated for MEDL were net2's; and its regex matched only `val_base_dice` lines, so on
               Fundus / M&Ms it returned "ok" with zero parsed domains (a vacuous pass, R-29). Its test
               `hd >= 100` also differs from the registered rule text "HD = 100" (the empty-prediction
               sentinel): a genuine HD of 134.6 is not the sentinel.
  E-35 (mine)  a run with tens of thousands of skipped non-finite iterations was reported as plain PASS;
               v02's table_runs carried only a boolean.
New / corrected outputs (v03's files are left as v03 wrote them, for comparison):
  table_run_screen_v04.csv   every run log: headline-student block of the last evaluation located by the
                             "Testing student... / Testing EMA teacher..." markers (works for all four
                             datasets and for MEDL), parse self-check against the logged student_avg,
                             R-34 under the REGISTERED rule (domain Dice <= 0.01 or HD == 100.00 sentinel on
                             every class of the domain) and, in a separate column, under v03's code (hd >= 100)
  table_nonfinite.csv        per run: number of skipped non-finite iterations, first/last, fraction of
                             max_iterations, lines whose printed terms are themselves nan/inf; flag
                             UNSTABLE if the fraction exceeds NONFINITE_FRACTION
  table_gate_stats.csv       per run: logged gate pass rate (w_mean), teacher U_epi mean and student S mean
                             after 20 % of training and at the end (from the periodic stats lines)
  table_phaseB_per_seed.csv  every Phase-B run and threshold reference: best/last Dice, per-domain Dice at
                             best, flags (R-34, UNSTABLE, frozen-student indicator), resume events, script
                             version at first launch and in the last code snapshot
  table_ramp_control.csv     Phase R (D-13) under the rule pre-registered in HANDOFF v1.9 section 3
  docs_tables_s11.md         markdown of the Phase-B tables for the docs (never hand-typed)
  assertions.json            v03 entries + R34_screen_v04, nonfinite_screen, frozen_student_indicator,
                             phaseR_rule, phaseB_health
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_derived_v02 as v02          # noqa: E402
import make_derived_v03 as v03          # noqa: E402

R34_FRACTION = v03.R34_FRACTION
R34_DICE = v03.R34_DICE_DEGENERATE
HD_SENTINEL = 100.0
NONFINITE_FRACTION = 0.01   # POST-HOC threshold (set in S11 after seeing the data): the two affected runs sit at
                            # 0.21 and 0.42, every other run at exactly 0, so any threshold in (0, 0.2) gives the same flags
FROZEN_TOL = 5e-4           # student and EMA per-domain Dice equal to 4 decimals at the last evaluation
SCRIPT_VERSIONS = {'ee2b7242': 'v6', '6314f5b8': 'v7', 'c55498c4': 'v7c/v8', 'a2731564': 'v9'}

DOM_RX = re.compile(r'domain(\d+) epoch (\d+): ((?:val_\w+_dice=[\d.]+\s*)+)((?:hd=[\d.]+\s*)+)')
SUM_RX = re.compile(r'\[epoch (\d+)\] student_avg=([\d.]+)\s+ema_avg=([\d.]+)\s+best_avg=([\d.]+) @ iter (\d+)')
STAT_RX = re.compile(r'\] iter (\d+): loss=([\w.]+), sup=([\w.]+), unsup=([\w.]+), spec=([\w.]+), cw=([\w.]+), '
                     r'w_mean=([\w.]+), w_min=([\w.]+), U_ale=([\w.]+), U_epi=([\w.]+), U_dom=([\w.]+), S_mean=([\w.]+)')
NF_RX = re.compile(r'\[iter (\d+)\] non-finite total loss, skipping\.(.*)')


def fl(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def parse_evaluations(txt):
    """Return a list of evaluations; each = {'epoch', 'student_avg', 'ema_avg', 'best_avg', 'best_iter',
    'student_blocks': [block, ...], 'ema_blocks': [...]} where block = {domain: (dice list, hd list)}.
    Blocks are assigned by the 'Testing student...' / 'Testing EMA teacher...' / 'Testing SWAD' markers, so the
    headline student is student_blocks[-1] for every method (MEDL logs net2, ensemble, net1 in that order)."""
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
                pending[int(m.group(1))] = ([float(x) for x in re.findall(r'dice=([\d.]+)', m.group(3))],
                                            [float(x) for x in re.findall(r'hd=([\d.]+)', m.group(4))])
            elif re.search(r'\] epoch \d+: avg ', line) or re.match(r'epoch \d+: avg ', line):
                if pending:
                    cur[phase].append(pending)
                pending = {}
        s = SUM_RX.search(line)
        if s and cur is not None:
            evals.append({'epoch': int(s.group(1)), 'student_avg': float(s.group(2)), 'ema_avg': float(s.group(3)),
                          'best_avg': float(s.group(4)), 'best_iter': int(s.group(5)),
                          'student_blocks': cur['student'], 'ema_blocks': cur['ema']})
            cur, phase = None, None
    return evals


def block_avg(block):
    return float(np.mean([np.mean(d) for d, _ in block.values()])) if block else float('nan')


def r34_flags(block):
    """(registered-rule degenerate domains, v03-code degenerate domains) for one per-domain block."""
    reg, old = [], []
    for d, (dice, hd) in sorted(block.items()):
        dd = float(np.mean(dice))
        sentinel = bool(hd) and all(abs(h - HD_SENTINEL) < 0.005 for h in hd)
        if dd <= R34_DICE or sentinel:
            reg.append(d)
        if dd <= R34_DICE or any(h >= HD_SENTINEL for h in hd):
            old.append(d)
    return reg, old


def screen_run(path):
    txt = open(path, errors='replace').read()
    mm = re.search(r'max_iterations=(\d+)', txt); max_it = int(mm.group(1)) if mm else None
    ne = re.search(r'num_eval_iter=(\d+)', txt); n_eval = int(ne.group(1)) if ne else 500
    evals = parse_evaluations(txt)
    out = {'max_iterations': max_it, 'complete': 'TRAINING COMPLETE' in txt, 'n_evals': len(evals),
           'resumed_iters': [int(x) for x in re.findall(r'RESUMED from iter (\d+)', txt)],
           'n_headers': txt.count('python train_fourierfuse'), 'is_medl': '[medl] net2_avg' in txt}
    nf = [(int(i), rest) for i, rest in NF_RX.findall(txt)]
    out['nf_n'] = len(nf); out['nf_first'] = nf[0][0] if nf else None; out['nf_last'] = nf[-1][0] if nf else None
    out['nf_printed_nonfinite'] = sum(1 for _, r in nf if re.search(r'=(nan|inf|-inf)\b', r))
    out['nf_frac'] = (len(nf) / max_it) if (nf and max_it) else 0.0
    stats = [(int(m[0]),) + tuple(fl(x) for x in m[1:]) for m in STAT_RX.findall(txt)]
    out['stats'] = stats
    if not evals:
        out.update({'last_iter': None}); return out
    e = evals[-1]
    head = e['student_blocks'][-1] if e['student_blocks'] else {}
    ema = e['ema_blocks'][0] if e['ema_blocks'] else {}
    reg, old = r34_flags(head)
    out.update({'last_iter': e['epoch'] * n_eval, 'student_avg_last': e['student_avg'], 'ema_avg_last': e['ema_avg'],
                'best_avg': e['best_avg'], 'best_iter': e['best_iter'], 'head': head, 'ema': ema,
                'n_student_blocks': len(e['student_blocks']),
                'parse_ok': bool(head) and abs(block_avg(head) - e['student_avg']) <= 6e-4,
                'r34_reg': reg, 'r34_old': old,
                'net2_avg_last': block_avg(e['student_blocks'][0]) if len(e['student_blocks']) == 3 else None,
                'ens_avg_last': block_avg(e['student_blocks'][1]) if len(e['student_blocks']) == 3 else None,
                'frozen': bool(head) and bool(ema) and set(head) == set(ema) and all(
                    abs(a - b) < FROZEN_TOL for d in head for a, b in zip(head[d][0], ema[d][0]))})
    return out


def flag_of(t, which='r34_reg'):
    if t.get('last_iter') is None:
        return 'no_eval_yet'
    if t['max_iterations'] and t['last_iter'] < R34_FRACTION * t['max_iterations']:
        return 'too_early'
    if not t.get('parse_ok'):
        return 'PARSE-MISMATCH'
    return ('DEGENERATE:dom' + ','.join(map(str, t[which]))) if t[which] else 'ok'


def sha_short(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8] if os.path.exists(p) else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sendback', default='.'); ap.add_argument('--out', default='derived_v04')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    sys.argv = [sys.argv[0], '--sendback', a.sendback, '--out', a.out]
    v03.main()
    A = json.load(open(os.path.join(a.out, 'assertions.json')))
    build(a.sendback, a.out, A)
    A['generator'] = 'make_derived_v04.py (S11)'
    json.dump(A, open(os.path.join(a.out, 'assertions.json'), 'w'), indent=1)
    print(f"[derived_v04] done -> {a.out}/assertions.json")


def build(sendback, out, A):
    T = {}
    for log in sorted(glob.glob(os.path.join(sendback, 'runs', '*', '*', 'log.txt'))):
        ds, run = log.split(os.sep)[-3], log.split(os.sep)[-2]
        T[(ds, run)] = screen_run(log)
    # ---- corrected screen ------------------------------------------------------------------------
    rows, deg_c, deg_i, differs, mism, frozen = [], [], [], [], [], []
    for (ds, run), t in sorted(T.items()):
        f_reg, f_old = flag_of(t, 'r34_reg'), flag_of(t, 'r34_old')
        head = t.get('head', {})
        rows.append([ds, run, 'COMPLETE' if t['complete'] else 'partial', int(v02.quarantined(run)), t['max_iterations'],
                     t.get('last_iter'), t['n_evals'], v02.f3(t.get('student_avg_last')), v02.f3(t.get('ema_avg_last')),
                     v02.f3(t.get('best_avg')), t.get('best_iter'), t.get('n_student_blocks', ''),
                     ';'.join(f"{np.mean(head[d][0]):.4f}" for d in sorted(head)),
                     ';'.join('/'.join(f"{h:.1f}" for h in head[d][1]) for d in sorted(head)),
                     v02.f3(t.get('net2_avg_last')), v02.f3(t.get('ens_avg_last')), int(bool(t.get('frozen'))),
                     ';'.join(map(str, t['resumed_iters'])), t['n_headers'], f_reg, f_old])
        if v02.quarantined(run):
            continue
        rec = {'dataset': ds, 'run': run, 'last_iter': t.get('last_iter'), 'student_avg_last': t.get('student_avg_last'),
               'best_avg': t.get('best_avg'), 'best_iter': t.get('best_iter'), 'flag': f_reg, 'complete': t['complete']}
        if f_reg.startswith('DEGENERATE'):
            (deg_c if t['complete'] else deg_i).append(rec)
        if f_reg == 'PARSE-MISMATCH':
            mism.append(run)
        if f_reg != f_old:
            differs.append({'run': run, 'registered_rule': f_reg, 'v03_code_hd_ge_100': f_old})
        if t.get('frozen') and t.get('last_iter') and t['max_iterations'] and t['last_iter'] >= R34_FRACTION * t['max_iterations']:
            frozen.append({'run': run, 'student_avg_last': t.get('student_avg_last'), 'ema_avg_last': t.get('ema_avg_last')})
    v02.wcsv(os.path.join(out, 'table_run_screen_v04.csv'), rows,
             ['dataset', 'run', 'state', 'quarantined', 'max_iterations', 'last_iter', 'n_evals', 'student_avg_last',
              'ema_avg_last', 'best_avg', 'best_iter', 'n_student_side_blocks', 'headline_student_dice_last_per_domain',
              'headline_hd_last_per_domain', 'medl_net2_avg_last', 'medl_ensemble_avg_last', 'ema_equals_student_last',
              'resumed_at_iters', 'n_launch_headers', 'R34_flag_registered_rule', 'R34_flag_v03_code'])
    A['R34_screen_v04'] = {
        'rule': f"REGISTERED rule (HANDOFF section 3 / RULES R-34): after {int(R34_FRACTION * 100)} % of max_iterations, a test domain "
                f"is degenerate if its headline-student Dice (mean over foreground classes) <= {R34_DICE} or its HD equals the "
                f"empty-prediction sentinel {HD_SENTINEL:.2f} on every class, at the latest evaluation. Multi-class extension "
                f"stated in S11 before any Fundus/M&Ms per-domain value was parsed.",
        'supersedes': 'R34_screen (v03): parsed net2 for MEDL, parsed nothing on Fundus/M&Ms, used hd >= 100 (E-33)',
        'n_runs_screened': len(rows), 'parse_mismatch': mism,
        'degenerate_incomplete': deg_i, 'degenerate_complete': deg_c, 'differs_from_v03_code': differs}
    A['frozen_student_indicator'] = {
        'definition': f'headline student and EMA teacher per-domain Dice agree within {FROZEN_TOL} on every domain/class at the '
                      f'last evaluation (after 20 % of training)',
        'verdict': 'NOT DIAGNOSTIC: read the base rate below before reading it on any suspect run (R-43)',
        'n_runs_with_indicator': len(frozen), 'n_runs_screened': len(rows), 'runs': [f['run'] for f in frozen]}
    # ---- non-finite screen -----------------------------------------------------------------------
    nrows, unstable = [], []
    for (ds, run), t in sorted(T.items()):
        fl_ = 'UNSTABLE' if t['nf_frac'] > NONFINITE_FRACTION else ('skips' if t['nf_n'] else 'clean')
        nrows.append([ds, run, int(t['complete']), int(v02.quarantined(run)), t['max_iterations'], t['nf_n'], t['nf_first'] or '',
                      t['nf_last'] or '', f"{t['nf_frac']:.4f}", t['nf_printed_nonfinite'], fl_])
        if fl_ == 'UNSTABLE' and not v02.quarantined(run):
            unstable.append({'run': run, 'n_skipped': t['nf_n'], 'first_iter': t['nf_first'], 'last_iter': t['nf_last'],
                             'fraction_of_max_iterations': round(t['nf_frac'], 4), 'best_iter': t.get('best_iter'),
                             'best_before_first_skip': bool(t.get('best_iter') is not None and t['nf_first'] and t['best_iter'] < t['nf_first'])})
    v02.wcsv(os.path.join(out, 'table_nonfinite.csv'), nrows,
             ['dataset', 'run', 'complete', 'quarantined', 'max_iterations', 'n_nonfinite_skips', 'first_skip_iter',
              'last_skip_iter', 'fraction_of_max_iterations', 'n_lines_with_nonfinite_printed_terms', 'flag'])
    A['nonfinite_screen'] = {
        'rule': f'a run is UNSTABLE if more than {NONFINITE_FRACTION:.0%} of max_iterations were skipped for a non-finite total '
                f'loss. POST-HOC threshold (S11, set after seeing the data); flags are identical for any threshold in (0, 0.2)',
        'n_runs': len(nrows), 'n_runs_with_any_skip': sum(1 for r in nrows if r[5]), 'unstable': unstable}
    # ---- gate statistics -------------------------------------------------------------------------
    grows = []
    for (ds, run), t in sorted(T.items()):
        st = [s for s in t['stats'] if t['max_iterations'] and s[0] >= R34_FRACTION * t['max_iterations']]
        if not st or v02.quarantined(run):
            continue
        # STAT_RX groups: 0 iter, 1 loss, 2 sup, 3 unsup, 4 spec, 5 cw, 6 w_mean, 7 w_min, 8 U_ale, 9 U_epi, 10 U_dom, 11 S_mean
        w = np.array([s[6] for s in st]); ue = np.array([s[9] for s in st]); sm = np.array([s[11] for s in st])
        grows.append([ds, run, len(st), v02.f3(float(np.nanmean(w))), v02.f3(float(np.nanmin(w))), v02.f3(st[-1][6]),
                      f"{np.nanmean(ue):.4f}", f"{st[-1][9]:.4f}", f"{np.nanmean(sm):.1f}", f"{st[-1][11]:.1f}",
                      int(np.isnan(np.array([s[1:] for s in st], dtype=float)).any())])
    v02.wcsv(os.path.join(out, 'table_gate_stats.csv'), grows,
             ['dataset', 'run', 'n_stat_lines_after_20pct', 'w_mean_avg', 'w_mean_min', 'w_mean_last', 'teacher_U_epi_avg',
              'teacher_U_epi_last', 'student_S_mean_avg', 'student_S_mean_last', 'any_nan_in_logged_stats'])
    # ---- Phase-B per-seed table with provenance --------------------------------------------------
    rx = re.compile(r'v18B_(\w+?)_(softmax_fixmatch|mcdropout2|devis|medl|dropoutctl2_fixmatch|devisobj_threshold|fixmatch)_'
                    r'(s(\d+)|split1337_init(\d+))$')
    prow, health = [], {}
    for (ds, run), t in sorted(T.items()):
        m = rx.match(run)
        if not m:
            continue
        p = v02.parse_log(os.path.join(sendback, 'runs', ds, run, 'log.txt')) or {}
        dom = {d: float(np.mean(list(v.values()))) for d, v in p.get('per_domain', {}).items()}
        rc_path = os.path.join(sendback, 'runs', ds, run, 'run_config.json')
        rc = json.load(open(rc_path)) if os.path.exists(rc_path) else {}
        first = SCRIPT_VERSIONS.get((rc.get('script_sha256') or '')[:8], (rc.get('script_sha256') or '')[:8])
        snap = sha_short(os.path.join(sendback, 'runs', ds, run, 'code', 'train_fourierfuse_v18_revision.py'))
        snap = SCRIPT_VERSIONS.get(snap, snap)
        f_reg = flag_of(t, 'r34_reg')
        nf = 'UNSTABLE' if t['nf_frac'] > NONFINITE_FRACTION else ''
        # student == EMA at the last evaluation is NOT a health flag: it holds for most healthy runs (base rate checked
        # in S11, see frozen_student_indicator); it stays a descriptive column of table_run_screen_v04.csv only (R-43).
        flags = ';'.join(x for x in [f_reg if f_reg != 'ok' else '', nf] if x)
        seed = m.group(4) or f"init{m.group(5)}"
        prow.append([ds, m.group(2), seed, run, int(t['complete']), v02.f3(p.get('best_avg')), p.get('best_iter', ''),
                     v02.f3(t.get('student_avg_last')), v02.f3(p.get('swad_best'))] +
                    [v02.f3(dom.get(d, float('nan'))) for d in range(1, 7)] +
                    [flags or 'ok', t['nf_n'], ';'.join(map(str, t['resumed_iters'])), len(rc.get('resume_events', [])), first, snap])
        if flags:
            health[run] = flags
    v02.wcsv(os.path.join(out, 'table_phaseB_per_seed.csv'), prow,
             ['dataset', 'method', 'seed', 'run', 'complete', 'best_avg', 'best_iter', 'student_avg_last', 'swad_best'] +
             [f'dom{d}_dice_at_best' for d in range(1, 7)] +
             ['health_flags', 'n_nonfinite_skips', 'resumed_at_iters', 'n_resume_events', 'script_first_launch', 'script_last_snapshot'])
    A['phaseB_health'] = {'n_phaseB_runs': len(prow), 'n_complete': sum(r[4] for r in prow), 'flagged': health,
                          'note': 'R-38: flagged runs stay in every table with their per-seed values; nothing is dropped or re-seeded'}
    # ---- Phase R (D-13), rule pre-registered 2026-10-03 in HANDOFF v1.9 section 3 ----------------
    R = {k: v02.parse_log(os.path.join(sendback, 'runs', k[0], k[1], 'log.txt')) for k in T}
    R = {k: v for k, v in R.items() if v and v['complete']}
    names = {'thr_short': 'v18R_prostate_fixmatch_ramp10_s1337', 'daep_short': 'v18R_prostate_daep_ramp10_s1337',
             'thr_whole': 'v18B_prostate_fixmatch_split1337_init1337', 'daep_whole': 'v18A_prostate_daep_default_s1337',
             'rep_a': 'v18A_prostate_daep_default_s1337', 'rep_b': 'v18A_prostate_daep_default_s1337_rep2'}
    got = {k: R.get(('prostate', n), {}).get('best_avg') for k, n in names.items()}
    res = {'rule': 'PRE-REGISTERED 2026-10-03 (before any v18R run existed): d_short = Dice(DAEP) - Dice(threshold) under the short '
                   'ramp, d_whole = the same difference for the headline s1337 pair; if |d_short - d_whole| <= the B2 replicate '
                   'tolerance |Dice(rep_a) - Dice(rep_b)|, the schedule is a disclosure with no consequence; otherwise the '
                   'Discussion paragraph is rewritten to report the dependence. Single seed -> exploratory.',
           'runs': names, 'best_avg': got, 'status': 'no data'}
    rrow = []
    if all(got[k] is not None for k in ('thr_short', 'daep_short', 'thr_whole', 'daep_whole')):
        d_s, d_w = got['daep_short'] - got['thr_short'], got['daep_whole'] - got['thr_whole']
        tol = abs(got['rep_a'] - got['rep_b']) if got['rep_a'] is not None and got['rep_b'] is not None else None
        res.update({'d_short': d_s, 'd_whole': d_w, 'abs_diff': abs(d_s - d_w), 'replicate_tolerance': tol,
                    'status': 'tolerance missing (replicate not complete)' if tol is None else
                              ('no consequence' if abs(d_s - d_w) <= tol else 'DEPENDENCE: rewrite the Discussion paragraph')})
        rrow.append([v02.f3(got['thr_short']), v02.f3(got['daep_short']), v02.f3(got['thr_whole']), v02.f3(got['daep_whole']),
                     v02.f3(d_s), v02.f3(d_w), v02.f3(abs(d_s - d_w)), v02.f3(tol), res['status']])
    v02.wcsv(os.path.join(out, 'table_ramp_control.csv'), rrow,
             ['thr_short', 'daep_short', 'thr_whole', 'daep_whole', 'd_short', 'd_whole', 'abs_diff', 'replicate_tolerance', 'status'])
    A['phaseR_rule'] = res
    docs_markdown(out, A, prow)


def docs_markdown(out, A, prow):
    B = list(csv.DictReader(open(os.path.join(out, 'table_baselines.csv'))))
    L = ['# Phase-B tables (generated by make_derived_v04.py -- never hand-edited)', '',
         '## Paired comparison with the threshold gate (pre-registered rule: paired t p < 0.05 AND mean delta >= 0.005)', '',
         '| dataset | method | n | mean | std | threshold ref | mean delta | per-seed delta (1337;42;2024) | t | p | sign-test p | beats | ref code-matched |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in B:
        L.append(f"| {r['dataset']} | {r['method']} | {r['n_seeds']} | {r['mean_dice']} | {r['std_dice']} | {r['threshold_ref_mean']} | "
                 f"{r['delta_vs_threshold']} | {r['per_seed_delta']} | {r['t_paired']} | {r['p_paired']} | {r['sign_test_p']} | "
                 f"{r['beats_threshold_rule']} | {r['reference_code_matched']} |")
    L += ['', '## Per-seed values and health flags (R-38: nothing dropped)', '',
          '| dataset | method | seed | best avg | best iter | last avg | flags | non-finite skips | resumed at | script first/last |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in prow:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[5]} | {r[6]} | {r[7]} | {r[15]} | {r[16]} | {r[17]} | {r[19]}/{r[20]} |")
    g = A.get('v18_reproduction_gate', {})
    L += ['', '## Reproduction gate (SUSPENDED until the D-10 replicate, E-17/R-32; values only)', '',
          '| key | v18 best avg | S01 reference | abs diff |', '|---|---|---|---|']
    for k, v in g.items():
        L.append(f"| {k} | {v['v18_best_avg']:.6f} | {v['S01_reference']:.6f} | {v['abs_diff']:.4f} |")
    L += ['', '## Health screens', '', '```', json.dumps({k: A[k] for k in ('nonfinite_screen', 'frozen_student_indicator')}, indent=1),
          json.dumps({k: A['R34_screen_v04'][k] for k in ('parse_mismatch', 'degenerate_incomplete', 'degenerate_complete',
                                                           'differs_from_v03_code')}, indent=1), '```', '']
    with open(os.path.join(out, 'docs_tables_s11.md'), 'w') as f:
        f.write('\n'.join(L))
    print(f"  wrote {out}/docs_tables_s11.md")


def _selftest():
    import tempfile
    d = tempfile.mkdtemp()

    def mk(ds, run, body):
        p = os.path.join(d, 'runs', ds, run); os.makedirs(p)
        open(os.path.join(p, 'log.txt'), 'w').write('\n'.join(body)); return os.path.join(p, 'log.txt')
    hdr = "[00:00:00.000] Namespace(dataset='prostate', max_iterations=60000, num_eval_iter=500)"
    # MEDL: net2 block empty (sentinel), ensemble, headline healthy, EMA different -> must NOT be degenerate
    b = [hdr]
    for e in (119, 120):
        b.append('[x] Testing student...')
        b += [f'[x] domain{k} epoch {e}: val_base_dice=0.0003 hd=100.00' for k in range(1, 7)] + [f'[x] epoch {e}: avg val_base_dice=0.0003 DSDV=0']
        b.append('[x] [medl] net2_avg=0.0003 ensemble_avg=0.4000')
        b += [f'[x] domain{k} epoch {e}: val_base_dice=0.4000 hd=50.00' for k in range(1, 7)] + [f'[x] epoch {e}: avg val_base_dice=0.4000 DSDV=0']
        b += [f'[x] domain{k} epoch {e}: val_base_dice=0.8000 hd=134.60' for k in range(1, 7)] + [f'[x] epoch {e}: avg val_base_dice=0.8000 DSDV=0']
        b.append('[x] Testing EMA teacher...')
        b += [f'[x] domain{k} epoch {e}: val_base_dice=0.8100 hd=12.00' for k in range(1, 7)] + [f'[x] epoch {e}: avg val_base_dice=0.8100 DSDV=0']
        b.append(f'[x] [epoch {e}] student_avg=0.8000  ema_avg=0.8100  best_avg=0.8200 @ iter 30000  swad_best=0 @ iter -1')
    b.append('[x] iter 60000: loss=1.0000, sup=0.0200, unsup=1.1000, spec=0.0001, cw=0.500, w_mean=0.990, w_min=0.970, '
             'U_ale=nan, U_epi=0.050, U_dom=0.006, S_mean=88.33, medl_R_mean=nan')
    b += ['[x] [iter 40001] non-finite total loss, skipping. sup=0.020 unsup=1.100 spec=0.000'] * 700 + \
         ['[x] [iter 40002] non-finite total loss, skipping. sup=nan unsup=1.100 spec=0.000'] + ['[x] TRAINING COMPLETE']
    t = screen_run(mk('prostate', 'v18B_prostate_medl_s1', b))
    assert t['n_student_blocks'] == 3 and t['parse_ok'] and t['r34_reg'] == [] and t['r34_old'] == [1, 2, 3, 4, 5, 6], t
    assert abs(t['net2_avg_last'] - 0.0003) < 1e-9 and not t['frozen'] and t['nf_n'] == 701 and t['nf_printed_nonfinite'] == 1
    assert flag_of(t) == 'ok' and flag_of(t, 'r34_old').startswith('DEGENERATE') and t['nf_frac'] > NONFINITE_FRACTION
    # Fundus: two classes per domain; domain 2 empty on both classes -> degenerate dom2; student == EMA -> frozen
    f = ["[00:00] Namespace(dataset='fundus', max_iterations=30000, num_eval_iter=500)"]
    for e in (59, 60):
        for tag in ('Testing student...', 'Testing EMA teacher...'):
            f.append(f'[x] {tag}')
            f += [f'[x] domain1 epoch {e}: val_cup_dice=0.8000 val_disc_dice=0.9000 hd=10.00 hd=8.00',
                  f'[x] domain2 epoch {e}: val_cup_dice=0.0003 val_disc_dice=0.0001 hd=100.00 hd=100.00',
                  f'[x] epoch {e}: avg val_cup_dice=0.4 val_disc_dice=0.45 DSDV=0']
        f.append(f'[x] [epoch {e}] student_avg=0.4251  ema_avg=0.4251  best_avg=0.5 @ iter 1000  swad_best=0 @ iter -1')
    t2 = screen_run(mk('fundus', 'v18B_fundus_devis_s1', f))
    assert t2['parse_ok'] and t2['r34_reg'] == [2] and t2['frozen'] and t2['nf_n'] == 0 and not t2['complete'], t2
    out = os.path.join(d, 'out'); os.makedirs(out)
    open(os.path.join(out, 'table_baselines.csv'), 'w').write('dataset,method,n_seeds,mean_dice,std_dice,threshold_ref_mean,delta_vs_threshold,'
                                                               'per_seed_delta,t_paired,p_paired,n_seeds_positive,sign_test_p,beats_threshold_rule,'
                                                               'reference_code_matched,reference_sources\n')
    A = {}
    build(d, out, A)
    g = list(csv.DictReader(open(os.path.join(out, 'table_gate_stats.csv'))))
    assert len(g) == 1 and g[0]['w_mean_last'] == '0.9900' and g[0]['teacher_U_epi_last'] == '0.0500' \
        and g[0]['student_S_mean_last'] == '88.3' and g[0]['any_nan_in_logged_stats'] == '1', g
    assert [x['run'] for x in A['nonfinite_screen']['unstable']] == ['v18B_prostate_medl_s1']
    assert [x['run'] for x in A['R34_screen_v04']['degenerate_incomplete']] == ['v18B_fundus_devis_s1']
    assert A['R34_screen_v04']['differs_from_v03_code'][0]['run'] == 'v18B_prostate_medl_s1' and A['phaseR_rule']['status'] == 'no data'
    assert A['frozen_student_indicator']['runs'] == ['v18B_fundus_devis_s1']
    print('[make_derived_v04] self-test PASSED')


if __name__ == '__main__':
    main()
