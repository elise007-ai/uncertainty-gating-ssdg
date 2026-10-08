#!/usr/bin/env python3
"""
make_derived_v03.py (S08; supersedes v02, which it imports and runs unchanged) — regenerates EVERY
revision table from raw artefacts. Nothing is typed by hand (R-14).

  python make_derived_v03.py --sendback <unzipped return dir> --out derived_v03
  python make_derived_v03.py --selftest

v03 adds (S08, first return with populated mm bands after the E-21 fix):
  table_mm_bands.csv          per run x domain x signal x band(mm): AUROC, central-80% range, pixel fraction,
                              error fraction, in-band error rate, plus the site's mm/px and "5 px in mm"
  table_mm_bands_summary.csv  per run x signal: 6-domain means of the mm curve (0.5/1/2/3/5 mm) next to the
                              pre-registered EDT-5px value and the all-pixel value
  table_run_screen.csv        R-34 screen of EVERY run log (complete or not): last iteration, best, last
                              per-domain student Dice, HD=100 domains, and the flag
                              (too_early / ok / DEGENERATE) under the pre-registered R-34 rule
  docs_table_mm_bands.md      markdown of the summary for the docs/handoff (never hand-typed)
  assertions.json             v02 entries + `mm_band_curve` (descriptive, not a pass/fail rule) + `R34_screen`
Reading rule for mm bands (pre-registered in RUNBOOK_S07 step 4 / HANDOFF v1.6 §3, Phase D): mm numbers are
quoted only from JSONs whose `n_images_with_mm == n_images` in every domain; any other JSON is listed under
`mm_band_curve.excluded`.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_derived_v02 as v02          # noqa: E402  (v02 is unchanged and stays in the tree)

MM_BANDS = ['0.5', '1.0', '2.0', '3.0', '5.0']
# in-plane mm/px at grid 384 from prostate_spacing_index.json v3 (N-15: I2CVB is anisotropic, ½(sx+sy) used)
SITE_MM_PER_PX = {'BIDMC': 0.3646, 'BMC': 0.4167, 'HK': 0.5208, 'I2CVB': 0.5 * (0.5343 + 0.4285),
                  'RUNMC': 0.5000, 'UCL': 0.4688}
R34_FRACTION = 0.20            # screen applies after 20 % of max_iterations (R-34)
R34_DICE_DEGENERATE = 0.01     # any test domain at or below this student Dice -> degenerate
R34_HD_DEGENERATE = 100.0      # HD = 100 is the "empty prediction" sentinel of the v16 metric


def mm_tables(H, out, A):
    """Per-domain and per-run mm-band tables from harness JSONs whose mm lookup was complete."""
    rows, summ, excluded = [], [], []
    for (ds, run, which), J in sorted(H.items()):
        doms = J['per_domain']
        n_mm = {d: r.get('n_images_with_mm', 0) for d, r in doms.items()}
        n_im = {d: r['n_images'] for d, r in doms.items()}
        if not all(n_mm[d] == n_im[d] and n_im[d] > 0 for d in doms):
            excluded.append({'run': f"{ds}:{run}:{which}", 'n_images_with_mm': n_mm, 'n_images': n_im})
            continue
        per_sig = {}
        for d, res in sorted(doms.items(), key=lambda kv: int(kv[0])):
            dname = v02.DOMAIN_NAMES[ds][int(d)]
            mmpx = SITE_MM_PER_PX.get(dname, float('nan'))
            for sig in v02.SIGNALS_ORDER:
                if sig not in res['signals']:
                    continue
                s = res['signals'][sig]; b = s['band']
                if not any(f'mm_gt_{m}' in b for m in MM_BANDS):
                    continue
                edt5 = b.get('edt_gt_5', {})
                for m in MM_BANDS:
                    e = b.get(f'mm_gt_{m}')
                    if e is None:
                        continue
                    rows.append([ds, run, which, d, dname, sig, m, f"{mmpx:.4f}", f"{5 * mmpx:.2f}",
                                 v02.f3(e.get('auroc')), f"{(e.get('range80') or float('nan')):.3e}",
                                 v02.f3(e.get('frac_pixels')), v02.f3(e.get('frac_errors')),
                                 v02.f3(e.get('error_rate')), v02.f3(s['auroc_all']), v02.f3(edt5.get('auroc'))])
                per_sig.setdefault(sig, []).append(
                    [b[f'mm_gt_{m}']['auroc'] if f'mm_gt_{m}' in b and b[f'mm_gt_{m}'].get('auroc') is not None
                     else float('nan') for m in MM_BANDS] +
                    [b[f'mm_gt_{m}']['frac_errors'] if f'mm_gt_{m}' in b else float('nan') for m in MM_BANDS] +
                    [s['auroc_all'], edt5.get('auroc', float('nan')), 5 * mmpx])
        for sig, vals in per_sig.items():
            a = np.array(vals, dtype=float); mean = np.nanmean(a, axis=0)
            summ.append([ds, run, which, sig, len(vals)] + [v02.f3(x) for x in mean[:5]] +
                        [v02.f3(x) for x in mean[5:10]] + [v02.f3(mean[10]), v02.f3(mean[11]), f"{mean[12]:.2f}"])
    v02.wcsv(os.path.join(out, 'table_mm_bands.csv'), rows,
             ['dataset', 'run', 'which', 'domain', 'domain_name', 'signal', 'band_mm', 'mm_per_px', 'five_px_in_mm',
              'auroc', 'range80', 'frac_pixels', 'frac_errors', 'error_rate_in_band', 'auroc_all', 'auroc_edt5px'])
    v02.wcsv(os.path.join(out, 'table_mm_bands_summary.csv'), summ,
             ['dataset', 'run', 'which', 'signal', 'n_domains'] + [f'auroc_mm{m}' for m in MM_BANDS] +
             [f'frac_errors_within_{m}mm' for m in MM_BANDS] + ['auroc_all', 'auroc_edt5px', 'five_px_in_mm_mean'])
    # descriptive assertion block (no new pass/fail rule; the pre-registered rule stays EDT-5px, R-21)
    curve = {}
    for r in summ:
        if r[3] in ('u_epi', 'conf') and r[0] == 'prostate':
            curve[f"{r[1]}:{r[3]}"] = {f'mm{m}': float(r[5 + i]) for i, m in enumerate(MM_BANDS)} | \
                                      {'edt5px': float(r[16]), 'all': float(r[15])}
    monotone = {k: all(v[f'mm{MM_BANDS[i]}'] <= v[f'mm{MM_BANDS[i + 1]}'] + 1e-9 for i in range(4))
                for k, v in curve.items()}
    phaseA = {k: v for k, v in curve.items() if k.startswith('v18A_') and k.endswith(':u_epi')}
    A['mm_band_curve'] = {
        'note': 'descriptive: u_epi/conf AUROC inside GT-boundary bands of physical width; the pre-registered '
                'collapse rule remains EDT-5px (R-21); quote the whole curve, not one width',
        'n_runs_with_complete_mm_lookup': len(set(r[1] for r in summ)),
        'excluded': excluded,
        'monotone_in_band_width_all_runs': bool(monotone) and all(monotone.values()),
        'phaseA_u_epi_max_auroc_by_band': {f'mm{m}': (max(v[f'mm{m}'] for v in phaseA.values()) if phaseA else None)
                                           for m in MM_BANDS},
        'phaseA_u_epi_n_above_0.70_by_band': {f'mm{m}': (sum(v[f'mm{m}'] > 0.70 for v in phaseA.values()) if phaseA else None)
                                              for m in MM_BANDS},
        'phaseA_n_configs': len(phaseA),
        'edt5px_bracketed_by_mm2_and_mm3_all_phaseA': bool(phaseA) and all(v['mm2.0'] <= v['edt5px'] <= v['mm3.0']
                                                                         for v in phaseA.values()),
        'curve': curve}
    return summ


def docs_mm_markdown(summ, out):
    lines = ['| run | signal | 0.5 mm | 1 mm | 2 mm | 3 mm | 5 mm | EDT-5 px | all pixels | 5 px ≈ mm |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in summ:
        if r[3] not in ('u_epi', 'conf'):
            continue
        want = 'conf' if 'softmax' in r[1] else 'u_epi'   # E-16: the two signals coincide; show the native one
        if r[3] != want:
            continue
        lines.append(f"| {r[1]} | {r[3]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | {r[9]} | {r[16]} | {r[15]} | {r[17]} |")
    with open(os.path.join(out, 'docs_table_mm_bands.md'), 'w') as f:
        f.write('# mm-band AUROC (6-vendor mean; generated by make_derived_v03.py — never hand-edited)\n\n'
                + '\n'.join(lines) + '\n')
    print(f"  wrote {out}/docs_table_mm_bands.md ({len(lines) - 2} rows)")


def parse_log_trajectory(path):
    txt = open(path, errors='replace').read()
    m_max = re.search(r'max_iterations=(\d+)', txt)
    max_it = int(m_max.group(1)) if m_max else None
    ep = re.findall(r'\[epoch (\d+)\] student_avg=([\d.]+)\s+ema_avg=([\d.]+)\s+best_avg=([\d.]+) @ iter (\d+)', txt)
    dom = re.findall(r'domain(\d+) epoch (\d+): val_base_dice=([\d.]+) hd=([\d.]+)', txt)
    n_eval = re.search(r'num_eval_iter=(\d+)', txt)
    n_eval = int(n_eval.group(1)) if n_eval else 500
    last = ep[-1] if ep else None
    stud = {}
    if last:
        # the student block is logged before the EMA block of the same epoch: keep the FIRST value per domain
        for d, e, v, hd in dom:
            if e == last[0] and int(d) not in stud:
                stud[int(d)] = (float(v), float(hd))
    return {'max_iterations': max_it, 'n_epochs_logged': len(ep), 'last_epoch': int(last[0]) if last else None,
            'last_iter': int(last[0]) * n_eval if last else None, 'student_avg_last': float(last[1]) if last else None,
            'ema_avg_last': float(last[2]) if last else None, 'best_avg': float(last[3]) if last else None,
            'best_iter': int(last[4]) if last else None, 'student_last_per_domain': stud,
            'complete': 'TRAINING COMPLETE' in txt, 'oom': 'out of memory' in txt,
            'resumed_iters': [int(x) for x in re.findall(r'RESUMED from iter (\d+)', txt)],
            'n_headers': txt.count('python train_fourierfuse')}


def run_screen_table(sendback, out, A):
    rows, flagged, degenerate_complete = [], [], []
    for log in sorted(glob.glob(os.path.join(sendback, 'runs', '*', '*', 'log.txt'))):
        ds = log.split(os.sep)[-3]; run = log.split(os.sep)[-2]
        t = parse_log_trajectory(log)
        state = 'COMPLETE' if t['complete'] else ('OOM-attempt' if t['oom'] and t['last_iter'] is None else 'partial')
        if t['last_iter'] is None:
            flag = 'no_eval_yet'
        elif t['max_iterations'] and t['last_iter'] < R34_FRACTION * t['max_iterations']:
            flag = 'too_early'
        else:
            bad = [d for d, (v, hd) in t['student_last_per_domain'].items()
                   if v <= R34_DICE_DEGENERATE or hd >= R34_HD_DEGENERATE]
            flag = f"DEGENERATE:dom{','.join(map(str, sorted(bad)))}" if bad else 'ok'
        if flag.startswith('DEGENERATE') and not v02.quarantined(run):
            rec = {'dataset': ds, 'run': run, 'last_iter': t['last_iter'], 'student_avg_last': t['student_avg_last'],
                   'best_avg': t['best_avg'], 'best_iter': t['best_iter'], 'flag': flag, 'complete': t['complete']}
            (degenerate_complete if t['complete'] else flagged).append(rec)   # R-34 is about runs still consuming compute
        pd = t['student_last_per_domain']
        rows.append([ds, run, state, int(v02.quarantined(run)), t['max_iterations'], t['last_iter'], t['n_epochs_logged'],
                     v02.f3(t['student_avg_last']), v02.f3(t['ema_avg_last']), v02.f3(t['best_avg']), t['best_iter'],
                     ';'.join(f"{pd[d][0]:.3f}" for d in sorted(pd)), ';'.join(f"{pd[d][1]:.1f}" for d in sorted(pd)),
                     ';'.join(map(str, t['resumed_iters'])), t['n_headers'], flag])
    v02.wcsv(os.path.join(out, 'table_run_screen.csv'), rows,
             ['dataset', 'run', 'state', 'quarantined', 'max_iterations', 'last_iter', 'n_evals', 'student_avg_last',
              'ema_avg_last', 'best_avg', 'best_iter', 'student_dice_last_per_domain', 'hd_last_per_domain',
              'resumed_at_iters', 'n_launch_headers', 'R34_flag'])
    A['R34_screen'] = {'rule': f"after {int(R34_FRACTION * 100)} % of max_iterations, any test domain with student Dice "
                               f"<= {R34_DICE_DEGENERATE} or HD >= {R34_HD_DEGENERATE} at the latest evaluation is DEGENERATE "
                               f"(R-34, S07); quarantined names are screened but not listed",
                       'n_runs_screened': len(rows), 'flagged_incomplete': flagged,
                       'degenerate_but_complete_informational': degenerate_complete}
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sendback', default='.'); ap.add_argument('--out', default='derived_v03')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    sys.argv = [sys.argv[0], '--sendback', a.sendback, '--out', a.out]
    v02.main()                                     # every v02 table + assertions.json (v02 part)
    A = json.load(open(os.path.join(a.out, 'assertions.json')))
    H = v02.load_harness(a.sendback)
    summ = mm_tables(H, a.out, A) if H else []
    if summ:
        docs_mm_markdown(summ, a.out)
    flagged = run_screen_table(a.sendback, a.out, A)
    A['generator'] = 'make_derived_v03.py (S08)'
    json.dump(A, open(os.path.join(a.out, 'assertions.json'), 'w'), indent=1)
    print(f"[derived_v03] mm rows for {len(summ)} run×signal; R-34 flagged: "
          f"{[f['run'] for f in flagged] or 'none'}; assertions -> {a.out}/assertions.json")


def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    log = os.path.join(d, 'runs', 'prostate', 'v18B_prostate_x_s1', 'log.txt'); os.makedirs(os.path.dirname(log))
    body = ["[00:00:00.000] Namespace(dataset='prostate', max_iterations=60000, num_eval_iter=500)"]
    for e in range(1, 27):
        for dd in range(1, 7):
            body.append(f"domain{dd} epoch {e}: val_base_dice={0.0003 if e >= 20 else 0.3:.4f} hd={100.0 if e >= 20 else 20.0:.2f}")
        for dd in range(1, 7):
            body.append(f"domain{dd} epoch {e}: val_base_dice=0.0003 hd=100.00")   # EMA block, must be ignored
        body.append(f"[epoch {e}] student_avg=0.2  ema_avg=0.0003  best_avg=0.25 @ iter 4000  swad_best=0 @ iter -1  best_DSDV=0")
    open(log, 'w').write('\n'.join(body))
    t = parse_log_trajectory(log)
    assert t['last_iter'] == 13000 and t['max_iterations'] == 60000 and len(t['student_last_per_domain']) == 6
    A = {}
    out = os.path.join(d, 'out'); os.makedirs(out)
    fl = run_screen_table(d, out, A)
    assert len(fl) == 1 and fl[0]['flag'] == 'DEGENERATE:dom1,2,3,4,5,6', fl
    # an early run must not be flagged
    body2 = [body[0]] + [x for x in body if ' epoch 1:' in x] + [body[13]]
    open(log, 'w').write('\n'.join(body2))
    fl = run_screen_table(d, out, A)
    assert fl == [] and A['R34_screen']['n_runs_screened'] == 1
    # mm table: a fake JSON with complete lookup and one with 0 matches (must be excluded)
    doms = {str(i): {'n_images': 3, 'n_images_with_mm': 3, 'error_rate': 0.01, 'signals': {'u_epi': {
        'auroc_all': 0.9, 'range80_all': 1e-4, 'median_all': 1e-3, 'band': {
            'edt_gt_5': {'auroc': 0.68, 'range80': 1e-4, 'frac_pixels': 0.05, 'frac_errors': 0.6, 'error_rate': 0.1},
            **{f'mm_gt_{m}': {'auroc': 0.55 + 0.05 * k, 'range80': 1e-4, 'frac_pixels': 0.01 * (k + 1),
                              'frac_errors': 0.4 + 0.1 * k, 'error_rate': 0.3} for k, m in enumerate(MM_BANDS)}}}}}
            for i in range(1, 7)}
    H = {('prostate', 'v18A_prostate_daep_default_s1337', 'ema'): {'per_domain': doms}}
    bad = json.loads(json.dumps(doms)); bad['1']['n_images_with_mm'] = 0
    H[('prostate', 'v18A_prostate_daep_x_s1337', 'ema')] = {'per_domain': bad}
    A = {}
    s = mm_tables(H, out, A)
    assert len(s) == 1 and A['mm_band_curve']['n_runs_with_complete_mm_lookup'] == 1 and len(A['mm_band_curve']['excluded']) == 1
    assert A['mm_band_curve']['monotone_in_band_width_all_runs'] and A['mm_band_curve']['phaseA_u_epi_n_above_0.70_by_band']['mm5.0'] == 1
    assert A['mm_band_curve']['edt5px_bracketed_by_mm2_and_mm3_all_phaseA']
    print('[make_derived_v03] self-test PASSED')


if __name__ == '__main__':
    main()
