#!/usr/bin/env python3
"""
make_derived_v01.py — regenerates EVERY revision table from raw artefacts
(harness JSONs, spectral JSON, run logs). Nothing is typed by hand (R-14).

  python make_derived_v01.py --sendback <unzipped sendback dir> --out derived_v01
  python make_derived_v01.py --selftest

Inputs (all optional; a table is skipped, not faked, when its input is absent):
  <sendback>/diag_out/<ds>__<run>__<which>.json      harness outputs
  <sendback>/spectral_out/spectral_distances.json    spectral analysis
  <sendback>/runs/<ds>/<run>/log.txt                 training logs (best_avg, per-domain)
Outputs (CSV + one JSON with the assertion results):
  table_band_generality.csv   per run x domain x signal: AUROC all / bands, ranges, error fractions
  table_band_summary.csv      per run: dataset means (the paper's Table-8 style row)
  table_calibration.csv       per run x domain: ECE / AUROC / range raw vs TS vs isotonic
  table_oracle_ceiling.csv    per run x domain: Dice as-is vs band-corrected (Fig. 5 style)
  table_threshold_sweep.csv   per run x domain: Dice vs tau (binary datasets)
  table_error_by_distance.csv per run x domain: error mass by distance bin
  table_spectral_loo.csv      leave-one-site-out z / hull / SWD / Frechet (with CIs)
  table_runs.csv              per run: best_avg, best_iter, per-domain Dice at best (from log.txt)
  table_evid_sweep.csv        Phase A: config -> Dice + boundary AUROC + range (+ rule check)
  table_baselines.csv         Phase B: method x dataset: 3-seed mean +- std, paired t vs threshold
  table_fda_sweep.csv         Phase C: beta / p -> avg Dice and BIDMC Dice
  assertions.json             pre-registered reading rules, evaluated (never hand-edited)
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, re, sys
from collections import defaultdict
import numpy as np

SIGNALS_ORDER = ['u_epi', 'conf', 'u_ale', 'entropy', 'mc_entropy', 'mc_mi', 'ens_entropy', 'ens_mi', 'ens_var', 'caef_u']
DOMAIN_NAMES = {'prostate': {1: 'BIDMC', 2: 'BMC', 3: 'HK', 4: 'I2CVB', 5: 'RUNMC', 6: 'UCL'},
                'fundus': {1: 'D1', 2: 'D2', 3: 'D3', 4: 'D4'}, 'MNMS': {1: 'A', 2: 'B', 3: 'C', 4: 'D'},
                'BUSI': {1: 'benign', 2: 'malignant'}}
# Threshold-gate reference (seed-split) best_avg Dice from the S01 release logs, for paired tests.
THRESHOLD_REF = {'prostate': {1337: 0.862933, 2024: 0.856871, 42: 0.851439},
                 'fundus': {1337: 0.882202, 2024: 0.886827, 42: 0.885062},
                 'MNMS': {1337: 0.807008, 2024: 0.808239, 42: 0.809903},
                 'BUSI': {1337: 0.703234, 2024: 0.714094, 42: 0.688449}}
FIXMATCH_S1337_PROSTATE_BIDMC = 0.807287   # RESULTS/adaptive_threshold/prostate_logs.txt, fixmatch_s1337 d1


def wcsv(path, rows, header):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path} ({len(rows)} rows)")


def f3(x):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


# ---------------------------------------------------------------------------
# Harness JSONs
# ---------------------------------------------------------------------------

def load_harness(sendback):
    out = {}
    for f in sorted(glob.glob(os.path.join(sendback, 'diag_out', '*__*__*.json'))):
        stem = os.path.basename(f)[:-5]
        ds, run, which = stem.split('__')
        try:
            out[(ds, run, which)] = json.load(open(f))
        except Exception as e:
            print(f"  [skip] {f}: {e!r}")
    return out


def band_tables(H, out):
    rows, summ = [], []
    for (ds, run, which), J in sorted(H.items()):
        per_run = defaultdict(list)
        for d, res in sorted(J['per_domain'].items(), key=lambda kv: int(kv[0])):
            for sig in SIGNALS_ORDER:
                if sig not in res['signals']:
                    continue
                s = res['signals'][sig]; b = s['band']
                def g(k, f='auroc'):
                    return b.get(k, {}).get(f, float('nan'))
                row = [ds, run, which, d, DOMAIN_NAMES[ds][int(d)], sig, res['n_images'], f3(res['error_rate']),
                       f3(s['auroc_all']), f3(g('edt_gt_1')), f3(g('edt_gt_2')), f3(g('edt_gt_3')), f3(g('edt_gt_5')),
                       f3(g('edt_gt_8')), f3(g('edt_gt_12')), f3(g('cheb_gt_5')), f3(g('edt_pred_5')), f3(g('edt_union_5')),
                       f3(g('rel_gt_0.05')), f3(g('rel_gt_0.1')), f3(g('mm_gt_1.0')), f3(g('mm_gt_2.0')),
                       f"{s['range80_all']:.3e}", f"{g('edt_gt_5', 'range80'):.3e}",
                       f3(g('edt_gt_5', 'frac_errors')), f3(g('edt_gt_5', 'frac_pixels')), f3(g('edt_gt_5', 'error_rate'))]
                rows.append(row)
                per_run[sig].append((s['auroc_all'], g('edt_gt_5'), s['range80_all'], g('edt_gt_5', 'range80'),
                                     g('edt_gt_5', 'frac_errors')))
        for sig, vals in per_run.items():
            a = np.array(vals, dtype=float)
            m = np.nanmean(a, axis=0)
            summ.append([ds, run, which, sig, len(vals), f3(m[0]), f3(m[1]), f3(m[0] - m[1]), f"{m[2]:.3e}", f"{m[3]:.3e}", f3(m[4])])
    wcsv(os.path.join(out, 'table_band_generality.csv'), rows,
         ['dataset', 'run', 'which', 'domain', 'domain_name', 'signal', 'n_images', 'error_rate', 'auroc_all',
          'auroc_edt1', 'auroc_edt2', 'auroc_edt3', 'auroc_edt5', 'auroc_edt8', 'auroc_edt12', 'auroc_cheb5',
          'auroc_predband5', 'auroc_unionband5', 'auroc_rel5pct', 'auroc_rel10pct', 'auroc_mm1', 'auroc_mm2',
          'range80_all', 'range80_edt5', 'frac_errors_in_edt5', 'frac_pixels_in_edt5', 'error_rate_in_edt5'])
    wcsv(os.path.join(out, 'table_band_summary.csv'), summ,
         ['dataset', 'run', 'which', 'signal', 'n_domains', 'auroc_all_mean', 'auroc_edt5_mean', 'drop',
          'range80_all_mean', 'range80_edt5_mean', 'frac_errors_in_edt5_mean'])
    return summ


def calibration_table(H, out):
    rows = []
    for (ds, run, which), J in sorted(H.items()):
        C = J.get('calibration', {})
        if 'per_domain' not in C:
            continue
        for d, r in sorted(C['per_domain'].items(), key=lambda kv: int(kv[0])):
            rows.append([ds, run, which, d, DOMAIN_NAMES[ds][int(d)], f3(C['temperature']), f3(r['acc']),
                         f3(r['ece_raw']), f3(r['ece_ts']), f3(r['ece_iso']), f3(r['auroc_raw']), f3(r['auroc_ts']),
                         f3(r['auroc_iso']), f"{r['range80_raw']:.3e}", f"{r['range80_ts']:.3e}", f"{r['range80_iso']:.3e}"])
    wcsv(os.path.join(out, 'table_calibration.csv'), rows,
         ['dataset', 'run', 'which', 'domain', 'domain_name', 'T_fit_on_labelled', 'acc', 'ece_raw', 'ece_ts', 'ece_iso',
          'auroc_raw', 'auroc_ts', 'auroc_iso', 'range80_raw', 'range80_ts', 'range80_iso'])


def oracle_threshold_tables(H, out):
    orows, trows, erows = [], [], []
    for (ds, run, which), J in sorted(H.items()):
        for d, res in sorted(J['per_domain'].items(), key=lambda kv: int(kv[0])):
            o = res['oracle_band']
            orows.append([ds, run, which, d, DOMAIN_NAMES[ds][int(d)], f3(o['dice_asis'])] +
                         [f3(o.get(f'dice_band_{r}', float('nan'))) for r in res['bands_px']] +
                         [f3(o.get('dice_band_5', float('nan')) - o['dice_asis'])])
            if 'threshold_sweep' in res:
                for t, v in sorted(res['threshold_sweep'].items(), key=lambda kv: float(kv[0])):
                    trows.append([ds, run, which, d, DOMAIN_NAMES[ds][int(d)], t, f3(v)])
            for e in res['error_by_distance']:
                erows.append([ds, run, which, d, DOMAIN_NAMES[ds][int(d)], e['d_lo'], e['d_hi'],
                              f3(e['frac_pixels']), f3(e['frac_errors']), f3(e['error_rate'])])
    bands = sorted({r for (ds, run, which), J in H.items() for res in J['per_domain'].values() for r in res['bands_px']})
    wcsv(os.path.join(out, 'table_oracle_ceiling.csv'), orows,
         ['dataset', 'run', 'which', 'domain', 'domain_name', 'dice_asis'] + [f'dice_band_{r}' for r in bands] + ['gain_band5'])
    wcsv(os.path.join(out, 'table_threshold_sweep.csv'), trows, ['dataset', 'run', 'which', 'domain', 'domain_name', 'tau', 'dice'])
    wcsv(os.path.join(out, 'table_error_by_distance.csv'), erows,
         ['dataset', 'run', 'which', 'domain', 'domain_name', 'd_lo', 'd_hi', 'frac_pixels', 'frac_errors', 'error_rate'])


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------

def spectral_table(sendback, out):
    f = os.path.join(sendback, 'spectral_out', 'spectral_distances.json')
    if not os.path.exists(f):
        print('  [skip] spectral_distances.json absent'); return None
    S = json.load(open(f)); rows = []
    for site, r in S['leave_one_out'].items():
        if site.startswith('_'):
            continue
        z = r['zscore_fda_scalar']
        rows.append([site, r['n']] + [f3(z[b]['z']) for b in ('0.01', '0.02', '0.05', '0.1', '0.2') if b in z] +
                    [r['inside_pca2_hull_of_others'], f3(r['swd_vs_others']['point']), f3(r['swd_vs_others']['lo95']),
                     f3(r['swd_vs_others']['hi95']), f3(r['frechet_vs_others']['point']), f3(r['frechet_vs_others']['lo95']),
                     f3(r['frechet_vs_others']['hi95']), f3(r['frechet_low_vs_others']['point']),
                     S['leave_one_out']['_rank_swd_vs_others'][site], S['leave_one_out']['_rank_frechet_vs_others'][site]])
    wcsv(os.path.join(out, 'table_spectral_loo.csv'), rows,
         ['site', 'n_slices', 'z_b0.01', 'z_b0.02', 'z_b0.05', 'z_b0.10', 'z_b0.20', 'inside_hull', 'swd', 'swd_lo95', 'swd_hi95',
          'frechet', 'fd_lo95', 'fd_hi95', 'frechet_lowfreq', 'rank_swd', 'rank_frechet'][:len(rows[0])] if rows else [])
    return S


# ---------------------------------------------------------------------------
# Run logs
# ---------------------------------------------------------------------------

def parse_log(path):
    txt = open(path, errors='replace').read()
    m = re.findall(r'Best avg dice \(student\): ([\d.]+) at iter (\d+)', txt)
    if not m:
        return None
    best, it = float(m[-1][0]), int(m[-1][1])
    dom = {}
    for d, part, v in re.findall(r'Domain (\d+) — (\w+): Dice=([\d.]+)', txt):
        dom.setdefault(int(d), {})[part] = float(v)
    swad = re.findall(r'Best SWAD dice: ([\d.]+)', txt)
    resumes = [int(x) for x in re.findall(r'RESUMED from iter (\d+)', txt)]
    return {'best_avg': best, 'best_iter': it, 'per_domain': dom, 'complete': 'TRAINING COMPLETE' in txt,
            'swad_best': float(swad[-1]) if swad else None, 'nonfinite': 'non-finite' in txt,
            'resumed_iters': resumes}


def runs_table(sendback, out):
    R = {}
    for log in sorted(glob.glob(os.path.join(sendback, 'runs', '*', '*', 'log.txt'))):
        ds = log.split(os.sep)[-3]; run = log.split(os.sep)[-2]
        p = parse_log(log)
        if p:
            R[(ds, run)] = p
    rows = []
    for (ds, run), p in sorted(R.items()):
        dom_avg = {d: np.mean(list(v.values())) for d, v in p['per_domain'].items()}
        rows.append([ds, run, int(p['complete']), int(p['nonfinite']), f3(p['best_avg']), p['best_iter'], f3(p['swad_best']),
                     ';'.join(map(str, p['resumed_iters']))] +
                    [f3(dom_avg.get(d, float('nan'))) for d in range(1, 7)])
    wcsv(os.path.join(out, 'table_runs.csv'), rows,
         ['dataset', 'run', 'complete', 'nonfinite', 'best_avg', 'best_iter', 'swad_best', 'resumed_at_iters'] + [f'dom{d}_dice' for d in range(1, 7)])
    return R


def evid_sweep_table(R, summ, out, A):
    """Phase A: v18A_prostate_daep_<cfg>_s1337 -> Dice + u_epi boundary metrics."""
    ref_band = {(r[1]): r for r in summ if r[3] == 'u_epi' and r[0] == 'prostate'}
    rows, checks = [], {}
    for (ds, run), p in sorted(R.items()):
        if not run.startswith('v18A_'):
            continue
        cfg = re.sub(r'^v18A_prostate_daep_|_s1337$', '', run)
        b = ref_band.get(run)
        auroc_all = float(b[5]) if b else float('nan'); auroc_b5 = float(b[6]) if b else float('nan')
        rg = float(b[8]) if b else float('nan')
        bidmc = np.mean(list(p['per_domain'].get(1, {}).values())) if p['per_domain'].get(1) else float('nan')
        rows.append([cfg, run, f3(p['best_avg']), f3(bidmc), f3(auroc_all), f3(auroc_b5), f"{rg:.3e}"])
        if b:
            checks[cfg] = {'band5_auroc': auroc_b5, 'range80': rg,
                           'collapse_persists': bool(auroc_b5 <= 0.70 and rg <= 1e-2)}
    wcsv(os.path.join(out, 'table_evid_sweep.csv'), rows,
         ['config', 'run', 'best_avg_dice', 'BIDMC_dice', 'u_epi_auroc_all', 'u_epi_auroc_edt5', 'u_epi_range80_all'])
    A['phaseA_rule'] = {'rule': 'collapse persists iff u_epi EDT-5px AUROC <= 0.70 AND central-80% range <= 1e-2 (6-vendor mean)',
                        'per_config': checks,
                        'counter_examples': [k for k, v in checks.items() if not v['collapse_persists']]}


def baselines_table(R, out, A):
    """Phase B: 3-seed mean +- std and paired t vs the seed-split threshold reference."""
    from scipy import stats
    groups = defaultdict(dict)
    for (ds, run), p in R.items():
        m = re.match(r'v18B_(\w+?)_(softmax_fixmatch|mcdropout|devis|medl|dropoutctl_fixmatch|devisobj_threshold)_s(\d+)$', run)
        if m:
            groups[(m.group(1), m.group(2))][int(m.group(3))] = p['best_avg']
    rows, verdicts = [], {}
    for (ds, meth), seeds in sorted(groups.items()):
        vals = [seeds[s] for s in (1337, 42, 2024) if s in seeds]
        ref = [THRESHOLD_REF[ds][s] for s in (1337, 42, 2024) if s in seeds]
        mean, std = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else float('nan'))
        delta = np.mean(np.array(vals) - np.array(ref))
        if len(vals) == 3:
            t, pval = stats.ttest_rel(vals, ref)
        else:
            t, pval = float('nan'), float('nan')
        beats = bool(len(vals) == 3 and pval < 0.05 and delta >= 0.005)
        rows.append([ds, meth, len(vals), f3(mean), f3(std), f3(np.mean(ref)), f3(delta), f3(t), f3(pval), int(beats)])
        verdicts[f'{ds}:{meth}'] = {'n': len(vals), 'delta_vs_threshold': float(delta), 'p': float(pval), 'beats_threshold': beats}
    wcsv(os.path.join(out, 'table_baselines.csv'), rows,
         ['dataset', 'method', 'n_seeds', 'mean_dice', 'std_dice', 'threshold_ref_mean', 'delta_vs_threshold', 't_paired', 'p_paired', 'beats_threshold_rule'])
    A['phaseB_rule'] = {'rule': 'a baseline beats the threshold iff paired t (3 seeds) p < 0.05 AND mean delta >= 0.5 Dice',
                        'verdicts': verdicts}
    # v18 reproduction gate (R-27)
    rep = {}
    for ds, ref, key in (('prostate-daep', 0.857467, ('prostate', 'v18A_prostate_daep_default_s1337')),
                         ('prostate', 0.862933, ('prostate', 'v18B_prostate_fixmatch_split1337_init1337')),
                         ('fundus', 0.882202, ('fundus', 'v18B_fundus_fixmatch_split1337_init1337')),
                         ('MNMS', 0.807008, ('MNMS', 'v18B_MNMS_fixmatch_s1337')),
                         ('BUSI', 0.703234, ('BUSI', 'v18B_BUSI_fixmatch_s1337'))):
        if key in R:
            rep[ds] = {'v18_best_avg': R[key]['best_avg'], 'S01_reference': ref, 'abs_diff': abs(R[key]['best_avg'] - ref),
                       'pass_within_0.005': abs(R[key]['best_avg'] - ref) <= 0.005,
                       'resumed_at_iters': R[key]['resumed_iters'],
                       'note': ('resumed run: data order after the resume is a fresh draw, so bit-exact '
                                'reproduction is not expected' if R[key]['resumed_iters'] else '')}
    A['v18_reproduction_gate'] = rep


def fda_table(R, out, A):
    rows, resp = [], {}
    for (ds, run), p in sorted(R.items()):
        m = re.match(r'v18C_prostate_fixmatch_(beta|pfda)([\d.]+)_s1337$', run)
        if not m:
            continue
        bidmc = np.mean(list(p['per_domain'].get(1, {}).values())) if p['per_domain'].get(1) else float('nan')
        rows.append([m.group(1), m.group(2), run, f3(p['best_avg']), f3(bidmc), f3(bidmc - FIXMATCH_S1337_PROSTATE_BIDMC)])
        resp[f'{m.group(1)}={m.group(2)}'] = float(bidmc - FIXMATCH_S1337_PROSTATE_BIDMC)
    wcsv(os.path.join(out, 'table_fda_sweep.csv'), rows,
         ['param', 'value', 'run', 'best_avg_dice', 'BIDMC_dice', 'BIDMC_delta_vs_default'])
    A['phaseC_rule'] = {'rule': 'BIDMC responds to FDA iff any beta/p setting raises BIDMC Dice by > 0.015 over the default '
                                '(fixmatch s1337 BIDMC = 0.807287); single seed -> exploratory',
                        'bidmc_delta': resp, 'responds': bool(any(v > 0.015 for v in resp.values()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sendback', default='.'); ap.add_argument('--out', default='derived_v01')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    os.makedirs(a.out, exist_ok=True)
    A = {}
    H = load_harness(a.sendback)
    print(f"[derived] {len(H)} harness JSONs")
    summ = band_tables(H, a.out) if H else []
    if H:
        calibration_table(H, a.out); oracle_threshold_tables(H, a.out)
    spectral_table(a.sendback, a.out)
    R = runs_table(a.sendback, a.out)
    print(f"[derived] {len(R)} run logs")
    evid_sweep_table(R, summ, a.out, A)
    baselines_table(R, a.out, A)
    fda_table(R, a.out, A)
    json.dump(A, open(os.path.join(a.out, 'assertions.json'), 'w'), indent=1)
    print(f"[derived] assertions -> {a.out}/assertions.json")


def _selftest():
    import tempfile
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, 'diag_out')); os.makedirs(os.path.join(d, 'runs', 'prostate', 'v18A_prostate_daep_kl1.0_lin_s1337'))
    band = {f'edt_gt_{r}': {'auroc': 0.6, 'range80': 1e-4, 'frac_pixels': 0.1, 'frac_errors': 0.9, 'error_rate': 0.05} for r in (1, 2, 3, 5, 8, 12)}
    band.update({'cheb_gt_5': dict(band['edt_gt_5']), 'edt_pred_5': dict(band['edt_gt_5']), 'edt_union_5': dict(band['edt_gt_5']),
                 'rel_gt_0.05': dict(band['edt_gt_5']), 'rel_gt_0.1': dict(band['edt_gt_5'])})
    sig = {'auroc_all': 0.87, 'range80_all': 9e-5, 'median_all': 1e-3, 'band': band}
    dom = {'n_images': 10, 'n_pixels': 100, 'error_rate': 0.01, 'signals': {'u_epi': sig, 'conf': sig}, 'bands_px': [1, 2, 3, 5, 8, 12],
           'error_by_distance': [{'d_lo': 0, 'd_hi': 1, 'frac_pixels': 0.1, 'frac_errors': 0.5, 'error_rate': 0.05}],
           'oracle_band': {'dice_asis': 0.78, 'dice_band_5': 0.90, 'dice_band_1': 0.8, 'dice_band_2': 0.85, 'dice_band_3': 0.87, 'dice_band_8': 0.93, 'dice_band_12': 0.95},
           'threshold_sweep': {'0.5': 0.78, '0.9': 0.775}}
    J = {'meta': {}, 'per_domain': {str(i): dom for i in range(1, 7)},
         'calibration': {'temperature': 1.6, 'per_domain': {'1': {'acc': 0.99, 'ece_raw': 0.01, 'ece_ts': 0.005, 'ece_iso': 0.004,
                                                                  'auroc_raw': 0.87, 'auroc_ts': 0.87, 'auroc_iso': 0.87,
                                                                  'range80_raw': 1e-4, 'range80_ts': 2e-3, 'range80_iso': 1e-2}}}}
    json.dump(J, open(os.path.join(d, 'diag_out', 'prostate__v18A_prostate_daep_kl1.0_lin_s1337__ema.json'), 'w'))
    log = ("[00:00] TRAINING COMPLETE — FourierFuse v16 summary\n[00:00]   Best avg dice (student): 0.851234 at iter 42000\n"
           "[00:00]     Domain 1 — base: Dice=0.812345, HD95=1.0, ASD=1.0\n[00:00]     Domain 2 — base: Dice=0.85, HD95=1.0, ASD=1.0\n")
    open(os.path.join(d, 'runs', 'prostate', 'v18A_prostate_daep_kl1.0_lin_s1337', 'log.txt'), 'w').write(log)
    out = os.path.join(d, 'derived'); os.makedirs(out)
    A = {}
    H = load_harness(d); summ = band_tables(H, out); calibration_table(H, out); oracle_threshold_tables(H, out)
    R = runs_table(d, out); evid_sweep_table(R, summ, out, A); baselines_table(R, out, A); fda_table(R, out, A)
    assert A['phaseA_rule']['per_config']['kl1.0_lin']['collapse_persists'] is True
    assert R[('prostate', 'v18A_prostate_daep_kl1.0_lin_s1337')]['best_avg'] == 0.851234
    rows = list(csv.reader(open(os.path.join(out, 'table_band_summary.csv'))))
    assert rows[1][5] == '0.8700' and rows[1][6] == '0.6000', rows[1]
    print('[make_derived_v01] self-test PASSED')


if __name__ == '__main__':
    main()
