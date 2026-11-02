#!/usr/bin/env python3
"""
make_extra_s43b.py (S43, 2026-10-27) -- reader for the S43b text layer: the queued review items N-75, N-81, F-S43b-1, C9 and N-82 of
WRITING/S43b_factcheck.md, assertion-first (R-48); nothing typed by hand (R-14). CPU only, from files already held: the harness JSONs under
RETURNS/*/diag_out, the derived band summaries and threshold-sweep table, and the release data loader. No training, no new measurement.
  python make_extra_s43b.py --returns <RETURNS> --derived <derived_v04_S12> --release <release_s12> --out <derived_v04_S12/extra_s43b>
Writes table_s43b_table11_identity.csv, table_s43b_families.csv, table_s43b_sweep_spans.csv, facts_s43b.json.
  N-75     the vacuity-confidence AUROC gap of the four Table 11 teachers (caption: "to within 10^-3"), and one testbed-objective teacher that exceeds
           it (so the unscoped sentence of Section 4.6 is false; the registered tolerance is that of Section 4.5).
  N-81     range (central 80 %, all pixels, mean over domains: the measure of the Phase-A rule) and boundary-band AUROC of every signal family the
           letter's General comment names; F-S43b-1 the same facts for the 30 DEviS and MEDL checkpoints of S9.
  C9       the operating-point sweep: per-teacher and mean-curve spans, against the text ("at most 1.5 ... less than 0.8") and Figure 4 ("0.1 to 1.0").
  N-82     the BUSI domains of the data loader.
"""
from __future__ import annotations
import argparse, ast, csv, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_extra_s23 import need, wcsv, PASSED, RULE_BAND, RULE_RANGE   # noqa: E402

T11 = ('v18B_prostate_fixmatch_split1337_init1337', 'v18B_prostate_fixmatch_s42', 'v18B_prostate_fixmatch_s2024', 'v18A_prostate_daep_default_s1337')
S9_EXCEPTION = ('fundus', 'v18B_fundus_fixmatch_split1337_init1337')   # the one testbed-objective checkpoint S9 names as exceeding 0.005
SEED = r'_s(1337|42|2024)$'
FAMILIES = {   # family -> (run regex, signals): the gate signal first
    'softmax head': (r'^v18B_(prostate|fundus|MNMS|BUSI)_softmax_fixmatch' + SEED, ('conf',)),
    'MC-dropout gate': (r'^v18B_(prostate|fundus)_mcdropout2' + SEED, ('mc_entropy', 'u_epi')),
    'DEviS': (r'^v18[BDE]_(prostate|fundus)_(devis|devisobj_threshold|devisrel_threshold|devisrel_belief)' + SEED, ('u_epi', 'conf')),
    'MEDL': (r'^v18B_(prostate|fundus)_medl' + SEED, ('caef_u', 'u_epi')),
    'ensemble': (r'^ENSEMBLE3_split1337_fixmatch$', ('conf', 'ens_entropy')),
}
# the signal each run's gate acts on (trainer v12 lines 1417-1443): threshold gates on the confidence; the MC-dropout gate on the MC entropy; the
# DEviS soft and belief-mass gates on U_epi (1 - u); the MEDL gate on the reliability of the CAEF-fused output (caef_u is its uncertainty). The
# ensembles were never used as a gate ("No ensemble-gated training was run"); their confidence stands for the signal such a gate would use.
GATE_SIGNAL = ((r'_softmax_fixmatch_', 'conf'), (r'_mcdropout2_', 'mc_entropy'), (r'_devisobj_threshold_|_devisrel_threshold_', 'conf'),
               (r'_devis_s|_devisrel_belief_', 'u_epi'), (r'_medl_', 'caef_u'), (r'^ENSEMBLE3_', 'conf'))
f2 = lambda x: f'{x:.2f}'


def jsons(rt, ds, run):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(rt, '*', 'diag_out', f'{ds}__{run}__ema.json')))]


def gaps(J):
    """max over domains of |AUROC(conf) - AUROC(u_epi)|, over all pixels and in the 5-px band"""
    a = b = 0.0
    for res in J['per_domain'].values():
        s = res['signals']
        a = max(a, abs(s['conf']['auroc_all'] - s['u_epi']['auroc_all']))
        b = max(b, abs(s['conf']['band']['edt_gt_5']['auroc'] - s['u_epi']['band']['edt_gt_5']['auroc']))
    return a, b


def main():
    ap = argparse.ArgumentParser()
    for k in ('returns', 'derived', 'release', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    F = {'session': 'S43 (for the S43b layer)', 'reader': 'make_extra_s43b.py', 'source': 'WRITING/S43b_factcheck.md'}
    # ------------------------------------------------ N-75
    rows_t = []; wa = wb = 0.0
    for run in T11:
        C = jsons(a.returns, 'prostate', run); need(bool(C), f'{run}: a harness JSON is held')
        G = [gaps(J) for J in C]
        need(all(abs(g[0] - G[0][0]) < 1e-9 and abs(g[1] - G[0][1]) < 1e-9 for g in G), f'{run}: every held copy of its harness JSON gives the same vacuity-confidence gaps')
        wa, wb = max(wa, G[0][0]), max(wb, G[0][1]); rows_t.append(['prostate', run, len(C), f'{G[0][0]:.6f}', f'{G[0][1]:.6f}'])
    need(wa <= 1e-3 and wb <= 1e-3, 'the four Table 11 teachers: confidence and vacuity give the same AUROC to within 1e-3, over all pixels and in the band (the caption holds)')
    ex = gaps(jsons(a.returns, *S9_EXCEPTION)[-1]); rows_t.append([S9_EXCEPTION[0], S9_EXCEPTION[1], 1, f'{ex[0]:.6f}', f'{ex[1]:.6f}'])
    need(ex[0] > 1e-3 and 0.005 < ex[0] < 0.0065, 'a testbed-objective teacher (the S9 exception) exceeds 1e-3, by about 0.001 above the 0.005 tolerance: '
         '"for the testbed objective ... to within 10^-3" (Section 4.6) is false unscoped; the registered tolerance is that of Section 4.5')
    F['N75'] = {'table11_max_gap_all': wa, 'table11_max_gap_band': wb, 'exception_run': S9_EXCEPTION[1], 'exception_gap_all': ex[0], 'exception_gap_band': ex[1]}
    wcsv(os.path.join(a.out, 'table_s43b_table11_identity.csv'), rows_t, ['dataset', 'run', 'n_json_copies', 'max_gap_all', 'max_gap_band5'])
    # ------------------------------------------------ N-81 and F-S43b-1: the signal families, from the derived band summaries
    B, clash, n_rows = {}, [], 0
    for p in sorted(glob.glob(os.path.join(a.derived, '**', 'table_band_summary*.csv'), recursive=True)):
        for r in csv.DictReader(open(p, encoding='utf-8')):
            if r['which'] != 'ema':
                continue
            k = (r['dataset'], r['run'], r['signal']); v = (float(r['auroc_edt5_mean']), float(r['range80_all_mean']), float(r['range80_edt5_mean']))
            if k in B and not all(abs(x - y) < 1e-9 for x, y in zip(B[k], v)):
                clash.append(k)
            B[k] = v; n_rows += 1
    need(not clash and n_rows > 500, f'the derived band summaries ({n_rows} EMA rows) agree wherever a run and signal appear twice')
    fam, rows_f = {}, []
    for name, (rx, sigs) in FAMILIES.items():
        for sig in sigs:
            sel = {(d, r): v for (d, r, s), v in B.items() if s == sig and re.search(rx, r)}
            for ds in sorted({d for d, _ in sel}):
                vs = [v for (d, _), v in sel.items() if d == ds]
                e = fam.setdefault(name, {}).setdefault(sig, {})[ds] = {
                    'n': len(vs), 'band_min': min(v[0] for v in vs), 'band_max': max(v[0] for v in vs), 'range_all_min': min(v[1] for v in vs),
                    'range_all_max': max(v[1] for v in vs), 'range_band_min': min(v[2] for v in vs), 'range_band_max': max(v[2] for v in vs),
                    'n_phaseA_persists': sum(v[0] <= RULE_BAND and v[1] <= RULE_RANGE for v in vs)}
                rows_f.append([name, sig, ds, e['n'], f2(e['band_min']), f2(e['band_max']), f"{e['range_all_min']:.2e}", f"{e['range_all_max']:.2e}",
                               f"{e['range_band_min']:.2e}", f"{e['range_band_max']:.2e}", e['n_phaseA_persists']])
    n = lambda fm, sg, ds: fam[fm][sg][ds]['n']
    need(all(n('softmax head', 'conf', d) == 3 for d in ('prostate', 'fundus', 'MNMS', 'BUSI')) and all(n('MC-dropout gate', 'mc_entropy', d) == 3 for d in ('prostate', 'fundus'))
         and all(n('DEviS', 'u_epi', d) == 12 for d in ('prostate', 'fundus')) and all(n('MEDL', 'u_epi', d) == 3 for d in ('prostate', 'fundus'))
         and all(n('ensemble', 'conf', d) == 1 for d in ('prostate', 'fundus')),
         'run counts: softmax head 3 per benchmark on four; MC-dropout gate 3 x 2; DEviS 12 x 2 (published, objective, released, belief gate); MEDL 3 x 2; ensemble 1 x 2')
    s9 = [B[k][0] for k in B if k[2] == 'u_epi' and (re.search(FAMILIES['DEviS'][0], k[1]) or re.search(FAMILIES['MEDL'][0], k[1]))]
    need(len(s9) == 30 and (f2(min(s9)), f2(max(s9))) == ('0.51', '0.72'), 'the 30 DEviS and MEDL checkpoints of S9: band AUROC of U_epi 0.51-0.72 (S9 and Section 4.5 hold)')
    mf = fam['MEDL']['u_epi']['fundus']; mc = fam['MEDL']['caef_u']['fundus']
    need(mf['range_all_min'] > RULE_RANGE and mf['band_max'] < 0.65 and mf['n_phaseA_persists'] == 0,
         'F-S43b-1: MEDL\'s U_epi on Fundus has a central range above the 10^-2 line of the Phase-A rule on all three runs, while its band AUROC stays below 0.65: '
         '"neither escapes the collapse" (S9, Section 4.5) is false for it as a range statement')
    need(mc['range_all_max'] <= RULE_RANGE, 'MEDL\'s fused uncertainty (CAEF; its gate thresholds the reliability computed from it, trainer line 1439) on Fundus '
         'stays below the 10^-2 line: the per-network vacuity, not the gate signal, exceeds it')
    GS, amb = {}, []                                # (dataset, run) -> the band AUROC, all-pixel range and band range of the signal its gate acts on
    for name, (rx, _) in FAMILIES.items():
        for (d, r, s), v in B.items():
            if re.search(rx, r):
                g = [sg for pat, sg in GATE_SIGNAL if re.search(pat, r)]
                if len(g) != 1:
                    amb.append(r)
                elif s == g[0]:
                    GS[(d, r)] = (name, s, v)
    need(not amb, f'every run of the five families maps to exactly one gate signal (ambiguous: {sorted(set(amb))[:3]})')
    need(len(GS) == 12 + 6 + 24 + 6 + 2 and {d for d, _ in GS} == {'prostate', 'fundus', 'MNMS', 'BUSI'},
         'the gate signal of every run of the five families: softmax 12, MC dropout 6, DEviS 24, MEDL 6, ensembles 2')
    need(all(v[1] <= RULE_RANGE for (d, r), (nm, s, v) in GS.items() if nm != 'ensemble'),
         'the signal every trained gate acts on stays within the 10^-2 range line of the Phase-A rule, on every benchmark it was trained on')
    pf = [v[0] for (d, r), (nm, s, v) in GS.items() if d in ('prostate', 'fundus')]
    need((f2(min(pf)), f2(max(pf))) == ('0.51', '0.72'), 'on Prostate and Fundus the band AUROC of the signal each gate acts on, the ensembles\' confidence included, lies in 0.51-0.72')
    need(sum(v[0] <= RULE_BAND for (d, r), (nm, s, v) in GS.items() if d in ('prostate', 'fundus')) >= 30, 'most of them at or below the 0.70 line ("at or near")')
    gate = (('softmax head', 'conf', ('prostate', 'fundus', 'MNMS', 'BUSI')), ('MC-dropout gate', 'mc_entropy', ('prostate', 'fundus')), ('MEDL', 'caef_u', ('prostate', 'fundus')))
    need(all(fam[f][s][d]['range_all_max'] <= RULE_RANGE for f, s, ds in gate for d in ds), 'the same, per family')
    u30 = {(d, r): B[(d, r, 'u_epi')] for (d, r, s) in B if s == 'u_epi' and (re.search(FAMILIES['DEviS'][0], r) or re.search(FAMILIES['MEDL'][0], r))}
    wide = sorted((d, r) for (d, r), v in u30.items() if v[1] > RULE_RANGE)
    need(len(u30) == 30 and len(wide) == 3 and all(d == 'fundus' and '_medl_' in r for d, r in wide),
         'the U_epi range of the 30 DEviS and MEDL checkpoints: 27 below the 10^-2 line; the three above it are MEDL on Fundus')
    F['gate_signals'] = {f'{d}|{r}': {'family': nm, 'signal': s, 'band_auroc': v[0], 'range80_all': v[1], 'range80_band': v[2]} for (d, r), (nm, s, v) in sorted(GS.items())}
    F['S9_u_epi_wide'] = {'n_checkpoints': len(u30), 'n_below_line': len(u30) - len(wide), 'wide': [r for _, r in wide],
                          'range_min': mf['range_all_min'], 'range_max': mf['range_all_max'], 'band_min': mf['band_min'], 'band_max': mf['band_max']}
    sm = fam['softmax head']['conf']['MNMS']
    need(sm['band_min'] > 0.72, 'the softmax confidence on M&Ms ranks band errors above 0.72 on all three runs: the letter\'s "0.51 to 0.72" is false unscoped')
    en = fam['ensemble']
    need(all(en['conf'][d]['range_all_max'] < 1e-3 for d in ('prostate', 'fundus')) and all(en['ens_entropy'][d]['range_band_min'] > 0.5 for d in ('prostate', 'fundus')),
         'the ensembles: over all pixels the range stays below 1e-3; in the boundary band the entropy range exceeds 0.5 (Section 4.6: 0.63 / 0.52)')
    F['families'] = fam; F['S9_band_u_epi'] = [min(s9), max(s9)]; F['prostate_fundus_band_all_gate_signals'] = [min(pf), max(pf)]
    wcsv(os.path.join(a.out, 'table_s43b_families.csv'), rows_f, ['family', 'signal', 'dataset', 'n_runs', 'band_auroc_min', 'band_auroc_max', 'range80_all_min',
         'range80_all_max', 'range80_band_min', 'range80_band_max', 'n_phaseA_collapse_persists'])
    # ------------------------------------------------ C9: the operating-point sweep of the three threshold teachers
    SW = {}
    for r in csv.DictReader(open(os.path.join(a.derived, 'table_threshold_sweep.csv'), encoding='utf-8')):
        if r['dataset'] == 'prostate' and r['run'] in T11[:3] and r['which'] == 'ema':
            SW.setdefault((r['run'], r['domain_name']), []).append((float(r['tau']), float(r['dice'])))
    vend = sorted({v for _, v in SW}); need(len(vend) == 6 and len(SW) == 18 and all(len(c) == 19 for c in SW.values()), 'sweep: 3 teachers x 6 vendors x 19 thresholds')
    span = {k: 100 * (max(d for _, d in c) - min(d for _, d in c)) for k, c in SW.items()}
    mean = {v: [sum(dict(SW[(t, v)])[tau] for t in T11[:3]) / 3 for tau in sorted(dict(SW[(T11[0], v)]))] for v in vend}
    mspan = {v: 100 * (max(c) - min(c)) for v, c in mean.items()}
    top = max(span, key=span.get)
    need(top[1] == 'I2CVB' and span[top] <= 1.5 and all(s < 0.8 for (t, v), s in span.items() if v != 'I2CVB'),
         'C9: per teacher, a vendor moves by at most 1.5 points (I2CVB) and by less than 0.8 on the other five (the text holds)')
    need((f'{min(mspan.values()):.1f}', f'{max(mspan.values()):.1f}') == ('0.1', '1.0'), 'C9: the mean curve moves by 0.1 to 1.0 points (the Figure 4 caption holds)')
    F['C9'] = {'per_teacher_max': span[top], 'per_teacher_max_at': list(top), 'mean_curve_min': min(mspan.values()), 'mean_curve_max': max(mspan.values()), 'consistent': True}
    wcsv(os.path.join(a.out, 'table_s43b_sweep_spans.csv'), [[t, v, f'{s:.2f}'] for (t, v), s in sorted(span.items())] + [['mean curve', v, f'{s:.2f}'] for v, s in sorted(mspan.items())],
         ['teacher', 'vendor', 'dice_span_points'])
    # ------------------------------------------------ N-82: the BUSI domains of the data loader
    dl = open(os.path.join(a.release, 'CODE', 'source', 'dataloaders', 'dataloader.py'), encoding='utf-8').read()
    blk = dl[dl.index('class BUSISegmentation'):]; blk = blk[:blk.index('\nclass ', 1)]
    dn = ast.literal_eval(re.search(r'self\.domain_name\s*=\s*(\{[^}]*\})', blk).group(1))
    need(dn == {1: 'benign', 2: 'malignant'}, 'N-82: the two BUSI "domains" of the data loader are the benign and malignant subsets (pathology classes, not acquisition sites)')
    F['N82'] = {'busi_domains': dn}
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s43b.json'), 'w'), indent=1, default=float)
    print(f"[s43b] N-75: Table 11 teachers max gap all {wa:.6f}, band {wb:.6f}; exception {S9_EXCEPTION[1]} all {ex[0]:.6f}")
    for r in rows_f:
        print('[s43b] ' + ' | '.join(map(str, r)))
    print(f"[s43b] C9: per-teacher max {span[top]:.2f} at {top}; mean curve {min(mspan.values()):.2f}-{max(mspan.values()):.2f}")
    print(f'[make_extra_s43b] {len(PASSED)} assertions PASSED; wrote {len(os.listdir(a.out))} files to {a.out}')


if __name__ == '__main__':
    main()
