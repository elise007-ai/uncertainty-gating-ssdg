#!/usr/bin/env python3
"""
make_extra_s44.py (S44, 2026-10-28) -- reader for the S44 text layer, assertion-first (R-48); nothing typed by hand (R-14). CPU only, from files
already held: the training logs and harness JSONs under RETURNS, the trainers in CODE/source, and the facts files of S25, S30, S31, S41 and S43b.
No training, no new measurement.
  python make_extra_s44.py --returns <RETURNS> --derived <derived_v04_S12> --code <CODE/source> --out <derived_v04_S12/extra_s44>
Facts:
- E-92: the runs continued from a saved state;
- the training constants of Section 4.2;
- the checkpoint the harness measures (sixth review M2);
- the criterion at the three checkpoints (E-94);
- the softmax head's band ranking (M4);
- the ensembles against the 0.70 line (E-95);
- the D-34 shares and late collapse (B1, B8);
- the composition of the 124 identity checkpoints (E-95).
"""
import argparse, csv, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_extra_s23 import need, PASSED, RULE_BAND   # noqa: E402
import make_derived_v02 as v02                          # noqa: E402

f2 = lambda x: f'{x:.2f}'
CONT = {('prostate', 'v18B_prostate_devis_s1337'): [28000], ('prostate', 'v18B_prostate_medl_s1337'): [20000, 29000],
        ('prostate', 'v18B_prostate_devis_s42'): [12000], ('fundus', 'v18I_fundus_pctmatch_s2024'): [27000]}


def held_logs(rt):
    """(dataset, run) -> the longest held copy of its log.txt; smoke runs excluded"""
    out = {}
    for p in glob.glob(os.path.join(rt, '*', 'runs', '*', '*', 'log.txt')):
        ds, run = p.split(os.sep)[-3], p.split(os.sep)[-2]
        if run.startswith('v18smoke'):
            continue
        if (ds, run) not in out or os.path.getsize(p) > os.path.getsize(out[(ds, run)]):
            out[(ds, run)] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ('returns', 'derived', 'code', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    F = {'session': 'S44', 'reader': 'make_extra_s44.py'}
    J = lambda *p: json.load(open(os.path.join(a.derived, *p), encoding='utf-8'))
    F25, F30, F31, F41, F43 = (J('extra_s25_selection', 'facts_s25.json'), J('extra_s30', 'facts_s30.json'), J('extra_s31', 'facts_s31.json'),
                               J('extra_s41', 'facts_s41.json'), J('extra_s43b', 'facts_s43b.json'))
    L = held_logs(a.returns)
    # ------------------------------------------------ E-92: the runs continued from a saved state
    res = {}
    for k, p in L.items():
        t = open(p, encoding='utf-8', errors='replace').read()
        its = [int(x) for x in re.findall(r'RESUMED from iter (\d+)', t)]
        if its:
            q = v02.parse_log(p); res[k] = {'resumed_from': sorted(set(its)), 'best_iter': q['best_iter'], 'best': q['best_avg']}
    need(set(res) == set(CONT) and all(res[k]['resumed_from'] == CONT[k] for k in CONT),
         f'exactly four held runs were continued from a saved state, from the registered iterations (E-92): {sorted(res)}')
    after = sorted(k[1] for k, v in res.items() if v['best_iter'] > v['resumed_from'][0])
    need(after == ['v18B_prostate_devis_s1337', 'v18B_prostate_medl_s1337'], 'the best checkpoint follows the continuation in two runs: the published-form DEviS '
         'run under its own gate and the MEDL run, both Prostate seed 1337')
    F['continued'] = {f'{d}|{r}': v for (d, r), v in sorted(res.items())}
    s12 = open(os.path.join(a.code, 'v12', 'train_fourierfuse_v18_revision.py'), encoding='utf-8').read()
    need('seg_scheduler = LambdaLR(seg_optimizer, lr_lambda)' in s12 and 'seg_scheduler.load_state_dict' not in s12 and "'scheduler':" not in s12,
         'trainer v12 builds its scheduler at every launch and neither saves nor restores it (E-92)')
    t = open(L[('prostate', 'v18B_prostate_medl_s1337')], encoding='utf-8', errors='replace').read()
    nf = [int(x) for x in re.findall(r'\[iter (\d+)\] non-finite', t)]
    need(bool(nf) and min(nf) > max(CONT[('prostate', 'v18B_prostate_medl_s1337')]), 'MEDL Prostate seed 1337: its first skipped update follows its last continuation')
    Tm = int(re.search(r'max_iterations=(\d+)', t).group(1)); skp = f'{100 * len(set(nf)) / Tm:.0f}'
    need(skp == '42', f'MEDL Prostate seed 1337 is the run that skipped 42 % of its updates ({skp} %; Section 4.4 states 42 and 21)')
    ens, last = [], None                         # the fused (two-network) output is logged before each epoch line
    for line in t.split('\n'):
        m = re.search(r'\[medl\] net2_avg=([\d.]+) ensemble_avg=([\d.]+)', line)
        if m:
            last = float(m.group(2)); continue
        m = re.search(r'\[epoch (\d+)\] student_avg=', line)
        if m and last is not None:
            ens.append(((int(m.group(1)) + 1) * 500, last)); last = None
    bi, bv = max(ens, key=lambda x: x[1])
    need(len(ens) >= 100 and f'{100 * bv:.2f}' == '84.61', f'MEDL Prostate seed 1337: the best fused value of the log is the 84.61 of Table 7 ({100 * bv:.2f} at {bi})')
    need(bi > CONT[('prostate', 'v18B_prostate_medl_s1337')][0], f'its fused best ({bi}) also lies after the first continuation')
    F['medl_fused'] = {'best_iter': bi, 'best': bv}
    m24 = ('prostate', 'v18B_prostate_medl_s2024')
    need(m24 in L and m24 not in res and '] non-finite' in open(L[m24], encoding='utf-8', errors='replace').read(),
         'MEDL seed 2024 was never continued and skipped updates as well')
    import math
    lam = lambda s, T: s / 150 if s < 150 else 0.5 * (1 + math.cos(math.pi * s / T))
    it0, T = CONT[('fundus', 'v18I_fundus_pctmatch_s2024')][0], 30000
    lo_act, hi_int = min(lam(s - it0, T) for s in range(it0 + 150, T + 1)), max(lam(s, T) for s in range(it0, T + 1))
    need(lo_act > 0.97 and hi_int < 0.03, f'the continued D-34 segment ran at {lo_act:.3f} or more of the peak learning rate where the schedule gives {hi_int:.3f} or less')
    F['d34_lr'] = {'actual_min': lo_act, 'intended_max': hi_int, 'medl_first_skip': min(nf), 'medl_skip_pct': skp}
    # ------------------------------------------------ the training constants of Section 4.2, from every held log and from both trainers
    bs, bad, noargs = {}, [], []
    for (ds, run), p in L.items():
        t = open(p, encoding='utf-8', errors='replace').read(60000); m = re.search(r'Namespace\((.*?)\)\s*$', t, re.M)
        if not m:
            noargs.append(run); continue
        A = dict(re.findall(r"(\w+)=('[^']*'|[^,]+)", m.group(1)))
        if (A.get('AdamW'), A.get('base_lr'), A.get('warmup'), A.get('warmup_period')) != ('True', '0.0005', 'True', '150'):
            bad.append(run)
        bs.setdefault(ds, set()).add((A.get('label_bs'), A.get('unlabel_bs')))
    need(not noargs and len(L) >= 120, f'every held training log ({len(L)}, smoke runs excluded) records its arguments (without: {noargs[:3]})')
    need(not bad, f'every held log trains with AdamW, learning rate 5e-4 and a 150-iteration warm-up (exceptions {bad[:3]})')
    F['n_logs'] = len(L)
    need(bs == {'prostate': {('4', '4')}, 'fundus': {('4', '4')}, 'BUSI': {('4', '4')}, 'MNMS': {('2', '2')}},
         f'batches of 4 labeled and 4 unlabeled images, 2 and 2 on M&Ms ({bs})')
    for f in ('train_fourierfuse_v16.py', os.path.join('v12', 'train_fourierfuse_v18_revision.py')):
        s = open(os.path.join(a.code, f), encoding='utf-8').read()
        need('optim.AdamW(seg_params, lr=base_lr, weight_decay=0.01)' in s and 'return 0.5 * (1 + math.cos(math.pi * step / args.max_iterations))' in s
             and 'return step / args.warmup_period' in s, f'{f}: AdamW with weight decay 0.01; linear warm-up, then cosine decay to zero')
    F['train'] = {'weight_decay': 0.01, 'batch': {k: sorted(v)[0] for k, v in bs.items()}}
    # ------------------------------------------------ the checkpoint the harness measures (sixth review M2)
    n = same = 0
    for p in sorted(glob.glob(os.path.join(a.returns, '*', 'diag_out', '*__*__ema.json'))):
        ds, run, _ = os.path.basename(p)[:-5].split('__')
        if run.startswith(('v18smoke', 'ENSEMBLE')) or (ds, run) not in L:
            continue
        me = json.load(open(p)).get('meta', {}); q = v02.parse_log(L[(ds, run)])
        if not q or not q['complete'] or me.get('iter') is None:
            continue
        n += 1; same += int(me['iter']) == q['best_iter'] and os.path.basename(str(me.get('ckpt', ''))) == 'best_student.pth'
    need(n >= 150 and same == n, f'every harness JSON with a complete held log measured best_student.pth at the best iteration ({same} of {n})')
    F['harness'] = {'n': n, 'best_checkpoint': same}
    # ------------------------------------------------ E-94: the criterion at the three checkpoints (oracle minus threshold)
    meets = lambda c: c['p'] < 0.05 and c['delta_points'] >= 0.5
    C = F31['checkpoints']; cell = {(ds, ck): meets(C[f'{ds}_oracle_minus_threshold_{ck}']) for ds in ('prostate', 'fundus') for ck in ('best', 'last', 'swad')}
    need(cell[('prostate', 'best')] and cell[('fundus', 'best')] and [k for k in cell if k[1] != 'best' and cell[k]] == [('fundus', 'last')],
         'the oracle meets the criterion at the best checkpoint on both benchmarks and at one of the four test-free checkpoints (Fundus, last iteration)')
    need(all(C[f'{ds}_oracle_minus_threshold_{ck}']['delta_points'] > 0 for ds in ('prostate', 'fundus') for ck in ('best', 'last', 'swad')), 'positive at every checkpoint')
    # ------------------------------------------------ M4: the softmax head's band ranking; E-95: the ensembles against their members
    sm = F43['families']['softmax head']['conf']
    need(all(sm[d]['n'] == 3 for d in ('prostate', 'fundus')), 'three softmax-head teachers on each of Prostate and Fundus')
    F['softmax_band'] = {d: [f2(sm[d]['band_min']), f2(sm[d]['band_max'])] for d in ('prostate', 'fundus')}
    need(F['softmax_band'] == {'prostate': ['0.71', '0.72'], 'fundus': ['0.70', '0.72']}, f'softmax band AUROC {F["softmax_band"]}')
    B = {}
    for p in sorted(glob.glob(os.path.join(a.derived, '**', 'table_band_summary*.csv'), recursive=True)):
        for r in csv.DictReader(open(p, encoding='utf-8')):
            if r['which'] == 'ema' and r['signal'] == 'conf':
                B[(r['dataset'], r['run'])] = float(r['auroc_edt5_mean'])
    gain = []
    for d, ens, lo, hi in (('prostate', '0.720', '0.691', '0.697'), ('fundus', '0.652', '0.614', '0.626')):
        mem = [v for (dd, r), v in B.items() if dd == d and re.fullmatch(rf'v18B_{d}_fixmatch_split1337_init(1337|43|44)', r)]
        e = B[(d, 'ENSEMBLE3_split1337_fixmatch')]
        need(len(mem) == 3 and (f'{e:.3f}', f'{min(mem):.3f}', f'{max(mem):.3f}') == (ens, lo, hi), f'{d}: ensemble {e:.3f} against members {sorted(mem)} (Section 4.6)')
        gain += [e - max(mem), e - min(mem)]
    F['ensemble_gain'] = [f2(min(gain)), f2(max(gain))]
    need(F['ensemble_gain'] == ['0.02', '0.04'] and min(gain) > 0, 'the ensemble raises the band ranking over its members by 0.02 to 0.04')
    need(B[('prostate', 'ENSEMBLE3_split1337_fixmatch')] > RULE_BAND > B[('fundus', 'ENSEMBLE3_split1337_fixmatch')],
         'the Prostate ensemble passes the 0.70 line of the Phase-A rule and the Fundus one does not (E-95)')
    # ------------------------------------------------ B1, B8: the D-34 gate's achieved share and its late collapse on Prostate
    cov, kept = F41['coverage'], {}
    for ds in ('prostate', 'fundus'):
        lo, hi = cov[ds]['w_after_median']
        need(lo == hi, f'{ds}: the three D-34 runs have the same median kept share after the first fifth ({lo}, {hi})')
        kept[ds] = f'{100 * lo:.1f}'
    need(kept == {'prostate': '99.6', 'fundus': '99.0'}, f'D-34 kept 99.6 % (Prostate) and 99.0 % (Fundus) of the pseudo-label term after the first fifth ({kept})')
    bml, ref = sorted(F41['prostate_descriptive']['best_minus_last_points']), F41['prostate_descriptive']['ref_best_minus_last_points']
    need(sum(19 <= x <= 21 for x in bml) == 2 and bml[0] < 10, f'two of the three D-34 Prostate runs end about 20 points below their best ({bml})')
    need(f'{max(ref):.1f}' == '10.4', 'the threshold references end at most 10.4 points below theirs (Section 4.2 states 10.4)')
    F['d34'] = {'kept_after_fifth': kept, 'prostate_best_minus_last': bml, 'ref_max': f'{max(ref):.1f}'}
    # ------------------------------------------------ M3 / Point 9: the outcome of the FDA rule fixed before the runs (S6)
    rows = list(csv.DictReader(open(os.path.join(a.derived, 'table_fda_sweep.csv'), encoding='utf-8')))
    met = [r['run'] for r in rows if float(r['BIDMC_delta_vs_default']) > 0.015]
    need(len(rows) == 7 and len(met) == 6, f'six of the seven FDA settings meet the registered rule on BIDMC (> +1.5 points over the default run): {len(met)} of {len(rows)}')
    need([r['run'] for r in rows if r['run'] not in met] == ['v18C_prostate_fixmatch_pfda1.0_s1337'], 'the one setting that does not meet it is p_fda = 1')
    F['fda_rule'] = {'n_settings': len(rows), 'n_met': len(met)}
    # ------------------------------------------------ E-95: the 124 identity checkpoints of S9
    rows = {}
    for p in sorted(glob.glob(os.path.join(a.derived, '**', 'table_signal_identity*.csv'), recursive=True)):
        for r in csv.DictReader(open(p, encoding='utf-8')):
            rows.setdefault((r['dataset'], r['run']), r['identity_holds'])
    dm = [k for k in rows if re.search(r'(?i)devis|medl', k[1])]; en = [k for k in rows if k[1].startswith('ENSEMBLE3_')]
    ce = [k for k in rows if k[1] == 'v18A_prostate_daep_act_exp_s1337']
    rest = len(rows) - len(dm) - len(en) - len(ce)
    need((len(rows), len(dm), len(en), len(ce), rest) == (124, 30, 2, 1, 91), f'S9: 124 = 91 testbed objective + 30 DEviS/MEDL + 1 clamped exponential + 2 ensembles ({len(rows)}, {len(dm)}, {rest})')
    need(all(rows[k] == '0' for k in en + ce), 'the clamped-exponential teacher and both ensembles break the identity')
    F['s9'] = {'n': len(rows), 'testbed': rest, 'devis_medl': len(dm), 'clamped': len(ce), 'ensembles': len(en)}
    F['assertions_passed'] = PASSED
    json.dump(F, open(os.path.join(a.out, 'facts_s44.json'), 'w', encoding='utf-8'), indent=1, default=float)
    print(f"[s44] continued: {F['continued']}")
    print(f"[s44] train {F['train']}; harness {F['harness']}; softmax band {F['softmax_band']}; ensemble gain {F['ensemble_gain']}")
    print(f"[s44] D-34 {F['d34']}; FDA rule {F['fda_rule']}; S9 {F['s9']}")
    print(f'[make_extra_s44] {len(PASSED)} assertions PASSED; wrote {a.out}')


if __name__ == '__main__':
    main()
