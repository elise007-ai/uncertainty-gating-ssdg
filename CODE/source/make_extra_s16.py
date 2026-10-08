#!/usr/bin/env python3
"""make_extra_s16.py -- S16 reader (N-39 / D-22, N-37 / E-52, E-53). Analysis side only; no GPU, no checkpoint.
  python make_extra_s16.py --derived <BSPC_results/derived/derived_v04_S12> --release <release root> [--out <dir>]
Reads  <derived>/table_band_generality.csv, table_oracle_ceiling.csv, table_threshold_sweep.csv, table_runs.csv,
       <release>/REFERENCE/synfoc_published_tables.csv, <release>/RESULTS/{adaptive_threshold,seed_split_migration}/*.txt
Writes <derived>/extra_s16/ : table_collapse_prostate_s16.csv, table_collapse_cross_s16.csv, table_oracle_band_s16.csv,
       table_threshold_sweep_s16.csv, table_synfoc_like_for_like_s16.csv, facts_s16.json
D-22: the headline measurements of 4.5-4.7 (Tables collapse, collapse-cross, band-auroc; Figures 3-5; oracle band; threshold
sweep) are re-derived on SEED-SPLIT teachers: the three threshold-gate teachers of the reference row of Table baselines
(v18B, seeds 1337 / 42 / 2024) and, for reference, the DAEP default teacher (v18A, seed 1337). Every statement the S16 text
makes about them is an assertion below (R-48); the thresholds were written after the values had been printed once.
N-37: the published SynFoC table is read by LABELLED domain (its Section 4.2); the arithmetic 'Avg. = mean of the columns'
is asserted for every dataset x method, and the column of OUR labelled domain is the like-for-like reference.
"""
import argparse, json, os, re, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd

VEND = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
TH = ['v18B_prostate_fixmatch_split1337_init1337', 'v18B_prostate_fixmatch_s42', 'v18B_prostate_fixmatch_s2024']
DA = 'v18A_prostate_daep_default_s1337'
INITS = ['v18B_prostate_fixmatch_split1337_init1337', 'v18B_prostate_fixmatch_split1337_init43', 'v18B_prostate_fixmatch_split1337_init44']
CROSS = {'prostate': TH,
         'fundus': ['v18B_fundus_fixmatch_split1337_init1337', 'v18B_fundus_fixmatch_s42', 'v18B_fundus_fixmatch_s2024'],
         'MNMS': ['v18B_MNMS_fixmatch_s1337', 'v18B_MNMS_fixmatch_s42', 'v18B_MNMS_fixmatch_s2024'],
         'BUSI': ['v18B_BUSI_fixmatch_s1337', 'v18B_BUSI_fixmatch_s42', 'v18B_BUSI_fixmatch_s2024']}
CROSS_DAEP = {'MNMS': ['adaptive_cmp_mnms_daep_s1337', 'adaptive_cmp_mnms_daep_s42', 'adaptive_cmp_mnms_daep_s2024'],
              'BUSI': ['adaptive_cmp_busi_daep_s1337', 'adaptive_cmp_busi_daep_s42', 'adaptive_cmp_busi_daep_s2024']}
OUR_LB = {'prostate': 'RUNMC', 'fundus': 'Domain 3', 'MNMS': 'Vendor D', 'BUSI': 'Malignant'}
NDOM = {'prostate': 6, 'fundus': 4, 'MNMS': 4, 'BUSI': 2}
# Table main, testbed rows as printed in the manuscript (seed-split, S01 logs / digests), mean and sd in Dice points
MAIN_PRINTED = {'prostate': {'daep': (85.38, 0.58), 'thr': (85.71, 0.57)}, 'fundus': {'daep': (88.54, 0.17), 'thr': (88.47, 0.23)},
                'MNMS': {'daep': (80.75, 0.60), 'thr': (80.84, 0.15)}, 'BUSI': {'daep': (70.93, 1.25), 'thr': (70.19, 1.29)}}


def r2(x):
    """round half up on the decimal representation (89.535 -> 89.54), so that a class mean never depends on binary floats"""
    return float(Decimal(str(round(float(x), 6))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def need(cond, msg):
    if not cond:
        raise AssertionError('[make_extra_s16] ' + msg)


def synfoc(release):
    p = os.path.join(release, 'REFERENCE', 'synfoc_published_tables.csv'); d = pd.read_csv(p)
    need(len(d) == 48, f'{p}: expected 48 rows, got {len(d)}')
    d['value'] = [sum(Decimal(x) for x in str(c).split('/')) / len(str(c).split('/')) for c in d.class_values]
    out, n = {}, 0
    for (ds, m), g in d.groupby(['dataset', 'method']):
        need(len(g) == NDOM[ds], f'{ds} {m}: {len(g)} labelled-domain columns')
        mean = float(sum(g.value) / len(g)); avg = float(g.published_avg_dsc.iloc[0])
        need(abs(mean - avg) <= 0.011, f'{ds} {m}: mean of the labelled-domain columns {mean:.4f} != published Avg. {avg}')   # the reading rule
        n += 1
        row = g[g.labelled_domain == OUR_LB[ds]]; need(len(row) == 1, f'{ds} {m}: our labelled domain {OUR_LB[ds]}')
        out.setdefault(ds, {})[m] = {'like': r2(row.value.iloc[0]), 'avg_over_labelled_domains': avg,
                                     'columns': {k: r2(v) for k, v in zip(g.labelled_domain, g.value)}}
    need(n == 12, 'expected 12 dataset x method groups')
    return out


def digests(release, runs):
    """per-seed best-average Dice of the runs behind Table main, from the submission digests (threshold: all four datasets;
    DAEP: prostate, fundus) and from the held logs via table_runs (DAEP: MNMS, BUSI). Returns values in [0, 1]."""
    R = os.path.join(release, 'RESULTS'); out = {}
    pro = open(os.path.join(R, 'adaptive_threshold', 'prostate_logs.txt')).read()
    pv = {}
    for gate, seed, avg, dom in re.findall(r'^(fixmatch|daep)_s(\d+):\s+avg=([\d.]+)\s+per-domain\(d1-6\)=([\d.,]+)', pro, flags=re.M):
        pv[(gate, int(seed))] = (float(avg), [float(x) for x in dom.split(',')])
    need(len(pv) == 6 and all(len(v[1]) == 6 for v in pv.values()), f'prostate digest: {len(pv)} gate x seed lines')
    out['prostate'] = {'thr': [pv[('fixmatch', s)][0] for s in (1337, 42, 2024)], 'daep': [pv[('daep', s)][0] for s in (1337, 42, 2024)]}
    fun = open(os.path.join(R, 'adaptive_threshold', 'fundus_logs.txt')).read()
    fv = {(g, int(s)): float(a) for g, s, a in re.findall(r'^(fixmatch|daep)_s(\d+):\s+avg=([\d.]+)', fun, flags=re.M)}
    need(len(fv) == 6, 'fundus digest'); out['fundus'] = {'thr': [fv[('fixmatch', s)] for s in (1337, 42, 2024)], 'daep': [fv[('daep', s)] for s in (1337, 42, 2024)]}
    mb = open(os.path.join(R, 'seed_split_migration', 'mnms_busi_threshold_logs.txt')).read()
    tv = {(d, int(s)): float(a) for d, s, a in re.findall(r'(mnms|busi)_fixmatch_s(\d+):\s*([\d.]+)', mb)}
    need(len(tv) == 6, 'mnms/busi threshold digest')
    for ds, key in (('MNMS', 'mnms'), ('BUSI', 'busi')):
        logs = [float(runs.set_index('run').best_avg[f'adaptive_cmp_{key}_daep_s{s}']) for s in (1337, 42, 2024)]
        out[ds] = {'thr': [tv[(key, s)] for s in (1337, 42, 2024)], 'daep_held_logs': logs}
    return out, pv


def build(derived, release, out):
    os.makedirs(out, exist_ok=True); F = {}
    g = pd.read_csv(os.path.join(derived, 'table_band_generality.csv')); o = pd.read_csv(os.path.join(derived, 'table_oracle_ceiling.csv'))
    t = pd.read_csv(os.path.join(derived, 'table_threshold_sweep.csv')); runs = pd.read_csv(os.path.join(derived, 'table_runs.csv'))
    u = g[g.signal == 'u_epi']; c = g[g.signal == 'conf']
    for r in TH + [DA] + INITS + sum(CROSS.values(), []) + sum(CROSS_DAEP.values(), []):
        need((u.run == r).sum() >= 2, f'no harness rows for {r}')
    F['teachers'] = {'threshold': TH, 'daep': DA, 'inits': INITS}
    # ---------------------------------------------------------------- 1. Prostate per vendor (Tables collapse, band-auroc; Figure 3)
    ut = u[u.run.isin(TH)]; ud = u[u.run == DA].set_index('domain_name')
    need(len(ut) == 18 and len(ud) == 6 and sorted(ud.index) == VEND, 'prostate rows')
    rows, col = [], {}
    for v in VEND:
        x = ut[ut.domain_name == v]; need(len(x) == 3, v)
        er, fe, fp = x.error_rate.mean(), x.frac_errors_in_edt5.mean(), x.frac_pixels_in_edt5.mean()
        col[v] = dict(thr_all_m=x.auroc_all.mean(), thr_all_sd=x.auroc_all.std(), thr_band_m=x.auroc_edt5.mean(), thr_band_sd=x.auroc_edt5.std(),
                      thr_all_min=x.auroc_all.min(), thr_all_max=x.auroc_all.max(), thr_band_min=x.auroc_edt5.min(), thr_band_max=x.auroc_edt5.max(),
                      thr_drop=x.auroc_all.mean() - x.auroc_edt5.mean(), thr_1px_m=x.auroc_edt1.mean(), thr_range_m=x.range80_all.mean(),
                      thr_range_max=x.range80_all.max(), thr_band_range_m=x.range80_edt5.mean(), daep_all=ud.auroc_all[v], daep_band=ud.auroc_edt5[v],
                      daep_range=ud.range80_all[v], frac_errors_in_band=fe, frac_pixels_in_band=fp, err_rate_in_band=x.error_rate_in_edt5.mean(),
                      err_rate_outside_band=float((x.error_rate * (1 - x.frac_errors_in_edt5) / (1 - x.frac_pixels_in_edt5)).mean()))
        need((x.auroc_edt5 < x.auroc_all).all() and ud.auroc_edt5[v] < ud.auroc_all[v], f'{v}: band AUROC not below all-pixel AUROC')
        rows.append(dict(vendor=v, **col[v]))
    T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(out, 'table_collapse_prostate_s16.csv'), index=False, float_format='%.6g')
    mean = {k: float(T1[k].mean()) for k in T1.columns if k != 'vendor'}
    per_t = ut.groupby('run')[['auroc_all', 'auroc_edt5', 'range80_all']].mean().loc[TH]
    cg = c[c.run.isin(TH + [DA])].set_index(['run', 'domain_name']); ug = u[u.run.isin(TH + [DA])].set_index(['run', 'domain_name'])
    F['collapse'] = {'vendors': {v: {k: float(x) for k, x in col[v].items()} for v in VEND}, 'mean': mean,
                     'thr_all_per_teacher': [float(x) for x in per_t.auroc_all], 'thr_band_per_teacher': [float(x) for x in per_t.auroc_edt5],
                     'all_vendor_min': float(T1.thr_all_m.min()), 'all_vendor_max': float(T1.thr_all_m.max()),
                     'band_vendor_min': float(T1.thr_band_m.min()), 'band_vendor_max': float(T1.thr_band_m.max()),
                     'drop_min': float(T1.thr_drop.min()), 'drop_max': float(T1.thr_drop.max()),
                     'band_sd_min': float(T1.thr_band_sd.min()), 'band_sd_max': float(T1.thr_band_sd.max()), 'all_sd_max': float(T1.thr_all_sd.max()),
                     'range_max_any': float(max(ut.range80_all.max(), ud.range80_all.max())), 'range_min_any': float(min(ut.range80_all.min(), ud.range80_all.min())),
                     'band_range_max': float(T1.thr_band_range_m.max()),
                     'fe_min': float(T1.frac_errors_in_band.min()), 'fe_max': float(T1.frac_errors_in_band.max()),
                     'err_out_min': float(T1.err_rate_outside_band.min()), 'err_out_max': float(T1.err_rate_outside_band.max()),
                     'identity_gap_all': float((cg.auroc_all - ug.auroc_all).abs().max()), 'identity_gap_band': float((cg.auroc_edt5 - ug.auroc_edt5).abs().max())}
    C = F['collapse']
    need(0.865 <= mean['thr_all_m'] <= 0.875 and 0.67 <= mean['thr_band_m'] <= 0.68, 'headline means moved (0.87 / 0.68)')
    need(0.875 <= mean['daep_all'] <= 0.885 and 0.675 <= mean['daep_band'] <= 0.685, 'DAEP teacher means moved (0.88 / 0.68)')
    need(0.81 <= C['all_vendor_min'] and C['all_vendor_max'] <= 0.915, 'vendor all-pixel means outside 0.82-0.91')
    need(0.645 <= C['band_vendor_min'] and C['band_vendor_max'] <= 0.705, 'vendor band means outside 0.65-0.70')
    need(0.165 <= C['drop_min'] and C['drop_max'] <= 0.215, 'drop outside 0.17-0.21')
    need(0.555 <= mean['thr_1px_m'] <= 0.575, '1-px band mean moved (0.57)')
    need(2e-5 <= C['range_min_any'] and C['range_max_any'] <= 1.5e-4, 'range not of order 1e-4 on every vendor and teacher')
    need(C['band_range_max'] <= 5e-3, 'in-band range above 5e-3')
    need(C['identity_gap_all'] <= 1.5e-4 and C['identity_gap_band'] <= 1.05e-3, 'confidence and vacuity differ by more than 1e-3')
    need(0.015 <= C['band_sd_min'] and C['band_sd_max'] <= 0.035 and 0.11 <= C['all_sd_max'] <= 0.125, 'seed sd statement (band 0.02-0.03; all-pixel up to 0.12)')
    need(0.60 <= C['fe_min'] and C['fe_max'] <= 0.86 and 0.72 <= mean['frac_errors_in_band'] <= 0.75, 'share of errors in the band (61-85 %, mean 74 %)')
    need(0.017 <= T1.frac_pixels_in_band.min() and T1.frac_pixels_in_band.max() <= 0.026, 'band is not 2 % of the pixels')
    need(0.0005 <= C['err_out_min'] and C['err_out_max'] <= 0.004, 'error rate outside the band not 0.1-0.4 %')
    # three initialisations on one labelled split: the all-pixel value moves, the band value does not
    ui = u[u.run.isin(INITS)].groupby('run')[['auroc_all', 'auroc_edt5']].mean().loc[INITS]
    F['inits'] = {'all': [float(x) for x in ui.auroc_all], 'band': [float(x) for x in ui.auroc_edt5]}
    need(ui.auroc_all.max() - ui.auroc_all.min() >= 0.09 and ui.auroc_edt5.max() - ui.auroc_edt5.min() <= 0.01, 'init spread statement')
    # ---------------------------------------------------------------- 2. cross-dataset (Table collapse-cross)
    rows, F['cross'] = [], {}
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        for gate, rr in (('threshold', CROSS[ds]), ('daep', CROSS_DAEP.get(ds))):
            if not rr: continue
            x = u[u.run.isin(rr)].groupby('run')[['auroc_all', 'auroc_edt5', 'range80_all', 'frac_errors_in_edt5']].mean()
            need(len(x) == 3 and (u[u.run.isin(rr)].groupby('run').size() == NDOM[ds]).all(), f'{ds} {gate}: teachers / domains')
            d = dict(dataset=ds, gate=gate, n_teachers=3, n_domains=NDOM[ds], auroc_all_m=x.auroc_all.mean(), auroc_all_min=x.auroc_all.min(), auroc_all_max=x.auroc_all.max(),
                     band_m=x.auroc_edt5.mean(), band_min=x.auroc_edt5.min(), band_max=x.auroc_edt5.max(), range_m=x.range80_all.mean(), range_max=x.range80_all.max(),
                     frac_errors_in_band=x.frac_errors_in_edt5.mean())
            rows.append(d); F['cross'].setdefault(ds, {})[gate] = {k: (float(v) if not isinstance(v, (str, int)) else v) for k, v in d.items()}
            need(d['auroc_all_min'] > 0.70 and d['band_m'] < d['auroc_all_m'] and d['range_max'] < 2e-3, f'{ds} {gate}: dissociation statement')
    pd.DataFrame(rows).to_csv(os.path.join(out, 'table_collapse_cross_s16.csv'), index=False, float_format='%.6g')
    X = {ds: F['cross'][ds]['threshold'] for ds in F['cross']}
    need(min(X, key=lambda k: X[k]['auroc_all_m']) == 'BUSI' and max(X, key=lambda k: X[k]['auroc_all_m']) == 'MNMS', 'BUSI lowest / M&Ms highest')
    need(7e-5 <= min(v['range_m'] for v in X.values()) and max(v['range_m'] for v in X.values()) <= 1.05e-3, 'cross range 8e-5 .. 1e-3')
    need(X['BUSI']['frac_errors_in_band'] < 0.35 and min(X[k]['frac_errors_in_band'] for k in ('prostate', 'fundus', 'MNMS')) > 0.70, 'BUSI is the benchmark whose errors are not at the boundary')
    # ---------------------------------------------------------------- 3. oracle band (Figure 5)
    ot = o[o.run.isin(TH)]; need(len(ot) == 18, 'oracle rows')
    T3 = ot.groupby('domain_name').agg(asis_m=('dice_asis', 'mean'), asis_sd=('dice_asis', 'std'), band5_m=('dice_band_5', 'mean'), band5_sd=('dice_band_5', 'std'),
                                       gain=('gain_band5', 'mean')).loc[VEND].reset_index().rename(columns={'domain_name': 'vendor'})
    T3.to_csv(os.path.join(out, 'table_oracle_band_s16.csv'), index=False, float_format='%.6g')
    pt = ot.groupby('run').gain_band5.mean().loc[TH]
    F['oracle'] = {'vendors': {r.vendor: {k: float(getattr(r, k)) for k in ('asis_m', 'asis_sd', 'band5_m', 'band5_sd', 'gain')} for r in T3.itertuples()},
                   'gain_mean': float(pt.mean()), 'gain_vendor_min': float(T3.gain.min()), 'gain_vendor_max': float(T3.gain.max()), 'gain_per_teacher': [float(x) for x in pt]}
    need(0.093 <= F['oracle']['gain_mean'] <= 0.095 and 0.084 <= F['oracle']['gain_vendor_min'] and F['oracle']['gain_vendor_max'] <= 0.103, 'oracle gain (+8.5 .. +10.2, mean +9.4)')
    need(T3.set_index('vendor').gain.idxmax() == 'BIDMC' or abs(T3.gain.max() - T3.set_index('vendor').gain['BIDMC']) < 1e-3, 'largest oracle gain is not on BIDMC')
    # ---------------------------------------------------------------- 4. threshold sweep (Figure 4)
    tt = t[t.run.isin(TH)]; need(len(tt) == 3 * 6 * 19, f'sweep rows {len(tt)}')
    T4 = tt.groupby(['domain_name', 'tau']).dice.mean().reset_index().rename(columns={'domain_name': 'vendor', 'dice': 'dice_mean'})
    T4.to_csv(os.path.join(out, 'table_threshold_sweep_s16.csv'), index=False, float_format='%.6g')
    span_t = tt.groupby(['run', 'domain_name']).dice.agg(lambda x: x.max() - x.min()).unstack()
    span_m = T4.groupby('vendor').dice_mean.agg(lambda x: x.max() - x.min())
    best = tt.groupby(['run', 'domain_name']).dice.max().unstack(); at5 = tt[np.isclose(tt.tau, 0.5)].set_index(['run', 'domain_name']).dice.unstack()
    rec = (best - at5) / ot.set_index(['run', 'domain_name']).gain_band5.unstack()
    F['sweep'] = {'span_mean_curve': {v: float(span_m[v]) for v in VEND}, 'span_teacher_max': {v: float(span_t[v].max()) for v in VEND},
                  'span_max_any': float(span_t.values.max()), 'span_max_vendor': str(span_t.max().idxmax()),
                  'span_max_other_five': float(span_t.drop(columns=[str(span_t.max().idxmax())]).values.max()),
                  'oracle_threshold_recovery_mean': float(rec.values.mean()), 'oracle_threshold_recovery_min': float(rec.values.min()), 'oracle_threshold_recovery_max': float(rec.values.max())}
    S = F['sweep']
    need(S['span_max_vendor'] == 'I2CVB' and 0.014 <= S['span_max_any'] <= 0.015 and S['span_max_other_five'] < 0.008, 'sweep span statement (<= 1.5 pts on I2CVB, < 0.8 elsewhere)')
    need(0.0009 <= min(S['span_mean_curve'].values()) and max(S['span_mean_curve'].values()) <= 0.0111, 'mean-curve span 0.1-1.1 pts')
    need(S['oracle_threshold_recovery_min'] >= 0 and S['oracle_threshold_recovery_max'] <= 0.08 and 0.015 <= S['oracle_threshold_recovery_mean'] <= 0.025, 'oracle threshold recovers 0-8 % (mean 2 %)')
    # ---------------------------------------------------------------- 5. SynFoC, like for like (Table main)  -- N-37 / E-52
    SF = synfoc(release); dg, pv = digests(release, runs); rows = []; F['synfoc'] = {'by_dataset': {}, 'our_labelled_domain': OUR_LB}
    for ds in ('prostate', 'fundus', 'MNMS', 'BUSI'):
        thr = np.array(dg[ds]['thr']) * 100; m, s = MAIN_PRINTED[ds]['thr']
        need(abs(thr.mean() - m) < 0.006 and abs(thr.std(ddof=1) - s) < 0.006, f'{ds}: threshold row of Table main not reproduced from the digest ({thr.mean():.3f} +- {thr.std(ddof=1):.3f})')
        if 'daep' in dg[ds]:
            da = np.array(dg[ds]['daep']) * 100; m, s = MAIN_PRINTED[ds]['daep']
            need(abs(da.mean() - m) < 0.006 and abs(da.std(ddof=1) - s) < 0.006, f'{ds}: DAEP row of Table main not reproduced from the digest')
        d = {'like_unet': SF[ds]['SynFoC-UNet']['like'], 'like_medsam': SF[ds]['SynFoC-MedSAM']['like'], 'like_fixmatch': SF[ds]['FixMatch']['like'],
             'avg_unet': SF[ds]['SynFoC-UNet']['avg_over_labelled_domains'], 'avg_medsam': SF[ds]['SynFoC-MedSAM']['avg_over_labelled_domains'],
             'testbed_thr': MAIN_PRINTED[ds]['thr'][0], 'testbed_daep': MAIN_PRINTED[ds]['daep'][0]}
        d.update(thr_minus_unet=r2(d['testbed_thr'] - d['like_unet']), daep_minus_unet=r2(d['testbed_daep'] - d['like_unet']), thr_minus_fixmatch=r2(d['testbed_thr'] - d['like_fixmatch']),
                 thr_minus_unet_as_submitted=r2(d['testbed_thr'] - d['avg_unet']))      # two decimals, decimal arithmetic: M&Ms is exactly +0.25
        F['synfoc']['by_dataset'][ds] = {k: float(v) for k, v in d.items()}; rows.append(dict(dataset=ds, labelled_domain=OUR_LB[ds], **d))
    pd.DataFrame(rows).to_csv(os.path.join(out, 'table_synfoc_like_for_like_s16.csv'), index=False, float_format='%.6g')
    B = F['synfoc']['by_dataset']
    F['synfoc']['mean4'] = {k: r2(np.mean([B[ds][k] for ds in B])) for k in ('like_unet', 'like_medsam', 'testbed_thr', 'testbed_daep', 'avg_unet')}
    need([B[ds]['thr_minus_unet'] for ds in ('prostate', 'fundus', 'MNMS', 'BUSI')] == [-2.38, -0.72, 0.25, -2.5], 'threshold minus SynFoC-UNet, like for like')
    need([B[ds]['thr_minus_fixmatch'] for ds in ('prostate', 'fundus', 'MNMS', 'BUSI')] == [4.02, 1.79, -1.26, 0.39], 'threshold minus published FixMatch')
    need(max(abs(B[ds][k]) for ds in B for k in ('thr_minus_unet', 'daep_minus_unet')) < 3.0, 'within three Dice points of SynFoC-UNet on every benchmark')
    need(F['synfoc']['mean4'] == {'like_unet': 82.64, 'like_medsam': 83.33, 'testbed_thr': 81.30, 'testbed_daep': 81.40, 'avg_unet': 80.02}, f"four-benchmark means {F['synfoc']['mean4']}")
    # E-53: the M&Ms / BUSI DAEP rows of Table main are NOT the means of the held logs that carry those run names (recorded, not resolved)
    F['main_table_vs_held_logs'] = {}
    for ds in ('MNMS', 'BUSI'):
        lg = np.array(dg[ds]['daep_held_logs']) * 100; thr = np.array(dg[ds]['thr']) * 100; dd = lg - thr
        tstat = dd.mean() / (dd.std(ddof=1) / np.sqrt(3))
        F['main_table_vs_held_logs'][ds] = {'printed_mean': MAIN_PRINTED[ds]['daep'][0], 'printed_sd': MAIN_PRINTED[ds]['daep'][1], 'held_logs_mean': float(lg.mean()),
                                            'held_logs_sd': float(lg.std(ddof=1)), 'held_logs_minus_threshold': float(dd.mean()), 'paired_t': float(tstat)}
    M = F['main_table_vs_held_logs']
    need(abs(M['MNMS']['held_logs_mean'] - 80.85) < 0.01 and abs(M['BUSI']['held_logs_mean'] - 70.83) < 0.01, 'E-53 record changed: re-adjudicate')
    # ---------------------------------------------------------------- 6. per-vendor Dice, the runs of Table main (digest; E-49 caveat stands)
    lowest = all(int(np.argmin(v[1])) == 0 for v in pv.values())     # d1 = BIDMC
    gm = {gate: np.mean([pv[(gate, s)][1] for s in (1337, 42, 2024)], axis=0) * 100 for gate in ('fixmatch', 'daep')}
    F['per_vendor'] = {'bidmc_lowest_in_all_six_runs': bool(lowest), 'second_lowest': {k: VEND[int(np.argsort(v)[1])] for k, v in gm.items()},
                       'gap_bidmc_to_next': {k: float(np.sort(v)[1] - np.sort(v)[0]) for k, v in gm.items()},
                       'other_four_min': float(min(np.delete(v, [0, 4]).min() for v in gm.values())), 'other_four_max': float(max(np.delete(v, [0, 4]).max() for v in gm.values())),
                       'runmc': {k: float(v[4]) for k, v in gm.items()}, 'bidmc': {k: float(v[0]) for k, v in gm.items()}}
    P = F['per_vendor']
    need(lowest and set(P['second_lowest'].values()) == {'RUNMC'} and all(2.0 <= x <= 2.3 for x in P['gap_bidmc_to_next'].values()), 'BIDMC lowest in every run; RUNMC next, about two points above')
    need(85.45 <= P['other_four_min'] and P['other_four_max'] <= 89.35, 'other four vendors 85.5-89.3')
    json.dump(F, open(os.path.join(out, 'facts_s16.json'), 'w'), indent=1, sort_keys=True)
    return F


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--derived', required=True); ap.add_argument('--release', required=True); ap.add_argument('--out', default='')
    a = ap.parse_args(); out = a.out or os.path.join(a.derived, 'extra_s16')
    F = build(a.derived, a.release, out); C = F['collapse']; B = F['synfoc']['by_dataset']
    print(f"[collapse] threshold x3: all {C['mean']['thr_all_m']:.3f} (vendors {C['all_vendor_min']:.2f}-{C['all_vendor_max']:.2f}), band {C['mean']['thr_band_m']:.3f} "
          f"({C['band_vendor_min']:.2f}-{C['band_vendor_max']:.2f}), 1 px {C['mean']['thr_1px_m']:.3f}, range max {C['range_max_any']:.2e}; DAEP {C['mean']['daep_all']:.3f} / {C['mean']['daep_band']:.3f}")
    print(f"[cross] " + '; '.join(f"{ds} {v['threshold']['auroc_all_m']:.3f} / {v['threshold']['band_m']:.3f} / {v['threshold']['range_m']:.1e} / err-in-band {v['threshold']['frac_errors_in_band']:.2f}" for ds, v in F['cross'].items()))
    print(f"[oracle] gain mean {F['oracle']['gain_mean']*100:.2f} (vendors {F['oracle']['gain_vendor_min']*100:.1f}-{F['oracle']['gain_vendor_max']*100:.1f}); "
          f"[sweep] max span {F['sweep']['span_max_any']*100:.2f} ({F['sweep']['span_max_vendor']}), others < {F['sweep']['span_max_other_five']*100:.2f}; oracle threshold recovers {F['sweep']['oracle_threshold_recovery_mean']*100:.1f} %")
    print('[synfoc] like for like (UNet / MedSAM / FixMatch | as submitted): ' + '; '.join(f"{ds} {v['like_unet']} / {v['like_medsam']} / {v['like_fixmatch']} | {v['avg_unet']}" for ds, v in B.items()))
    print('[synfoc] threshold - SynFoC-UNet: ' + ', '.join(f"{ds} {v['thr_minus_unet']:+.2f}" for ds, v in B.items()) + f"; four-benchmark means {F['synfoc']['mean4']}")
    print(f"[E-53] Table main DAEP rows vs held logs: " + '; '.join(f"{ds} printed {v['printed_mean']} +- {v['printed_sd']}, held logs {v['held_logs_mean']:.2f} +- {v['held_logs_sd']:.2f}" for ds, v in F['main_table_vs_held_logs'].items()))
    print(f'[done] {out}: 5 tables + facts_s16.json; all assertions passed')
