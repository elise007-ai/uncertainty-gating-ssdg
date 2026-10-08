#!/usr/bin/env python3
"""make_extra_s15.py -- S15 readers.

  python make_extra_s15.py --derived <BSPC_results/derived/derived_v04_S12> --returns <BSPC_results/RETURNS>
                           --release_results <release>/RESULTS [--out <dir>]

A. Spectral robustness and forensic check (V-13 / N-36). The script behind the submitted "-1.19 sigma, inside the
   hull" is not available (owner, S15). From the per-slice radial profiles of the reproducible analysis
   (RETURNS/S06/spectral_out/slice_profiles.npz) this reader (i) recomputes the leave-one-vendor-out sliced
   Wasserstein and Frechet distances under six feature variants (raw; lowest bin dropped; per-slice centred;
   per-slice standardised = spectral shape only; only the bins above the FDA band; standardised without the
   lowest bin) and (ii) evaluates 21 z-score conventions (7 low-frequency scalars x 3 denominators) to see whether
   any natural convention gives the submitted value.
B. Seed-split per-vendor Prostate table (D-21 / N-34) and the per-vendor DAEP-minus-threshold differences (D-19),
   from RESULTS/adaptive_threshold/prostate_logs.txt: the digests of the six seed-split runs behind Table main
   (85.71 / 85.38), the same file make_derived_v01/v02 take the registered references from.
C. Band-definition sensitivity (R1 #7; N-15): error-ranking AUROC of the teacher's signal for Euclidean bands of
   1-12 px, a Chebyshev band, bands around the predicted / union boundary, object-size-relative bands and
   millimetre bands, for the seed-split DAEP default teacher and the three seed-split threshold teachers.
Writes <derived>/extra_s15/: table_spectral_robustness_s15.csv, table_spectral_zconventions_s15.csv,
table_prostate_per_vendor_seedsplit_s15.csv, table_band_definitions_s15.csv, facts_s15.json, console log.
Every qualitative statement of the S15 text is an assertion here (R-48).
"""
import argparse, csv, json, math, os, re, statistics as st, sys, glob
import numpy as np

VENDORS = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)


def rd(path):
    return list(csv.DictReader(open(path, newline='')))


def latest(returns, rel):
    m = sorted(glob.glob(os.path.join(returns, 'S*', rel))); assert m, f'not in any return: {rel}'; return m[-1]


def wcsv(path, rows):
    w = csv.DictWriter(open(path, 'w', newline=''), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def t_p_two_sided_df2(d):
    """Paired t over three seeds; two-sided p for 2 df in closed form (as in S13)."""
    m = st.mean(d); s = st.stdev(d)
    if s == 0: return float('inf'), 0.0
    t = m / (s / math.sqrt(len(d))); return t, 1 - abs(t) / math.sqrt(2 + t * t)


# ---------------------------------------------------------------------------------------------- A
def spectral_robustness(R, F, out):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from spectral_distance_analysis import sliced_wasserstein, frechet_distance, in_convex_hull_2d
    z = np.load(latest(R, 'spectral_out/slice_profiles.npz'), allow_pickle=True)
    P = {s: z[s].astype(np.float64) for s in VENDORS}
    assert sum(len(P[s]) for s in VENDORS) == 1867 and all(P[s].shape[1] == 64 for s in VENDORS)
    J = json.load(open(latest(R, 'spectral_out/spectral_distances.json'))); low = int(J['meta']['lowfreq_bins']); assert low == 6
    swd = lambda a, b: sliced_wasserstein(a, b, rs=np.random.RandomState(1))
    ctr = lambda X: X - X.mean(1, keepdims=True)
    std = lambda X: ctr(X) / X.std(1, keepdims=True)
    variants = [('raw', 'as in Table tab:spectral', lambda X: X),
                ('drop_bin0', 'lowest radial bin (DC and its neighbours) removed', lambda X: X[:, 1:]),
                ('centred', 'each profile minus its own mean', ctr),
                ('standardised', 'each profile centred and scaled to unit variance: spectral shape only', std),
                ('above_fda_band', f'bins {low}-63 only, i.e. above the band FDA swaps at beta = 0.05', lambda X: X[:, low:]),
                ('standardised_no_bin0', 'shape only, lowest bin removed first', lambda X: std(X[:, 1:]))]
    rows = []; F['spec_rob'] = {'variants': {}}
    for key, desc, fn in variants:
        Pv = {s: fn(P[s]) for s in VENDORS}; a = {}; b = {}
        for s in VENDORS:
            Xo = np.concatenate([Pv[o] for o in VENDORS if o != s]); a[s] = swd(Pv[s], Xo); b[s] = frechet_distance(Pv[s], Xo)
        sm = np.stack([Pv[s].mean(0) for s in VENDORS]); mu = sm.mean(0); _, Sg, Vt = np.linalg.svd(sm - mu, full_matrices=False); pc = (sm - mu) @ Vt[:2].T
        hull = bool(in_convex_hull_2d(np.delete(pc, 0, axis=0), pc[0]))
        ra = sorted(VENDORS, key=lambda s: -a[s]); rb = sorted(VENDORS, key=lambda s: -b[s])
        f = {'swd_bidmc': a['BIDMC'], 'swd_next': a[ra[1]], 'swd_next_vendor': ra[1], 'swd_ratio': a['BIDMC'] / a[ra[1]], 'swd_rank_bidmc': ra.index('BIDMC') + 1,
             'fd_bidmc': b['BIDMC'], 'fd_next': b[rb[1]], 'fd_next_vendor': rb[1], 'fd_ratio': b['BIDMC'] / b[rb[1]], 'fd_rank_bidmc': rb.index('BIDMC') + 1,
             'bidmc_inside_hull': hull, 'pc1_share': float(Sg[0] ** 2 / (Sg ** 2).sum()), 'pc12_share': float((Sg[:2] ** 2).sum() / (Sg ** 2).sum())}
        assert f['swd_rank_bidmc'] == 1 and f['fd_rank_bidmc'] == 1 and not hull, (key, f)       # the statement of the text
        F['spec_rob']['variants'][key] = f
        rows.append({'variant': key, 'description': desc, 'swd_BIDMC': f"{a['BIDMC']:.4f}", 'swd_next_vendor': ra[1], 'swd_next': f"{a[ra[1]]:.4f}",
                     'swd_ratio': f"{f['swd_ratio']:.2f}", 'frechet_BIDMC': f"{b['BIDMC']:.4f}", 'frechet_next_vendor': rb[1], 'frechet_next': f"{b[rb[1]]:.4f}",
                     'frechet_ratio': f"{f['fd_ratio']:.2f}", 'rank_BIDMC_swd': 1, 'rank_BIDMC_frechet': 1, 'BIDMC_inside_hull': int(hull),
                     'pc1_share': f"{f['pc1_share']:.3f}", 'pc1_pc2_share': f"{f['pc12_share']:.3f}"})
    V = F['spec_rob']['variants']
    assert abs(V['raw']['swd_bidmc'] - J['leave_one_out']['BIDMC']['swd_vs_others']['point']) < 1e-9          # the raw variant IS the published analysis
    assert abs(V['raw']['fd_bidmc'] - J['leave_one_out']['BIDMC']['frechet_vs_others']['point']) < 2e-3   # float32 profiles re-read as float64 (matrix square root)
    oth = [k for k in V if k != 'raw']
    F['spec_rob']['swd_ratio_min'] = min(V[k]['swd_ratio'] for k in oth); F['spec_rob']['swd_ratio_max'] = max(V[k]['swd_ratio'] for k in oth)
    F['spec_rob']['fd_ratio_min'] = min(V[k]['fd_ratio'] for k in oth); F['spec_rob']['fd_ratio_max'] = max(V[k]['fd_ratio'] for k in oth)
    F['spec_rob']['n_variants_beyond_raw'] = len(oth)
    wcsv(os.path.join(out, 'table_spectral_robustness_s15.csv'), rows)
    # --- forensic grid: can any natural convention give the submitted -1.19 (with UCL "comparable")?
    sc = {'mean_bins_0_5': lambda X: X[:, :low].mean(1), 'bin_0': lambda X: X[:, 0], 'mean_bins_1_5': lambda X: X[:, 1:low].mean(1),
          'mean_bins_0_5_centred': lambda X: ctr(X)[:, :low].mean(1), 'mean_bins_0_5_standardised': lambda X: std(X)[:, :low].mean(1),
          'mean_bins_1_5_standardised_no_bin0': lambda X: std(X[:, 1:])[:, :low - 1].mean(1), 'mean_all_bins': lambda X: X.mean(1)}
    zrows = []; allz = []
    for name, fn in sc.items():
        v = {s: fn(P[s]) for s in VENDORS}
        for s in ('BIDMC', 'UCL'):
            o = [x for x in VENDORS if x != s]; vo = np.concatenate([v[x] for x in o]); n = {x: len(v[x]) for x in o}
            pooled = math.sqrt(sum(np.var(v[x], ddof=1) * (n[x] - 1) for x in o) / sum(n[x] - 1 for x in o))
            den = {'pooled_within_vendor_sd': (vo.mean(), pooled), 'sd_of_all_other_slices': (vo.mean(), float(np.std(vo, ddof=1))),
                   'sd_of_other_vendor_means': (float(np.mean([v[x].mean() for x in o])), float(np.std([v[x].mean() for x in o], ddof=1)))}
            for dn, (c, d) in den.items():
                zz = float((v[s].mean() - c) / d); zrows.append({'scalar': name, 'denominator': dn, 'vendor': s, 'z': f'{zz:+.3f}'})
                if s == 'BIDMC': allz.append(zz)
    assert len(allz) == 21 and all(zz < 0 for zz in allz)
    F['spec_rob']['n_z_conventions'] = len(allz); F['spec_rob']['z_bidmc_smallest_abs'] = min(abs(zz) for zz in allz); F['spec_rob']['z_bidmc_largest_abs'] = max(abs(zz) for zz in allz)
    assert F['spec_rob']['z_bidmc_smallest_abs'] > 2.0            # no convention comes near the submitted -1.19
    wcsv(os.path.join(out, 'table_spectral_zconventions_s15.csv'), zrows)
    say(f"[spectral] BIDMC rank 1 (SWD and Frechet) and outside the hull in all 6 variants; ratios to the next vendor SWD "
        f"{F['spec_rob']['swd_ratio_min']:.2f}-{F['spec_rob']['swd_ratio_max']:.2f}, Frechet {F['spec_rob']['fd_ratio_min']:.2f}-{F['spec_rob']['fd_ratio_max']:.2f}; "
        f"21 z conventions: |z_BIDMC| from {F['spec_rob']['z_bidmc_smallest_abs']:.2f} to {F['spec_rob']['z_bidmc_largest_abs']:.2f}")


# ---------------------------------------------------------------------------------------------- B
def per_vendor(D, RR, F, out):
    txt = open(os.path.join(RR, 'adaptive_threshold', 'prostate_logs.txt')).read()
    runs = {}; mismatch = []
    for m in re.finditer(r'^(\w+?)_s(\d+):\s+avg=([\d.]+)\s+per-domain\(d1-6\)=([\d.,]+)', txt, flags=re.M):
        g, seed, avg, doms = m.group(1), m.group(2), float(m.group(3)), [float(x) for x in m.group(4).split(',')]
        assert len(doms) == 6 and abs(st.mean(doms) - avg) < 1.5e-4, (g, seed)          # the digest's avg is the six-vendor mean ...
        if abs(st.mean(doms) - avg) > 2e-6:                                          # ... to 1e-6 except where recorded here (E-49)
            mismatch.append({'run': f'{g}_s{seed}', 'digest_avg': avg, 'mean_of_six_vendors': st.mean(doms)})
        runs[(g, seed)] = doms
    assert len(runs) == 9 and all((g, s) in runs for g in ('fixmatch', 'daep') for s in ('1337', '42', '2024'))
    seeds = ('1337', '42', '2024'); f = {'seeds': list(seeds), 'gates': {}}; rows = []
    f['digest_avg_mismatch'] = mismatch; assert len(mismatch) <= 1 and all(m['run'] == 'daep_s2024' for m in mismatch), mismatch
    for g in ('daep', 'fixmatch'):
        per = {v: [runs[(g, s)][i] for s in seeds] for i, v in enumerate(VENDORS)}
        avg = [st.mean(runs[(g, s)]) for s in seeds]
        f['gates'][g] = {'vendor_mean': {v: st.mean(x) for v, x in per.items()}, 'vendor_sd': {v: st.stdev(x) for v, x in per.items()},
                         'avg_mean': st.mean(avg), 'avg_sd': st.stdev(avg), 'per_seed': per}
    # these six runs are the ones behind Table main / Table ablation (seed_split_migration/ANALYSIS.md: 85.71 +- 0.57, 85.38 +- 0.58)
    assert f"{f['gates']['fixmatch']['avg_mean'] * 100:.2f} {f['gates']['fixmatch']['avg_sd'] * 100:.2f}" == '85.71 0.57'
    assert f"{f['gates']['daep']['avg_mean'] * 100:.2f} {f['gates']['daep']['avg_sd'] * 100:.2f}" == '85.38 0.58'
    f['diff'] = {}
    for i, v in enumerate(VENDORS):
        d = [runs[('daep', s)][i] - runs[('fixmatch', s)][i] for s in seeds]; t, p = t_p_two_sided_df2(d)
        f['diff'][v] = {'mean': st.mean(d), 'per_seed': d, 't': t, 'p': p}
        rows.append({'vendor': v, 'daep_mean': f"{f['gates']['daep']['vendor_mean'][v]:.4f}", 'daep_sd': f"{f['gates']['daep']['vendor_sd'][v]:.4f}",
                     'threshold_mean': f"{f['gates']['fixmatch']['vendor_mean'][v]:.4f}", 'threshold_sd': f"{f['gates']['fixmatch']['vendor_sd'][v]:.4f}",
                     'daep_minus_threshold': f'{st.mean(d):+.4f}', 'per_seed_1337_42_2024': ';'.join(f'{x:+.4f}' for x in d), 'paired_t_p': f'{p:.3f}'})
    wcsv(os.path.join(out, 'table_prostate_per_vendor_seedsplit_s15.csv'), rows)
    davg = [st.mean(runs[('daep', s)]) - st.mean(runs[('fixmatch', s)]) for s in seeds]; t, p = t_p_two_sided_df2(davg)
    f['diff_avg'] = {'mean': st.mean(davg), 'per_seed': davg, 'p': p}
    assert f"{f['diff_avg']['mean'] * 100:+.2f}" == '-0.33' and abs(p - 0.18) < 0.01          # seed_split_migration/ANALYSIS.md: -0.33, p 0.18
    # Published single-run per-vendor Dice of SynFoC-UNet as typed in the submitted Table tab:prostate-domains (the S15 patch asserts
    # equality with that table). [UNVERIFIED against the SynFoC paper in S15 -> N-37, proof.]
    f['synfoc'] = {'BIDMC': 88.1, 'BMC': 85.3, 'HK': 89.6, 'I2CVB': 84.8, 'RUNMC': 88.1, 'UCL': 85.4}
    f['gap'] = {g: {v: f['gates'][g]['vendor_mean'][v] * 100 - f['synfoc'][v] for v in VENDORS} for g in ('daep', 'fixmatch')}
    four = [f['gap'][g][v] for g in ('daep', 'fixmatch') for v in ('BMC', 'HK', 'I2CVB', 'UCL')]
    f['gap_other_four_min'] = min(four); f['gap_other_four_max'] = max(four)
    assert all(f['gap'][g]['BIDMC'] < -6 and -4.5 < f['gap'][g]['RUNMC'] < -4 for g in f['gap']) and -2.5 < min(four) and max(four) < 2.5
    dm = {v: f['diff'][v]['mean'] for v in VENDORS}
    # D-19: the submitted "BIDMC inversion" (DAEP 1.93 below the threshold on BIDMC, fixed-split) is absent under the seed-split protocol
    assert abs(dm['BIDMC']) < 0.002 and min(dm, key=dm.get) == 'HK' and dm['HK'] < -0.015, dm
    assert all(f['diff'][v]['p'] > 0.05 for v in VENDORS if v != 'HK'), {v: f['diff'][v]['p'] for v in VENDORS}
    f['largest_abs_diff_vendor'] = max(dm, key=lambda v: abs(dm[v])); f['n_seeds_bidmc_daep_higher'] = sum(x > 0 for x in f['diff']['BIDMC']['per_seed'])
    # cross-check with the revision-code threshold references (three seed-splits) and the same-configuration spread
    ps = {r['run']: r for r in rd(os.path.join(D, 'table_phaseB_per_seed.csv')) if r['dataset'] == 'prostate'}
    v18 = [float(ps[r]['dom1_dice_at_best']) for r in ('v18B_prostate_fixmatch_split1337_init1337', 'v18B_prostate_fixmatch_s42', 'v18B_prostate_fixmatch_s2024')]
    f['v18_threshold_bidmc'] = v18; f['v18_threshold_bidmc_mean'] = st.mean(v18); f['v18_threshold_bidmc_sd'] = st.stdev(v18)
    s14 = json.load(open(os.path.join(D, 'extra_s14', 'facts_s14.json')))['fda']
    f['init_range_per_vendor'] = s14['init_range_per_vendor']; f['init_bidmc_sd'] = s14['init_bidmc_sd']
    assert min(f['init_range_per_vendor'].values()) > 0.0099 and max(f['init_range_per_vendor'].values()) < 0.03
    assert sum(abs(dm[v]) <= f['init_range_per_vendor'][v] for v in VENDORS) == 5          # only HK's difference exceeds its same-configuration range
    F['pv'] = f
    say('[per-vendor, seed-split] DAEP', {v: round(f['gates']['daep']['vendor_mean'][v] * 100, 2) for v in VENDORS}, '| threshold',
        {v: round(f['gates']['fixmatch']['vendor_mean'][v] * 100, 2) for v in VENDORS}, '| DAEP-thr', {v: round(dm[v] * 100, 2) for v in VENDORS},
        '| p', {v: round(f['diff'][v]['p'], 3) for v in VENDORS})


# ---------------------------------------------------------------------------------------------- C
def bands(D, F, out):
    bg = rd(os.path.join(D, 'table_band_generality.csv')); mm = rd(os.path.join(D, 'table_mm_bands_summary.csv')); mmd = rd(os.path.join(D, 'table_mm_bands.csv'))
    DAEP = 'v18A_prostate_daep_default_s1337'; THR = ['v18B_prostate_fixmatch_split1337_init1337', 'v18B_prostate_fixmatch_s42', 'v18B_prostate_fixmatch_s2024']
    cols = [('edt1', 'auroc_edt1'), ('edt2', 'auroc_edt2'), ('edt3', 'auroc_edt3'), ('edt5', 'auroc_edt5'), ('edt8', 'auroc_edt8'), ('edt12', 'auroc_edt12'),
            ('cheb5', 'auroc_cheb5'), ('pred5', 'auroc_predband5'), ('union5', 'auroc_unionband5'), ('rel5', 'auroc_rel5pct'), ('rel10', 'auroc_rel10pct'), ('all', 'auroc_all')]
    mmc = [('mm0.5', 'auroc_mm0.5'), ('mm1', 'auroc_mm1.0'), ('mm2', 'auroc_mm2.0'), ('mm3', 'auroc_mm3.0'), ('mm5', 'auroc_mm5.0')]

    def one(run):
        r = [x for x in bg if x['dataset'] == 'prostate' and x['run'] == run and x['signal'] == 'u_epi' and x['which'] == 'ema']; assert len(r) == 6, (run, len(r))
        o = {k: st.mean(float(x[c]) for x in r) for k, c in cols}
        m = [x for x in mm if x['dataset'] == 'prostate' and x['run'] == run and x['signal'] == 'u_epi' and x['which'] == 'ema']; assert len(m) == 1 and m[0]['n_domains'] == '6', run
        o.update({k: float(m[0][c]) for k, c in mmc}); o['frac_err'] = {k: float(m[0][f'frac_errors_within_{k[2:] if "." in k else k[2:] + ".0"}mm']) for k, _ in mmc}
        o['frac_pix_edt5'] = st.mean(float(x['frac_pixels_in_edt5']) for x in r); o['frac_err_edt5'] = st.mean(float(x['frac_errors_in_edt5']) for x in r)
        return o
    d = one(DAEP); t3 = [one(r) for r in THR]
    keys = [k for k, _ in cols] + [k for k, _ in mmc]
    t = {k: st.mean(x[k] for x in t3) for k in keys}; tmin = {k: min(x[k] for x in t3) for k in keys}; tmax = {k: max(x[k] for x in t3) for k in keys}
    f = {'daep': d, 'thr_mean': t, 'thr_min': tmin, 'thr_max': tmax, 'thr_frac_err': {k: st.mean(x['frac_err'][k] for x in t3) for k, _ in mmc}}
    # statements of the text
    for o in [d] + t3:
        e = [o[k] for k in ('edt1', 'edt2', 'edt3', 'edt5', 'edt8', 'edt12')]; m_ = [o[k] for k in ('mm0.5', 'mm1', 'mm2', 'mm3', 'mm5')]
        assert all(a < b for a, b in zip(e, e[1:])) and all(a < b for a, b in zip(m_, m_[1:])), 'not monotone in width'
        assert e[-1] < o['all'], 'widest band above the all-pixel value'
        assert o['mm2'] < o['edt5'] < o['mm3'], 'EDT-5 px not bracketed by the 2 and 3 mm bands'
        assert 0 < o['cheb5'] - o['edt5'] < 0.03 and 0 < o['pred5'] - o['edt5'] < 0.04 and 0 < o['union5'] - o['edt5'] < 0.03   # wider / shifted bands rank slightly better
        assert o['edt1'] < 0.58 and o['mm0.5'] < 0.57 and o['mm1'] < 0.61 and o['mm5'] < 0.78
    f['max_abs_edt5_vs_cheb5'] = max(abs(o['edt5'] - o['cheb5']) for o in [d] + t3)
    f['max_abs_edt5_vs_pred5'] = max(abs(o['edt5'] - o['pred5']) for o in [d] + t3); f['max_abs_edt5_vs_union5'] = max(abs(o['edt5'] - o['union5']) for o in [d] + t3)
    px = [x for x in mmd if x['dataset'] == 'prostate' and x['run'] == DAEP and x['signal'] == 'u_epi' and x['band_mm'] in ('1', '1.0')]
    five = {x['domain_name']: float(x['five_px_in_mm']) for x in px}; assert len(five) == 6, five
    f['five_px_in_mm'] = five; f['five_px_in_mm_min'] = min(five.values()); f['five_px_in_mm_max'] = max(five.values())
    # mm quoted only from JSONs with full spacing coverage (HANDOFF 3, Phase D): every test image carries a verified spacing
    cov = 0; tot = 0
    for run in [DAEP] + THR:
        J = json.load(open(latest(ARGS.returns, f'diag_out/prostate__{run}__ema.json')))
        for dk, dv in J['per_domain'].items(): cov += dv['n_images_with_mm']; tot += dv['n_images']
        if run == DAEP: f['n_test_images'] = sum(dv['n_images'] for dv in J['per_domain'].values())
    assert cov == tot and tot > 0; f['mm_coverage'] = [cov, tot]
    oc = rd(os.path.join(D, 'table_oracle_ceiling.csv')); g = []; bid = []
    for run in THR:
        r = [x for x in oc if x['dataset'] == 'prostate' and x['run'] == run and x['which'] == 'ema']; assert len(r) == 6, run
        g.append(st.mean(float(x['gain_band5']) for x in r)); bb = [x for x in r if x['domain_name'] == 'BIDMC'][0]; bid.append([float(bb['dice_asis']), float(bb['dice_band_5'])])
    assert 0.085 < min(g) and max(g) < 0.10
    F['oracle_seedsplit'] = {'gain_per_run': g, 'gain_mean': st.mean(g), 'bidmc_asis_band5': bid}
    F['bands'] = f
    lab = {'edt1': 'Euclidean, 1 px', 'edt2': 'Euclidean, 2 px', 'edt3': 'Euclidean, 3 px', 'edt5': 'Euclidean, 5 px (main analysis)', 'edt8': 'Euclidean, 8 px', 'edt12': 'Euclidean, 12 px',
           'cheb5': 'Chebyshev, 5 px', 'pred5': 'around the predicted boundary, 5 px', 'union5': 'around either boundary, 5 px', 'rel5': '5 % of the equivalent radius', 'rel10': '10 % of the equivalent radius',
           'mm0.5': '0.5 mm', 'mm1': '1 mm', 'mm2': '2 mm', 'mm3': '3 mm', 'mm5': '5 mm', 'all': 'all pixels'}
    wcsv(os.path.join(out, 'table_band_definitions_s15.csv'),
         [{'band': k, 'label': lab[k], 'daep_default_s1337': f'{d[k]:.4f}', 'threshold_mean_3_seed_splits': f'{t[k]:.4f}', 'threshold_min': f'{tmin[k]:.4f}', 'threshold_max': f'{tmax[k]:.4f}'}
          for k in ['edt1', 'edt2', 'edt3', 'edt5', 'edt8', 'edt12', 'cheb5', 'pred5', 'union5', 'rel5', 'rel10', 'mm0.5', 'mm1', 'mm2', 'mm3', 'mm5', 'all']])
    say('[bands] DAEP', {k: round(d[k], 3) for k in keys}, '| thr mean', {k: round(t[k], 3) for k in keys}, '| 5 px =', round(f['five_px_in_mm_min'], 2), '-', round(f['five_px_in_mm_max'], 2),
        'mm | errors within 1 mm (DAEP)', round(d['frac_err']['mm1'], 3), '| mm coverage', cov, '/', tot, '| test images', f['n_test_images'])


def build(derived, returns, release_results, out):
    global ARGS
    ARGS = argparse.Namespace(returns=returns)
    os.makedirs(out, exist_ok=True); F = {}
    spectral_robustness(returns, F, out); per_vendor(derived, release_results, F, out); bands(derived, F, out)
    json.dump(F, open(os.path.join(out, 'facts_s15.json'), 'w'), indent=1, sort_keys=True)
    say(f'[done] {out}: 4 tables + facts_s15.json; all assertions passed')
    open(os.path.join(out, 'make_extra_s15.console.log'), 'w').write('\n'.join(LOG) + '\n')
    return F


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', required=True); ap.add_argument('--returns', required=True); ap.add_argument('--release_results', required=True); ap.add_argument('--out', default='')
    a = ap.parse_args()
    build(a.derived, a.returns, a.release_results, a.out or os.path.join(a.derived, 'extra_s15'))
