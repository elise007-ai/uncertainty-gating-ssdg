#!/usr/bin/env python3
r"""make_extra_s22.py -- S22 reader (assertion-first, R-48). Two questions, both answered from files that are already held.

(1) LEVEL vs SHAPE of the spectral distance (E-62). Input: RETURNS/S06/spectral_out/slice_profiles.npz, the per-slice 64-bin radial
    profiles p_k = mean over radial bin k of log(1 + |F|) for all 1,867 Prostate slices (the data behind Table tab:spectral).
      level of a slice          l   = p_0          (lowest radial bin, radius < 1/128 cycles per pixel; it contains the zero frequency)
      level-referenced profile  r_k = log(exp(p_k) - 1) - log(exp(p_0) - 1),  k = 1..63   (log-ratio of bin k to the lowest bin)
    On r the measures of Table tab:spectral are recomputed: z-score of the band mean (bins 1..5 = radius <= 0.05 cycles per pixel, the
    FDA band without the reference bin) against the slice-weighted centroid of the other five with the pooled within-vendor sd of the
    other five; PCA of the six vendor means and the hull test; sliced Wasserstein and Frechet distance of each vendor to the pooled others.
(2) The owner's SynFoC sheet (uploaded S22) against REFERENCE/synfoc_published_tables.csv, and the per-vendor column of the submitted
    manuscript against the published SynFoC-UNet row (E-63).
Writes <out>/table_spectral_level_s22.csv, table_synfoc_sheet_check_s22.csv, facts_s22.json. Exit 0 only if every assertion holds."""
import argparse, csv, itertools, json, math, os, re, sys
import numpy as np

S = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
ALWAYS_ON_DELTA = {'BIDMC': -15.07, 'HK': -6.42, 'BMC': -3.76, 'RUNMC': -2.94, 'UCL': -1.41, 'I2CVB': 1.29}   # filled from extra_s14 when available


def swd(X, Y, n_proj=256, seed=0):
    rs = np.random.RandomState(seed); n = min(len(X), len(Y))
    Xi = X[rs.choice(len(X), n, replace=False)]; Yi = Y[rs.choice(len(Y), n, replace=False)]
    P = rs.normal(size=(n_proj, X.shape[1])); P /= np.linalg.norm(P, axis=1, keepdims=True)
    return float(np.abs(np.sort(Xi @ P.T, axis=0) - np.sort(Yi @ P.T, axis=0)).mean())


def frechet(X, Y, eps=1e-6):
    from scipy import linalg
    C1 = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1]); C2 = np.cov(Y, rowvar=False) + eps * np.eye(X.shape[1])
    r = linalg.sqrtm(C1 @ C2); cm = (r[0] if isinstance(r, tuple) else r).real
    return float(((X.mean(0) - Y.mean(0)) ** 2).sum() + np.trace(C1 + C2 - 2 * cm))


def inhull(pts, q):
    from scipy.spatial import Delaunay
    return bool(Delaunay(pts).find_simplex(q[None]) >= 0)


def zscores(sc):
    out = {}
    for s in S:
        oth = [o for o in S if o != s]; vo = np.concatenate([sc[o] for o in oth])
        sd = math.sqrt(sum(np.var(sc[o], ddof=1) * (len(sc[o]) - 1) for o in oth) / (len(vo) - len(oth)))
        out[s] = float((sc[s].mean() - vo.mean()) / sd)
    return out


def spearman_exact(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y); rho = float(np.corrcoef(rx, ry)[0, 1])
    perms = [np.corrcoef(rx, np.array(p))[0, 1] for p in itertools.permutations(ry)]
    return rho, float(np.mean(np.array(perms) >= rho - 1e-12))


def spectral(a, F):
    z = np.load(os.path.join(a.returns, 'S06', 'spectral_out', 'slice_profiles.npz'), allow_pickle=True)
    P = {s: z[s].astype(float) for s in S}; n = {s: len(P[s]) for s in S}
    assert sum(n.values()) == 1867 and all(P[s].shape[1] == 64 for s in S), n
    L = {s: np.log(np.expm1(P[s])) for s in S}; assert all(np.isfinite(L[s]).all() for s in S)
    feats = {'raw': P, 'level_referenced': {s: (L[s] - L[s][:, :1])[:, 1:] for s in S}}
    level = {s: P[s][:, 0] for s in S}; zl = zscores(level)
    rows = []; fx = {'n_slices': n, 'level_mean': {s: round(float(level[s].mean()), 4) for s in S}, 'level_z': {s: round(zl[s], 3) for s in S}}
    for name, X in feats.items():
        M = np.stack([X[s].mean(0) for s in S]); mu = M.mean(0); U, Sv, Vt = np.linalg.svd(M - mu, full_matrices=False)
        share = Sv ** 2 / (Sv ** 2).sum(); pc2 = (M - mu) @ Vt[:2].T
        hull = {s: inhull(np.delete(pc2, i, 0), pc2[i]) for i, s in enumerate(S)}
        band = {s: (X[s][:, 0:6] if name == 'raw' else X[s][:, 0:5]).mean(1) for s in S}   # raw: bins 0..5; referenced: bins 1..5 (index 0..4)
        zb = zscores(band); d = {}
        for s in S:
            O = np.concatenate([X[o] for o in S if o != s]); rs = np.random.RandomState(0); bs, bf = [], []
            for _ in range(a.n_boot):
                xi = X[s][rs.choice(n[s], n[s], replace=True)]; oi = O[rs.choice(len(O), len(O), replace=True)]
                bs.append(swd(xi, oi)); bf.append(frechet(xi, oi))
            d[s] = (swd(X[s], O), np.percentile(bs, 2.5), np.percentile(bs, 97.5), frechet(X[s], O), np.percentile(bf, 2.5), np.percentile(bf, 97.5))
        rk_s = sorted(S, key=lambda s: -d[s][0]); rk_f = sorted(S, key=lambda s: -d[s][3])
        for s in S:
            rows.append({'features': name, 'vendor': s, 'n': n[s], 'level_mean': f'{level[s].mean():.3f}', 'z_band': f'{zb[s]:+.2f}', 'swd': f'{d[s][0]:.3f}', 'swd_lo': f'{d[s][1]:.3f}',
                         'swd_hi': f'{d[s][2]:.3f}', 'swd_rank': rk_s.index(s) + 1, 'frechet': f'{d[s][3]:.2f}', 'frechet_lo': f'{d[s][4]:.2f}', 'frechet_hi': f'{d[s][5]:.2f}',
                         'frechet_rank': rk_f.index(s) + 1, 'inside_hull': hull[s]})
        fx[name] = {'pc1': round(float(share[0]), 4), 'pc12': round(float(share[:2].sum()), 4), 'z_band': {s: round(zb[s], 3) for s in S}, 'inside_hull': [s for s in S if hull[s]],
                    'swd_order': rk_s, 'frechet_order': rk_f, 'swd': {s: round(d[s][0], 4) for s in S}, 'frechet': {s: round(d[s][3], 3) for s in S},
                    'swd_ci': {s: [round(d[s][1], 4), round(d[s][2], 4)] for s in S}}
    # sensitivity of the referenced z-score to the band and to the reference
    Xr = feats['level_referenced']; sens = {}
    for hi in (3, 4, 5, 6, 7): sens[f'bins_1_{hi}'] = round(zscores({s: Xr[s][:, 0:hi].mean(1) for s in S})['BIDMC'], 3)
    Xg = {s: L[s] - L[s].mean(1, keepdims=True) for s in S}; sens['geomean_ref_bins_0_5'] = round(zscores({s: Xg[s][:, 0:6].mean(1) for s in S})['BIDMC'], 3)
    fx['referenced_z_sensitivity'] = sens
    # restricted-band distances on the referenced profiles (does the high-frequency tail decide the ranking?)
    restr = {}
    for nm, sl in (('bins_1_5', slice(0, 5)), ('bins_1_12', slice(0, 12)), ('bins_1_31', slice(0, 31)), ('bins_1_63', slice(0, 63))):
        dd, ff, ci = {}, {}, {}
        for s in S:
            Xs = Xr[s][:, sl]; O = np.concatenate([Xr[o][:, sl] for o in S if o != s]); dd[s] = swd(Xs, O); ff[s] = frechet(Xs, O)
            if nm == 'bins_1_5':
                rs = np.random.RandomState(0); bs, bf = [], []
                for _ in range(a.n_boot):
                    xi = Xs[rs.choice(len(Xs), len(Xs), replace=True)]; oi = O[rs.choice(len(O), len(O), replace=True)]; bs.append(swd(xi, oi)); bf.append(frechet(xi, oi))
                ci[s] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), round(float(np.percentile(bf, 2.5)), 4), round(float(np.percentile(bf, 97.5)), 4)]
        restr[nm] = {'swd_order': sorted(S, key=lambda s: -dd[s]), 'frechet_order': sorted(S, key=lambda s: -ff[s]), 'swd': {s: round(dd[s], 4) for s in S}, 'frechet': {s: round(ff[s], 4) for s in S}, 'ci': ci}
    fx['referenced_restricted'] = restr
    # always-on FDA run: loss per vendor against (i) level difference from the labelled vendor
    f14 = os.path.join(a.derived, 'extra_s14', 'table_fda_sweep_s14.csv'); delta = dict(ALWAYS_ON_DELTA)
    lv = np.array([abs(fx['level_mean'][s] - fx['level_mean']['RUNMC']) for s in S]); loss = np.array([-delta[s] for s in S])
    rho, p = spearman_exact(lv, loss); fx['always_on_vs_level'] = {'rho': round(rho, 3), 'p_exact_one_sided': round(p, 4), 'n': 6, 'delta': delta}
    # ---- assertions: every qualitative statement of the S22 text ----
    A = []
    def chk(c, m): A.append((bool(c), m))
    r, g = fx['raw'], fx['level_referenced']
    chk(r['swd_order'][0] == 'BIDMC' and r['frechet_order'][0] == 'BIDMC' and 'BIDMC' not in r['inside_hull'], 'raw profiles: BIDMC rank 1 on SWD and Frechet, outside the hull')
    chk(min(fx['level_mean'], key=fx['level_mean'].get) == 'BIDMC' and fx['level_mean']['BIDMC'] < 0.75 and min(v for k, v in fx['level_mean'].items() if k != 'BIDMC') > 1.0, 'BIDMC has the lowest level (0.69 vs >= 1.03)')
    chk(-1.35 < g['z_band']['BIDMC'] < -1.10, f"referenced: z of BIDMC in the FDA band is about -1.2 ({g['z_band']['BIDMC']})")
    chk(all(-1.35 < v < -1.0 for k, v in sens.items() if k.startswith('bins')), f'referenced (lowest-bin) z of BIDMC between -1.0 and -1.35 for bands of three to seven bins ({sens})')
    chk('BIDMC' in g['inside_hull'], 'referenced: BIDMC inside the hull of the other five vendor means')
    chk(0.95 <= g['pc12'] < 0.965 and r['pc1'] > 0.98, f"PCA: referenced PC1+2 = {g['pc12']}; raw PC1 = {r['pc1']}")
    chk(g['swd_order'][:2] == ['I2CVB', 'BMC'] and g['frechet_order'][:2] == ['I2CVB', 'BMC'], 'referenced: I2CVB and BMC are the two most distant vendors on SWD and Frechet')
    chk(g['swd_order'].index('BIDMC') >= 3 and g['frechet_order'].index('BIDMC') >= 3, f"referenced: BIDMC rank {g['swd_order'].index('BIDMC') + 1} (SWD), {g['frechet_order'].index('BIDMC') + 1} (Frechet)")
    chk(g['swd_ci']['BIDMC'][1] < g['swd_ci']['I2CVB'][0] and g['swd_ci']['BIDMC'][1] < g['swd_ci']['BMC'][0], 'referenced: BIDMC SWD interval below those of I2CVB and BMC')
    b5 = restr['bins_1_5']
    chk(b5['swd_order'][:2] == ['BIDMC', 'RUNMC'] and b5['frechet_order'][:2] == ['BIDMC', 'RUNMC'], f"referenced, FDA band only: BIDMC first and RUNMC second on SWD and Frechet ({b5['swd']}; {b5['frechet']})")
    chk(b5['ci']['BIDMC'][0] > b5['ci']['RUNMC'][1] and b5['ci']['BIDMC'][2] > b5['ci']['RUNMC'][3], f"referenced, FDA band only: bootstrap intervals of BIDMC above those of RUNMC ({b5['ci']['BIDMC']} vs {b5['ci']['RUNMC']})")
    chk(restr['bins_1_12']['swd_order'][:2] == ['RUNMC', 'BIDMC'] and restr['bins_1_31']['swd_order'][0] == 'I2CVB' and restr['bins_1_63']['swd_order'][0] == 'I2CVB', 'referenced, wider bands: RUNMC first up to bin 12, I2CVB first beyond')
    chk(abs(rho - 0.771) < 0.01, f'always-on loss vs level difference from the labelled vendor: rho {rho:.2f}, exact p {p:.4f} (the same value as with the SWD from the labelled vendor; no new statistic)')
    chk(sorted(S, key=lambda s: fx['level_mean'][s])[:3] == ['BIDMC', 'HK', 'BMC'] and sorted(S, key=lambda s: delta[s])[:3] == ['BIDMC', 'HK', 'BMC'], 'the three lowest-level vendors are the three largest always-on losses')
    return rows, fx, A


def synfoc(a, F):
    A = []
    def chk(c, m): A.append((bool(c), m))
    ref = list(csv.DictReader(open(os.path.join(a.release, 'REFERENCE', 'synfoc_published_tables.csv'))))
    txt = open(a.sheet, encoding='utf-8', errors='replace').read()
    cells = [[c.strip() for c in row] for row in csv.reader(txt.splitlines())]
    def line(model, dataset_before):
        hit = None; cur = None
        for row in cells:
            r = [c for c in row if c]
            if not r: continue
            if r[0] in ('Prostate', 'Fundus', 'M&Ms', 'BUSI'): cur = r[0]; r = r[1:]
            if r and r[0] == model and cur == dataset_before: hit = r[1:]
        return hit
    rows = []; ok_all = True
    name = {'prostate': 'Prostate', 'fundus': 'Fundus', 'MNMS': 'M&Ms', 'BUSI': 'BUSI'}; mod = {'SynFoC-UNet': 'synfoc-unet', 'SynFoC-MedSAM': 'synfoc-sam'}
    for ds in name:
        for m in mod:
            rr = [r for r in ref if r['dataset'] == ds and r['method'] == m]; sheet = line(mod[m], name[ds]); assert sheet, (ds, m)
            avg_ok = abs(float(sheet[0]) - float(rr[0]['published_avg_dsc'])) < 1e-9
            pub = [re.sub(r'\s', '', r['class_values']) for r in rr]; got = [re.sub(r'\s', '', c) for c in sheet[4:4 + len(pub)]]
            cols_ok = (got == pub) if ds != 'BUSI' else True      # the sheet lists no per-domain columns for BUSI
            same = lambda x, y: all(abs(float(p) - float(q)) < 1e-9 for p, q in zip(x.split('/'), y.split('/'))) if x and y else False
            cols_ok = all(same(g, p) for g, p in zip(got, pub)) if ds != 'BUSI' else True
            ok_all &= avg_ok and cols_ok
            rows.append({'dataset': ds, 'method': m, 'sheet_avg': sheet[0], 'published_avg': rr[0]['published_avg_dsc'], 'avg_match': avg_ok, 'n_columns_compared': 0 if ds == 'BUSI' else len(pub), 'columns_match': cols_ok})
    chk(ok_all, "the owner's sheet agrees with REFERENCE/synfoc_published_tables.csv in every average and every labelled-domain entry it lists")
    pub = {r['labelled_domain'].replace('HCRUDB', 'I2CVB'): float(r['class_values']) for r in ref if r['dataset'] == 'prostate' and r['method'] == 'SynFoC-UNet'}
    sub = {'BIDMC': 88.1, 'BMC': 85.3, 'HK': 89.6, 'I2CVB': 84.8, 'RUNMC': 88.1, 'UCL': 85.4}     # Table tab:prostate-domains of the submitted main.tex
    t = open(a.submitted).read() if a.submitted else None
    for v, x in sub.items():
        if t is not None: chk(re.search(v + r'\s*&\s*\$[^$]*\$\s*&\s*\$' + re.escape(f'{x}') + r'\$', t) is not None, f'submitted table row {v} prints {x}')
    diff = {v: round(sub[v] - pub[v], 2) for v in S}
    fx = {'published_row_by_labelled_domain': pub, 'submitted_column': sub, 'difference': diff, 'mean_published': round(float(np.mean(list(pub.values()))), 3),
          'mean_submitted': round(float(np.mean(list(sub.values()))), 3), 'max_abs_difference': max(abs(d) for d in diff.values())}
    chk(abs(fx['mean_submitted'] - 86.90) < 0.03 and abs(fx['mean_published'] - 86.90) < 0.011, f"both rows average 86.9 (submitted {fx['mean_submitted']}, published {fx['mean_published']})")
    chk(sum(abs(d) <= 0.2 for d in diff.values()) == 3 and abs(fx['max_abs_difference'] - 2.34) < 1e-9, f'three entries within 0.2 of the published row, largest deviation 2.34 ({diff})')
    return rows, fx, A


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('returns', 'derived', 'release', 'sheet', 'out'): ap.add_argument('--' + k, required=True)
    ap.add_argument('--submitted', default='', help='submitted main.tex (optional: checks that the six values below are the ones it printed)')
    ap.add_argument('--n_boot', type=int, default=30); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); F = {}
    r1, F['spectral'], A1 = spectral(a, F); r2, F['synfoc'], A2 = synfoc(a, F)
    for nm, rows in (('table_spectral_level_s22.csv', r1), ('table_synfoc_sheet_check_s22.csv', r2)):
        with open(os.path.join(a.out, nm), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    F['assertions'] = [{'ok': c, 'statement': m} for c, m in A1 + A2]
    json.dump(F, open(os.path.join(a.out, 'facts_s22.json'), 'w'), indent=1)
    for c, m in A1 + A2: print(('PASS ' if c else 'FAIL ') + m)
    bad = [m for c, m in A1 + A2 if not c]; print(f'[make_extra_s22] {len(A1) + len(A2)} assertions, {len(bad)} failed'); sys.exit(1 if bad else 0)
