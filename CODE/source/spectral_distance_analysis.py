#!/usr/bin/env python3
"""
spectral_distance_analysis.py
============================================================================
BSPC-D-26-12287 revision, reviewer point 9: stronger distributional measures
for the "BIDMC is not outside FDA's spectral support" claim, plus sensitivity
to the FDA band beta.

Per slice (every PNG under <data>/ProstateSlice/<site>/{train,test}/image,
resized to the model grid exactly as the dataloader does) we compute the
2-D log-amplitude spectrum and reduce it to
  * a 64-bin radial profile  (log(1 + |F|), centred, radius in cycles/px), and
  * the FDA-band scalar: mean log-amplitude inside the EXACT low-frequency
    block the training code swaps (four floor(beta*H) x floor(beta*W) corner
    blocks of the unshifted FFT, fourierfuse_v16_modules.fda_spectral_augment).

Then, for every site k (leave-one-site-out, so BIDMC's value has five
reference values to be compared against):
  1. z-score of the site's mean FDA-band scalar vs the slice-count-weighted
     centroid of the other five, pooled within-others std          (paper)
  2. inside/outside the convex hull of the other five site-mean radial
     profiles in PCA-2 space                                        (paper)
  3. sliced Wasserstein-1 distance between the site's slice-profile
     distribution and the pooled others (256 projections, equal-size
     subsamples, 30 bootstrap draws -> 95% CI)                       (NEW)
  4. Frechet (FID-style) distance on the 64-d profiles, and on the
     low-frequency bins only                                        (NEW)
  5. the same z-score for beta in {0.01,0.02,0.05,0.10,0.20}         (NEW)
  6. the full 6x6 pairwise SWD and Frechet matrices                 (NEW)

Outputs JSON + CSV. No GPU. ~1-3 min for 1,510 slices.
  python spectral_distance_analysis.py --data_root ../../data/ProstateSlice --out spectral_out
  python spectral_distance_analysis.py --selftest
============================================================================
"""
from __future__ import annotations
import argparse, glob, hashlib, json, math, os, time
from typing import Dict, List
import numpy as np
from PIL import Image

SITES = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

def load_slice(path: str, grid: int) -> np.ndarray:
    img = Image.open(path).resize((grid, grid), Image.LANCZOS)
    if img.mode != 'L':
        img = img.convert('L')
    return np.asarray(img, dtype=np.float64) / 255.0


def fda_corner_mask(H: int, W: int, beta: float) -> np.ndarray:
    """Exactly the block fda_spectral_augment swaps (unshifted FFT layout)."""
    h = int(math.floor(H * beta)); w = int(math.floor(W * beta))
    m = np.zeros((H, W), dtype=bool)
    if h == 0 or w == 0:
        return m
    m[:h, :w] = True; m[:h, -w:] = True; m[-h:, :w] = True; m[-h:, -w:] = True
    return m


def radial_bins(H: int, W: int, n_bins: int = 64) -> np.ndarray:
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)                     # cycles / px, 0 .. ~0.707
    return np.minimum((r / 0.5 * n_bins).astype(int), n_bins - 1)


def slice_features(x: np.ndarray, masks: Dict[float, np.ndarray], bins: np.ndarray,
                   n_bins: int = 64) -> Dict:
    Fa = np.abs(np.fft.fft2(x, norm='ortho'))
    logA = np.log1p(Fa)
    prof = np.bincount(bins.ravel(), weights=logA.ravel(), minlength=n_bins) / \
        np.maximum(np.bincount(bins.ravel(), minlength=n_bins), 1)
    scal = {str(b): float(logA[m].mean()) for b, m in masks.items() if m.any()}
    return {'profile': prof.astype(np.float32), 'fda_scalar': scal}


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_proj: int = 256, rs=None) -> float:
    rs = rs or np.random.RandomState(0)
    n = min(len(X), len(Y))
    Xi = X[rs.choice(len(X), n, replace=False)]; Yi = Y[rs.choice(len(Y), n, replace=False)]
    d = X.shape[1]
    P = rs.normal(size=(n_proj, d)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    px = np.sort(Xi @ P.T, axis=0); py = np.sort(Yi @ P.T, axis=0)
    return float(np.abs(px - py).mean())


def frechet_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    from scipy import linalg
    mu1, mu2 = X.mean(0), Y.mean(0)
    C1 = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    C2 = np.cov(Y, rowvar=False) + eps * np.eye(X.shape[1])
    r = linalg.sqrtm(C1 @ C2)
    covmean = (r[0] if isinstance(r, tuple) else r).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(C1 + C2 - 2 * covmean))


def bootstrap(fn, X, Y, n_boot: int, seed: int = 0):
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        xi = X[rs.choice(len(X), len(X), replace=True)]
        yi = Y[rs.choice(len(Y), len(Y), replace=True)]
        vals.append(fn(xi, yi))
    vals = np.array(vals)
    return {'point': float(fn(X, Y)), 'lo95': float(np.percentile(vals, 2.5)),
            'hi95': float(np.percentile(vals, 97.5)), 'n_boot': n_boot}


def in_convex_hull_2d(points: np.ndarray, q: np.ndarray) -> bool:
    from scipy.spatial import ConvexHull, Delaunay
    if len(points) < 3:
        return False
    try:
        return bool(Delaunay(points).find_simplex(q[None]) >= 0)
    except Exception:
        hull = ConvexHull(points)
        eq = hull.equations
        return bool(np.all(eq[:, :2] @ q + eq[:, 2] <= 1e-9))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(data_root: str, out_dir: str, grid: int = 384, betas=(0.01, 0.02, 0.05, 0.10, 0.20),
        n_boot: int = 30, phases=('train', 'test'), max_per_site: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    masks = {b: fda_corner_mask(grid, grid, b) for b in betas}
    bins = radial_bins(grid, grid)
    feats = {s: {'profiles': [], 'scalars': {str(b): [] for b in betas}, 'files': []} for s in SITES}
    t0 = time.time()
    for s in SITES:
        files = []
        for ph in phases:
            files += sorted(glob.glob(os.path.join(data_root, s, ph, 'image', '*.png')))
        if max_per_site:
            files = files[:max_per_site]
        for f in files:
            fe = slice_features(load_slice(f, grid), masks, bins)
            feats[s]['profiles'].append(fe['profile'])
            for b in betas:
                feats[s]['scalars'][str(b)].append(fe['fda_scalar'][str(b)])
            feats[s]['files'].append(os.path.relpath(f, data_root))
        print(f"[spectra] {s}: {len(files)} slices ({time.time()-t0:.0f}s)")
    n = {s: len(feats[s]['profiles']) for s in SITES}
    assert all(v > 0 for v in n.values()), f"missing sites: {n}"
    P = {s: np.stack(feats[s]['profiles']) for s in SITES}
    lowbins = int(round(0.05 / 0.5 * 64))            # radial bins within r<=0.05 cyc/px (~6 of 64)
    res = {'meta': {'data_root': data_root, 'grid': grid, 'betas': list(betas), 'n_slices': n,
                    'n_boot': n_boot, 'lowfreq_bins': lowbins, 'phases': list(phases),
                    'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
           'leave_one_out': {}, 'pairwise': {'swd': {}, 'frechet': {}, 'frechet_low': {}}}

    # site means / PCA-2 hull on site-mean profiles (paper convention)
    site_mean = np.stack([P[s].mean(0) for s in SITES])
    mu = site_mean.mean(0)
    U, S, Vt = np.linalg.svd(site_mean - mu, full_matrices=False)
    pc2 = (site_mean - mu) @ Vt[:2].T
    var_expl = float((S[:2] ** 2).sum() / (S ** 2).sum())
    res['pca2'] = {'coords': {s: pc2[i].tolist() for i, s in enumerate(SITES)}, 'var_explained_2': var_expl}

    rs = np.random.RandomState(0)
    for i, s in enumerate(SITES):
        others = [o for o in SITES if o != s]
        Xo = np.concatenate([P[o] for o in others]); Xs = P[s]
        row = {'n': n[s]}
        # 1. + 5. z-scores per beta (slice-count-weighted centroid == pooled mean)
        row['zscore_fda_scalar'] = {}
        for b in betas:
            vs = np.array(feats[s]['scalars'][str(b)])
            vo = np.concatenate([np.array(feats[o]['scalars'][str(b)]) for o in others])
            centroid = vo.mean()
            pooled_std = math.sqrt(sum(np.var(np.array(feats[o]['scalars'][str(b)]), ddof=1) * (n[o] - 1)
                                       for o in others) / max(sum(n[o] - 1 for o in others), 1))
            row['zscore_fda_scalar'][str(b)] = {'site_mean': float(vs.mean()), 'others_centroid': float(centroid),
                                                'pooled_std': pooled_std,
                                                'z': float((vs.mean() - centroid) / pooled_std) if pooled_std > 0 else float('nan')}
        # 2. convex hull of the other five site means in PCA-2 (fit on all six, as in the paper)
        row['inside_pca2_hull_of_others'] = in_convex_hull_2d(np.delete(pc2, i, axis=0), pc2[i])
        # 3./4. distribution distances, site vs pooled others, with bootstrap CI
        row['swd_vs_others'] = bootstrap(lambda a, b: sliced_wasserstein(a, b, rs=np.random.RandomState(1)), Xs, Xo, n_boot)
        row['frechet_vs_others'] = bootstrap(frechet_distance, Xs, Xo, n_boot)
        row['frechet_low_vs_others'] = bootstrap(frechet_distance, Xs[:, :lowbins], Xo[:, :lowbins], n_boot)
        row['swd_low_vs_others'] = bootstrap(lambda a, b: sliced_wasserstein(a, b, rs=np.random.RandomState(1)),
                                             Xs[:, :lowbins], Xo[:, :lowbins], n_boot)
        res['leave_one_out'][s] = row
        print(f"[loo] {s:6s} z(beta=0.05)={row['zscore_fda_scalar']['0.05']['z']:+.2f} "
              f"hull={row['inside_pca2_hull_of_others']} SWD={row['swd_vs_others']['point']:.3f} "
              f"FD={row['frechet_vs_others']['point']:.3f} FDlow={row['frechet_low_vs_others']['point']:.3f}")
    # 6. pairwise matrices
    for a in SITES:
        for b in SITES:
            if a >= b:
                continue
            res['pairwise']['swd'][f'{a}|{b}'] = sliced_wasserstein(P[a], P[b], rs=np.random.RandomState(2))
            res['pairwise']['frechet'][f'{a}|{b}'] = frechet_distance(P[a], P[b])
            res['pairwise']['frechet_low'][f'{a}|{b}'] = frechet_distance(P[a][:, :lowbins], P[b][:, :lowbins])
    # rank of BIDMC among the six leave-one-out distances (1 = most distant)
    for key in ('swd_vs_others', 'frechet_vs_others', 'frechet_low_vs_others', 'swd_low_vs_others'):
        vals = {s: res['leave_one_out'][s][key]['point'] for s in SITES}
        order = sorted(vals, key=lambda k: -vals[k])
        res['leave_one_out']['_rank_' + key] = {s: order.index(s) + 1 for s in SITES}
    # save
    with open(os.path.join(out_dir, 'spectral_distances.json'), 'w') as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, 'spectral_loo_table.csv'), 'w') as f:
        f.write('site,n,z_beta0.01,z_beta0.02,z_beta0.05,z_beta0.10,z_beta0.20,inside_hull,swd,swd_lo95,swd_hi95,'
                'frechet,fd_lo95,fd_hi95,frechet_low,fdl_lo95,fdl_hi95\n')
        for s in SITES:
            r = res['leave_one_out'][s]; z = r['zscore_fda_scalar']
            f.write(f"{s},{r['n']}," + ','.join(f"{z[str(b)]['z']:.3f}" for b in betas) +
                    f",{r['inside_pca2_hull_of_others']},{r['swd_vs_others']['point']:.4f},{r['swd_vs_others']['lo95']:.4f},"
                    f"{r['swd_vs_others']['hi95']:.4f},{r['frechet_vs_others']['point']:.4f},{r['frechet_vs_others']['lo95']:.4f},"
                    f"{r['frechet_vs_others']['hi95']:.4f},{r['frechet_low_vs_others']['point']:.4f},"
                    f"{r['frechet_low_vs_others']['lo95']:.4f},{r['frechet_low_vs_others']['hi95']:.4f}\n")
    np.savez_compressed(os.path.join(out_dir, 'slice_profiles.npz'),
                        **{s: P[s] for s in SITES}, **{f'files_{s}': np.array(feats[s]['files']) for s in SITES})
    print(f"[done] {out_dir}/spectral_distances.json, spectral_loo_table.csv, slice_profiles.npz")
    return res


def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp()
    rs = np.random.RandomState(0)
    yy, xx = np.mgrid[:96, :96]
    for i, s in enumerate(SITES):
        for ph in ('train', 'test'):
            d = os.path.join(tmp, s, ph, 'image'); os.makedirs(d)
            for k in range(6):
                base = 0.5 + 0.3 * np.sin(2 * np.pi * (xx / 30.0 + i * 0.05)) * np.exp(-((yy - 48) ** 2) / 800)
                noise = rs.normal(0, 0.05 * (1 + (i == 0) * 3), size=(96, 96))   # site 0 = noisier
                img = np.clip(base + noise, 0, 1)
                Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(d, f'Case{k:02d}_{ph}.png'))
    out = os.path.join(tmp, 'out')
    res = run(tmp, out, grid=64, betas=(0.05, 0.1), n_boot=5)
    assert set(res['leave_one_out']) >= set(SITES)
    assert os.path.exists(os.path.join(out, 'spectral_loo_table.csv'))
    m = fda_corner_mask(64, 64, 0.05); assert m.sum() == 4 * 3 * 3
    # SWD of identical distributions ~0, of shifted ones > 0
    X = rs.normal(size=(200, 8)); assert sliced_wasserstein(X, X) < 1e-9
    assert sliced_wasserstein(X, X + 2.0) > 1.0 and frechet_distance(X, X + 2.0) > 8 * 3.9
    assert in_convex_hull_2d(np.array([[0, 0], [1, 0], [0, 1], [1, 1.]]), np.array([0.5, 0.5]))
    assert not in_convex_hull_2d(np.array([[0, 0], [1, 0], [0, 1], [1, 1.]]), np.array([2.0, 2.0]))
    print('[spectral_distance_analysis] self-test PASSED')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='../../data/ProstateSlice')
    ap.add_argument('--out', default='spectral_out')
    ap.add_argument('--grid', type=int, default=384)
    ap.add_argument('--n_boot', type=int, default=30)
    ap.add_argument('--max_per_site', type=int, default=0, help='debug: cap slices per site')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        run(a.data_root, a.out, grid=a.grid, n_boot=a.n_boot, max_per_site=a.max_per_site)
