#!/usr/bin/env python3
"""
prostate_spacing_index.py  (v3, S03)
============================================================================
BSPC-D-26-12287 revision, D-1 / reviewer point 7: millimetre boundary bands.

v1/v2 assumed that ProstateSlice PNG names encode the NIfTI case and z-index.
They do not (S02 return: per-site sequential case ids, slice ids that are not
volume z, 0/36 NCC verifications). v3 makes NO naming assumption: for every
PNG it searches ALL slices of ALL volumes of the same site, under a set of
candidate crops (centre crops, mask-bounding-box crops, full-frame resize) and
the 8 axis orientations, and keeps the best normalised cross-correlation. A
match is accepted only if image NCC >= --ncc_pass AND the mask IoU of the same
(volume, z, crop) >= --iou_pass (or both masks empty). mm per pixel on the
model grid then follows from the matched crop width and the NIfTI spacing.

Per site it also reports the modal (crop, orientation) and the fraction of
PNGs agreeing with it: a real preprocessing pipeline gives one mode with
agreement close to 1.0; anything else is reported, not papered over (R-29).

  python prostate_spacing_index.py --nii_root /opt/data/private/data/Prostate_nii \
      --png_root ../../data/ProstateSlice --out diag_out/prostate_spacing_index.json
  python prostate_spacing_index.py --selftest           # numpy-only synthetic test

Output JSON schema (consumed by diag_uncertainty_harness_v18.make_spacing_lookup):
  slices["<site>/<png stem>"] = {site, phase, png, png_size, match: {vol, z, crop, orient,
                        ncc, iou}, crop_px, spacing_xyz, pass, mm_per_px_at_grid: {G: v}}
  site_summary[site] = {n_png, n_pass, pass_rate, mode_crop, mode_orient,
                        mode_agreement, spacing_x, n_volumes, usable_for_mm_bands}
============================================================================
"""
from __future__ import annotations
import argparse, glob, json, os, re, time
from collections import Counter
import numpy as np
from PIL import Image

SITES = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
D = 48                                       # matching resolution
CENTER_FRACS = [1.0, 0.875, 0.75, 0.625, 0.5]
BBOX_MARGINS = [1.0, 1.5, 2.0, 3.0]
ORIENTS = [f"T{t}Y{fy}X{fx}" for t in (0, 1) for fy in (0, 1) for fx in (0, 1)]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_volume_and_spacing(path):
    try:
        import SimpleITK as sitk
        im = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(im).astype(np.float32), list(map(float, im.GetSpacing()))  # [z,y,x], (x,y,z)
    except ImportError:
        import nibabel as nib
        im = nib.load(path)
        arr = np.transpose(np.asanyarray(im.dataobj).astype(np.float32), (2, 1, 0))
        return arr, list(map(float, im.header.get_zooms()[:3]))


def is_seg_name(name):
    low = name.lower()
    return any(k in low for k in ('seg', 'label', 'mask', 'gt'))


def find_seg_for(img_path):
    d, b = os.path.dirname(img_path), os.path.basename(img_path)
    stem = re.sub(r'\.nii(\.gz)?$', '', b)
    for cand in sorted(glob.glob(os.path.join(d, stem + '*'))):
        if cand != img_path and is_seg_name(os.path.basename(cand)):
            return cand
    return None


def load_site_volumes(nii_root, site):
    files = sorted(f for f in glob.glob(os.path.join(nii_root, '**', '*.nii*'), recursive=True)
                   if site.lower() in [p.lower() for p in f.split(os.sep)] and not is_seg_name(os.path.basename(f)))
    vols = []
    for f in files:
        arr, sp = read_volume_and_spacing(f)
        seg_path = find_seg_for(f)
        seg = read_volume_and_spacing(seg_path)[0] if seg_path else None
        vols.append({'path': f, 'seg_path': seg_path, 'arr': arr, 'seg': seg, 'spacing': sp})
    return vols


# ---------------------------------------------------------------------------
# Candidate generation (numpy core, unit-testable)
# ---------------------------------------------------------------------------

def orient(a, code):
    t, fy, fx = int(code[1]), int(code[3]), int(code[5])
    b = a.T if t else a
    b = b[::-1] if fy else b
    b = b[:, ::-1] if fx else b
    return np.ascontiguousarray(b)


def resize(a, size, nearest=False):
    m = Image.fromarray(np.ascontiguousarray(a, dtype=np.float32), mode='F')
    return np.asarray(m.resize((size, size), Image.NEAREST if nearest else Image.BILINEAR), dtype=np.float32)


def norm_slice(s):
    lo, hi = np.percentile(s, [1, 99])
    return np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)


def crop_candidates(H, W, mask2d):
    """Yields (name, y0, y1, x0, x1) crops in ORIGINAL pixel coordinates."""
    m = min(H, W); cy, cx = H // 2, W // 2
    yield 'full', 0, H, 0, W                           # whole frame (aspect-distorting resize if H != W)
    for f in CENTER_FRACS:
        s = int(round(m * f)); y0 = max(cy - s // 2, 0); x0 = max(cx - s // 2, 0)
        if f == 1.0 and H == W:
            continue                                   # identical to 'full'
        yield f'center{f}', y0, min(y0 + s, H), x0, min(x0 + s, W)
    if mask2d is not None and mask2d.any():
        ys, xs = np.where(mask2d > 0)
        by, bx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        side = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        for mg in BBOX_MARGINS:
            s = max(int(round(side * mg)), 16)
            y0 = int(round(by - s / 2)); x0 = int(round(bx - s / 2))
            y0 = min(max(y0, 0), max(H - s, 0)); x0 = min(max(x0, 0), max(W - s, 0))
            yield f'bbox{mg}', y0, min(y0 + s, H), x0, min(x0 + s, W)


def unit(v):
    v = v - v.mean(axis=1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def build_bank(vols, orient_codes):
    """Returns (unit vectors [N, D*D], meta list, mask bank [N, D*D] bool)."""
    vecs, masks, meta = [], [], []
    for vi, v in enumerate(vols):
        arr, seg = v['arr'], v['seg']
        for z in range(arr.shape[0]):
            sl = norm_slice(arr[z]); mk = (seg[z] > 0).astype(np.float32) if seg is not None else None
            for name, y0, y1, x0, x1 in crop_candidates(*sl.shape, mk):
                c = sl[y0:y1, x0:x1]; cm = mk[y0:y1, x0:x1] if mk is not None else None
                for oc in orient_codes:
                    vecs.append(resize(orient(c, oc), D).ravel())
                    masks.append(resize(orient(cm, oc), D, nearest=True).ravel() > 0.5 if cm is not None else np.zeros(D * D, bool))
                    meta.append((vi, z, name, oc, (y1 - y0, x1 - x0)))
    return unit(np.stack(vecs)), meta, np.stack(masks)


def png_vectors(png_paths):
    imgs, msks, sizes = [], [], []
    for p in png_paths:
        im = Image.open(p).convert('L'); sizes.append(list(im.size))
        imgs.append(resize(np.asarray(im, dtype=np.float32) / 255.0, D).ravel())
        mp = p.replace('image', 'mask')
        if os.path.exists(mp):
            mk = np.asarray(Image.open(mp).convert('L'), dtype=np.float32)
            fg = (mk < 128) if (mk < 128).mean() < 0.5 else (mk >= 128)      # foreground = minority class
            msks.append(resize(fg.astype(np.float32), D, nearest=True).ravel() > 0.5)
        else:
            msks.append(None)
    return unit(np.stack(imgs)), msks, sizes


def iou(a, b):
    if a.sum() == 0 and b.sum() == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / (np.logical_or(a, b).sum() + 1e-8))


def match_site(vols, png_paths, ncc_pass, iou_pass, log=print):
    P, pm, sizes = png_vectors(png_paths)
    # pass 1: orientation vote on a subset of PNGs against all 8 orientations
    sub = P[:: max(1, len(P) // 25)]
    B, meta, _ = build_bank(vols[:8], ORIENTS)          # orientation is a site-level property
    best = (sub @ B.T).argmax(axis=1)
    votes = Counter(meta[j][3] for j in best)
    oc_mode = votes.most_common(1)[0][0]
    log(f"    orientation vote: {votes.most_common(3)} -> {oc_mode}")
    del B
    # pass 2: full match with the modal orientation only
    B, meta, Bm = build_bank(vols, [oc_mode])
    S = P @ B.T
    results = []
    for i, p in enumerate(png_paths):
        j = int(S[i].argmax()); vi, z, crop, oc, (ch, cw) = meta[j]
        ncc = float(S[i, j])
        io = iou(pm[i], Bm[j]) if pm[i] is not None else None
        ok = ncc >= ncc_pass and (io is None or io >= iou_pass)
        results.append({'png': p, 'png_size': sizes[i],
                        'match': {'vol': vi, 'vol_path': vols[vi]['path'], 'z': int(z), 'crop': crop,
                                  'orient': oc, 'ncc': ncc, 'iou': io},
                        'crop_px': [int(ch), int(cw)], 'spacing_xyz': vols[vi]['spacing'], 'pass': bool(ok)})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nii_root', required=True); ap.add_argument('--png_root', default='../../data/ProstateSlice')
    ap.add_argument('--out', default='prostate_spacing_index.json'); ap.add_argument('--grids', default='384')
    ap.add_argument('--ncc_pass', type=float, default=0.85); ap.add_argument('--iou_pass', type=float, default=0.6)
    ap.add_argument('--sites', default=','.join(SITES))
    a = ap.parse_args()
    grids = [int(g) for g in a.grids.split(',')]
    out = {'meta': {'nii_root': a.nii_root, 'png_root': a.png_root, 'grids': grids, 'ncc_pass': a.ncc_pass,
                    'iou_pass': a.iou_pass, 'method': 'brute-force slice/crop/orientation search (v3)',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, 'site_summary': {}, 'slices': {}}
    t0 = time.time()
    for site in a.sites.split(','):
        vols = load_site_volumes(a.nii_root, site)
        pngs = []
        for ph in ('train', 'test'):
            pngs += sorted(glob.glob(os.path.join(a.png_root, site, ph, 'image', '*.png')))
        if not vols or not pngs:
            print(f"[{site}] volumes={len(vols)} pngs={len(pngs)} -> skipped"); continue
        shapes = Counter(v['arr'].shape[1:] for v in vols)
        print(f"[{site}] {len(vols)} volumes (in-plane shapes {dict(shapes)}, spacing_x {vols[0]['spacing'][0]:.4f}), "
              f"{len(pngs)} PNGs, seg found for {sum(v['seg'] is not None for v in vols)} volumes ({time.time()-t0:.0f}s)")
        res = match_site(vols, pngs, a.ncc_pass, a.iou_pass, log=print)
        n_pass = sum(r['pass'] for r in res)
        modes = Counter((r['match']['crop'], r['match']['orient']) for r in res if r['pass'])
        (mode, mode_n) = modes.most_common(1)[0] if modes else ((None, None), 0)
        dup = Counter((r['match']['vol'], r['match']['z']) for r in res if r['pass'])
        n_dup = sum(1 for c in dup.values() if c > 1)
        nccs = [r['match']['ncc'] for r in res]; ious = [r['match']['iou'] for r in res if r['match']['iou'] is not None]
        print(f"    pass {n_pass}/{len(res)} (ncc median {np.median(nccs):.3f}, iou median "
              f"{(np.median(ious) if ious else float('nan')):.3f}); mode crop/orient = {mode} agreement "
              f"{mode_n / max(n_pass, 1):.2f}; {n_dup} NIfTI slices matched by >1 PNG; png sizes {Counter(tuple(r['png_size']) for r in res).most_common(2)}")
        for r in res:
            stem = os.path.splitext(os.path.basename(r['png']))[0]
            ph = 'test' if f'{os.sep}test{os.sep}' in r['png'] else 'train'
            rec = {'site': site, 'phase': ph, **r}
            if r['pass']:
                sx, sy = r['spacing_xyz'][0], r['spacing_xyz'][1]; ch, cw = r['crop_px']
                rec['mm_per_px_at_grid'] = {str(g): 0.5 * (sx * cw / g + sy * ch / g) for g in grids}
                rec['mm_per_px_xy_at_grid'] = {str(g): [sx * cw / g, sy * ch / g] for g in grids}
            out['slices'][f"{site}/{stem}"] = rec        # stems collide across sites (every site has 00_00)
        out['site_summary'][site] = {'n_png': len(res), 'n_pass': n_pass, 'pass_rate': n_pass / len(res),
                                     'mode_crop': mode[0], 'mode_orient': mode[1], 'mode_agreement': mode_n / max(n_pass, 1),
                                     'n_dup_slices': n_dup, 'spacing_x': vols[0]['spacing'][0], 'spacing_y': vols[0]['spacing'][1],
                                     'n_volumes': len(vols), 'inplane_shapes': {str(k): v for k, v in shapes.items()},
                                     'usable_for_mm_bands': bool(n_pass / len(res) >= 0.75 and mode_n / max(n_pass, 1) >= 0.75)}
        del vols
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out['site_summary'], indent=1))
    print(f"[done] {a.out}")


# ---------------------------------------------------------------------------
# Self-test: synthetic volumes; PNGs made by a hidden crop + flip + resize
# ---------------------------------------------------------------------------

def _selftest():
    import tempfile
    rs = np.random.RandomState(0); tmp = tempfile.mkdtemp()
    H, W = 160, 160; vols = []
    yy, xx = np.mgrid[:H, :W]
    for vi in range(3):
        arr = np.zeros((6, H, W), np.float32); seg = np.zeros((6, H, W), np.float32)
        for z in range(6):
            cy, cx, r = 70 + 5 * z + 10 * vi, 85 - 3 * z, 18 + 2 * z
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r + 8) ** 2))
            arr[z] = blob * 200 + rs.normal(0, 8, (H, W)) + 20 * np.sin(xx / (7 + z))
            seg[z] = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2)
        vols.append({'path': f'vol{vi}', 'seg_path': None, 'arr': arr, 'seg': seg, 'spacing': [0.5, 0.5, 3.0]})
    os.makedirs(os.path.join(tmp, 'image')); os.makedirs(os.path.join(tmp, 'mask'))
    truth = []
    for vi, v in enumerate(vols):
        for z in (1, 3, 5):
            sl = norm_slice(v['arr'][z]); mk = v['seg'][z]
            cands = {n: (y0, y1, x0, x1) for n, y0, y1, x0, x1 in crop_candidates(H, W, mk)}
            y0, y1, x0, x1 = cands['bbox2.0']
            im = orient(sl[y0:y1, x0:x1], 'T0Y1X0'); mm = orient(mk[y0:y1, x0:x1], 'T0Y1X0')
            Image.fromarray((resize(im, 96) * 255).astype(np.uint8)).save(os.path.join(tmp, 'image', f'{vi:02d}_{z:02d}.png'))
            Image.fromarray((255 - (resize(mm, 96, True) > 0.5) * 255).astype(np.uint8)).save(os.path.join(tmp, 'mask', f'{vi:02d}_{z:02d}.png'))
            truth.append((vi, z, y1 - y0))
    pngs = sorted(glob.glob(os.path.join(tmp, 'image', '*.png')))
    res = match_site(vols, pngs, 0.85, 0.6, log=lambda s: None)
    ok = sum(1 for r, (vi, z, cw) in zip(res, truth)
             if r['pass'] and r['match']['vol'] == vi and r['match']['z'] == z and r['match']['crop'] == 'bbox2.0'
             and r['match']['orient'] == 'T0Y1X0' and abs(r['crop_px'][1] - cw) <= 1)
    assert ok == len(truth), f"recovered {ok}/{len(truth)}: " + str(
        [(r['match']['vol'], r['match']['z'], r['match']['crop'], r['match']['orient'], round(r['match']['ncc'], 3), r['match']['iou']) for r in res])
    print(f'[prostate_spacing_index v3] self-test PASSED ({ok}/{len(truth)} slices recovered with crop, orientation, z)')


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        _selftest()
    else:
        main()
