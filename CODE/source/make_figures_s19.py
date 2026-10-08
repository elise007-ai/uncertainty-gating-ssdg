#!/usr/bin/env python3
"""make_figures_s19.py -- Figures 3-5 as VECTOR PDF (S19; journal artwork: line art must not be a 200-dpi bitmap).
  python make_figures_s19.py --derived <derived_v04_S12> --out <manuscript dir>
Same drawing code and same input tables as make_figures_s16.py (imported from the same folder); only the output format changes.
Writes Figure3.pdf, Figure4.pdf, Figure5.pdf with embedded TrueType fonts and <derived>/extra_s19/figures_s19.json (sha256, for the pins)."""
import argparse, hashlib, importlib.util, json, os
import matplotlib
matplotlib.use('Agg')
import pandas as pd

def build(derived, out):
    here = os.path.dirname(os.path.abspath(__file__)); spec = importlib.util.spec_from_file_location('mf16', os.path.join(here, 'make_figures_s16.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42}); m.META = {'Creator': 'make_figures_s19.py', 'Producer': 'matplotlib', 'CreationDate': None}
    E = os.path.join(derived, 'extra_s16'); os.makedirs(out, exist_ok=True); H = {}
    for name, fn, tab in (('Figure3.pdf', m.fig3, 'table_collapse_prostate_s16.csv'), ('Figure4.pdf', m.fig4, 'table_threshold_sweep_s16.csv'), ('Figure5.pdf', m.fig5, 'table_oracle_band_s16.csv')):
        T = pd.read_csv(os.path.join(E, tab)); assert sorted(T.vendor.unique()) == m.VEND, tab
        p = os.path.join(out, name); fn(T, p); H[name] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return H

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--derived', required=True); ap.add_argument('--out', required=True); ap.add_argument('--no_json', action='store_true'); a = ap.parse_args()
    H = build(a.derived, a.out)
    if not a.no_json:
        d = os.path.join(a.derived, 'extra_s19'); os.makedirs(d, exist_ok=True); json.dump(H, open(os.path.join(d, 'figures_s19.json'), 'w'), indent=1, sort_keys=True)
    for k, v in H.items(): print(f'[fig] {k} sha256 {v[:16]}')
