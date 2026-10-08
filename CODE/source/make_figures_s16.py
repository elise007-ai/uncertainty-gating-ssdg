#!/usr/bin/env python3
"""make_figures_s16.py -- Figures 3-5 regenerated from the S16 tables (N-35, D-22; settles E-06 for these three figures).
  python make_figures_s16.py --derived <derived_v04_S12> --out <manuscript dir>
Reads <derived>/extra_s16/table_collapse_prostate_s16.csv, table_threshold_sweep_s16.csv, table_oracle_band_s16.csv (make_extra_s16.py);
writes Figure3.png, Figure4.png, Figure5.png and <derived>/extra_s16/figures_s16.json (sha256 of each PNG, for the pins).
Layout rules (protocol 12, R-16): no claim in a figure title (claims live in the caption, which is pinned to the tables); legends sit
above the axes, outside the data; no reference line to a number we cannot source (the 'SynFoC (BIDMC)' line of the old Figure 5: E-52)."""
import argparse, hashlib, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd

VEND = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200, 'figure.dpi': 100,
                     'font.family': 'DejaVu Sans', 'svg.hashsalt': 's16'})
META = {'Software': 'make_figures_s16.py'}          # fixed PNG metadata -> reproducible bytes in one environment


def fig3(T, path):
    T = T.set_index('vendor').loc[VEND]; x = np.arange(6); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ea = np.vstack([T.thr_all_m - T.thr_all_min, T.thr_all_max - T.thr_all_m]); eb = np.vstack([T.thr_band_m - T.thr_band_min, T.thr_band_max - T.thr_band_m])
    ax.bar(x - w / 2, T.thr_all_m, w, yerr=ea, color='#9ecae1', ecolor='black', capsize=3, label='all pixels (threshold teachers: mean, range)')   # whiskers = min..max of 3 teachers
    ax.bar(x + w / 2, T.thr_band_m, w, yerr=eb, color='#08519c', ecolor='black', capsize=3, label='±5 px boundary band')
    ax.plot(x - w / 2, T.daep_all, 'D', ms=5, mfc='white', mec='black', ls='none', label='DAEP teacher (seed 1337)')
    ax.plot(x + w / 2, T.daep_band, 'D', ms=5, mfc='white', mec='black', ls='none')
    ax.axhline(0.5, color='0.3', ls='--', lw=1); ax.set_xlim(-0.6, 6.0); ax.text(5.97, 0.508, 'chance', ha='right', va='bottom', fontsize=8, color='0.3')   # right of the last bar (ends at 5.38)
    ax.set_xticks(x); ax.set_xticklabels(VEND); ax.set_ylim(0.45, 1.03); ax.set_yticks(np.arange(0.5, 1.01, 0.1)); ax.set_ylabel('error-ranking AUROC')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(); fig.savefig(path, metadata=META); plt.close(fig)


def fig4(S, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.5)); cols = plt.cm.viridis(np.linspace(0.0, 0.92, 6))
    for v, c in zip(VEND, cols):
        d = S[S.vendor == v].sort_values('tau'); ax.plot(d.tau, d.dice_mean * 100, color=c, lw=2, label=v)
    ax.axvline(0.5, color='0.4', ls=':', lw=1)
    ax.set_xlabel(r'foreground-probability threshold $\tau$'); ax.set_ylabel('Dice (%)'); ax.set_xlim(0.03, 0.97)
    lo, hi = S.dice_mean.min() * 100, S.dice_mean.max() * 100; ax.set_ylim(np.floor(lo) - 1, np.ceil(hi) + 1)
    ax.text(0.505, np.floor(lo) - 0.8, r'argmax ($\tau$ = 0.5)', fontsize=8, color='0.3', va='bottom')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=8, columnspacing=1.4)
    fig.tight_layout(); fig.savefig(path, metadata=META); plt.close(fig)


def fig5(O, path):
    O = O.set_index('vendor').loc[VEND]; x = np.arange(6); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.bar(x - w / 2, O.asis_m * 100, w, yerr=O.asis_sd * 100, color='0.72', ecolor='black', capsize=3, label='teacher as is')
    ax.bar(x + w / 2, O.band5_m * 100, w, yerr=O.band5_sd * 100, color='#238b45', ecolor='black', capsize=3, label='±5 px band replaced by ground truth')
    for i, v in enumerate(VEND):
        ax.text(i + w / 2, (O.band5_m[v] + O.band5_sd[v]) * 100 + 0.6, f"+{O.gain[v] * 100:.1f}", ha='center', va='bottom', fontsize=8, color='#00441b', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(VEND); ax.set_ylim(70, 104); ax.set_yticks(range(70, 101, 5)); ax.set_ylabel('Dice (%)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, metadata=META); plt.close(fig)


def build(derived, out):
    E = os.path.join(derived, 'extra_s16'); os.makedirs(out, exist_ok=True); H = {}
    for name, fn, tab in (('Figure3.png', fig3, 'table_collapse_prostate_s16.csv'), ('Figure4.png', fig4, 'table_threshold_sweep_s16.csv'), ('Figure5.png', fig5, 'table_oracle_band_s16.csv')):
        T = pd.read_csv(os.path.join(E, tab)); assert sorted(T.vendor.unique()) == VEND, tab
        p = os.path.join(out, name); fn(T, p); H[name] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return H


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--derived', required=True); ap.add_argument('--out', required=True); ap.add_argument('--no_json', action='store_true')
    a = ap.parse_args(); H = build(a.derived, a.out)
    if not a.no_json: json.dump(H, open(os.path.join(a.derived, 'extra_s16', 'figures_s16.json'), 'w'), indent=1, sort_keys=True)
    for k, v in H.items(): print(f'[fig] {k} sha256 {v[:16]}')
