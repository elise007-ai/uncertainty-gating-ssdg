#!/usr/bin/env python3
"""make_extra_s14.py -- S14 readers for writing blocks 5 (R1 #8: deep ensembles, MC-dropout signals),
8 (E-09: reproducible spectral measures) and 6 (R1 #9: FDA beta / p sweep).

  python make_extra_s14.py --derived <BSPC_results/derived/derived_v04_S12> --returns <BSPC_results/RETURNS> [--out <dir>]

Reads the authoritative derived tables (make_derived_v04.py output) plus three kinds of raw return that v04
does not tabulate: the ENSEMBLE3 JSON metadata (member provenance, N-28), spectral_distances.json (pairwise
matrices, PCA coordinates) and the run_config.json / split.json of the Phase-C runs (controlled-sweep check).
Writes <derived>/extra_s14/:
  table_ensemble_s14.csv, table_mcdropout_signal_s14.csv, table_fda_sweep_s14.csv, table_spectral_s14.csv,
  facts_s14.json (every scalar the S14 text quotes) and make_extra_s14.console.log.
Nothing is typed by hand (R-14, R-25); every qualitative statement of the S14 text is an assertion here
(protocol 9), so a change in the data stops the script instead of silently changing a sentence.
"""
import argparse, csv, glob, itertools, json, os, statistics as st, sys

VENDORS = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
LABELED = 'RUNMC'                       # Prostate labelled domain 5 (HANDOFF 1; run_config lb_domain = 5)
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)


def rd(path):
    return list(csv.DictReader(open(path, newline='')))


def latest(returns, rel):
    """Overlay semantics of the Results tier: the copy in the latest RETURNS/S<NN>/ wins."""
    m = sorted(glob.glob(os.path.join(returns, 'S*', rel)))
    assert m, f'not in any return: {rel}'
    return m[-1]


def flat(d, p=''):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, p + k + '.'))
        else: o[p + k] = v
    return o


def spearman_exact(x, y):
    """Spearman rho and exact one-sided p (P[rho >= observed] over all n! orderings); no ties allowed."""
    n = len(x); assert len(set(x)) == n and len(set(y)) == n, 'ties'
    rk = lambda v: [sorted(v).index(a) + 1 for a in v]
    rx, ry = rk(x), rk(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    cnt = tot = 0
    for perm in itertools.permutations(range(1, n + 1)):
        tot += 1
        cnt += sum((a - b) ** 2 for a, b in zip(rx, perm)) <= d2
    return rho, cnt / tot, d2


# ------------------------------------------------------------------------------------------------------
def ensembles(D, R, F, out):
    bs = rd(os.path.join(D, 'table_band_summary.csv')); bg = rd(os.path.join(D, 'table_band_generality.csv'))
    oc = rd(os.path.join(D, 'table_oracle_ceiling.csv')); cal = rd(os.path.join(D, 'table_calibration.csv'))
    ENS = 'ENSEMBLE3_split1337_fixmatch'; rows = []; F['ens'] = {}

    def S(ds, run, sig, col):
        r = [x for x in bs if x['dataset'] == ds and x['run'] == run and x['signal'] == sig and x['which'] == 'ema']
        assert len(r) == 1, (ds, run, sig, len(r)); return float(r[0][col])

    def mean_bg(ds, run, col):
        v = [float(x[col]) for x in bg if x['dataset'] == ds and x['run'] == run and x['signal'] == 'conf']
        assert v; return st.mean(v)

    for ds in ('prostate', 'fundus'):
        members = [f'v18B_{ds}_fixmatch_split1337_init{i}' for i in (1337, 43, 44)]
        # --- provenance (N-28): the ensemble JSON's members are exactly the three member checkpoints
        J = json.load(open(latest(R, f'diag_out/{ds}__{ENS}__ema.json')))['meta']
        shas = [J['ckpt_sha256']] + [e['sha256'] for e in J.get('ensemble_members', [])]
        msha = [json.load(open(latest(R, f'diag_out/{ds}__{m}__ema.json')))['meta']['ckpt_sha256'] for m in members]
        assert shas == msha, f'{ds}: ensemble members are not the three init checkpoints'
        assert J['which'] == 'ema' and int(J['args'].get('mc_T', 0)) == 0
        n_mem = len(shas); assert n_mem == 3
        say(f'[ens] {ds}: {n_mem} members verified by sha256 (meta.n_models = {J["n_models"]} is written before the '
            f'extra members are appended: stale by construction, N-28)')
        # --- harness identities (N-28): entropy == ens_entropy; K = 2 -> entropy is monotone in confidence
        for col in ('auroc_all_mean', 'auroc_edt5_mean', 'range80_edt5_mean'):
            assert S(ds, ENS, 'entropy', col) == S(ds, ENS, 'ens_entropy', col), (ds, col)
        gap = abs(S(ds, ENS, 'conf', 'auroc_edt5_mean') - S(ds, ENS, 'entropy', 'auroc_edt5_mean'))
        assert gap < 5e-4, (ds, gap)
        per_v = [(x['domain_name'], float(x['range80_edt5'])) for x in bg
                 if x['dataset'] == ds and x['run'] == ENS and x['signal'] == 'conf']
        f = {'n_members': n_mem, 'n_domains': len(per_v),
             'conf_all': S(ds, ENS, 'conf', 'auroc_all_mean'), 'conf_band': S(ds, ENS, 'conf', 'auroc_edt5_mean'),
             'mi_band': S(ds, ENS, 'ens_mi', 'auroc_edt5_mean'), 'var_band': S(ds, ENS, 'ens_var', 'auroc_edt5_mean'),
             'conf_range_all': S(ds, ENS, 'conf', 'range80_all_mean'), 'conf_range_band': S(ds, ENS, 'conf', 'range80_edt5_mean'),
             'ent_range_band': S(ds, ENS, 'entropy', 'range80_edt5_mean'), 'conf_vs_entropy_band_gap': gap,
             'conf_range_band_per_vendor': per_v,
             'n_vendors_within_0.01_of_third': sum(abs(v - 1 / 3) <= 0.01 for _, v in per_v),
             'err_band': mean_bg(ds, ENS, 'error_rate_in_edt5'), 'err_all': mean_bg(ds, ENS, 'error_rate'),
             'mem_conf_band': [S(ds, m, 'conf', 'auroc_edt5_mean') for m in members],
             'mem_conf_all': [S(ds, m, 'conf', 'auroc_all_mean') for m in members],
             'mem_conf_range_band': [S(ds, m, 'conf', 'range80_edt5_mean') for m in members],
             'mem_ent_range_band': [S(ds, m, 'entropy', 'range80_edt5_mean') for m in members],
             'mem_err_band': [mean_bg(ds, m, 'error_rate_in_edt5') for m in members]}
        f['gain_band_min'] = f['conf_band'] - max(f['mem_conf_band']); f['gain_band_max'] = f['conf_band'] - min(f['mem_conf_band'])
        # the statements the text makes
        assert f['gain_band_min'] > 0.02 and f['gain_band_max'] < 0.04, (ds, f['gain_band_min'], f['gain_band_max'])
        assert all(0.009 <= v <= 0.011 for v in f['mem_ent_range_band']), f['mem_ent_range_band']
        assert all(v < 2e-3 for v in f['mem_conf_range_band']) and f['conf_range_band'] > 0.2
        assert f['conf_all'] - f['conf_band'] > 0.2                       # band ranking far below all-pixel ranking
        F['ens'][ds] = f
        for unit, run in [('ENSEMBLE3', ENS)] + list(zip(('init1337', 'init43', 'init44'), members)):
            dice = st.mean(float(x['dice_asis']) for x in oc if x['dataset'] == ds and x['run'] == run)
            ece = st.mean(float(x['ece_raw']) for x in cal if x['dataset'] == ds and x['run'] == run)
            rows.append({'dataset': ds, 'unit': unit, 'run': run, 'n_domains': len(per_v),
                         'conf_auroc_all': f"{S(ds, run, 'conf', 'auroc_all_mean'):.4f}",
                         'conf_auroc_edt5': f"{S(ds, run, 'conf', 'auroc_edt5_mean'):.4f}",
                         'conf_range80_all': f"{S(ds, run, 'conf', 'range80_all_mean'):.3e}",
                         'conf_range80_edt5': f"{S(ds, run, 'conf', 'range80_edt5_mean'):.3e}",
                         'entropy_range80_edt5': f"{S(ds, run, 'entropy', 'range80_edt5_mean'):.3e}",
                         'ens_mi_auroc_edt5': f"{S(ds, run, 'ens_mi', 'auroc_edt5_mean'):.4f}" if unit == 'ENSEMBLE3' else '',
                         'ens_var_auroc_edt5': f"{S(ds, run, 'ens_var', 'auroc_edt5_mean'):.4f}" if unit == 'ENSEMBLE3' else '',
                         'error_rate_all': f"{mean_bg(ds, run, 'error_rate'):.5f}",
                         'error_rate_in_edt5': f"{mean_bg(ds, run, 'error_rate_in_edt5'):.4f}",
                         'harness_dice_asis': f'{dice:.4f}', 'ece_raw': f'{ece:.4f}'})
    assert F['ens']['prostate']['n_vendors_within_0.01_of_third'] == 5 and F['ens']['prostate']['n_domains'] == 6
    # Prostate: MI is level with a member's confidence (within 0.002 of the best member), not above the ensemble confidence
    p = F['ens']['prostate']; assert abs(p['mi_band'] - max(p['mem_conf_band'])) < 0.002 and p['mi_band'] < p['conf_band']
    assert F['ens']['fundus']['mi_band'] < F['ens']['fundus']['conf_band']
    w = csv.DictWriter(open(os.path.join(out, 'table_ensemble_s14.csv'), 'w', newline=''), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # --- MC-dropout teachers (the gated runs of Table baselines), signals measured with --mc_T 8
    mrows = []; F['mc'] = {}
    for ds in ('prostate', 'fundus'):
        runs = [f'v18B_{ds}_mcdropout2_s{s}' for s in (1337, 42, 2024)]
        g = lambda sig, col: [S(ds, r, sig, col) for r in runs]
        for r in runs:                                              # the MC signals were measured with eight stochastic passes
            assert int(json.load(open(latest(R, f'diag_out/{ds}__{r}__ema.json')))['meta']['args']['mc_T']) == 8, r
        F['mc'][ds] = {'ent_band': g('mc_entropy', 'auroc_edt5_mean'), 'ent_range_band': g('mc_entropy', 'range80_edt5_mean'),
                       'mi_band': g('mc_mi', 'auroc_edt5_mean'), 'mi_range_band': g('mc_mi', 'range80_edt5_mean')}
        assert all(0.011 <= v <= 0.018 for v in F['mc'][ds]['ent_range_band']), F['mc'][ds]['ent_range_band']
        assert max(F['mc'][ds]['ent_band']) < F['ens'][ds]['conf_band']            # MC entropy never ranks better than the ensemble
        assert max(F['mc'][ds]['ent_band']) <= max(F['ens'][ds]['mem_conf_band']) + 1e-9   # ... nor than the best single member
        for r in runs:
            mrows.append({'dataset': ds, 'run': r, 'mc_entropy_auroc_edt5': f"{S(ds, r, 'mc_entropy', 'auroc_edt5_mean'):.4f}",
                          'mc_entropy_range80_edt5': f"{S(ds, r, 'mc_entropy', 'range80_edt5_mean'):.3e}",
                          'mc_mi_auroc_edt5': f"{S(ds, r, 'mc_mi', 'auroc_edt5_mean'):.4f}",
                          'mc_mi_range80_edt5': f"{S(ds, r, 'mc_mi', 'range80_edt5_mean'):.3e}"})
    w = csv.DictWriter(open(os.path.join(out, 'table_mcdropout_signal_s14.csv'), 'w', newline=''), fieldnames=list(mrows[0])); w.writeheader(); w.writerows(mrows)
    say('[ens] prostate conf band', f"{p['conf_band']:.4f}", 'members', [f'{v:.4f}' for v in p['mem_conf_band']], 'MI', f"{p['mi_band']:.4f}",
        '| fundus', f"{F['ens']['fundus']['conf_band']:.4f}", [f'{v:.4f}' for v in F['ens']['fundus']['mem_conf_band']])


# ------------------------------------------------------------------------------------------------------
def fda(D, R, F, out):
    runs = {r['run']: r for r in rd(os.path.join(D, 'table_runs.csv')) if r['dataset'] == 'prostate'}
    DEF = 'v18B_prostate_fixmatch_split1337_init1337'
    inits = [DEF, 'v18B_prostate_fixmatch_split1337_init43', 'v18B_prostate_fixmatch_split1337_init44']
    v04 = rd(os.path.join(D, 'table_fda_sweep.csv')); assert len(v04) == 7
    reg_ref = 0.807287          # D-06 as registered (HANDOFF 3, Phase C): the S01 threshold-gate s1337 BIDMC Dice; make_derived_v02.py constant
    for r in v04:               # ... and it is the reference the v04 table's delta column was computed against (4-decimal rounding)
        assert abs(float(r['BIDMC_dice']) - float(r['BIDMC_delta_vs_default']) - reg_ref) < 1.5e-4, r['run']
    dom = lambda r: [float(runs[r][f'dom{i}_dice']) for i in range(1, 7)]
    for r in inits + [x['run'] for x in v04]:
        assert runs[r]['complete'] == '1' and runs[r]['quotable'] == '1', r
    # --- controlled-sweep check (R-33 spirit): each Phase-C run differs from the default only in the swept flag
    base = flat(json.load(open(latest(R, f'runs/prostate/{DEF}/run_config.json'))))
    assert base['args.split_seed'] == 1337 and base['args.seed'] == 1337 and base['args.lb_domain'] == 5 and base['args.lb_num'] == 20
    assert base['args.ff_fda_beta'] == 0.05 and base['args.ff_fda_prob'] == 0.5 and base['args.method'] == 'fixmatch'
    split_ref = json.load(open(latest(R, 'runs/prostate/v18A_prostate_daep_default_s1337/split.json')))
    assert split_ref['split_seed'] == 1337 and split_ref['lb_domain'] == 5 and split_ref['lb_num'] == 20
    ALLOWED = {'args.save_name', 'args.gpu', 'argv', 'snapshot_path', 'script_sha256', 'args.split_seed',
               'args.consistency_ramp_frac', 'args.selftest_ramp'}
    shas = set(); rows = []
    for x in v04:
        c = flat(json.load(open(latest(R, f"runs/prostate/{x['run']}/run_config.json"))))
        key = 'args.ff_fda_beta' if x['param'] == 'beta' else 'args.ff_fda_prob'
        diff = {k for k in set(base) | set(c) if base.get(k) != c.get(k)}
        assert key in diff and abs(float(c[key]) - float(x['value'])) < 1e-9, (x['run'], key)
        assert diff - ALLOWED == {key}, (x['run'], sorted(diff - ALLOWED))
        # split: --split_seed unset (-1) resolves to --seed (train script: split_seed = args.seed if args.split_seed < 0)
        assert c['args.split_seed'] == -1 and c['args.seed'] == 1337
        assert c.get('args.consistency_ramp_frac') == -1.0                     # v9 flag at its default = whole-run ramp (E-28)
        sp = json.load(open(latest(R, f"runs/prostate/{x['run']}/split.json")))
        assert all(sp[k] == split_ref[k] for k in ('split_seed', 'lb_domain', 'lb_num', 'lb_idxs')), x['run']
        shas.add(c['script_sha256'][:8])
    assert len(shas) == 1 and base['script_sha256'][:8] not in shas
    say(f'[fda] 7 Phase-C configs differ from {DEF} only in the swept flag; same 20-slice labelled split (seed 1337); '
        f'training script {shas.pop()} (tools v9, ramp flag at default) vs {base["script_sha256"][:8]} for the default runs')
    d_avg = float(runs[DEF]['best_avg']); d_dom = dom(DEF)
    spread_avg = [float(runs[r]['best_avg']) for r in inits]; spread_b = [dom(r)[0] for r in inits]
    f = {'default_avg': d_avg, 'default_bidmc': d_dom[0], 'init_avg': spread_avg, 'init_bidmc': spread_b,
         'init_avg_range': max(spread_avg) - min(spread_avg), 'init_bidmc_range': max(spread_b) - min(spread_b),
         'registered_ref_bidmc': reg_ref, 'settings': {}}
    for lab, r in zip(('init1337', 'init43', 'init44'), inits):
        rows.append({'kind': 'default', 'param': 'init', 'value': lab.replace('init', ''), 'run': r, 'best_avg': runs[r]['best_avg'],
                     **{v: f'{d:.4f}' for v, d in zip(VENDORS, dom(r))},
                     'd_avg_vs_default': f"{float(runs[r]['best_avg']) - d_avg:+.4f}", 'd_bidmc_vs_default': f'{dom(r)[0] - d_dom[0]:+.4f}'})
    for x in v04:
        r = x['run']; a = float(runs[r]['best_avg']); b = dom(r)
        assert abs(a - float(x['best_avg_dice'])) < 1e-9 and abs(b[0] - float(x['BIDMC_dice'])) < 1e-9
        f['settings'][f"{x['param']}={float(x['value']):g}"] = {'avg': a, 'bidmc': b[0], 'd_avg': a - d_avg, 'd_bidmc': b[0] - d_dom[0],
                                                           'd_bidmc_registered': b[0] - reg_ref, 'vendors': b}
        rows.append({'kind': 'sweep', 'param': x['param'], 'value': f"{float(x['value']):g}", 'run': r, 'best_avg': runs[r]['best_avg'],
                     **{v: f'{d:.4f}' for v, d in zip(VENDORS, b)},
                     'd_avg_vs_default': f'{a - d_avg:+.4f}', 'd_bidmc_vs_default': f'{b[0] - d_dom[0]:+.4f}'})
    w = csv.DictWriter(open(os.path.join(out, 'table_fda_sweep_s14.csv'), 'w', newline=''), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    S_ = f['settings']; on = S_['pfda=1']; rest = {k: v for k, v in S_.items() if k != 'pfda=1'}
    # registered rule D-06 (fixed before the runs): BIDMC "responds" iff a setting exceeds the default by > 0.015
    f['n_exceed_rule_registered_ref'] = sum(v['d_bidmc_registered'] > 0.015 for v in S_.values())
    f['n_exceed_rule_codematched_ref'] = sum(v['d_bidmc'] > 0.015 for v in S_.values())
    assert f['n_exceed_rule_registered_ref'] == 6 and f['n_exceed_rule_codematched_ref'] == 6
    # post-hoc yardstick (labelled as such in the text): the three same-configuration initialisations
    assert f['default_bidmc'] == min(spread_b)                      # the default run is the lowest of its three inits on BIDMC
    f['init_bidmc_mean'] = st.mean(spread_b); f['init_bidmc_sd'] = st.stdev(spread_b)
    f['init_avg_mean'] = st.mean(spread_avg); f['init_avg_sd'] = st.stdev(spread_avg)
    for v in rest.values():
        v['bidmc_minus_init_mean'] = v['bidmc'] - f['init_bidmc_mean']; v['bidmc_z_init'] = v['bidmc_minus_init_mean'] / f['init_bidmc_sd']
    zs = {k: v['bidmc_z_init'] for k, v in rest.items()}
    f['n_within_1sd_of_init_mean'] = sum(abs(z) <= 1 for z in zs.values()); assert f['n_within_1sd_of_init_mean'] == 4, zs
    f['max_z_setting'] = max(zs, key=zs.get); f['max_z'] = zs[f['max_z_setting']]
    assert f['max_z_setting'] == 'beta=0.02' and 1.7 < f['max_z'] < 1.9 and all(abs(z) < 2 for z in zs.values()), zs
    f['bidmc_minus_init_mean_min'] = min(v['bidmc_minus_init_mean'] for v in rest.values())
    f['bidmc_minus_init_mean_max'] = max(v['bidmc_minus_init_mean'] for v in rest.values())
    # E-44: ADJUDICATION_S11 5.4 said every response except beta 0.02 is inside the init range; beta 0.10 is above its top too
    f['settings_above_init_top'] = {k: v['bidmc'] - max(spread_b) for k, v in rest.items() if v['bidmc'] > max(spread_b)}
    assert sorted(f['settings_above_init_top']) == ['beta=0.02', 'beta=0.1']
    f['bidmc_gain_min'] = min(v['d_bidmc'] for v in rest.values()); f['bidmc_gain_max'] = max(v['d_bidmc'] for v in rest.values())
    # average Dice: shallow optimum
    within = sorted(k for k, v in rest.items() if abs(v['d_avg']) <= 0.005)
    assert within == ['beta=0.02', 'beta=0.1', 'pfda=0.25', 'pfda=0.75'], within
    assert rest['beta=0.01']['d_avg'] < -0.005 and rest['beta=0.2']['d_avg'] < -0.015 and on['d_avg'] < -0.04
    assert max(abs(v['d_avg']) for k, v in rest.items() if k in within) > f['init_avg_range']     # the sweep moves the average more than init does
    assert max(v['bidmc'] for v in S_.values()) < 0.881                                        # nothing reaches SynFoC-UNet's published 88.1
    # always-on FDA, per vendor, against the default run and the per-vendor init range
    f['alwayson_change'] = {v: on['vendors'][i] - d_dom[i] for i, v in enumerate(VENDORS)}
    f['init_range_per_vendor'] = {v: max(dom(r)[i] for r in inits) - min(dom(r)[i] for r in inits) for i, v in enumerate(VENDORS)}
    F['fda'] = f
    say('[fda] default', d_avg, d_dom[0], '| init ranges avg/BIDMC', f"{f['init_avg_range']:.4f}", f"{f['init_bidmc_range']:.4f}",
        '| always-on change', {k: round(v, 4) for k, v in f['alwayson_change'].items()})


# ------------------------------------------------------------------------------------------------------
def spectral(D, R, F, out):
    loo = {r['site']: r for r in rd(os.path.join(D, 'table_spectral_loo.csv'))}; assert list(loo) == VENDORS
    J = json.load(open(latest(R, 'spectral_out/spectral_distances.json')))
    L = J['leave_one_out']; betas = ['0.01', '0.02', '0.05', '0.1', '0.2']; zcol = ['z_b0.01', 'z_b0.02', 'z_b0.05', 'z_b0.10', 'z_b0.20']
    for s in VENDORS:                                               # derived table == raw JSON
        for b, c in zip(betas, zcol):
            assert abs(L[s]['zscore_fda_scalar'][b]['z'] - float(loo[s][c])) < 2e-3, (s, b)
        assert abs(L[s]['swd_vs_others']['point'] - float(loo[s]['swd'])) < 1e-4 and str(L[s]['inside_pca2_hull_of_others']) == loo[s]['inside_hull']
    pw = lambda key, a, b: J['pairwise'][key][f'{a}|{b}' if a < b else f'{b}|{a}']
    pc = J['pca2']['coords']; ss1 = sum(pc[s][0] ** 2 for s in VENDORS); ss2 = sum(pc[s][1] ** 2 for s in VENDORS)
    f = {'n_slices_total': sum(int(loo[s]['n_slices']) for s in VENDORS), 'n_slices': {s: int(loo[s]['n_slices']) for s in VENDORS},
         'pc1_share': ss1 / (ss1 + ss2) * J['pca2']['var_explained_2'], 'pc1': {s: pc[s][0] for s in VENDORS},
         'z': {s: float(loo[s]['z_b0.05']) for s in VENDORS}, 'z_bidmc_all_beta': [float(loo['BIDMC'][c]) for c in zcol],
         'swd': {s: [float(loo[s][k]) for k in ('swd', 'swd_lo95', 'swd_hi95')] for s in VENDORS},
         'fd': {s: [float(loo[s][k]) for k in ('frechet', 'fd_lo95', 'fd_hi95')] for s in VENDORS},
         'inside_hull': {s: loo[s]['inside_hull'] == 'True' for s in VENDORS},
         'swd_to_labeled': {s: (0.0 if s == LABELED else pw('swd', s, LABELED)) for s in VENDORS},
         'fd_to_labeled': {s: (0.0 if s == LABELED else pw('frechet', s, LABELED)) for s in VENDORS},
         'n_boot': J['meta']['n_boot'], 'grid': J['meta']['grid']}
    assert J['meta']['n_slices'] == f['n_slices'] and J['meta']['phases'] == ['train', 'test']
    # the statements the text makes
    for key in ('_rank_swd_vs_others', '_rank_frechet_vs_others', '_rank_frechet_low_vs_others', '_rank_swd_low_vs_others'):
        assert L[key]['BIDMC'] == 1, key
    for c in zcol:                                                  # BIDMC has the largest |z| at every beta
        assert all(abs(float(loo['BIDMC'][c])) > abs(float(loo[s][c])) for s in VENDORS if s != 'BIDMC'), c
    oth = [s for s in VENDORS if s != 'BIDMC']
    assert f['swd']['BIDMC'][1] > max(f['swd'][s][2] for s in oth) and f['fd']['BIDMC'][1] > max(f['fd'][s][2] for s in oth)   # CIs do not overlap
    f['swd_ratio_to_next'] = f['swd']['BIDMC'][0] / max(f['swd'][s][0] for s in oth); f['fd_ratio_to_next'] = f['fd']['BIDMC'][0] / max(f['fd'][s][0] for s in oth)
    f['n_outside_hull'] = sum(not v for v in f['inside_hull'].values()); assert f['n_outside_hull'] == 3 and not f['inside_hull']['BIDMC']
    for key in ('swd', 'frechet'):                                  # the largest pairwise entry is BIDMC - labelled vendor
        assert max(J['pairwise'][key], key=J['pairwise'][key].get) == 'BIDMC|RUNMC', key
    order = sorted(VENDORS, key=lambda s: f['pc1'][s]); f['pc1_order'] = order
    assert order[0] == 'BIDMC' and order[-1] == LABELED and set(order[1:3]) == {'BMC', 'HK'} and set(order[3:]) == {'I2CVB', 'UCL', 'RUNMC'}
    assert f['z'][LABELED] == max(f['z'].values()) and f['z']['BIDMC'] == min(f['z'].values())
    # D-18 (post hoc, one run): per-vendor change under always-on FDA vs spectral distance from the labelled vendor
    ch = F['fda']['alwayson_change']
    rho6, p6, _ = spearman_exact([f['swd_to_labeled'][s] for s in VENDORS], [-ch[s] for s in VENDORS])
    v5 = [s for s in VENDORS if s != LABELED]
    rho5, p5, _ = spearman_exact([f['swd_to_labeled'][s] for s in v5], [-ch[s] for s in v5])
    rho6f, p6f, _ = spearman_exact([f['fd_to_labeled'][s] for s in VENDORS], [-ch[s] for s in VENDORS])
    f['alwayson_vs_distance'] = {'rho_six_vendors_swd': rho6, 'p_one_sided_exact_six': p6, 'rho_five_unlabeled_vendors_swd': rho5,
                                 'p_one_sided_exact_five': p5, 'rho_six_vendors_frechet': rho6f, 'p_one_sided_exact_six_frechet': p6f}
    assert abs(rho6 - rho6f) < 1e-12                                # same ordering under either distance
    near = sorted(VENDORS, key=lambda s: f['swd_to_labeled'][s])[:3]; assert set(near) == {'RUNMC', 'UCL', 'I2CVB'}
    f['alwayson_near_three'] = [ch[s] for s in near]
    assert ch['BIDMC'] < ch['HK'] < ch['BMC'] < min(f['alwayson_near_three'])
    F['spectral'] = f
    rows = [{'vendor': s, 'labeled': int(s == LABELED), 'n_slices': f['n_slices'][s], 'z_fda_band_b0.05': f"{f['z'][s]:+.3f}",
             'swd': f"{f['swd'][s][0]:.4f}", 'swd_lo95': f"{f['swd'][s][1]:.4f}", 'swd_hi95': f"{f['swd'][s][2]:.4f}",
             'frechet': f"{f['fd'][s][0]:.4f}", 'fd_lo95': f"{f['fd'][s][1]:.4f}", 'fd_hi95': f"{f['fd'][s][2]:.4f}",
             'pc1': f"{f['pc1'][s]:+.3f}", 'inside_hull_of_others': int(f['inside_hull'][s]),
             'swd_to_labeled': f"{f['swd_to_labeled'][s]:.4f}", 'frechet_to_labeled': f"{f['fd_to_labeled'][s]:.4f}",
             'dice_change_alwayson_fda': f'{ch[s]:+.4f}', 'dice_init_range': f"{F['fda']['init_range_per_vendor'][s]:.4f}"} for s in VENDORS]
    w = csv.DictWriter(open(os.path.join(out, 'table_spectral_s14.csv'), 'w', newline=''), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    say('[spectral] BIDMC z', f['z']['BIDMC'], 'SWD', f['swd']['BIDMC'], 'FD', f['fd']['BIDMC'], '| PC1 share', f"{f['pc1_share']:.4f}",
        '| rho6', f'{rho6:.3f}', 'p', f'{p6:.4f}', '| rho5', f'{rho5:.3f}', 'p', f'{p5:.4f}')


def build(derived, returns, out):
    os.makedirs(out, exist_ok=True); F = {}
    ensembles(derived, returns, F, out); fda(derived, returns, F, out); spectral(derived, returns, F, out)
    json.dump(F, open(os.path.join(out, 'facts_s14.json'), 'w'), indent=1, sort_keys=True)
    say(f'[done] {out}: 4 tables + facts_s14.json; all assertions passed')
    open(os.path.join(out, 'make_extra_s14.console.log'), 'w').write('\n'.join(LOG) + '\n')
    return F


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', required=True); ap.add_argument('--returns', required=True); ap.add_argument('--out', default='')
    a = ap.parse_args()
    build(a.derived, a.returns, a.out or os.path.join(a.derived, 'extra_s14'))
