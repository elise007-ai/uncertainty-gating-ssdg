#!/usr/bin/env python3
"""make_docs_tables_s07.py — renders the markdown tables quoted in DOCS/adjudications/ADJUDICATION_S06.md
and HANDOFF from derived_v02/ (R-14: ledger tables are script-generated, never typed).
  python make_docs_tables_s07.py --derived derived_v02
"""
import argparse, csv, json, os

def rd(p):
    return list(csv.DictReader(open(p)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--derived', default='derived_v02'); a = ap.parse_args()
    D = a.derived; A = json.load(open(os.path.join(D, 'assertions.json')))
    print("### Phase A — evidential configuration sweep (Prostate, DAEP gate, seed 1337, seed-split; source table_evid_sweep.csv, u_epi from EMA teacher, 6-vendor means)")
    print("| config | best avg Dice | BIDMC Dice | u_epi AUROC all | u_epi AUROC EDT-5px | range80 (all) | margin to 0.70 | Chebyshev-5 | rel-5% | rule |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    pc = A['phaseA_rule']['per_config']
    for r in rd(os.path.join(D, 'table_evid_sweep.csv')):
        v = pc.get(r['config'], {})
        print(f"| {r['config']} | {r['best_avg_dice']} | {r['BIDMC_dice']} | {r['u_epi_auroc_all']} | {r['u_epi_auroc_edt5']} | {r['u_epi_range80_all']} | {r['margin_edt5_vs_0.70']} | {r['u_epi_auroc_cheb5']} | {r['u_epi_auroc_rel5pct']} | {'collapse persists' if v.get('collapse_persists') else 'COUNTER-EXAMPLE'} |")
    pa = A['phaseA_rule']
    print(f"\nRule: {pa['rule']}. Result: {pa['n_collapse_persists']}/{pa['n_configs']} configurations satisfy it; counter-examples {pa['counter_examples']} with margins {pa['counter_example_margins']}; max band AUROC {pa['max_band5_auroc']}, max range80 {pa['max_range80']:.2e}; range condition met by all: {pa['range_condition_met_by_all']}. Post-hoc (NOT pre-registered) sensitivity: {pa['post_hoc_sensitivity_NOT_preregistered']}.")
    print("\n### Phase B so far (source table_baselines.csv; paired vs the seed-split threshold gate)")
    print("| dataset | method | n | mean Dice | std | threshold ref | mean Δ | per-seed Δ (1337;42;2024) | paired t | p | sign-test p | beats rule | code-matched ref |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rd(os.path.join(D, 'table_baselines.csv')):
        print(f"| {r['dataset']} | {r['method']} | {r['n_seeds']} | {r['mean_dice']} | {r['std_dice']} | {r['threshold_ref_mean']} | {r['delta_vs_threshold']} | {r['per_seed_delta']} | {r['t_paired']} | {r['p_paired']} | {r['sign_test_p']} | {r['beats_threshold_rule']} | {r['reference_code_matched']} ({r['reference_sources'].split(';')[0]}) |")
    print("\n### v18 reproduction gate (source assertions.json)")
    for k, v in A['v18_reproduction_gate'].items():
        print(f"- {k}: v18 {v['v18_best_avg']:.6f} vs S01 {v['S01_reference']:.6f}, |Δ| = {v['abs_diff']:.4f}, within ±0.005: {v['pass_within_0.005']}, resumed: {v['resumed_at_iters'] or 'no'}")
    e = A['E16_signal_identity']
    print(f"\n### E-16 identity check (source table_signal_identity.csv): {e['n_holds']}/{e['n_checkpoints']} converged evidential checkpoints satisfy u_epi = K/(K-1)(1-p_max) within tolerance; fails: {e['fails']}")
    print("\n### Runs (source table_runs.csv; quotable = complete and not quarantined)")
    rows = rd(os.path.join(D, 'table_runs.csv'))
    q = [r for r in rows if r['quotable'] == '1']
    print(f"{len(rows)} run logs received; {len(q)} complete and quotable; incomplete: {[r['run'] for r in rows if r['complete']=='0']}; quarantined-complete: {[r['run'] for r in rows if r['complete']=='1' and r['quotable']=='0' and not r['run'].startswith('v18smoke')]}")
    print("| run | best avg | best iter | BIDMC | BMC | HK | I2CVB | RUNMC | UCL |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in q:
        if r['run'].startswith(('v18A_', 'v18B_')):
            print(f"| {r['run']} | {r['best_avg']} | {r['best_iter']} | {r['dom1_dice']} | {r['dom2_dice']} | {r['dom3_dice']} | {r['dom4_dice']} | {r['dom5_dice']} | {r['dom6_dice']} |")

if __name__ == '__main__':
    main()
