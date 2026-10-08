#!/usr/bin/env python3
"""make_medl_heads_s13.py — D-05 reader (S13). For every MEDL run under <sendback>/runs/<ds>/<run>/log.txt pair each
`[medl] net2_avg=A ensemble_avg=B` line with the `[epoch N] student_avg=... best_avg=... @ iter I` line that follows it,
keep the LAST evaluation per epoch (a resumed run re-evaluates), and write one row per run:
  best_net1 (must equal the log's best_avg — asserted), best_net2, best_ensemble, their epochs, and the net2 / ensemble
  values at net1's best epoch. Headline per D-05 = best_ensemble; net1/net2 are reported alongside.
Writes <out>/table_medl_heads.csv. Will be folded into make_derived_v05 (N-26); until then it lives in extra_s13/.
  python make_medl_heads_s13.py --sendback <tree> --out derived_v04_S12/extra_s13
"""
import argparse, csv, glob, os, re

RX_MEDL = re.compile(r'\[medl\] net2_avg=([\d.]+) ensemble_avg=([\d.]+)')
RX_EP = re.compile(r'\[epoch (\d+)\] student_avg=([\d.]+)\s+ema_avg=([\d.]+)\s+best_avg=([\d.]+) @ iter (\d+)')

def read_run(log):
    txt = open(log, errors='replace').read().splitlines()
    rows = {}; pending = None; last_best = None
    for line in txt:
        m = RX_MEDL.search(line)
        if m: pending = (float(m.group(1)), float(m.group(2))); continue
        e = RX_EP.search(line)
        if e:
            ep, s1, ema, best, bit = int(e.group(1)), float(e.group(2)), float(e.group(3)), float(e.group(4)), int(e.group(5))
            last_best = (best, bit)
            if pending is not None:
                rows[ep] = (s1, pending[0], pending[1])
            pending = None
    return rows, last_best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sendback', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); out = []
    for log in sorted(glob.glob(os.path.join(a.sendback, 'runs', '*', '*medl*', 'log.txt'))):
        ds, run = log.split(os.sep)[-3], log.split(os.sep)[-2]
        if run.startswith('v18smoke'): continue
        rows, last_best = read_run(log)
        assert rows and last_best, log
        ep1 = max(rows, key=lambda k: rows[k][0]); ep2 = max(rows, key=lambda k: rows[k][1]); epE = max(rows, key=lambda k: rows[k][2])
        best1 = rows[ep1][0]
        assert abs(best1 - last_best[0]) < 5e-5, (run, best1, last_best)   # net1 best reproduces the log's own best_avg
        out.append([ds, run, len(rows), f'{best1:.4f}', ep1 * 500, f'{rows[ep2][1]:.4f}', ep2 * 500, f'{rows[epE][2]:.4f}', epE * 500,
                    f'{rows[ep1][1]:.4f}', f'{rows[ep1][2]:.4f}'])
        print(f'[medl-heads] {run}: net1 {best1:.4f}@{ep1*500}  net2 {rows[ep2][1]:.4f}@{ep2*500}  ensemble {rows[epE][2]:.4f}@{epE*500}  (at net1 best: net2 {rows[ep1][1]:.4f}, ens {rows[ep1][2]:.4f}; {len(rows)} evals)')
    with open(os.path.join(a.out, 'table_medl_heads.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset', 'run', 'n_evals', 'best_net1', 'iter_net1', 'best_net2', 'iter_net2', 'best_ensemble', 'iter_ensemble', 'net2_at_net1_best', 'ensemble_at_net1_best']); w.writerows(out)
    print(f'[medl-heads] wrote {os.path.join(a.out, "table_medl_heads.csv")} ({len(out)} runs)')

if __name__ == '__main__':
    main()
