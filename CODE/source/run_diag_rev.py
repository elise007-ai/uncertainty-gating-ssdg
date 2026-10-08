#!/usr/bin/env python3
"""
run_diag_rev.py — phase D driver (inference-only diagnostics), run from
/opt/data/private/SynFoC/source after the training phases (or right away on
the existing checkpoints).

  python run_diag_rev.py inventory                 # lists every best_student.pth it can find
  python run_diag_rev.py spacing                   # builds prostate_spacing_index.json (D-1)
  python run_diag_rev.py spectral                  # spectral_out/ (R1 #9)
  python run_diag_rev.py harness --select "adaptive_cmp|v16_.*_s1337|v18|fixmatch_s|daep_s" --gpu 0
  python run_diag_rev.py ensemble --gpu 1          # same-split deep ensembles (R1 #8)

Every call tees to console_logs/<session>/diag_<cmd>.console.log; harness
outputs land in diag_out/<dataset>__<run_name>__<which>.json and are never
overwritten unless --force (provenance stays attached to the first run).
"""
import argparse, glob, json, os, re, subprocess, sys, time

SEARCH = ['../model/*/train_fourierfuse_v16/*/best_student.pth',
          '/opt/data/private/checkpoints/**/best_student.pth']
DS_FROM_PATH = [('/prostate/', 'prostate'), ('/fundus/', 'fundus'), ('/MNMS/', 'MNMS'), ('/BUSI/', 'BUSI'),
                ('prostate', 'prostate'), ('fundus', 'fundus'), ('mnms', 'MNMS'), ('busi', 'BUSI')]
NII_ROOT = '/opt/data/private/data/Prostate_nii'


def find_ckpts():
    seen, out = set(), []
    for pat in SEARCH:
        for f in sorted(glob.glob(pat, recursive=True)):
            f = os.path.abspath(f)
            if f in seen:
                continue
            seen.add(f)
            name = os.path.basename(os.path.dirname(f))
            ds = next((d for k, d in DS_FROM_PATH if k in f or k in name.lower()), None)
            out.append({'path': f, 'run': name, 'dataset': ds, 'size_mb': os.path.getsize(f) / 1e6,
                        'has_run_config': os.path.exists(os.path.join(os.path.dirname(f), 'run_config.json')),
                        'has_split': os.path.exists(os.path.join(os.path.dirname(f), 'split.json'))})
    return out


def sh(cmd, logf):
    print(f"$ {cmd}", flush=True)
    with open(logf, 'a') as f:
        f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')}\n$ {cmd}\n"); f.flush()
        rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"  -> rc={rc} (log: {logf})", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['inventory', 'spacing', 'spectral', 'harness', 'ensemble'])
    ap.add_argument('--session', default='S01'); ap.add_argument('--gpu', default='0')
    ap.add_argument('--select', default='adaptive_cmp|v16_(prostate|fundus|mnms|busi)_s1337$|v18|fixmatch_s|daep_s')
    ap.add_argument('--which', default='ema'); ap.add_argument('--force', action='store_true')
    ap.add_argument('--mc_T', type=int, default=8)
    a = ap.parse_args()
    logdir = os.path.join('console_logs', a.session); os.makedirs(logdir, exist_ok=True)
    os.makedirs('diag_out', exist_ok=True); os.makedirs('spectral_out', exist_ok=True)
    logf = os.path.join(logdir, f'diag_{a.cmd}.console.log')
    class _Tee:                                   # own stdout copy -> no shell tee/redirect needed
        def __init__(self, path): self.f = open(path, 'a', buffering=1); self.o = sys.stdout
        def write(self, x): self.o.write(x); self.f.write(x)
        def flush(self): self.o.flush(); self.f.flush()
    sys.stdout = _Tee(os.path.join(logdir, f'diag_{a.cmd}.stdout.log'))
    env = f"CUDA_VISIBLE_DEVICES={a.gpu} "
    if a.cmd == 'inventory':
        inv = find_ckpts()
        json.dump(inv, open('diag_out/checkpoint_inventory.json', 'w'), indent=1)
        for r in inv:
            print(f"{r['dataset'] or '?':8s} {r['run']:55s} {r['size_mb']:7.1f}MB cfg={int(r['has_run_config'])} split={int(r['has_split'])}")
        print(f"[inventory] {len(inv)} checkpoints -> diag_out/checkpoint_inventory.json")
        # Second listing: every .pth anywhere under the archive dir and the model tree,
        # so renamed/relocated checkpoints (e.g. the seed-split adaptive_cmp_prostate_*
        # runs, absent from the S01 inventory) can be found and mapped by hand.
        others = []
        for pat in ('/opt/data/private/checkpoints/**/*.pth', '/opt/data/private/checkpoints/**/*.pt',
                    '../model/**/*.pth', '../../checkpoints/**/*.pth'):
            for f in sorted(glob.glob(pat, recursive=True)):
                f = os.path.abspath(f)
                if os.path.basename(f) != 'best_student.pth':
                    others.append({'path': f, 'size_mb': os.path.getsize(f) / 1e6})
        json.dump(others, open('diag_out/checkpoint_inventory_other_pth.json', 'w'), indent=1)
        print(f"[inventory] {len(others)} other .pth files (not named best_student.pth):")
        for r in others[:200]:
            print(f"   {r['size_mb']:7.1f}MB  {r['path']}")
        for root in ('/opt/data/private/checkpoints', '../model'):
            print(f"[inventory] top-level of {root}: {sorted(os.listdir(root))[:60] if os.path.isdir(root) else 'MISSING'}")
    elif a.cmd == 'spacing':
        sh(f"python prostate_spacing_index.py --nii_root {NII_ROOT} --png_root ../../data/ProstateSlice "
           f"--out diag_out/prostate_spacing_index.json", logf)
    elif a.cmd == 'spectral':
        sh("python spectral_distance_analysis.py --data_root ../../data/ProstateSlice --out spectral_out", logf)
    elif a.cmd == 'harness':
        inv = find_ckpts(); rx = re.compile(a.select)
        sp = 'diag_out/prostate_spacing_index.json'
        n_run = n_skip = n_fail = 0
        for r in inv:
            if not r['dataset'] or not rx.search(r['run']):
                continue
            out = f"diag_out/{r['dataset']}__{r['run']}__{a.which}.json"
            if os.path.exists(out) and not a.force:
                n_skip += 1; continue
            extra = f" --spacing_index {sp}" if (r['dataset'] == 'prostate' and os.path.exists(sp)) else ''
            if 'mcdropout' in r['run'] or 'dropoutctl' in r['run']:   # S07: MC signals on the dropout-only control too
                extra += f" --mc_T {a.mc_T}"
            rc = sh(env + f"python diag_uncertainty_harness_v18.py --dataset {r['dataset']} --ckpt {r['path']} "
                    f"--which {a.which} --out {out}{extra}", logf)
            ok = rc == 0 and os.path.exists(out) and os.path.getsize(out) > 1000
            n_run += 1; n_fail += (not ok)
            print(f"  [{'OK' if ok else 'FAIL'}] {out}")
        print(f"[harness] ran {n_run} (failed {n_fail}), skipped {n_skip} existing")
        sys.exit(1 if n_fail else 0)
    elif a.cmd == 'ensemble':
        sp = 'diag_out/prostate_spacing_index.json'
        inv = {r['run']: r for r in find_ckpts()}
        for ds in ('prostate', 'fundus'):
            base = f'v18B_{ds}_fixmatch_split1337_init1337'
            if base not in inv:                      # fall back to the original S01 run if present
                base = f'adaptive_cmp_{ds}_fixmatch_s1337'
            members = [f'v18B_{ds}_fixmatch_split1337_init43', f'v18B_{ds}_fixmatch_split1337_init44']
            if base not in inv or any(m not in inv for m in members):
                print(f"[ensemble] {ds}: missing {[m for m in [base]+members if m not in inv]} -> skipped"); continue
            out = f"diag_out/{ds}__ENSEMBLE3_split1337_fixmatch__{a.which}.json"
            extra = f" --spacing_index {sp}" if (ds == 'prostate' and os.path.exists(sp)) else ''
            sh(env + f"python diag_uncertainty_harness_v18.py --dataset {ds} --ckpt {inv[base]['path']} "
               f"--ensemble_ckpts {inv[members[0]]['path']},{inv[members[1]]['path']} --which {a.which} "
               f"--out {out}{extra}", logf)


if __name__ == '__main__':
    main()
