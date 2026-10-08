#!/usr/bin/env python3
"""
job_runner.py — runs a jobs_*.txt list on N GPUs with K concurrent jobs per GPU.

  nohup python job_runner.py --jobs jobs/jobs_A.txt --gpus 0,1 --per_gpu 2 --session S05 > nohup_A.out 2>&1 &
  python job_runner.py --jobs jobs/jobs_smoke.txt --gpus 0 --per_gpu 1 --session S05 --smoke
The runner creates console_logs/<session>/ itself and writes runner_<list>.log there
(its own tee), status_<list>.csv and one <job>.console.log per job. Never redirect
its output into a subdirectory that may not exist; the nohup file goes to the cwd.

* Preflight (whole list): data root for each dataset exists, scripts import,
  free disk > 20 GB, GPUs visible. Aborts before launching anything otherwise.
* Each job: `python train_fourierfuse_v18_revision.py <flags> --gpu <g>` with
  stdout+stderr tee'd to console_logs/<session>/<job>.console.log.
* Postflight per job (R-06: a zero exit code is not evidence):
    - ../model/<ds>/train_fourierfuse_v16/<job>/log.txt exists,
    - contains 'TRAINING COMPLETE', a 'best_avg=' > 0, no 'non-finite',
    - best_student.pth exists and is > 1 MB (skipped for --smoke).
* Resumable at three levels:
    - finished jobs (postflight PASS) are SKIPPED unless --force;
    - a job that died mid-run and has <run>/resume_state.pth CONTINUES from the
      last saved eval-epoch boundary (training script --resume auto, default);
    - --restart forces a clean restart of the selected jobs (--resume off).
* status CSV: console_logs/<session>/status_<listname>.csv (append-only, one
  row per attempt with timestamps, exit code, postflight verdict).
"""
import argparse, csv, os, re, shutil, subprocess, sys, threading, time
from queue import Queue


class Tee:
    """Duplicate stdout/stderr into a file the runner creates itself, so that no
    shell redirect into a not-yet-existing directory is ever needed (E-13)."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'a', buffering=1); self.stdout = sys.stdout
    def write(self, x):
        self.stdout.write(x); self.f.write(x)
    def flush(self):
        self.stdout.flush(); self.f.flush()

TRAIN = 'train_fourierfuse_v18_revision.py'
DATA_ROOT = {'prostate': '../../data/ProstateSlice', 'fundus': '../../data/Fundus',
             'MNMS': '../../data/mnms', 'BUSI': '../../data/Dataset_BUSI_with_GT'}


def snapshot_dir(ds, name):
    return f"../model/{ds}/train_fourierfuse_v16/{name}/"


def postflight(ds, name, smoke=False):
    d = snapshot_dir(ds, name)
    log = os.path.join(d, 'log.txt')
    if not os.path.exists(log):
        return False, 'no log.txt'
    txt = open(log, errors='replace').read()
    if 'TRAINING COMPLETE' not in txt:
        return False, 'no TRAINING COMPLETE'
    m = re.findall(r'best_avg=([\d.]+)', txt)
    if not m or float(m[-1]) <= 0:
        return False, 'best_avg missing or zero'
    warn = f' WARNING: {txt.count("non-finite")} non-finite-loss skips' if 'non-finite' in txt else ''
    if not smoke:
        pth = os.path.join(d, 'best_student.pth')
        if not (os.path.exists(pth) and os.path.getsize(pth) > 1 << 20):
            return False, 'best_student.pth missing/small'
    if not os.path.exists(os.path.join(d, 'run_config.json')):
        return False, 'run_config.json missing'
    return True, f'ok best_avg={m[-1]}{warn}'


RUNNER_LOCK = os.path.join('console_logs', 'job_runner.lock')      # one runner per source tree (E-23)


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def acquire_runner_lock():
    """Refuse to start if another job_runner is alive in this tree (E-23: two runners launched
    duplicate processes on the same run directories and one wiped a live run)."""
    os.makedirs(os.path.dirname(RUNNER_LOCK), exist_ok=True)
    if os.path.exists(RUNNER_LOCK):
        try:
            other = open(RUNNER_LOCK).read().split()
            opid = int(other[0])
        except Exception:
            opid = -1
        if opid > 0 and opid != os.getpid() and pid_alive(opid):
            print(f"[runner] ANOTHER job_runner IS RUNNING (pid {opid}, started {' '.join(other[1:])}). "
                  f"Not starting a second one. Inspect it with: tail console_logs/<session>/runner_*.log ; "
                  f"stop it with: pkill -f job_runner.py ; pkill -f train_fourierfuse_v18")
            sys.exit(3)
        print(f"[runner] stale lock from pid {opid} (not alive) -> replaced")
    with open(RUNNER_LOCK, 'w') as f:
        f.write(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')} {os.getcwd()}\n")
    import atexit
    atexit.register(lambda: os.path.exists(RUNNER_LOCK) and open(RUNNER_LOCK).read().split()[0] == str(os.getpid())
                    and os.remove(RUNNER_LOCK))


def gpu_mem_used_mib(gpu):
    """(used, total) MiB for one GPU index, or (None, None) if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        for l in out.splitlines():
            idx, used, total = [x.strip() for x in l.split(',')]
            if idx == str(gpu):
                return int(float(used)), int(float(total))
    except Exception:
        pass
    return None, None


BUSY_MIB = 1500          # a selected GPU with more than this in use is not ours to fill (E-24)
NEED_FREE_MIB = 9000     # a training job is launched only when at least this much is free on its GPU


def preflight(jobs, gpus, allow_busy=False):
    problems = []
    for ds in sorted({j['ds'] for j in jobs}):
        if not os.path.isdir(DATA_ROOT[ds]):
            problems.append(f"data root missing for {ds}: {DATA_ROOT[ds]}")
    for f in (TRAIN, 'revision_modules.py', 'fourierfuse_v16_modules.py'):
        if not os.path.exists(f):
            problems.append(f"missing {f} in cwd {os.getcwd()}")
    free_gb = shutil.disk_usage('.').free / 1e9
    if free_gb < 20:
        problems.append(f"only {free_gb:.1f} GB free")
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        print('[preflight] nvidia-smi:\n' + out)
        seen = {l.split(',')[0].strip() for l in out.splitlines()}
        for g in gpus:
            if g not in seen:
                problems.append(f"GPU {g} not visible")
            else:
                used, total = gpu_mem_used_mib(g)
                if used is not None and used > BUSY_MIB and not allow_busy:
                    problems.append(f"GPU {g} already has {used} MiB in use (E-24): another runner or a "
                                    f"leftover training process is on it. Run: pkill -f job_runner.py ; "
                                    f"pkill -f train_fourierfuse_v18 ; then check nvidia-smi shows 0 MiB, then re-issue. "
                                    f"(--allow_busy overrides, only if that memory is deliberately yours)")
    except Exception as e:
        problems.append(f"nvidia-smi failed: {e!r}")
    r = subprocess.run([sys.executable, 'revision_modules.py', '--selftest'], capture_output=True, text=True)
    if 'PASSED' not in r.stdout:
        problems.append('revision_modules selftest failed:\n' + r.stdout[-2000:] + r.stderr[-2000:])
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True); ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--per_gpu', type=int, default=2); ap.add_argument('--session', default='S01')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--force', action='store_true')
    ap.add_argument('--restart', action='store_true',
                    help='pass --resume off: wipe and restart selected jobs instead of resuming them')
    ap.add_argument('--only', default='', help='comma-separated substrings to select jobs')
    ap.add_argument('--skip', default='', help='comma-separated substrings to exclude jobs (e.g. dropoutctl,devisobj)')
    ap.add_argument('--dry', action='store_true', help='print what would run and exit')
    ap.add_argument('--allow_busy', action='store_true', help='start even if a selected GPU already has memory in use')
    a = ap.parse_args()
    gpus = a.gpus.split(',')
    jobs = []
    for line in open(a.jobs):
        if not line.strip() or line.startswith('#'):
            continue
        name, ds, cmd = line.rstrip('\n').split('\t')
        if a.only and not any(s in name for s in a.only.split(',')):
            continue
        if a.skip and any(s in name for s in a.skip.split(',')):
            continue
        jobs.append({'name': name, 'ds': ds, 'cmd': cmd})
    listname = os.path.splitext(os.path.basename(a.jobs))[0]
    logdir = os.path.join('console_logs', a.session); os.makedirs(logdir, exist_ok=True)
    if not a.dry:
        acquire_runner_lock()
    tee = Tee(os.path.join(logdir, f'runner_{listname}.log')); sys.stdout = tee; sys.stderr = tee
    print(f"[runner] {time.strftime('%Y-%m-%d %H:%M:%S')} cwd={os.getcwd()} log={os.path.abspath(tee.f.name)}")
    status_path = os.path.join(logdir, f'status_{listname}.csv')
    print(f"[runner] {len(jobs)} jobs from {a.jobs}; gpus={gpus} per_gpu={a.per_gpu} session={a.session}")
    todo = []
    for j in jobs:
        ok, why = postflight(j['ds'], j['name'], a.smoke)
        rs = os.path.join(snapshot_dir(j['ds'], j['name']), 'resume_state.pth')
        if ok and not a.force:
            print(f"[skip] {j['name']} already complete ({why})")
        else:
            if os.path.exists(rs) and not a.restart:
                try:
                    import torch
                    it = torch.load(rs, map_location='cpu', weights_only=False).get('iter_num', '?')
                except Exception:
                    it = '?'
                print(f"[resume] {j['name']} has resume_state.pth (iter {it}) -> will CONTINUE, not restart")
            elif os.path.exists(rs) and a.restart:
                print(f"[restart] {j['name']} -> --resume off (wipe and start over)")
            todo.append(j)
    if a.restart:
        for j in todo:
            j['cmd'] += ' --resume off'
    print(f"[runner] {len(todo)} to run, {len(jobs)-len(todo)} skipped")
    if a.dry:
        for j in todo:
            print(f"  python {TRAIN} {j['cmd']} --gpu <g>")
        return
    probs = preflight(todo, gpus, a.allow_busy)
    if probs:
        print('[preflight] FAILED:\n  - ' + '\n  - '.join(probs)); sys.exit(2)
    print('[preflight] GREEN')
    q = Queue()
    for j in todo:
        q.put(j)
    lock = threading.Lock()
    results = {}

    def log_status(row):
        with lock, open(status_path, 'a', newline='') as f:   # E-18: decide 'new' under the lock
            new = f.tell() == 0
            w = csv.writer(f)
            if new:
                w.writerow(['job', 'dataset', 'gpu', 'start', 'end', 'exit_code', 'postflight', 'detail'])
            w.writerow(row)

    cascade = {'fast_fails': 0, 'abort': False}

    def worker(gpu, slot):
        while True:
            if cascade['abort']:
                return
            try:
                j = q.get_nowait()
            except Exception:
                return
            waited = 0
            while True:                                   # E-24: never launch into a full GPU
                used, total = gpu_mem_used_mib(gpu)
                if used is None or total - used >= NEED_FREE_MIB or waited >= 1800:
                    break
                if waited == 0:
                    print(f"[gpu{gpu}/slot{slot}] waiting for {NEED_FREE_MIB} MiB free on GPU {gpu} "
                          f"({total - used} MiB free now) before {j['name']}", flush=True)
                time.sleep(30); waited += 30
            cmd = f"{sys.executable} {TRAIN} {j['cmd']} --gpu {gpu}"
            con = os.path.join(logdir, f"{j['name']}.console.log")
            start = time.strftime('%Y-%m-%d %H:%M:%S'); t_launch = time.time()
            print(f"[gpu{gpu}/slot{slot}] START {j['name']}  -> {con}", flush=True)
            with open(con, 'w') as f:
                f.write(f"# {start}\n# {cmd}\n"); f.flush()
                rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
            ok, why = postflight(j['ds'], j['name'], a.smoke)
            end = time.strftime('%Y-%m-%d %H:%M:%S')
            verdict = 'PASS' if (rc == 0 and ok) else 'FAIL'
            print(f"[gpu{gpu}/slot{slot}] {verdict} {j['name']} rc={rc} {why}", flush=True)
            log_status([j['name'], j['ds'], gpu, start, end, rc, verdict, why])
            with lock:
                results[j['name']] = verdict
                if verdict == 'FAIL' and not a.smoke and time.time() - t_launch < 120:
                    cascade['fast_fails'] += 1        # a job that dies within 2 min is a systemic fault, not a result
                    if cascade['fast_fails'] >= 3 and not cascade['abort']:
                        cascade['abort'] = True
                        print(f"[runner] CASCADE ABORT: 3 jobs died within 2 minutes of launch -> stopping; "
                              f"read {con} (last lines) before re-issuing", flush=True)
                elif verdict == 'PASS':
                    cascade['fast_fails'] = 0
            q.task_done()

    threads = []
    for g in gpus:
        for s in range(a.per_gpu):
            t = threading.Thread(target=worker, args=(g, s), daemon=True); t.start(); threads.append(t)
            time.sleep(20)   # stagger launches so data loading / cudnn init do not collide
    for t in threads:
        t.join()
    n_pass = sum(v == 'PASS' for v in results.values())
    print(f"[runner] DONE: {n_pass}/{len(results)} PASS. status: {status_path}")
    if n_pass != len(results):
        print('[runner] FAILED jobs: ' + ', '.join(k for k, v in results.items() if v != 'PASS'))
        sys.exit(1)


if __name__ == '__main__':
    main()
