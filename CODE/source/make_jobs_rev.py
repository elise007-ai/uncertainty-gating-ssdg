#!/usr/bin/env python3
"""
make_jobs_rev.py — generates the revision job lists (one command per line).

The COMMON / per-dataset blocks are copied VERBATIM from
run_v16_experiments_v2.sh / run_adaptive_threshold.sh, so every new run
inherits exactly the configuration that produced the submitted numbers; the
only differences are the flags a job explicitly adds.

  python make_jobs_rev.py --out_dir jobs            # writes jobs/jobs_{smoke,A,B,C,B2,R}.txt
  python make_jobs_rev.py --out_dir jobs --lists R  # S10: writes ONLY jobs_R.txt (nothing else touched)
Each line:  <job_name>\t<dataset>\t<command>
The runner (job_runner.py) prepends `python train_fourierfuse_v18_revision.py`
and appends `--gpu <g>` itself.
"""
import argparse, os

COMMON = ("--warmup --warmup_period 150 --AdamW --amp 1 --ff_base_features 64 "
          "--ff_spec_consistency_weight 0.5 --ff_fda_beta 0.05 "
          "--consistency 1.0 --consistency_rampup 150.0 --ema_decay 0.995 "
          "--daep_alpha 1.0 --daep_beta 1.0 --daep_gamma 1.0 "
          "--proto_per_domain 2 --proto_ema 0.99 "
          "--evid_lambda_kl_max 0.02 --evid_anneal_iters 10000 "
          "--save_model --overwrite")

DATASET = {
    'prostate': ("--dataset prostate --lb_num 20 --lb_domain 5 --domain_num 6 "
                 "--max_iterations 60000 --num_eval_iter 500 --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 "
                 "--swad_start_iter 15000 --swad_update_every 100 --swad_bn_recalib_batches 20"),
    'fundus':   ("--dataset fundus --lb_num 20 --lb_domain 3 --domain_num 4 "
                 "--max_iterations 30000 --num_eval_iter 500 --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 "
                 "--swad_start_iter 10000 --swad_update_every 100 --swad_bn_recalib_batches 20"),
    'BUSI':     ("--dataset BUSI --lb_num 64 --lb_domain 2 --domain_num 2 "
                 "--max_iterations 30000 --num_eval_iter 500 --base_lr 0.0005 --label_bs 4 --unlabel_bs 4 "
                 "--swad_start_iter 10000 --swad_update_every 100 --swad_bn_recalib_batches 20"),
    'MNMS':     ("--dataset MNMS --lb_num 5 --lb_domain 4 --domain_num 4 "
                 "--max_iterations 60000 --num_eval_iter 500 --base_lr 0.0005 --label_bs 2 --unlabel_bs 2 "
                 "--swad_start_iter 15000 --swad_update_every 100 --swad_bn_recalib_batches 20"),
}
SEEDS = [1337, 42, 2024]
# Ordering: longest jobs first so the two GPUs stay full (Prostate/MNMS 60k > Fundus/BUSI 30k).
ORDER = {'prostate': 0, 'MNMS': 1, 'fundus': 2, 'BUSI': 3}


def job(name, ds, extra, seed, smoke=False):
    flags = DATASET[ds]
    if smoke:  # 20 iterations, one eval, same flags otherwise
        flags = flags.replace('--max_iterations 60000', '--max_iterations 20').replace(
            '--max_iterations 30000', '--max_iterations 20').replace('--num_eval_iter 500', '--num_eval_iter 10')
    return f"{name}\t{ds}\t{flags} {COMMON} {extra} --seed {seed} --save_name {name}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out_dir', default='jobs')
    ap.add_argument('--lists', default='', help='comma-separated subset of lists to write (e.g. R); default all')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    J = {'smoke': [], 'A': [], 'B': [], 'C': []}

    # ---- SMOKE: every code path, 20 iterations each, Prostate (K=2) and MNMS (K=4) ----
    smoke_cfgs = [
        ('fixmatch_default', '--method fixmatch'), ('daep_default', '--method daep'),
        ('softmax', '--method fixmatch --head softmax'), ('adaptive', '--method adaptive'),
        ('mcdropout', '--method mcdropout --mc_dropout_p 0.5 --mc_T 4'),
        ('devis', '--method devis'), ('medl', '--method medl --medl_report net1'),
        ('evid_relu_ce_w10', '--method daep --evid_activation relu --evid_loss ce --evid_prior_w 10'),
        ('evid_exp_log_cap', '--method daep --evid_activation exp --evid_loss log --evid_max_evidence 100 --evid_kl_schedule const'),
        ('percentile', '--method percentile_gate'), ('consistency', '--method consistency_gate'),
        ('split_seed', '--method fixmatch --split_seed 1337'),
        ('devis_thr', '--method devis --devis_gate threshold'),          # S04 control paths
        ('dropoutctl', '--method fixmatch --mc_dropout_p 0.5'),
        # S07 (E-19/E-20): gate-only MC-dropout (deterministic teacher target, UA-MT placement)
        ('mcdropout2', '--method mcdropout --mc_dropout_p 0.5 --mc_T 4 --mc_target teacher --mc_dropout_place uamt'),
        ('dropoutctl2', '--method fixmatch --mc_dropout_p 0.5 --mc_dropout_place uamt'),
    ]
    for n, ex in smoke_cfgs:
        J['smoke'].append(job(f'v18smoke_prostate_{n}', 'prostate', ex, 43, smoke=True))
    for n, ex in smoke_cfgs[:7]:
        J['smoke'].append(job(f'v18smoke_mnms_{n}', 'MNMS', ex, 43, smoke=True))

    # ---- A: evidential configuration sweep (R1 #4, #6, #10), DAEP gate, Prostate, seed 1337 ----
    # Reference = existing adaptive_cmp_prostate_daep_s1337 (softplus, mse, KL 0.02 linear-10k, W=1, T=1).
    A = [
        ('kl0.1_lin',   '--evid_lambda_kl_max 0.1'),
        ('kl0.5_lin',   '--evid_lambda_kl_max 0.5'),
        ('kl1.0_lin',   '--evid_lambda_kl_max 1.0'),                 # Sensoy's annealed-to-1 KL
        ('kl0.02_const', '--evid_kl_schedule const'),
        ('kl_none',     '--evid_kl_schedule none'),
        ('act_relu',    '--evid_activation relu'),                   # Sensoy 2018 original
        ('act_exp',     '--evid_activation exp'),
        ('loss_ce',     '--evid_loss ce'),                           # digamma Bayes risk (DEviS form)
        ('loss_log',    '--evid_loss log'),
        ('priorw0.1',   '--evid_prior_w 0.1'),
        ('priorw10',    '--evid_prior_w 10'),
        ('temp0.5',     '--evid_temperature 0.5'),
        ('temp2.0',     '--evid_temperature 2.0'),
        ('cap100',      '--evid_max_evidence 100'),
    ]
    # Reference row re-run under v18 (the S01 adaptive_cmp_prostate_daep_s1337 checkpoint is not on the
    # server): same code version + measurable teacher; reproduction gate vs S01 log 0.857467.
    J['A'].append(job('v18A_prostate_daep_default_s1337', 'prostate', '--method daep', 1337))
    for n, ex in A:
        J['A'].append(job(f'v18A_prostate_daep_{n}_s1337', 'prostate', f'--method daep {ex}', 1337))

    # ---- B: alternative estimators / baselines (R1 #5, #8) ----
    for ds in ['prostate', 'MNMS', 'fundus', 'BUSI']:           # D-2: softmax FixMatch, all four, 3 seeds
        for s in SEEDS:
            J['B'].append(job(f'v18B_{ds}_softmax_fixmatch_s{s}', ds, '--method fixmatch --head softmax', s))
    for ds in ['prostate', 'fundus']:                           # D-3: Prostate + Fundus, 3 seeds
        for s in SEEDS:
            # S07: `mcdropout2` = gate-only MC-dropout (E-19: the S06 `mcdropout` jobs used the MC mean
            # as the pseudo-label target; E-20: dropout placement now UA-MT's). Old names are retired.
            J['B'].append(job(f'v18B_{ds}_mcdropout2_s{s}', ds,
                              '--method mcdropout --mc_dropout_p 0.5 --mc_T 8 --mc_target teacher --mc_dropout_place uamt', s))
            J['B'].append(job(f'v18B_{ds}_devis_s{s}', ds, '--method devis', s))
            J['B'].append(job(f'v18B_{ds}_medl_s{s}', ds, '--method medl --medl_report net1', s))
        for s in (1337, 43, 44):                                # deep-ensemble members at the s1337 split;
            # init1337 == exact re-run of adaptive_cmp_<ds>_fixmatch_s1337 under v18 defaults
            # (R-27 reproduction gate: best_avg must match the S01 logs, prostate 0.862933 / fundus 0.882202)
            J['B'].append(job(f'v18B_{ds}_fixmatch_split1337_init{s}', ds, '--method fixmatch --split_seed 1337', s))

    # D-08 gate-only controls (fairness): the MC-dropout run changes the MODEL (dropout regulariser) and the
    # DEviS run changes the OBJECTIVE; each needs a partner with the same model/objective but the standard
    # 0.80 threshold gate, otherwise a Dice difference cannot be attributed to the gate.
    for ds in ['prostate', 'fundus']:
        for s in SEEDS:
            J['B'].append(job(f'v18B_{ds}_dropoutctl2_fixmatch_s{s}', ds,
                              '--method fixmatch --mc_dropout_p 0.5 --mc_dropout_place uamt', s))
            J['B'].append(job(f'v18B_{ds}_devisobj_threshold_s{s}', ds, '--method devis --devis_gate threshold', s))
    for ds in ['MNMS', 'BUSI']:                                # D-07: regenerate the missing threshold teachers
        for s in SEEDS:                                         # (reproduction gate vs RESULTS/seed_split_migration logs)
            J['B'].append(job(f'v18B_{ds}_fixmatch_s{s}', ds, '--method fixmatch', s))

    # ---- B2 (S07, D-09/D-10): within-v18 threshold-gate references so every Phase-B paired test is
    # code-matched (the s1337 member is the `init1337` reproduction run; s42/s2024 had only v16 logs),
    # plus one same-code same-seed replicate of the Phase-A reference to MEASURE run-to-run
    # non-determinism (E-17: the +/-0.005 reproduction gate was set without one).
    J['B2'] = []
    for ds in ['prostate', 'fundus']:
        for s in (42, 2024):
            J['B2'].append(job(f'v18B_{ds}_fixmatch_s{s}', ds, '--method fixmatch', s))
    J['B2'].append(job('v18A_prostate_daep_default_s1337_rep2', 'prostate', '--method daep', 1337))

    # ---- C: FDA parameter sensitivity (R1 #9), threshold gate, Prostate, seed 1337 ----
    for b in ['0.01', '0.02', '0.10', '0.20']:
        J['C'].append(job(f'v18C_prostate_fixmatch_beta{b}_s1337', 'prostate',
                          f'--method fixmatch --ff_fda_beta {b}', 1337))
    for p in ['0.25', '0.75', '1.0']:
        J['C'].append(job(f'v18C_prostate_fixmatch_pfda{p}_s1337', 'prostate',
                          f'--method fixmatch --ff_fda_prob {p}', 1337))

    # ---- R (S10, D-13 approved 2026-10-03): standard-ramp control for E-28 ----
    # The testbed's consistency weight ramps over the WHOLE run (150 buckets; 0.04 at 20 %, cap 0.5
    # from 63 %), identical for every gate. Both headline gates are re-trained on Prostate s1337 with
    # the conventional short ramp (--consistency_ramp_frac 0.10: per-iteration sigmoid over the first
    # 10 % of iterations, same cap 0.5, cap reached at 6.3 %); everything else is the headline
    # configuration. PRE-REGISTERED reading rule (HANDOFF v1.9 §3): the pair is compared with the
    # same-seed headline pair; if |delta_short - delta_whole| <= the B2 replicate tolerance (E-17),
    # E-28 is a disclosure only; otherwise the Discussion paragraph on the schedule is rewritten.
    # Single seed -> exploratory; new names (v18R_*) match no existing table regex.
    J['R'] = []
    for n, ex in [('fixmatch', '--method fixmatch'), ('daep', '--method daep')]:
        J['R'].append(job(f'v18R_prostate_{n}_ramp10_s1337', 'prostate',
                          f'{ex} --consistency_ramp_frac 0.10', 1337))

    only = set(a.lists.split(',')) - {''}
    for k, lines in J.items():
        if only and k not in only:
            continue
        lines.sort(key=lambda l: ORDER[l.split('\t')[1]])
        with open(os.path.join(a.out_dir, f'jobs_{k}.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"jobs_{k}.txt: {len(lines)} jobs")


if __name__ == '__main__':
    main()
