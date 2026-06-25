# ===========================================================================
# [交接導向註解]
# runner：n=3 multi-seed 變異（headline 格）。
# 流程：AI_model_train。產出對應 results/ 之 results.csv（見 README 對照表）。
# ===========================================================================

"""
Multi-seed (n=3) variance sweep for headline cells.

Existing seed=2024 runs come from v7 / v9 / v9_ft. This sweep adds seeds 2025
and 2026 for the headline 12 cells, giving n=3 per cell with mean ± std.

Headline cells (1-bit, 200ep, lr=0.005, milestones=100,150):
  SVHN (6):
    - v1/v2-eq M=1 rc, M=4 rc        (HW deployed equivalent)
    - v6-eq    M=1 rc, M=4 rc        (SW best)
    - full_ft, frozen_only            (upper/lower bounds)
  STL10 (2):       v1/v2 M=4 rc, full_ft
  FashionMNIST(2): v1/v2 M=4 rc, full_ft
  CINIC10 (2):     v1/v2 M=4 rc, full_ft

12 cells × 2 extra seeds = 24 runs.

Output:
  claude/paper_results_bitwidth/v_seed/
    experiments/Transfer_seed{N}_{cell}_e200/
    logs/seed{N}_{cell}.log
    results.csv  cell,seed,acc,params,returncode
"""

import argparse
import os
import sys
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))

OUTPUT_ROOT = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'v_seed')
EXP_DIR = os.path.join(OUTPUT_ROOT, 'experiments')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BIT = 1

# Headline cell specs
# Each: (label, mode, dataset, num_branches, kernel, act, alpha, bias, mid_basis)
# mode in {adapter, full_ft, frozen_only}; for non-adapter modes the design fields are None.
CELLS = [
    # SVHN — v1/v2-eq (1×1 + sign + scalar α + bias, mid='in')
    ('SVHN_v1v2_M1_rc', 'adapter', 'SVHN', 1, 1, 'signed', 'scalar', True, 'in'),
    ('SVHN_v1v2_M4_rc', 'adapter', 'SVHN', 4, 1, 'signed', 'scalar', True, 'in'),
    # SVHN — v6-eq (3×3 + ReLU/sign + per-ch α + bias, mid='in')  ★ SW best
    ('SVHN_v6_M1_rc', 'adapter', 'SVHN', 1, 3, 'relu', 'per_channel', True, 'in'),
    ('SVHN_v6_M4_rc', 'adapter', 'SVHN', 4, 3, 'relu', 'per_channel', True, 'in'),
    # SVHN — bounds
    ('SVHN_full_ft', 'full_ft', 'SVHN', 0, None, None, None, False, None),
    ('SVHN_frozen_only', 'frozen_only', 'SVHN', 0, None, None, None, False, None),
    # STL10
    ('STL10_v1v2_M4_rc', 'adapter', 'STL10', 4, 1, 'signed', 'scalar', True, 'in'),
    ('STL10_full_ft', 'full_ft', 'STL10', 0, None, None, None, False, None),
    # FashionMNIST
    ('FashionMNIST_v1v2_M4_rc', 'adapter', 'FashionMNIST', 4, 1, 'signed', 'scalar', True, 'in'),
    ('FashionMNIST_full_ft', 'full_ft', 'FashionMNIST', 0, None, None, None, False, None),
    # CINIC10
    ('CINIC10_v1v2_M4_rc', 'adapter', 'CINIC10', 4, 1, 'signed', 'scalar', True, 'in'),
    ('CINIC10_full_ft', 'full_ft', 'CINIC10', 0, None, None, None, False, None),
]

EXTRA_SEEDS = [2025, 2026]   # seed 2024 already exists from v7 / v9 / v9_ft

_PRINT_LOCK = threading.Lock()


def is_done(label):
    log_path = os.path.join(LOG_DIR, f"{label}.log")
    try:
        with open(log_path) as f:
            return "Final Best Accuracy" in f.read()
    except Exception:
        return False


def stream_cmd(cmd, label, gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    log_path = os.path.join(LOG_DIR, f"{label}.log")
    log_fp = open(log_path, 'w')

    with _PRINT_LOCK:
        print(f"\n>>> [GPU {gpu_id}] {label} <<<")
        print(' '.join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=PROJ_ROOT, env=env)
    best_acc = 0.0
    total_params = 0
    tag = f"[{label}]"
    try:
        for line in proc.stdout:
            log_fp.write(line)
            log_fp.flush()
            with _PRINT_LOCK:
                print(f"{tag} {line}", end='')
                sys.stdout.flush()
            if "Final Best Accuracy" in line:
                try:
                    best_acc = round(float(line.split(':')[-1].replace('%', '').strip()), 2)
                except Exception:
                    pass
            if "[Model Stats] Total Params" in line:
                try:
                    total_params = int(line.split(':')[-1].strip())
                except Exception:
                    pass
    finally:
        rc = proc.wait()
        log_fp.close()
    if rc != 0:
        with _PRINT_LOCK:
            print(f"FAILED: {label} (exit {rc})")
    return best_acc, total_params, rc


def build_cmd(cell, seed, args):
    label_root, mode, dataset, M, kernel, act, alpha, bias, mid = cell
    bp = os.path.join(BACKBONE_DIR, f'cifar10_{BIT}w{BIT}a.tar')
    exp_name = f"Transfer_seed{seed}_{label_root}_e{args.epochs}"
    label = f"seed{seed}_{label_root}"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', mode,
        '--net_bit', str(BIT),
        '--dataset', dataset,
        '--finetune_checkpoint', bp,
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--scheduler', 'STEP',
        '--milestones', ms,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_seed', str(seed),
        '--experiments', EXP_DIR,
        '--experiment_name', exp_name,
    ]
    if mode == 'adapter':
        cmd += [
            '--num_branches', str(M),
            '--adapter_bit_width', str(BIT),
            '--adapter_kernel', str(kernel),
            '--adapter_act', act,
            '--adapter_alpha', alpha,
            '--adapter_mid_basis', mid,
            '--no_rc',
        ]
        if bias:
            cmd.append('--adapter_bias')
    return label, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--parallel', type=int, default=3)
    p.add_argument('--gpu', type=int, nargs='+', default=[0])
    args = p.parse_args()

    print(f"v_seed sweep: {len(CELLS)} cells x {len(EXTRA_SEEDS)} extra seeds = "
          f"{len(CELLS) * len(EXTRA_SEEDS)} runs, 1-bit, "
          f"epochs={args.epochs}, parallel={args.parallel}, gpu={args.gpu}")

    jobs = []
    for cell in CELLS:
        for seed in EXTRA_SEEDS:
            _label, _cmd = build_cmd(cell, seed, args)
            if is_done(_label):
                print(f"SKIP done: {_label}")
                continue
            jobs.append((_label, _cmd))

    gpu_iter = cycle(args.gpu)
    job_gpu = [next(gpu_iter) for _ in jobs]

    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {}
        for i, (label, cmd) in enumerate(jobs):
            fut = pool.submit(stream_cmd, cmd, label, job_gpu[i])
            futs[fut] = i
        for fut in as_completed(futs):
            i = futs[fut]
            acc, params, rc = fut.result()
            label, _ = jobs[i]
            results[i] = (label, acc, params, rc)

    rows = []
    for label, acc, params, rc in results:
        # label = "seed{N}_{cell_label}"
        seed_str, cell_label = label.split('_', 1)
        seed = int(seed_str.replace('seed', ''))
        rows.append({'cell': cell_label, 'seed': seed, 'acc': acc,
                     'params': params, 'returncode': rc})

    df = pd.DataFrame(rows).sort_values(['cell', 'seed']).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_ROOT, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[CSV] -> {csv_path}")
    print(df.to_string())


if __name__ == '__main__':
    main()
