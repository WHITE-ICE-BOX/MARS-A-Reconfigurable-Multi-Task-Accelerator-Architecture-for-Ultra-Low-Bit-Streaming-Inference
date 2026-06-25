"""
Generic 1-bit v6 sweep: {source} -> all other datasets (mirror svhn_to_others).

For each source, target = full 4-set minus source itself.
Adapter design: v6 (kernel=3, ReLU/sign, per-ch alpha, mid='in', bias toggleable).

Cells: |targets|=4 (after self-exclusion of source from canonical set) x
       (M=1-4 x {norc,rc} + full_ft) = 36 cells, 1-bit, 200ep.

Usage:
  python3 claude/run_xx_to_others.py --source STL10 --parallel 3

Backbone required:
  claude/pretrained_backbones/{source.lower()}_1w1a.tar

Output:
  claude/paper_results_bitwidth/{source.lower()}_to_others/
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
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

CANONICAL = ['CIFAR10', 'SVHN', 'STL10', 'FashionMNIST', 'CINIC10']
MS = [1, 2, 3, 4]
RC_VALUES = [False, True]
BIT = 1

_PRINT_LOCK = threading.Lock()


def is_done(log_dir, label):
    log_path = os.path.join(log_dir, f"{label}.log")
    try:
        with open(log_path) as f:
            return "Final Best Accuracy" in f.read()
    except Exception:
        return False


def stream_cmd(cmd, label, gpu_id, log_dir):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    log_path = os.path.join(log_dir, f"{label}.log")
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
            log_fp.write(line); log_fp.flush()
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


def build_full_ft(source, target, args, exp_dir):
    bp = os.path.join(BACKBONE_DIR, f'{source.lower()}_{BIT}w{BIT}a.tar')
    exp_name = f"Transfer_{source.lower()}2oAC_{target}_full_ft_e{args.epochs}"
    label = f"{target}_full_ft"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'full_ft', '--net_bit', str(BIT),
        '--dataset', target,
        '--finetune_checkpoint', bp,
        '--epochs', str(args.epochs), '--lr', str(args.lr),
        '--scheduler', 'STEP', '--milestones', ms,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_seed', str(args.seed),
        '--experiments', exp_dir,
        '--experiment_name', exp_name,
    ]
    return label, cmd


def build_adapter(source, target, M, rc_on, args, exp_dir):
    bp = os.path.join(BACKBONE_DIR, f'{source.lower()}_{BIT}w{BIT}a.tar')
    rc_tag = 'rc' if rc_on else 'norc'
    exp_name = f"Transfer_{source.lower()}2oAC_{target}_M{M}_{rc_tag}_e{args.epochs}"
    label = f"{target}_M{M}_{rc_tag}"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'adapter', '--net_bit', str(BIT),
        '--dataset', target,
        '--finetune_checkpoint', bp,
        '--epochs', str(args.epochs), '--lr', str(args.lr),
        '--scheduler', 'STEP', '--milestones', ms,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_seed', str(args.seed),
        '--experiments', exp_dir,
        '--experiment_name', exp_name,
        '--num_branches', str(M),
        '--adapter_bit_width', str(BIT),
        '--adapter_kernel', '1', '--adapter_act', 'signed',
        '--adapter_alpha', 'scalar', '--adapter_mid_basis', 'in',
        '--no_rc',
    ]
    if rc_on:
        cmd.append('--adapter_bias')
    return label, cmd


def parse_label(label):
    if label.endswith('_full_ft'):
        return label[:-len('_full_ft')], 'full_ft', 0, False
    parts = label.rsplit('_', 2)
    return parts[0], 'adapter', int(parts[1][1:]), parts[2] == 'rc'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=str, required=True,
                   choices=['CIFAR10', 'SVHN', 'STL10', 'FashionMNIST', 'CINIC10'])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--gpu', type=int, nargs='+', default=[0])
    p.add_argument('--targets', type=str, nargs='+', default=None)
    p.add_argument('--rc_only', action='store_true')
    args = p.parse_args()

    output_root = os.path.join(THIS_DIR, 'paper_results_bitwidth',
                               f'{args.source.lower()}_configA_cross')
    exp_dir = os.path.join(output_root, 'experiments')
    log_dir = os.path.join(output_root, 'logs')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    bp = os.path.join(BACKBONE_DIR, f'{args.source.lower()}_{BIT}w{BIT}a.tar')
    if not os.path.exists(bp):
        print(f"ERROR: backbone not found at {bp}")
        sys.exit(1)

    targets = args.targets if getattr(args, 'targets', None) else [d for d in CANONICAL if d != args.source]
    print(f"{args.source}_to_others sweep: targets={targets}, M={MS}, RC={RC_VALUES} + full_ft, "
          f"1-bit, epochs={args.epochs}, parallel={args.parallel}, gpu={args.gpu}")

    jobs = []
    for tgt in targets:
        _l, _c = build_full_ft(args.source, tgt, args, exp_dir)
        if is_done(log_dir, _l):
            print(f"SKIP done: {_l}")
        else:
            jobs.append((_l, _c))
        for M in MS:
            for rc_on in ([True] if args.rc_only else RC_VALUES):
                _l, _c = build_adapter(args.source, tgt, M, rc_on, args, exp_dir)
                if is_done(log_dir, _l):
                    print(f"SKIP done: {_l}")
                    continue
                jobs.append((_l, _c))

    gpu_iter = cycle(args.gpu)
    job_gpu = [next(gpu_iter) for _ in jobs]

    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {}
        for i, (lbl, cmd) in enumerate(jobs):
            fut = pool.submit(stream_cmd, cmd, lbl, job_gpu[i], log_dir)
            futs[fut] = i
        for fut in as_completed(futs):
            i = futs[fut]
            acc, params, rc = fut.result()
            lbl, _ = jobs[i]
            results[i] = (lbl, acc, params, rc)

    rows = []
    for lbl, acc, params, rc in results:
        dataset, mode, M, rc_on = parse_label(lbl)
        rows.append({'dataset': dataset, 'mode': mode, 'M': M, 'rc': rc_on,
                     'acc': acc, 'params': params, 'returncode': rc})
    df = pd.DataFrame(rows).sort_values(['dataset', 'mode', 'M', 'rc']).reset_index(drop=True)
    csv_path = os.path.join(output_root, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[CSV] -> {csv_path}")
    print(df.to_string())


if __name__ == '__main__':
    main()
