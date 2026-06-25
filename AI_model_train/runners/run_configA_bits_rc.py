"""
Generic bit-width sweep: {source} -> all other datasets.

Mirrors Table 1 (adapter no-RC v3 design) + Table 1b (full_ft) for cross-source.
Adapter design: v3 k=3 (kernel=3, ReLU/sign, per-ch alpha, mid='out', no bias, M=1) —
same as v_v3_cross (CIFAR10 source).

Usage:
  python3 claude/run_xx_to_others_bits.py --source SVHN --bits 2 4 8 16 32 --parallel 3
  python3 claude/run_xx_to_others_bits.py --source STL10 --bits 1 2 4 8 16 32 --parallel 3

Backbones required:
  claude/pretrained_backbones/{source.lower()}_{N}w{N}a.tar for each --bits N

Cells: |targets|=4 x |bits| x 2 modes (adapter_rc, full_ft).
Output: claude/paper_results_bitwidth/{source.lower()}_to_others_bits/
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


def build_full_ft(source, target, bit, args, exp_dir):
    bp = os.path.join(BACKBONE_DIR, f'{source.lower()}_{bit}w{bit}a.tar')
    exp_name = f"Transfer_{source.lower()}2oCAbitsRC_{target}_b{bit}_full_ft_e{args.epochs}"
    label = f"{target}_b{bit}_full_ft"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'full_ft', '--net_bit', str(bit),
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


def build_adapter_rc(source, target, bit, args, exp_dir):
    bp = os.path.join(BACKBONE_DIR, f'{source.lower()}_{bit}w{bit}a.tar')
    exp_name = f"Transfer_{source.lower()}2oCAbitsRC_{target}_b{bit}_adapter_rc_e{args.epochs}"
    label = f"{target}_b{bit}_adapter_rc"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'adapter', '--net_bit', str(bit),
        '--dataset', target,
        '--finetune_checkpoint', bp,
        '--epochs', str(args.epochs), '--lr', str(args.lr),
        '--scheduler', 'STEP', '--milestones', ms,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_seed', str(args.seed),
        '--experiments', exp_dir,
        '--experiment_name', exp_name,
        '--num_branches', '1',
        '--adapter_bit_width', str(bit),
        '--adapter_kernel', '1',
        '--adapter_act', 'signed',
        '--adapter_alpha', 'scalar',
        '--adapter_mid_basis', 'in',
        '--no_rc',
        '--adapter_bias',  # RC ON
        # No --adapter_bias -> Table 1 style (no RC)
    ]
    return label, cmd


def parse_label(label):
    """label = '{dataset}_b{N}_{full_ft|adapter_rc}'."""
    parts = label.split('_')
    dataset = parts[0]
    bit = int(parts[1][1:])
    mode = '_'.join(parts[2:])
    return dataset, bit, mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=str, required=True,
                   choices=['CIFAR10', 'SVHN', 'STL10', 'FashionMNIST', 'CINIC10'])
    p.add_argument('--bits', type=int, nargs='+', required=True)
    p.add_argument('--modes', type=str, nargs='+',
                   default=['adapter_rc'],
                   choices=['adapter_rc', 'full_ft'])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--gpu', type=int, nargs='+', default=[0])
    p.add_argument('--targets', type=str, nargs='+', default=None)
    args = p.parse_args()

    output_root = os.path.join(THIS_DIR, 'paper_results_bitwidth',
                               f'{args.source.lower()}_configA_bits_rc')
    exp_dir = os.path.join(output_root, 'experiments')
    log_dir = os.path.join(output_root, 'logs')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    targets = args.targets if getattr(args, 'targets', None) else [d for d in CANONICAL if d != args.source]
    print(f"{args.source}_to_others_bits sweep: targets={targets}, bits={args.bits}, "
          f"modes={args.modes}, epochs={args.epochs}, parallel={args.parallel}, gpu={args.gpu}")

    for bit in args.bits:
        bp = os.path.join(BACKBONE_DIR, f'{args.source.lower()}_{bit}w{bit}a.tar')
        if not os.path.exists(bp):
            print(f"ERROR: backbone {bp} not found")
            sys.exit(1)

    jobs = []
    for tgt in targets:
        for bit in args.bits:
            if 'full_ft' in args.modes:
                _l, _c = build_full_ft(args.source, tgt, bit, args, exp_dir)
                if is_done(log_dir, _l):
                    print(f"SKIP done: {_l}")
                else:
                    jobs.append((_l, _c))
            if 'adapter_rc' in args.modes:
                _l, _c = build_adapter_rc(args.source, tgt, bit, args, exp_dir)
                if is_done(log_dir, _l):
                    print(f"SKIP done: {_l}")
                else:
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
        dataset, bit, mode = parse_label(lbl)
        rows.append({'dataset': dataset, 'bit': bit, 'mode': mode,
                     'acc': acc, 'params': params, 'returncode': rc})
    df = pd.DataFrame(rows).sort_values(['dataset', 'mode', 'bit']).reset_index(drop=True)
    csv_path = os.path.join(output_root, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[CSV] -> {csv_path}")
    print(df.to_string())


if __name__ == '__main__':
    main()
