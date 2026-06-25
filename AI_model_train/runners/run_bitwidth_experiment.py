"""
Bitwidth-vs-Adapter experiment runner.

Phase A (Pretrain):  CIFAR10 backbone for each bitwidth in BITS.
                     1-bit can be skipped via --reuse_1bit (uses experiments/Cifar10_backbone.tar).
Phase B (Transfer):  For each bitwidth, fine-tune on SVHN with 3 modes:
                       - Full FT             : whole network trainable
                       - Adapter (matched b) : freeze backbone + single adapter
                                               whose bit-width = backbone bit-width
                       - Frozen-only         : freeze backbone, only train FC3 (linear probe lower bound)
                     Adapter bit-width follows --net_bit (v2 design).

Outputs:
    claude/paper_results_bitwidth/
      experiments/                  per-run checkpoints + logs
      results.csv                   bitwidth x mode -> best_acc, params
      plots/bitwidth_vs_acc.png     summary figure
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from queue import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))

OUTPUT_ROOT = os.path.join(THIS_DIR, 'paper_results_bitwidth')
EXP_DIR     = os.path.join(OUTPUT_ROOT, 'experiments')
PLOT_DIR    = os.path.join(OUTPUT_ROOT, 'plots')
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(BACKBONE_DIR, exist_ok=True)

BITS = [1, 2, 4, 8, 16, 32]
MODES = ['full_ft', 'adapter', 'frozen_only']

TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')
LEGACY_1BIT_BACKBONE = os.path.join(PROJ_ROOT, 'experiments', 'Cifar10_backbone.tar')


_PRINT_LOCK = threading.Lock()


def stream_cmd(cmd, label, gpu_id=None, log_dir=None):
    """Run a subprocess, mirror its output to console (thread-safe), parse acc/params."""
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log_path = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{label}.log")
    log_fp = open(log_path, 'w') if log_path else None

    with _PRINT_LOCK:
        gpu_tag = f"[GPU {gpu_id}] " if gpu_id is not None else ""
        print(f"\n>>> {gpu_tag}{label} <<<")
        print(' '.join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=PROJ_ROOT, env=env)
    best_acc = 0.0
    total_params = 0
    tag = f"[{label}]"
    try:
        for line in proc.stdout:
            if log_fp:
                log_fp.write(line)
                log_fp.flush()
            with _PRINT_LOCK:
                # Tag every line so interleaved output stays readable.
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
        if log_fp:
            log_fp.close()
    if rc != 0:
        with _PRINT_LOCK:
            print(f"FAILED: {label} (exit {rc})")
    return best_acc, total_params, rc


def run_jobs(jobs, parallel, gpu_pool, log_dir):
    """jobs: list of (label, cmd). Returns list of (label, best_acc, params, rc)."""
    results = [None] * len(jobs)
    if parallel <= 1:
        for i, (label, cmd) in enumerate(jobs):
            gpu = gpu_pool[0] if gpu_pool else None
            acc, params, rc = stream_cmd(cmd, label, gpu_id=gpu, log_dir=log_dir)
            results[i] = (label, acc, params, rc)
        return results

    # Static GPU assignment: round-robin across pool.
    gpu_iter = cycle(gpu_pool if gpu_pool else [None])
    job_gpu = [next(gpu_iter) for _ in jobs]

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = {}
        for i, (label, cmd) in enumerate(jobs):
            fut = pool.submit(stream_cmd, cmd, label, job_gpu[i], log_dir)
            futs[fut] = i
        for fut in as_completed(futs):
            i = futs[fut]
            acc, params, rc = fut.result()
            label, _ = jobs[i]
            results[i] = (label, acc, params, rc)
    return results


def backbone_path(bit):
    return os.path.join(BACKBONE_DIR, f'cifar10_{bit}w{bit}a.tar')


def build_pretrain_cmd(bit, args):
    exp_name = f"Pretrain_b{bit}_e{args.pretrain_epochs}"
    pre_ms = f"{int(args.pretrain_epochs*0.6)},{int(args.pretrain_epochs*0.8)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'pretrain',
        '--net_bit', str(bit),
        '--dataset', 'CIFAR10',
        '--epochs', str(args.pretrain_epochs),
        '--lr', str(args.pretrain_lr),
        '--scheduler', 'STEP',
        '--milestones', pre_ms,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_seed', str(args.seed),
        '--experiments', EXP_DIR,
        '--experiment_name', exp_name,
    ]
    return f"PRETRAIN_b{bit}", cmd


def build_transfer_cmd(bit, mode, args):
    bp = backbone_path(bit)
    exp_name = f"Transfer_b{bit}_{mode}_e{args.transfer_epochs}"
    tr_ms = f"{int(args.transfer_epochs*0.5)},{int(args.transfer_epochs*0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', mode,
        '--net_bit', str(bit),
        '--dataset', 'SVHN',
        '--finetune_checkpoint', bp,
        '--epochs', str(args.transfer_epochs),
        '--lr', str(args.transfer_lr),
        '--scheduler', 'STEP',
        '--milestones', tr_ms,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_seed', str(args.seed),
        '--experiments', EXP_DIR,
        '--experiment_name', exp_name,
    ]
    if mode == 'adapter':
        # v2: adapter bit-width matches backbone bit-width
        cmd += ['--num_branches', '1', '--adapter_bit_width', str(bit), '--no_rc']
    return f"TRANSFER_b{bit}_{mode}", cmd


def plot_results(df, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bits_sorted = sorted(df['bit'].unique())
    for mode in MODES:
        sub = df[df['mode'] == mode].sort_values('bit')
        if len(sub) == 0:
            continue
        ax.plot(sub['bit'], sub['acc'], marker='o', label=mode, linewidth=2)
    ax.set_xscale('log', base=2)
    ax.set_xticks(bits_sorted)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Backbone bit-width (W & A)', fontsize=12)
    ax.set_ylabel('SVHN Test Accuracy (%)', fontsize=12)
    ax.set_title('Adapter vs. Full FT vs. Frozen-only across backbone bit-widths\n(Adapter bit-width = backbone bit-width)', fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[Plot] -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bits', type=int, nargs='+', default=BITS,
                   help=f'Subset of {BITS}')
    p.add_argument('--modes', type=str, nargs='+', default=MODES,
                   choices=MODES)
    p.add_argument('--pretrain_epochs', type=int, default=500)
    p.add_argument('--transfer_epochs', type=int, default=200)
    p.add_argument('--pretrain_lr', type=float, default=0.02)
    p.add_argument('--transfer_lr', type=float, default=0.005)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--reuse_1bit', action='store_true',
                   help='Reuse experiments/Cifar10_backbone.tar for the 1-bit backbone')
    p.add_argument('--force_pretrain', action='store_true',
                   help='Re-run pretraining even if checkpoint exists')
    p.add_argument('--skip_pretrain', action='store_true')
    p.add_argument('--skip_transfer', action='store_true')
    p.add_argument('--smoke_test', action='store_true',
                   help='Override epochs to 1 for smoke test')
    p.add_argument('--parallel', type=int, default=1,
                   help='Concurrent processes (1 = serial). Each process is one '
                        'training run; round-robin assigned to GPUs in --gpu pool.')
    p.add_argument('--gpu', type=int, nargs='+', default=[0],
                   help='GPU id pool. Workers are assigned round-robin. '
                        'Default [0]. Use [0,1] to spread across two GPUs.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.pretrain_epochs = 1
        args.transfer_epochs = 1

    log_dir = os.path.join(OUTPUT_ROOT, 'logs')

    print(f"Bits      : {args.bits}")
    print(f"Modes     : {args.modes}")
    print(f"Pretrain  : {args.pretrain_epochs} ep  lr={args.pretrain_lr}")
    print(f"Transfer  : {args.transfer_epochs} ep  lr={args.transfer_lr}")
    print(f"Parallel  : {args.parallel} workers  GPUs={args.gpu}")
    print(f"Reuse 1b  : {args.reuse_1bit}")

    # ---- Phase A: build pretrain job list (skip those already done) ----
    pretrain_jobs = []
    for bit in args.bits:
        bp = backbone_path(bit)
        if bit == 1 and args.reuse_1bit and os.path.exists(LEGACY_1BIT_BACKBONE):
            if not os.path.exists(bp):
                shutil.copy2(LEGACY_1BIT_BACKBONE, bp)
            print(f"[Pretrain] Reuse legacy 1-bit -> {bp}")
            continue
        if os.path.exists(bp) and not args.force_pretrain:
            print(f"[Pretrain] Skip (exists): {bp}")
            continue
        pretrain_jobs.append(build_pretrain_cmd(bit, args))

    if not args.skip_pretrain and pretrain_jobs:
        print(f"[Pretrain] Launching {len(pretrain_jobs)} jobs (parallel={args.parallel})")
        run_jobs(pretrain_jobs, args.parallel, args.gpu, log_dir)

    # ---- Phase B: build transfer job list ----
    transfer_jobs = []
    for bit in args.bits:
        if not os.path.exists(backbone_path(bit)):
            print(f"[Transfer] Skip bit={bit} (no backbone)")
            continue
        for mode in args.modes:
            transfer_jobs.append(build_transfer_cmd(bit, mode, args))

    rows = []
    if not args.skip_transfer and transfer_jobs:
        print(f"[Transfer] Launching {len(transfer_jobs)} jobs (parallel={args.parallel})")
        results = run_jobs(transfer_jobs, args.parallel, args.gpu, log_dir)
        # Map result back to (bit, mode)
        for label, acc, params, rc in results:
            # label like TRANSFER_b4_full_ft
            parts = label.split('_')
            bit = int(parts[1][1:])
            mode = '_'.join(parts[2:])
            rows.append({'bit': bit, 'mode': mode, 'acc': acc,
                         'params': params, 'returncode': rc})

    if rows:
        df = pd.DataFrame(rows).sort_values(['bit', 'mode']).reset_index(drop=True)
        csv_path = os.path.join(OUTPUT_ROOT, 'results.csv')
        df.to_csv(csv_path, index=False)
        print(f"[CSV] -> {csv_path}")
        print(df.pivot(index='bit', columns='mode', values='acc'))
        plot_results(df, os.path.join(PLOT_DIR, 'bitwidth_vs_acc.png'))


if __name__ == '__main__':
    main()
