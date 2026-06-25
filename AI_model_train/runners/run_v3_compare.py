# ===========================================================================
# [交接導向註解]
# runner：單軸消融：3×3 vs 1×1 down-conv（kernel 軸，50ep no-RC）。
# 流程：AI_model_train。產出對應 results/ 之 results.csv（見 README 對照表）。
# ===========================================================================

"""
v3 comparison sweep: 3x3 vs 1x1 down-conv, full-alignment QuantConvAdapter
(QuantReLU/binary make_act + per-channel alpha), 6 bits each, 50 epochs.

Reuses pretrained backbones from claude/pretrained_backbones/.
Only runs adapter mode -- full_ft and frozen_only numbers are reused from v2.

Output:
  claude/paper_results_bitwidth/v3_compare/
    experiments/Transfer_k{1|3}_b{N}_adapter_50ep/   per-cell logs/checkpoints
    logs/                                            per-cell stdout logs
    results.csv                                      kernel,bit,acc,params,returncode
    plots/v3_compare.png                             summary figure
"""

import argparse
import os
import sys
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))

OUTPUT_ROOT = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'v3_compare')
EXP_DIR = os.path.join(OUTPUT_ROOT, 'experiments')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
PLOT_DIR = os.path.join(OUTPUT_ROOT, 'plots')
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

BITS = [1, 2, 4, 8, 16, 32]
KERNELS = [3, 1]

_PRINT_LOCK = threading.Lock()


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


def build_cmd(bit, kernel, args):
    bp = os.path.join(BACKBONE_DIR, f'cifar10_{bit}w{bit}a.tar')
    exp_name = f"Transfer_k{kernel}_b{bit}_adapter_e{args.epochs}"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'adapter',
        '--net_bit', str(bit),
        '--dataset', 'SVHN',
        '--finetune_checkpoint', bp,
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--scheduler', 'STEP',
        '--milestones', ms,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_seed', str(args.seed),
        '--experiments', EXP_DIR,
        '--experiment_name', exp_name,
        '--num_branches', '1',
        '--adapter_bit_width', str(bit),
        '--adapter_kernel', str(kernel),
        '--no_rc',
    ]
    return f"k{kernel}_b{bit}", cmd


def plot_results(df, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bits_sorted = sorted(df['bit'].unique())
    for k in KERNELS:
        sub = df[df['kernel'] == k].sort_values('bit')
        if len(sub) == 0:
            continue
        ax.plot(sub['bit'], sub['acc'], marker='o', linewidth=2,
                label=f'down kernel={k}x{k}')
    ax.set_xscale('log', base=2)
    ax.set_xticks(bits_sorted)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Backbone bit-width (W & A)', fontsize=12)
    ax.set_ylabel('SVHN test accuracy (%)', fontsize=12)
    ax.set_title('v3 compare: 3x3 vs 1x1 down adapter, '
                 'QuantReLU + per-channel alpha, 50 epochs', fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[Plot] -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--gpu', type=int, nargs='+', default=[0])
    args = p.parse_args()

    print(f"v3 compare: kernels={KERNELS}, bits={BITS}, "
          f"epochs={args.epochs}, parallel={args.parallel}, gpu={args.gpu}")

    jobs = []
    for kernel in KERNELS:
        for bit in BITS:
            jobs.append(build_cmd(bit, kernel, args))

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
        # label: k3_b32 -> kernel=3, bit=32
        parts = label.split('_')
        kernel = int(parts[0][1:])
        bit = int(parts[1][1:])
        rows.append({'kernel': kernel, 'bit': bit, 'acc': acc,
                     'params': params, 'returncode': rc})

    df = pd.DataFrame(rows).sort_values(['kernel', 'bit']).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_ROOT, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[CSV] -> {csv_path}")
    print(df.pivot(index='bit', columns='kernel', values='acc'))
    plot_results(df, os.path.join(PLOT_DIR, 'v3_compare.png'))


if __name__ == '__main__':
    main()
