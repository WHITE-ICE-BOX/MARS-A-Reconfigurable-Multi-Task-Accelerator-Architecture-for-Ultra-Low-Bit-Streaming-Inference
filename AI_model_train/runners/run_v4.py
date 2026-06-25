"""
v4 sweep: 3x3 down adapter + signed QuantIdentity activation + scalar alpha.

Combines v2 adapter design (signed act + scalar alpha, used in QuantAdapter/MultiBranchAdapter)
with v3's 3x3 down kernel. The other v3 piece (QuantReLU + per-channel alpha) is dropped.

Bits processed in order [1, 2, 4, 8, 16, 32] (1-bit first, per user request).

Reuses pretrained backbones from claude/pretrained_backbones/ (same as v3).

Output:
  claude/paper_results_bitwidth/v4/
    experiments/Transfer_v4_b{N}_adapter_e200/   per-cell logs/checkpoints
    logs/v4_b{N}.log                             per-cell stdout (orchestrator copy)
    results.csv                                  bit,acc,params,returncode
    plots/v4.png                                 vs full_ft / v3 k=3
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

OUTPUT_ROOT = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'v4')
EXP_DIR = os.path.join(OUTPUT_ROOT, 'experiments')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
PLOT_DIR = os.path.join(OUTPUT_ROOT, 'plots')
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

BITS = [1, 2, 4, 8, 16, 32]   # 1-bit first

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


def build_cmd(bit, args):
    bp = os.path.join(BACKBONE_DIR, f'cifar10_{bit}w{bit}a.tar')
    exp_name = f"Transfer_v4_b{bit}_adapter_e{args.epochs}"
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
        '--adapter_kernel', '3',
        '--adapter_act', 'signed',
        '--adapter_alpha', 'scalar',
        '--no_rc',
    ]
    return f"v4_b{bit}", cmd


def plot_results(df, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bits_sorted = sorted(df['bit'].unique())
    sub = df.sort_values('bit')
    ax.plot(sub['bit'], sub['acc'], marker='o', linewidth=2, label='v4: 3x3 + signed + scalar α')
    ax.set_xscale('log', base=2)
    ax.set_xticks(bits_sorted)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Backbone bit-width (W & A)', fontsize=12)
    ax.set_ylabel('SVHN test accuracy (%)', fontsize=12)
    ax.set_title('v4: 3x3 down + signed QuantIdentity + scalar α', fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[Plot] -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--gpu', type=int, nargs='+', default=[0])
    p.add_argument('--bits', type=int, nargs='+', default=BITS)
    args = p.parse_args()

    print(f"v4 sweep: bits={args.bits}, epochs={args.epochs}, "
          f"parallel={args.parallel}, gpu={args.gpu}")

    jobs = [build_cmd(bit, args) for bit in args.bits]

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
        # label: v4_b32 -> bit=32
        bit = int(label.split('_b')[-1])
        rows.append({'bit': bit, 'acc': acc, 'params': params, 'returncode': rc})

    df = pd.DataFrame(rows).sort_values('bit').reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_ROOT, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[CSV] -> {csv_path}")
    print(df)
    plot_results(df, os.path.join(PLOT_DIR, 'v4.png'))


if __name__ == '__main__':
    main()
