# ===========================================================================
# [交接導向註解]
# runner：v9 adapter 主實驗（single-seed headline 來源）。
# 流程：AI_model_train。產出對應 results/ 之 results.csv（見 README 對照表）。
# ===========================================================================

"""
v9 sweep: cross-dataset 1×1 transfer, RC × M ablation.

Adapter design fixed = v1/v2 HW-deployed style:
  kernel=1, signed QuantIdentity, scalar α, bias toggleable, mid_basis='in'

Datasets: STL10, FashionMNIST, CINIC10  (CIFAR-10 backbone → each)

Per dataset: 4 M × 2 RC = 8 cells. 3 datasets × 8 = 24 cells. 1-bit only, 200ep.

Output:
  claude/paper_results_bitwidth/v9_cross_dataset/
    experiments/Transfer_v9_{dataset}_M{M}_{rc|norc}_e200/
    logs/{dataset}_M{M}_{rc|norc}.log
    results.csv  dataset,M,rc,acc,params,returncode
    plots/v9.png 3 subplots (one per dataset), x=M, two lines (rc on/off)
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

OUTPUT_ROOT = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'v9_cross_dataset')
EXP_DIR = os.path.join(OUTPUT_ROOT, 'experiments')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
PLOT_DIR = os.path.join(OUTPUT_ROOT, 'plots')
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DATASETS = ['STL10', 'FashionMNIST', 'CINIC10']
M_VALUES = [1, 2, 3, 4]
RC_VALUES = [False, True]
BIT = 1

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


def build_cmd(dataset, M, rc_on, args):
    bp = os.path.join(BACKBONE_DIR, f'cifar10_{BIT}w{BIT}a.tar')
    rc_tag = 'rc' if rc_on else 'norc'
    exp_name = f"Transfer_v9_{dataset}_M{M}_{rc_tag}_e{args.epochs}"
    label = f"{dataset}_M{M}_{rc_tag}"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'adapter',
        '--net_bit', str(BIT),
        '--dataset', dataset,
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
        '--num_branches', str(M),
        '--adapter_bit_width', str(BIT),
        '--adapter_kernel', '1',         # v1/v2 design fixed
        '--adapter_act', 'signed',
        '--adapter_alpha', 'scalar',
        '--adapter_mid_basis', 'in',     # HW-aligned, matches v1/v2 deployed
        '--no_rc',
    ]
    if rc_on:
        cmd.append('--adapter_bias')
    return label, cmd


def plot_results(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        sub = df[df['dataset'] == dataset]
        for rc_on in [False, True]:
            line = sub[sub['rc'] == rc_on].sort_values('M')
            label = 'RC on (bias)' if rc_on else 'RC off (no bias)'
            ax.plot(line['M'], line['acc'], marker='o', linewidth=2, label=label)
        ax.set_xticks(M_VALUES)
        ax.set_xlabel('num_branches (M)')
        ax.set_title(f'CIFAR10 → {dataset}')
        ax.grid(True, alpha=0.4)
        ax.legend()
    axes[0].set_ylabel('Test accuracy (%) — 1-bit')
    fig.suptitle('v9 cross-dataset RC × M ablation (v1/v2 design: 1×1+bias+sign+scalar α, mid=in)',
                 fontsize=12)
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
    p.add_argument('--parallel', type=int, default=3)
    p.add_argument('--gpu', type=int, nargs='+', default=[0])
    p.add_argument('--datasets', type=str, nargs='+', default=DATASETS)
    args = p.parse_args()

    print(f"v9 sweep: {len(args.datasets)} datasets x 4 M x 2 RC = "
          f"{len(args.datasets) * 8} cells, 1-bit, mid_basis=in, "
          f"epochs={args.epochs}, parallel={args.parallel}, gpu={args.gpu}")

    jobs = []
    for dataset in args.datasets:
        for M in M_VALUES:
            for rc_on in RC_VALUES:
                _label, _cmd = build_cmd(dataset, M, rc_on, args)
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
        # label: {dataset}_M{N}_{rc|norc}
        parts = label.rsplit('_', 2)
        dataset = parts[0]
        M = int(parts[1][1:])
        rc_on = parts[2] == 'rc'
        rows.append({'dataset': dataset, 'M': M, 'rc': rc_on, 'acc': acc,
                     'params': params, 'returncode': rc})

    df = pd.DataFrame(rows).sort_values(['dataset', 'M', 'rc']).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_ROOT, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[CSV] -> {csv_path}")
    print(df.to_string())
    plot_results(df, os.path.join(PLOT_DIR, 'v9.png'))


if __name__ == '__main__':
    main()
