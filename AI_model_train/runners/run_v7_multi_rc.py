# ===========================================================================
# [交接導向註解]
# runner：多 Adapter(M=1..4) × RC 開關消融。
# 流程：AI_model_train。產出對應 results/ 之 results.csv（見 README 對照表）。
# ===========================================================================

"""
v7 sweep: multi-adapter + RC ablation, 1-bit only.

Three base recipe variants × 4 num_branches × 2 RC settings = 24 cells.

Variants:
  v1v2-eq: kernel=1, signed, scalar α            (matches v1/v2 design)
  v5-eq:   kernel=3, signed, scalar α            (matches v5 design)
  v6-eq:   kernel=3, ReLU/signed, per-channel α  (matches v6 design)

For all cells: mid_channels = in_ch // 4 (HW-friendly, matches v1/v2 original).
"RC" = use_bias on down conv (8-bit Int8Bias).
M = num_branches.

Output:
  claude/paper_results_bitwidth/v7_multi_rc/
    experiments/Transfer_v7_{variant}_M{M}_{rc|norc}_e200/
    logs/{variant}_M{M}_{rc|norc}.log
    results.csv      variant,M,rc,acc,params,returncode
    plots/v7.png     3 subplots (one per variant), x=M, two lines (rc on/off)
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

OUTPUT_ROOT = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'v7_multi_rc')
EXP_DIR = os.path.join(OUTPUT_ROOT, 'experiments')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
PLOT_DIR = os.path.join(OUTPUT_ROOT, 'plots')
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Variant definitions: (kernel, act_mode, alpha_mode)
VARIANTS = {
    'v1v2': {'kernel': 1, 'act_mode': 'signed', 'alpha_mode': 'scalar'},
    'v5':   {'kernel': 3, 'act_mode': 'signed', 'alpha_mode': 'scalar'},
    'v6':   {'kernel': 3, 'act_mode': 'relu',   'alpha_mode': 'per_channel'},
}
M_VALUES = [1, 2, 3, 4]
RC_VALUES = [False, True]   # False = no bias (no RC); True = bias on (RC)
BIT = 1

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


def build_cmd(variant_name, variant_cfg, M, rc_on, args):
    bp = os.path.join(BACKBONE_DIR, f'cifar10_{BIT}w{BIT}a.tar')
    rc_tag = 'rc' if rc_on else 'norc'
    exp_name = f"Transfer_v7_{variant_name}_M{M}_{rc_tag}_e{args.epochs}"
    label = f"{variant_name}_M{M}_{rc_tag}"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'adapter',
        '--net_bit', str(BIT),
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
        '--num_branches', str(M),
        '--adapter_bit_width', str(BIT),
        '--adapter_kernel', str(variant_cfg['kernel']),
        '--adapter_act', variant_cfg['act_mode'],
        '--adapter_alpha', variant_cfg['alpha_mode'],
        '--adapter_mid_basis', 'in',   # HW-friendly, matches v1/v2 original
        '--no_rc',                      # use_rc flag is dead code in legacy paths; harmless
    ]
    if rc_on:
        cmd.append('--adapter_bias')   # this is the actual "RC" knob
    return label, cmd


def plot_results(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, variant in zip(axes, ['v1v2', 'v5', 'v6']):
        sub = df[df['variant'] == variant]
        for rc_on in [False, True]:
            line = sub[sub['rc'] == rc_on].sort_values('M')
            label = 'RC on (bias)' if rc_on else 'RC off (no bias)'
            ax.plot(line['M'], line['acc'], marker='o', linewidth=2, label=label)
        ax.set_xticks(M_VALUES)
        ax.set_xlabel('num_branches (M)')
        ax.set_title(f"{variant} (kernel={VARIANTS[variant]['kernel']}, "
                     f"{VARIANTS[variant]['act_mode']}, {VARIANTS[variant]['alpha_mode']} α)")
        ax.grid(True, alpha=0.4)
        ax.legend()
    axes[0].set_ylabel('SVHN test accuracy (%) — 1-bit')
    fig.suptitle('v7 multi-adapter + RC ablation (1-bit, 200ep)', fontsize=13)
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
    args = p.parse_args()

    print(f"v7 sweep: 3 variants x 4 M x 2 RC = 24 cells, 1-bit, "
          f"epochs={args.epochs}, parallel={args.parallel}, gpu={args.gpu}")

    jobs = []
    for variant_name, variant_cfg in VARIANTS.items():
        for M in M_VALUES:
            for rc_on in RC_VALUES:
                jobs.append(build_cmd(variant_name, variant_cfg, M, rc_on, args))

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
        # label: variant_M{N}_{rc|norc}
        parts = label.split('_')
        variant = parts[0]
        M = int(parts[1][1:])
        rc_on = parts[2] == 'rc'
        rows.append({'variant': variant, 'M': M, 'rc': rc_on, 'acc': acc,
                     'params': params, 'returncode': rc})

    df = pd.DataFrame(rows).sort_values(['variant', 'M', 'rc']).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_ROOT, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[CSV] -> {csv_path}")
    print(df.to_string())
    plot_results(df, os.path.join(PLOT_DIR, 'v7.png'))


if __name__ == '__main__':
    main()
