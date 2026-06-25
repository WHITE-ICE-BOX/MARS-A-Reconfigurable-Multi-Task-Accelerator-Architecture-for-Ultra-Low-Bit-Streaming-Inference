# ===========================================================================
# [交接導向註解]
# runner：full-FT × bit-width（2..32-bit）。
# 流程：AI_model_train。產出對應 results/ 之 results.csv（見 README 對照表）。
# ===========================================================================

"""
v9ft_cross_bit: full_ft baseline x bit-width x 3 new datasets.
Already have 1-bit from v9_ft. This sweep adds 2/4/8/16/32-bit.

3 datasets x 5 bits (2,4,8,16,32) = 15 cells, full_ft mode, 200ep.

Output:
  claude/paper_results_bitwidth/v9ft_cross_bit/
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

OUTPUT_ROOT = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'v9ft_cross_bit')
EXP_DIR = os.path.join(OUTPUT_ROOT, 'experiments')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DATASETS = ['STL10', 'FashionMNIST', 'CINIC10']
BITS = [2, 4, 8, 16, 32]   # 1-bit already in v9_ft

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


def build_cmd(dataset, bit, args):
    bp = os.path.join(BACKBONE_DIR, f'cifar10_{bit}w{bit}a.tar')
    exp_name = f"Transfer_v9ftbit_{dataset}_b{bit}_full_ft_e{args.epochs}"
    label = f"{dataset}_b{bit}"
    ms = f"{int(args.epochs * 0.5)},{int(args.epochs * 0.75)}"
    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'full_ft',
        '--net_bit', str(bit),
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
    ]
    return label, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--gpu', type=int, nargs='+', default=[0])
    args = p.parse_args()

    print(f"v9ft_cross_bit sweep: 3 datasets x 5 bits (2..32) = 15 cells full_ft, "
          f"epochs={args.epochs}, parallel={args.parallel}, gpu={args.gpu}")

    jobs = []
    for dataset in DATASETS:
        for bit in BITS:
            _label, _cmd = build_cmd(dataset, bit, args)
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
        parts = label.rsplit('_', 1)
        dataset = parts[0]
        bit = int(parts[1][1:])
        rows.append({'dataset': dataset, 'bit': bit, 'acc': acc,
                     'params': params, 'returncode': rc})

    df = pd.DataFrame(rows).sort_values(['dataset', 'bit']).reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_ROOT, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[CSV] -> {csv_path}")
    print(df.to_string())


if __name__ == '__main__':
    main()
