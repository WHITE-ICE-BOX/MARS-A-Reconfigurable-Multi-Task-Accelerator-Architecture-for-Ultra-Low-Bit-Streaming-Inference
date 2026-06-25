# ===========================================================================
# [交接導向註解]
# runner：預訓練 5 個 1W1A backbone（每個 source 一個）。
# 流程：AI_model_train。產出對應 results/ 之 results.csv（見 README 對照表）。
# ===========================================================================

"""
Generic backbone pretrainer: pretrain any source dataset at any list of bit-widths.

Usage:
  python3 claude/run_xx_pretrain.py --source SVHN --bits 2 4 8 16 32 --parallel 3
  python3 claude/run_xx_pretrain.py --source STL10 --bits 1 2 4 8 16 32 --parallel 3

Output:
  claude/paper_results_bitwidth/{source.lower()}_pretrain/experiments/Pretrain_{src}_b{N}_e500/
  claude/pretrained_backbones/{source.lower()}_{N}w{N}a.tar   (copied on success)
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

os.makedirs(BACKBONE_DIR, exist_ok=True)

_PRINT_LOCK = threading.Lock()


def backbone_path(source, bit):
    return os.path.join(BACKBONE_DIR, f'{source.lower()}_{bit}w{bit}a.tar')


def is_done(source, bit):
    return os.path.exists(backbone_path(source, bit))


def stream_cmd(cmd, label, log_path, gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    log_fp = open(log_path, 'w')
    with _PRINT_LOCK:
        print(f"\n>>> [GPU {gpu_id}] {label} <<<")
        print(' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=PROJ_ROOT, env=env)
    tag = f"[{label}]"
    try:
        for line in proc.stdout:
            log_fp.write(line); log_fp.flush()
            with _PRINT_LOCK:
                print(f"{tag} {line}", end='')
                sys.stdout.flush()
    finally:
        rc = proc.wait()
        log_fp.close()
    return rc


def run_one(source, bit, args, gpu_id, output_root):
    exp_dir = os.path.join(output_root, 'experiments')
    log_dir = os.path.join(output_root, 'logs')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    exp_name = f"Pretrain_{source.lower()}_b{bit}_e{args.epochs}"
    label = f"{source.lower()}_b{bit}"
    ms = f"{int(args.epochs * 0.6)},{int(args.epochs * 0.8)}"
    log_path = os.path.join(log_dir, f"{label}.log")

    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'pretrain',
        '--net_bit', str(bit),
        '--dataset', source,
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--scheduler', 'STEP',
        '--milestones', ms,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--random_seed', str(args.seed),
        '--experiments', exp_dir,
        '--experiment_name', exp_name,
    ]
    rc = stream_cmd(cmd, label, log_path, gpu_id)
    if rc != 0:
        with _PRINT_LOCK:
            print(f"FAILED: {label} (exit {rc})")
        return label, rc

    # Copy backbone (the train script only auto-copies for CIFAR10)
    if source != 'CIFAR10':
        src = os.path.join(exp_dir, exp_name, 'checkpoints', 'best.tar')
        dst = backbone_path(source, bit)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            with _PRINT_LOCK:
                print(f"[Copy] {label} backbone -> {dst}")
        else:
            with _PRINT_LOCK:
                print(f"WARNING: {label} best.tar not found at {src}")
            return label, 2
    return label, rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=str, required=True,
                   choices=['CIFAR10', 'SVHN', 'STL10', 'FashionMNIST', 'CINIC10'])
    p.add_argument('--bits', type=int, nargs='+', required=True)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--gpu', type=int, nargs='+', default=[0])
    args = p.parse_args()

    output_root = os.path.join(THIS_DIR, 'paper_results_bitwidth',
                               f'{args.source.lower()}_pretrain')
    os.makedirs(output_root, exist_ok=True)

    todo = []
    for bit in args.bits:
        if is_done(args.source, bit):
            print(f"SKIP done: {args.source} b{bit} (backbone exists)")
            continue
        todo.append(bit)

    print(f"{args.source} pretrain: bits={todo}, parallel={args.parallel}, gpu={args.gpu}, "
          f"epochs={args.epochs}")

    gpu_iter = cycle(args.gpu)
    job_gpu = [next(gpu_iter) for _ in todo]

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = []
        for i, bit in enumerate(todo):
            futs.append(pool.submit(run_one, args.source, bit, args, job_gpu[i], output_root))
        for fut in as_completed(futs):
            label, rc = fut.result()
            print(f"DONE {label} rc={rc}")


if __name__ == '__main__':
    main()
