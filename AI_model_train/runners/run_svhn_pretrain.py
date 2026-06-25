"""
Pretrain SVHN 1-bit backbone for reverse-direction transfer experiments.

Mirrors CIFAR10 pretrain hyperparams: 500ep, lr=0.02, STEP@300,400, ADAM, seed=2024.

After completion, copies best.tar to pretrained_backbones/svhn_1w1a.tar.
(The train script auto-copies only for --dataset CIFAR10, so we do it here.)

Output:
  claude/paper_results_bitwidth/svhn_pretrain/
    experiments/Pretrain_svhn_b1_e500/
    logs/svhn_b1.log
  claude/pretrained_backbones/svhn_1w1a.tar  (copied on success)
"""

import argparse
import os
import shutil
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))

OUTPUT_ROOT = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'svhn_pretrain')
EXP_DIR = os.path.join(OUTPUT_ROOT, 'experiments')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'logs')
BACKBONE_DIR = os.path.join(THIS_DIR, 'pretrained_backbones')
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BACKBONE_DIR, exist_ok=True)

BIT = 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    exp_name = f"Pretrain_svhn_b{BIT}_e{args.epochs}"
    label = f"svhn_b{BIT}"
    ms = f"{int(args.epochs * 0.6)},{int(args.epochs * 0.8)}"
    log_path = os.path.join(LOG_DIR, f"{label}.log")

    cmd = [
        sys.executable, '-u', TRAIN_SCRIPT,
        '--mode', 'pretrain',
        '--net_bit', str(BIT),
        '--dataset', 'SVHN',
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

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f">>> [GPU {args.gpu}] {label} <<<")
    print(' '.join(cmd))

    with open(log_path, 'w') as log_fp:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=PROJ_ROOT, env=env)
        for line in proc.stdout:
            log_fp.write(line)
            log_fp.flush()
            print(f"[{label}] {line}", end='')
            sys.stdout.flush()
        rc = proc.wait()

    if rc != 0:
        print(f"FAILED: {label} (exit {rc})")
        sys.exit(rc)

    src = os.path.join(EXP_DIR, exp_name, 'checkpoints', 'best.tar')
    dst = os.path.join(BACKBONE_DIR, f'svhn_{BIT}w{BIT}a.tar')
    if not os.path.exists(src):
        print(f"WARNING: best.tar not found at {src}")
        sys.exit(2)
    shutil.copy2(src, dst)
    print(f"[Copy] backbone -> {dst}")


if __name__ == '__main__':
    main()
