"""
Forgetting evaluation: source domain (CIFAR-10) preservation after transfer.

For each transferred checkpoint (full_ft or adapter), reconstruct the model,
load weights, and evaluate on CIFAR-10 test set.

Compares:
  - full_ft: expected catastrophic forgetting (low CIFAR-10 acc)
  - adapter: expected preservation (high CIFAR-10 acc, same as backbone)

Output:
  claude/paper_results_bitwidth/forgetting_eval/results.csv
    cell, target_dataset, target_acc, cifar10_acc, forgetting (cifar10_pretrain - cifar10_acc)
"""

import os
import sys
import csv
import argparse
from configparser import ConfigParser

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, THIS_DIR)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models_bitwidth import model_with_cfg_bitwidth
from models_bitwidth.CNV_param import cnv_param

OUTPUT_DIR = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'forgetting_eval')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- cell list --
# Each tuple: (label, ckpt_path, num_branches, kernel, act, alpha, bias, mid_basis, target, target_acc)
# num_branches=0 means no adapter (full_ft/frozen_only/pretrain)
CELLS = [
    # (1) Pretrain backbone reference
    ('CIFAR10_pretrain', os.path.join(THIS_DIR, 'pretrained_backbones/cifar10_1w1a.tar'),
     0, None, None, None, False, None, 'CIFAR10', None),
    # (2) SVHN full_ft (v1/v2 archive)
    ('SVHN_full_ft', os.path.join(THIS_DIR, 'paper_results_bitwidth/archive_v2_2026-05-03/experiments_transfer/Transfer_b1_full_ft_e200/checkpoints/best.tar'),
     0, None, None, None, False, None, 'SVHN', 94.91),
    # (3) SVHN frozen_only (v1/v2 archive) -- backbone only, FC3 retrained
    ('SVHN_frozen_only', os.path.join(THIS_DIR, 'paper_results_bitwidth/archive_v2_2026-05-03/experiments_transfer/Transfer_b1_frozen_only_e200/checkpoints/best.tar'),
     0, None, None, None, False, None, 'SVHN', 28.81),
    # (4) SVHN v1/v2-eq M=1 rc (v7) -- HW-deployed best single-branch equivalent
    ('SVHN_v1v2_M1_rc', os.path.join(THIS_DIR, 'paper_results_bitwidth/v7_multi_rc/experiments/Transfer_v7_v1v2_M1_rc_e200/checkpoints/best.tar'),
     1, 1, 'signed', 'scalar', True, 'in', 'SVHN', 73.72),
    # (5) SVHN v1/v2-eq M=4 rc (v7) -- HW-deployed best multi-branch
    ('SVHN_v1v2_M4_rc', os.path.join(THIS_DIR, 'paper_results_bitwidth/v7_multi_rc/experiments/Transfer_v7_v1v2_M4_rc_e200/checkpoints/best.tar'),
     4, 1, 'signed', 'scalar', True, 'in', 'SVHN', 79.81),
    # (6) SVHN v6-eq M=4 rc (v7) -- SW best
    ('SVHN_v6_M4_rc', os.path.join(THIS_DIR, 'paper_results_bitwidth/v7_multi_rc/experiments/Transfer_v7_v6_M4_rc_e200/checkpoints/best.tar'),
     4, 3, 'relu', 'per_channel', True, 'in', 'SVHN', 83.31),
    # (7) STL10 adapter best (v9)
    ('STL10_v1v2_M4_rc', os.path.join(THIS_DIR, 'paper_results_bitwidth/v9_cross_dataset/experiments/Transfer_v9_STL10_M4_rc_e200/checkpoints/best.tar'),
     4, 1, 'signed', 'scalar', True, 'in', 'STL10', 68.06),
    # (8) STL10 full_ft (v9_ft)
    ('STL10_full_ft', os.path.join(THIS_DIR, 'paper_results_bitwidth/v9_ft_baseline/experiments/Transfer_v9ft_STL10_full_ft_e200/checkpoints/best.tar'),
     0, None, None, None, False, None, 'STL10', 71.05),
    # (9) FashionMNIST adapter best so far (v9, M=3 rc since M=4 still running at write time)
    ('FashionMNIST_v1v2_M3_rc', os.path.join(THIS_DIR, 'paper_results_bitwidth/v9_cross_dataset/experiments/Transfer_v9_FashionMNIST_M3_rc_e200/checkpoints/best.tar'),
     3, 1, 'signed', 'scalar', True, 'in', 'FashionMNIST', 82.82),
    # (10) FashionMNIST full_ft (v9_ft)
    ('FashionMNIST_full_ft', os.path.join(THIS_DIR, 'paper_results_bitwidth/v9_ft_baseline/experiments/Transfer_v9ft_FashionMNIST_full_ft_e200/checkpoints/best.tar'),
     0, None, None, None, False, None, 'FashionMNIST', 92.36),
    # (11) CINIC10 full_ft (v9_ft)
    ('CINIC10_full_ft', os.path.join(THIS_DIR, 'paper_results_bitwidth/v9_ft_baseline/experiments/Transfer_v9ft_CINIC10_full_ft_e200/checkpoints/best.tar'),
     0, None, None, None, False, None, 'CINIC10', 68.02),
]


def build_model_for_cell(num_branches, kernel, act, alpha, bias, mid_basis):
    """Reconstruct the 1-bit CNV with optional adapter, mirroring bnn_pynq_train_bitwidth.build_model."""
    _, cfg = model_with_cfg_bitwidth('cnv_1w1a')
    if num_branches > 0:
        if not cfg.has_section('ADAPTER'):
            cfg.add_section('ADAPTER')
        cfg.set('ADAPTER', 'NUM_BRANCHES', str(num_branches))
        cfg.set('ADAPTER', 'BIT_WIDTH', '1')
        cfg.set('ADAPTER', 'USE_RC', 'False')
        cfg.set('ADAPTER', 'RC_BIT_WIDTH', '8')
        cfg.set('ADAPTER', 'KERNEL_SIZE', str(kernel))
        cfg.set('ADAPTER', 'ACT_MODE', act)
        cfg.set('ADAPTER', 'ALPHA_MODE', alpha)
        cfg.set('ADAPTER', 'USE_BIAS', str(bias))
        cfg.set('ADAPTER', 'MID_BASIS', mid_basis)
    model = cnv_param(cfg)
    if num_branches > 0:
        model.use_adapter = True
    return model


def get_cifar10_test_loader(batch_size, num_workers, datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = CIFAR10(root=datadir, train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate_top1(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def load_pretrain_classifier_state(pretrain_ckpt_path):
    """Pull classifier (linear_features.6 = QuantLinear) and TensorNorm (linear_features.7) state from pretrain checkpoint."""
    ckpt = torch.load(pretrain_ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    keep = {k: v for k, v in sd.items() if k.startswith('linear_features.6') or k.startswith('linear_features.7')}
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--datadir', type=str, default=os.path.join(PROJ_ROOT, 'data'))
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_loader = get_cifar10_test_loader(args.batch_size, args.num_workers, args.datadir)
    print(f"CIFAR-10 test set: {len(test_loader.dataset)} samples")

    pretrain_ckpt_path = os.path.join(THIS_DIR, 'pretrained_backbones/cifar10_1w1a.tar')
    pretrain_classifier_state = load_pretrain_classifier_state(pretrain_ckpt_path)
    print(f"Loaded pretrain classifier state: {list(pretrain_classifier_state.keys())}")

    rows = []
    for label, ckpt_path, num_branches, kernel, act, alpha, bias, mid_basis, target, target_acc in CELLS:
        print(f"\n=== {label} ===")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP: checkpoint not found: {ckpt_path}")
            rows.append({'cell': label, 'target': target, 'target_acc': target_acc,
                         'cifar10_acc_orig_head': None, 'cifar10_acc_pretrain_head': None,
                         'cifar10_acc_no_adapter': None, 'note': 'ckpt_missing'})
            continue

        try:
            model = build_model_for_cell(num_branches, kernel, act, alpha, bias, mid_basis)
            model = model.to(device)

            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('state_dict', ckpt)
            msg = model.load_state_dict(sd, strict=False)
            if msg.missing_keys:
                print(f"  Missing keys ({len(msg.missing_keys)}): {msg.missing_keys[:3]}...")
            if msg.unexpected_keys:
                print(f"  Unexpected keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:3]}...")

            # (a) eval as-is (transferred head, adapter on if present)
            acc_orig = evaluate_top1(model, test_loader, device)

            # (b) swap pretrain classifier head, keep adapter on
            cur_state = model.state_dict()
            for k, v in pretrain_classifier_state.items():
                if k in cur_state:
                    cur_state[k].copy_(v.to(device))
            acc_with_pretrain_head = evaluate_top1(model, test_loader, device)

            # (c) swap pretrain head AND disable adapter (only meaningful if num_branches > 0)
            acc_no_adapter = None
            if num_branches > 0:
                model.use_adapter = False
                acc_no_adapter = evaluate_top1(model, test_loader, device)
                model.use_adapter = True  # restore

            print(f"  acc(transferred head, adapter on)         = {acc_orig:.2f}%")
            print(f"  acc(pretrain head, adapter on)            = {acc_with_pretrain_head:.2f}%")
            if acc_no_adapter is not None:
                print(f"  acc(pretrain head, adapter OFF) ★ key      = {acc_no_adapter:.2f}%  ← backbone preservation test")
            print(f"  (target {target}: {target_acc})")

            rows.append({
                'cell': label, 'target': target, 'target_acc': target_acc,
                'cifar10_acc_orig_head': round(acc_orig, 2),
                'cifar10_acc_pretrain_head': round(acc_with_pretrain_head, 2),
                'cifar10_acc_no_adapter': round(acc_no_adapter, 2) if acc_no_adapter is not None else None,
                'note': ''})
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({'cell': label, 'target': target, 'target_acc': target_acc,
                         'cifar10_acc_orig_head': None, 'cifar10_acc_pretrain_head': None,
                         'cifar10_acc_no_adapter': None, 'note': f'error: {e}'})

    pretrain_row = next((r for r in rows if r['cell'] == 'CIFAR10_pretrain' and r['cifar10_acc_orig_head'] is not None), None)
    pretrain_acc = pretrain_row['cifar10_acc_orig_head'] if pretrain_row else None

    csv_path = os.path.join(OUTPUT_DIR, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell', 'target', 'target_acc',
                    'cifar10_orig_head', 'cifar10_pretrain_head', 'cifar10_no_adapter',
                    'forgetting (pretrain_acc - cifar10_no_adapter or pretrain_head)', 'note'])
        for r in rows:
            ref = r['cifar10_acc_no_adapter'] if r['cifar10_acc_no_adapter'] is not None else r['cifar10_acc_pretrain_head']
            d = ''
            if pretrain_acc is not None and ref is not None and r['cell'] != 'CIFAR10_pretrain':
                d = round(pretrain_acc - ref, 2)
            w.writerow([r['cell'], r['target'], r['target_acc'],
                        r['cifar10_acc_orig_head'], r['cifar10_acc_pretrain_head'], r['cifar10_acc_no_adapter'],
                        d, r['note']])

    print(f"\n[CSV] -> {csv_path}")
    if pretrain_acc is not None:
        print(f"\nPretrain CIFAR-10 baseline (orig head, no adapter): {pretrain_acc:.2f}%")
    print(f"\n{'cell':<32} {'target':<14} {'target_acc':>10} {'orig_head':>10} {'pre_head':>10} {'no_adp':>8} {'forget':>8}")
    for r in rows:
        ref = r['cifar10_acc_no_adapter'] if r['cifar10_acc_no_adapter'] is not None else r['cifar10_acc_pretrain_head']
        d = ''
        if pretrain_acc is not None and ref is not None and r['cell'] != 'CIFAR10_pretrain':
            d = f"{pretrain_acc - ref:+.2f}"
        print(f"{r['cell']:<32} {str(r['target']):<14} {str(r['target_acc']):>10} "
              f"{str(r['cifar10_acc_orig_head']):>10} {str(r['cifar10_acc_pretrain_head']):>10} "
              f"{str(r['cifar10_acc_no_adapter']):>8} {d:>8}")


if __name__ == '__main__':
    main()
