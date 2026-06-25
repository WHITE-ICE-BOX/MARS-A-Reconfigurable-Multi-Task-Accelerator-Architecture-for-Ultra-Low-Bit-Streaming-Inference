# ===========================================================================
# [交接導向註解]
# 舊版單一 bit-width 訓練入口（保留供參考；主用 bnn_pynq_train_bitwidth.py）。
# ===========================================================================

# bnn_pynq_train.py
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import torch
import random
import numpy as np
import shutil  # [新增] 用於自動搬移檔案
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN
from torch.utils.data import DataLoader
from packaging.version import parse
import torchvision
from models import model_with_cfg
from trainer import Trainer

# === Reproducibility Setup ===
def set_global_seed(seed):
    if seed is not None:
        print(f"🔒 [Seed] Resetting Global Seed to {seed}...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# [工具] 檢查模型初始權重是否一致
def print_weight_checksum(model):
    checksum = 0.0
    count = 0
    for param in model.parameters():
        checksum += torch.sum(torch.abs(param.data)).item()
        count += 1
    print(f"📊 [Checksum] Model Weight Sum: {checksum:.4f} (Params: {count})")
    return checksum

# [工具] 計算並印出模型參數量
def print_model_details(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 [Model Stats] Total Params: {total_params}")
    print(f"📊 [Model Stats] Trainable Params: {trainable_params}")
    return total_params

# === Dataset Definitions ===
class MirrorMNIST(MNIST):
    if parse(torchvision.__version__) < parse('0.9.1'):
        resources = [
            ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")]

class CommonSVHN(SVHN):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        split = "train" if train else "test"
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)

def get_dataloader(args):
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'CIFAR10':
        train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        builder = CIFAR10
        transform_train = train_transforms
    elif args.dataset == 'MNIST':
        builder = MirrorMNIST
        transform_train = transform_to_tensor
    elif args.dataset == 'SVHN':
        train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        builder = CommonSVHN
        transform_train = train_transforms
    else:
        raise Exception("Dataset not supported: {}".format(args.dataset))

    train_set = builder(root=args.datadir, train=True, download=True, transform=transform_train)
    test_set = builder(root=args.datadir, train=False, download=True, transform=transform_to_tensor)
    
    g = torch.Generator()
    if args.random_seed is not None:
        g.manual_seed(args.random_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    return train_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SVHN/CIFAR Training')
    parser.add_argument('--network', type=str, default='CNV', help='Network architecture')
    parser.add_argument('--dataset', type=str, default='SVHN', help='Dataset')
    parser.add_argument('--datadir', type=str, default='./data', help='Dataset root')
    parser.add_argument('--experiments', type=str, default='./experiments', help='Output dir')
    parser.add_argument('--experiment_name', type=str, default='default', help='Exp name')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate') 
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay') 
    parser.add_argument('--optim', type=str, default='ADAM')
    parser.add_argument('--loss', type=str, default='SqrHinge') 
    parser.add_argument('--scheduler', type=str, default='FIXED') 
    parser.add_argument('--milestones', type=str, default='30,40') 
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--export_qonnx', action='store_true')
    parser.add_argument('--export_qcdq_onnx', action='store_true')
    parser.add_argument('--detect_nan', action='store_true')
    parser.add_argument('--strict', action='store_true') 
    
    # New Arguments
    parser.add_argument("--num_branches", type=int, default=0)
    parser.add_argument("--adapter_bit_width", type=int, default=1)
    parser.add_argument("--rc_bit_width", type=int, default=8)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--finetune_checkpoint", type=str, default=None)
    parser.add_argument("--distill_alpha", type=float, default=0.1)
    parser.add_argument("--distill_temp", type=float, default=10.0)
    parser.add_argument("--export_finn_assets", action="store_true")
    parser.add_argument("--no_rc", action="store_true")

    # Override Args
    parser.add_argument("--override_wbw", type=int, default=None)
    parser.add_argument("--override_abw", type=int, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_global_seed(args.random_seed)

    # 1. Config Setup
    model, cfg = model_with_cfg(args.network, args.pretrained)
    
    config_changed = False
    if args.override_wbw is not None:
        cfg.set('QUANT', 'WEIGHT_BIT_WIDTH', str(args.override_wbw))
        config_changed = True
    if args.override_abw is not None:
        cfg.set('QUANT', 'ACT_BIT_WIDTH', str(args.override_abw))
        config_changed = True
        
    if args.num_branches > 0:
        if not cfg.has_section('ADAPTER'): cfg.add_section('ADAPTER')
        cfg.set('ADAPTER', 'NUM_BRANCHES', str(args.num_branches))
        cfg.set('ADAPTER', 'BIT_WIDTH', str(args.adapter_bit_width))
        cfg.set('ADAPTER', 'USE_RC', str(not args.no_rc))
        cfg.set('ADAPTER', 'RC_BIT_WIDTH', str(args.rc_bit_width))
        config_changed = True

    if config_changed or args.network == 'CNV':
        print(f"-> Re-building model with overridden config...")
        from models.CNV import cnv
        set_global_seed(args.random_seed) # 重建前重置種子確保對齊
        model = cnv(cfg)
        if args.num_branches > 0:
            model.use_adapter = True

    # 2. Data Loading
    train_loader, test_loader = get_dataloader(args)
    
    # 3. Trainer Init
    trainer = Trainer(model, train_loader, test_loader, args)
    
    print("--- Pre-training Check ---")
    print_weight_checksum(model)
    print_model_details(model)
    
    # 4. Run Training
    best_acc = trainer.train_model()
    print(f"Final Best Accuracy: {best_acc}%")

    # ============================================================
    # [關鍵自動化] 針對 CIFAR-10 骨幹訓練後的自動歸位邏輯
    # ============================================================
    if args.dataset == 'CIFAR10' and args.experiment_name == 'Cifar10_backbone_exp':
        # 產出位置通常在: ./experiments/Cifar10_backbone_exp/checkpoints/best.tar
        src_path = os.path.join(trainer.checkpoints_dir_path, "best.tar")
        dst_path = os.path.join(args.experiments, "Cifar10_backbone.tar")
        
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"✅ [自動歸位成功] 權重已複製至：{dst_path}")
        else:
            print(f"⚠️ [自動歸位失敗] 找不到產出的 best.tar 於 {src_path}")

if __name__ == '__main__':
    main()