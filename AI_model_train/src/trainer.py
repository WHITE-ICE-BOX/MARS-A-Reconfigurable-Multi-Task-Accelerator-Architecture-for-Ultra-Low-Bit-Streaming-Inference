# ===========================================================================
# [交接導向註解]
# 訓練/評估迴圈：forward/backward、scheduler、checkpoint 存取、輸出 Final Best Accuracy。
# ===========================================================================

# trainer.py
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime
import os
import random
import time
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import brevitas.config as config
from models import model_with_cfg
from models.losses import SqrHingeLoss, DistillationLoss
from models.CNV import MultiBranchAdapter
from logger import EvalEpochMeters, Logger, TrainingEpochMeters
import numpy as np 

# [新增] 內部也需要重置 Seed
def set_global_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device(gpu_id=1):
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if gpu_id >= n: gpu_id = 0
        
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        print(f"✅ 使用 CUDA:{gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ 使用 Apple MPS")
        return torch.device("mps")
    else:
        print("⚠️ 使用 CPU")
        return torch.device("cpu")

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# [關鍵修正] 1-bit 權重匯出邏輯 (Fix for FPGA Hardware)
def save_tensor_to_dat(tensor, filepath, bit_width=8):
    if tensor is None: return
    
    # 確保轉成 numpy array 並且攤平
    data = tensor.detach().cpu().numpy().flatten()
    
    with open(filepath, 'w') as f:
        for val in data:
            # --- 針對 1-bit (Bipolar) 的特殊編碼處理 ---
            if bit_width == 1:
                # Brevitas 1-bit 權重通常是 +1/-1
                # FPGA 硬體 (XNOR Engine) 通常預期: +1 -> 1, -1 -> 0
                if val >= 0:
                    int_val = 1
                else:
                    int_val = 0
            else:
                # 針對多位元 (例如 8-bit Bias/Weight)
                # 四捨五入取整
                int_val = int(round(val))
            
            # Mask 處理 (確保負數正確轉成補數 Hex，例如 -1 在 8bit 變成 0xff)
            mask = (1 << bit_width) - 1
            hex_val = "{:02x}".format(int_val & mask)
            f.write(hex_val + "\n")

class Trainer(object):
    def __init__(self, model, train_loader, test_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.validate(args)

        experiment_name = args.experiment_name if args.experiment_name != 'default' else \
                          f"{args.network}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.output_dir_path = os.path.join(args.experiments, experiment_name)
        if self.args.resume:
            self.output_dir_path, _ = os.path.split(args.resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)

        if not args.dry_run:
            self.checkpoints_dir_path = os.path.join(self.output_dir_path, 'checkpoints')
            if not args.resume:
                os.makedirs(self.output_dir_path, exist_ok=True)
                os.makedirs(self.checkpoints_dir_path, exist_ok=True)
        
        self.logger = Logger(self.output_dir_path, args.dry_run)
        self.device = get_device()
        
        self.dataset_name = args.dataset
        self.num_classes = 10 
        self.starting_epoch = 1
        self.best_val_acc = 0

        self.teacher_model = None
        if hasattr(args, 'teacher_checkpoint') and args.teacher_checkpoint:
            print(f"======== Loading Teacher from {args.teacher_checkpoint} ========")
            teacher_ckpt = torch.load(args.teacher_checkpoint, map_location='cpu')
            
            self.teacher_model, _ = model_with_cfg(args.network, False)
            self.teacher_model.load_state_dict(teacher_ckpt['state_dict'], strict=False)
            self.teacher_model.eval()
            self.teacher_model.to(self.device)
            
            if args.random_seed is not None:
                print("🔄 [Seed] Reseeding after Teacher Init to ensure consistency...")
                set_global_seed(args.random_seed)

        if args.resume:
            self.model = self.load_checkpoint(self.model, args.resume, args.strict)
        
        # [關鍵修正] Finetune Checkpoint 載入邏輯 (防止 module. 前綴問題)
        if hasattr(args, 'finetune_checkpoint') and args.finetune_checkpoint:
            print(f"======== Loading Backbone from {args.finetune_checkpoint} ========")
            checkpoint = torch.load(args.finetune_checkpoint, map_location='cpu')
            
            # 1. 取得 state_dict
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # 2. [修正] 處理 'module.' 前綴問題 (DataParallel 產生的)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
            state_dict = new_state_dict

            # 3. 移除 FC3
            keys_to_remove = [k for k in state_dict.keys() if 'linear_features.6' in k]
            for k in keys_to_remove: del state_dict[k]
            
            # 4. [修正] 載入並檢查是否成功
            msg = self.model.load_state_dict(state_dict, strict=False)
            
            # 5. [關鍵] 檢查 Backbone (ConvFeatures) 是否真的載入了
            missing_backbone = [k for k in msg.missing_keys if 'conv_features' in k]
            if len(missing_backbone) > 0:
                print(f"❌ [Error] Missing Keys: {missing_backbone[:5]}...")
                raise RuntimeError("❌ 嚴重錯誤！Backbone 權重載入失敗！請檢查 Checkpoint Key 是否匹配 (例如 module. 前綴)。")
            else:
                print("✅ Backbone 權重載入成功！(Verified)")

        if hasattr(args, 'freeze_backbone') and args.freeze_backbone:
            print("======== Freezing Backbone (Conv1-6, FC1-2) ========")
            for name, param in self.model.named_parameters():
                if 'adapters' in name or 'linear_features.6' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.model.to(self.device)

        if self.teacher_model:
            print(f"-> Using Distillation Loss (Base: {args.loss})")
            print(f"   ★ Config: Alpha={args.distill_alpha}, Temp={args.distill_temp}")
            
            # [修改] 使用 args 傳入的參數
            self.criterion = DistillationLoss(
                alpha=args.distill_alpha, 
                temperature=args.distill_temp, 
                base_loss=args.loss
            )
            self.model.return_features = True
            self.teacher_model.return_features = True
        elif args.loss == 'SqrHinge':
            self.criterion = SqrHingeLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if args.optim == 'ADAM':
            self.optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'SGD':
            self.optimizer = optim.SGD(trainable_params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        if args.scheduler == 'STEP':
            milestones = [int(i) for i in args.milestones.split(',')]
            self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=milestones, gamma=0.1)
        else:
            self.scheduler = None

    def validate(self, args):
        pass

    def load_checkpoint(self, model, checkpoint_path, strict):
        print('Loading model checkpoint at: {}'.format(checkpoint_path))
        package = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(package['state_dict'], strict=strict)
        return model

    def checkpoint_best(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            'state_dict': self.model.state_dict(),
            'epoch': epoch,
            'best_val_acc': self.best_val_acc,
            'optimizer': self.optimizer.state_dict(),
        }, best_path)

        if self.args.export_finn_assets:
            full_path = os.path.join(self.checkpoints_dir_path, "full.tar")
            torch.save({
                'state_dict': self.model.state_dict(),
                'epoch': epoch,
                'best_val_acc': self.best_val_acc,
                'optimizer': self.optimizer.state_dict(),
            }, full_path)

    def train_model(self):
        if self.args.random_seed is not None:
            set_global_seed(self.args.random_seed)

        for epoch in range(self.starting_epoch, self.args.epochs + 1):
            self.model.train()
            self.criterion.train()
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()
            
            for i, data in enumerate(self.train_loader):
                (input, target) = data
                input, target = input.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                is_sqr_hinge = isinstance(self.criterion, SqrHingeLoss)
                if isinstance(self.criterion, DistillationLoss):
                    is_sqr_hinge = isinstance(self.criterion.base_criterion, SqrHingeLoss)

                if is_sqr_hinge:
                    target_var = target.unsqueeze(1)
                    target_onehot = torch.Tensor(target_var.size(0), self.num_classes).to(self.device).fill_(-1)
                    target_onehot.scatter_(1, target_var, 1)
                    target_var = target_onehot
                else:
                    target_var = target

                epoch_meters.data_time.update(time.time() - start_data_loading)
                start_batch = time.time()
                
                output = self.model(input)
                
                if self.teacher_model:
                    s_logits, s_feats = output
                    with torch.no_grad():
                        t_logits, t_feats = self.teacher_model(input)
                    loss = self.criterion(s_logits, s_feats, t_logits, t_feats, target_var)
                    output = s_logits 
                else:
                    loss = self.criterion(output, target_var)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if hasattr(self.model, 'clip_weights'):
                    self.model.clip_weights(-1, 1)

                epoch_meters.batch_time.update(time.time() - start_batch)
                
                if i % int(self.args.log_freq) == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(epoch_meters, epoch, i, len(self.train_loader))
                start_data_loading = time.time()

            if self.scheduler:
                self.scheduler.step(epoch)
            
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            if top1avg >= self.best_val_acc and not self.args.dry_run:
                self.best_val_acc = top1avg
                self.checkpoint_best(epoch, "best.tar")
                
                if self.args.export_finn_assets:
                    self.save_components(epoch)
                    self.save_finn_backbone(epoch, top1avg)

            elif not self.args.dry_run:
                self.checkpoint_best(epoch, "checkpoint.tar")
        
        return self.best_val_acc

    def eval_model(self, epoch=None):
        eval_meters = EvalEpochMeters()
        self.model.eval()
        self.criterion.eval()
        for i, data in enumerate(self.test_loader):
            (input, target) = data
            input, target = input.to(self.device), target.to(self.device)
            output = self.model(input)
            if isinstance(output, tuple): output = output[0] 
            loss = nn.CrossEntropyLoss()(output, target) 
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))
        return eval_meters.top1.avg

    def save_finn_backbone(self, epoch, acc):
        full_dict = self.model.state_dict()
        clean_dict = {k: v for k, v in full_dict.items() if 'adapters' not in k}
        finn_export_dir = os.path.join(self.output_dir_path, 'finn_export')
        os.makedirs(finn_export_dir, exist_ok=True)
        save_path = os.path.join(finn_export_dir, "svhn.tar")
        torch.save({'epoch': epoch, 'state_dict': clean_dict, 'best_val_acc': acc}, save_path)
        print(f"-> [Export] Saved clean backbone to {save_path}")

    # [關鍵修正] 權重匯出函式 (使用 quant_weight().value 來獲取真正參與運算的權重)
    def save_components(self, epoch):
        if hasattr(self.args, 'adapter_bit_width') and self.args.adapter_bit_width > 1:
            # 如果是多位元 Adapter，目前的邏輯可能需要微調，但你的實驗主要是 1-bit
            pass 

        finn_export_dir = os.path.join(self.output_dir_path, 'finn_export')
        hw_params_dir = os.path.join(finn_export_dir, 'hardware_params')
        os.makedirs(hw_params_dir, exist_ok=True)
        
        # 1. 儲存 FC3 (通常還是用 PyTorch 格式，因為是 float 或 int8 訓練的)
        state_dict = self.model.state_dict()
        fc3_dict = {k: v for k, v in state_dict.items() if 'linear_features.6' in k}
        torch.save(fc3_dict, os.path.join(hw_params_dir, 'fc3_svhn_weights.pth'))
        
        # 2. 儲存 Adapter 組件 (Down/Up/RC)
        for name, module in self.model.named_modules():
            if isinstance(module, MultiBranchAdapter):
                layer_id = name.split('.')[-1]
                for b_idx, branch in enumerate(module.branches):
                    # --- 匯出 .pth (給 PyTorch 驗證用) ---
                    torch.save(branch.down.state_dict(), f"{hw_params_dir}/adapter{layer_id}_m{b_idx}_down.pth")
                    torch.save(branch.up.state_dict(),   f"{hw_params_dir}/adapter{layer_id}_m{b_idx}_up.pth")
                    
                    # --- 匯出 .dat (給 FPGA Verilog 用) ---
                    # [修正] 獲取「量化後」的權重 (Quantized Weights)
                    # 注意：Brevitas 的 quant_weight() 可能回傳 TensorQuant，要取 .value 拿到 Tensor
                    # 如果還沒呼叫過 forward，可能需要先 trigger 一次，但在 save 時應該都跑過訓練了
                    
                    w_down = branch.down.quant_weight()
                    if hasattr(w_down, 'value'): w_down = w_down.value
                    
                    w_up = branch.up.quant_weight()
                    if hasattr(w_up, 'value'): w_up = w_up.value
                    
                    # 判斷 bit-width (Down 層如果輸入是 8-bit，權重也可能是 8-bit)
                    down_bit = branch.down.weight_bit_width if hasattr(branch.down, 'weight_bit_width') else 8
                    up_bit = branch.up.weight_bit_width if hasattr(branch.up, 'weight_bit_width') else 1

                    save_tensor_to_dat(w_down, f"{hw_params_dir}/adapter{layer_id}_m{b_idx}_down.dat", bit_width=down_bit)
                    save_tensor_to_dat(w_up,   f"{hw_params_dir}/adapter{layer_id}_m{b_idx}_up.dat",   bit_width=up_bit)

                    # [修正] 匯出 Bias (RC)
                    if branch.down.bias is not None:
                        # 嘗試獲取量化後的 Bias
                        if hasattr(branch.down, 'quant_bias'):
                            bias_val = branch.down.quant_bias()
                            if hasattr(bias_val, 'value'): bias_val = bias_val.value
                        else:
                            bias_val = branch.down.bias # Fallback

                        torch.save({'beta': branch.down.bias}, f"{hw_params_dir}/rc{layer_id}_m{b_idx}.pth")
                        # RC bit width 這裡假設是 8
                        save_tensor_to_dat(bias_val, f"{hw_params_dir}/rc{layer_id}_m{b_idx}.dat", bit_width=8)

    def export_qonnx(self): pass 
    def export_qcdq_onnx(self): pass