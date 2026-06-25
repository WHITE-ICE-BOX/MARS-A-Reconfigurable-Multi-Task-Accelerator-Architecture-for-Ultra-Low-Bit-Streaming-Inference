# ===========================================================================
# [交接導向註解]
# 損失函數（SqrHinge 等）。
# ===========================================================================

# models/losses.py
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

class squared_hinge_loss(Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        output = 1. - predictions.mul(targets)
        output[output.le(0.)] = 0.
        loss = torch.mean(output.mul(output))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        output = 1. - predictions.mul(targets)
        output[output.le(0.)] = 0.
        grad_output.resize_as_(predictions).copy_(targets).mul_(-2.).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(predictions.numel())
        return grad_output, None

class SqrHingeLoss(nn.Module):
    def __init__(self):
        super(SqrHingeLoss, self).__init__()

    def forward(self, input, target):
        return squared_hinge_loss.apply(input, target)

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0, base_loss='CrossEntropy'):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        
        # [關鍵修正] 允許切換基礎 Task Loss
        if base_loss == 'SqrHinge':
            self.base_criterion = SqrHingeLoss()
        else:
            self.base_criterion = nn.CrossEntropyLoss()

    def forward(self, student_logits, student_features, teacher_logits, teacher_features, target):
        # 1. Task Loss (使用設定好的 base_criterion)
        task_loss = self.base_criterion(student_logits, target)

        # 2. Logits Distillation (KL Divergence)
        distill_loss = self.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1)
        ) * (self.T * self.T)

        # 3. Feature Distillation (MSE)
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
             for sf, tf in zip(student_features, teacher_features):
                 feature_loss += self.mse(sf, tf)

        total_loss = (1. - self.alpha) * task_loss + self.alpha * distill_loss + feature_loss
        return total_loss