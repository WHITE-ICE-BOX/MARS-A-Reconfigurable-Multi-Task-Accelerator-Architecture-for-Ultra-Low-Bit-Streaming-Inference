# ===========================================================================
# [交接導向註解]
# CNV backbone 定義（Brevitas QuantConv2d/Linear，1W1A）。被訓練與 FINN 匯出共用。
# ===========================================================================

# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList

from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.quant.solver import BiasQuantSolver

try:
    from .common import CommonActQuant, CommonWeightQuant, CommonQuant
    from .tensor_norm import TensorNorm
except ImportError:
    from brevitas_examples.bnn_pynq.models.common import CommonActQuant, CommonWeightQuant, CommonQuant
    from brevitas_examples.bnn_pynq.models.tensor_norm import TensorNorm

# 設定參數 (完全對齊官方)
CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3

# =============================================================================
# Adapter 組件定義 (統一使用純 1-bit，移除 8-bit 切換)
# =============================================================================
class Int8Bias(CommonQuant, BiasQuantSolver):
    bit_width = 8
    requires_input_scale = False
    requires_weight_scale = False
    scaling_const = 1.0

class QuantAdapter(Module):
    def __init__(self, in_channels, out_channels, reduction=4, bit_width=1, rc_bit_width=8):
        super(QuantAdapter, self).__init__()
        reduced_channels = max(1, in_channels // reduction)
        bias_quantizer = Int8Bias if rc_bit_width == 8 else None

        self.down = QuantConv2d(
            in_channels=in_channels, 
            out_channels=reduced_channels, 
            kernel_size=1,            
            bias=True,                
            bias_quant=bias_quantizer, 
            weight_quant=CommonWeightQuant, 
            weight_bit_width=bit_width 
        )
        self.act = QuantIdentity(act_quant=CommonActQuant, bit_width=bit_width)
        self.up = QuantConv2d(
            in_channels=reduced_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            bias=False, 
            weight_quant=CommonWeightQuant, 
            weight_bit_width=bit_width
        )

    def forward(self, x):
        return self.up(self.act(self.down(x)))

class MultiBranchAdapter(Module):
    def __init__(self, in_channels, out_channels, num_branches=1, bit_width=1, rc_bit_width=8, use_rc=True):
        super(MultiBranchAdapter, self).__init__()
        self.use_rc = use_rc
        self.branches = ModuleList()
        self.alphas = nn.ParameterList()
        
        for _ in range(num_branches):
            self.branches.append(QuantAdapter(
                in_channels, 
                out_channels, 
                bit_width=bit_width, 
                rc_bit_width=rc_bit_width 
            ))
            self.alphas.append(nn.Parameter(torch.tensor(1.0)))

    def forward(self, x):
        out = 0
        for i in range(len(self.branches)):
            branch_out = self.branches[i](x)
            alpha_val = self.alphas[i] if self.training else float(self.alphas[i])
            out = out + branch_out * alpha_val
        return out


# =============================================================================
# 主模型 CNV (Backbone 絕對對齊官方設計)
# =============================================================================
class CNV(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch, adapter_config=None):
        super(CNV, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()
        
        self.adapters = ModuleList()
        self.use_adapter = adapter_config is not None

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for i, (out_ch, is_pool_enabled) in enumerate(CNV_OUT_CH_POOL):
            self.conv_features.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            
            # Adapter 插入邏輯 (第一層 i == 0 跳過不生成分支)
            if self.use_adapter and i > 0:
                self.adapters.append(MultiBranchAdapter(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_branches=adapter_config['num_branches'],
                    bit_width=adapter_config['bit_width'],
                    rc_bit_width=adapter_config['rc_bit_width'],
                    use_rc=adapter_config.get('use_rc', True)
                ))
            elif self.use_adapter:
                self.adapters.append(nn.Identity()) # 為第一層站位，確保 Index 對齊

            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=POOL_SIZE))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))

        self.linear_features.append(
            QuantLinear(
                in_features=LAST_FC_IN_FEATURES,
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    # 官方的權重裁剪功能
    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        # 官方的 Tensor 預處理
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        
        if not self.use_adapter:
            # 純 Backbone 模式 (完全等於原版)
            for mod in self.conv_features:
                x = mod(x)
        else:
            # Adapter 融合模式
            adapter_idx = 0
            x = self.conv_features[0](x) # 第 0 層是初始 QuantIdentity
            ptr = 1 
            
            for i, (_, is_pool_enabled) in enumerate(CNV_OUT_CH_POOL):
                x_in = x 
                
                # 1. 官方 Backbone 卷積
                x_conv = self.conv_features[ptr](x); ptr += 1
                
                # 2. Adapter 融合計算 (第一層跳過)
                if i > 0:
                    x_adp = self.adapters[adapter_idx](x_in)
                    x = x_conv + x_adp[:, :, 1:-1, 1:-1]
                else:
                    x = x_conv
                adapter_idx += 1
                
                # 3. 官方後續處理 (BN -> Act -> Pool)
                x = self.conv_features[ptr](x); ptr += 1 # BN
                x = self.conv_features[ptr](x); ptr += 1 # Act
                
                if is_pool_enabled:
                    x = self.conv_features[ptr](x); ptr += 1 # Pool

        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


def cnv(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    
    adapter_config = None
    if cfg.has_section('ADAPTER'):
        num_branches = cfg.getint('ADAPTER', 'NUM_BRANCHES')
        if num_branches > 0:
            adapter_config = {
                'num_branches': num_branches,
                'bit_width': cfg.getint('ADAPTER', 'BIT_WIDTH'),
                'rc_bit_width': cfg.getint('ADAPTER', 'RC_BIT_WIDTH'),
                'use_rc': cfg.getboolean('ADAPTER', 'USE_RC', fallback=True)
            }

    net = CNV(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        num_classes=num_classes,
        in_ch=in_channels,
        adapter_config=adapter_config)
    return net