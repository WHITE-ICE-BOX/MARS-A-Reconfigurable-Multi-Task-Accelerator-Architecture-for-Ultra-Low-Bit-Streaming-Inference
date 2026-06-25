"""
Parametrizable CNV with selectable W/A bit-widths (1, 2, 4, 8, 16, 32).

Architecture follows models/CNV.py exactly. The only difference:
- act_bit_width == 1  -> QuantIdentity + CommonActQuant (binary +/-1, original behavior)
- act_bit_width >= 2  -> QuantReLU (unsigned, Brevitas convention for multi-bit)

Adapter components (QuantAdapter / MultiBranchAdapter) are imported from the
original models.CNV unchanged: adapter is always 1-bit in this experiment.
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, BatchNorm2d, MaxPool2d, Module, ModuleList

from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU

from models.common import CommonActQuant, CommonWeightQuant
from models.tensor_norm import TensorNorm
from .conv_adapter import QuantConvAdapter
from .multi_adapter import MultiAdapter

CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
POOL_SIZE = 2
KERNEL_SIZE = 3


def make_act(bit_width):
    """1-bit -> binary signed (HardTanh-like); >=2-bit -> QuantReLU (unsigned)."""
    if bit_width == 1:
        return QuantIdentity(act_quant=CommonActQuant, bit_width=1)
    return QuantReLU(bit_width=bit_width)


class CNVParam(Module):
    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch, adapter_config=None):
        super().__init__()
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()
        self.adapters = ModuleList()
        self.use_adapter = adapter_config is not None

        self.conv_features.append(QuantIdentity(
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=-1.0,
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

            if self.use_adapter and i > 0:
                num_branches = adapter_config.get('num_branches', 1)
                if num_branches > 1:
                    self.adapters.append(MultiAdapter(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        num_branches=num_branches,
                        bit_width=adapter_config['bit_width'],
                        kernel_size=adapter_config.get('kernel_size', 3),
                        act_mode=adapter_config.get('act_mode', 'relu'),
                        alpha_mode=adapter_config.get('alpha_mode', 'per_channel'),
                        use_bias=adapter_config.get('use_bias', False),
                        mid_basis=adapter_config.get('mid_basis', 'out')))
                else:
                    self.adapters.append(QuantConvAdapter(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        bit_width=adapter_config['bit_width'],
                        kernel_size=adapter_config.get('kernel_size', 3),
                        act_mode=adapter_config.get('act_mode', 'relu'),
                        alpha_mode=adapter_config.get('alpha_mode', 'per_channel'),
                        use_bias=adapter_config.get('use_bias', False),
                        mid_basis=adapter_config.get('mid_basis', 'out')))
            elif self.use_adapter:
                self.adapters.append(nn.Identity())

            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(make_act(act_bit_width))
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
            self.linear_features.append(make_act(act_bit_width))

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

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)

        if not self.use_adapter:
            for mod in self.conv_features:
                x = mod(x)
        else:
            adapter_idx = 0
            x = self.conv_features[0](x)
            ptr = 1
            for i, (_, is_pool_enabled) in enumerate(CNV_OUT_CH_POOL):
                x_in = x
                x_conv = self.conv_features[ptr](x); ptr += 1
                if i > 0:
                    x_adp = self.adapters[adapter_idx](x_in)
                    # Adapter 3x3 padding=0 output already matches backbone 3x3 padding=0 spatial size
                    x = x_conv + x_adp
                else:
                    x = x_conv
                adapter_idx += 1
                x = self.conv_features[ptr](x); ptr += 1  # BN
                x = self.conv_features[ptr](x); ptr += 1  # Act
                if is_pool_enabled:
                    x = self.conv_features[ptr](x); ptr += 1  # Pool

        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


def cnv_param(cfg):
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
                'use_rc': cfg.getboolean('ADAPTER', 'USE_RC', fallback=True),
                'kernel_size': cfg.getint('ADAPTER', 'KERNEL_SIZE', fallback=3),
                'act_mode': cfg.get('ADAPTER', 'ACT_MODE', fallback='relu'),
                'alpha_mode': cfg.get('ADAPTER', 'ALPHA_MODE', fallback='per_channel'),
                'use_bias': cfg.getboolean('ADAPTER', 'USE_BIAS', fallback=False),
                'mid_basis': cfg.get('ADAPTER', 'MID_BASIS', fallback='out'),
            }

    return CNVParam(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        num_classes=num_classes,
        in_ch=in_channels,
        adapter_config=adapter_config)
