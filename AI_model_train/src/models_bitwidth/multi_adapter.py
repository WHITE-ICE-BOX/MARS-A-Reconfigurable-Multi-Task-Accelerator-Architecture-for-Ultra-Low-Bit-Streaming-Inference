# ===========================================================================
# [交接導向註解]
# 多分支 Adapter：M 條平行 conv_adapter 分支聚合（對應論文 M=1..4）。
# ===========================================================================

"""
MultiAdapter: parallel composition of N QuantConvAdapter branches.

Each branch is an independent QuantConvAdapter (same kernel/act/alpha/bias settings).
Branches are summed at the output. Each branch carries its own alpha (managed inside
QuantConvAdapter), so the multi-branch output is:

    out = sum_i  branch_i(x)
        = sum_i  alpha_i * up_i(act(down_i(x)))

This is the v7-style multi-adapter used for ablation against the original
MultiBranchAdapter in models/CNV.py (which had the same parallel-sum semantics
but with kernel=1 and only inside QuantAdapter).
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
import torch.nn as nn

from .conv_adapter import QuantConvAdapter


class MultiAdapter(nn.Module):
    """N parallel QuantConvAdapter branches, summed at output."""

    def __init__(self, in_channels, out_channels, num_branches, bit_width,
                 reduction=4, kernel_size=3, act_mode='relu',
                 alpha_mode='per_channel', use_bias=False, mid_basis='out'):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            QuantConvAdapter(
                in_channels=in_channels,
                out_channels=out_channels,
                bit_width=bit_width,
                reduction=reduction,
                kernel_size=kernel_size,
                act_mode=act_mode,
                alpha_mode=alpha_mode,
                use_bias=use_bias,
                mid_basis=mid_basis,
            )
            for _ in range(num_branches)
        ])

    def forward(self, x):
        out = self.branches[0](x)
        for b in self.branches[1:]:
            out = out + b(x)
        return out
