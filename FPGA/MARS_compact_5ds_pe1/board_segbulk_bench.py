#!/usr/bin/env python3
"""Bench: baseline fancy-scatter vs SegBulk full-set contiguous slices.
Both write all 6757 words (bit-identical cfg) — pure software, existing
bitstream. Also #segments per ds. No accuracy run needed (bit-identical to
baseline => same inference by construction)."""
import os, sys, time, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from runtime_3ds import RuntimeSwitcher
from runtime_3ds_segbulk import SegBulkRuntimeSwitcher

from pynq import Overlay
from pynq.ps import Clocks
bit = os.path.join(SCRIPT_DIR, "resizer_3ds_v3.bit")
print(f"Loading {bit} (existing, no rebuild) ...")
ol = Overlay(bit); Clocks.fclk0_mhz = 100.0
WR = os.path.join(SCRIPT_DIR, "runtime_weights")
DS = ("cifar10", "svhn", "fashion")
R = 8

base = RuntimeSwitcher(weights_root=WR)
bt = {}
for _ in range(R):
    for d in DS:
        t = time.time(); base.switch(d); bt.setdefault(d, []).append((time.time()-t)*1000)
del base

seg = SegBulkRuntimeSwitcher(weights_root=WR)
st = {}; nseg = {}
for _ in range(R):
    for d in DS:
        ms, ns = seg.switch(d); st.setdefault(d, []).append(ms); nseg[d] = ns

print(f"\n{'ds':9}{'baseline(scatter)':>20}{'segbulk(full)':>16}{'#seg':>7}{'speedup':>9}")
for d in DS:
    b = min(bt[d]); s = min(st[d])
    print(f"{d:9}{b:18.3f}ms{s:14.3f}ms{nseg[d]:7d}{b/s:8.2f}x")
print("\nDONE_SEGBULK")
