#!/usr/bin/env python3
"""Write garbage to ONE MVAU's rom_* at a time, see if SVHN accuracy drops."""
import os, sys, numpy as np, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from pynq import Overlay
from pynq.ps import Clocks
from runtime_3ds_v1bit import open_cfg_mmio, byte_addr, write_bank, write_dataset, run_one

def run_acc(ol, name, max_samples=30):
    idma = ol.idma0; odma = ol.odma0
    x = np.load(f"{SCRIPT_DIR}/{name}_test_x.npy")[:max_samples]
    y = np.load(f"{SCRIPT_DIR}/{name}_test_y.npy")[:max_samples]
    c = n = 0
    for i in range(len(x)):
        out = run_one(idma, odma, x[i])
        if out is None: print("TIMEOUT"); break
        if int(out[0]) == int(y[i]): c += 1
        n += 1
    return c/n*100 if n > 0 else 0

ol = Overlay(f"{SCRIPT_DIR}/resizer_v1.bit")
Clocks.fclk0_mhz = 100.0
cfg = open_cfg_mmio()

print("=== Baseline: SVHN with proper cfg ===")
write_dataset(cfg, "svhn", adapter_on=True)
base = run_acc(ol, "svhn", 30)
print(f"  acc={base:.1f}%")

# For each MVAU 1-5, overwrite ONLY its rom_rc with garbage (0xDEADBEEF)
GARBAGE = np.full(256, 0xDEADBEEF, dtype=np.uint32)
for mvau in (1, 2, 3, 4, 5):
    # restore SVHN first
    write_dataset(cfg, "svhn", adapter_on=True)
    # Now corrupt ONLY this MVAU's rom_rc
    write_bank(cfg, mvau, 4, GARBAGE[:16])  # 16 hidden_ch entries
    acc = run_acc(ol, "svhn", 30)
    print(f"  MVAU{mvau} rom_rc=garbage: acc={acc:.1f}%  (baseline={base:.1f}%)")

print()
# Also try thresh corruption — known to be writable
write_dataset(cfg, "svhn", adapter_on=True)
write_bank(cfg, 1, 1152, GARBAGE[:64])  # MVAU1 thresh
acc = run_acc(ol, "svhn", 30)
print(f"  MVAU1 thresh=garbage (sanity): acc={acc:.1f}%  (expect big drop)")
