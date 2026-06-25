#!/usr/bin/env python3
# ===========================================================================
# [交接導向註解]
# 切換延遲(switch latency)量測（-> 1.86 ms）。流程：FPGA(compact)。
# ===========================================================================

"""SOFTWARE-ONLY switch-time benchmark + correctness spot-check on board.
Uses the EXISTING resizer_3ds_v3.bit (no rebuild, no RTL). Compares baseline
RuntimeSwitcher.switch (full fancy-scatter write) vs FastRuntimeSwitcher
(delta + static-once + cifar-skip, contiguous slices), and verifies the
fast switch keeps 3-dataset accuracy on real hardware.

Run on board:
  cd /home/xilinx/runtime_3ds_pe1 && echo xilinx | sudo -S XILINX_XRT=/usr \
    BOARD=Pynq-Z2 /usr/local/share/pynq-venv/bin/python3 \
    board_switch_fast_bench.py --spot 2000
"""
import numpy as np, os, sys, time, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from board_test_v3force import run_inference, DATA_DIR, DATASETS
from runtime_3ds import RuntimeSwitcher
from runtime_3ds_fast import FastRuntimeSwitcher

ap = argparse.ArgumentParser()
ap.add_argument("--spot", type=int, default=2000, help="accuracy spot-check samples/ds")
ap.add_argument("--rounds", type=int, default=4)
a = ap.parse_args()

from pynq import Overlay
from pynq.ps import Clocks
bitfile = os.path.join(SCRIPT_DIR, "resizer_3ds_v3.bit")
print(f"Loading {bitfile} (existing bitstream, no rebuild) ...")
ol = Overlay(bitfile); Clocks.fclk0_mhz = 100.0
WR = os.path.join(SCRIPT_DIR, "runtime_weights")
DS = ("cifar10", "svhn", "fashion")

# ---- 1. baseline switch timing (full fancy-scatter, the shipped path) ----
print("\n[1] baseline RuntimeSwitcher.switch (full write):")
base = RuntimeSwitcher(weights_root=WR)
base_ms = {}
for _ in range(a.rounds):
    for d in DS:
        t = time.time(); base.switch(d); ms = (time.time() - t) * 1000
        base_ms.setdefault(d, []).append(ms)
for d in DS:
    print(f"  -> {d:8} min={min(base_ms[d]):.3f} ms  ({len(base._cache[d][0])} writes)")
del base

# ---- 2. fast switch timing (delta + static-once + cifar-skip) ----
print("\n[2] FastRuntimeSwitcher.switch (delta+static+cifar-skip):")
fast = FastRuntimeSwitcher(weights_root=WR)
for d in DS:                       # warmup: first switch fully establishes
    fast.switch(d)
seq = ["cifar10", "svhn", "fashion", "cifar10", "svhn", "cifar10", "fashion", "svhn"]
ft = {}
for _ in range(a.rounds):
    for d in seq:
        ms, n = fast.switch(d)
        ft.setdefault(d, []).append((ms, n))
for d in DS:
    xs = ft[d]
    print(f"  -> {d:8} min={min(x[0] for x in xs):.3f} ms "
          f"mean={sum(x[0] for x in xs)/len(xs):.3f} ms  words~{xs[-1][1]}")

# ---- 3. APPLES-TO-APPLES correctness: baseline vs fast on SAME subset ----
# Proves the reduced fast-switch writes produce identical inference as the
# full baseline switch on the very same images (rules out subset-variance
# ambiguity). They must match within run-to-run noise (ideally exactly).
print(f"\n[3] correctness: baseline vs fast on SAME first-{a.spot} per ds:")
base2 = RuntimeSwitcher(weights_root=WR)
for d in DS:
    tx = np.load(f"{DATA_DIR}/{d}_test_x.npy")
    ty = np.load(f"{DATA_DIR}/{d}_test_y.npy")
    base2.switch(d)
    run_inference(ol, tx[:100], ty[:100], batch_size=100)
    ba, bc, bn, _ = run_inference(ol, tx, ty, batch_size=100, max_samples=a.spot)
    fast.switch(d)
    run_inference(ol, tx[:100], ty[:100], batch_size=100)
    fa, fc, fn, _ = run_inference(ol, tx, ty, batch_size=100, max_samples=a.spot)
    flag = "OK match" if bc == fc else f"MISMATCH (Δ={fc-bc})"
    print(f"  {d:8} baseline={ba:.2f}% ({bc}/{bn})  fast={fa:.2f}% ({fc}/{fn})  -> {flag}")

print("\nDONE_FAST_BENCH")
