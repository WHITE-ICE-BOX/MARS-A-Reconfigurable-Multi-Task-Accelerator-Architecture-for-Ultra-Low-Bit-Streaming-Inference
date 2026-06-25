#!/usr/bin/env python3
"""Empirical proof: SegBulkRuntimeSwitcher (full-set contiguous-segment write,
bit-identical cfg to baseline) reproduces the validated 3-dataset accuracy.
Pure software, existing resizer_3ds_v3.bit, no rebuild. @10000 per dataset
to compare directly against the canonical numbers
CIFAR 80.99 / SVHN 73.23 / Fashion 77.68."""
import os, sys, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from board_test_v3force import run_inference, DATA_DIR
from runtime_3ds_segbulk import SegBulkRuntimeSwitcher
from pynq import Overlay
from pynq.ps import Clocks

EXPECT = {"cifar10": 80.99, "svhn": 73.23, "fashion": 77.68}
bit = os.path.join(SCRIPT_DIR, "resizer_3ds_v3.bit")
print(f"Loading {bit} (existing bitstream, no rebuild) ...", flush=True)
ol = Overlay(bit); Clocks.fclk0_mhz = 100.0
sw = SegBulkRuntimeSwitcher(weights_root=os.path.join(SCRIPT_DIR, "runtime_weights"))
print("SegBulkRuntimeSwitcher ready (writes all 6757 words, bit-identical to "
      "baseline, via 33 contiguous segments).", flush=True)

print(f"\n{'ds':9}{'segbulk acc':>14}{'expected':>12}{'Δ':>8}{'switch ms':>11}")
for d in ("cifar10", "svhn", "fashion"):
    tx = np.load(f"{DATA_DIR}/{d}_test_x.npy")
    ty = np.load(f"{DATA_DIR}/{d}_test_y.npy")
    ms, nseg = sw.switch(d)
    run_inference(ol, tx[:100], ty[:100], batch_size=100)            # drain
    acc, c, n, sec = run_inference(ol, tx, ty, batch_size=100, max_samples=10000)
    e = EXPECT[d]
    print(f"{d:9}{acc:12.2f}% {e:10.2f}% {acc-e:+7.2f}{ms:10.3f} ({nseg}seg)",
          flush=True)
print("\nDONE_SEGBULK_ACC", flush=True)
