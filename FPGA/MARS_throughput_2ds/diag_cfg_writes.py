#!/usr/bin/env python3
"""Diagnose if cfg writes to rom_rc/down/up actually take effect.

Test: load SVHN cfg (matching baked SVHN data) → run SVHN, expect ~76%.
      Then OVERWRITE rom_* with Fashion values (which don't match SVHN data).
      Run SVHN → if accuracy drops below 76% → rom_* cfg writes WORK.
                  if accuracy stays 76% → rom_* cfg writes are NO-OP."""
import os, sys, mmap, time, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from pynq import Overlay, allocate
from pynq.ps import Clocks
from runtime_3ds_v1bit import (
    open_cfg_mmio, byte_addr, write_bank, load_bin, write_dataset, run_one
)

def run_acc(ol, name, max_samples=30):
    idma = ol.idma0; odma = ol.odma0
    x = np.load(f"{SCRIPT_DIR}/{name}_test_x.npy")[:max_samples]
    y = np.load(f"{SCRIPT_DIR}/{name}_test_y.npy")[:max_samples]
    c = 0; n = 0; t0 = time.time()
    for i in range(len(x)):
        out = run_one(idma, odma, x[i])
        if out is None: print(f"  TIMEOUT"); break
        if int(out[0]) == int(y[i]): c += 1
        n += 1
    print(f"  acc={c/n*100:.1f}% ({c}/{n})  {n/(time.time()-t0):.1f} FPS")
    return c/n*100 if n > 0 else 0


def main():
    ol = Overlay(f"{SCRIPT_DIR}/resizer_v1.bit")
    Clocks.fclk0_mhz = 100.0
    cfg = open_cfg_mmio()

    print("=== Step 1: Write SVHN cfg, test SVHN (baseline) ===")
    write_dataset(cfg, "svhn", adapter_on=True)
    base_acc = run_acc(ol, "svhn", max_samples=30)

    print("\n=== Step 2: Overwrite ONLY MVAU1-5 rom_rc/down/up with Fashion values ===")
    for mvau in (1, 2, 3, 4, 5):
        for word, suffixes in [(4, ("rom_rc", "rc")),
                                (128, ("rom_down", "down")),
                                (640, ("rom_up", "up"))]:
            for suf in suffixes:
                a = load_bin(f"{SCRIPT_DIR}/mvau{mvau}_{suf}_fashion.bin")
                if a is not None:
                    write_bank(cfg, mvau, word, a)
                    break
    print("  Fashion rom_* written over SVHN's. Now test SVHN again:")
    overwrite_acc = run_acc(ol, "svhn", max_samples=30)

    print(f"\n[Conclusion]")
    print(f"  Baseline SVHN acc:  {base_acc:.1f}%")
    print(f"  After Fashion overwrite SVHN acc: {overwrite_acc:.1f}%")
    if overwrite_acc < base_acc - 10:
        print("  → rom_* cfg writes ARE TAKING EFFECT (drop confirms write)")
    elif overwrite_acc > base_acc - 5:
        print("  → rom_* cfg writes ARE NO-OP (baked SVHN rom_* unchanged)")
    else:
        print("  → ambiguous, need more samples")

if __name__ == "__main__":
    main()
