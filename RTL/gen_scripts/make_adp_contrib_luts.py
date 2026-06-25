#!/usr/bin/env python3
# ===========================================================================
# [交接導向註解]
# 腳本：產生 Adapter contribution LUT（.dat），供 Stream_Adder_Threshold 零-DSP 縮放。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

"""Precompute adp_contrib_lut for MVAU1..MVAU4.

Vivado synth refuses to fold (i - OFFSET) * alpha_q8_rom[0] inside an initial
block because alpha_q8_rom is register-backed. The LUT stays at 0 in the
bitstream, so the adapter contribution is always 0. Fix: precompute the
256 entries in software and load them with $readmemh at elaboration time.

LUT entry i stores  (i - OFFSET) * ALPHA_Q8, Q8 popcount domain, signed.
"""
import os
import sys

# (offset, alpha_q8_dat, contrib_lut_dat)
MVAUS = {
    1: (24, "mvau1/data/mvau1_alpha_q8.dat", "mvau1/data/mvau1_contrib_lut.dat"),
    2: (24, "mvau2/data/mvau2_alpha_q8.dat", "mvau2/data/mvau2_contrib_lut.dat"),
    3: (16, "mvau3/data/mvau3_alpha_q8.dat", "mvau3/data/mvau3_contrib_lut.dat"),
    4: (16, "mvau4/data/mvau4_alpha_q8.dat", "mvau4/data/mvau4_contrib_lut.dat"),
}
ROOT = "/home/barkie1/mvau_pipeline/mvau_adapter"


def read_alpha(path):
    with open(path) as f:
        tok = f.read().strip().split()[0]
    v = int(tok, 16)
    if v & 0x80000000:
        v -= 1 << 32
    return v


def to_u32(x):
    if x < 0:
        x += 1 << 32
    return x & 0xFFFFFFFF


def main():
    for n, (offset, alpha_rel, lut_rel) in MVAUS.items():
        alpha_path = os.path.join(ROOT, alpha_rel)
        lut_path = os.path.join(ROOT, lut_rel)
        alpha = read_alpha(alpha_path)
        print(f"MVAU{n}: offset={offset}, alpha_q8={alpha} ({alpha/256.0:+.5f})")
        with open(lut_path, "w") as f:
            for i in range(256):
                entry = (i - offset) * alpha
                f.write(f"{to_u32(entry):08x}\n")
        print(f"  -> {lut_path}  (256 entries)")


if __name__ == "__main__":
    main()
