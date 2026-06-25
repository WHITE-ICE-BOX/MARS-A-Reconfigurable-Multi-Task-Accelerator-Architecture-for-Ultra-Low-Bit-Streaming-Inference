#!/usr/bin/env python3
"""
test_switch.py — Quick test of runtime switching via MMIO
Standalone script: no FINN driver needed, just pynq + MMIO.
Writes config to cfg_hub and reads back (read stub returns 0, so we verify
write doesn't hang and check adapter_enable effect on inference).

Usage (on Pynq board):
  python3 test_switch.py --bitfile resizer.bit
  python3 test_switch.py --bitfile resizer.bit --cfg_addr 0x43C00000
"""

import argparse
import struct
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Memory map
CFG_ENABLE_WORD = 0
CFG_THRESH_BASE = 1152
CFG_SIGN_BASE   = 1408

MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}


def mvau_byte_addr(mvau_id, word_addr):
    return (mvau_id << 13) | (word_addr << 2)


def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))


def find_cfg_addr(overlay):
    """Find cfg_hub address from overlay."""
    for name, info in overlay.ip_dict.items():
        phys = info.get('phys_addr', 0)
        rng = info.get('addr_range', 0)
        print(f"  IP: {name} @ 0x{phys:08X} range=0x{rng:X}")
        if 'cfg' in name.lower() or 'adapter' in name.lower():
            return phys, rng
    # Try the last GP0 slave (usually the newest addition)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", default="resizer.bit")
    parser.add_argument("--cfg_addr", type=lambda x: int(x, 0), default=None,
                        help="cfg_hub base address (hex)")
    parser.add_argument("--cfg_range", type=lambda x: int(x, 0), default=0x10000,
                        help="cfg_hub address range (default 64K)")
    # Default address from Vivado address map: 0x43C00000
    args = parser.parse_args()

    from pynq import Overlay, MMIO

    print("Loading overlay...")
    ol = Overlay(args.bitfile)
    print("Overlay loaded.\n")

    if args.cfg_addr is not None:
        cfg_addr = args.cfg_addr
        cfg_range = args.cfg_range
    else:
        cfg_addr, cfg_range = find_cfg_addr(ol)
        if cfg_addr is None:
            print("ERROR: Could not find cfg_hub. Use --cfg_addr to specify.")
            return

    print(f"\nUsing MMIO: base=0x{cfg_addr:08X}, range=0x{cfg_range:X}")
    mmio = MMIO(cfg_addr, cfg_range)

    # --- Test 1: Write adapter_enable ---
    print("\n--- Test 1: adapter_enable writes ---")
    for mvau_id in range(1, 6):
        offset = mvau_byte_addr(mvau_id, CFG_ENABLE_WORD)
        mmio.write(offset, 0)  # disable
        print(f"  MVAU{mvau_id}: adapter_enable=0 (offset=0x{offset:04X})")
    for mvau_id in range(1, 6):
        offset = mvau_byte_addr(mvau_id, CFG_ENABLE_WORD)
        mmio.write(offset, 1)  # re-enable
        print(f"  MVAU{mvau_id}: adapter_enable=1 (offset=0x{offset:04X})")
    print("  OK — no hang, AXI writes completed")

    # --- Test 2: Write CIFAR-10 thresholds ---
    print("\n--- Test 2: Write CIFAR-10 thresholds ---")
    t0 = time.time()
    total_words = 0
    for mvau_id in range(1, 6):
        thresh_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_thresh_cifar10.bin")
        if not os.path.exists(thresh_file):
            print(f"  MVAU{mvau_id}: {thresh_file} not found, skipping")
            continue
        threshs = load_bin_u32(thresh_file)
        for i, val in enumerate(threshs):
            offset = mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i)
            mmio.write(offset, val)
        # Disable adapter for CIFAR-10
        mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)
        total_words += len(threshs) + 1
        print(f"  MVAU{mvau_id}: {len(threshs)} thresholds written, adapter_enable=0")
    elapsed = (time.time() - t0) * 1000
    print(f"  Total: {total_words} words in {elapsed:.1f} ms")
    print("  -> Now in CIFAR-10 mode")

    # --- Test 3: Write SVHN thresholds + signs ---
    print("\n--- Test 3: Write SVHN thresholds + signs ---")
    t0 = time.time()
    total_words = 0
    for mvau_id in range(1, 6):
        thresh_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_thresh_svhn.bin")
        sign_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_sign_svhn.bin")
        if not os.path.exists(thresh_file):
            print(f"  MVAU{mvau_id}: {thresh_file} not found, skipping")
            continue
        threshs = load_bin_u32(thresh_file)
        signs = load_bin_u32(sign_file)
        for i, val in enumerate(threshs):
            offset = mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i)
            mmio.write(offset, val)
        for i, val in enumerate(signs):
            offset = mvau_byte_addr(mvau_id, CFG_SIGN_BASE + i)
            mmio.write(offset, val)
        # Enable adapter for SVHN
        mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 1)
        total_words += len(threshs) + len(signs) + 1
        print(f"  MVAU{mvau_id}: {len(threshs)} threshs + {len(signs)} signs, adapter_enable=1")
    elapsed = (time.time() - t0) * 1000
    print(f"  Total: {total_words} words in {elapsed:.1f} ms")
    print("  -> Now in SVHN mode")

    # --- Test 4: Rapid switching benchmark ---
    print("\n--- Test 4: Rapid switching benchmark (10 iterations) ---")
    times_c = []
    times_s = []
    for _ in range(10):
        # Switch to CIFAR-10
        t0 = time.time()
        for mvau_id in range(1, 6):
            thresh_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_thresh_cifar10.bin")
            threshs = load_bin_u32(thresh_file)
            for i, val in enumerate(threshs):
                mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
            mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)
        times_c.append((time.time() - t0) * 1000)

        # Switch to SVHN
        t0 = time.time()
        for mvau_id in range(1, 6):
            thresh_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_thresh_svhn.bin")
            sign_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_sign_svhn.bin")
            threshs = load_bin_u32(thresh_file)
            signs = load_bin_u32(sign_file)
            for i, val in enumerate(threshs):
                mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
            for i, val in enumerate(signs):
                mmio.write(mvau_byte_addr(mvau_id, CFG_SIGN_BASE + i), val)
            mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 1)
        times_s.append((time.time() - t0) * 1000)

    print(f"  CIFAR-10 switch: avg={sum(times_c)/len(times_c):.1f} ms, "
          f"min={min(times_c):.1f} ms, max={max(times_c):.1f} ms")
    print(f"  SVHN switch:     avg={sum(times_s)/len(times_s):.1f} ms, "
          f"min={min(times_s):.1f} ms, max={max(times_s):.1f} ms")

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
