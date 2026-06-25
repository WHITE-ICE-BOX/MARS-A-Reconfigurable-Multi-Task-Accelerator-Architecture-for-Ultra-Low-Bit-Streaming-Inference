#!/usr/bin/env python3
"""Test 3-dataset runtime switching using v1 bitstream + cfg writes."""
import os, sys, mmap, time, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from pynq import Overlay, allocate
from pynq.ps import Clocks

# v1 cfg_hub at 0x43C00000 (16-bit address space = 64KB)
CFG_BASE = 0x43C00000

# Unit-shift = 13 bits. Unit i → byte offset (i << 13).
# Per-MVAU word addresses inside each unit's 8KB window:
#   word 0          : adapter_enable
#   word 4..       : rom_rc
#   word 128..     : rom_down
#   word 640..     : rom_up (one 32-bit cfg slot per (out_step, pe))
#   word 1152..    : thresh
#   word 1408..    : sign
#   word 1664..    : contrib_lut

MVAU_PARAMS = {
    1: dict(IN_CH=64,  OUT_CH=64,  PE=32, SIMD=32, HIDDEN_CH=16),
    2: dict(IN_CH=64,  OUT_CH=128, PE=16, SIMD=32, HIDDEN_CH=16),
    3: dict(IN_CH=128, OUT_CH=128, PE=16, SIMD=32, HIDDEN_CH=32),
    4: dict(IN_CH=128, OUT_CH=256, PE=8,  SIMD=32, HIDDEN_CH=32),
    5: dict(IN_CH=256, OUT_CH=256, PE=8,  SIMD=32, HIDDEN_CH=64),
}
for n, p in MVAU_PARAMS.items():
    p["IN_CHUNKS"]  = p["IN_CH"] // p["SIMD"]
    p["OUT_STEPS"]  = p["OUT_CH"] // p["PE"]


def byte_addr(unit, word):
    return (unit << 13) | (word << 2)


def open_cfg_mmio():
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    m = mmap.mmap(fd, 0x10000, offset=CFG_BASE)
    os.close(fd)
    return m


def write_bank(cfg, unit, word_base, vals):
    """vals: 1D array of uint32. Writes consecutive cfg words from word_base."""
    arr = np.asarray(vals, dtype=np.uint32)
    # Build a uint32 array view of the MMIO at the right offset.
    addr = byte_addr(unit, word_base)
    view = np.frombuffer(cfg, dtype=np.uint32)
    n_words = len(arr)
    start_word = addr // 4
    view[start_word:start_word + n_words] = arr


def load_bin(path):
    """Load a .bin file as uint32 LE array."""
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.uint32)


def write_dataset(cfg, name, adapter_on):
    """Write all cfg banks for the given dataset.
    Files used (each dataset has its own .bin per bank):
      mvauN_rom_rc_<name>.bin  /  mvauN_rom_down_<name>.bin  /  mvauN_rom_up_<name>.bin
      mvauN_thresh_<name>.bin  /  mvauN_sign_<name>.bin  /  mvauN_lut_<name>.bin
      mvau0_thresh_<name>.bin  fc1_thresh_<name>.bin  fc2_thresh_<name>.bin  cls_weights_<name>.bin
    """
    t0 = time.time()
    n_writes = 0
    # MVAU0 thresholds (unit 0 low half, word 0..63)
    a = load_bin(f"{SCRIPT_DIR}/mvau0_thresh_{name}.bin")
    if a is not None:
        write_bank(cfg, 0, 0, a); n_writes += len(a)

    # Classifier weights (unit 0 high half, word offset 0x1000/4 = 1024)
    a = load_bin(f"{SCRIPT_DIR}/cls_weights_{name}.bin")
    if a is not None:
        write_bank(cfg, 0, 1024, a); n_writes += len(a)

    # MVAU1-5 banks
    for mvau in (1, 2, 3, 4, 5):
        # adapter_enable
        view = np.frombuffer(cfg, dtype=np.uint32)
        view[byte_addr(mvau, 0) // 4] = 1 if adapter_on else 0
        n_writes += 1

        # rom_rc @ word 4
        for fname in (f"mvau{mvau}_rom_rc_{name}.bin", f"mvau{mvau}_rc_{name}.bin"):
            a = load_bin(f"{SCRIPT_DIR}/{fname}")
            if a is not None:
                write_bank(cfg, mvau, 4, a); n_writes += len(a); break
        # rom_down @ word 128
        for fname in (f"mvau{mvau}_rom_down_{name}.bin", f"mvau{mvau}_down_{name}.bin"):
            a = load_bin(f"{SCRIPT_DIR}/{fname}")
            if a is not None:
                write_bank(cfg, mvau, 128, a); n_writes += len(a); break
        # rom_up @ word 640
        for fname in (f"mvau{mvau}_rom_up_{name}.bin", f"mvau{mvau}_up_{name}.bin"):
            a = load_bin(f"{SCRIPT_DIR}/{fname}")
            if a is not None:
                write_bank(cfg, mvau, 640, a); n_writes += len(a); break
        # thresh @ word 1152
        a = load_bin(f"{SCRIPT_DIR}/mvau{mvau}_thresh_{name}.bin")
        if a is not None:
            write_bank(cfg, mvau, 1152, a); n_writes += len(a)
        # sign @ word 1408
        a = load_bin(f"{SCRIPT_DIR}/mvau{mvau}_sign_{name}.bin")
        if a is not None:
            write_bank(cfg, mvau, 1408, a); n_writes += len(a)
        # contrib_lut @ word 1664
        a = load_bin(f"{SCRIPT_DIR}/mvau{mvau}_lut_{name}.bin")
        if a is not None:
            write_bank(cfg, mvau, 1664, a); n_writes += len(a)

    # FC1 (unit 6), FC2 (unit 7)
    a = load_bin(f"{SCRIPT_DIR}/fc1_thresh_{name}.bin")
    if a is not None: write_bank(cfg, 6, 0, a); n_writes += len(a)
    a = load_bin(f"{SCRIPT_DIR}/fc2_thresh_{name}.bin")
    if a is not None: write_bank(cfg, 7, 0, a); n_writes += len(a)

    elapsed = (time.time() - t0) * 1000
    print(f"  [{name}] {n_writes} cfg writes, {elapsed:.2f} ms")


def run_one(idma, odma, x_byte, batch=1):
    ibuf = allocate(shape=(batch, 32, 32, 3), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch, 1, 1), dtype=np.uint8, cacheable=True)
    ibuf[0] = x_byte; ibuf.flush()
    odma.write(0x10, obuf.device_address); odma.write(0x1C, batch); odma.write(0x00, 1)
    idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch); idma.write(0x00, 1)
    t0 = time.time()
    while odma.read(0x00) & 0x2 == 0:
        if time.time() - t0 > 30: return None
    obuf.invalidate()
    out = obuf[:, 0, 0].copy()
    return out


def run_dataset_test(ol, name, max_samples=50):
    idma = ol.idma0; odma = ol.odma0
    xpath = f"{SCRIPT_DIR}/{name}_test_x.npy"
    ypath = f"{SCRIPT_DIR}/{name}_test_y.npy"
    if not os.path.exists(xpath): print(f"  [{name}] no test data, skip"); return
    x = np.load(xpath)[:max_samples]
    y = np.load(ypath)[:max_samples]
    correct = 0; total = 0
    t0 = time.time()
    for i in range(len(x)):
        out = run_one(idma, odma, x[i])
        if out is None: print(f"  [{name}] DMA TIMEOUT at sample {i}"); break
        pred = int(out[0])
        if pred == int(y[i]): correct += 1
        total += 1
    elapsed = time.time() - t0
    fps = total / elapsed if elapsed > 0 else 0
    acc = correct / total * 100 if total > 0 else 0
    print(f"  [{name}] acc={acc:.2f}% ({correct}/{total})  {fps:.1f} FPS  {elapsed:.2f}s")
    return acc


def main():
    bit = f"{SCRIPT_DIR}/resizer_v1.bit"  # uploaded as v1 bitstream
    print(f"Loading {bit}...")
    ol = Overlay(bit)
    Clocks.fclk0_mhz = 100.0
    print(f"fclk0 = {Clocks.fclk0_mhz} MHz")

    cfg = open_cfg_mmio()
    print("cfg_hub MMIO opened at 0x43C00000")

    # Test all 3 datasets
    for name, adapter_on in [("svhn", True), ("cifar10", False), ("fashion", True)]:
        print(f"\n=== {name.upper()} (adapter_on={adapter_on}) ===")
        write_dataset(cfg, name, adapter_on)
        run_dataset_test(ol, name)

if __name__ == "__main__":
    main()
