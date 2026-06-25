#!/usr/bin/env python3
"""Debug: test with ROM defaults (no threshold writes) and compare modes."""
import struct, time, numpy as np
from pynq import Overlay, MMIO, allocate

ol = Overlay("resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)

CFG_ENABLE_WORD = 0
CFG_THRESH_BASE = 1152
CFG_SIGN_BASE   = 1408

def mvau_byte_addr(mvau_id, word_addr):
    return (mvau_id << 13) | (word_addr << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))

MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}

# Setup DMA
idma = ol.idma0
odma = ol.odma0
ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)

deploy_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn"
img = np.load(deploy_dir + "/input.npy")
ibuf[:] = img.reshape(1, 32, 32, 3, 1)

def run_single():
    ibuf.flush()
    obuf[:] = 0
    obuf.flush()
    odma.write(0x10, obuf.device_address)
    odma.write(0x1C, 1)
    odma.write(0x00, 1)
    idma.write(0x10, ibuf.device_address)
    idma.write(0x1C, 1)
    idma.write(0x00, 1)
    for _ in range(100000):
        if odma.read(0x00) & 0x2:
            break
    obuf.invalidate()
    return int(np.array(obuf).flatten()[0])

# Test 1: ROM defaults, adapter OFF
print("=== Test 1: ROM defaults, adapter OFF ===")
for mvau_id in range(1, 6):
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)
pred = run_single()
print(f"  Prediction: {pred} (expected: 3)")

# Test 2: ROM defaults, adapter ON (no sign/thresh writes)
print("\n=== Test 2: ROM defaults, adapter ON ===")
for mvau_id in range(1, 6):
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 1)
pred = run_single()
print(f"  Prediction: {pred}")

# Test 3: Load SVHN thresholds + signs, adapter ON
print("\n=== Test 3: SVHN mode (full load) ===")
for mvau_id in range(1, 6):
    threshs = load_bin_u32(f"mvau{mvau_id}_thresh_svhn.bin")
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
    signs = load_bin_u32(f"mvau{mvau_id}_sign_svhn.bin")
    for i, val in enumerate(signs):
        mmio.write(mvau_byte_addr(mvau_id, CFG_SIGN_BASE + i), val)
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 1)
pred = run_single()
print(f"  Prediction: {pred}")

# Test 4: Load CIFAR-10 thresholds, adapter OFF
print("\n=== Test 4: CIFAR-10 mode (new thresholds) ===")
for mvau_id in range(1, 6):
    threshs = load_bin_u32(f"mvau{mvau_id}_thresh_cifar10.bin")
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)
pred = run_single()
print(f"  Prediction: {pred} (expected: 3)")

# Read back threshold values from hardware for MVAU1
print("\n=== MVAU1 threshold readback (first 8 channels) ===")
for i in range(8):
    val = mmio.read(mvau_byte_addr(1, CFG_THRESH_BASE + i))
    print(f"  ch{i}: 0x{val:08X} = {val} (Q8={val})")

# Compare with CIFAR-10 bin file
print("\n=== MVAU1 CIFAR-10 bin values (first 8) ===")
c10 = load_bin_u32("mvau1_thresh_cifar10.bin")
for i in range(8):
    print(f"  ch{i}: 0x{c10[i]:08X} = {c10[i]} (orig_thresh={c10[i]//256})")

ibuf.freebuffer()
obuf.freebuffer()
