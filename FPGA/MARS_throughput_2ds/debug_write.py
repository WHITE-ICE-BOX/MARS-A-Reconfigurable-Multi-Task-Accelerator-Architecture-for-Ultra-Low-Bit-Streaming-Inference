#!/usr/bin/env python3
"""Debug: verify threshold writes actually take effect."""
import struct, time, numpy as np
from pynq import Overlay, MMIO, allocate

ol = Overlay("resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)

CFG_ENABLE_WORD = 0
CFG_THRESH_BASE = 1152
MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}

def mvau_byte_addr(mvau_id, word_addr):
    return (mvau_id << 13) | (word_addr << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))

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

# All adapters OFF
for mvau_id in range(1, 6):
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)

# Test A: ROM defaults (no thresh write), adapter OFF
print("=== A: ROM defaults, adapter OFF ===")
pred = run_single()
print(f"  Prediction: {pred}")

# Test B: Write threshold=0 for ALL MVAUs (everything should be >= 0 → all 1s)
print("\n=== B: All thresholds = 0, adapter OFF ===")
for mvau_id in range(1, 6):
    for i in range(MVAU_OUT_CH[mvau_id]):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), 0)
pred = run_single()
print(f"  Prediction: {pred}  (all-1s output → specific class)")

# Test C: Write threshold=MAX for ALL MVAUs (nothing should be >= MAX → all 0s)
print("\n=== C: All thresholds = 0x7FFFFFFF, adapter OFF ===")
for mvau_id in range(1, 6):
    for i in range(MVAU_OUT_CH[mvau_id]):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), 0x7FFFFFFF)
pred = run_single()
print(f"  Prediction: {pred}  (all-0s output → specific class)")

# Test D: Restore ROM defaults by reloading overlay, adapter OFF
print("\n=== D: Reload overlay, ROM defaults, adapter OFF ===")
ol = Overlay("resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)
idma = ol.idma0
odma = ol.odma0
for mvau_id in range(1, 6):
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)
pred = run_single()
print(f"  Prediction: {pred}  (should match test A)")

# Test E: Write CIFAR-10 thresholds, adapter OFF
print("\n=== E: CIFAR-10 thresholds, adapter OFF ===")
for mvau_id in range(1, 6):
    threshs = load_bin_u32(f"mvau{mvau_id}_thresh_cifar10.bin")
    print(f"  MVAU{mvau_id}: {len(threshs)} entries, first={threshs[0]}, last={threshs[-1]}")
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)
pred = run_single()
print(f"  Prediction: {pred}")

# Test F: SVHN thresholds + signs, adapter ON
print("\n=== F: SVHN mode ===")
CFG_SIGN_BASE = 1408
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

ibuf.freebuffer()
obuf.freebuffer()
