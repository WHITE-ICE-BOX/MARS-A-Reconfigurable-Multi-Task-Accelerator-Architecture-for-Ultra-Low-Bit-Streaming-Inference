#!/usr/bin/env python3
"""Golden test: single image, expect class 3, using CIFAR-10 mode."""
import struct, time, numpy as np
from pynq import Overlay, MMIO, allocate

ol = Overlay("resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)

CFG_ENABLE_WORD = 0
CFG_THRESH_BASE = 1152

def mvau_byte_addr(mvau_id, word_addr):
    return (mvau_id << 13) | (word_addr << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))

# Switch to CIFAR-10 mode
print("Switching to CIFAR-10 mode...")
t0 = time.time()
for mvau_id in range(1, 6):
    threshs = load_bin_u32(f"mvau{mvau_id}_thresh_cifar10.bin")
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)
elapsed = (time.time() - t0) * 1000
print(f"  Switch took {elapsed:.1f} ms")

# Setup DMA
idma = ol.idma0
odma = ol.odma0
ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)

# Load golden test image
deploy_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn"
img = np.load(deploy_dir + "/input.npy")
print(f"Input shape: {img.shape}, dtype: {img.dtype}")
ibuf[:] = img.reshape(1, 32, 32, 3, 1)
ibuf.flush()

# Run inference
obuf[:] = 0
obuf.flush()
status = odma.read(0x00)
print(f"ODMA status before: 0x{status:X}")

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
pred = int(np.array(obuf).flatten()[0])
print(f"Prediction: {pred} (expected: 3)")
print("GOLDEN TEST " + ("PASSED" if pred == 3 else "FAILED"))

ibuf.freebuffer()
obuf.freebuffer()
