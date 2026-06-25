#!/usr/bin/env python3
"""Quick sanity test: does SVHN mode still work at all?"""
import struct, numpy as np
from pynq import Overlay, MMIO, allocate

ol = Overlay("/home/xilinx/runtime_switch/resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)
idma = ol.idma0
odma = ol.odma0
ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)

deploy_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn"
img = np.load(deploy_dir + "/input.npy")
ibuf[:] = img.reshape(1, 32, 32, 3, 1)
print("Golden image: shape=%s dtype=%s first10=%s" % (img.shape, img.dtype, img.flatten()[:10]))

def mvau_byte_addr(m, w):
    return (m << 13) | (w << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack("<%dI" % (len(data)//4), data))

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
    for _ in range(200000):
        if odma.read(0x00) & 0x2:
            break
    obuf.invalidate()
    return int(np.array(obuf).flatten()[0])

BD = "/home/xilinx/runtime_switch"

# Test A: ROM defaults, adapter OFF
print("=== A: Fresh overlay, ROM defaults, adapter OFF ===")
for m in range(1, 6):
    mmio.write(mvau_byte_addr(m, 0), 0)
pred = run_single()
print("  pred=%d" % pred)

# Test B: ROM defaults, adapter ON (ROM has SVHN fused thresholds)
print("=== B: ROM defaults, adapter ON ===")
for m in range(1, 6):
    mmio.write(mvau_byte_addr(m, 0), 1)
pred = run_single()
print("  pred=%d" % pred)

# Test C: Write SVHN thresholds + signs, adapter ON
print("=== C: SVHN fused thresh+signs, adapter ON ===")
for m in range(1, 6):
    threshs = load_bin_u32(BD + "/mvau%d_thresh_svhn.bin" % m)
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(m, 1152 + i), val)
    signs = load_bin_u32(BD + "/mvau%d_sign_svhn.bin" % m)
    for i, val in enumerate(signs):
        mmio.write(mvau_byte_addr(m, 1408 + i), val)
    mmio.write(mvau_byte_addr(m, 0), 1)
pred = run_single()
print("  pred=%d" % pred)

# Test D: CIFAR-10 thresholds, adapter OFF
print("=== D: CIFAR-10 build thresh, adapter OFF ===")
for m in range(1, 6):
    threshs = load_bin_u32(BD + "/mvau%d_thresh_cifar10.bin" % m)
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(m, 1152 + i), val)
    mmio.write(mvau_byte_addr(m, 0), 0)
pred = run_single()
print("  pred=%d" % pred)

# Test E: CIFAR-10 thresholds, adapter ON
print("=== E: CIFAR-10 build thresh, adapter ON ===")
for m in range(1, 6):
    mmio.write(mvau_byte_addr(m, 0), 1)
pred = run_single()
print("  pred=%d" % pred)

# Test F: SVHN backbone thresh, adapter OFF
print("=== F: SVHN backbone thresh, adapter OFF ===")
for m in range(1, 6):
    threshs = load_bin_u32(BD + "/mvau%d_thresh_svhn_backbone.bin" % m)
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(m, 1152 + i), val)
    mmio.write(mvau_byte_addr(m, 0), 0)
pred = run_single()
print("  pred=%d" % pred)

# Test G: Run 3 times for consistency
print("=== G: Consistency (3 runs, current config) ===")
preds = [run_single() for _ in range(3)]
print("  preds=%s" % preds)

ibuf.freebuffer()
obuf.freebuffer()
print("Done.")
