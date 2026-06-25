#!/usr/bin/env python3
"""Debug: test threshold range to find accumulation value distribution."""
import struct, time, os, numpy as np
from pynq import Overlay, MMIO, allocate
from glob import glob
from PIL import Image

ol = Overlay("resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)

CFG_ENABLE_WORD = 0
CFG_THRESH_BASE = 1152
MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}
# K = total XNOR ops per output channel
MVAU_K = {1: 64*9, 2: 64*9, 3: 128*9, 4: 128*9, 5: 256*9}

def mvau_byte_addr(mvau_id, word_addr):
    return (mvau_id << 13) | (word_addr << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))

# DMA setup
idma = ol.idma0
odma = ol.odma0
ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)

# Load a test image
cifar_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn/cifar10_finn_dataset"
# Find first available image
first_img_path = sorted(glob(os.path.join(cifar_dir, "0", "*.png")) + glob(os.path.join(cifar_dir, "0", "*.jpg")))[0]
print(f"Using test image: {first_img_path}")
test_img = np.array(Image.open(first_img_path).convert("RGB"))
ibuf[:] = test_img.reshape(1, 32, 32, 3, 1)

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

# Adapter OFF for all
for mvau_id in range(1, 6):
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)

# Sweep MVAU1 threshold to find accumulation range
print("=== Sweeping MVAU1 threshold (K=576, expected accum range [0, 576]) ===")
# Use correct thresholds for MVAU2-5 (CIFAR-10)
for mvau_id in range(2, 6):
    threshs = load_bin_u32(f"mvau{mvau_id}_thresh_cifar10.bin")
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)

# Try different uniform threshold values for MVAU1
K1 = 576
for thresh_val in [0, 100, 200, 288, 300, 336, 400, 500, 576]:
    thresh_q8 = thresh_val * 256
    for i in range(MVAU_OUT_CH[1]):
        mmio.write(mvau_byte_addr(1, CFG_THRESH_BASE + i), thresh_q8)
    pred = run_single()
    print(f"  MVAU1 uniform threshold={thresh_val} (Q8={thresh_q8}): pred={pred}")

# Now try correct per-channel thresholds for MVAU1
print("\n=== Per-channel CIFAR-10 thresholds ===")
threshs = load_bin_u32("mvau1_thresh_cifar10.bin")
for i, val in enumerate(threshs):
    mmio.write(mvau_byte_addr(1, CFG_THRESH_BASE + i), val)
pred = run_single()
print(f"  CIFAR-10 thresholds: pred={pred}")

# Try swapping step order for MVAU1
print("\n=== Swap step 0 and step 1 for MVAU1 ===")
PE1 = 32
STEPS1 = 2
for i in range(MVAU_OUT_CH[1]):
    pe = i % PE1
    step = i // PE1
    new_step = 1 - step  # swap
    new_ch = new_step * PE1 + pe
    mmio.write(mvau_byte_addr(1, CFG_THRESH_BASE + i), threshs[new_ch])
pred = run_single()
print(f"  Step-swapped thresholds: pred={pred}")

# Also check: what if step ordering for ALL MVAUs needs swapping?
print("\n=== Swap step order for ALL MVAUs ===")
MVAU_PE = {1: 32, 2: 16, 3: 16, 4: 4, 5: 1}
for mvau_id in range(1, 6):
    threshs = load_bin_u32(f"mvau{mvau_id}_thresh_cifar10.bin")
    pe = MVAU_PE[mvau_id]
    out_ch = MVAU_OUT_CH[mvau_id]
    steps = out_ch // pe
    for i in range(out_ch):
        p = i % pe
        step = i // pe
        new_step = (steps - 1) - step  # reverse step order
        new_ch = new_step * pe + p
        if new_ch < len(threshs):
            mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), threshs[new_ch])
pred = run_single()
print(f"  All MVAUs step-reversed: pred={pred}")

# Load CIFAR-10 images and test accuracy with step-swapped thresholds
print("\n=== Batch test with step-swapped (10 per class) ===")
images, labels = [], []
for cls in range(10):
    cls_dir = os.path.join(cifar_dir, str(cls))
    if os.path.isdir(cls_dir):
        files = sorted(glob(os.path.join(cls_dir, "*.png")))[:10]
        for f in files:
            img = np.array(Image.open(f).convert("RGB"))
            if img.shape == (32, 32, 3):
                images.append(img)
                labels.append(cls)

correct = 0
for i in range(len(images)):
    pred = run_single() if i == 0 else None
    ibuf[:] = images[i].reshape(1, 32, 32, 3, 1)
    pred = run_single()
    if pred == labels[i]:
        correct += 1
acc = 100 * correct / len(images) if images else 0
print(f"  Step-swapped accuracy: {correct}/{len(images)} = {acc:.1f}%")

# Now test with original ordering
print("\n=== Batch test with original ordering (10 per class) ===")
for mvau_id in range(1, 6):
    threshs = load_bin_u32(f"mvau{mvau_id}_thresh_cifar10.bin")
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
correct = 0
for i in range(len(images)):
    ibuf[:] = images[i].reshape(1, 32, 32, 3, 1)
    pred = run_single()
    if pred == labels[i]:
        correct += 1
acc = 100 * correct / len(images) if images else 0
print(f"  Original ordering accuracy: {correct}/{len(images)} = {acc:.1f}%")

ibuf.freebuffer()
obuf.freebuffer()
