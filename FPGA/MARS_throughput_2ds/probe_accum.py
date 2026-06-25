#!/usr/bin/env python3
"""
Probe actual MVAU accumulation values by sweeping uniform thresholds.
Binary search to find the transition point for each channel.
"""
import struct, os, numpy as np
from pynq import Overlay, MMIO, allocate
from glob import glob
from PIL import Image

ol = Overlay("/home/xilinx/runtime_switch/resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)
idma = ol.idma0
odma = ol.odma0

MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}
MVAU_PE = {1: 32, 2: 16, 3: 16, 4: 4, 5: 1}
# Max popcount per output channel (input_channels * kernel_size)
# CNV architecture: 64->64->128->128->256->256 channels, 3x3 kernels
# Layer 1: 64 input channels, 3x3 kernel = 576 XNOR ops
# Layer 2: 64 input channels, 3x3 kernel = 576
# Layer 3: 128 input, 3x3 = 1152
# Layer 4: 128 input, 3x3 = 1152
# Layer 5: 256 input, 3x3 = 2304
MVAU_K = {1: 576, 2: 576, 3: 1152, 4: 1152, 5: 2304}

def mvau_byte_addr(m, w):
    return (m << 13) | (w << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack("<%dI" % (len(data)//4), data))

BD = "/home/xilinx/runtime_switch"

# Use a CIFAR-10 image
cifar_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn/cifar10_finn_dataset"
# Load first image from class 0
img_path = sorted(glob(os.path.join(cifar_dir, "0", "*.png")))[0]
test_img = np.array(Image.open(img_path).convert("RGB"))
print("Test image: %s, shape=%s" % (img_path, test_img.shape))

ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)
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

# All adapter OFF
for m in range(1, 6):
    mmio.write(mvau_byte_addr(m, 0), 0)

# Set MVAU2-5 to correct CIFAR-10 thresholds
for m in range(2, 6):
    threshs = load_bin_u32(BD + "/mvau%d_thresh_cifar10.bin" % m)
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(m, 1152 + i), val)

# Sweep MVAU1 with uniform threshold to see how prediction changes
print("\n=== MVAU1 uniform threshold sweep (K=%d) ===" % MVAU_K[1])
print("When adapter OFF: comparison is mvau_pop*256 >= thresh_q8")
print("So thresh_q8 = thresh * 256")
for thresh in range(0, MVAU_K[1]+1, 50):
    thresh_q8 = thresh * 256
    for i in range(MVAU_OUT_CH[1]):
        mmio.write(mvau_byte_addr(1, 1152 + i), thresh_q8 & 0xFFFFFFFF)
    pred = run_single()
    # Count how many channels would fire for this threshold
    print("  thresh=%3d (q8=%6d): pred=%d" % (thresh, thresh_q8, pred))

# Now check: what if the accumulation values are NOT popcount but bipolar?
# Bipolar: mac = 2*popcount - K, range [-K, K]
# If bipolar, threshold in FINN would be in [-K, K] range
# Let's check CIFAR-10 FINN threshold range
print("\n=== CIFAR-10 FINN threshold ranges ===")
for m in range(1, 6):
    threshs = load_bin_u32(BD + "/mvau%d_thresh_cifar10.bin" % m)
    # Interpret as signed
    signed_vals = [struct.unpack("<i", struct.pack("<I", v))[0] for v in threshs]
    raw_vals = [v // 256 for v in signed_vals]
    K = MVAU_K[m]
    print("  MVAU%d: K=%d, thresh range=[%d, %d] (raw), mean=%.1f" % (
        m, K, min(raw_vals), max(raw_vals), sum(raw_vals)/len(raw_vals)))
    print("    ratio to K: [%.2f, %.2f], mean=%.2f" % (
        min(raw_vals)/K, max(raw_vals)/K, sum(raw_vals)/(len(raw_vals)*K)))

# Check if thresholds might need sign inversion or complement
# If FINN uses <=, output=1 when accum <= threshold
# This means output=1 for SMALL values, 0 for LARGE values
# Our HW uses >=, output=1 when pop*256 >= thresh*256
# This means output=1 for LARGE values, 0 for SMALL values
# If the polarity is INVERTED, we need to fix it!
print("\n=== Testing INVERTED threshold hypothesis ===")
print("If FINN outputs 1 for accum <= threshold (low values),")
print("then we need: output = (mvau_pop <= threshold) ? 1 : 0")
print("Which is equivalent to: output = (K - mvau_pop >= K - threshold) ? 1 : 0")
print("So inverted_threshold = K - finn_threshold")

for m in range(1, 6):
    threshs = load_bin_u32(BD + "/mvau%d_thresh_cifar10.bin" % m)
    signed_vals = [struct.unpack("<i", struct.pack("<I", v))[0] for v in threshs]
    K = MVAU_K[m]
    inverted = []
    for sv in signed_vals:
        raw = sv // 256
        inv_raw = K - raw
        inv_q8 = inv_raw * 256
        inverted.append(inv_q8 & 0xFFFFFFFF)
    for i, val in enumerate(inverted):
        mmio.write(mvau_byte_addr(m, 1152 + i), val)

# Load multiple CIFAR-10 images and test
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
    ibuf[:] = images[i].reshape(1, 32, 32, 3, 1)
    pred = run_single()
    if pred == labels[i]:
        correct += 1
acc = 100.0 * correct / len(images)
print("  Inverted thresholds: %d/%d = %.1f%%" % (correct, len(images), acc))

# Also test with regular thresholds for comparison
for m in range(1, 6):
    threshs = load_bin_u32(BD + "/mvau%d_thresh_cifar10.bin" % m)
    for i, val in enumerate(threshs):
        mmio.write(mvau_byte_addr(m, 1152 + i), val)
correct2 = 0
for i in range(len(images)):
    ibuf[:] = images[i].reshape(1, 32, 32, 3, 1)
    pred = run_single()
    if pred == labels[i]:
        correct2 += 1
acc2 = 100.0 * correct2 / len(images)
print("  Regular thresholds:  %d/%d = %.1f%%" % (correct2, len(images), acc2))

ibuf.freebuffer()
obuf.freebuffer()
print("\nDone.")
