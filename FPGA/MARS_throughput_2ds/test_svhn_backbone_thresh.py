#!/usr/bin/env python3
"""
Test SVHN-pipeline backbone thresholds with adapter OFF.

Hypothesis: The CIFAR-10 FINN build used different BN parameters than the
SVHN pipeline compilation. Since the bitstream weights come from the SVHN
pipeline, the SVHN pipeline's raw backbone thresholds should be correct
for adapter-OFF inference.
"""
import struct, os, numpy as np
from pynq import Overlay, MMIO, allocate
from glob import glob
from PIL import Image

ol = Overlay("/home/xilinx/runtime_switch/resizer.bit")
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

def load_bin_i32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}i", data))

# DMA setup
idma = ol.idma0
odma = ol.odma0
ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)

def run_single(img):
    ibuf[:] = img.reshape(1, 32, 32, 3, 1)
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

def write_thresh(mvau_id, values):
    for i, val in enumerate(values):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val & 0xFFFFFFFF)

def write_signs(mvau_id, values):
    for i, val in enumerate(values):
        mmio.write(mvau_byte_addr(mvau_id, CFG_SIGN_BASE + i), val)

def set_enable(mvau_id, enable):
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 1 if enable else 0)

# Load golden test image
deploy_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn"
golden_img = np.load(deploy_dir + "/input.npy")

# Load threshold files
svhn_bb_thresh = {}
cifar10_thresh = {}
svhn_fused_thresh = {}
svhn_sign = {}
for m in range(1, 6):
    svhn_bb_thresh[m] = load_bin_u32(f"mvau{m}_thresh_svhn_backbone.bin")
    cifar10_thresh[m] = load_bin_u32(f"mvau{m}_thresh_cifar10.bin")
    svhn_fused_thresh[m] = load_bin_u32(f"mvau{m}_thresh_svhn.bin")
    svhn_sign[m] = load_bin_u32(f"mvau{m}_sign_svhn.bin")

print("=" * 70)
print("Testing threshold variants (golden image, expected class 3)")
print("=" * 70)

# Test 1: SVHN backbone thresholds, adapter OFF
print("\n--- Test 1: SVHN backbone thresh, adapter OFF ---")
for m in range(1, 6):
    write_thresh(m, svhn_bb_thresh[m])
    set_enable(m, False)
pred = run_single(golden_img)
print(f"  pred={pred} (SVHN backbone thresholds, adapter OFF)")

# Test 2: CIFAR-10 build thresholds, adapter OFF (known broken)
print("\n--- Test 2: CIFAR-10 build thresh, adapter OFF ---")
for m in range(1, 6):
    write_thresh(m, cifar10_thresh[m])
    set_enable(m, False)
pred = run_single(golden_img)
print(f"  pred={pred} (CIFAR-10 build thresholds, adapter OFF)")

# Test 3: SVHN fused thresholds, adapter ON (known working)
print("\n--- Test 3: SVHN fused thresh + signs, adapter ON ---")
for m in range(1, 6):
    write_thresh(m, svhn_fused_thresh[m])
    write_signs(m, svhn_sign[m])
    set_enable(m, True)
pred = run_single(golden_img)
print(f"  pred={pred} (SVHN fused mode, adapter ON)")

# Test 4: SVHN backbone thresh, adapter ON (should be wrong - no adapter contribution)
print("\n--- Test 4: SVHN backbone thresh, adapter ON (wrong signs from previous) ---")
for m in range(1, 6):
    write_thresh(m, svhn_bb_thresh[m])
    set_enable(m, True)
pred = run_single(golden_img)
print(f"  pred={pred} (SVHN backbone thresh + adapter ON)")

# Test 5: ROM defaults, adapter OFF
print("\n--- Test 5: ROM defaults, adapter OFF ---")
ol = Overlay("/home/xilinx/runtime_switch/resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)
idma = ol.idma0
odma = ol.odma0
for m in range(1, 6):
    set_enable(m, False)
pred = run_single(golden_img)
print(f"  pred={pred} (ROM defaults, adapter OFF)")

# Batch test with SVHN backbone thresholds
print("\n" + "=" * 70)
print("Batch accuracy test with SVHN backbone thresholds, adapter OFF")
print("=" * 70)

cifar_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn/cifar10_finn_dataset"

# Write SVHN backbone thresholds
for m in range(1, 6):
    write_thresh(m, svhn_bb_thresh[m])
    set_enable(m, False)

if os.path.isdir(cifar_dir):
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
        pred = run_single(images[i])
        if pred == labels[i]:
            correct += 1
    acc = 100 * correct / len(images)
    print(f"  SVHN backbone thresh: {correct}/{len(images)} = {acc:.1f}%")

    # Compare with CIFAR-10 build thresholds
    for m in range(1, 6):
        write_thresh(m, cifar10_thresh[m])
        set_enable(m, False)
    correct2 = 0
    for i in range(len(images)):
        pred = run_single(images[i])
        if pred == labels[i]:
            correct2 += 1
    acc2 = 100 * correct2 / len(images)
    print(f"  CIFAR-10 build thresh: {correct2}/{len(images)} = {acc2:.1f}%")

ibuf.freebuffer()
obuf.freebuffer()
print("\nDone.")
