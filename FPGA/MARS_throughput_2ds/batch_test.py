#!/usr/bin/env python3
# ===========================================================================
# [交接導向註解]
# batch-1000 吞吐量量測（-> 1866 img/s）。流程：FPGA(throughput build)。
# ===========================================================================

"""Batch accuracy test for CIFAR-10 and SVHN modes."""
import struct, time, os, numpy as np
from pynq import Overlay, MMIO, allocate
from glob import glob
from PIL import Image

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

# Setup DMA
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

def switch_cifar10():
    for mvau_id in range(1, 6):
        threshs = load_bin_u32(f"mvau{mvau_id}_thresh_cifar10.bin")
        for i, val in enumerate(threshs):
            mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
        mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)

def switch_svhn():
    for mvau_id in range(1, 6):
        threshs = load_bin_u32(f"mvau{mvau_id}_thresh_svhn.bin")
        for i, val in enumerate(threshs):
            mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)
        signs = load_bin_u32(f"mvau{mvau_id}_sign_svhn.bin")
        for i, val in enumerate(signs):
            mmio.write(mvau_byte_addr(mvau_id, CFG_SIGN_BASE + i), val)
        mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 1)

def switch_rom_defaults():
    """Use ROM defaults (reload overlay) and adapter OFF."""
    for mvau_id in range(1, 6):
        mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)

def load_images(base_dir, max_per_class=20):
    images, labels = [], []
    for cls in range(10):
        cls_dir = os.path.join(base_dir, str(cls))
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(glob(os.path.join(cls_dir, "*.png")) + glob(os.path.join(cls_dir, "*.jpg")))
        for f in files[:max_per_class]:
            img = np.array(Image.open(f).convert("RGB"))
            if img.shape == (32, 32, 3):
                images.append(img)
                labels.append(cls)
    return np.array(images, dtype=np.uint8), np.array(labels)

def test_accuracy(images, labels, mode_name):
    correct = 0
    total = len(images)
    preds = []
    for i in range(total):
        pred = run_single(images[i])
        preds.append(pred)
        if pred == labels[i]:
            correct += 1
        if (i+1) % 50 == 0:
            print(f"  {mode_name}: {i+1}/{total}, acc={100*correct/(i+1):.1f}%", flush=True)
    acc = 100 * correct / total
    print(f"  {mode_name} FINAL: {correct}/{total} = {acc:.2f}%")
    return acc, preds

# Load datasets
cifar_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn/cifar10_finn_dataset"
svhn_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn/svhn_finn_dataset"

print("=" * 60)

# Test CIFAR-10 with ROM defaults (no writes, adapter OFF)
print("\n--- Test 1: ROM defaults, adapter OFF, CIFAR-10 images ---")
switch_rom_defaults()
if os.path.isdir(cifar_dir):
    c_imgs, c_labels = load_images(cifar_dir, max_per_class=10)
    print(f"Loaded {len(c_imgs)} CIFAR-10 images")
    test_accuracy(c_imgs, c_labels, "ROM-defaults-CIFAR10")

# Test CIFAR-10 with our CIFAR-10 thresholds
print("\n--- Test 2: CIFAR-10 thresholds, adapter OFF, CIFAR-10 images ---")
switch_cifar10()
if os.path.isdir(cifar_dir):
    test_accuracy(c_imgs, c_labels, "CIFAR10-thresh-CIFAR10")

# Test SVHN mode
print("\n--- Test 3: SVHN mode, SVHN images ---")
switch_svhn()
if os.path.isdir(svhn_dir):
    s_imgs, s_labels = load_images(svhn_dir, max_per_class=10)
    print(f"Loaded {len(s_imgs)} SVHN images")
    test_accuracy(s_imgs, s_labels, "SVHN-mode-SVHN")

# Test SVHN mode on CIFAR-10 images (sanity: should be low)
print("\n--- Test 4: SVHN mode, CIFAR-10 images ---")
if os.path.isdir(cifar_dir):
    test_accuracy(c_imgs, c_labels, "SVHN-mode-CIFAR10")

print("\n" + "=" * 60)
ibuf.freebuffer()
obuf.freebuffer()
