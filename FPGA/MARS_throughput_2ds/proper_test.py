#!/usr/bin/env python3
"""
Proper accuracy test: correct dataset for each mode.
- CIFAR-10 mode: backbone-only thresholds, adapter OFF, CIFAR-10 images
- SVHN mode: fused thresholds, adapter ON, SVHN images
Also tests SVHN mode with SVHN golden image.
"""
import struct, os, numpy as np
from pynq import Overlay, MMIO, allocate
from glob import glob
from PIL import Image

ol = Overlay("/home/xilinx/runtime_switch/resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)
idma = ol.idma0
odma = ol.odma0
ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)

BD = "/home/xilinx/runtime_switch"
deploy = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn"

def mvau_byte_addr(m, w):
    return (m << 13) | (w << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack("<%dI" % (len(data)//4), data))

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

def switch_svhn_mode():
    for m in range(1, 6):
        threshs = load_bin_u32(BD + "/mvau%d_thresh_svhn.bin" % m)
        for i, val in enumerate(threshs):
            mmio.write(mvau_byte_addr(m, 1152 + i), val)
        signs = load_bin_u32(BD + "/mvau%d_sign_svhn.bin" % m)
        for i, val in enumerate(signs):
            mmio.write(mvau_byte_addr(m, 1408 + i), val)
        mmio.write(mvau_byte_addr(m, 0), 1)

def switch_cifar10_mode(thresh_variant="cifar10"):
    for m in range(1, 6):
        threshs = load_bin_u32(BD + "/mvau%d_thresh_%s.bin" % (m, thresh_variant))
        for i, val in enumerate(threshs):
            mmio.write(mvau_byte_addr(m, 1152 + i), val)
        mmio.write(mvau_byte_addr(m, 0), 0)

def switch_rom_defaults():
    ol2 = Overlay("/home/xilinx/runtime_switch/resizer.bit")
    global mmio, idma, odma
    mmio = MMIO(0x43C00000, 0x10000)
    idma = ol2.idma0
    odma = ol2.odma0
    for m in range(1, 6):
        mmio.write(mvau_byte_addr(m, 0), 0)

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
    return images, labels

def test_accuracy(images, labels, mode_name):
    correct = 0
    total = len(images)
    class_correct = [0]*10
    class_total = [0]*10
    preds_list = []
    for i in range(total):
        pred = run_single(images[i])
        preds_list.append(pred)
        class_total[labels[i]] += 1
        if pred == labels[i]:
            correct += 1
            class_correct[labels[i]] += 1
    acc = 100.0 * correct / total if total > 0 else 0
    print("  %s: %d/%d = %.1f%%" % (mode_name, correct, total, acc))
    # Per-class accuracy
    for c in range(10):
        if class_total[c] > 0:
            ca = 100.0 * class_correct[c] / class_total[c]
            print("    class %d: %d/%d = %.0f%%" % (c, class_correct[c], class_total[c], ca))
    # Prediction distribution
    from collections import Counter
    pred_counts = Counter(preds_list)
    print("    pred distribution: %s" % dict(sorted(pred_counts.items())))
    return acc

cifar_dir = deploy + "/cifar10_finn_dataset"
svhn_dir = deploy + "/svhn_finn_dataset"

print("=" * 60)

# Test 1: SVHN golden image with SVHN mode (should be class 3)
print("\n--- Test 1: SVHN golden image with SVHN mode ---")
switch_svhn_mode()
golden = np.load(deploy + "/input.npy")
pred = run_single(golden)
print("  SVHN golden: pred=%d (expected 3)" % pred)

# Test 2: SVHN golden with ROM defaults (adapter OFF)
print("\n--- Test 2: SVHN golden with ROM defaults, adapter OFF ---")
switch_rom_defaults()
pred = run_single(golden)
print("  ROM defaults: pred=%d" % pred)

# Test 3: SVHN golden with ROM defaults, adapter ON
print("\n--- Test 3: SVHN golden with ROM defaults, adapter ON ---")
for m in range(1, 6):
    mmio.write(mvau_byte_addr(m, 0), 1)
pred = run_single(golden)
print("  ROM defaults + adapter ON: pred=%d" % pred)

# Test 4: SVHN mode batch test
print("\n--- Test 4: SVHN mode, SVHN images (20/class) ---")
switch_svhn_mode()
if os.path.isdir(svhn_dir):
    s_imgs, s_labels = load_images(svhn_dir, max_per_class=20)
    print("  Loaded %d SVHN images" % len(s_imgs))
    test_accuracy(s_imgs, s_labels, "SVHN-mode")

# Test 5: CIFAR-10 mode (CIFAR-10 build thresholds) with CIFAR-10 images
print("\n--- Test 5: CIFAR-10 build thresh, adapter OFF, CIFAR-10 images ---")
switch_cifar10_mode("cifar10")
if os.path.isdir(cifar_dir):
    c_imgs, c_labels = load_images(cifar_dir, max_per_class=20)
    print("  Loaded %d CIFAR-10 images" % len(c_imgs))
    test_accuracy(c_imgs, c_labels, "CIFAR10-build-thresh")

# Test 6: CIFAR-10 mode (SVHN backbone thresholds) with CIFAR-10 images
print("\n--- Test 6: SVHN backbone thresh, adapter OFF, CIFAR-10 images ---")
switch_cifar10_mode("svhn_backbone")
if os.path.isdir(cifar_dir):
    test_accuracy(c_imgs, c_labels, "SVHN-backbone-thresh")

# Test 7: ROM defaults (adapter OFF) with CIFAR-10 images
print("\n--- Test 7: ROM defaults, adapter OFF, CIFAR-10 images ---")
switch_rom_defaults()
if os.path.isdir(cifar_dir):
    test_accuracy(c_imgs, c_labels, "ROM-defaults")

print("\n" + "=" * 60)
ibuf.freebuffer()
obuf.freebuffer()
print("Done.")
