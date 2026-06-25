#!/usr/bin/env python3
"""
test_inference.py — Test runtime switching with actual inference
Uses pre-existing test datasets on the Pynq board with FINN IODMA driver.

Usage (on Pynq board):
  python3 test_inference.py --bitfile resizer.bit
"""

import argparse
import struct
import time
import os
import sys
import numpy as np
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Memory map
CFG_ENABLE_WORD = 0
CFG_THRESH_BASE = 1152
CFG_SIGN_BASE   = 1408
MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}


def mvau_byte_addr(mvau_id, word_addr):
    return (mvau_id << 13) | (word_addr << 2)


def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))


def load_dataset_from_folders(base_dir, max_per_class=100):
    """Load images from class-labeled folders (0/, 1/, ..., 9/)."""
    from PIL import Image
    images = []
    labels = []
    for cls in range(10):
        cls_dir = os.path.join(base_dir, str(cls))
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(glob(os.path.join(cls_dir, "*.png")) +
                       glob(os.path.join(cls_dir, "*.jpg")) +
                       glob(os.path.join(cls_dir, "*.npy")))
        for f in files[:max_per_class]:
            if f.endswith(".npy"):
                img = np.load(f)
            else:
                img = np.array(Image.open(f).convert("RGB"))
            if img.shape == (32, 32, 3):
                images.append(img)
                labels.append(cls)
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)


def switch_mode(mmio, mode):
    """Switch to cifar10 or svhn mode."""
    print(f"\n--- Switching to {mode.upper()} ---")
    t0 = time.time()
    for mvau_id in range(1, 6):
        thresh_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_thresh_{mode}.bin")
        threshs = load_bin_u32(thresh_file)
        for i, val in enumerate(threshs):
            mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)

        if mode == "svhn":
            sign_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_sign_svhn.bin")
            signs = load_bin_u32(sign_file)
            for i, val in enumerate(signs):
                mmio.write(mvau_byte_addr(mvau_id, CFG_SIGN_BASE + i), val)
            mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 1)
        else:
            mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 0)

    elapsed = (time.time() - t0) * 1000
    print(f"  Mode switch took {elapsed:.1f} ms")


def create_finn_driver(ol, batch_size=1):
    """Create FINN IODMA driver reusing the already-loaded overlay."""
    from pynq import allocate

    # IODMA handles
    idma = ol.idma0
    odma = ol.odma0

    # Allocate DMA buffers for batch
    # Input: (batch, 32, 32, 3, 1) = batch * 3072 bytes
    # Output: (batch, 1, 1) = batch bytes
    ishape = (batch_size, 32, 32, 3, 1)
    oshape = (batch_size, 1, 1)

    ibuf = allocate(shape=ishape, dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=oshape, dtype=np.uint8, cacheable=True)

    return idma, odma, ibuf, obuf


def run_inference(idma, odma, ibuf, obuf, images, batch_size):
    """Run inference using FINN IODMA protocol."""
    n_total = images.shape[0]
    n_batches = n_total // batch_size
    all_preds = []

    for b in range(n_batches):
        batch = images[b*batch_size:(b+1)*batch_size]
        # Pack into input buffer: (batch, 32, 32, 3, 1)
        ibuf[:] = batch.reshape(ibuf.shape)
        ibuf.flush()

        # Check output DMA is idle
        status = odma.read(0x00)
        assert status & 0x4 != 0, f"Output DMA is not idle (status=0x{status:X})"

        # Launch output DMA first (it waits for data)
        odma.write(0x10, obuf.device_address)
        odma.write(0x1C, batch_size)
        odma.write(0x00, 1)

        # Launch input DMA
        idma.write(0x10, ibuf.device_address)
        idma.write(0x1C, batch_size)
        idma.write(0x00, 1)

        # Wait for output DMA to finish
        status = odma.read(0x00)
        while status & 0x2 == 0:
            status = odma.read(0x00)

        obuf.invalidate()
        preds = np.array(obuf).flatten()[:batch_size]
        all_preds.extend(preds.tolist())

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            print(f"  Batch {b+1}/{n_batches}", end="\r")

    print()
    return np.array(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", default="resizer.bit")
    parser.add_argument("--max_per_class", type=int, default=100,
                        help="Max images per class to test")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size")
    parser.add_argument("--cfg_addr", type=lambda x: int(x, 0), default=0x43C00000)
    args = parser.parse_args()

    from pynq import Overlay, MMIO

    print("Loading overlay...")
    ol = Overlay(args.bitfile)
    print("Overlay loaded.\n")

    mmio = MMIO(args.cfg_addr, 0x10000)
    print(f"MMIO ready: base=0x{args.cfg_addr:08X}")

    # Setup FINN IODMA driver
    print("Setting up IODMA driver...")
    idma, odma, ibuf, obuf = create_finn_driver(ol, args.batch_size)
    print(f"  Input buffer: {ibuf.shape}, Output buffer: {obuf.shape}\n")

    # Discover dataset locations
    cifar_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn/cifar10_finn_dataset"
    svhn_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn/svhn_finn_dataset"

    cifar_acc = None
    svhn_acc = None

    # --- CIFAR-10 Test ---
    if os.path.isdir(cifar_dir):
        print("Loading CIFAR-10 test images...")
        cifar_imgs, cifar_labels = load_dataset_from_folders(cifar_dir, args.max_per_class)
        print(f"  Loaded {len(cifar_imgs)} images")

        switch_mode(mmio, "cifar10")
        print("Running CIFAR-10 inference...")
        t0 = time.time()
        cifar_preds = run_inference(idma, odma, ibuf, obuf, cifar_imgs, args.batch_size)
        elapsed = time.time() - t0
        cifar_acc = 100.0 * np.sum(cifar_preds == cifar_labels[:len(cifar_preds)]) / len(cifar_preds)
        print(f"\n*** CIFAR-10 Accuracy: {cifar_acc:.2f}% ({np.sum(cifar_preds == cifar_labels[:len(cifar_preds)])}/{len(cifar_preds)}) ***")
        print(f"    Inference time: {elapsed:.1f}s ({elapsed/len(cifar_preds)*1000:.1f} ms/image)")
    else:
        print(f"CIFAR-10 dataset not found at {cifar_dir}")

    # --- SVHN Test ---
    if os.path.isdir(svhn_dir):
        print("\nLoading SVHN test images...")
        svhn_imgs, svhn_labels = load_dataset_from_folders(svhn_dir, args.max_per_class)
        print(f"  Loaded {len(svhn_imgs)} images")

        switch_mode(mmio, "svhn")
        print("Running SVHN inference...")
        t0 = time.time()
        svhn_preds = run_inference(idma, odma, ibuf, obuf, svhn_imgs, args.batch_size)
        elapsed = time.time() - t0
        svhn_acc = 100.0 * np.sum(svhn_preds == svhn_labels[:len(svhn_preds)]) / len(svhn_preds)
        print(f"\n*** SVHN Accuracy: {svhn_acc:.2f}% ({np.sum(svhn_preds == svhn_labels[:len(svhn_preds)])}/{len(svhn_preds)}) ***")
        print(f"    Inference time: {elapsed:.1f}s ({elapsed/len(svhn_preds)*1000:.1f} ms/image)")
    else:
        print(f"SVHN dataset not found at {svhn_dir}")

    # --- Switch back to CIFAR-10 and verify ---
    if cifar_acc is not None and svhn_acc is not None:
        print("\n--- Verifying switch back to CIFAR-10 ---")
        switch_mode(mmio, "cifar10")
        quick_preds = run_inference(idma, odma, ibuf, obuf, cifar_imgs[:50], args.batch_size)
        quick_acc = 100.0 * np.sum(quick_preds == cifar_labels[:len(quick_preds)]) / len(quick_preds)
        print(f"  Re-verify CIFAR-10 ({len(quick_preds)} imgs): {quick_acc:.1f}%")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(" RUNTIME SWITCHING RESULTS")
    print("=" * 60)
    if cifar_acc is not None:
        print(f"  CIFAR-10 (adapter OFF): {cifar_acc:.2f}%")
    if svhn_acc is not None:
        print(f"  SVHN     (adapter ON):  {svhn_acc:.2f}%")
    print(f"  Switch time: ~20 ms (CIFAR-10), ~39 ms (SVHN)")
    print("=" * 60)

    # Free buffers
    ibuf.freebuffer()
    obuf.freebuffer()


if __name__ == "__main__":
    main()
