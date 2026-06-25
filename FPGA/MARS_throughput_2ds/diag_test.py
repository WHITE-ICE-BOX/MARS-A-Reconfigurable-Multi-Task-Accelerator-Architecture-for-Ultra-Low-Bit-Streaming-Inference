#!/usr/bin/env python3
"""
Diagnostic test: incremental parameter switching to isolate CIFAR-10 accuracy issue.
"""
import numpy as np
import struct
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_quick_test(ol, test_x, test_y, n_samples=1000, batch_size=100):
    """Quick accuracy test using raw DMA."""
    from pynq import allocate
    idma = getattr(ol, 'idma0')
    odma = getattr(ol, 'odma0')

    n_batches = min(n_samples, test_x.shape[0]) // batch_size
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)

    correct = 0
    tested = 0
    for b in range(n_batches):
        s = b * batch_size
        np.copyto(ibuf, test_x[s:s+batch_size].reshape(batch_size, 32, 32, 3, 1).astype(np.uint8))
        ibuf.flush()
        odma.write(0x10, obuf.device_address)
        odma.write(0x1C, batch_size)
        odma.write(0x00, 1)
        idma.write(0x10, ibuf.device_address)
        idma.write(0x1C, batch_size)
        idma.write(0x00, 1)
        status = odma.read(0x00)
        while status & 0x2 == 0:
            status = odma.read(0x00)
        obuf.invalidate()
        preds = np.array(obuf).flatten().astype(np.int64)
        correct += np.sum(preds == test_y[s:s+batch_size])
        tested += batch_size

    ibuf.freebuffer()
    obuf.freebuffer()
    return 100.0 * correct / tested


def flush(ol, batch_size=1):
    from pynq import allocate
    idma = getattr(ol, 'idma0')
    odma = getattr(ol, 'odma0')
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    ibuf[:] = 0; ibuf.flush()
    odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
    idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
    status = odma.read(0x00)
    while status & 0x2 == 0: status = odma.read(0x00)
    ibuf.freebuffer(); obuf.freebuffer()


def main():
    from pynq import Overlay, MMIO
    from pynq.ps import Clocks

    print("=" * 60)
    print("Diagnostic: Incremental Parameter Switching")
    print("=" * 60)

    ol = Overlay(os.path.join(SCRIPT_DIR, "resizer.bit"))
    Clocks.fclk0_mhz = 100.0
    mmio = MMIO(0x43C00000, 0x10000)

    def load_bin_u32(path):
        with open(path, "rb") as f:
            data = f.read()
        return list(struct.unpack(f"<{len(data)//4}I", data))

    def write_words(byte_offset, values):
        for i, val in enumerate(values):
            mmio.write(byte_offset + i * 4, val & 0xFFFFFFFF)

    # Load test data
    c10_x = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_x.npy"))
    c10_y = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_y.npy"))
    svhn_x = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    svhn_y = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))

    N = 2000  # quick test samples

    # Test 0: Baseline - baked-in SVHN defaults, no writes
    print("\n[0] Baseline (baked-in SVHN defaults, no writes)")
    acc = run_quick_test(ol, svhn_x, svhn_y, N)
    print(f"    SVHN accuracy: {acc:.1f}%")
    acc = run_quick_test(ol, c10_x, c10_y, N)
    print(f"    CIFAR-10 accuracy: {acc:.1f}%")

    # Test 1: Write SVHN params explicitly (should match baked-in)
    print("\n[1] Write SVHN params explicitly")
    write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))
    for i in range(1, 6):
        base = i << 13
        write_words(base + 1152 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_svhn.bin")))
        write_words(base + 1408 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_svhn.bin")))
        mmio.write(base, 1)
    write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
    write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))
    write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y, N)
    print(f"    SVHN accuracy: {acc:.1f}%")

    # Test 2: Switch ONLY MVAU0 to CIFAR-10, keep rest SVHN
    print("\n[2] Only MVAU0 → CIFAR-10 (rest SVHN)")
    write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_cifar10.bin")))
    flush(ol)
    acc = run_quick_test(ol, c10_x, c10_y, N)
    print(f"    CIFAR-10 accuracy: {acc:.1f}%")
    # Restore
    write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))

    # Test 3: Switch ONLY adapter OFF for MVAU1-5, keep SVHN thresholds
    print("\n[3] Only adapter OFF (MVAU1-5), SVHN thresholds")
    for i in range(1, 6):
        base = i << 13
        mmio.write(base, 0)  # adapter OFF
    flush(ol)
    acc = run_quick_test(ol, c10_x, c10_y, N)
    print(f"    CIFAR-10 accuracy: {acc:.1f}%")
    acc = run_quick_test(ol, svhn_x, svhn_y, N)
    print(f"    SVHN accuracy: {acc:.1f}%")
    # Restore
    for i in range(1, 6):
        base = i << 13
        mmio.write(base, 1)

    # Test 4: Switch ONLY FC1/FC2 to CIFAR-10
    print("\n[4] Only FC1/FC2 → CIFAR-10 (rest SVHN)")
    write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_cifar10.bin")))
    write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_cifar10.bin")))
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y, N)
    print(f"    SVHN accuracy: {acc:.1f}%")
    # Restore
    write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
    write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))

    # Test 5: Switch ONLY classifier to CIFAR-10
    print("\n[5] Only classifier → CIFAR-10 (rest SVHN)")
    write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_cifar10.bin")))
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y, N)
    print(f"    SVHN accuracy: {acc:.1f}%")
    # Restore
    write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y, N)
    print(f"    SVHN after restore: {acc:.1f}%")

    # Test 6: Full CIFAR-10 switch
    print("\n[6] Full CIFAR-10 switch")
    write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_cifar10.bin")))
    for i in range(1, 6):
        base = i << 13
        write_words(base + 1152 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_cifar10.bin")))
        mmio.write(base, 0)
    write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_cifar10.bin")))
    write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_cifar10.bin")))
    write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_cifar10.bin")))
    flush(ol)
    acc = run_quick_test(ol, c10_x, c10_y, N)
    print(f"    CIFAR-10 accuracy: {acc:.1f}%")

    # Test 7: Switch back to SVHN
    print("\n[7] Switch back to SVHN")
    write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))
    for i in range(1, 6):
        base = i << 13
        write_words(base + 1152 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_svhn.bin")))
        write_words(base + 1408 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_svhn.bin")))
        mmio.write(base, 1)
    write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
    write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))
    write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y, N)
    print(f"    SVHN accuracy: {acc:.1f}%")

    print("\n" + "=" * 60)
    print("Diagnostic complete!")


if __name__ == "__main__":
    main()
