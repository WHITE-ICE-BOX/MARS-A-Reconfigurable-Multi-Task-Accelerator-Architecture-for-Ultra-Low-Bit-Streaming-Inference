#!/usr/bin/env python3
# ===========================================================================
# [交接導向註解]
# 板上精度測試。流程：FPGA(throughput build)。
# ===========================================================================

"""
On-board accuracy verification for runtime dataset switching.
Self-contained: no qonnx/finn dependencies, uses raw MMIO DMA control.
"""
import numpy as np
import struct
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_accuracy_test(ol, mmio_cfg, test_x, test_y, batch_size=100, max_samples=None):
    """Run inference and compute accuracy using raw DMA register control."""
    from pynq import allocate

    # Get DMA IP handles
    idma = getattr(ol, 'idma0')
    odma = getattr(ol, 'odma0')

    total = test_x.shape[0]
    if max_samples:
        total = min(total, max_samples)
    n_batches = total // batch_size
    total = n_batches * batch_size

    correct = 0
    tested = 0

    # Allocate DMA buffers for the batch
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)

    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size

        batch_x = test_x[start:end].astype(np.uint8)
        batch_y = test_y[start:end]

        # Copy input data
        np.copyto(ibuf, batch_x.reshape(batch_size, 32, 32, 3, 1))
        ibuf.flush()

        # Launch DMA transfers via register writes (FINN IODMA style)
        # Output DMA first, then input DMA
        odma.write(0x10, obuf.device_address)
        odma.write(0x1C, batch_size)
        odma.write(0x00, 1)

        idma.write(0x10, ibuf.device_address)
        idma.write(0x1C, batch_size)
        idma.write(0x00, 1)

        # Wait for output DMA to finish
        status = odma.read(0x00)
        while status & 0x2 == 0:
            status = odma.read(0x00)

        # Read results
        obuf.invalidate()
        preds = np.array(obuf).flatten().astype(np.int64)

        correct += np.sum(preds == batch_y)
        tested += batch_size

        if tested % 2000 == 0:
            acc = 100.0 * correct / tested
            print(f"  {tested}/{total}: acc={acc:.1f}%")

    ibuf.freebuffer()
    obuf.freebuffer()

    acc = 100.0 * correct / tested
    return acc, tested


def flush_pipeline(ol, batch_size=1):
    """Run one dummy inference to flush stale weight data from FIFOs."""
    from pynq import allocate
    idma = getattr(ol, 'idma0')
    odma = getattr(ol, 'odma0')
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    ibuf[:] = 0
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
    ibuf.freebuffer()
    obuf.freebuffer()


def main():
    from pynq import Overlay, MMIO
    from pynq.ps import Clocks

    print("=" * 60)
    print("Runtime Dataset Switching — On-Board Verification")
    print("=" * 60)

    # Load bitstream
    print("\nLoading bitstream...")
    ol = Overlay(os.path.join(SCRIPT_DIR, "resizer.bit"))
    Clocks.fclk0_mhz = 100.0
    print("Bitstream loaded.")

    # Set up cfg_hub MMIO
    cfg_addr = 0x43C00000
    cfg_range = 0x10000
    mmio = MMIO(cfg_addr, cfg_range)
    print(f"MMIO ready: 0x{cfg_addr:08X}")

    def load_bin_u32(path):
        with open(path, "rb") as f:
            data = f.read()
        return list(struct.unpack(f"<{len(data)//4}I", data))

    def write_words(byte_offset, values):
        for i, val in enumerate(values):
            mmio.write(byte_offset + i * 4, val & 0xFFFFFFFF)

    batch_size = 100

    # ============================================================
    # Test 1: SVHN mode
    # ============================================================
    print("\n" + "=" * 40)
    print("TEST 1: SVHN mode (adapter ON)")
    print("=" * 40)

    t0 = time.time()
    write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))
    for i in range(1, 6):
        base = i << 13
        write_words(base + 1152 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_svhn.bin")))
        write_words(base + 1408 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_svhn.bin")))
        mmio.write(base, 1)
    write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
    write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))
    write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))
    switch_time = (time.time() - t0) * 1000
    print(f"Switch to SVHN: {switch_time:.1f} ms")

    print("Flushing pipeline...")
    flush_pipeline(ol)

    svhn_x = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    svhn_y = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))
    print(f"SVHN test set: {svhn_x.shape[0]} images")

    acc, n = run_accuracy_test(ol, mmio, svhn_x, svhn_y, batch_size=batch_size, max_samples=10000)
    print(f"\n*** SVHN Accuracy: {acc:.2f}% ({int(acc*n/100)}/{n}) ***")
    print(f"    Target: 71.8%")

    # ============================================================
    # Test 2: CIFAR-10 mode
    # ============================================================
    print("\n" + "=" * 40)
    print("TEST 2: CIFAR-10 mode (adapter OFF)")
    print("=" * 40)

    t0 = time.time()
    write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_cifar10.bin")))
    for i in range(1, 6):
        base = i << 13
        write_words(base + 1152 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_cifar10.bin")))
        mmio.write(base, 0)
    write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_cifar10.bin")))
    write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_cifar10.bin")))
    write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_cifar10.bin")))
    switch_time = (time.time() - t0) * 1000
    print(f"Switch to CIFAR-10: {switch_time:.1f} ms")

    print("Flushing pipeline...")
    flush_pipeline(ol)

    c10_x = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_x.npy"))
    c10_y = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_y.npy"))
    print(f"CIFAR-10 test set: {c10_x.shape[0]} images")

    acc, n = run_accuracy_test(ol, mmio, c10_x, c10_y, batch_size=batch_size, max_samples=10000)
    print(f"\n*** CIFAR-10 Accuracy: {acc:.2f}% ({int(acc*n/100)}/{n}) ***")
    print(f"    Target: ~74%")

    # ============================================================
    # Test 3: Rapid switching
    # ============================================================
    print("\n" + "=" * 40)
    print("TEST 3: Rapid switching benchmark")
    print("=" * 40)

    times_c10 = []
    times_svhn = []
    for _ in range(10):
        t0 = time.time()
        write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_cifar10.bin")))
        for i in range(1, 6):
            base = i << 13
            write_words(base + 1152 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_cifar10.bin")))
            mmio.write(base, 0)
        write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_cifar10.bin")))
        write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_cifar10.bin")))
        write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_cifar10.bin")))
        times_c10.append((time.time() - t0) * 1000)

        t0 = time.time()
        write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))
        for i in range(1, 6):
            base = i << 13
            write_words(base + 1152 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_svhn.bin")))
            write_words(base + 1408 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_svhn.bin")))
            mmio.write(base, 1)
        write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
        write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))
        write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))
        times_svhn.append((time.time() - t0) * 1000)

    print(f"  →CIFAR-10: avg={np.mean(times_c10):.1f}ms, min={np.min(times_c10):.1f}ms, max={np.max(times_c10):.1f}ms")
    print(f"  →SVHN:     avg={np.mean(times_svhn):.1f}ms, min={np.min(times_svhn):.1f}ms, max={np.max(times_svhn):.1f}ms")

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
