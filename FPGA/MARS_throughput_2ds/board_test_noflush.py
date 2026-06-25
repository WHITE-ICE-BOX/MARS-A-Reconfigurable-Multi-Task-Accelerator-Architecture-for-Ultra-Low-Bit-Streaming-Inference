#!/usr/bin/env python3
"""
Verify: switching WITHOUT pipeline flush still produces correct inference.
Measures true end-to-end latency (switch + first inference).
"""
import numpy as np
import mmap
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def infer_batch(ol, ibuf, obuf, test_x, test_y, start, batch_size):
    idma = getattr(ol, 'idma0')
    odma = getattr(ol, 'odma0')
    np.copyto(ibuf, test_x[start:start+batch_size].reshape(batch_size, 32, 32, 3, 1).astype(np.uint8))
    ibuf.flush()
    odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
    idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
    status = odma.read(0x00)
    while status & 0x2 == 0:
        status = odma.read(0x00)
    obuf.invalidate()
    preds = np.array(obuf).flatten().astype(np.int64)
    return int(np.sum(preds == test_y[start:start+batch_size]))


def main():
    from pynq import Overlay, allocate
    from pynq.ps import Clocks

    print("=" * 60)
    print("Switch WITHOUT flush — accuracy & latency verification")
    print("=" * 60)

    ol = Overlay(os.path.join(SCRIPT_DIR, "resizer.bit"))
    Clocks.fclk0_mhz = 100.0

    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    mem = mmap.mmap(fd, 0x10000, offset=0x43C00000)
    os.close(fd)
    cfg = np.frombuffer(mem, dtype=np.uint32)

    def load_u32(name):
        return np.fromfile(os.path.join(SCRIPT_DIR, name), dtype=np.uint32)

    c10 = {
        'mvau0': load_u32("mvau0_thresh_cifar10.bin"),
        'mvau_thresh': [load_u32(f"mvau{i}_thresh_cifar10.bin") for i in range(1, 6)],
        'fc1': load_u32("fc1_thresh_cifar10.bin"),
        'fc2': load_u32("fc2_thresh_cifar10.bin"),
        'cls': load_u32("cls_weights_cifar10.bin"),
    }
    svhn = {
        'mvau0': load_u32("mvau0_thresh_svhn.bin"),
        'mvau_thresh': [load_u32(f"mvau{i}_thresh_svhn.bin") for i in range(1, 6)],
        'mvau_sign': [load_u32(f"mvau{i}_sign_svhn.bin") for i in range(1, 6)],
        'fc1': load_u32("fc1_thresh_svhn.bin"),
        'fc2': load_u32("fc2_thresh_svhn.bin"),
        'cls': load_u32("cls_weights_svhn.bin"),
    }

    def switch_c10():
        cfg[0:len(c10['mvau0'])] = c10['mvau0']
        for i in range(5):
            base = (i + 1) << 11
            t = c10['mvau_thresh'][i]
            cfg[base + 1152 : base + 1152 + len(t)] = t
            cfg[base] = 0
        cfg[0xC000//4 : 0xC000//4 + len(c10['fc1'])] = c10['fc1']
        cfg[0xE000//4 : 0xE000//4 + len(c10['fc2'])] = c10['fc2']
        cfg[0x1000//4 : 0x1000//4 + len(c10['cls'])] = c10['cls']

    def switch_svhn():
        cfg[0:len(svhn['mvau0'])] = svhn['mvau0']
        for i in range(5):
            base = (i + 1) << 11
            t = svhn['mvau_thresh'][i]
            s = svhn['mvau_sign'][i]
            cfg[base + 1152 : base + 1152 + len(t)] = t
            cfg[base + 1408 : base + 1408 + len(s)] = s
            cfg[base] = 1
        cfg[0xC000//4 : 0xC000//4 + len(svhn['fc1'])] = svhn['fc1']
        cfg[0xE000//4 : 0xE000//4 + len(svhn['fc2'])] = svhn['fc2']
        cfg[0x1000//4 : 0x1000//4 + len(svhn['cls'])] = svhn['cls']

    c10_x = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_x.npy"))
    c10_y = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_y.npy"))
    svhn_x = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    svhn_y = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))

    BATCH = 100
    # Pre-allocate buffers once (avoid per-call overhead)
    ibuf = allocate(shape=(BATCH, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(BATCH, 1, 1), dtype=np.uint8, cacheable=True)

    # Warm-up
    switch_svhn()
    infer_batch(ol, ibuf, obuf, svhn_x, svhn_y, 0, BATCH)

    # ============================================================
    # Test 1: Alternating SVHN/CIFAR-10 batches WITHOUT flush
    # ============================================================
    print("\n[1] Alternating batches, NO flush between switches")
    print("    Each round: switch → inference (batch=100)")

    svhn_correct = 0
    c10_correct = 0
    svhn_n = 0
    c10_n = 0

    switch_timings = []
    first_infer_timings = []

    for r in range(20):
        # Switch to SVHN
        t0 = time.time()
        switch_svhn()
        t_sw = (time.time() - t0) * 1000
        switch_timings.append(t_sw)

        t0 = time.time()
        svhn_correct += infer_batch(ol, ibuf, obuf, svhn_x, svhn_y, r*BATCH, BATCH)
        t_inf = (time.time() - t0) * 1000
        first_infer_timings.append(t_inf)
        svhn_n += BATCH

        # Switch to CIFAR-10
        t0 = time.time()
        switch_c10()
        t_sw = (time.time() - t0) * 1000
        switch_timings.append(t_sw)

        t0 = time.time()
        c10_correct += infer_batch(ol, ibuf, obuf, c10_x, c10_y, r*BATCH, BATCH)
        t_inf = (time.time() - t0) * 1000
        first_infer_timings.append(t_inf)
        c10_n += BATCH

    svhn_acc = 100.0 * svhn_correct / svhn_n
    c10_acc = 100.0 * c10_correct / c10_n

    print(f"\n  SVHN accuracy (alternating, no flush):   {svhn_acc:.2f}%  ({svhn_correct}/{svhn_n})")
    print(f"  CIFAR-10 accuracy (alternating, no flush): {c10_acc:.2f}%  ({c10_correct}/{c10_n})")

    print(f"\n  Switch time (no flush):")
    print(f"    avg={np.mean(switch_timings):.3f} ms")
    print(f"    min={np.min(switch_timings):.3f} ms")
    print(f"    max={np.max(switch_timings):.3f} ms")

    print(f"\n  First inference after switch (batch=100):")
    print(f"    avg={np.mean(first_infer_timings):.2f} ms")
    print(f"    min={np.min(first_infer_timings):.2f} ms")

    # ============================================================
    # Test 2: Single image switch → 1 inference latency
    # ============================================================
    print("\n[2] Single-image switch → infer (most realistic latency)")

    ibuf1 = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf1 = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)

    e2e_times = []
    correct_single = 0
    for r in range(30):
        # SVHN
        t0 = time.time()
        switch_svhn()
        correct_single += infer_batch(ol, ibuf1, obuf1, svhn_x, svhn_y, r, 1)
        e2e_times.append((time.time() - t0) * 1000)

        # CIFAR-10
        t0 = time.time()
        switch_c10()
        correct_single += infer_batch(ol, ibuf1, obuf1, c10_x, c10_y, r, 1)
        e2e_times.append((time.time() - t0) * 1000)

    print(f"  End-to-end (switch + 1 inference), no flush:")
    print(f"    avg={np.mean(e2e_times):.3f} ms")
    print(f"    min={np.min(e2e_times):.3f} ms")
    print(f"    max={np.max(e2e_times):.3f} ms")
    print(f"  Accuracy (single samples): {correct_single}/60")

    ibuf.freebuffer(); obuf.freebuffer()
    ibuf1.freebuffer(); obuf1.freebuffer()
    mem.close()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
