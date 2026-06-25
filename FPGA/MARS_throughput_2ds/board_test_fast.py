#!/usr/bin/env python3
"""
Fast runtime switching benchmark — numpy/mmap bulk writes vs Python loop.
Measures switching latency and verifies accuracy after fast switch.
"""
import numpy as np
import mmap
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_quick_test(ol, test_x, test_y, n_samples=2000, batch_size=100):
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
        odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
        idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
        status = odma.read(0x00)
        while status & 0x2 == 0:
            status = odma.read(0x00)
        obuf.invalidate()
        preds = np.array(obuf).flatten().astype(np.int64)
        correct += np.sum(preds == test_y[s:s+batch_size])
        tested += batch_size
    ibuf.freebuffer(); obuf.freebuffer()
    return 100.0 * correct / tested


def flush(ol):
    from pynq import allocate
    idma = getattr(ol, 'idma0')
    odma = getattr(ol, 'odma0')
    ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)
    ibuf[:] = 0; ibuf.flush()
    odma.write(0x10, obuf.device_address); odma.write(0x1C, 1); odma.write(0x00, 1)
    idma.write(0x10, ibuf.device_address); idma.write(0x1C, 1); idma.write(0x00, 1)
    status = odma.read(0x00)
    while status & 0x2 == 0:
        status = odma.read(0x00)
    ibuf.freebuffer(); obuf.freebuffer()


def main():
    from pynq import Overlay, MMIO
    from pynq.ps import Clocks

    print("=" * 60)
    print("Fast Runtime Switching Benchmark")
    print("=" * 60)

    ol = Overlay(os.path.join(SCRIPT_DIR, "resizer.bit"))
    Clocks.fclk0_mhz = 100.0

    # --- Method 1: original MMIO (slow, for comparison) ---
    mmio = MMIO(0x43C00000, 0x10000)

    # --- Method 2: numpy array backed by /dev/mem mmap (fast) ---
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    mem = mmap.mmap(fd, 0x10000, offset=0x43C00000)
    os.close(fd)
    cfg = np.frombuffer(mem, dtype=np.uint32)

    # Pre-load all binary param files as numpy uint32 arrays
    def load_u32(name):
        return np.fromfile(os.path.join(SCRIPT_DIR, name), dtype=np.uint32)

    # CIFAR-10 config
    c10 = {
        'mvau0': load_u32("mvau0_thresh_cifar10.bin"),
        'mvau_thresh': [load_u32(f"mvau{i}_thresh_cifar10.bin") for i in range(1, 6)],
        'fc1': load_u32("fc1_thresh_cifar10.bin"),
        'fc2': load_u32("fc2_thresh_cifar10.bin"),
        'cls': load_u32("cls_weights_cifar10.bin"),
    }
    # SVHN config
    svhn = {
        'mvau0': load_u32("mvau0_thresh_svhn.bin"),
        'mvau_thresh': [load_u32(f"mvau{i}_thresh_svhn.bin") for i in range(1, 6)],
        'mvau_sign': [load_u32(f"mvau{i}_sign_svhn.bin") for i in range(1, 6)],
        'fc1': load_u32("fc1_thresh_svhn.bin"),
        'fc2': load_u32("fc2_thresh_svhn.bin"),
        'cls': load_u32("cls_weights_svhn.bin"),
    }

    # Test data
    c10_x = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_x.npy"))
    c10_y = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_y.npy"))
    svhn_x = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    svhn_y = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))

    # ---- Fast switch functions (numpy bulk write) ----

    def fast_switch_cifar10():
        n = len(c10['mvau0'])
        cfg[0:n] = c10['mvau0']
        for i in range(5):
            base = (i + 1) << 11          # unit (i+1): byte (i+1)<<13, word (i+1)<<11
            t = c10['mvau_thresh'][i]
            cfg[base + 1152 : base + 1152 + len(t)] = t
            cfg[base] = 0                 # adapter OFF
        n1 = len(c10['fc1'])
        cfg[0xC000//4 : 0xC000//4 + n1] = c10['fc1']
        n2 = len(c10['fc2'])
        cfg[0xE000//4 : 0xE000//4 + n2] = c10['fc2']
        nc = len(c10['cls'])
        cfg[0x1000//4 : 0x1000//4 + nc] = c10['cls']

    def fast_switch_svhn():
        n = len(svhn['mvau0'])
        cfg[0:n] = svhn['mvau0']
        for i in range(5):
            base = (i + 1) << 11
            t = svhn['mvau_thresh'][i]
            s = svhn['mvau_sign'][i]
            cfg[base + 1152 : base + 1152 + len(t)] = t
            cfg[base + 1408 : base + 1408 + len(s)] = s
            cfg[base] = 1                 # adapter ON
        n1 = len(svhn['fc1'])
        cfg[0xC000//4 : 0xC000//4 + n1] = svhn['fc1']
        n2 = len(svhn['fc2'])
        cfg[0xE000//4 : 0xE000//4 + n2] = svhn['fc2']
        nc = len(svhn['cls'])
        cfg[0x1000//4 : 0x1000//4 + nc] = svhn['cls']

    # ---- Slow switch functions (original mmio.write loop) ----

    def slow_write(byte_offset, values):
        for i, val in enumerate(values):
            mmio.write(byte_offset + i * 4, int(val) & 0xFFFFFFFF)

    def slow_switch_cifar10():
        slow_write(0x0000, c10['mvau0'])
        for i in range(5):
            base = (i + 1) << 13
            slow_write(base + 1152 * 4, c10['mvau_thresh'][i])
            mmio.write(base, 0)
        slow_write(0xC000, c10['fc1'])
        slow_write(0xE000, c10['fc2'])
        slow_write(0x1000, c10['cls'])

    def slow_switch_svhn():
        slow_write(0x0000, svhn['mvau0'])
        for i in range(5):
            base = (i + 1) << 13
            slow_write(base + 1152 * 4, svhn['mvau_thresh'][i])
            slow_write(base + 1408 * 4, svhn['mvau_sign'][i])
            mmio.write(base, 1)
        slow_write(0xC000, svhn['fc1'])
        slow_write(0xE000, svhn['fc2'])
        slow_write(0x1000, svhn['cls'])

    # ============================================================
    # Test 1: Verify fast switch produces correct accuracy
    # ============================================================
    print("\n" + "=" * 40)
    print("TEST 1: Fast switch accuracy check")
    print("=" * 40)

    fast_switch_svhn()
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y, 2000)
    print(f"  SVHN:     {acc:.1f}%")

    fast_switch_cifar10()
    flush(ol)
    acc = run_quick_test(ol, c10_x, c10_y, 2000)
    print(f"  CIFAR-10: {acc:.1f}%")

    # ============================================================
    # Test 2: Speed comparison — slow vs fast (no flush)
    # ============================================================
    print("\n" + "=" * 40)
    print("TEST 2: Write-only switching speed")
    print("=" * 40)

    N_ROUNDS = 50

    # Slow
    slow_c10 = []
    slow_svhn = []
    for _ in range(N_ROUNDS):
        t0 = time.time(); slow_switch_cifar10(); slow_c10.append((time.time()-t0)*1000)
        t0 = time.time(); slow_switch_svhn();    slow_svhn.append((time.time()-t0)*1000)

    print(f"\n  Slow (mmio.write loop), {N_ROUNDS} rounds:")
    print(f"    →CIFAR-10: avg={np.mean(slow_c10):.2f} ms, min={np.min(slow_c10):.2f} ms")
    print(f"    →SVHN:     avg={np.mean(slow_svhn):.2f} ms, min={np.min(slow_svhn):.2f} ms")

    # Fast
    fast_c10 = []
    fast_svhn = []
    for _ in range(N_ROUNDS):
        t0 = time.time(); fast_switch_cifar10(); fast_c10.append((time.time()-t0)*1000)
        t0 = time.time(); fast_switch_svhn();    fast_svhn.append((time.time()-t0)*1000)

    print(f"\n  Fast (numpy/mmap bulk), {N_ROUNDS} rounds:")
    print(f"    →CIFAR-10: avg={np.mean(fast_c10):.3f} ms, min={np.min(fast_c10):.3f} ms, max={np.max(fast_c10):.3f} ms")
    print(f"    →SVHN:     avg={np.mean(fast_svhn):.3f} ms, min={np.min(fast_svhn):.3f} ms, max={np.max(fast_svhn):.3f} ms")

    speedup_c10 = np.mean(slow_c10) / np.mean(fast_c10)
    speedup_svhn = np.mean(slow_svhn) / np.mean(fast_svhn)
    print(f"\n  Speedup: {speedup_c10:.0f}x (CIFAR-10), {speedup_svhn:.0f}x (SVHN)")

    # ============================================================
    # Test 3: Switch + flush total latency
    # ============================================================
    print("\n" + "=" * 40)
    print("TEST 3: Fast switch + pipeline flush")
    print("=" * 40)

    sf_times = []
    for _ in range(30):
        t0 = time.time()
        fast_switch_cifar10()
        flush(ol)
        sf_times.append((time.time()-t0)*1000)

        t0 = time.time()
        fast_switch_svhn()
        flush(ol)
        sf_times.append((time.time()-t0)*1000)

    print(f"  Switch + flush (60 switches):")
    print(f"    avg={np.mean(sf_times):.3f} ms")
    print(f"    min={np.min(sf_times):.3f} ms")
    print(f"    max={np.max(sf_times):.3f} ms")

    # ============================================================
    # Test 4: Full accuracy on 10000 samples after fast switch
    # ============================================================
    print("\n" + "=" * 40)
    print("TEST 4: Full accuracy verification (10k)")
    print("=" * 40)

    fast_switch_svhn()
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y, 10000)
    print(f"  SVHN:     {acc:.2f}%  (target: 71.8%)")

    fast_switch_cifar10()
    flush(ol)
    acc = run_quick_test(ol, c10_x, c10_y, 10000)
    print(f"  CIFAR-10: {acc:.2f}%  (target: ~74%)")

    mem.close()

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
