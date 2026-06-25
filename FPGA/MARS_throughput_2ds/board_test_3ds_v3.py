#!/usr/bin/env python3
"""3-dataset on-board test (v3): full v2 RTL — all 5 MVAU adapter ROMs cfg-writable.

Per-dataset writes (compared to v2 which only had MVAU1 rom_*):
  every MVAU 1..5 gets rom_rc + rom_down + rom_up + contrib_lut + thresh + sign
  + adapter_enable.

Address layout (matches mvau_adapter_v2.v):
  word_base = i << 11        # i in 1..5
  RC_BASE   = word_base + 4
  DOWN_BASE = word_base + 128
  UP_BASE   = word_base + 640
  TH_BASE   = word_base + 1152
  SG_BASE   = word_base + 1408
  LUT_BASE  = word_base + 1664

Bitstream expected: fpga/resizer_v2.bit + resizer_v2.hwh (Phase 4 output)
"""
import numpy as np, struct, time, os, mmap
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_acc(ol, test_x, test_y, batch_size=100, max_samples=None):
    from pynq import allocate
    idma = getattr(ol, 'idma0'); odma = getattr(ol, 'odma0')
    total = test_x.shape[0]
    if max_samples: total = min(total, max_samples)
    n = total // batch_size; total = n * batch_size
    correct = 0
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    for b in range(n):
        s = b * batch_size
        np.copyto(ibuf, test_x[s:s+batch_size].astype(np.uint8).reshape(batch_size,32,32,3,1))
        ibuf.flush()
        odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
        idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
        while odma.read(0x00) & 0x2 == 0: pass
        obuf.invalidate()
        correct += int(np.sum(np.array(obuf).flatten().astype(np.int64) == test_y[s:s+batch_size]))
    ibuf.freebuffer(); obuf.freebuffer()
    return 100.0 * correct / total, total


def flush(ol):
    from pynq import allocate
    idma = getattr(ol, 'idma0'); odma = getattr(ol, 'odma0')
    ibuf = allocate(shape=(1,32,32,3,1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(1,1,1), dtype=np.uint8, cacheable=True)
    ibuf[:] = 0; ibuf.flush()
    odma.write(0x10, obuf.device_address); odma.write(0x1C, 1); odma.write(0x00, 1)
    idma.write(0x10, ibuf.device_address); idma.write(0x1C, 1); idma.write(0x00, 1)
    while odma.read(0x00) & 0x2 == 0: pass
    ibuf.freebuffer(); obuf.freebuffer()


def main():
    from pynq import Overlay
    from pynq.ps import Clocks

    bit_path = os.path.join(SCRIPT_DIR, "resizer_v2.bit")
    if not os.path.exists(bit_path):
        bit_path = os.path.join(SCRIPT_DIR, "resizer.bit")
    print(f"Using bitstream: {bit_path}")
    ol = Overlay(bit_path)
    Clocks.fclk0_mhz = 100.0  # v2 timing met (WNS=0.000ns) at 100 MHz

    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    # v2 BD auto-assigned cfg_hub to 0x40010000 (vs v1 0x43C00000)
    mem = mmap.mmap(fd, 0x10000, offset=0x40010000)
    os.close(fd)
    cfg = np.frombuffer(mem, dtype=np.uint32)

    def load(name):
        with open(os.path.join(SCRIPT_DIR, name),"rb") as f:
            return np.frombuffer(f.read(), dtype=np.uint32)

    def build_writes(tag, adapter_on):
        """Return list of (word_addr, np.uint32 array) tuples."""
        w = []
        # MVAU0 thresh
        w.append((0, load(f"mvau0_thresh_{tag}.bin")))
        # MVAU1..5
        for i in range(1, 6):
            base = i << 11
            w.append((base + 1152, load(f"mvau{i}_thresh_{tag}.bin")))
            if adapter_on:
                w.append((base + 1408, load(f"mvau{i}_sign_{tag}.bin")))
                w.append((base + 4,    load(f"mvau{i}_rc_{tag}.bin")))
                w.append((base + 128,  load(f"mvau{i}_down_{tag}.bin")))
                w.append((base + 640,  load(f"mvau{i}_up_{tag}.bin")))
                w.append((base + 1664, load(f"mvau{i}_lut_{tag}.bin")))
            # adapter_enable @ word 0 (per-MVAU)
            w.append((base, np.array([1 if adapter_on else 0], dtype=np.uint32)))
        # FC1, FC2, classifier
        w.append((6 << 11, load(f"fc1_thresh_{tag}.bin")))
        w.append((7 << 11, load(f"fc2_thresh_{tag}.bin")))
        w.append((0x1000 >> 2, load(f"cls_weights_{tag}.bin")))
        return w

    def apply(writes):
        for word_addr, vals in writes:
            cfg[word_addr : word_addr + len(vals)] = vals

    svhn_w    = build_writes("svhn",    adapter_on=True)
    c10_w     = build_writes("cifar10", adapter_on=False)
    fashion_w = build_writes("fashion", adapter_on=True)

    # ---- SVHN ----
    print("\n[SVHN]")
    t0 = time.time(); apply(svhn_w); ts = (time.time()-t0)*1000
    flush(ol)
    sx = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    sy = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))
    acc, n = run_acc(ol, sx, sy, max_samples=200)
    print(f"  switch={ts:.2f}ms  acc={acc:.2f}% ({int(acc*n/100)}/{n})")

    # ---- CIFAR-10 ----
    print("\n[CIFAR-10]")
    t0 = time.time(); apply(c10_w); ts = (time.time()-t0)*1000
    flush(ol)
    cx = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_x.npy"))
    cy = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_y.npy"))
    acc, n = run_acc(ol, cx, cy, max_samples=200)
    print(f"  switch={ts:.2f}ms  acc={acc:.2f}% ({int(acc*n/100)}/{n})")

    # ---- FashionMNIST ----
    print("\n[FashionMNIST]")
    t0 = time.time(); apply(fashion_w); ts = (time.time()-t0)*1000
    flush(ol)
    fx = np.load(os.path.join(SCRIPT_DIR, "fashion_test_x.npy"))
    fy = np.load(os.path.join(SCRIPT_DIR, "fashion_test_y.npy"))
    acc, n = run_acc(ol, fx, fy, max_samples=200)
    print(f"  switch={ts:.2f}ms  acc={acc:.2f}% ({int(acc*n/100)}/{n})")

    # ---- Round-robin switch latency ----
    print("\n[Switch latency round-robin]")
    seq = [("CIFAR-10", c10_w), ("SVHN", svhn_w), ("Fashion", fashion_w)]
    times = []
    for i in range(20):
        name, ws = seq[i % 3]
        t0 = time.time(); apply(ws); times.append((time.time()-t0)*1000)
    print(f"  avg={np.mean(times):.2f}ms  max={np.max(times):.2f}ms  min={np.min(times):.2f}ms")


if __name__ == "__main__":
    main()
