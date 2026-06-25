#!/usr/bin/env python3
"""3-dataset on-board test (v2): MVAU1 cfg-writable adapter; MVAU2-5 share
SVHN-baked adapter (resource budget constraint).

Per-dataset writes:
  CIFAR-10 (adapter OFF): mvau0_thresh + mvau{1-5}_thresh + fc{1,2}_thresh + cls + adapter_enable=0
  SVHN     (adapter ON ): + sign + MVAU1 rom_rc/down/up (SVHN values)
  Fashion  (adapter ON ): + sign + MVAU1 rom_rc/down/up (Fashion values)
                          (MVAU2-5 reuse SVHN bake)
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
    from pynq import Overlay, MMIO
    from pynq.ps import Clocks

    ol = Overlay(os.path.join(SCRIPT_DIR, "resizer.bit"))
    Clocks.fclk0_mhz = 100.0

    # Fast mmap path
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    mem = mmap.mmap(fd, 0x10000, offset=0x43C00000)
    os.close(fd)
    cfg = np.frombuffer(mem, dtype=np.uint32)  # uint32 view

    def load_u32_arr(p):
        with open(p,"rb") as f: return np.frombuffer(f.read(), dtype=np.uint32)

    def build_writes(tag, adapter_on, mvau1_rom_files=None):
        """Return list of (word_addr, np.uint32 array) tuples to apply via bulk slice."""
        w = []
        # MVAU0 thresh (word 0..63)
        w.append((0, load_u32_arr(os.path.join(SCRIPT_DIR, f"mvau0_thresh_{tag}.bin"))))
        for i in range(1, 6):
            base = i << 11  # word base = i*2048
            # thresh @ word 1152
            w.append((base + 1152, load_u32_arr(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_{tag}.bin"))))
            if adapter_on:
                # sign @ word 1408
                w.append((base + 1408, load_u32_arr(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_{tag}.bin"))))
        # MVAU1 rom_* (per-bank, only refactored MVAU)
        if adapter_on and mvau1_rom_files is not None:
            rc_f, down_f, up_f = mvau1_rom_files
            base1 = 1 << 11
            w.append((base1 + 4,   load_u32_arr(os.path.join(SCRIPT_DIR, rc_f))))
            w.append((base1 + 128, load_u32_arr(os.path.join(SCRIPT_DIR, down_f))))
            w.append((base1 + 640, load_u32_arr(os.path.join(SCRIPT_DIR, up_f))))
        # adapter_enable per MVAU (1-bit at word 0 of each)
        for i in range(1, 6):
            w.append((i << 11, np.array([1 if adapter_on else 0], dtype=np.uint32)))
        # FC1, FC2, classifier
        w.append((6 << 11, load_u32_arr(os.path.join(SCRIPT_DIR, f"fc1_thresh_{tag}.bin"))))
        w.append((7 << 11, load_u32_arr(os.path.join(SCRIPT_DIR, f"fc2_thresh_{tag}.bin"))))
        w.append((0x1000 >> 2, load_u32_arr(os.path.join(SCRIPT_DIR, f"cls_weights_{tag}.bin"))))
        return w

    def apply(writes):
        for word_addr, vals in writes:
            cfg[word_addr : word_addr + len(vals)] = vals

    svhn_w = build_writes("svhn", adapter_on=True,
                          mvau1_rom_files=("mvau1_rom_rc_svhn.bin","mvau1_rom_down_svhn.bin","mvau1_rom_up_svhn.bin"))
    c10_w = build_writes("cifar10", adapter_on=False)
    fashion_w = build_writes("fashion", adapter_on=True,
                             mvau1_rom_files=("mvau1_rom_rc_fashion.bin","mvau1_rom_down_fashion.bin","mvau1_rom_up_fashion.bin"))

    # ---- SVHN ----
    print("\n[SVHN]")
    t0 = time.time(); apply(svhn_w); ts = (time.time()-t0)*1000
    flush(ol)
    sx = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    sy = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))
    acc, n = run_acc(ol, sx, sy, max_samples=10000)
    print(f"  switch={ts:.2f}ms  acc={acc:.2f}% ({int(acc*n/100)}/{n})")

    # ---- CIFAR-10 ----
    print("\n[CIFAR-10]")
    t0 = time.time(); apply(c10_w); ts = (time.time()-t0)*1000
    flush(ol)
    cx = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_x.npy"))
    cy = np.load(os.path.join(SCRIPT_DIR, "cifar10_test_y.npy"))
    acc, n = run_acc(ol, cx, cy, max_samples=10000)
    print(f"  switch={ts:.2f}ms  acc={acc:.2f}% ({int(acc*n/100)}/{n})")

    # ---- FashionMNIST ----
    print("\n[FashionMNIST]")
    t0 = time.time(); apply(fashion_w); ts = (time.time()-t0)*1000
    flush(ol)
    fx = np.load(os.path.join(SCRIPT_DIR, "fashion_test_x.npy"))
    fy = np.load(os.path.join(SCRIPT_DIR, "fashion_test_y.npy"))
    acc, n = run_acc(ol, fx, fy, max_samples=10000)
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
