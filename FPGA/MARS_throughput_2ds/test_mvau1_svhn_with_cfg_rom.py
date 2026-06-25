#!/usr/bin/env python3
"""Verify MVAU1's per-bank distRAM works: explicitly cfg-write SVHN rom_*
values, then run SVHN inference. If acc returns to ~71%, the per-bank
refactor is functioning correctly.
"""
import numpy as np, struct, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_acc(ol, test_x, test_y, batch_size=100, max_samples=2000):
    from pynq import allocate
    idma = getattr(ol, 'idma0'); odma = getattr(ol, 'odma0')
    n_batches = min(test_x.shape[0], max_samples) // batch_size
    total = n_batches * batch_size
    correct = 0
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    for b in range(n_batches):
        s = b * batch_size
        np.copyto(ibuf, test_x[s:s+batch_size].astype(np.uint8).reshape(batch_size, 32, 32, 3, 1))
        ibuf.flush()
        odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
        idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
        while odma.read(0x00) & 0x2 == 0: pass
        obuf.invalidate()
        correct += int(np.sum(np.array(obuf).flatten().astype(np.int64) == test_y[s:s+batch_size]))
    ibuf.freebuffer(); obuf.freebuffer()
    return 100.0 * correct / total


def flush(ol):
    from pynq import allocate
    idma = getattr(ol, 'idma0'); odma = getattr(ol, 'odma0')
    ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)
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
    mmio = MMIO(0x43C00000, 0x10000)
    def load_u32(p):
        with open(p, "rb") as f: d = f.read()
        return list(struct.unpack(f"<{len(d)//4}I", d))
    def write_words(off, vs):
        for i, v in enumerate(vs): mmio.write(off + i*4, v & 0xFFFFFFFF)

    # SVHN baseline setup (thresh, sign, cls, fc — all MVAUs)
    write_words(0x0000, load_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))
    for i in range(1, 6):
        base = i << 13
        write_words(base + 1152*4, load_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_svhn.bin")))
        write_words(base + 1408*4, load_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_svhn.bin")))
        mmio.write(base, 1)
    write_words(0xC000, load_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
    write_words(0xE000, load_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))
    write_words(0x1000, load_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))
    flush(ol)

    sx = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    sy = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))
    acc_before = run_acc(ol, sx, sy)
    print(f"BEFORE writing rom_*: SVHN acc = {acc_before:.2f}%")

    # Now write MVAU1's rom_rc/down/up SVHN values via cfg
    base_mvau1 = 1 << 13  # 0x2000
    write_words(base_mvau1 + 4*4,   load_u32(os.path.join(SCRIPT_DIR, "mvau1_rom_rc_svhn.bin")))
    write_words(base_mvau1 + 128*4, load_u32(os.path.join(SCRIPT_DIR, "mvau1_rom_down_svhn.bin")))
    write_words(base_mvau1 + 640*4, load_u32(os.path.join(SCRIPT_DIR, "mvau1_rom_up_svhn.bin")))
    flush(ol)

    acc_after = run_acc(ol, sx, sy)
    print(f"AFTER  writing rom_*: SVHN acc = {acc_after:.2f}%")
    print(f"  delta = {acc_after - acc_before:+.2f} pp")
    if acc_after > 65.0:
        print("✅ MVAU1 per-bank rom_* works perfectly!")
    elif acc_after > acc_before + 10:
        print("⚠️  Some improvement but not full — partial mapping issue?")
    else:
        print("❌ rom_* explicit write didn't help — issue elsewhere")


if __name__ == "__main__":
    main()
