#!/usr/bin/env python3
"""
ROM writability sanity test v2 — stronger corruption + control.

Three back-to-back tests on MVAU1:
  (A) Corrupt rom_down with 0xFFFFFFFF (binary projection matrix — should break HARD)
  (B) Corrupt rom_rc with 0xFFFF       (16-bit RC bias)
  (C) Corrupt thresh_mem[0] with 0x7FFFFFFF (control — we know this is writable)

Between each test, restore baseline by re-writing the official .bin files.
"""
import numpy as np
import struct, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_acc(ol, test_x, test_y, batch_size=100, max_samples=1000):
    from pynq import allocate
    idma = getattr(ol, 'idma0'); odma = getattr(ol, 'odma0')
    total = min(test_x.shape[0], max_samples)
    n_batches = total // batch_size
    total = n_batches * batch_size
    correct = 0
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    for b in range(n_batches):
        s = b * batch_size
        bx = test_x[s:s+batch_size].astype(np.uint8); by = test_y[s:s+batch_size]
        np.copyto(ibuf, bx.reshape(batch_size, 32, 32, 3, 1)); ibuf.flush()
        odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
        idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
        while odma.read(0x00) & 0x2 == 0: pass
        obuf.invalidate()
        correct += int(np.sum(np.array(obuf).flatten().astype(np.int64) == by))
    ibuf.freebuffer(); obuf.freebuffer()
    return 100.0 * correct / total


def main():
    from pynq import Overlay, MMIO
    from pynq.ps import Clocks
    print("="*60); print("ROM writability sanity test v2"); print("="*60)
    ol = Overlay(os.path.join(SCRIPT_DIR, "resizer.bit"))
    Clocks.fclk0_mhz = 100.0
    mmio = MMIO(0x43C00000, 0x10000)

    def load_u32(p):
        with open(p, "rb") as f: d = f.read()
        return list(struct.unpack(f"<{len(d)//4}I", d))

    def write_words(byte_off, values):
        for i, v in enumerate(values):
            mmio.write(byte_off + i*4, v & 0xFFFFFFFF)

    def setup_svhn():
        write_words(0x0000, load_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))
        for i in range(1, 6):
            base = i << 13
            write_words(base + 1152*4, load_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_svhn.bin")))
            write_words(base + 1408*4, load_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_svhn.bin")))
            mmio.write(base, 1)
        write_words(0xC000, load_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
        write_words(0xE000, load_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))
        write_words(0x1000, load_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))

    svhn_x = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    svhn_y = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))
    MVAU1 = 1 << 13   # byte base 0x2000

    print("\n--- Baseline (SVHN) ---")
    setup_svhn()
    base_acc = run_acc(ol, svhn_x, svhn_y)
    print(f"  acc = {base_acc:.2f}%")

    # (C) Control: corrupt MVAU1's thresh_mem[0..63] to 0x7FFFFFFF (max signed) — should break
    print("\n--- (C) Control: overwrite MVAU1 thresh_mem[0..63] = 0x7FFFFFFF ---")
    setup_svhn()
    for w in range(0, 64):  # only first 64 channels
        mmio.write(MVAU1 + (1152 + w)*4, 0x7FFFFFFF)
    c_acc = run_acc(ol, svhn_x, svhn_y)
    print(f"  acc = {c_acc:.2f}%  (drop={base_acc-c_acc:+.2f})  [expected: big drop]")

    # (A) Corrupt MVAU1's rom_down with 0xFFFFFFFF (32 words: addr 128..159)
    print("\n--- (A) Overwrite MVAU1 rom_down[128..159] = 0xFFFFFFFF (32 writes) ---")
    setup_svhn()
    for w in range(128, 128 + 32):
        mmio.write(MVAU1 + w*4, 0xFFFFFFFF)
    a_acc = run_acc(ol, svhn_x, svhn_y)
    print(f"  acc = {a_acc:.2f}%  (drop={base_acc-a_acc:+.2f})")

    # (B) Corrupt MVAU1's rom_rc with 0xFFFF (effectively -1 in 16-bit signed)
    print("\n--- (B) Overwrite MVAU1 rom_rc[4..19] = 0x0000FFFF (16 writes) ---")
    setup_svhn()
    for w in range(4, 4 + 16):
        mmio.write(MVAU1 + w*4, 0x0000FFFF)
    b_acc = run_acc(ol, svhn_x, svhn_y)
    print(f"  acc = {b_acc:.2f}%  (drop={base_acc-b_acc:+.2f})")

    # ------- Verdict -------
    print("\n="*1 + "=== Verdict ===")
    print(f"  baseline = {base_acc:.2f}%")
    print(f"  (C) thresh write   → {c_acc:.2f}%  (control — sanity check)")
    print(f"  (A) rom_down write → {a_acc:.2f}%")
    print(f"  (B) rom_rc   write → {b_acc:.2f}%")
    if base_acc - c_acc < 5.0:
        print("  ⚠️  Control DID NOT break inference — cfg path itself may be broken!")
    else:
        if base_acc - a_acc >= 5.0 or base_acc - b_acc >= 5.0:
            print("  ✅ rom_down/rom_rc IS writable. Multi-dataset adapter switching is software-only.")
        else:
            print("  ❌ thresh_mem writable but rom_down/rom_rc NOT — these specific RAMs are ROM in netlist.")
            print("     → Need bitstream rebuild that forces them to RAM.")


if __name__ == "__main__":
    main()
