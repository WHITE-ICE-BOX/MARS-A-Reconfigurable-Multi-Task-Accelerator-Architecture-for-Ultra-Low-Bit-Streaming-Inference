#!/usr/bin/env python3
"""
ROM writability sanity test.
Baseline: standard SVHN mode → ~71% acc.
Then overwrite MVAU1's rom_rc[0..15] with zeros (CFG_RC_BASE=4, HIDDEN_CH=16) → re-run.
If acc drops → distributed RAM is runtime-writable (path is wired).
If acc unchanged → it really is ROM / cfg path not wired → need bitstream rebuild.
"""
import numpy as np
import struct, time, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_acc(ol, mmio, test_x, test_y, batch_size=100, max_samples=2000):
    from pynq import allocate
    idma = getattr(ol, 'idma0')
    odma = getattr(ol, 'odma0')

    total = min(test_x.shape[0], max_samples)
    n_batches = total // batch_size
    total = n_batches * batch_size

    correct = 0
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    for b in range(n_batches):
        s = b * batch_size
        bx = test_x[s:s+batch_size].astype(np.uint8)
        by = test_y[s:s+batch_size]
        np.copyto(ibuf, bx.reshape(batch_size, 32, 32, 3, 1))
        ibuf.flush()
        odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
        idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
        while odma.read(0x00) & 0x2 == 0: pass
        obuf.invalidate()
        preds = np.array(obuf).flatten().astype(np.int64)
        correct += int(np.sum(preds == by))
    ibuf.freebuffer(); obuf.freebuffer()
    return 100.0 * correct / total, total


def flush(ol, batch_size=1):
    from pynq import allocate
    idma = getattr(ol, 'idma0'); odma = getattr(ol, 'odma0')
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    ibuf[:] = 0; ibuf.flush()
    odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
    idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
    while odma.read(0x00) & 0x2 == 0: pass
    ibuf.freebuffer(); obuf.freebuffer()


def main():
    from pynq import Overlay, MMIO
    from pynq.ps import Clocks

    print("=" * 60)
    print("ROM writability sanity test (MVAU1 rom_rc)")
    print("=" * 60)

    ol = Overlay(os.path.join(SCRIPT_DIR, "resizer.bit"))
    Clocks.fclk0_mhz = 100.0
    mmio = MMIO(0x43C00000, 0x10000)

    def load_u32(path):
        with open(path, "rb") as f: data = f.read()
        return list(struct.unpack(f"<{len(data)//4}I", data))

    def write_words(byte_off, values):
        for i, v in enumerate(values):
            mmio.write(byte_off + i*4, v & 0xFFFFFFFF)

    # ----- Standard SVHN mode -----
    print("\n[1/3] Set SVHN mode (adapter ON, SVHN thresholds)…")
    write_words(0x0000, load_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))
    for i in range(1, 6):
        base = i << 13
        write_words(base + 1152*4, load_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_svhn.bin")))
        write_words(base + 1408*4, load_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_svhn.bin")))
        mmio.write(base, 1)  # adapter_enable=1
    write_words(0xC000, load_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
    write_words(0xE000, load_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))
    write_words(0x1000, load_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))
    flush(ol)

    svhn_x = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    svhn_y = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))

    acc_before, n = run_acc(ol, mmio, svhn_x, svhn_y, batch_size=100, max_samples=2000)
    print(f"  Baseline SVHN acc: {acc_before:.2f}%  ({n} samples)  (target ~71%)")

    # ----- Corrupt MVAU1's rom_rc -----
    # MVAU1 base = 0x2000, CFG_RC_BASE word = 4 (byte 0x10), HIDDEN_CH = 16 → words 4..19
    print("\n[2/3] Overwrite MVAU1 rom_rc[0..15] with 0x0000 (16 writes)…")
    base = 1 << 13  # MVAU1 byte base = 0x2000
    for w in range(4, 4 + 16):
        mmio.write(base + w*4, 0x00000000)
    flush(ol)

    acc_after, _ = run_acc(ol, mmio, svhn_x, svhn_y, batch_size=100, max_samples=2000)
    print(f"  After-corruption SVHN acc: {acc_after:.2f}%")

    # ----- Verdict -----
    print("\n[3/3] Verdict:")
    drop = acc_before - acc_after
    print(f"  Drop = {drop:+.2f} pp")
    if drop >= 3.0:
        print("  ✅ Distributed RAM IS runtime-writable. cfg path is live.")
        print("     → Multi-dataset adapter switching only needs software (.bin files).")
    elif abs(drop) < 1.0:
        print("  ❌ Accuracy unchanged. rom_rc writes had NO effect.")
        print("     → Either it really is ROM in the netlist, or cfg path isn't wired.")
        print("     → Need bitstream rebuild.")
    else:
        print(f"  ⚠️  Borderline drop ({drop:.2f}). Try more aggressive corruption "
              "or smaller batch noise margin.")


if __name__ == "__main__":
    main()
