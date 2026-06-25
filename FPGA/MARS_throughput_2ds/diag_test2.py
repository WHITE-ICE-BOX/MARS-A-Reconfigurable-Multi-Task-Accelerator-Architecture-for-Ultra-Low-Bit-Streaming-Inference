#!/usr/bin/env python3
"""
Diagnostic test 2: Write zeros to specific layers to verify writes take effect.
"""
import numpy as np
import struct
import time
import os

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
        while status & 0x2 == 0: status = odma.read(0x00)
        obuf.invalidate()
        preds = np.array(obuf).flatten().astype(np.int64)
        correct += np.sum(preds == test_y[s:s+batch_size])
        tested += batch_size
    ibuf.freebuffer(); obuf.freebuffer()
    return 100.0 * correct / tested


def flush(ol, n=3):
    """Flush pipeline with n dummy inferences."""
    from pynq import allocate
    idma = getattr(ol, 'idma0')
    odma = getattr(ol, 'odma0')
    for _ in range(n):
        ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
        obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)
        ibuf[:] = 0; ibuf.flush()
        odma.write(0x10, obuf.device_address); odma.write(0x1C, 1); odma.write(0x00, 1)
        idma.write(0x10, ibuf.device_address); idma.write(0x1C, 1); idma.write(0x00, 1)
        status = odma.read(0x00)
        while status & 0x2 == 0: status = odma.read(0x00)
        ibuf.freebuffer(); obuf.freebuffer()


def main():
    from pynq import Overlay, MMIO
    from pynq.ps import Clocks

    print("=" * 60)
    print("Diagnostic 2: Verify writes with zero-writes")
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

    def write_all_svhn():
        write_words(0x0000, load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin")))
        for i in range(1, 6):
            base = i << 13
            write_words(base + 1152 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_thresh_svhn.bin")))
            write_words(base + 1408 * 4, load_bin_u32(os.path.join(SCRIPT_DIR, f"mvau{i}_sign_svhn.bin")))
            mmio.write(base, 1)
        write_words(0xC000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin")))
        write_words(0xE000, load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin")))
        write_words(0x1000, load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin")))

    svhn_x = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    svhn_y = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))

    # Baseline: write all SVHN
    print("\n[A] Write all SVHN params")
    write_all_svhn()
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN accuracy: {acc:.1f}%")

    # Test B: Write ZEROS to FC1 thresholds (512 words)
    print("\n[B] Zero out FC1 thresholds")
    write_words(0xC000, [0] * 512)
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN accuracy: {acc:.1f}%  (should drop if FC1 write works)")

    # Restore
    print("    Restoring FC1...")
    write_all_svhn()
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN after restore: {acc:.1f}%")

    # Test C: Write ZEROS to FC2 thresholds (512 words)
    print("\n[C] Zero out FC2 thresholds")
    write_words(0xE000, [0] * 512)
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN accuracy: {acc:.1f}%  (should drop if FC2 write works)")

    # Restore
    print("    Restoring FC2...")
    write_all_svhn()
    flush(ol)

    # Test D: Write ZEROS to classifier (1024 words)
    print("\n[D] Zero out classifier weights")
    write_words(0x1000, [0] * 1024)
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN accuracy: {acc:.1f}%  (should drop if classifier write works)")

    # Restore
    print("    Restoring classifier...")
    write_all_svhn()
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN after restore: {acc:.1f}%")

    # Test E: Write ZEROS to MVAU0 thresholds (64 words)
    print("\n[E] Zero out MVAU0 thresholds")
    write_words(0x0000, [0] * 64)
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN accuracy: {acc:.1f}%  (should drop if MVAU0 write works)")

    # Restore
    print("    Restoring MVAU0...")
    write_all_svhn()
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN after restore: {acc:.1f}%")

    # Test F: Read back classifier state - write known pattern
    print("\n[F] Classifier write verification: write 0xAA to entry 0")
    # Write pattern 0xAA to first entry
    mmio.write(0x1000, 0x000000AA)
    import time; time.sleep(0.01)
    # Read it back via MMIO (if cfg_hub supports read - it doesn't, but let's try)
    # Instead, test with a distinctive known value

    # Write inverted classifier weights (bitwise NOT of each byte)
    cls_svhn = load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin"))
    cls_inverted = [(~v) & 0xFF for v in cls_svhn]
    print(f"    Writing inverted classifier weights (first 5: {cls_inverted[:5]})")
    write_words(0x1000, cls_inverted)
    flush(ol, n=5)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    SVHN accuracy with inverted cls: {acc:.1f}%")

    # Final restore
    write_all_svhn()
    flush(ol)
    acc = run_quick_test(ol, svhn_x, svhn_y)
    print(f"    Final restore: {acc:.1f}%")

    print("\n" + "=" * 60)
    print("Diagnostic 2 complete!")


if __name__ == "__main__":
    main()
