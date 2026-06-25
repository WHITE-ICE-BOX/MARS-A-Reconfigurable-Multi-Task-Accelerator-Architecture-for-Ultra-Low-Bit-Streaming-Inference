#!/usr/bin/env python3
"""
Explicitly cfg-write SVHN adapter weights for MVAU1 (rom_rc/down/up),
then run SVHN inference. If accuracy returns to ~71%, the cfg path is
functioning correctly and the previous low baseline was just due to
$readmemh init failing to populate the per-bank distRAMs.
"""
import numpy as np, struct, time, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_acc(ol, test_x, test_y, batch_size=100, max_samples=2000):
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

    def write_words(byte_off, vals):
        for i, v in enumerate(vals):
            mmio.write(byte_off + i*4, v & 0xFFFFFFFF)

    # ======== Set SVHN mode (thresh + sign + cls + fc, all MVAUs) ========
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

    # Load SVHN test data
    sx = np.load(os.path.join(SCRIPT_DIR, "svhn_test_x.npy"))
    sy = np.load(os.path.join(SCRIPT_DIR, "svhn_test_y.npy"))

    # Baseline WITHOUT explicit rom_* load (relying on initial $readmemh)
    acc_no = run_acc(ol, sx, sy, max_samples=2000)
    print(f"WITHOUT explicit MVAU1 rom_* load: SVHN acc = {acc_no:.2f}% (baseline)")

    # Now explicitly cfg-write MVAU1's rom_rc, rom_down, rom_up
    # Generate SVHN values from .dat files (need to extract from MVAU1 data)
    DATA = "/home/xilinx/runtime_switch/svhn_adapter_data"  # we'll need to ship these
    # For now: read the existing SVHN .bin files we have... but those are for adapter_thresh/sign,
    # not rom_*. Need NEW .bin files for rom_*.

    # As a quick verification: write some KNOWN-distinct value to ALL of rom_rc/down/up of MVAU1
    # and confirm accuracy changes drastically.
    base_mvau1 = 1 << 13  # 0x2000

    print("\n>>> Writing rom_rc = 0 (16 words at off 4..19) ...")
    for w in range(4, 4 + 16):
        mmio.write(base_mvau1 + w*4, 0)
    flush(ol)
    acc_rc0 = run_acc(ol, sx, sy, max_samples=2000)
    print(f"  acc with rom_rc=0: {acc_rc0:.2f}% (was {acc_no:.2f}%)")

    print("\n>>> Writing rom_down = 0xFFFFFFFF (32 words at off 128..159) ...")
    for w in range(128, 128 + 32):
        mmio.write(base_mvau1 + w*4, 0xFFFFFFFF)
    flush(ol)
    acc_d = run_acc(ol, sx, sy, max_samples=2000)
    print(f"  acc with rom_down=ALL-1s: {acc_d:.2f}%")

    print("\n>>> Writing rom_up = 0 (64 words at off 640..703) ...")
    for w in range(640, 640 + 64):
        mmio.write(base_mvau1 + w*4, 0)
    flush(ol)
    acc_u = run_acc(ol, sx, sy, max_samples=2000)
    print(f"  acc with rom_up=0: {acc_u:.2f}%")

    print(f"\nSummary:  no-touch={acc_no}  rc=0:{acc_rc0}  down=1:{acc_d}  up=0:{acc_u}")
    if abs(acc_d - acc_no) > 2 or abs(acc_u - acc_no) > 2 or abs(acc_rc0 - acc_no) > 2:
        print("✅ rom_* IS cfg-writable on MVAU1!")
    else:
        print("❌ rom_* writes have minimal effect — cfg path may not be reaching banks.")


if __name__ == "__main__":
    main()
