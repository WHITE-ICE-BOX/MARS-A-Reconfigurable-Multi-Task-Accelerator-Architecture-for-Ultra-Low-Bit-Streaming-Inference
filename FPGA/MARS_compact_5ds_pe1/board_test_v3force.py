#!/usr/bin/env python3
"""
On-board 3-dataset accuracy verification for PE=1 BNN bitstream.
Bitstream: resizer_3ds.bit (PE=1 backbone + 5 cfg-writable adapter wrappers).
cfg_hub base: 0x40010000.

Note: MVAU0/FC1/FC2/classifier remain STATIC (baked CIFAR thresholds in Phase 1).
Expected: CIFAR ~74%, SVHN/Fashion degraded vs paper but should show adapter effect.
"""
import numpy as np
import os, time, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/xilinx/runtime_3ds_pe1/data"  # uploaded test data on board

DATASETS = ["cifar10", "svhn", "fashion"]


def run_inference(ol, test_x, test_y, batch_size=100, max_samples=None):
    from pynq import allocate
    idma = ol.idma0
    odma = ol.odma0
    total = test_x.shape[0]
    if max_samples: total = min(total, max_samples)
    n_batches = total // batch_size
    total = n_batches * batch_size
    correct = 0
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    t0 = time.time()
    for b in range(n_batches):
        s, e = b * batch_size, (b + 1) * batch_size
        np.copyto(ibuf, test_x[s:e].astype(np.uint8).reshape(batch_size, 32, 32, 3, 1))
        ibuf.flush()
        odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
        idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
        while odma.read(0x00) & 0x2 == 0: pass
        obuf.invalidate()
        preds = np.array(obuf).flatten().astype(np.int64)
        correct += int(np.sum(preds == test_y[s:e]))
    elapsed = time.time() - t0
    ibuf.freebuffer(); obuf.freebuffer()
    return 100.0 * correct / total, correct, total, elapsed


def main():
    from pynq import Overlay
    from pynq.ps import Clocks
    # Allow runtime_3ds.py import from same dir
    sys.path.insert(0, SCRIPT_DIR)
    from runtime_3ds import RuntimeSwitcher

    # Prefer v2 (Phase 2 cfg-writable MVAU0/FC1/FC2) if available; fall back to v1 (Phase 1).
    bitfile_v2 = os.path.join(SCRIPT_DIR, "resizer_3ds_v2.bit")
    bitfile_v1 = os.path.join(SCRIPT_DIR, "resizer_3ds.bit")
    bitfile = os.path.join(SCRIPT_DIR, "resizer_3ds_v3.bit")  # FORCE v3
    print(f"Loading {bitfile} ...")
    ol = Overlay(bitfile)
    Clocks.fclk0_mhz = 100.0
    print("Overlay loaded.")

    print("Initializing RuntimeSwitcher ...")
    sw = RuntimeSwitcher(weights_root=os.path.join(SCRIPT_DIR, "runtime_weights"))
    print(f"  cfg_base = 0x{sw.cfg_base:08X}")

    results = {}
    for ds in DATASETS:
        print(f"\n=== {ds.upper()} ===")
        test_x = np.load(f"{DATA_DIR}/{ds}_test_x.npy")
        test_y = np.load(f"{DATA_DIR}/{ds}_test_y.npy")
        ms = sw.switch(ds)
        print(f"  cfg switch: {ms:.2f} ms ({len(sw._cache[ds][0])} writes)")
        # 1 dummy batch to drain pipe
        dummy_x = test_x[:100]; dummy_y = test_y[:100]
        run_inference(ol, dummy_x, dummy_y, batch_size=100)
        # Real run
        acc, c, n, sec = run_inference(ol, test_x, test_y, batch_size=100)
        fps = n / sec if sec > 0 else 0
        print(f"  Accuracy: {acc:.2f}% ({c}/{n}), FPS = {fps:.1f}")
        results[ds] = (acc, c, n, fps)

    print("\n" + "=" * 60)
    print(f"{'Dataset':<10} {'Accuracy':>10} {'Correct':>10} {'FPS':>10}")
    print("-" * 60)
    for ds, (acc, c, n, fps) in results.items():
        print(f"{ds:<10} {acc:>9.2f}% {c:>5d}/{n:<5d} {fps:>9.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
