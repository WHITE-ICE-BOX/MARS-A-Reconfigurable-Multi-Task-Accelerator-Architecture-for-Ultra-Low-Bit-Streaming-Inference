#!/usr/bin/env python3
"""
Staged board test with defensive guards against PYNQ state issues.

Stages (cumulative):
  0: load bitstream ONLY, no inference, no cfg writes
  1: load + inference with baked CIFAR thresh (no cfg writes)
  2: load + write SVHN cfg + inference
  3: load + write Fashion cfg + inference
  4: load + write CIFAR cfg + inference (verify cfg path goes both ways)
  all: 1..4 in sequence

Run: python3 board_test_staged.py --stage 0|1|2|3|4|all (default: 1)
"""
import argparse, os, sys, time, traceback, gc
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/xilinx/runtime_3ds_pe1/data"


def wait_idle(idma, odma, timeout=5.0):
    """Wait for both DMAs idle (bit 2 of status = idle)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        si = idma.read(0x04)
        so = odma.read(0x04)
        if (si & 0x2) and (so & 0x2):
            return True
        time.sleep(0.01)
    return False


def run_inference(ol, test_x, test_y, batch_size=100, label="", batches=None):
    from pynq import allocate
    idma = ol.idma0
    odma = ol.odma0
    total = test_x.shape[0]
    n_batches = total // batch_size
    if batches: n_batches = min(n_batches, batches)
    total = n_batches * batch_size
    correct = 0
    ibuf = allocate(shape=(batch_size, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
    obuf = allocate(shape=(batch_size, 1, 1), dtype=np.uint8, cacheable=True)
    t0 = time.time()
    try:
        for b in range(n_batches):
            s, e = b * batch_size, (b + 1) * batch_size
            np.copyto(ibuf, test_x[s:e].astype(np.uint8).reshape(batch_size, 32, 32, 3, 1))
            ibuf.flush()
            odma.write(0x10, obuf.device_address); odma.write(0x1C, batch_size); odma.write(0x00, 1)
            idma.write(0x10, ibuf.device_address); idma.write(0x1C, batch_size); idma.write(0x00, 1)
            timeout = time.time() + 30
            while odma.read(0x00) & 0x2 == 0:
                if time.time() > timeout:
                    print(f"  [{label}] TIMEOUT at batch {b}")
                    return None, 0, 0, 0
            obuf.invalidate()
            preds = np.array(obuf).flatten().astype(np.int64)
            correct += int(np.sum(preds == test_y[s:e]))
    finally:
        ibuf.freebuffer(); obuf.freebuffer()
    elapsed = time.time() - t0
    return 100.0 * correct / total, correct, total, total / elapsed if elapsed > 0 else 0


def stage_0(ol):
    print("=== Stage 0: bitstream loaded, no inference, no cfg ===")
    idma = ol.idma0; odma = ol.odma0
    si = idma.read(0x04); so = odma.read(0x04)
    print(f"  idma status: 0x{si:08X} (idle bit={bool(si&0x2)})")
    print(f"  odma status: 0x{so:08X} (idle bit={bool(so&0x2)})")
    print("  Bitstream healthy.")


def stage_inference(ol, dataset, label, batches=10):
    x = np.load(f"{DATA_DIR}/{dataset}_test_x.npy")
    y = np.load(f"{DATA_DIR}/{dataset}_test_y.npy")
    acc, c, n, fps = run_inference(ol, x, y, batches=batches, label=label)
    if acc is None:
        print(f"  [{label}] FAIL: hung at inference"); return False
    print(f"  [{label}] {dataset}: {acc:.2f}% ({c}/{n}), {fps:.1f} fps")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="1")
    parser.add_argument("--bitfile", default="resizer_3ds_v3.bit")
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()

    stage = args.stage
    from pynq import Overlay
    from pynq.ps import Clocks

    bitfile = os.path.join(SCRIPT_DIR, args.bitfile)
    if not os.path.exists(bitfile):
        print(f"FATAL: {bitfile} not found")
        return
    print(f"Loading {bitfile}")
    ol = Overlay(bitfile)
    Clocks.fclk0_mhz = 100.0
    print("Overlay loaded.")

    if stage == "0":
        stage_0(ol)
        return

    if stage == "1":
        print("\n=== Stage 1: CIFAR baked, no cfg writes ===")
        stage_inference(ol, "cifar10", "stage1_baked", args.batches)
        return

    # Stages 2-4: cfg writes
    sys.path.insert(0, SCRIPT_DIR)
    from runtime_3ds import RuntimeSwitcher
    print("\nInit RuntimeSwitcher...")
    sw = RuntimeSwitcher(weights_root=os.path.join(SCRIPT_DIR, "runtime_weights"))
    print(f"  cfg_base = 0x{sw.cfg_base:08X}")
    gc.collect()

    do = [stage] if stage != "all" else ["2", "3", "4"]

    for s in do:
        if s == "2":
            print("\n=== Stage 2: SVHN ===")
            try: ms = sw.switch("svhn"); print(f"  switched in {ms:.2f} ms")
            except Exception as e: print(f"  switch FAIL: {e}"); continue
            stage_inference(ol, "svhn", "stage2_svhn", args.batches)
        if s == "3":
            print("\n=== Stage 3: Fashion ===")
            try: ms = sw.switch("fashion"); print(f"  switched in {ms:.2f} ms")
            except Exception as e: print(f"  switch FAIL: {e}"); continue
            stage_inference(ol, "fashion", "stage3_fashion", args.batches)
        if s == "4":
            print("\n=== Stage 4: back to CIFAR ===")
            try: ms = sw.switch("cifar10"); print(f"  switched in {ms:.2f} ms")
            except Exception as e: print(f"  switch FAIL: {e}"); continue
            stage_inference(ol, "cifar10", "stage4_cifar_again", args.batches)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
