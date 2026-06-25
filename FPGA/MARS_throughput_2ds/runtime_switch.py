#!/usr/bin/env python3
# ===========================================================================
# [交接導向註解]
# runtime 切換流程（throughput build, 2 資料集）。流程：FPGA。
# ===========================================================================

"""
runtime_switch.py — Runtime dataset switching on FPGA
Switches between CIFAR-10 (backbone-only) and SVHN (adapter-on) modes
by writing thresholds via the adapter_cfg_hub AXI-Lite slave.

Usage:
  python3 runtime_switch.py --bitfile resizer.bit --mode cifar10
  python3 runtime_switch.py --bitfile resizer.bit --mode svhn
  python3 runtime_switch.py --bitfile resizer.bit --mode demo
"""

import argparse
import struct
import time
import os
import numpy as np

# ---------------------------------------------------------------------------
# Memory map constants (from adapter_cfg_hub.v)
# ---------------------------------------------------------------------------
# cfg_hub byte address = (unit_select << 13) | (word_addr << 2)
#
# Unit 0 (MVAU_hls_0 + classifier):
#   Low half (bit[12]=0): MVAU_hls_0 thresholds, word 0-63
#   High half (bit[12]=1): Classifier weights, word 0-1023
#     => byte offset = 0x1000 + (word << 2)
#
# Units 1-5 (MVAU1-5 adapters):
#   0          : adapter_enable
#   1152..1407 : thresh_mem[0..255]
#   1408..1663 : sign_mem[0..255]
#
# Unit 6 (FC1/MVAU_hls_6): word 0-511, 8-bit thresholds
# Unit 7 (FC2/MVAU_hls_7): word 0-511, 10-bit thresholds

CFG_ENABLE_WORD   = 0
CFG_THRESH_BASE   = 1152
CFG_SIGN_BASE     = 1408

MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def mvau_byte_addr(mvau_id, word_addr):
    """Compute byte offset within cfg_hub's 64 KB space."""
    return (mvau_id << 13) | (word_addr << 2)


class AdapterCfgHub:
    """Driver for the adapter_cfg_hub AXI-Lite slave."""

    def __init__(self, overlay):
        """Find the cfg_hub in the overlay and create MMIO handle."""
        from pynq import MMIO

        # The cfg_hub is exposed through the stitch IP's s_axilite_cfg port.
        # Look for it in the overlay's IP dict.
        cfg_addr = None
        cfg_range = None

        for name, info in overlay.ip_dict.items():
            # The address might be under the stitch IP's cfg port
            if 'cfg' in name.lower() or 'adapter' in name.lower():
                cfg_addr = info['phys_addr']
                cfg_range = info['addr_range']
                print(f"  Found cfg hub: {name} @ 0x{cfg_addr:08X} (range={cfg_range})")
                break

        if cfg_addr is None:
            # Try looking in the address map more broadly
            for name, info in overlay.ip_dict.items():
                phys = info.get('phys_addr', 0)
                rng = info.get('addr_range', 0)
                print(f"  IP: {name} @ 0x{phys:08X} range=0x{rng:X}")

            # Fallback: try to find via mem_dict
            if hasattr(overlay, 'mem_dict'):
                for name, info in overlay.mem_dict.items():
                    if 'cfg' in name.lower():
                        cfg_addr = info['phys_addr']
                        cfg_range = info['addr_range']
                        print(f"  Found in mem_dict: {name} @ 0x{cfg_addr:08X}")
                        break

        if cfg_addr is None:
            # Fallback to known address from Vivado address map
            cfg_addr = 0x43C00000
            cfg_range = 0x10000
            print(f"  Using fallback address: 0x{cfg_addr:08X} (range=0x{cfg_range:X})")

        self.mmio = MMIO(cfg_addr, cfg_range)
        self.base_addr = cfg_addr
        print(f"  MMIO ready: base=0x{cfg_addr:08X}, range=0x{cfg_range:X}")

    def write_word(self, mvau_id, word_addr, value):
        """Write a single 32-bit word to a specific MVAU's config space."""
        offset = mvau_byte_addr(mvau_id, word_addr)
        self.mmio.write(offset, value & 0xFFFFFFFF)

    def set_adapter_enable(self, mvau_id, enable):
        """Set adapter_enable for a specific MVAU (0=off, 1=on)."""
        self.write_word(mvau_id, CFG_ENABLE_WORD, 1 if enable else 0)

    def write_thresh_array(self, mvau_id, values):
        """Write threshold array for a specific MVAU."""
        n_ch = MVAU_OUT_CH[mvau_id]
        assert len(values) == n_ch, f"MVAU{mvau_id}: expected {n_ch} thresh, got {len(values)}"
        for i, val in enumerate(values):
            self.write_word(mvau_id, CFG_THRESH_BASE + i, val)

    def write_sign_array(self, mvau_id, values):
        """Write sign array for a specific MVAU."""
        n_ch = MVAU_OUT_CH[mvau_id]
        assert len(values) == n_ch, f"MVAU{mvau_id}: expected {n_ch} sign, got {len(values)}"
        for i, val in enumerate(values):
            self.write_word(mvau_id, CFG_SIGN_BASE + i, val)

    def write_mvau0_thresholds(self, values):
        """Write MVAU_hls_0 thresholds (unit 0, low half, 64 words)."""
        assert len(values) == 64, f"Expected 64, got {len(values)}"
        for i, val in enumerate(values):
            # Unit 0, word addr i → byte addr = (0 << 13) | (i << 2) = i*4
            self.mmio.write(i * 4, val & 0xFFFFFFFF)

    def write_classifier_weights(self, values):
        """Write classifier weights (unit 0, high half, 1024 words).
        Byte addr = 0x1000 + (word << 2)
        """
        assert len(values) == 1024, f"Expected 1024, got {len(values)}"
        for i, val in enumerate(values):
            self.mmio.write(0x1000 + i * 4, val & 0xFFFFFFFF)

    def write_fc1_thresholds(self, values):
        """Write FC1 thresholds (unit 6, 512 words).
        Byte addr = (6 << 13) | (word << 2) = 0xC000 + word*4
        """
        assert len(values) == 512, f"Expected 512, got {len(values)}"
        for i, val in enumerate(values):
            self.mmio.write(0xC000 + i * 4, val & 0xFFFFFFFF)

    def write_fc2_thresholds(self, values):
        """Write FC2 thresholds (unit 7, 512 words).
        Byte addr = (7 << 13) | (word << 2) = 0xE000 + word*4
        """
        assert len(values) == 512, f"Expected 512, got {len(values)}"
        for i, val in enumerate(values):
            self.mmio.write(0xE000 + i * 4, val & 0xFFFFFFFF)

    def load_bin_u32(self, path):
        """Load a binary file of little-endian uint32 values."""
        with open(path, "rb") as f:
            data = f.read()
        return list(struct.unpack(f"<{len(data)//4}I", data))


def switch_to_cifar10(hub):
    """Switch all layers to CIFAR-10 backbone-only mode."""
    print("\n=== Switching to CIFAR-10 mode (adapter OFF) ===")
    t0 = time.time()

    # MVAU_hls_0: first conv thresholds
    mvau0_th = hub.load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_cifar10.bin"))
    hub.write_mvau0_thresholds(mvau0_th)
    print(f"  MVAU_hls_0: {len(mvau0_th)} thresholds written")

    # MVAU1-5: adapter layers
    for mvau_id in range(1, 6):
        thresh_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_thresh_cifar10.bin")
        threshs = hub.load_bin_u32(thresh_file)
        hub.write_thresh_array(mvau_id, threshs)
        hub.set_adapter_enable(mvau_id, False)
        print(f"  MVAU{mvau_id}: {len(threshs)} thresholds written, adapter OFF")

    # FC1: thresholds
    fc1_th = hub.load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_cifar10.bin"))
    hub.write_fc1_thresholds(fc1_th)
    print(f"  FC1: {len(fc1_th)} thresholds written")

    # FC2: thresholds
    fc2_th = hub.load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_cifar10.bin"))
    hub.write_fc2_thresholds(fc2_th)
    print(f"  FC2: {len(fc2_th)} thresholds written")

    # Classifier: weights
    cls_w = hub.load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_cifar10.bin"))
    hub.write_classifier_weights(cls_w)
    print(f"  Classifier: {len(cls_w)} weights written")

    elapsed = (time.time() - t0) * 1000
    print(f"  Switch complete in {elapsed:.1f} ms")


def switch_to_svhn(hub):
    """Switch all layers to SVHN adapter mode."""
    print("\n=== Switching to SVHN mode (adapter ON) ===")
    t0 = time.time()

    # MVAU_hls_0: first conv thresholds
    mvau0_th = hub.load_bin_u32(os.path.join(SCRIPT_DIR, "mvau0_thresh_svhn.bin"))
    hub.write_mvau0_thresholds(mvau0_th)
    print(f"  MVAU_hls_0: {len(mvau0_th)} thresholds written")

    # MVAU1-5: adapter layers
    for mvau_id in range(1, 6):
        thresh_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_thresh_svhn.bin")
        threshs = hub.load_bin_u32(thresh_file)
        hub.write_thresh_array(mvau_id, threshs)

        sign_file = os.path.join(SCRIPT_DIR, f"mvau{mvau_id}_sign_svhn.bin")
        signs = hub.load_bin_u32(sign_file)
        hub.write_sign_array(mvau_id, signs)

        hub.set_adapter_enable(mvau_id, True)
        print(f"  MVAU{mvau_id}: {len(threshs)} threshs + {len(signs)} signs, adapter ON")

    # FC1: thresholds
    fc1_th = hub.load_bin_u32(os.path.join(SCRIPT_DIR, "fc1_thresh_svhn.bin"))
    hub.write_fc1_thresholds(fc1_th)
    print(f"  FC1: {len(fc1_th)} thresholds written")

    # FC2: thresholds
    fc2_th = hub.load_bin_u32(os.path.join(SCRIPT_DIR, "fc2_thresh_svhn.bin"))
    hub.write_fc2_thresholds(fc2_th)
    print(f"  FC2: {len(fc2_th)} thresholds written")

    # Classifier: weights
    cls_w = hub.load_bin_u32(os.path.join(SCRIPT_DIR, "cls_weights_svhn.bin"))
    hub.write_classifier_weights(cls_w)
    print(f"  Classifier: {len(cls_w)} weights written")

    elapsed = (time.time() - t0) * 1000
    print(f"  Switch complete in {elapsed:.1f} ms")


def run_inference(overlay, test_imgs, test_labels, batch_size=100):
    """Run inference and return accuracy."""
    from driver_base import FINNExampleOverlay
    from qonnx.core.datatype import DataType

    io_shape_dict = {
        "idt": [DataType['UINT8']],
        "odt": [DataType['UINT8']],
        "ishape_normal": [(1, 32, 32, 3)],
        "oshape_normal": [(1, 1)],
        "ishape_folded": [(1, 32, 32, 3, 1)],
        "oshape_folded": [(1, 1, 1)],
        "ishape_packed": [(1, 32, 32, 3, 1)],
        "oshape_packed": [(1, 1, 1)],
        "input_dma_name": ['idma0'],
        "output_dma_name": ['odma0'],
        "number_of_external_weights": 0,
        "num_inputs": 1,
        "num_outputs": 1,
    }

    driver = FINNExampleOverlay(
        bitfile_name="already_loaded",
        platform="zynq-iodma",
        io_shape_dict=io_shape_dict,
        batch_size=batch_size,
        fclk_mhz=100.0,
    )
    # Reuse the already-loaded overlay
    driver.ol = overlay

    total = test_imgs.shape[0]
    n_batches = total // batch_size
    ok = 0

    test_imgs_batched = test_imgs[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    test_labels_batched = test_labels[:n_batches * batch_size].reshape(n_batches, batch_size)

    for i in range(n_batches):
        ibuf = test_imgs_batched[i].reshape(driver.ibuf_packed_device[0].shape)
        driver.copy_input_data_to_device(ibuf)
        driver.execute_on_buffers()
        obuf = np.empty_like(driver.obuf_packed_device[0])
        driver.copy_output_data_from_device(obuf)
        preds = obuf.flatten()
        ok += np.sum(preds == test_labels_batched[i])

    acc = 100.0 * ok / (n_batches * batch_size)
    return acc


def load_cifar10(dataset_root="/tmp"):
    """Load CIFAR-10 test set."""
    try:
        from dataset_loading import cifar
        _, _, testx, testy, _, _ = cifar.load_cifar_data(dataset_root, download=True, one_hot=False)
        return testx, testy
    except ImportError:
        print("  WARNING: dataset_loading not available, trying torchvision...")
        import torchvision
        import torchvision.transforms as transforms
        ds = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True)
        testx = np.array(ds.data).astype(np.uint8)  # (10000, 32, 32, 3)
        testy = np.array(ds.targets).astype(np.int64)
        return testx, testy


def load_svhn(dataset_root="/tmp"):
    """Load SVHN test set."""
    try:
        import torchvision
        ds = torchvision.datasets.SVHN(root=dataset_root, split='test', download=True)
        testx = np.transpose(ds.data, (0, 2, 3, 1)).astype(np.uint8)  # NCHW -> NHWC
        testy = np.array(ds.labels).astype(np.int64)
        return testx, testy
    except ImportError:
        print("  ERROR: torchvision not available for SVHN loading")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Runtime dataset switching on FPGA")
    parser.add_argument("--bitfile", default="resizer.bit", help="Bitstream file")
    parser.add_argument("--mode", choices=["cifar10", "svhn", "demo", "switch_only"],
                        required=True, help="Mode: cifar10, svhn, demo (both), or switch_only")
    parser.add_argument("--batchsize", type=int, default=100, help="Inference batch size")
    parser.add_argument("--dataset_root", default="/tmp", help="Dataset download root")
    parser.add_argument("--cfg_addr", type=lambda x: int(x, 0), default=None,
                        help="Override cfg_hub base address (hex, e.g. 0x43C00000)")
    args = parser.parse_args()

    from pynq import Overlay

    print("Loading bitstream:", args.bitfile)
    ol = Overlay(args.bitfile)
    print("Bitstream loaded.")

    hub = AdapterCfgHub(ol)

    if args.mode == "switch_only":
        # Just demonstrate switching speed
        for _ in range(3):
            switch_to_cifar10(hub)
            switch_to_svhn(hub)
        return

    if args.mode in ("cifar10", "demo"):
        switch_to_cifar10(hub)
        print("\nLoading CIFAR-10 test set...")
        testx, testy = load_cifar10(args.dataset_root)
        if testx is not None:
            acc = run_inference(ol, testx, testy, args.batchsize)
            print(f"\n*** CIFAR-10 Accuracy: {acc:.2f}% ***")

    if args.mode in ("svhn", "demo"):
        switch_to_svhn(hub)
        print("\nLoading SVHN test set...")
        testx, testy = load_svhn(args.dataset_root)
        if testx is not None:
            acc = run_inference(ol, testx, testy, args.batchsize)
            print(f"\n*** SVHN Accuracy: {acc:.2f}% ***")


if __name__ == "__main__":
    main()
