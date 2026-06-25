# ===========================================================================
# [交接導向註解]
# ★ 5 資料集 runtime 切換主 driver。流程：FPGA(compact build)。
# 依資料集從 runtime_weights/<dataset>/ 載入 per-task 參數、經 cfg_hub 寫入後推論。
# ===========================================================================

"""
Runtime 3-dataset switcher for Pynq-Z2 PE=1 BNN.

On boot:
    overlay = Overlay("resizer_3ds_v3.bit")
    sw = RuntimeSwitcher(overlay)

Per-dataset switch (< 1ms):
    sw.switch("cifar10")  # or "svhn", "fashion"

Then run inference via overlay's DMA IO.

cfg_hub base default 0x40010000 (Phase-1+ Vivado allocation).

Address map (per `mvau_adapter_ip/adapter_cfg_hub/ip/src/adapter_cfg_hub.v`):
    byte_addr[15:13] = unit:
      000 (0x0000): MVAU_hls_0 (low half) + Classifier (high half)
      001..101: MVAU1..5 adapter
      110:    FC1 (MVAU_hls_6)
      111:    FC2 (MVAU_hls_7)

    Unit 0 layout (split by bit 12):
      0x0000..0x00FC: MVAU0 thresh, 64 u32 entries (low 11 bits used)
      0x1000..0x19FC: Classifier packed weights, 640 u32 entries (low 8 bits used).
                     Each byte holds 8 consecutive binary weights (LSB-first).
                     Bridge (cls_cfg_bridge) expands each into 8 AXI-Lite writes
                     to memstream s_axilite, populating rows 0..5119.

    MVAU1..5 layout (word offsets within each 8KB unit):
      0      : adapter_enable
      4      : rc[0..HIDDEN-1]
      128    : down[0..IN_CHUNKS*HIDDEN-1]
      640    : up[0..OUT_CH-1] (or 2*OUT_CH for MVAU5 64-bit-wide rows)
      1152   : thresh[0..OUT_CH-1]
      1408   : sign[0..OUT_CH-1]
      1664   : contrib_lut[0..255]

    FC1 layout (word offsets, byte addr 0xC000-0xDFFF):
      0..511: thresh[0..511] (low 8 bits used)

    FC2 layout (word offsets, byte addr 0xE000-0xFFFF):
      0..511: thresh[0..511] (low 10 bits used)
"""
import os, struct, mmap
import numpy as np

# cfg word offsets within a unit (per CLAUDE.md)
WORD = {
    "enable": 0,
    "rc": 4,
    "down": 128,
    "up": 640,
    "thresh": 1152,
    "sign": 1408,
    "contrib": 1664,
}

MVAU_CFG = {
    1: dict(IN_CH=64,  OUT_CH=64,  HIDDEN=16, UP_WORDS_PER_OC=1),
    2: dict(IN_CH=64,  OUT_CH=128, HIDDEN=16, UP_WORDS_PER_OC=1),
    3: dict(IN_CH=128, OUT_CH=128, HIDDEN=32, UP_WORDS_PER_OC=1),
    4: dict(IN_CH=128, OUT_CH=256, HIDDEN=32, UP_WORDS_PER_OC=1),
    5: dict(IN_CH=256, OUT_CH=256, HIDDEN=64, UP_WORDS_PER_OC=2),  # 64-bit up = 2 u32
}
SIMD = 32

CFG_BASE_DEFAULT = 0x40010000
CFG_SIZE = 0x10000  # 64 KB

def _load_bin(path, dtype=np.uint32):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=dtype).copy()

class RuntimeSwitcher:
    def __init__(self, cfg_base=CFG_BASE_DEFAULT, weights_root=None):
        if weights_root is None:
            weights_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "sw", "runtime_weights")
        self.weights_root = weights_root
        self.cfg_base = cfg_base
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(fd, CFG_SIZE, offset=cfg_base)
        os.close(fd)
        self.cfg = np.frombuffer(self.mm, dtype=np.uint32)
        # Build cfg blob per dataset upfront so switch is one bulk write
        self._cache = {}
        for ds in ("cifar10", "svhn", "fashion"):
            self._cache[ds] = self._build_blob(ds)

    def _unit_word(self, unit, word):
        """byte address = (unit << 13) + (word << 2), divide by 4 for u32 index"""
        return ((unit << 13) >> 2) + word

    def _build_blob(self, dataset):
        """Pre-compute all (u32_index, value) writes for one dataset."""
        ds_dir = os.path.join(self.weights_root, dataset)
        writes = []
        adapter_on = (dataset != "cifar10")

        # === MVAU0 thresh (unit 0 low, byte 0x0000-0x00FC) ===
        # 64 u32 entries, addr in bytes 0..252 (word 0..63)
        # Map: unit=0 byte_addr=word*4 ⇒ aw[15:13]=0, aw[12]=0, aw[7:2]=word
        # u32 index in mmap = (byte_addr) / 4
        mvau0_path = f"{ds_dir}/mvau0_thresh.bin"
        if os.path.exists(mvau0_path):
            mvau0_t = _load_bin(mvau0_path)
            for i, v in enumerate(mvau0_t):
                # Byte addr = i*4 in unit 0 low half (bit12=0)
                # u32 idx = (0 + i*4)/4 = i
                writes.append((i, int(v) & 0xFFFFFFFF))

        # === Classifier packed weights (unit 0 high, byte 0x1000-0x19FC) ===
        # cls_weight.bin from gen_3ds_cfg.py is 640 packed bytes (8 weight bits/byte,
        # LSB-first). Each byte goes to one cls_cfg pulse; the on-FPGA bridge
        # (cls_cfg_bridge) expands it into 8 AXI-Lite memstream writes.
        # Memory map: unit 0 high half starts at byte 0x1000 → u32 idx 1024.
        # cls_cfg_waddr = aw[11:2] (10 bits, 0..1023). Use slots 0..639.
        cls_path = f"{ds_dir}/cls_weight.bin"
        if os.path.exists(cls_path):
            cls_packed = _load_bin(cls_path, dtype=np.uint8)
            n = min(len(cls_packed), 640)
            for i in range(n):
                writes.append((1024 + i, int(cls_packed[i]) & 0xFF))

        # === MVAU1..5 (unit 1..5) ===
        for n in [1, 2, 3, 4, 5]:
            cfg = MVAU_CFG[n]
            unit = n
            # enable
            writes.append((self._unit_word(unit, WORD["enable"]), 1 if adapter_on else 0))
            # rc
            rc = _load_bin(f"{ds_dir}/mvau{n}_rc.bin")
            for i, v in enumerate(rc):
                writes.append((self._unit_word(unit, WORD["rc"] + i), int(v) & 0xFFFFFFFF))
            # down
            down = _load_bin(f"{ds_dir}/mvau{n}_down.bin")
            for i, v in enumerate(down):
                writes.append((self._unit_word(unit, WORD["down"] + i), int(v) & 0xFFFFFFFF))
            # up
            up = _load_bin(f"{ds_dir}/mvau{n}_up.bin")
            for i, v in enumerate(up):
                writes.append((self._unit_word(unit, WORD["up"] + i), int(v) & 0xFFFFFFFF))
            # thresh
            thr = _load_bin(f"{ds_dir}/mvau{n}_thresh.bin")
            for i, v in enumerate(thr):
                writes.append((self._unit_word(unit, WORD["thresh"] + i), int(v) & 0xFFFFFFFF))
            # sign
            sgn = _load_bin(f"{ds_dir}/mvau{n}_sign.bin")
            for i, v in enumerate(sgn):
                writes.append((self._unit_word(unit, WORD["sign"] + i), int(v) & 0xFFFFFFFF))
            # contrib
            ctb = _load_bin(f"{ds_dir}/mvau{n}_contrib.bin")
            for i, v in enumerate(ctb):
                writes.append((self._unit_word(unit, WORD["contrib"] + i), int(v) & 0xFFFFFFFF))

        # === FC1 thresh (unit 6, byte 0xC000-0xDFFC, 512 u32) ===
        fc1_path = f"{ds_dir}/fc1_thresh.bin"
        if os.path.exists(fc1_path):
            fc1 = _load_bin(fc1_path)
            for i, v in enumerate(fc1):
                writes.append((self._unit_word(6, i), int(v) & 0xFFFFFFFF))

        # === FC2 thresh (unit 7, byte 0xE000-0xFFFC, 512 u32) ===
        fc2_path = f"{ds_dir}/fc2_thresh.bin"
        if os.path.exists(fc2_path):
            fc2 = _load_bin(fc2_path)
            for i, v in enumerate(fc2):
                writes.append((self._unit_word(7, i), int(v) & 0xFFFFFFFF))

        # Convert to sorted numpy arrays for fast bulk write
        writes.sort(key=lambda x: x[0])
        idxs = np.array([w[0] for w in writes], dtype=np.uint32)
        vals = np.array([w[1] for w in writes], dtype=np.uint32)
        return (idxs, vals)

    def switch(self, dataset):
        """Apply per-dataset cfg. Returns elapsed ms."""
        import time
        t0 = time.time()
        idxs, vals = self._cache[dataset]
        self.cfg[idxs] = vals
        return (time.time() - t0) * 1000.0


if __name__ == "__main__":
    sw = RuntimeSwitcher()
    for ds in ("cifar10", "svhn", "fashion"):
        ms = sw.switch(ds)
        print(f"Switched to {ds} in {ms:.2f} ms ({len(sw._cache[ds][0])} writes)")
