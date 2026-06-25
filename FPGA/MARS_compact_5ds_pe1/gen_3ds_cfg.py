# ===========================================================================
# [交接導向註解]
# ★ 把 PyTorch 參數打包成板上 runtime_weights/<dataset>/*.bin。流程：FPGA。
# 輸出 little-endian u32 stream，可直接 mmap 寫入 cfg_hub（位址 = (unit<<13)+(word<<2)）。
# 這就是『板上把參數包成 .bin』的腳本；產生 runtime 切換用的 per-task 參數。
# ===========================================================================

"""
Generate per-dataset cfg write payloads for 3 datasets:
  - CIFAR-10: adapter OFF, use Phase-0 baked thresholds (no cfg write needed for MVAU1-5; just adapter_enable=0)
  - SVHN: SVHN_m1_rc adapter weights + BN-fused thresh
  - FashionMNIST: Fashion_m1_rc adapter weights + BN-fused thresh

Output: sw/runtime_weights/{cifar10,svhn,fashion}/mvau{1..5}_{rc,down,up,thresh,sign,contrib_lut}.bin
Each .bin is little-endian u32 stream ready to write to cfg_hub via mmap.

cfg_hub byte address = (unit << 13) + (word_addr << 2)
  unit 1..5 -> MVAU1..5
  word 0: adapter_enable
  word 4..4+HIDDEN-1: rom_rc
  word 128..128+IN_CHUNKS*HIDDEN-1: rom_down
  word 640..640+OUT_CH-1 (or 2*OUT_CH for MVAU5): rom_up
  word 1152..1152+OUT_CH-1: thresh_rom
  word 1408..1408+OUT_CH-1: sign_rom
  word 1664..1664+255: contrib_lut
"""
import os, sys, struct, json
sys.path.insert(0, "/home/barkie1/finn_brevitis/brevitas/src")
import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np

ROOT = "/home/barkie1/mvau_pipeline_runtime_3ds_pe1"
CKPT_DIR = f"{ROOT}/sw/checkpoints"
OUT_DIR = f"{ROOT}/sw/runtime_weights"

# Consistent trio: SVHN/Fashion adapters trained with the SAME frozen
# cifar10_1w1a backbone (verified: backbone conv sign agreement = 1.000).
CKPTS = {
    "cifar10": f"{CKPT_DIR}/cifar10_1w1a_BACKBONE.tar",          # adapter OFF
    "svhn":    f"{CKPT_DIR}/SVHN_v7_m1rc_73.72_FROZENbb.tar",
    "fashion": f"{CKPT_DIR}/Fashion_v9_m1rc_80.42_FROZENbb.tar",
    "stl10":   f"{CKPT_DIR}/STL10_v9_m1rc_67.46_FROZENbb.tar",
    "cinic10": f"{CKPT_DIR}/CINIC10_v9_m1rc_64.80_FROZENbb.tar",
}
HAS_ADAPTER = {"cifar10": False, "svhn": True, "fashion": True, "stl10": True, "cinic10": True}

# MVAU configs: IN_CH, OUT_CH, SIMD, HIDDEN, UP_WIDTH, K (kernel total = 3*3=9), MAC_TOTAL = K*IN_CH
MVAU_CFG = {
    1: dict(IN_CH=64,  OUT_CH=64,  HIDDEN=16, UP_WIDTH=32),
    2: dict(IN_CH=64,  OUT_CH=128, HIDDEN=16, UP_WIDTH=32),
    3: dict(IN_CH=128, OUT_CH=128, HIDDEN=32, UP_WIDTH=32),
    4: dict(IN_CH=128, OUT_CH=256, HIDDEN=32, UP_WIDTH=32),
    5: dict(IN_CH=256, OUT_CH=256, HIDDEN=64, UP_WIDTH=64),
}

SIMD = 32

def binarize(w):
    """bit-packed: bit i = (w[..., i] >= 0)."""
    return (w >= 0).to(torch.uint8)

def pack_bits(bits, nbits):
    """Pack last-dim bits LSB-first into integer of nbits."""
    val = 0
    for i in range(nbits):
        if bits[i]: val |= (1 << i)
    return val

def quantize_bias_to_int16(bias_t, IN_CH):
    """Adapter down bias → 16-bit signed (v1 convention: floor(bias/2 - IN_CH/2)).

    Derivation: PyT forward computes y = 2*pop - IN_CH + bias. RTL accumulates
    hidden_acc = rc + pop. For sign(y)==sign(hidden_acc) we need
    rc = floor((bias - IN_CH) / 2) = floor(bias/2 - IN_CH/2).

    The previous formula `round(bias) - IN_CH//2` was off by factor 2 on bias
    (and used round vs floor), causing 1-2 unit error at boundary, mostly
    benign but can flip sign-quantized hidden_act bits.
    """
    b = bias_t.to(torch.float64).numpy()
    rc = np.floor(b / 2.0 - IN_CH / 2.0).astype(np.int32)
    return np.clip(rc, -32768, 32767).astype(np.int16)

def gen_mvau_adapter(sd, mvau_idx):
    """Extract adapter weights for MVAU N, return dict of arrays for cfg packing."""
    cfg = MVAU_CFG[mvau_idx]
    IN_CH, OUT_CH = cfg["IN_CH"], cfg["OUT_CH"]
    HIDDEN, UP_WIDTH = cfg["HIDDEN"], cfg["UP_WIDTH"]
    IN_CHUNKS = IN_CH // SIMD

    down_w = sd[f"adapters.{mvau_idx}.down.weight"]  # (HIDDEN, IN_CH, 1, 1)
    down_b = sd[f"adapters.{mvau_idx}.down.bias"]    # (HIDDEN,)
    up_w   = sd[f"adapters.{mvau_idx}.up.weight"]    # (OUT_CH, HIDDEN, 1, 1)
    alpha  = float(sd[f"adapters.{mvau_idx}.alpha"])

    # rom_rc: per-hidden 16-bit signed init for hidden_acc
    rc = quantize_bias_to_int16(down_b, IN_CH)

    # rom_down: IN_CHUNKS wide rows, each HIDDEN*SIMD bits
    rom_down = np.zeros((IN_CHUNKS, HIDDEN), dtype=np.uint32)
    for c in range(IN_CHUNKS):
        for h in range(HIDDEN):
            bits = binarize(down_w[h, c*SIMD:(c+1)*SIMD, 0, 0])
            rom_down[c, h] = pack_bits(bits.tolist(), SIMD)

    # rom_up: per-OC UP_WIDTH bits
    rom_up = np.zeros((OUT_CH,), dtype=np.uint64 if UP_WIDTH == 64 else np.uint32)
    for oc in range(OUT_CH):
        bits = binarize(up_w[oc, :, 0, 0])
        # pad to UP_WIDTH bits with zeros (HIDDEN ≤ UP_WIDTH)
        padded = list(bits) + [0]*(UP_WIDTH - HIDDEN)
        rom_up[oc] = pack_bits(padded, UP_WIDTH)

    # contrib_lut: contrib[p] = (p - OFFSET) * alpha_q8
    # OFFSET accounts for adp_pop's xnor-padding bias:
    #   - rom_up is UP_WIDTH bits, but only HIDDEN bits are real weight; upper (UP_WIDTH-HIDDEN) bits are 0
    #   - hidden_act is also HIDDEN bits zero-padded to UP_WIDTH
    #   - xnor of two zero-padded sections always = 1 → adds (UP_WIDTH-HIDDEN) to popcount
    #   - real popcount = adp_pop - (UP_WIDTH - HIDDEN); bipolar dot = 2*real_pop - HIDDEN
    #   - contrib_q8 = alpha_q8 * (2*real_pop - HIDDEN) / 2 = alpha_q8 * (real_pop - HIDDEN/2)
    #                = alpha_q8 * (adp_pop - (UP_WIDTH - HIDDEN + HIDDEN/2))
    #                = alpha_q8 * (adp_pop - (UP_WIDTH - HIDDEN/2))
    # So OFFSET = UP_WIDTH - HIDDEN/2:
    #   MVAU1/2: UP_WIDTH=32 HIDDEN=16 → OFFSET=24
    #   MVAU3/4: UP_WIDTH=32 HIDDEN=32 → OFFSET=16
    #   MVAU5:   UP_WIDTH=64 HIDDEN=64 → OFFSET=32
    alpha_q8 = int(round(alpha * 256))
    OFFSET = UP_WIDTH - HIDDEN // 2
    contrib = np.zeros((256,), dtype=np.int32)
    for p in range(256):
        contrib[p] = (p - OFFSET) * alpha_q8
    return dict(rc=rc, rom_down=rom_down, rom_up=rom_up, contrib=contrib, alpha=alpha,
                IN_CH=IN_CH, OUT_CH=OUT_CH, HIDDEN=HIDDEN, IN_CHUNKS=IN_CHUNKS, UP_WIDTH=UP_WIDTH)

def gen_mvau_thresh_sign(sd, mvau_idx):
    """Extract per-OC BN-fused threshold (Q8 scale) + sign for Stream_Adder_Threshold.
    The conv2..conv6 BN params absorb the popcount mean. Computing BN-fused thresh:
        t_signed = (- beta + ...) / gamma ... (signed thresh of bipolar(2p-N))
        t_pop = (t_signed + IN_CH * K_PIX) / 2  (convert bipolar threshold to popcount threshold)
        sign = (gamma < 0)
        thresh_q8 = round(t_pop * 256)
        if sign: thresh_q8 = K * 256 - thresh_q8 (flip for negative gamma)
    """
    # BN module index per MVAU: conv_features layout is
    #   0: input QuantIdentity
    #   1..3 per conv layer: QuantConv2d, BatchNorm2d, QuantIdentity(act)
    #     and possibly MaxPool2d (after layers 1,3 in CNV_OUT_CH_POOL)
    # MVAU N corresponds to layer N (conv index N). After conv, there's BN.
    # For 1W1A CNV the BN params are at conv_features[bn_idx]
    # CNV_OUT_CH_POOL = [(64,F),(64,T),(128,F),(128,T),(256,F),(256,F)]
    # Per CNV.py iteration:
    #   i=0 conv_features[1]=conv, [2]=BN, [3]=act
    #   i=1 conv_features[4]=conv, [5]=BN, [6]=act, [7]=maxpool
    #   i=2 conv_features[8]=conv, [9]=BN, [10]=act
    #   i=3 conv_features[11]=conv, [12]=BN, [13]=act, [14]=maxpool
    #   i=4 conv_features[15]=conv, [16]=BN, [17]=act
    #   i=5 conv_features[18]=conv, [19]=BN, [20]=act
    # So MVAU N (conv index N, 1..5) BN is at conv_features[1 + N*3 + (skip pools)]
    # Actually counting properly with pools:
    #   N=0: conv idx 1, BN idx 2, act idx 3 (no pool inserted yet)
    #   N=1: conv idx 4, BN idx 5, act idx 6, pool 7
    #   N=2: conv idx 8, BN idx 9, act idx 10
    #   N=3: conv idx 11, BN idx 12, act idx 13, pool 14
    #   N=4: conv idx 15, BN idx 16, act idx 17
    #   N=5: conv idx 18, BN idx 19, act idx 20
    BN_IDX = {1: 5, 2: 9, 3: 12, 4: 16, 5: 19}
    cfg = MVAU_CFG[mvau_idx]
    OUT_CH = cfg["OUT_CH"]
    IN_CH = cfg["IN_CH"]
    K_PIX = 9  # 3x3 kernel
    K_TOTAL = K_PIX * IN_CH  # MAC count = max popcount

    bn_idx = BN_IDX[mvau_idx]
    rm = sd[f"conv_features.{bn_idx}.running_mean"].numpy()
    rv = sd[f"conv_features.{bn_idx}.running_var"].numpy()
    gm = sd[f"conv_features.{bn_idx}.weight"].numpy()
    bt = sd[f"conv_features.{bn_idx}.bias"].numpy()
    eps = 1e-4
    # BN: y = gamma * (x - rm) / sqrt(rv + eps) + beta
    # Activation: sign(y) → bit = (y >= 0)
    # y >= 0  →  gamma*(x - rm) >= -beta*sqrt(rv+eps)
    #         →  (x - rm) >= -beta*sqrt/gamma   (if gamma>0)
    #         →  x >= rm - beta*sqrt/gamma
    sqrt_rv = np.sqrt(rv + eps)
    sign = (gm < 0).astype(np.uint8)
    # Bipolar conv output: y_bipolar = 2 * popcount - K_TOTAL (range [-K, +K])
    # Threshold y_signed = -beta * sqrt / gamma, equiv pop threshold:
    # 2*pop - K = thr_signed → pop = (thr_signed + K) / 2
    # t_bip = bipolar accumulator threshold = rm - bt*sqrt/gm
    t_bip = rm - bt * sqrt_rv / gm
    # Convert bipolar→popcount: 2*pop - K >= t_bip → pop >= (K + t_bip)/2
    thr_pop = (K_TOTAL + t_bip) / 2
    # Q8 scale (so that comparison with (mvau_pop << 8) works)
    thr_q8 = np.round(thr_pop * 256).astype(np.int64)
    # For sign=1 (gamma<0): flip comparison: thresh_q8 = K*256 - thresh_q8
    for oc in range(OUT_CH):
        if sign[oc]:
            thr_q8[oc] = K_TOTAL * 256 - thr_q8[oc]
    # Clip to int32 range
    thr_q8 = np.clip(thr_q8, -(2**31), 2**31-1).astype(np.int32)
    return thr_q8, sign

def gen_mvau0_thresh(sd):
    """MVAU0 (first conv): 64 thresholds, 11-bit signed, BN params at conv_features.2.
    Formula matches FINN's: bit = (accum >= T) with input scale = 128 (int8 signed).
    For gamma<0: T = -ceil(T_pyth * 128), bit = (accum >= T) under flipped weight semantics."""
    gamma = sd["conv_features.2.weight"].numpy()
    beta  = sd["conv_features.2.bias"].numpy()
    mu    = sd["conv_features.2.running_mean"].numpy()
    var   = sd["conv_features.2.running_var"].numpy()
    sigma = np.sqrt(var + 1e-5)
    T_pyth = mu - beta * sigma / gamma
    SCALE = 128
    T_acc = np.where(gamma >= 0, T_pyth * SCALE, -T_pyth * SCALE)
    T = np.ceil(T_acc).astype(np.int32)
    # Clip to 11-bit signed range
    T = np.clip(T, -1024, 1023).astype(np.int32)
    # Store as 32-bit (low 11 bits used in RTL)
    return T


def gen_fc_thresh(sd, fc_idx, IN_DIM, OUT_DIM, data_width):
    """FC1 (linear_features.0, BN at .1, IN=256, OUT=512, 8-bit thresh)
       FC2 (linear_features.3, BN at .4, IN=512, OUT=512, 10-bit thresh)
    Formula: T_pop = ceil((IN_DIM ± T_pyth) / 2), bit = (popcount >= T_pop)."""
    BN_IDX = {1: 1, 2: 4}
    bn_idx = BN_IDX[fc_idx]
    gamma = sd[f"linear_features.{bn_idx}.weight"].numpy()
    beta  = sd[f"linear_features.{bn_idx}.bias"].numpy()
    mu    = sd[f"linear_features.{bn_idx}.running_mean"].numpy()
    var   = sd[f"linear_features.{bn_idx}.running_var"].numpy()
    sigma = np.sqrt(var + 1e-5)
    T_pyth = mu - beta * sigma / gamma
    T_pop = np.where(gamma >= 0, 0.5*(IN_DIM + T_pyth), 0.5*(IN_DIM - T_pyth))
    T = np.ceil(T_pop).astype(np.int64)
    # FINN clamps out-of-range to 0 — implement clipping
    max_v = (1 << data_width) - 1
    T = np.where((T < 0) | (T > max_v), 0, T).astype(np.uint32)
    return T


def gen_classifier_weight(sd):
    """Classifier (MVAU_hls_8, 512->10, no activation).
    FINN memstream config: DEPTH=5120 WIDTH=8, one binary weight per row stored
    in the LSB. Memblock layout per FINN MVAU PE=1 SIMD=1:
        row r = nf * SF + sf, nf in [0,10), sf in [0,512), total 5120 rows.

    Phase-3 packing for cls_cfg_bridge: pack 8 consecutive row bits into one
    byte (LSB-first within byte). Returns 640 uint8 values. The bridge expands
    each byte into 8 sequential AXI-Lite memstream writes.
    """
    W = sd["linear_features.6.weight"].numpy()  # (10, 512)
    bits = (W >= 0).astype(np.uint8)
    rows = np.zeros(5120, dtype=np.uint8)
    for nf in range(10):
        for sf in range(512):
            rows[nf * 512 + sf] = int(bits[nf, sf])
    # Pack 8 row bits per byte, LSB-first
    packed = np.zeros(640, dtype=np.uint8)
    for i in range(640):
        b = 0
        for k in range(8):
            b |= (int(rows[i * 8 + k]) & 1) << k
        packed[i] = b
    return packed  # 640 packed bytes


def write_dataset_cfg(name, sd):
    out = f"{OUT_DIR}/{name}"
    os.makedirs(out, exist_ok=True)
    has_adapter = HAS_ADAPTER[name]
    # MVAU0 thresh
    mvau0_thresh = gen_mvau0_thresh(sd)
    with open(f"{out}/mvau0_thresh.bin", "wb") as f:
        f.write(mvau0_thresh.astype(np.uint32).tobytes())
    # FC1 thresh
    fc1_thresh = gen_fc_thresh(sd, fc_idx=1, IN_DIM=256, OUT_DIM=512, data_width=8)
    with open(f"{out}/fc1_thresh.bin", "wb") as f:
        f.write(fc1_thresh.astype(np.uint32).tobytes())
    # FC2 thresh
    fc2_thresh = gen_fc_thresh(sd, fc_idx=2, IN_DIM=512, OUT_DIM=512, data_width=10)
    with open(f"{out}/fc2_thresh.bin", "wb") as f:
        f.write(fc2_thresh.astype(np.uint32).tobytes())
    # Classifier weight
    cls_w = gen_classifier_weight(sd)
    # Write as bytes (each is 0 or 1, low bit)
    with open(f"{out}/cls_weight.bin", "wb") as f:
        f.write(cls_w.astype(np.uint8).tobytes())
    print(f"  MVAU0/FC1/FC2 thresh + cls weight written")

    for mvau_idx in [1, 2, 3, 4, 5]:
        if has_adapter:
            adap = gen_mvau_adapter(sd, mvau_idx)
        else:
            # CIFAR: no adapter data; produce zeros for rc/down/up/contrib
            cfg = MVAU_CFG[mvau_idx]
            adap = dict(
                rc=np.zeros(cfg["HIDDEN"], dtype=np.int16),
                rom_down=np.zeros((cfg["IN_CH"]//SIMD, cfg["HIDDEN"]), dtype=np.uint32),
                rom_up=np.zeros(cfg["OUT_CH"], dtype=np.uint64 if cfg["UP_WIDTH"]==64 else np.uint32),
                contrib=np.zeros(256, dtype=np.int32),
                alpha=0.0,
                IN_CH=cfg["IN_CH"], OUT_CH=cfg["OUT_CH"], HIDDEN=cfg["HIDDEN"],
                IN_CHUNKS=cfg["IN_CH"]//SIMD, UP_WIDTH=cfg["UP_WIDTH"])
        thr, sgn = gen_mvau_thresh_sign(sd, mvau_idx)
        # Write each section as u32 little-endian
        # rc: 16 lo bits, hi bits 0
        rc_u32 = adap["rc"].astype(np.int32).astype(np.uint32) & 0xFFFF
        with open(f"{out}/mvau{mvau_idx}_rc.bin","wb") as f:
            f.write(rc_u32.tobytes())
        # down: IN_CHUNKS*HIDDEN u32, layout (c,h) → addr = c*HIDDEN + h
        with open(f"{out}/mvau{mvau_idx}_down.bin","wb") as f:
            f.write(adap["rom_down"].astype(np.uint32).tobytes())
        # up: OUT_CH * (1 or 2 u32 for MVAU5)
        if adap["UP_WIDTH"] == 64:
            up_arr = np.zeros((adap["OUT_CH"]*2,), dtype=np.uint32)
            for oc in range(adap["OUT_CH"]):
                v = int(adap["rom_up"][oc])
                up_arr[oc*2]   = v & 0xFFFFFFFF
                up_arr[oc*2+1] = (v >> 32) & 0xFFFFFFFF
            with open(f"{out}/mvau{mvau_idx}_up.bin","wb") as f: f.write(up_arr.tobytes())
        else:
            with open(f"{out}/mvau{mvau_idx}_up.bin","wb") as f:
                f.write(adap["rom_up"].astype(np.uint32).tobytes())
        # contrib: 256 int32
        with open(f"{out}/mvau{mvau_idx}_contrib.bin","wb") as f:
            f.write(adap["contrib"].astype(np.int32).tobytes())
        # thresh: OUT_CH int32
        with open(f"{out}/mvau{mvau_idx}_thresh.bin","wb") as f:
            f.write(thr.tobytes())
        # sign: OUT_CH u32
        with open(f"{out}/mvau{mvau_idx}_sign.bin","wb") as f:
            f.write(sgn.astype(np.uint32).tobytes())
        print(f"  MVAU{mvau_idx}: {len(adap['rc'])}rc {adap['rom_down'].size}down {len(adap['rom_up'])}up alpha={adap['alpha']:.3f}")

for name, path in CKPTS.items():
    print(f"\n=== {name.upper()} ===")
    print(f"Loading {path}")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k.replace("module.",""): v for k,v in sd.items()}
    write_dataset_cfg(name, sd)

print("\nCifar10 default = adapter OFF + Phase-0 baked thresh (no separate cfg needed for MVAU1-5)")
print(f"\nAll outputs at {OUT_DIR}/")
