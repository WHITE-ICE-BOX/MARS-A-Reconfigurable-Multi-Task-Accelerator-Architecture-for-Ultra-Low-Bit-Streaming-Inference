#!/usr/bin/env python3
# ===========================================================================
# [交接導向註解]
# 腳本：產生 MVAU1–5 的 golden 測資（RTL 模擬時逐模組比對用）。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

"""
generate_mvau_golden.py
---------------------------
Generate RTL-accurate golden data for individual MVAU1-5 testbenches.

CRITICAL: The MVAU popcount is computed from memblock.dat (FINN-compiled weights),
NOT from PyTorch conv weights. The conv weights in RC_m1_full.tar differ from
memblock.dat because adapter training also modified the backbone weights.

The golden generation:
  1. Generates INPUT data from PyTorch model forward pass (binary activations)
  2. Computes EXPECTED output by simulating RTL computation:
     - MVAU popcount from memblock.dat XNOR+popcount
     - Adapter popcount from PyTorch adapter weights (binary XNOR+popcount)
     - Contribution lookup from .dat LUTs
     - Threshold comparison from .dat threshold ROMs

10 random images using torch.manual_seed(2026 + i).
"""
import os, sys, math

sys.path.insert(0, "/home/barkie1/finn_brevitis/brevitas/src")
sys.path.insert(0, "/home/barkie1/mvau_pipeline/sw")

import numpy as np
import torch
import torch.nn.functional as F
from models.CNV import cnv

CKPT = "/home/barkie1/mvau_pipeline/sw/paper_results/stage1_models/RC_m1_full.tar"
GOLDEN_ROOT = "/home/barkie1/mvau_pipeline/mvau_adapter"
NUM_SAMPLES = 10

# Layer config
LAYER_INDEX_MAP = {
    0: (1, 2), 1: (4, 5), 2: (8, 9),
    3: (11, 12), 4: (15, 16), 5: (18, 19),
}
POOL_AFTER = [False, True, False, True, False, False]

# MVAU configs
MVAU_CFGS = {
    1: {"in_ch": 64,  "out_ch": 64,  "PE": 32, "SIMD": 32, "kernel": 3,
        "adp_idx": 1, "reduction": 4, "offset": 24, "up_width": 32},
    2: {"in_ch": 64,  "out_ch": 128, "PE": 16, "SIMD": 32, "kernel": 3,
        "adp_idx": 2, "reduction": 4, "offset": 24, "up_width": 32},
    3: {"in_ch": 128, "out_ch": 128, "PE": 16, "SIMD": 32, "kernel": 3,
        "adp_idx": 3, "reduction": 4, "offset": 16, "up_width": 32},
    4: {"in_ch": 128, "out_ch": 256, "PE": 4,  "SIMD": 32, "kernel": 3,
        "adp_idx": 4, "reduction": 4, "offset": 16, "up_width": 32},
    5: {"in_ch": 256, "out_ch": 256, "PE": 1,  "SIMD": 32, "kernel": 3,
        "adp_idx": 5, "reduction": 4, "offset": 32, "up_width": 64},
}


class MockCfg:
    def __init__(self):
        self.d = {
            "QUANT": {"WEIGHT_BIT_WIDTH": 1, "ACT_BIT_WIDTH": 1, "IN_BIT_WIDTH": 8},
            "MODEL": {"NUM_CLASSES": 10, "IN_CHANNELS": 3},
            "ADAPTER": {"NUM_BRANCHES": 1, "BIT_WIDTH": 1, "RC_BIT_WIDTH": 8, "USE_RC": True},
        }
    def getint(self, s, k): return self.d.get(s, {}).get(k, 0)
    def getboolean(self, s, k, fallback=None):
        v = self.d.get(s, {}).get(k)
        return v if v is not None else (fallback if fallback is not None else False)
    def has_section(self, s): return s in self.d


# =====================================================================
# Utility functions
# =====================================================================

def popcount32(x):
    return bin(x & 0xFFFFFFFF).count('1')

def popcount64(x):
    return bin(x & 0xFFFFFFFFFFFFFFFF).count('1')

def read_hex_signed32(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                v = int(line, 16)
                if v >= 0x80000000:
                    v -= (1 << 32)
                vals.append(v)
    return vals

def read_hex_signed16(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                v = int(line, 16)
                if v >= 0x8000:
                    v -= (1 << 16)
                vals.append(v)
    return vals

def to_u32(x):
    if x < 0:
        x += (1 << 32)
    return x & 0xFFFFFFFF


# =====================================================================
# Load .dat files
# =====================================================================

def load_memblock(mvau_idx):
    """Load memblock.dat as list of large integers."""
    path = os.path.join(GOLDEN_ROOT, f"mvau{mvau_idx}/data/memblock.dat")
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(line, 16))
    return vals

def load_contrib_lut(mvau_idx):
    if mvau_idx <= 4:
        path = os.path.join(GOLDEN_ROOT, f"mvau{mvau_idx}/data",
                            f"mvau{mvau_idx}_contrib_lut.dat")
    else:
        path = os.path.join(GOLDEN_ROOT, "mvau5/data", "adapter_5_contrib_lut.dat")
    return read_hex_signed32(path)

def load_thresholds(mvau_idx):
    cfg = MVAU_CFGS[mvau_idx]
    OUT_CH = cfg["out_ch"]
    PE = cfg["PE"]
    STEPS = OUT_CH // PE

    if mvau_idx <= 4:
        data_dir = os.path.join(GOLDEN_ROOT, f"mvau{mvau_idx}/data")
        thresh = [0] * OUT_CH
        sign = [0] * OUT_CH
        for p in range(PE):
            tvals = read_hex_signed32(os.path.join(data_dir, f"mvau{mvau_idx}_thresh_{p}_q8.dat"))
            spath = os.path.join(data_dir, f"mvau{mvau_idx}_adp_sign_{p}.dat")
            with open(spath) as f:
                svals = [int(line.strip()) for line in f if line.strip()]
            for s in range(STEPS):
                ch = s * PE + p
                thresh[ch] = tvals[s]
                sign[ch] = svals[s]
    else:
        data_dir = os.path.join(GOLDEN_ROOT, "mvau5/data")
        thresh = read_hex_signed32(os.path.join(data_dir, "threshs_ROM_MVAU5_q8.dat"))
        with open(os.path.join(data_dir, "threshs_pol_MVAU5.dat")) as f:
            sign = [int(line.strip()) for line in f if line.strip()]

    return thresh, sign


# =====================================================================
# BN parameters (for MVAU0 only — no adapter)
# =====================================================================

def bn_params(bn):
    gamma = bn.weight.data.double()
    beta  = bn.bias.data.double()
    mean  = bn.running_mean.data.double()
    var   = bn.running_var.data.double()
    eps   = bn.eps
    t_bip = mean - beta * torch.sqrt(var + eps) / gamma
    pol   = (gamma < 0)
    return t_bip, pol


# =====================================================================
# Forward pass to generate layer inputs
# =====================================================================
# We use the ORIGINAL q8_forward to compute layer inputs. These inputs
# are what gets packed into mvauN_in.dat. The testbench feeds them into
# the RTL. Even though the PyTorch weights differ from memblock.dat,
# the layer inputs are deterministic from the model forward pass.
#
# NOTE: For top-level simulation, the layer inputs would come from
# the RTL itself (cascaded MVAUs). For individual testbenches, we
# feed the inputs ourselves, so using PyTorch forward is fine.

def q8_forward(model, x):
    """Q8 forward pass to get layer inputs and outputs (PyTorch-based)."""
    layer_inputs = {}
    layer_outputs = {}

    x = 2.0 * x.double() - 1.0
    x = model.conv_features[0](x).double()

    for mvau_idx in range(6):
        conv_idx, bn_idx = LAYER_INDEX_MAP[mvau_idx]
        conv = model.conv_features[conv_idx]
        bn   = model.conv_features[bn_idx]

        w = conv.weight.data.double()
        w_bin = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))

        x_in = x
        layer_inputs[mvau_idx] = x_in.clone()

        x_conv = F.conv2d(x_in, w_bin, bias=None, padding=0)

        if mvau_idx > 0:
            with torch.no_grad():
                x_adp = model.adapters[mvau_idx](x_in.float()).double()
            x_adp_c = x_adp[:, :, 1:-1, 1:-1]
            x_combined = x_conv + x_adp_c
        else:
            x_combined = x_conv

        t_bip, pol = bn_params(bn)
        C = t_bip.shape[0]
        t_view = t_bip.view(1, C, 1, 1)
        pol_view = pol.view(1, C, 1, 1)

        if mvau_idx == 0:
            cmp = torch.round(x_combined) >= torch.ceil(t_view)
        else:
            cmp = torch.round(x_combined * 128.0) >= torch.round(t_view * 128.0)

        x = (cmp ^ pol_view).double() * 2.0 - 1.0
        layer_outputs[mvau_idx] = x.clone()

        if POOL_AFTER[mvau_idx]:
            x = F.max_pool2d(x, 2)

    return layer_inputs, layer_outputs


# =====================================================================
# Compute RTL-accurate expected output from input data + memblock
# =====================================================================

def compute_expected_output(input_words, mvau_idx, model):
    """
    Compute the expected RTL output for one MVAU given its input words.

    Args:
        input_words: list of 32-bit input values (from mvauN_in.dat)
        mvau_idx: 1-5
        model: PyTorch model (for adapter weights)

    Returns:
        output_words: list of output hex values
    """
    cfg = MVAU_CFGS[mvau_idx]
    PE = cfg["PE"]
    SIMD = cfg["SIMD"]
    IN_CH = cfg["in_ch"]
    OUT_CH = cfg["out_ch"]
    HIDDEN = IN_CH // cfg["reduction"]
    UP_WIDTH = cfg["up_width"]
    KERNEL = cfg["kernel"]
    K = IN_CH * KERNEL * KERNEL
    STEPS = OUT_CH // PE
    IN_CHUNKS = IN_CH // SIMD
    WORDS_PER_STEP = K // SIMD
    PIXELS = KERNEL * KERNEL
    WORDS_PER_WINDOW = PIXELS * IN_CHUNKS
    CENTER_PIXEL = (PIXELS - 1) // 2  # pixel 4 for 3x3

    # Load memblock
    memblock = load_memblock(mvau_idx)

    # Load threshold data
    thresh_list, sign_list = load_thresholds(mvau_idx)
    contrib_lut = load_contrib_lut(mvau_idx)

    # Load adapter weights from PyTorch
    adp_module = model.adapters[cfg["adp_idx"]]
    branch = adp_module.branches[0]

    w_down = branch.down.weight.data
    w_down_bin = (w_down >= 0).int().view(HIDDEN, IN_CH)

    w_up = branch.up.weight.data
    w_up_bin = (w_up >= 0).int().view(OUT_CH, HIDDEN)

    # Load RC bias from .dat file (matches what RTL loads)
    rc_path = os.path.join(GOLDEN_ROOT, f"mvau{mvau_idx}/data",
                           f"adapter_{mvau_idx}_rc.dat")
    hw_rc = read_hex_signed16(rc_path)

    # Precompute adapter down weight as packed SIMD words for each hidden channel
    # For chunk c, hidden h: packed_down[h][c] = 32-bit packed weight
    packed_down = []
    for h in range(HIDDEN):
        chunks = []
        for c in range(IN_CHUNKS):
            val = 0
            for s in range(SIMD):
                if w_down_bin[h, c * SIMD + s].item():
                    val |= (1 << s)
            chunks.append(val)
        packed_down.append(chunks)

    # Precompute adapter up weight as packed words per output channel
    # For MVAU1-4: 32-bit (HIDDEN zero-padded to 32)
    # For MVAU5: 64-bit
    packed_up = []
    for oc in range(OUT_CH):
        val = 0
        for h in range(HIDDEN):
            if w_up_bin[oc, h].item():
                val |= (1 << h)
        packed_up.append(val)

    # Compute number of windows
    num_windows = len(input_words) // WORDS_PER_WINDOW

    # Output words
    output_words = []

    # Hex length for output
    if PE >= 32:
        hex_len = 8
    elif PE >= 16:
        hex_len = 4
    else:
        hex_len = 2

    for win in range(num_windows):
        win_offset = win * WORDS_PER_WINDOW

        # --- 1. Adapter: extract center pixel and compute hidden activation ---
        center_start = CENTER_PIXEL * IN_CHUNKS
        center_words = []
        for c in range(IN_CHUNKS):
            center_words.append(input_words[win_offset + center_start + c])

        # Compute hidden accumulator
        hidden_acc = list(hw_rc)  # copy
        for c in range(IN_CHUNKS):
            act_word = center_words[c]
            for h in range(HIDDEN):
                xnor_val = ~(act_word ^ packed_down[h][c]) & 0xFFFFFFFF
                hidden_acc[h] += popcount32(xnor_val)

        # Sign activation
        hidden_act = 0
        for h in range(HIDDEN):
            if hidden_acc[h] >= 0:
                hidden_act |= (1 << h)

        # --- 2. For each output step/PE: compute MVAU pop + adapter pop ---
        for s in range(STEPS):
            out_val = 0

            for p in range(PE):
                ch = s * PE + p

                # MVAU popcount from memblock
                mvau_pop = 0
                for w in range(WORDS_PER_STEP):
                    in_word = input_words[win_offset + w]
                    mem_word = memblock[s * WORDS_PER_STEP + w]
                    w_bits = (mem_word >> (p * SIMD)) & 0xFFFFFFFF
                    xnor_val = ~(in_word ^ w_bits) & 0xFFFFFFFF
                    mvau_pop += popcount32(xnor_val)

                # Adapter popcount
                up_weight = packed_up[ch]
                if UP_WIDTH == 32:
                    xnor_up = ~(hidden_act ^ up_weight) & 0xFFFFFFFF
                    adp_pop = popcount32(xnor_up)
                else:  # 64-bit for MVAU5
                    xnor_up = ~(hidden_act ^ up_weight) & 0xFFFFFFFFFFFFFFFF
                    adp_pop = popcount64(xnor_up)

                # Contribution lookup
                contrib = contrib_lut[adp_pop]

                # Threshold comparison
                thresh = thresh_list[ch]
                adp_sign = sign_list[ch]

                if mvau_idx <= 4:
                    # MVAU1-4: total = (pop << 8) + (sign ? -contrib : contrib)
                    signed_c = -contrib if adp_sign else contrib
                    total_q8 = (mvau_pop << 8) + signed_c
                    result_bit = 1 if total_q8 >= thresh else 0
                else:
                    # MVAU5: total = ((pol ? (K-pop) : pop) << 8) + contrib
                    adjusted_pop = (K - mvau_pop) if adp_sign else mvau_pop
                    total_q8 = (adjusted_pop << 8) + contrib
                    compare = 1 if total_q8 >= thresh else 0
                    result_bit = compare ^ adp_sign

                if result_bit:
                    out_val |= (1 << p)

            output_words.append(f"{out_val:0{hex_len}x}")

    return output_words


# =====================================================================
# AXI packing (for input data)
# =====================================================================

def pack_input_axi(x_in, in_ch, simd, kernel_size):
    """Pack pre-conv binary activation into FINN AXI-Stream input format.
    Returns list of 32-bit integer values.
    """
    x_bin = (x_in > 0).to(torch.uint8)
    patches = F.unfold(x_bin.float(), kernel_size=kernel_size).to(torch.uint8)
    num_windows = patches.shape[2]
    pixels_per_window = kernel_size * kernel_size
    patches = patches.view(in_ch, pixels_per_window, num_windows)
    patches = patches.permute(2, 1, 0)
    chunks_per_pixel = in_ch // simd

    words = []
    for w in range(num_windows):
        for p in range(pixels_per_window):
            channels = patches[w, p, :]
            for chunk in range(chunks_per_pixel):
                chunk_vals = channels[chunk * simd : (chunk + 1) * simd]
                val32 = 0
                for bit_idx in range(simd):
                    val32 |= (int(chunk_vals[bit_idx].item()) << bit_idx)
                words.append(val32)
    return words


# =====================================================================
# Main
# =====================================================================

def main():
    print("Loading model...")
    model = cnv(MockCfg())
    model.use_adapter = True
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Preload .dat files
    print("Loading .dat files...")
    for n in [1, 2, 3, 4, 5]:
        t, s = load_thresholds(n)
        c = load_contrib_lut(n)
        m = load_memblock(n)
        cfg = MVAU_CFGS[n]
        expected_mem = cfg["out_ch"] // cfg["PE"] * (cfg["in_ch"] * 9 // cfg["SIMD"])
        print(f"  MVAU{n}: memblock={len(m)} words (expected {expected_mem}), "
              f"thresh={len(t)} ch, contrib_lut={len(c)} entries")

    # Open output files
    files_in = {}
    files_out = {}
    for n in [1, 2, 3, 4, 5]:
        golden_dir = os.path.join(GOLDEN_ROOT, f"mvau{n}", "golden_data")
        os.makedirs(golden_dir, exist_ok=True)
        files_in[n] = open(os.path.join(golden_dir, f"mvau{n}_in.dat"), "w")
        files_out[n] = open(os.path.join(golden_dir, f"mvau{n}_expected.dat"), "w")

    # Accumulate input words per MVAU across samples
    all_input_words = {n: [] for n in [1, 2, 3, 4, 5]}

    for sample_idx in range(NUM_SAMPLES):
        torch.manual_seed(2026 + sample_idx)
        dummy_image = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            layer_inputs, _ = q8_forward(model, dummy_image)

        for n in [1, 2, 3, 4, 5]:
            cfg = MVAU_CFGS[n]
            x_in = layer_inputs[n]

            # Pack input data
            input_words = pack_input_axi(x_in, cfg["in_ch"], cfg["SIMD"], cfg["kernel"])
            all_input_words[n].extend(input_words)

            # Write input words
            for w in input_words:
                files_in[n].write(f"{w:08x}\n")

        if (sample_idx + 1) % 5 == 0:
            print(f"  Packed inputs for {sample_idx + 1}/{NUM_SAMPLES} images")

    # Close input files
    for n in [1, 2, 3, 4, 5]:
        files_in[n].close()

    # Now compute expected outputs from input data + memblock
    print("\nComputing expected outputs from memblock.dat + adapter...")
    for n in [1, 2, 3, 4, 5]:
        print(f"  MVAU{n}...", end=" ", flush=True)
        output_words = compute_expected_output(all_input_words[n], n, model)
        for w in output_words:
            files_out[n].write(w + "\n")
        files_out[n].close()
        print(f"done ({len(output_words)} output words)")

    # Verify line counts
    print("\nLine count verification:")
    for n in [1, 2, 3, 4, 5]:
        golden_dir = os.path.join(GOLDEN_ROOT, f"mvau{n}", "golden_data")
        in_path = os.path.join(golden_dir, f"mvau{n}_in.dat")
        out_path = os.path.join(golden_dir, f"mvau{n}_expected.dat")
        with open(in_path) as f:
            in_lines = sum(1 for _ in f)
        with open(out_path) as f:
            out_lines = sum(1 for _ in f)
        print(f"  MVAU{n}: input={in_lines} lines, output={out_lines} lines")

    print("\nDone! Golden data regenerated with RTL-accurate computation.")


if __name__ == "__main__":
    main()
