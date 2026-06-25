# ===========================================================================
# [交接導向註解]
# 腳本：產生 MVAU5 的 Q8 threshold + polarity ROM（.dat）。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

"""Generate Q8 threshold ROM + polarity ROM for MVAU5's Stream_Adder_Threshold.

Outputs (hardware_assets_pe_simd_aligned_new/):
    threshs_ROM_MVAU5_q8.dat : 256 lines, signed int32 as 8-hex (2's complement).
                               = round(t_pop_float * 256), clamped to int32 range,
                                 out-of-range entries forced to INT32_MAX so the
                                 RTL compare becomes constantly false.
    threshs_pol_MVAU5.dat    : 256 lines, 1-bit per channel. 1 iff BN gamma<0.

Derivation (matches FINN popcount-domain sign activation):
    t_bip       = mean - beta*sqrt(var+eps)/gamma            (bipolar domain)
    t_pop_float = (t_bip + k) / 2                            (popcount domain)
    compare     = (popcount_q8 >= thresh_q8)                 (uniform >=)
    out         = compare XOR pol                            (pol=1 flips for gamma<0)

Reads BN params directly from the checkpoint's state_dict (no brevitas import
needed). conv5's BN = `conv_features.19.*`.
"""
import os
import torch

CHECKPOINT = "paper_results/stage1_models/RC_m1_full.tar"
OUT_DIR    = "hardware_assets_pe_simd_aligned_new"
MVAU_IDX   = 5
BN_PREFIX  = "conv_features.19"
CONV_PREFIX = "conv_features.18"
N_CH       = 256
MAX_POP    = 256 * 3 * 3          # conv5: 256 in-ch * 3x3 kernel = 2304
BN_EPS     = 1e-4                 # matches CNV.py: BatchNorm2d(..., eps=1e-4)
INT32_MAX  = 0x7FFFFFFF
INT32_MIN  = -INT32_MAX - 1

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    gamma = sd[f"{BN_PREFIX}.weight"].double()
    beta  = sd[f"{BN_PREFIX}.bias"].double()
    mu    = sd[f"{BN_PREFIX}.running_mean"].double()
    var   = sd[f"{BN_PREFIX}.running_var"].double()
    w = sd[f"{CONV_PREFIX}.weight"]
    k_val = w[0].numel()

    assert gamma.numel() == N_CH, f"unexpected N_CH={gamma.numel()}"
    assert k_val == MAX_POP,      f"unexpected k={k_val}"

    t_bip = mu - beta * torch.sqrt(var + BN_EPS) / gamma
    t_pop_float = (t_bip + k_val) / 2.0

    t_q8 = torch.round(t_pop_float * 256.0).long()
    pol  = (gamma < 0).long()

    q8_lo, q8_hi = 0, MAX_POP * 256
    oor = (t_q8 < q8_lo) | (t_q8 > q8_hi)
    t_q8 = torch.where(oor, torch.full_like(t_q8, INT32_MAX), t_q8)
    t_q8 = torch.clamp(t_q8, INT32_MIN, INT32_MAX)

    thr_path = os.path.join(OUT_DIR, "threshs_ROM_MVAU5_q8.dat")
    pol_path = os.path.join(OUT_DIR, "threshs_pol_MVAU5.dat")
    with open(thr_path, "w") as f:
        for v in t_q8.tolist():
            f.write(f"{v & 0xFFFFFFFF:08x}\n")
    with open(pol_path, "w") as f:
        for v in pol.tolist():
            f.write(f"{v:1d}\n")

    print(f"MVAU{MVAU_IDX}: wrote {thr_path}  ({N_CH} entries, Q8 signed int32)")
    print(f"MVAU{MVAU_IDX}: wrote {pol_path}  ({N_CH} entries, 1-bit)")
    print(f"  gamma<0 channels (pol=1): {int(pol.sum().item())}")
    print(f"  out-of-range thresholds forced to INT32_MAX: {int(oor.sum().item())}")
    print(f"  t_pop_float range before Q8: [{float(t_pop_float.min()):.3f}, "
          f"{float(t_pop_float.max()):.3f}]")

if __name__ == "__main__":
    main()
