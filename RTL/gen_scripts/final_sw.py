# ===========================================================================
# [交接導向註解]
# 腳本：★ 主匯出器：載入 PyTorch checkpoint(RC_m1_full.tar) → 產生全部硬體 .dat assets
# （輸出到 hardware_assets/）。PE=1；pack_bits_to_hex 把二值權重打包成 $readmemh hex。
# 這就是『PyTorch 參數 → .dat』的核心轉換腳本。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

import torch
import os
import numpy as np
import math
from models import model_with_cfg
from models.CNV import cnv

# === 1. 硬體規格設定 (PE 全部改為 1) ===
MVAU_CONFIGS = [
    (1, 3),    # mvau0 (Conv0)
    (1, 32),   # mvau1 (Conv1)
    (1, 32),   # mvau2 (Conv2)
    (1, 32),   # mvau3 (Conv3)
    (1, 32),   # mvau4 (Conv4)
    (1, 32),   # mvau5 (Conv5)
    (1, 4),    # mvau6 (FC0)
    (1, 8),    # mvau7 (FC1)
    (1, 1)     # mvau8 (FC2)
]

CHECKPOINT_PATH = "RC_m1_full.tar"
OUTPUT_DIR = "hardware_assets_pe_simd_aligned_new"

def pack_bits_to_hex(bits_array, bits_per_word):
    """ 將位元陣列打包成指定寬度的 Hex 格式 (LSB-first) """
    hex_lines = []
    for i in range(0, len(bits_array), bits_per_word):
        chunk = bits_array[i : i + bits_per_word]
        val = 0
        for bit_idx, b in enumerate(chunk):
            if b > 0: val |= (1 << bit_idx)
        hex_len = math.ceil(bits_per_word / 4)
        hex_lines.append(f"{val:0{hex_len}x}")
    return hex_lines

def interleave_mvau_weights(layer, pe, simd):
    """ 按照 FINN 標準進行權重交織: (Out_Ch/PE, In_Depth/SIMD, PE, SIMD) """
    w = layer.weight.data
    w_flat = w.view(w.shape[0], -1) 
    bits = (w_flat >= 0).int()
    
    interleaved_bits = []
    num_pe_groups = w.shape[0] // pe
    num_simd_chunks = w_flat.shape[1] // simd
    
    for g in range(num_pe_groups):
        for c in range(num_simd_chunks):
            for p in range(pe):
                chunk = bits[g*pe + p, c*simd : (c+1)*simd]
                interleaved_bits.extend(chunk.tolist())
    return interleaved_bits

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # 載入模型與權重
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    sd_clean = {k[7:] if k.startswith('module.') else k: v for k, v in (ckpt['state_dict'] if 'state_dict' in ckpt else ckpt).items()}
    
    # 分離並儲存 baseline.tar
    baseline_sd = {k: v for k, v in sd_clean.items() if "adapters" not in k}
    torch.save({'state_dict': baseline_sd}, os.path.join(OUTPUT_DIR, "baseline.tar"))

    # 建立模型實例
    model_cfg, cfg = model_with_cfg("CNV", False)
    cfg.add_section('ADAPTER'); cfg.set('ADAPTER', 'NUM_BRANCHES', '1')
    cfg.set('ADAPTER', 'BIT_WIDTH', '1'); cfg.set('ADAPTER', 'USE_RC', 'True'); cfg.set('ADAPTER', 'RC_BIT_WIDTH', '8')
    model = cnv(cfg); model.use_adapter = True
    model.load_state_dict(sd_clean, strict=False); model.eval()

    all_layers = [m for m in model.conv_features if isinstance(m, torch.nn.Conv2d)] + \
                 [m for m in model.linear_features if isinstance(m, torch.nn.Linear)]

    # --- 2. 匯出 MVAU 權重與門檻 ---
    for i, layer in enumerate(all_layers):
        pe, simd = MVAU_CONFIGS[i]
        
        interleaved = interleave_mvau_weights(layer, pe, simd)
        packed_w = pack_bits_to_hex(interleaved, pe * simd)
        with open(os.path.join(OUTPUT_DIR, f"mvau_{i}_weight.dat"), "w") as f:
            for line in packed_w: f.write(line + "\n")
            
        bn = None
        if i < 6:
            for idx, m in enumerate(model.conv_features):
                if m == layer and isinstance(model.conv_features[idx+1], torch.nn.BatchNorm2d):
                    bn = model.conv_features[idx+1]; break
        else:
            for idx, m in enumerate(model.linear_features):
                if m == layer and isinstance(model.linear_features[idx+1], torch.nn.BatchNorm1d):
                    bn = model.linear_features[idx+1]; break
        
        if bn:
            k_val = layer.weight.data[0].numel()
            t_bip = bn.running_mean.data - (bn.bias.data * torch.sqrt(bn.running_var.data + bn.eps)) / bn.weight.data
            t_pop = torch.ceil((t_bip + k_val) / 2.0).int()
            with open(os.path.join(OUTPUT_DIR, f"threshs_ROM_MVAU{i}.dat"), "w") as f:
                for v in t_pop: f.write(f"{v.item() & 0x7FF:03x}\n")

    # --- 3. 匯出 Adapter 0~5 ---
    for i, adp_module in enumerate(model.adapters):
        if not hasattr(adp_module, 'branches'): continue
        adp = adp_module.branches[0]
        _, simd_mvau = MVAU_CONFIGS[i] 
        
        w_down_bits = (adp.down.weight.data >= 0).int().flatten().tolist()
        packed_down = pack_bits_to_hex(w_down_bits, simd_mvau)
        with open(os.path.join(OUTPUT_DIR, f"adapter_{i}_down.dat"), "w") as f:
            for line in packed_down: f.write(line + "\n")
            
        hw_rc = torch.floor(adp.down.bias.data / 2.0 - (adp.down.in_channels / 2.0)).int()
        with open(os.path.join(OUTPUT_DIR, f"adapter_{i}_rc.dat"), "w") as f:
            for v in hw_rc: f.write(f"{v.item() & 0xFFFF:04x}\n")
            
        w_up_bits = (adp.up.weight.data >= 0).int()
        with open(os.path.join(OUTPUT_DIR, f"adapter_{i}_up.dat"), "w") as f:
            for oc in range(w_up_bits.shape[0]):
                bits_64 = torch.zeros(64, dtype=torch.int)
                actual = w_up_bits[oc].flatten()
                bits_64[:len(actual)] = actual[:64]
                low = 0
                for b in range(32):
                    if bits_64[b] > 0: low |= (1 << b)
                high = 0
                for b in range(32, 64):
                    if bits_64[b] > 0: high |= (1 << (b-32))
                f.write(f"{low:08x}\n{high:08x}\n")

        # D. 新增：匯出 Alpha 值 (轉換為 Q8 定點數整數)
        alpha_val = adp_module.alphas[0].item()
        alpha_q8 = int(round(alpha_val * 256.0))
        with open(os.path.join(OUTPUT_DIR, f"adapter_{i}_alpha.dat"), "w") as f:
            # 直接寫入整數值，方便 Verilog 讀取
            f.write(f"{alpha_q8}\n")

    print(f"✅ 成功匯出對齊 PE=1 且包含 Q8 格式 Alpha 值的硬體資產至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
