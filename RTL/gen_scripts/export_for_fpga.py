# ===========================================================================
# [交接導向註解]
# 腳本：PyTorch → FPGA 資產匯出（通用版）。流程：訓練後 → RTL/hardware_assets。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

# export_for_fpga.py
import torch
import os
import argparse
import sys
import numpy as np
from models.CNV import cnv

# 對應官方 FINN 層配置
LAYER_CONFIGS = {
    0: {'PE': 16, 'SIMD': 3,  'IN_CH': 3,   'OUT_CH': 64},
    1: {'PE': 32, 'SIMD': 32, 'IN_CH': 64,  'OUT_CH': 64},
    2: {'PE': 16, 'SIMD': 32, 'IN_CH': 64,  'OUT_CH': 128},
    3: {'PE': 16, 'SIMD': 32, 'IN_CH': 128, 'OUT_CH': 128},
    4: {'PE': 4,  'SIMD': 32, 'IN_CH': 128, 'OUT_CH': 256},
    5: {'PE': 1,  'SIMD': 32, 'IN_CH': 256, 'OUT_CH': 256},
}

def pack_weights_simple(weight_tensor, layer_type, config, bit_width=1):
    """
    修正版：針對 1-bit 採用 +1->1, -1->0 的編碼方式
    """
    if bit_width == 1:
        # [關鍵修正] 1-bit Encoding: 正數變 1, 負數變 0
        w_bin = (weight_tensor >= 0).int().cpu().numpy()
    else:
        # 多位元則四捨五入
        w_bin = torch.round(weight_tensor).int().cpu().numpy()

    hex_lines = []
    
    # 這裡的打包邏輯會根據你的 RTL 讀取方式調整
    # 目前維持 32-bit line 輸出以方便 Verilog $readmemh
    w_flat = w_bin.flatten()
    
    # 補齊 32 的倍數 (Padding)
    pad_len = (32 - (len(w_flat) % 32)) % 32
    if pad_len > 0:
        w_flat = np.pad(w_flat, (0, pad_len), 'constant')

    for i in range(0, len(w_flat), 32):
        chunk = w_flat[i : i + 32]
        val = 0
        for idx, bit in enumerate(chunk):
            val |= (int(bit) << idx)
        hex_lines.append(f"{val:08x}")
    
    return hex_lines

def export_rc_dat(branch, output_dir, idx):
    """匯出 Re-centering (Bias) 作為 8-bit 整數"""
    if branch.down.bias is None: return

    # 嘗試獲取量化後的 Bias 數據
    if hasattr(branch.down, 'quant_bias'):
         bias_val = branch.down.quant_bias()
         if hasattr(bias_val, 'value'): bias_val = bias_val.value
         bias_data = bias_val.detach().cpu().numpy()
    else:
         bias_data = branch.down.bias.data.detach().cpu().numpy()

    hex_lines = []
    for val in bias_data:
        # 轉成 8-bit 補數格式的 Hex
        int_val = int(round(float(val)))
        hex_lines.append(f"{int_val & 0xFF:02x}")
        
    with open(os.path.join(output_dir, f"adapter_{idx}_rc.dat"), "w") as f:
        f.write("\n".join(hex_lines) + "\n")

def export_adapter_dat(model, output_dir, net_bit):
    print(f"📦 Exporting Adapter Assets to {output_dir}...")
    for idx, adapter in enumerate(model.adapters):
        # 跳過 Identity 或沒有 branch 的層
        if type(adapter).__name__ == 'Identity': continue
        if not hasattr(adapter, 'branches'): continue
        if idx not in LAYER_CONFIGS: continue
        
        config = LAYER_CONFIGS[idx]
        branch = adapter.branches[0] # 預設匯出第一個分支 (m=1)
        
        # 獲取量化權重 (Brevitas 格式)
        w_down = branch.down.quant_weight().value if hasattr(branch.down, 'quant_weight') else branch.down.weight
        w_up = branch.up.quant_weight().value if hasattr(branch.up, 'quant_weight') else branch.up.weight

        # 下層權重位元數判斷 (第一層強制 8-bit)
        down_bit = 8 if (hasattr(branch.down, 'weight_bit_width') and branch.down.weight_bit_width > 1) else 1
        
        lines_down = pack_weights_simple(w_down, 'down', config, bit_width=down_bit)
        with open(os.path.join(output_dir, f"adapter_{idx}_down.dat"), "w") as f:
            f.write("\n".join(lines_down) + "\n")
            
        lines_up = pack_weights_simple(w_up, 'up', config, bit_width=1)
        with open(os.path.join(output_dir, f"adapter_{idx}_up.dat"), "w") as f:
            f.write("\n".join(lines_up) + "\n")
            
        export_rc_dat(branch, output_dir, idx)
    print("✅ Adapter assets exported.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--net_bit", type=int, default=1)
    parser.add_argument("--num_branches", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 模擬 Config 載入模型
    class MockConfig:
        def getint(self, sec, key):
            if key == 'WEIGHT_BIT_WIDTH': return args.net_bit
            if key == 'ACT_BIT_WIDTH': return args.net_bit
            if key == 'IN_BIT_WIDTH': return 8
            if key == 'NUM_CLASSES': return 10
            if key == 'IN_CHANNELS': return 3
            if key == 'NUM_BRANCHES': return args.num_branches
            if key == 'BIT_WIDTH': return args.net_bit
            if key == 'RC_BIT_WIDTH': return 8
            return 0
        
        # [修正] 增加 fallback 參數以相容 models/CNV.py 的呼叫
        def getboolean(self, sec, key, fallback=None): 
            return True
            
        def has_section(self, sec): return True
        def has_option(self, sec, key): return True

    model = cnv(MockConfig())
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    export_adapter_dat(model, args.output_dir, args.net_bit)

if __name__ == "__main__":
    main()