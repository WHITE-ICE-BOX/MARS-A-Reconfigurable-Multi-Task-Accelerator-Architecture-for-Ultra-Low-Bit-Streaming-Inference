# ===========================================================================
# [交接導向註解]
# 腳本：工具：修正 RC bias .dat 的格式。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

import torch
import numpy as np
import argparse
import sys
import math

try:
    from models import model_with_cfg
    from models.CNV import cnv
except ImportError:
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.tar or full.tar")
    parser.add_argument("--out_file", type=str, default="adapter_5_rc.dat")
    args = parser.parse_args()

    print("-> 載入模型擷取真實 Bias...")
    model_cfg, cfg = model_with_cfg("CNV", False)
    cfg.set('QUANT', 'WEIGHT_BIT_WIDTH', '1')
    cfg.set('QUANT', 'ACT_BIT_WIDTH', '1')
    cfg.add_section('ADAPTER')
    cfg.set('ADAPTER', 'NUM_BRANCHES', '1')
    cfg.set('ADAPTER', 'BIT_WIDTH', '1')
    cfg.set('ADAPTER', 'USE_RC', 'True')
    cfg.set('ADAPTER', 'RC_BIT_WIDTH', '8')
    
    model = cnv(cfg)
    model.use_adapter = True
    
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    # 取得第 6 層 Adapter 的 Bias (RC)
    bias_tensor = model.adapters[5].branches[0].down.bias.detach().numpy()
    
    print("-> 執行硬體數學轉換: RC = (Bias / 2) - 128 ...")
    
    with open(args.out_file, "w") as f:
        for b in bias_tensor:
            # 關鍵轉換公式！
            hw_rc = math.floor(b / 2.0) - 128
            
            # 確保以 16-bit 補數格式寫入 (處理負數)
            hex_val = hw_rc & 0xFFFF
            f.write(f"{hex_val:04x}\n")
            
    print(f"✅ 修正後的 RC 檔案已匯出至 {args.out_file}！")

if __name__ == "__main__":
    main()