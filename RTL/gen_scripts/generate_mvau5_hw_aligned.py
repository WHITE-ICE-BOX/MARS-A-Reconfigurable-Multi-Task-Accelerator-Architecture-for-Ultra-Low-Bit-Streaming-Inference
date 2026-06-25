# ===========================================================================
# [交接導向註解]
# 腳本：產生 MVAU5 硬體對齊（hw-aligned）資產 .dat。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

import torch
import os
import sys

# 確保能載入你的模型
try:
    from models import model_with_cfg
    from models.CNV import cnv
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import model_with_cfg
    from models.CNV import cnv

def main():
    print("-> 啟動【硬體同調版】Golden Data 生成器...")
    
    # 1. 建立模型並載入權重
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

    # ⚠️ 請確保路徑正確
    ckpt_path = "paper_results/hardware_export/S2_Distill_m1_T32_SVHN/svhn_full.tar"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 2. 讀取 FPGA 的 Threshold ROM (代表 Stage 1 凍結的真實閾值)
    # ⚠️ 請確保這個 .dat 檔與本 python 腳本在同一資料夾
    thresh_file = "StreamingDataflowPartition_1_MVAU_hls_5_Matrix_Vector_Activate_Stream_Batch_threshs_ROM_AUTO_1R.dat"
    thresh_rom = []
    with open(thresh_file, "r") as f:
        for line in f:
            thresh_rom.append(int(line.strip(), 16))

    # 3. 攔截特徵 (攔截 BN 前的純 MAC)
    captured = {}
    def hook_conv(module, input, output):
        captured['conv_mac'] = output.detach() 
    def hook_adp(module, input, output):
        captured['adp_mac'] = output.detach()

    model.conv_features[18].register_forward_hook(hook_conv)
    model.adapters[5].branches[0].up.register_forward_hook(hook_adp)

    # 用相同的 Seed 產生測試資料
    torch.manual_seed(2026)
    dummy_image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _ = model(dummy_image)

    # 取出中心點的 Bipolar MAC (-256 ~ 256)
    conv_mac = captured['conv_mac'][0, :, 0, 0] # 骨幹 (256,)
    adp_mac = captured['adp_mac'][0, :, 1, 1]   # Adapter (256,)

    # 4. 轉換為 FPGA 底層的 Popcount 數學 (0 ~ 256)
    # 公式: Popcount = (Bipolar MAC + 總輸入通道數) // 2
    p_conv = (conv_mac + 2304) // 2
    p_adp = (adp_mac + 64) // 2

    # 5. 模擬 Verilog 計算並比對 ROM
    hw_expected = []
    for ch in range(256):
        # 這行對應我們在 Stream_Adder_Threshold.v 寫的數學：
        total_mac = p_conv[ch] + p_adp[ch] - 32
        
        # 拿硬體的 ROM 當作標準答案 (不看 PyTorch 飄移的 BN)
        t_finn = thresh_rom[ch]
        
        # 啟動函數判定
        result = 1 if total_mac >= t_finn else 0
        hw_expected.append(result)

    # 6. 輸出最終的黃金解答
    out_file = "mvau5_expected_output.dat"
    with open(out_file, "w") as f:
        for val in hw_expected:
            f.write(f"{val:02x}\n")

    print(f"✅ 硬體同調版預期輸出已匯出至: {out_file} ！")

if __name__ == "__main__":
    main()