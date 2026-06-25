# FPGA — 最終上板產物（PYNQ-Z2）

PYNQ-Z2（Xilinx XC7Z020，100 MHz）最終上板的 `.bit`/`.hwh`、Python driver、以及所有
**runtime 參數**。共四種 build，對應論文資源/功耗/跨平台與 runtime 切換結果。

> **runtime 參數** = 換任務時透過 `cfg_hub` 寫入的 per-task 組態（各層 thresholds、classifier
> 權重、5 層 adapter blob）。換任務「不重燒 bitstream」，只寫這份約 26 KB 的參數（≈1.86 ms）。

---

## 一、四種 build 一覽

| 資料夾 | Bitstream | 折疊 | 角色 |
|---|---|---|---|
| `backbone_throughput/` | `resizer.bit` | high-PE | 純 backbone 吞吐量基準 |
| `MARS_throughput_2ds/` | `resizer_v1.bit` | high-PE | MARS（2 資料集）能效/吞吐量代表組態（1866 img/s、842.7 img/s/W） |
| `backbone_compact_pe1/` | `resizer.bit` | PE=1 | 純 backbone compact 基準 |
| `MARS_compact_5ds_pe1/` | `resizer_3ds_v3.bit` | PE=1 | MARS（**5 資料集** runtime 切換）板上精度正確性 |

---

## 二、結構樹

```
FPGA/
├── backbone_throughput/             # 純 backbone, high-PE
│   ├── resizer.bit / resizer.hwh    #   bitstream + 硬體交握檔
│   ├── driver.py / driver_base.py   #   FINN PYNQ driver（throughput_test）
│   └── validate.py                  #   精度驗證
│
├── backbone_compact_pe1/            # 純 backbone, PE=1
│   ├── resizer.bit / resizer.hwh
│   ├── driver.py / driver_base.py / validate.py
│   ├── input.npy / cifar10_test_y.npy   #   範例輸入 + 標籤
│   ├── runtime_weights/             #   （backbone-only，僅 thresholds/cls）
│   └── finn/ , qonnx/               #   driver 需要的 vendored 工具（data packing）
│
├── MARS_throughput_2ds/             # MARS（2 資料集）, high-PE
│   ├── resizer_v1.bit / resizer_v1.hwh   # ★ canonical bitstream（md5 c04e8195）
│   ├── driver.py / driver_base.py
│   ├── board_test.py                #   板上精度測試
│   ├── batch_test.py                #   batch-1000 吞吐量量測（→ 1866 img/s）
│   ├── runtime_switch.py            #   runtime 切換流程
│   ├── runtime_params/              #   77 個 .bin：cifar10/svhn/fashion 的 per-task 參數（扁平命名）
│   │                                #     mvau{0..5}_{down,up,rc,sign,lut,thresh}_{svhn,fashion}.bin、
│   │                                #     cls_weights_*.bin、fc1/fc2_thresh_*.bin …
│   └── data/                        #   *_test_y.npy 標籤
│
└── MARS_compact_5ds_pe1/            # ★ MARS（5 資料集）runtime 切換, PE=1
    ├── resizer_3ds_v3.bit / .hwh    #   canonical bitstream（單一 bitstream 服務 5 資料集）
    ├── driver.py / driver_base.py
    ├── runtime_3ds.py               #   ★ 主 runtime 切換 driver（依資料集載入 runtime_weights）
    ├── board_test_10k.py            #   各資料集 10k 張板上精度（→ 論文板上精度表）
    ├── board_switch_fast_bench.py   #   切換延遲量測（→ 1.86 ms）
    ├── data/                        #   *_test_y.npy 標籤
    └── runtime_weights/             # ★ 五資料集 per-task 參數（每組 34 個 .bin）
        ├── cifar10/  svhn/  fashion/  stl10/  cinic10/
        └─（每組）mvau0_thresh / mvau{1..5}_{down,up,rc,sign,contrib,thresh}
                  / fc1_thresh / fc2_thresh / cls_weight .bin
```

★ = 論文核心結果所用。

---

## 三、5 資料集 runtime 切換（重點）

`MARS_compact_5ds_pe1/runtime_weights/` 內含**全部五個資料集**（cifar10、svhn、fashion、
stl10、cinic10）的 per-task 參數，每組 34 個 `.bin`。換任務時 `runtime_3ds.py` 把某個資料夾
串流進 `cfg_hub` 即可，**無 fabric reconfiguration、不需 reconfiguration controller**。

> bitstream 名為 `resizer_3ds_v3`（歷史命名 `3ds`），但同一顆 bitstream 透過上述 weights
> 服務全部五個資料集。板上當時快照只存了 3 組，此處已從本地 canonical `sw/runtime_weights`
> 補齊為完整五組。

---

## 四、板上執行

```bash
# 吞吐量（throughput build）
sudo XILINX_XRT=/usr BOARD=Pynq-Z2 python3 driver.py \
     --exec_mode throughput_test --bitfile resizer_v1.bit --batchsize 1000

# 5 資料集精度 + runtime 切換（compact build）
sudo XILINX_XRT=/usr BOARD=Pynq-Z2 python3 runtime_3ds.py     # 載入 runtime_weights/<dataset>
sudo XILINX_XRT=/usr BOARD=Pynq-Z2 python3 board_test_10k.py  # 各資料集 10k 精度
```

> 各 build 的大型資料集測試輸入（`*_test_x.npy`，約 30 MB）已排除，可由 `../AI_model_train` 重新產生。
