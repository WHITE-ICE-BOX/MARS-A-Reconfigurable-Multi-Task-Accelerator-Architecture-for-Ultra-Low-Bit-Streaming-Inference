# MARS — 超低位元串流推論的可重組多任務加速器架構

> **MARS: A Reconfigurable Multi-Task Accelerator Architecture for Ultra-Low-Bit Streaming Inference**
> 碩士論文最終版程式碼與數據釋出。

---

## 一、這個專案在做什麼

MARS 把 **Conv-Adapter**（參數高效率遷移學習模組）整合進一個**凍結的 1W1A**（1-bit weight / 1-bit activation）**FINN streaming-dataflow CNV backbone**，並加入一個 AXI-Lite 的 **`cfg_hub`**（configuration hub）做**執行時期（runtime）多任務切換**。

核心貢獻：
1. **硬體整合**——把自研 Adapter RTL 以「Super Wrapper」方式包進 FINN 產生的 MVAU（Matrix-Vector-Activation Unit）串流資料流，週期精準（cycle-accurate）同步。
2. **單一 bitstream 多任務切換**——`cfg_hub` 讓**一顆 bitstream**在不做 fabric reconfiguration（不重組態 FPGA 邏輯、不需 reconfiguration controller）下，以 $O(1)$ 晶片成本服務任意數量的同 backbone 分類任務；切換僅是寫入約 26 KB 的 per-task 參數（≈1.86 ms）。
3. **RC（Residual Correction）的 1-bit 關鍵性發現**——down-convolution 的 Int8 bias 吸收進 accumulator reset，於 1-bit 貢獻極大、零硬體成本。

平台：**PYNQ-Z2（Xilinx XC7Z020）**，100 MHz；訓練於 **PyTorch + Brevitas**，硬體合成於 **FINN（Docker）+ Vivado 2022.2**。

---

## 二、資料夾結構（對應論文五大子專案）

| 資料夾 | 內容 | 來源主機 |
|---|---|---|
| [`AI_model_train/`](AI_model_train/) | 1W1A 量化感知訓練（QAT）與 Conv-Adapter 遷移學習的全部程式碼、各資料集 runner、預訓練 backbone、結果 CSV | RTX 4090 + A6000 |
| [`FINN_Compile/`](FINN_Compile/) | FINN end-to-end dataflow 合成（在 Docker 內）：end2end notebook、CNV 模型、匯出之 ONNX | 本地 FINN |
| [`RTL/`](RTL/) | Adapter／改過的 MVAU Super Wrapper／`cfg_hub` 切換控制器的 `.v`，以及打包/建置 `.tcl` | 本地 |
| [`SoC/`](SoC/) | 最頂層 block design（PS + input/output DMA + dataflow partition） | 本地 Vivado |
| [`FPGA/`](FPGA/) | 最終上板的 `.bit`/`.hwh`、driver、以及所有 runtime 參數（四種 build） | PYNQ-Z2 板 + 本地 |

每個資料夾內都有獨立的 `README.md` 做檔案層級說明。

---

## 三、端到端流程（5 個階段）

```
[1] 訓練 (AI_model_train)
    1W1A CNV backbone 預訓練 → 凍結 → 每個 target dataset 接 Conv-Adapter 遷移
        │  輸出：backbone .tar、adapter checkpoint、結果 CSV
        ▼
[2] 編譯 (FINN_Compile)   ← 在 FINN Docker 內
    Brevitas 匯出 ONNX → FINN streamline/折疊(fold)/dataflow 合成
        │  輸出：per-MVAU memblock .dat、StreamingDataflowPartition IP
        ▼
[3] RTL 整合 (RTL)
    每個 MVAU 外包一層 Adapter「Super Wrapper」；加入 cfg_hub；打包成 Vivado IP
        ▼
[4] SoC 縫合 (SoC)
    block design 接上 Zynq PS + input/output DMA + 縫合後的 dataflow partition → bitstream
        ▼
[5] 部署 (FPGA)
    .bit/.hwh + Python driver 上 PYNQ-Z2；runtime 換任務 = 透過 cfg_hub
    寫入該任務約 26 KB 參數（≈1.86 ms，無 fabric reconfiguration）
```

---

## 四、四種 FPGA build（對應論文資源/功耗/跨平台表）

| Build | 資料夾 | Bitstream | 用途 |
|---|---|---|---|
| Backbone, throughput（high-PE） | `FPGA/backbone_throughput/` | `resizer.bit` | 吞吐量基準（純 backbone） |
| MARS, throughput, 2-dataset（high-PE） | `FPGA/MARS_throughput_2ds/` | `resizer_v1.bit` | 能效/吞吐量代表組態 |
| Backbone, compact（PE=1） | `FPGA/backbone_compact_pe1/` | `resizer.bit` | compact 基準（純 backbone） |
| MARS, compact, 5-dataset（PE=1） | `FPGA/MARS_compact_5ds_pe1/` | `resizer_3ds_v3.bit` | 板上精度正確性 + 5 資料集 runtime 切換 |

> **PE** = Processing Element（MVAU 的平行折疊度）。`throughput` 採異質 per-layer PE 折疊（吞吐量高）；`compact` 採均勻 PE=1（換取 LUT 餘裕以容納多任務切換狀態）。

---

## 五、開發環境

| 階段 | 工具 |
|---|---|
| 訓練 | Python 3.8、PyTorch、Brevitas 0.12（1W1A QAT） |
| 合成 | FINN v0.9（Docker）、Vivado 2022.2 |
| 板端 | PYNQ-Z2（XC7Z020）、PYNQ runtime、100 MHz |

---

## 六、5 個資料集

| Dataset | 類別 | Train/Test | 原生解析度 | Modality | 領域 |
|---|---|---|---|---|---|
| CIFAR-10 | 10 | 50,000 / 10,000 | 32×32 | RGB | 一般自然物件（**預設 backbone**） |
| SVHN | 10 | 73,257 / 26,032 | 32×32 | RGB | 街景門牌數字 |
| STL10 | 10 | 5,000 / 8,000 | 96×96 | RGB | ImageNet 衍生（降採樣） |
| FashionMNIST | 10 | 60,000 / 10,000 | 28×28 | 灰階 | 服飾（複製通道） |
| CINIC10 | 10 | 90,000 / 90,000 | 32×32 | RGB | CIFAR-10 + 下採樣 ImageNet（總計 270k = 90k×3） |

全部統一調整成 **32×32×3** 輸入。

---

## 七、注意事項

- **大檔**：backbone（`*.tar`）、bitstream（`*.bit`/`*.hwh`）、ONNX、`*.npy`、`*.dat`、`*.bin` 皆為一般 git 物件（最大單檔 18.7 MB，皆在 GitHub 100 MB 限制內，故未使用 Git LFS）。
- **未收錄**：原始逐 epoch 訓練 log（約 6.4 GB）不放；數據以各 `results/*.csv` 與 `results/final_accuracy_summary.txt` 為準。Vivado 工程與 FINN `code_gen` 中間產物（數十 GB）不放，僅保留可重現所需的原始碼與最終產物。
- **資料集測試輸入**（如各 build 的 `*_test_x.npy`，約 30 MB）已排除，可由 `AI_model_train` 重新產生。
