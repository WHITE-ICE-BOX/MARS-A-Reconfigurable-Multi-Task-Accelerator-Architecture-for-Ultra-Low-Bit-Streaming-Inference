# FINN_Compile — FINN end-to-end dataflow 合成

在 **FINN（Docker）** 內，把訓練好的 1W1A CNV backbone 從 Brevitas 匯出成 ONNX，再經 FINN 的
**streamline → 轉硬體層 → 折疊（folding）→ dataflow 合成**，產出每個 MVAU 的 memblock 與
**StreamingDataflowPartition IP**，作為 `RTL/` 整合 Adapter 的輸入。

> **MVAU** = Matrix-Vector-Activation Unit（FINN 的卷積/全連接運算單元）。
> **folding / PE** = 平行折疊度；PE 越大越快、面積越大。throughput build 用異質高 PE，compact build 用 PE=1。

---

## 一、結構樹

```
FINN_Compile/
├── notebooks/                       # ── FINN end-to-end 流程（Jupyter）──
│   ├── pe1_cnv_end2end.ipynb        #  ★ 最終 compact(PE=1) 硬體的 backbone end2end（5資料集 build 用）
│   ├── pe1_adapter.ipynb            #  ★ 最終 compact(PE=1) 的 adapter 側 end2end
│   ├── cnv_end2end_example.ipynb    #    high-PE（throughput build）的 backbone end2end
│   ├── adapter.ipynb                #    high-PE 的 adapter 側 end2end
│   └── backbone_cifar.ipynb         #    CIFAR-10 backbone 匯出/驗證流程
├── model/                           # ── FINN 的輸入模型 ──
│   ├── CNV.py                       #    Brevitas CNV 定義（匯出成 ONNX 的來源）
│   └── cnv_6layer_fc3_cifar_w1a1.zip#    訓練好的 CIFAR-10 1W1A 模型（end2end 載入這個）
├── scripts/                         # ── 輔助腳本 ──
│   ├── pe1_refold_from_v1.py        #  ★ 把已驗證的 PE=32 dataflow model 重折疊(refold)成 PE=1
│   ├── verify_finn_stages.py        #    驗證 FINN 各階段 ONNX 輸出正確性
│   ├── verify_cifar1w1a.py          #    CIFAR-10 1W1A 端到端正確性檢查
│   ├── validate_custom.py           #    自訂資料集驗證
│   └── dump_canonical_cifar.py      #    匯出標準 CIFAR-10 測資供比對
└── onnx/                            # ── FINN 合成的關鍵中間/輸出 ONNX ──
    ├── end2end_cnv_w1a1_dataflow_model.onnx  #  dataflow 階段（含折疊資訊）
    └── end2end_cnv_w1a1_folded.onnx          #  折疊後模型（合成前最終形態）
```

★ = 產出論文最終硬體所用的關鍵檔。

---

## 二、「最終硬體」是用哪個 end2end？

| 最終 build | end2end 檔 | 折疊 |
|---|---|---|
| **compact, PE=1（5 資料集 runtime 切換）** | `notebooks/pe1_cnv_end2end.ipynb` + `notebooks/pe1_adapter.ipynb` + `scripts/pe1_refold_from_v1.py` | 均勻 PE=1 |
| **throughput, high-PE（2 資料集，能效代表）** | `notebooks/cnv_end2end_example.ipynb` + `notebooks/adapter.ipynb` | 異質 per-layer PE |

`pe1_refold_from_v1.py` 的做法：取 throughput build 已驗證的 dataflow model（PE=32），跳過會出錯的
streamline，直接把每層 MVAU 重折疊成 PE=1，產生 compact build 的折疊模型。

---

## 三、完整 FINN 階段（ONNX 命名對應）

```
export → tidy → streamlined → (convert to HW layers) → folded → dataflow_model → synth
```
本資料夾只保留 `dataflow_model` 與 `folded` 兩個關鍵 ONNX（其餘中間檔與 Vivado `code_gen`
產物數十 GB，未收錄）。完整重跑請在 FINN Docker 內依 `notebooks/` 執行。

---

## 四、與 RTL 的銜接

FINN 產出每個 MVAU 的 memblock `.dat` 與 StreamingDataflowPartition IP 之後，`RTL/` 會把
MVAU1–5 各包一層 Adapter「Super Wrapper」、加入 `cfg_hub`，再以 `RTL/tcl/` 打包成 Vivado IP，
最後由 `SoC/` 縫合成 bitstream。
