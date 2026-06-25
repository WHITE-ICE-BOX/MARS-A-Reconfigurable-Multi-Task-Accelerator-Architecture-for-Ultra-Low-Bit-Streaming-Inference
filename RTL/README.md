# RTL — Adapter / Super Wrapper / cfg_hub 硬體描述

自研 RTL，把 FINN 產生的每個 **MVAU** 包成「**Super Wrapper**」（MVAU 主幹 + Adapter 旁路融合），
並提供 **`cfg_hub`** 執行時期多任務切換控制器。共 5 個 MVAU 帶 Adapter（MVAU1–5），全部 1W1A。

> 命名：論文中「Super Wrapper」對應 5 個帶 Adapter 的卷積層（MVAU1–5）。MVAU0/FC 層無 Adapter。

---

## 一、結構樹

```
RTL/
├── adapter/                         # ── Adapter 資料路徑（1×1 down → sign → 1×1 up + RC）──
│   ├── Adapter_MVAU1.v              #  MVAU1 的 Adapter（4-stage pipeline：latch/ROM→XNOR→popcount→accumulate）
│   ├── Adapter_MVAU2.v             #  MVAU2 Adapter
│   ├── Adapter_MVAU3.v             #  MVAU3 Adapter
│   ├── Adapter_MVAU4.v             #  MVAU4 Adapter
│   └── Adapter_MVAU5.v             #  MVAU5 的 Adapter（泛用/參數化版，原名 Adapter_Generic.v）
│
├── super_wrapper/                   # ── 每個 MVAU 的 Super Wrapper 與其子模組 ──
│   ├── MVAU{1..5}_Super_Wrapper.v   #  頂層：Splitter→(MVAU主幹 ‖ Adapter)→FIFO→Adder+Threshold
│   ├── Stream_Splitter_mvau{N}.v    #  把輸入串流複製給 MVAU 主幹(Path A)與 Adapter(Path B)
│   ├── Simple_FIFO_mvau{N}.v        #  深度 4096 同步 FIFO，吸收兩路延遲差、確保 cycle 對齊
│   ├── Stream_Adder_Threshold_mvau{N}.v # 把 Adapter 貢獻量與 MVAU partial-sum 相加後做 Q8 閾值二值化
│   └── Stream_Adder_mvau{N}.v       #  純加法版（部分 MVAU 用）
│
├── cfg_hub/
│   └── adapter_cfg_hub.v            #  ★ AXI-Lite configuration hub（base 0x40010000，僅 19 LUT）
│                                    #    把 per-task 參數（thresholds、classifier 權重、5 層 adapter blob）
│                                    #    demux 到散落於 pipeline 的暫存器/RAM bank → 控制器無關的 runtime 切換
│
└── tcl/                             # ── Vivado 自動化腳本 ──
    ├── package_ips.tcl              #  把 5 個 MVAU+Adapter 打包成 Vivado IP
    ├── package_mvau1234.tcl         #  打包 MVAU1–4
    ├── package_mvau5_only.tcl       #  打包 MVAU5
    ├── make_project.tcl             #  建立 Vivado 工程
    └── build_bitstream.tcl          #  完整建置流程（stitch → zynq → bitstream）
```

★ = 對應論文核心貢獻「單一 bitstream、控制器無關的多任務切換」。

---

## 二、Adapter 資料路徑（4-stage pipeline）

```
輸入 → [S0] Input Latch + ROM Read（讀 down-proj 權重）
     → [S1] XNOR（1-bit 乘法）
     → [S2] Popcount（累計 +1/−1）
     → [S3] Accumulate（加上 RC = Int8 bias 作為 accumulator reset 初值）
     → Sign Extract（產生二值 hidden activation）
     → up-proj → Adapter 貢獻量
```
RC（Residual Correction）不需額外加法器/DSP/pipeline stage——只改 accumulator 的 reset 值。

---

## 三、Super Wrapper 資料流

```
        ┌──────────────► MVAU Core（FINN 原始，輸出整數 partial-sum）──┐
輸入 ──► Stream_Splitter                                               ├─► Stream_Adder_Threshold ─► 二值輸出
        └──► Adapter Core ─► (Simple_FIFO 對齊延遲) ─────────────────►─┘
```
`Stream_Adder_Threshold` 用合成期預計算的 contribution LUT 做縮放，**零 DSP**。

---

## 四、建置順序

1. `tcl/package_ips.tcl`（或 `package_mvau1234.tcl` + `package_mvau5_only.tcl`）打包 IP。
2. `tcl/make_project.tcl` 建工程、加入 `cfg_hub`。
3. `tcl/build_bitstream.tcl` 縫合並產生 bitstream（SoC 階段見 `../SoC/`）。
