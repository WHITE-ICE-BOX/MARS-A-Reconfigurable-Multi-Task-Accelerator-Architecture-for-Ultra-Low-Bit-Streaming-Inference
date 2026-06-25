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
│
├── mvau_core/                       # ── FINN 生成、本論文修改過的 MVAU 核心（MVAU0–8）──
│   ├── mvau1/ .. mvau5/             #  含 adapter 的卷積層：Matrix_Vector_Activate、memstream、
│   │                                #    threshold ROM、MVAU_hls 頂層…+ memblock.dat
│   └── mvau0/ mvau6/ mvau7/ mvau8/  #  FC/無 adapter 層（修改點：threshold/memstream 可由 cfg_hub 寫）
│                                    #  修改重點：MVAU 輸出『整數 partial-sum』供 Adapter 融合（非直接二值化）
│
├── hardware_assets/                 # ── RTL 消費的 .dat 權重（97 個）──
│   ├── mvau_{0..8}_memblock.dat     #  各 MVAU 的 FINN 二值權重 memblock
│   ├── adapter_{1..5}_{down,up,rc,alpha}.dat   #  各層 adapter 權重
│   └── *_contrib_lut.dat / thresh   #  contribution LUT、Q8 thresholds…
│
└── gen_scripts/                     # ── PyTorch 參數 → .dat/ROM 轉換腳本 ──
    ├── final_sw.py                  #  ★ 主匯出：載入 checkpoint → 產生全部 hardware_assets/*.dat（PE=1）
    ├── export_for_fpga.py           #  通用版匯出器
    ├── prepack_adapter_dat.py       #  把 adapter .dat 打包成 $readmemh 寬格式
    ├── make_adp_contrib_luts.py     #  產生 contribution LUT
    ├── generate_thresh_q8_mvau5.py  #  產生 MVAU5 Q8 threshold ROM
    ├── generate_mvau1234_golden.py / generate_mvau5_hw_aligned.py  #  golden / hw-aligned 資產
    ├── fix_rc_dat.py / gen_npy_golden.py   #  工具
    └── RC_m1_full.tar               #  final_sw.py 載入的 HW 匯出 checkpoint（deployed M1 + RC）
```

★ = 對應論文核心貢獻「單一 bitstream、控制器無關的多任務切換」。

> **PyTorch → 硬體流程**：訓練好的 checkpoint（`gen_scripts/RC_m1_full.tar`）經
> `final_sw.py` 轉成 `hardware_assets/*.dat`（FINN memblock + adapter 權重 + LUT/threshold），
> 由 `mvau_core/` 的 RTL 以 `$readmemh` 載入；`prepack_adapter_dat.py` 負責把 adapter 權重
> 打包成 ROM 寬格式。板端 runtime 切換則改用 `FPGA/.../runtime_weights/*.bin`（由相同來源產生）。

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
