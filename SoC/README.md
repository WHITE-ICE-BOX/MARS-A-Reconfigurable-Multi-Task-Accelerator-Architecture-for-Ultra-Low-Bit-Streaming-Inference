# SoC — 最頂層 Zynq 整合（block design + DMA）

最頂層的 Vivado **block design**：Zynq **PS**（Processing System）+ **input/output DMA** +
縫合後的 **StreamingDataflowPartition**（內含 RTL/ 的 5 個 Super Wrapper + cfg_hub）。
資料從 DRAM 經 IDMA 進入串流資料流，運算後經 ODMA 寫回 DRAM。

---

## 一、結構樹

```
SoC/
├── block_design/
│   ├── top.bd                # ★ Vivado block design：PS + IDMA + dataflow partition + ODMA 的連線定義
│   ├── top_wrapper.v         #   block design 的 HDL wrapper（合成頂層）
│   └── top_synth.v           #   block design 展開後的 synth 頂層 netlist
└── dma/
    ├── top_idma0_0_stub.v    #   Input DMA IP 介面（stub：埠定義）
    ├── top_odma0_0_stub.v    #   Output DMA IP 介面（stub：埠定義）
    └── IODMA_hls_0.v         #   FINN IODMA 控制器本體（HLS 產生的 Verilog：DRAM ↔ 串流的搬運邏輯）
```

★ = 最頂層整合來源；用 Vivado 開啟 `top.bd` 可見完整 block design。

---

## 二、資料流（DRAM ↔ FPGA）

```
DRAM ──(AXI)──► IDMA（Mem2Stream）──► StreamingDataflowPartition
                                         │  MVAU0
                                         │  MVAU1..5 Super Wrapper（含 Adapter）
                                         │  FC1/FC2、Thresholding、ConvInputGen、DWC、FIFO
                                         ▼
                              LabelSelect ──► ODMA（Stream2Mem）──► DRAM
            Zynq PS（ARM）──(AXI-Lite)──► cfg_hub（runtime 寫入 per-task 參數）
```

---

## 三、Block design 階層（參考）

```
top.bd
 ├─ Zynq PS
 ├─ IDMA (top_idma0_0)
 ├─ StreamingDataflowPartition_1   ← 由 RTL/ 的 IP 縫合而成
 │    ├─ mvau1_adapter .. mvau5_adapter（Super Wrapper）
 │    ├─ MVAU_hls_0 / 6 / 7 / 8（無 adapter 的層）
 │    └─ Thresholding / ConvInputGen / DWC / FIFO / LabelSelect
 └─ ODMA (top_odma0_0)
```

> 完整 Vivado 工程（`.xpr`、`.gen`、`.cache`，數十 GB）未收錄；此處保留 block design 與 DMA
> 介面/控制器的關鍵 `.v`。bitstream 由 `RTL/tcl/build_bitstream.tcl` 產生，最終產物在 `../FPGA/`。
