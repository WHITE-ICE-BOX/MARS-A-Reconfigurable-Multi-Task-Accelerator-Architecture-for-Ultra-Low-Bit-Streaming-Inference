# FPGA 上板量測結果（PYNQ-Z2, XC7Z020, 100 MHz）

板上測試（`board_test_10k.py` / `batch_test.py` / `board_switch_fast_bench.py`）為 print 到
stdout，未存檔；此處彙整論文回報之**實測數值**，並附各 build 的 Vivado routed 報告
（`<build>/top_wrapper_{utilization_placed,power_routed,timing_summary_routed}.rpt`）作為
資源／功耗／時序的原始依據。

---

## 一、板上準確率（MARS compact, 5-dataset, PE=1）

各資料集 10,000 張（STL10 為 8,000 張），與軟體同 `/255` NCHW 前處理比較：

| Dataset | 板上 | 軟體參考 | 差距 |
|---|---|---|---|
| CIFAR-10 | 80.99% | 81.46% | 0.47 pp |
| SVHN | 73.21% | 73.72% | 0.51 pp |
| FashionMNIST | 79.98% | 80.42% | 0.44 pp |
| STL10 | 66.77% | 67.46% | 0.69 pp |
| CINIC10 | 64.64% | 64.80% | 0.16 pp |

→ 五者皆 ≤ 0.69 pp（一般 FINN 定點量化損失），單一 bitstream、無 per-dataset 重建。
量測腳本：`MARS_compact_5ds_pe1/board_test_10k.py`（`DATASETS` 已含 5 個）。

> **重現說明**：板上原始快照僅常駐 cifar10/svhn/fashion 三組，故 `board_test_10k.py`
> 原本只跑 3 個（已改為 5 個）。STL10/CINIC10 的 66.77/64.64 為相同 harness 量測；
> 要在板上重跑這兩個，需先把它們的 `runtime_weights/<ds>/`（已在 repo 內）與測試資料
> `<ds>_test_x.npy/_test_y.npy` 上傳到板子。切換機制與 weights 對五個資料集皆完備。

## 二、Runtime 切換延遲

- **1.86 ± 0.04 ms**（50 次切換平均），每次寫 6,757 個 cfg word；blob 實體 25,088 bytes（≈25 KB；classifier 權重為 byte-packed，故非 6,757×4）。
- 無 fabric reconfiguration、無 reconfiguration controller。
- 量測腳本：`MARS_compact_5ds_pe1/board_switch_fast_bench.py`。

## 三、吞吐量 / 延遲 / 能效

| 指標 | MARS throughput (2ds, high-PE) | Backbone throughput | MARS compact (PE=1) |
|---|---|---|---|
| 端到端吞吐量 | 1,866 img/s | 1,859 img/s | ~107 img/s |
| 單張延遲 | 0.54 ms | — | — |
| 晶片功耗 | 2.214 W | — | 1.802 W |
| 能效 | 842.7 img/s/W | 879.4 img/s/W | 59.4 img/s/W |

- 跨平台：MARS throughput 842.7 img/s/W = **RTX 4090 的 11.9×**、**Jetson Orin NX 的 5.9×**
  （統一 batch-1000、10 次平均協定）。
- 量測腳本：`MARS_throughput_2ds/batch_test.py`。

## 四、資源（XC7Z020, post-route）

| Build | LUT | FF | BRAM | DSP | 時序 (100 MHz) |
|---|---|---|---|---|---|
| MARS compact 5ds (PE=1) | 39,967 | 40,926 | 84 tiles | 0 | WNS +0.055 ns（達成） |

- `cfg_hub` 本體僅 ~19 LUT。詳細數字見各 build 的 `*_utilization_placed.rpt`。

---

## 五、本資料夾結構

```
results/
├── onboard_measurements.md            # 本檔：實測數值彙整
├── backbone_throughput/               # 各 build 的 Vivado routed 報告
│   ├── top_wrapper_utilization_placed.rpt   #   資源使用
│   ├── top_wrapper_power_routed.rpt          #   功耗
│   └── top_wrapper_timing_summary_routed.rpt #   時序
├── backbone_compact_pe1/   ( 同上三份 )
├── MARS_throughput_2ds/    ( 同上三份 )
└── MARS_compact_5ds_pe1/   ( 同上三份 )
```

> 重現方式：上板執行對應 build 的測試腳本（見 `FPGA/README.md` 第四節）；資源/功耗/時序
> 由 Vivado 建置（`RTL/tcl/build_bitstream.tcl`）後在 `impl_1/` 產出對應 `.rpt`。
