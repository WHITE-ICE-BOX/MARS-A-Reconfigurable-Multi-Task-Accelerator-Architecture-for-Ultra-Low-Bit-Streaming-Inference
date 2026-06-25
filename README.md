# MARS — A Reconfigurable Multi-Task Accelerator Architecture for Ultra-Low-Bit Streaming Inference

Final code + data release for the thesis. MARS integrates Conv-Adapters into a
frozen 1W1A (1-bit weight / 1-bit activation) FINN streaming-dataflow CNV backbone,
with an AXI-Lite `cfg_hub` for **controller-free runtime multi-task switching** on a
**single bitstream** at O(1) on-chip cost.

## Repository structure

| Folder | Contents | Origin |
|---|---|---|
| `AI_model_train/` | PyTorch 1W1A training + transfer-learning code, per-dataset runners, pretrained backbones (LFS), result CSVs | RTX 4090 + A6000 |
| `FINN_Compile/` | FINN end-to-end dataflow synthesis (Docker): end2end notebooks, CNV model, exported ONNX | local FINN |
| `RTL/` | Adapter / modified-MVAU Super Wrapper / `cfg_hub` function-switch controller `.v` + packaging/build `.tcl` | local |
| `SoC/` | Top-level block design (PS + DMA): input/output DMA, top wrapper | local Vivado |
| `FPGA/` | Final on-board `.bit`/`.hwh` + drivers + runtime params for the 4 thesis builds | PYNQ-Z2 board + local |

## The four FPGA builds (thesis Table 5.11 / 5.12)

| Build | Folder | Bitstream |
|---|---|---|
| Backbone, throughput (high-PE) | `FPGA/backbone_throughput/` | `resizer.bit` |
| MARS, throughput, 2-dataset (high-PE) | `FPGA/MARS_throughput_2ds/` | `resizer_v1.bit` |
| Backbone, compact (PE=1) | `FPGA/backbone_compact_pe1/` | `resizer.bit` |
| MARS, compact, 5-dataset (PE=1) | `FPGA/MARS_compact_5ds_pe1/` | `resizer_3ds_v3.bit` |

## End-to-end flow

1. **Train** (`AI_model_train/`): 1W1A CNV backbone pre-train → freeze → Conv-Adapter
   transfer per target dataset (CIFAR-10, SVHN, STL10, FashionMNIST, CINIC10).
2. **Compile** (`FINN_Compile/`): export backbone to ONNX → FINN dataflow synthesis
   → per-MVAU memblocks + StreamingDataflowPartition IP.
3. **Integrate** (`RTL/`): wrap each MVAU with the Adapter Super Wrapper; add the
   `cfg_hub` runtime-switch controller; package as Vivado IPs (`.tcl`).
4. **Stitch** (`SoC/`): block design connects Zynq PS + input/output DMA + the
   stitched dataflow partition.
5. **Deploy** (`FPGA/`): `.bit`/`.hwh` + Python driver run on PYNQ-Z2; runtime task
   switch = a ~26 KB cfg blob written over MMIO (~1.86 ms, no fabric reconfiguration).

## Git LFS

Large binaries (`*.tar` checkpoints, `*.bit`, `*.hwh`, `*.onnx`, `*.npy`, `*.dat`,
`*.zip`) are tracked with Git LFS — run `git lfs install` before cloning.

## Notes
- Raw per-epoch training logs (~6.4 GB) are **not** included; `results/*.csv` and
  `results/final_accuracy_summary.txt` carry the reported numbers.
- See each folder's `README.md` for details.
