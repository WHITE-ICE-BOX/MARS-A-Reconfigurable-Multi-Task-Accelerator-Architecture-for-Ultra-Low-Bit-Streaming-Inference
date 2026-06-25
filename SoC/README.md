# SoC

Top-level Zynq integration (Vivado block design): PS + input/output DMA + the stitched
StreamingDataflowPartition.

- `block_design/top.bd` — block design (Zynq PS, IDMA, dataflow partition, ODMA)
- `block_design/top_wrapper.v`, `top_synth.v` — generated top
- `dma/top_idma0_0_stub.v`, `top_odma0_0_stub.v` — DMA IP interfaces
- `dma/IODMA_hls_0.v` — FINN IODMA controller (HLS-generated Verilog)
