// ===========================================================================
// [交接導向註解]
// MVAU5 — Conv5（帶 Adapter）
// 改造：同 Conv1，輸出整數 partial-sum 供 Adapter 融合。
// 
// 本檔：
//   threshold ROM bank（FINN 生成）。
// 
// 流程：FINN_Compile 產生 → 本論文修改 → RTL/super_wrapper 整合 → SoC 縫合 → FPGA。
// ===========================================================================

// ==============================================================
// FROZEN canonical MVAU5 hand-modified backbone (32-bit raw out).
// In this 32-bit raw mode the threshold ROM is unused — $readmemh is
// intentionally commented out. To enable thresholds in a future app:
// uncomment the $readmemh line below and override MVAU5_THRESH_FILE via:
//     -d "MVAU5_THRESH_FILE=\"path/to/thresholds.dat\""
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Version: 2022.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps

`ifndef MVAU5_THRESH_FILE
  `define MVAU5_THRESH_FILE "/home/barkie1/mvau_pipeline/sw/hardware_assets_pe_simd_aligned_new/StreamingDataflowPartition_1_MVAU_hls_5_Matrix_Vector_Activate_Stream_Batch_threshs_ROM_AUTO_1R.dat"
`endif
module StreamingDataflowPartition_1_MVAU_hls_5_Matrix_Vector_Activate_Stream_Batch_threshs_ROM_AUTO_1R (
    address0, ce0, q0, 
    reset, clk);

parameter DataWidth = 11;
parameter AddressWidth = 8;
parameter AddressRange = 256;
 
input[AddressWidth-1:0] address0;
input ce0;
output reg[DataWidth-1:0] q0;

input reset;
input clk;

 
reg [DataWidth-1:0] rom0[0:AddressRange-1];


initial begin
    $readmemh(`MVAU5_THRESH_FILE, rom0);
end

  
always @(posedge clk) 
begin 
    if (ce0) 
    begin
        q0 <= rom0[address0];
    end
end


endmodule

