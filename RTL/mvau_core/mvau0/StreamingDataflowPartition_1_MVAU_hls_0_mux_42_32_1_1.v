// ===========================================================================
// [交接導向註解]
// MVAU0 核心 RTL（FINN HLS 生成、本論文修改）。FC/無 adapter 層。
// 修改重點：輸出整數 partial-sum（不直接二值化）以供 Adapter 融合；
// memstream/threshold 改為 cfg_hub 可寫，支援 runtime 換任務。
// 流程：FINN_Compile 產生 → 本層修改 → SoC 縫合。
// ===========================================================================

// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================

`timescale 1ns/1ps

module StreamingDataflowPartition_1_MVAU_hls_0_mux_42_32_1_1 #(
parameter
    ID                = 0,
    NUM_STAGE         = 1,
    din0_WIDTH       = 32,
    din1_WIDTH       = 32,
    din2_WIDTH       = 32,
    din3_WIDTH       = 32,
    din4_WIDTH         = 32,
    dout_WIDTH            = 32
)(
    input  [31 : 0]     din0,
    input  [31 : 0]     din1,
    input  [31 : 0]     din2,
    input  [31 : 0]     din3,
    input  [1 : 0]    din4,
    output [31 : 0]   dout);

// puts internal signals
wire [1 : 0]     sel;
// level 1 signals
wire [31 : 0]         mux_1_0;
wire [31 : 0]         mux_1_1;
// level 2 signals
wire [31 : 0]         mux_2_0;

assign sel = din4;

// Generate level 1 logic
assign mux_1_0 = (sel[0] == 0)? din0 : din1;
assign mux_1_1 = (sel[0] == 0)? din2 : din3;

// Generate level 2 logic
assign mux_2_0 = (sel[1] == 0)? mux_1_0 : mux_1_1;

// output logic
assign dout = mux_2_0;

endmodule
