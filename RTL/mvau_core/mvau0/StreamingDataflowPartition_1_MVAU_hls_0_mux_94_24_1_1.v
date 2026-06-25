// ===========================================================================
// [交接導向註解]
// MVAU0 — Conv0（layer 0，第一層卷積，無 Adapter）
// 改造：threshs_ROM 由唯讀改為 cfg-可寫（64×11-bit），runtime 換任務時由 cfg_hub 改 MVAU0 threshold。
// 
// 本檔：
//   FINN HLS 介面多工/暫存切片（自動生成，未改）。
// 
// 流程：FINN_Compile 產生 → 本論文修改 → RTL/super_wrapper 整合 → SoC 縫合 → FPGA。
// ===========================================================================

// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================

`timescale 1ns/1ps

module StreamingDataflowPartition_1_MVAU_hls_0_mux_94_24_1_1 #(
parameter
    ID                = 0,
    NUM_STAGE         = 1,
    din0_WIDTH       = 32,
    din1_WIDTH       = 32,
    din2_WIDTH       = 32,
    din3_WIDTH       = 32,
    din4_WIDTH       = 32,
    din5_WIDTH       = 32,
    din6_WIDTH       = 32,
    din7_WIDTH       = 32,
    din8_WIDTH       = 32,
    din9_WIDTH         = 32,
    dout_WIDTH            = 32
)(
    input  [23 : 0]     din0,
    input  [23 : 0]     din1,
    input  [23 : 0]     din2,
    input  [23 : 0]     din3,
    input  [23 : 0]     din4,
    input  [23 : 0]     din5,
    input  [23 : 0]     din6,
    input  [23 : 0]     din7,
    input  [23 : 0]     din8,
    input  [3 : 0]    din9,
    output [23 : 0]   dout);

// puts internal signals
wire [3 : 0]     sel;
// level 1 signals
wire [23 : 0]         mux_1_0;
wire [23 : 0]         mux_1_1;
wire [23 : 0]         mux_1_2;
wire [23 : 0]         mux_1_3;
wire [23 : 0]         mux_1_4;
// level 2 signals
wire [23 : 0]         mux_2_0;
wire [23 : 0]         mux_2_1;
wire [23 : 0]         mux_2_2;
// level 3 signals
wire [23 : 0]         mux_3_0;
wire [23 : 0]         mux_3_1;
// level 4 signals
wire [23 : 0]         mux_4_0;

assign sel = din9;

// Generate level 1 logic
assign mux_1_0 = (sel[0] == 0)? din0 : din1;
assign mux_1_1 = (sel[0] == 0)? din2 : din3;
assign mux_1_2 = (sel[0] == 0)? din4 : din5;
assign mux_1_3 = (sel[0] == 0)? din6 : din7;
assign mux_1_4 = din8;

// Generate level 2 logic
assign mux_2_0 = (sel[1] == 0)? mux_1_0 : mux_1_1;
assign mux_2_1 = (sel[1] == 0)? mux_1_2 : mux_1_3;
assign mux_2_2 = mux_1_4;

// Generate level 3 logic
assign mux_3_0 = (sel[2] == 0)? mux_2_0 : mux_2_1;
assign mux_3_1 = mux_2_2;

// Generate level 4 logic
assign mux_4_0 = (sel[3] == 0)? mux_3_0 : mux_3_1;

// output logic
assign dout = mux_4_0;

endmodule
