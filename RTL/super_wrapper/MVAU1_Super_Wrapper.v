// ===========================================================================
// [交接導向註解]
// 模組：第 1 層 Super Wrapper（頂層）。流程：RTL，取代 FINN 預設 MVAU。
// 資料流：Stream_Splitter -> (MVAU主幹 ‖ Adapter) -> Simple_FIFO 對齊 -> Stream_Adder_Threshold。
// ===========================================================================

`timescale 1ns / 1ps

module MVAU1_Super_Wrapper (
    input  wire        ap_clk,
    input  wire        ap_rst_n,
    
    input  wire [31:0] in0_V_TDATA,
    input  wire        in0_V_TVALID,
    output wire        in0_V_TREADY,
    
    input  wire [1023:0] weights_V_TDATA,  // ✨ 1024-bit 權重
    input  wire          weights_V_TVALID,
    output wire          weights_V_TREADY,
    
    output wire [31:0] out_V_TDATA,        // ✨ 32-bit 輸出
    output wire        out_V_TVALID,
    input  wire        out_V_TREADY
);

    wire [31:0]  split_to_mvau_tdata, split_to_adapt_tdata;
    wire         split_to_mvau_tvalid, split_to_adapt_tvalid;
    wire         split_to_mvau_tready, split_to_adapt_tready;

    wire [1023:0] mvau_to_add_tdata; // ✨ 1024-bit MAC
    wire          mvau_to_add_tvalid, mvau_to_add_tready;

    wire [255:0] adapt_to_fifo_tdata, fifo_to_add_tdata; // ✨ 256-bit Popcount
    wire         adapt_to_fifo_tvalid, fifo_to_add_tvalid;
    wire         adapt_to_fifo_tready, fifo_to_add_tready;

    Stream_Splitter_mvau1 #(.DATA_WIDTH(32)) splitter_inst (
        .aclk(ap_clk), .aresetn(ap_rst_n),
        .s_axis_tdata(in0_V_TDATA), .s_axis_tvalid(in0_V_TVALID), .s_axis_tready(in0_V_TREADY),
        .m_axis_1_tdata(split_to_mvau_tdata), .m_axis_1_tvalid(split_to_mvau_tvalid), .m_axis_1_tready(split_to_mvau_tready),
        .m_axis_2_tdata(split_to_adapt_tdata), .m_axis_2_tvalid(split_to_adapt_tvalid), .m_axis_2_tready(split_to_adapt_tready)
    );

    StreamingDataflowPartition_1_StreamingDataflowPartition_1_MVAU_hls_1_0 mvau_inst (
        .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
        .in0_V_TVALID(split_to_mvau_tvalid), .in0_V_TREADY(split_to_mvau_tready), .in0_V_TDATA(split_to_mvau_tdata),
        .weights_V_TVALID(weights_V_TVALID), .weights_V_TREADY(weights_V_TREADY), .weights_V_TDATA(weights_V_TDATA),
        .out_V_TVALID(mvau_to_add_tvalid), .out_V_TREADY(mvau_to_add_tready), .out_V_TDATA(mvau_to_add_tdata)
    );

    Adapter_MVAU1 adapter_inst (
        .aclk(ap_clk), .aresetn(ap_rst_n),
        .s_axis_tdata(split_to_adapt_tdata), .s_axis_tvalid(split_to_adapt_tvalid), .s_axis_tready(split_to_adapt_tready),
        .m_axis_tdata(adapt_to_fifo_tdata), .m_axis_tvalid(adapt_to_fifo_tvalid), .m_axis_tready(adapt_to_fifo_tready)
    );

    Simple_FIFO_mvau1 #(.WIDTH(256), .DEPTH_LOG2(6)) fifo_inst (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .s_axis_tdata(adapt_to_fifo_tdata), .s_axis_tvalid(adapt_to_fifo_tvalid), .s_axis_tready(adapt_to_fifo_tready),
        .m_axis_tdata(fifo_to_add_tdata), .m_axis_tvalid(fifo_to_add_tvalid), .m_axis_tready(fifo_to_add_tready)
    );

    Stream_Adder_Threshold_mvau1 adder_thresh_inst (
        .aclk(ap_clk), .aresetn(ap_rst_n),
        .s_axis_1_tdata(mvau_to_add_tdata), .s_axis_1_tvalid(mvau_to_add_tvalid), .s_axis_1_tready(mvau_to_add_tready),
        .s_axis_2_tdata(fifo_to_add_tdata), .s_axis_2_tvalid(fifo_to_add_tvalid), .s_axis_2_tready(fifo_to_add_tready),
        .m_axis_tdata(out_V_TDATA), .m_axis_tvalid(out_V_TVALID), .m_axis_tready(out_V_TREADY)
    );

endmodule
