// ===========================================================================
// [交接導向註解]
// 模組：第 5 層 Super Wrapper（頂層）。流程：RTL，取代 FINN 預設 MVAU。
// 資料流：Stream_Splitter -> (MVAU主幹 ‖ Adapter) -> Simple_FIFO 對齊 -> Stream_Adder_Threshold。
// ===========================================================================

`timescale 1ns / 1ps

module MVAU5_Super_Wrapper (
    input  wire        ap_clk,
    input  wire        ap_rst_n,
    
    // 1. 特徵圖輸入 (來自上一層)
    input  wire [31:0] in0_V_TDATA,
    input  wire        in0_V_TVALID,
    output wire        in0_V_TREADY,
    
    // 2. 權重輸入 (來自 wstrm)
    input  wire [31:0] weights_V_TDATA,
    input  wire        weights_V_TVALID,
    output wire        weights_V_TREADY,
    
    // 3. 最終輸出 (去往下一層)
    output wire [7:0]  out_V_TDATA,
    output wire        out_V_TVALID,
    input  wire        out_V_TREADY
);

    // ==========================================
    // 內部接線宣告 (Internal Wires)
    // ==========================================
    wire [31:0] split_to_mvau_tdata;
    wire        split_to_mvau_tvalid;
    wire        split_to_mvau_tready;

    wire [31:0] split_to_adapt_tdata;
    wire        split_to_adapt_tvalid;
    wire        split_to_adapt_tready;

    wire [31:0] mvau_to_add_tdata;
    wire        mvau_to_add_tvalid;
    wire        mvau_to_add_tready;

    wire [7:0]  adapt_to_fifo_tdata;
    wire        adapt_to_fifo_tvalid;
    wire        adapt_to_fifo_tready;

    wire [7:0]  fifo_to_add_tdata;
    wire        fifo_to_add_tvalid;
    wire        fifo_to_add_tready;

    // ==========================================
    // 元件實例化 (Instantiations)
    // ==========================================
    
    // 1. 分流器
    Stream_Splitter_mvau5 #(.DATA_WIDTH(32)) splitter_inst (
        .aclk(ap_clk), .aresetn(ap_rst_n),
        .s_axis_tdata(in0_V_TDATA), .s_axis_tvalid(in0_V_TVALID), .s_axis_tready(in0_V_TREADY),
        .m_axis_1_tdata(split_to_mvau_tdata), .m_axis_1_tvalid(split_to_mvau_tvalid), .m_axis_1_tready(split_to_mvau_tready),
        .m_axis_2_tdata(split_to_adapt_tdata), .m_axis_2_tvalid(split_to_adapt_tvalid), .m_axis_2_tready(split_to_adapt_tready)
    );

    // 2. 被我們開過刀的 32-bit MVAU5
    // 注意這裡呼叫的是你修改過輸出為 32-bit 的那個 MVAU 核心
    StreamingDataflowPartition_1_StreamingDataflowPartition_1_MVAU_hls_5_0 mvau_core (
        .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
        .in0_V_TVALID(split_to_mvau_tvalid), .in0_V_TREADY(split_to_mvau_tready), .in0_V_TDATA(split_to_mvau_tdata),
        .weights_V_TVALID(weights_V_TVALID), .weights_V_TREADY(weights_V_TREADY), .weights_V_TDATA(weights_V_TDATA),
        .out_V_TVALID(mvau_to_add_tvalid), .out_V_TREADY(mvau_to_add_tready), .out_V_TDATA(mvau_to_add_tdata)
    );

    // 3. Adapter IP
    Adapter_Generic adapter_inst (
        .aclk(ap_clk), .aresetn(ap_rst_n),
        .s_axis_tdata(split_to_adapt_tdata), .s_axis_tvalid(split_to_adapt_tvalid), .s_axis_tready(split_to_adapt_tready),
        .m_axis_tdata(adapt_to_fifo_tdata), .m_axis_tvalid(adapt_to_fifo_tvalid), .m_axis_tready(adapt_to_fifo_tready)
    );

    // 4. FIFO 緩衝
    Simple_FIFO_mvau5 #(.WIDTH(8), .DEPTH_LOG2(6)) fifo_inst (
        .clk(ap_clk), .rst_n(ap_rst_n),
        .s_axis_tdata(adapt_to_fifo_tdata), .s_axis_tvalid(adapt_to_fifo_tvalid), .s_axis_tready(adapt_to_fifo_tready),
        .m_axis_tdata(fifo_to_add_tdata), .m_axis_tvalid(fifo_to_add_tvalid), .m_axis_tready(fifo_to_add_tready)
    );

    // 5. 智慧相加與閾值判斷
    Stream_Adder_Threshold_mvau5 adder_thresh_inst (
        .aclk(ap_clk), .aresetn(ap_rst_n),
        .s_axis_1_tdata(mvau_to_add_tdata), .s_axis_1_tvalid(mvau_to_add_tvalid), .s_axis_1_tready(mvau_to_add_tready),
        .s_axis_2_tdata(fifo_to_add_tdata), .s_axis_2_tvalid(fifo_to_add_tvalid), .s_axis_2_tready(fifo_to_add_tready),
        .m_axis_tdata(out_V_TDATA), .m_axis_tvalid(out_V_TVALID), .m_axis_tready(out_V_TREADY)
    );

endmodule
