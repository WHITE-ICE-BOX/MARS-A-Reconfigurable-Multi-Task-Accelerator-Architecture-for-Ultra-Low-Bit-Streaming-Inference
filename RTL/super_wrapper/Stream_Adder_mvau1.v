// ===========================================================================
// [交接導向註解]
// 模組：MVAU1 純加法子模組（不含閾值）。流程：RTL。
// ===========================================================================

`timescale 1ns / 1ps
module Stream_Adder_mvau1 #(parameter DATA_WIDTH = 8)(
    input  wire                  aclk,
    input  wire                  aresetn,
    input  wire [DATA_WIDTH-1:0] s_axis_1_tdata,
    input  wire                  s_axis_1_tvalid,
    output wire                  s_axis_1_tready,
    input  wire [DATA_WIDTH-1:0] s_axis_2_tdata,
    input  wire                  s_axis_2_tvalid,
    output wire                  s_axis_2_tready,
    output wire [DATA_WIDTH-1:0] m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready
);
    wire both_valid = s_axis_1_tvalid && s_axis_2_tvalid;
    // 等對方也 Valid 且下游 Ready，才拉高自己的 Ready
    assign s_axis_1_tready = m_axis_tready && s_axis_2_tvalid;
    assign s_axis_2_tready = m_axis_tready && s_axis_1_tvalid;
    assign m_axis_tvalid = both_valid;
    assign m_axis_tdata = s_axis_1_tdata + s_axis_2_tdata;
endmodule
