`timescale 1ns / 1ps
module Stream_Splitter_mvau1 #(parameter DATA_WIDTH = 32)(
    input  wire                  aclk,
    input  wire                  aresetn,
    input  wire [DATA_WIDTH-1:0] s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,
    output wire [DATA_WIDTH-1:0] m_axis_1_tdata,
    output wire                  m_axis_1_tvalid,
    input  wire                  m_axis_1_tready,
    output wire [DATA_WIDTH-1:0] m_axis_2_tdata,
    output wire                  m_axis_2_tvalid,
    input  wire                  m_axis_2_tready
);
    // 兩邊都 Ready 才讓上游灌資料，防止資料重複被吃
    assign s_axis_tready = m_axis_1_tready && m_axis_2_tready;
    assign m_axis_1_tdata = s_axis_tdata;
    assign m_axis_2_tdata = s_axis_tdata;
    assign m_axis_1_tvalid = s_axis_tvalid && s_axis_tready;
    assign m_axis_2_tvalid = s_axis_tvalid && s_axis_tready;
endmodule
