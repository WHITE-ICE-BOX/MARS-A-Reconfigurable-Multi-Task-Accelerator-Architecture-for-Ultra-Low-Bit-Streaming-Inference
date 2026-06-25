// ===========================================================================
// [交接導向註解]
// 模組：MVAU4 同步 FIFO（深度4096），吸收 MVAU 與 Adapter 延遲差、確保 cycle 對齊。流程：RTL。
// ===========================================================================

`timescale 1ns / 1ps
module Simple_FIFO_mvau4 #(parameter WIDTH = 8, parameter DEPTH_LOG2 = 10)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire [WIDTH-1:0] s_axis_tdata,
    input  wire             s_axis_tvalid,
    output wire             s_axis_tready,
    output wire [WIDTH-1:0] m_axis_tdata,
    output wire             m_axis_tvalid,
    input  wire             m_axis_tready
);
    localparam DEPTH = 1 << DEPTH_LOG2;
    (* ram_style = "distributed" *) reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [DEPTH_LOG2-1:0] wr_ptr, rd_ptr;
    reg [DEPTH_LOG2:0]   count;
    wire write_en = s_axis_tvalid && s_axis_tready;
    wire read_en  = m_axis_tvalid && m_axis_tready;

    assign s_axis_tready = (count < DEPTH);
    assign m_axis_tvalid = (count > 0);
    assign m_axis_tdata  = mem[rd_ptr];

    always @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= 0; rd_ptr <= 0; count <= 0;
        end else begin
            if (write_en) begin
                mem[wr_ptr] <= s_axis_tdata;
                wr_ptr <= wr_ptr + 1;
            end
            if (read_en) rd_ptr <= rd_ptr + 1;
            case ({write_en, read_en})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;
            endcase
        end
    end
endmodule
