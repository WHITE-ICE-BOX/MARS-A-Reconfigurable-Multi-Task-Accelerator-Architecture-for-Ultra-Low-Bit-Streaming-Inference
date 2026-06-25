// ===========================================================================
// [交接導向註解]
// 模組：cls_cfg_bridge — classifier(MVAU8) 權重的 runtime 寫入橋接。
// 流程：RTL，runtime 切換基礎架構（與 adapter_cfg_hub.v 搭配）。
// 做什麼：adapter_cfg_hub 每個 cls_cfg_* 脈衝帶 8 個打包的二值權重(LSB-first)；
//         本模組把每個脈衝展開成 8 筆對 MVAU8 memstream 的 AXI-Lite 寫入
//         （row = waddr*8 + i），於是 runtime 換任務即可改 classifier 投影權重。
// 這就是『MVAU8(FC) 也有 cfg』的實作——只是用 weight-writable 機制，非 threshold ROM patch。
// ===========================================================================

`timescale 1ns / 1ps

// =====================================================================
// cls_cfg_bridge — adapter_cfg_hub cls_cfg_* → memstream AXI-Lite
// =====================================================================
// One cls_cfg pulse carries 8 packed binary weights (LSB-first) at
// cfg index cls_cfg_waddr[9:0] (0..639). The bridge expands each
// pulse into 8 sequential AXI-Lite writes to memstream s_axilite.
// Each write puts 1 weight bit at memstream row r = waddr*8 + i.
// Memstream byte address = row * 4 (WIDTH=8 stores 1 byte per row,
// addressed by 4-byte word).
// =====================================================================

module cls_cfg_bridge #(
    parameter AXILITE_ADDR_WIDTH = 15  // memstream AXILITE_ADDR_WIDTH (DEPTH=5120,WIDTH=8)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Slave-side cfg pulse from adapter_cfg_hub
    input  wire [9:0]                    cls_cfg_waddr,
    input  wire [7:0]                    cls_cfg_wdata,
    input  wire                          cls_cfg_wen,

    // Master-side AXI-Lite to memstream s_axilite
    output reg                           m_awvalid,
    input  wire                          m_awready,
    output reg  [AXILITE_ADDR_WIDTH-1:0] m_awaddr,
    output wire [2:0]                    m_awprot,

    output reg                           m_wvalid,
    input  wire                          m_wready,
    output reg  [31:0]                   m_wdata,
    output wire [3:0]                    m_wstrb,

    input  wire                          m_bvalid,
    output wire                          m_bready,
    input  wire [1:0]                    m_bresp,

    // Read channel unused — tie off
    output wire                          m_arvalid,
    input  wire                          m_arready,
    output wire [AXILITE_ADDR_WIDTH-1:0] m_araddr,
    output wire [2:0]                    m_arprot,
    input  wire                          m_rvalid,
    output wire                          m_rready,
    input  wire [1:0]                    m_rresp,
    input  wire [31:0]                   m_rdata
);

    assign m_awprot  = 3'b000;
    assign m_wstrb   = 4'b1111;
    assign m_bready  = 1'b1;
    assign m_arvalid = 1'b0;
    assign m_araddr  = {AXILITE_ADDR_WIDTH{1'b0}};
    assign m_arprot  = 3'b000;
    assign m_rready  = 1'b1;

    // -----------------------------------------------------------------
    // 16-deep FIFO of pending (waddr, wdata) pairs
    // -----------------------------------------------------------------
    localparam FIFO_LOG2 = 4;
    localparam FIFO_LEN  = 1 << FIFO_LOG2;
    reg [17:0] fifo_mem [0:FIFO_LEN-1];
    reg [FIFO_LOG2:0] fifo_wptr;
    reg [FIFO_LOG2:0] fifo_rptr;
    wire fifo_empty = (fifo_wptr == fifo_rptr);
    wire fifo_full  = (fifo_wptr[FIFO_LOG2] != fifo_rptr[FIFO_LOG2]) &&
                      (fifo_wptr[FIFO_LOG2-1:0] == fifo_rptr[FIFO_LOG2-1:0]);

    always @(posedge clk) begin
        if (!rst_n) begin
            fifo_wptr <= 0;
        end else if (cls_cfg_wen && !fifo_full) begin
            fifo_mem[fifo_wptr[FIFO_LOG2-1:0]] <= {cls_cfg_waddr, cls_cfg_wdata};
            fifo_wptr <= fifo_wptr + 1;
        end
    end

    // -----------------------------------------------------------------
    // FSM: pop FIFO, issue 8 AXI-Lite writes per pop
    // -----------------------------------------------------------------
    localparam ST_IDLE = 2'd0;
    localparam ST_AW_W = 2'd1;
    localparam ST_B    = 2'd2;

    reg [1:0]  state;
    reg [9:0]  cur_waddr;
    reg [7:0]  cur_wdata;
    reg [2:0]  bit_idx;
    reg        aw_done;
    reg        w_done;

    // Compute memstream byte address for current bit
    function [AXILITE_ADDR_WIDTH-1:0] make_byte_addr;
        input [9:0] waddr;
        input [2:0] bidx;
        reg [12:0] row;
        begin
            row = {waddr, 3'b000} + {10'b0, bidx};
            make_byte_addr = {{(AXILITE_ADDR_WIDTH-15){1'b0}}, row, 2'b00};
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            m_awvalid <= 1'b0;
            m_wvalid  <= 1'b0;
            m_awaddr  <= {AXILITE_ADDR_WIDTH{1'b0}};
            m_wdata   <= 32'd0;
            cur_waddr <= 10'd0;
            cur_wdata <= 8'd0;
            bit_idx   <= 3'd0;
            aw_done   <= 1'b0;
            w_done    <= 1'b0;
            fifo_rptr <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    m_awvalid <= 1'b0;
                    m_wvalid  <= 1'b0;
                    aw_done   <= 1'b0;
                    w_done    <= 1'b0;
                    if (!fifo_empty) begin
                        cur_waddr <= fifo_mem[fifo_rptr[FIFO_LOG2-1:0]][17:8];
                        cur_wdata <= fifo_mem[fifo_rptr[FIFO_LOG2-1:0]][7:0];
                        fifo_rptr <= fifo_rptr + 1;
                        bit_idx   <= 3'd0;
                        m_awaddr  <= make_byte_addr(fifo_mem[fifo_rptr[FIFO_LOG2-1:0]][17:8], 3'd0);
                        m_wdata   <= {31'd0, fifo_mem[fifo_rptr[FIFO_LOG2-1:0]][0]};
                        m_awvalid <= 1'b1;
                        m_wvalid  <= 1'b1;
                        state     <= ST_AW_W;
                    end
                end
                ST_AW_W: begin
                    if (m_awvalid && m_awready) begin
                        m_awvalid <= 1'b0;
                        aw_done   <= 1'b1;
                    end
                    if (m_wvalid && m_wready) begin
                        m_wvalid <= 1'b0;
                        w_done   <= 1'b1;
                    end
                    if ((aw_done || (m_awvalid && m_awready)) &&
                        (w_done  || (m_wvalid  && m_wready ))) begin
                        state <= ST_B;
                    end
                end
                ST_B: begin
                    if (m_bvalid) begin
                        aw_done <= 1'b0;
                        w_done  <= 1'b0;
                        if (bit_idx == 3'd7) begin
                            state <= ST_IDLE;
                        end else begin
                            bit_idx   <= bit_idx + 1;
                            m_awaddr  <= make_byte_addr(cur_waddr, bit_idx + 1);
                            m_wdata   <= {31'd0, cur_wdata[bit_idx + 1]};
                            m_awvalid <= 1'b1;
                            m_wvalid  <= 1'b1;
                            state     <= ST_AW_W;
                        end
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
