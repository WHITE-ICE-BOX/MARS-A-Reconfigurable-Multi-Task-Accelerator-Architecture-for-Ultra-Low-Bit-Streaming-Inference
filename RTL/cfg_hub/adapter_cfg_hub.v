`timescale 1ns / 1ps

// =====================================================================
// adapter_cfg_hub — AXI-Lite slave → 5 MVAU config write ports
// =====================================================================
// Address space: 64 KB (16-bit byte address)
//   [15:13] selects MVAU:
//     001 = MVAU1 (0x2000-0x3FFF)
//     010 = MVAU2 (0x4000-0x5FFF)
//     011 = MVAU3 (0x6000-0x7FFF)
//     100 = MVAU4 (0x8000-0x9FFF)
//     101 = MVAU5 (0xA000-0xBFFF)
//
//   [12:2] = 11-bit word address within each MVAU (per-MVAU memory map)
//
// Each MVAU's memory map:
//   Word 0:              adapter_enable
//   Word 4..67:          rom_rc[0..63]
//   Word 128..639:       rom_down (entry*HIDDEN_CH + word)
//   Word 640..1151:      rom_up   (entry*UP_WORDS + word)
//   Word 1152..1407:     thresh_mem[0..255]
//   Word 1408..1663:     sign_mem[0..255]
//   Word 1664..1919:     contrib_lut[0..255]
// =====================================================================

module adapter_cfg_hub #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 16
)(
    // AXI-Lite Slave Interface
    input  wire                                S_AXI_ACLK,
    input  wire                                S_AXI_ARESETN,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       S_AXI_AWADDR,
    input  wire [2:0]                          S_AXI_AWPROT,
    input  wire                                S_AXI_AWVALID,
    output reg                                 S_AXI_AWREADY,

    input  wire [C_S_AXI_DATA_WIDTH-1:0]       S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0]   S_AXI_WSTRB,
    input  wire                                S_AXI_WVALID,
    output reg                                 S_AXI_WREADY,

    output reg  [1:0]                          S_AXI_BRESP,
    output reg                                 S_AXI_BVALID,
    input  wire                                S_AXI_BREADY,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       S_AXI_ARADDR,
    input  wire [2:0]                          S_AXI_ARPROT,
    input  wire                                S_AXI_ARVALID,
    output reg                                 S_AXI_ARREADY,

    output reg  [C_S_AXI_DATA_WIDTH-1:0]       S_AXI_RDATA,
    output reg  [1:0]                          S_AXI_RRESP,
    output reg                                 S_AXI_RVALID,
    input  wire                                S_AXI_RREADY,

    // Config output ports to 5 MVAUs
    output reg  [10:0]  mvau1_cfg_waddr,
    output reg  [31:0]  mvau1_cfg_wdata,
    output reg          mvau1_cfg_wen,

    output reg  [10:0]  mvau2_cfg_waddr,
    output reg  [31:0]  mvau2_cfg_wdata,
    output reg          mvau2_cfg_wen,

    output reg  [10:0]  mvau3_cfg_waddr,
    output reg  [31:0]  mvau3_cfg_wdata,
    output reg          mvau3_cfg_wen,

    output reg  [10:0]  mvau4_cfg_waddr,
    output reg  [31:0]  mvau4_cfg_wdata,
    output reg          mvau4_cfg_wen,

    output reg  [10:0]  mvau5_cfg_waddr,
    output reg  [31:0]  mvau5_cfg_wdata,
    output reg          mvau5_cfg_wen
);

    // =========================================================
    // AXI-Lite write state machine
    // =========================================================
    reg [C_S_AXI_ADDR_WIDTH-1:0] aw_addr_latched;
    reg aw_done, w_done;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            S_AXI_AWREADY <= 1'b0;
            S_AXI_WREADY  <= 1'b0;
            S_AXI_BVALID  <= 1'b0;
            S_AXI_BRESP   <= 2'b00;
            aw_done       <= 1'b0;
            w_done        <= 1'b0;
            aw_addr_latched <= {C_S_AXI_ADDR_WIDTH{1'b0}};

            mvau1_cfg_wen <= 1'b0;
            mvau2_cfg_wen <= 1'b0;
            mvau3_cfg_wen <= 1'b0;
            mvau4_cfg_wen <= 1'b0;
            mvau5_cfg_wen <= 1'b0;
        end else begin
            // Default: deassert write enables after one cycle
            mvau1_cfg_wen <= 1'b0;
            mvau2_cfg_wen <= 1'b0;
            mvau3_cfg_wen <= 1'b0;
            mvau4_cfg_wen <= 1'b0;
            mvau5_cfg_wen <= 1'b0;

            // AW channel
            if (S_AXI_AWVALID && !aw_done && !S_AXI_BVALID) begin
                S_AXI_AWREADY   <= 1'b1;
                aw_addr_latched <= S_AXI_AWADDR;
                aw_done         <= 1'b1;
            end else begin
                S_AXI_AWREADY <= 1'b0;
            end

            // W channel
            if (S_AXI_WVALID && !w_done && !S_AXI_BVALID) begin
                S_AXI_WREADY <= 1'b1;
                w_done       <= 1'b1;
            end else begin
                S_AXI_WREADY <= 1'b0;
            end

            // Both AW and W done → issue write + B response
            if (aw_done && w_done && !S_AXI_BVALID) begin
                S_AXI_BVALID <= 1'b1;
                S_AXI_BRESP  <= 2'b00;  // OKAY

                // Decode MVAU select and dispatch
                case (aw_addr_latched[15:13])
                    3'd1: begin
                        mvau1_cfg_waddr <= aw_addr_latched[12:2];
                        mvau1_cfg_wdata <= S_AXI_WDATA;
                        mvau1_cfg_wen   <= 1'b1;
                    end
                    3'd2: begin
                        mvau2_cfg_waddr <= aw_addr_latched[12:2];
                        mvau2_cfg_wdata <= S_AXI_WDATA;
                        mvau2_cfg_wen   <= 1'b1;
                    end
                    3'd3: begin
                        mvau3_cfg_waddr <= aw_addr_latched[12:2];
                        mvau3_cfg_wdata <= S_AXI_WDATA;
                        mvau3_cfg_wen   <= 1'b1;
                    end
                    3'd4: begin
                        mvau4_cfg_waddr <= aw_addr_latched[12:2];
                        mvau4_cfg_wdata <= S_AXI_WDATA;
                        mvau4_cfg_wen   <= 1'b1;
                    end
                    3'd5: begin
                        mvau5_cfg_waddr <= aw_addr_latched[12:2];
                        mvau5_cfg_wdata <= S_AXI_WDATA;
                        mvau5_cfg_wen   <= 1'b1;
                    end
                    default: ; // ignore writes to reserved addresses
                endcase

                aw_done <= 1'b0;
                w_done  <= 1'b0;
            end

            // B handshake
            if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end
        end
    end

    // =========================================================
    // AXI-Lite read (stub — return 0, write-only interface)
    // =========================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            S_AXI_ARREADY <= 1'b0;
            S_AXI_RVALID  <= 1'b0;
            S_AXI_RDATA   <= 32'd0;
            S_AXI_RRESP   <= 2'b00;
        end else begin
            if (S_AXI_ARVALID && !S_AXI_RVALID) begin
                S_AXI_ARREADY <= 1'b1;
                S_AXI_RVALID  <= 1'b1;
                S_AXI_RDATA   <= 32'd0;
                S_AXI_RRESP   <= 2'b00;
            end else begin
                S_AXI_ARREADY <= 1'b0;
            end

            if (S_AXI_RVALID && S_AXI_RREADY) begin
                S_AXI_RVALID <= 1'b0;
            end
        end
    end

endmodule
