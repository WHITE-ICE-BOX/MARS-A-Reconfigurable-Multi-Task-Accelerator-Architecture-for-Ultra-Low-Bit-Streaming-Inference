`timescale 1ns / 1ps

module Adapter_MVAU5 #(
    // --- 使用者參數 (Layer 5 / Conv6 規格) ---
    parameter LAYER_ID = 0,
    parameter IN_CH    = 256,
    parameter OUT_CH   = 256,
    parameter PE       = 1,
    parameter SIMD     = 32,
    parameter REDUCTION = 4,
    
    // --- 權重檔案路徑 ---
    parameter DOWN_WEIGHT_FILE = "/home/barkie1/mvau_pipeline/mvau_adapter/mvau5/data/adapter_5_down_packed.dat",
    parameter UP_WEIGHT_FILE   = "/home/barkie1/mvau_pipeline/mvau_adapter/mvau5/data/adapter_5_up_packed.dat",
    parameter RC_FILE          = "/home/barkie1/mvau_pipeline/mvau_adapter/mvau5/data/adapter_5_rc.dat",

    // --- 系統自動計算參數 ---
    parameter HIDDEN_CH   = (IN_CH/REDUCTION > 0) ? (IN_CH/REDUCTION) : 16,
    parameter IN_CHUNKS   = (IN_CH/SIMD > 0) ? (IN_CH/SIMD) : 1,
    parameter OUT_STEPS   = (OUT_CH/PE > 0) ? (OUT_CH/PE) : 1,

    parameter PIXEL_CYCLES = IN_CHUNKS,
    parameter WIN_CYCLES   = 9 * PIXEL_CYCLES,
    parameter CENTER_START = 4 * PIXEL_CYCLES,
    parameter CENTER_END   = 5 * PIXEL_CYCLES - 1
)(
    input  wire            aclk,
    input  wire            aresetn,
    input  wire [SIMD-1:0] s_axis_tdata,
    input  wire            s_axis_tvalid,
    output wire            s_axis_tready,
    output reg  [7:0]      m_axis_tdata,
    output reg             m_axis_tvalid,
    input  wire            m_axis_tready
);

    // --- 權重 ROM — pre-packed wide format ---
    (* rom_style = "distributed" *) reg [(HIDDEN_CH*SIMD)-1:0] rom_down [0 : IN_CHUNKS - 1];
    (* rom_style = "distributed" *) reg [63:0] rom_up [0 : OUT_STEPS - 1];
    (* rom_style = "distributed" *) reg [15:0] rom_rc [0 : HIDDEN_CH - 1];

    initial begin
        $readmemh(DOWN_WEIGHT_FILE, rom_down);
        $readmemh(UP_WEIGHT_FILE,   rom_up);
        $readmemh(RC_FILE,          rom_rc);
    end

    // --- 內部狀態 ---
    reg [15:0] cycle_cnt;
    reg [15:0] out_step_cnt;
    reg        is_outputting;

    reg signed [15:0] hidden_acc [0:HIDDEN_CH-1];
    reg        [HIDDEN_CH-1:0] hidden_act;
    reg        [HIDDEN_CH-1:0] hidden_act_next;

    // --- output path pipeline ---
    reg [63:0] xnor_pipe [0:PE-1];
    reg        valid_pipe;

    // --- hidden path pipeline ---
    reg [SIMD-1:0] s0_tdata;
    reg [15:0]     s0_idx;
    reg            s0_valid;

    reg [SIMD-1:0] s1_act;
    reg [SIMD-1:0] s1_down [0:HIDDEN_CH-1];
    reg            s1_valid;

    reg [SIMD-1:0] s2_xnor [0:HIDDEN_CH-1];
    reg            s2_valid;

    reg [7:0]      s3_pop [0:HIDDEN_CH-1];
    reg            s3_valid;

    integer i, j, k;

    // --- 握手與 Stall 邏輯 ---
    wire stall = m_axis_tvalid && !m_axis_tready;

    // 保留原本外部 ready 條件
    assign s_axis_tready = aresetn && !is_outputting && !valid_pipe && (!m_axis_tvalid || m_axis_tready);

    // --- Popcount 函數 ---
    function [7:0] popcount_simd;
        input [SIMD-1:0] vec;
        integer n;
        begin
            popcount_simd = 0;
            for (n = 0; n < SIMD; n = n + 1)
                popcount_simd = popcount_simd + vec[n];
        end
    endfunction

    function [7:0] popcount64;
        input [63:0] vec;
        integer n;
        begin
            popcount64 = 0;
            for (n = 0; n < 64; n = n + 1)
                popcount64 = popcount64 + vec[n];
        end
    endfunction

    // --- combinational 計算本次 window 的 hidden activation ---
    always @(*) begin
        for (k = 0; k < HIDDEN_CH; k = k + 1) begin
            hidden_act_next[k] = (hidden_acc[k] >= 0) ? 1'b1 : 1'b0;
        end
    end

    // --- 主運算與輸出邏輯 ---
    always @(posedge aclk) begin
        if (!aresetn) begin
            cycle_cnt     <= 0;
            out_step_cnt  <= 0;
            is_outputting <= 1'b0;

            m_axis_tvalid <= 1'b0;
            m_axis_tdata  <= 8'd0;
            hidden_act    <= {HIDDEN_CH{1'b0}};

            valid_pipe    <= 1'b0;
            for (j = 0; j < PE; j = j + 1)
                xnor_pipe[j] <= 64'd0;

            s0_tdata <= {SIMD{1'b0}};
            s0_idx   <= 16'd0;
            s0_valid <= 1'b0;

            s1_act   <= {SIMD{1'b0}};
            s1_valid <= 1'b0;

            s2_valid <= 1'b0;
            s3_valid <= 1'b0;

            for (i = 0; i < HIDDEN_CH; i = i + 1) begin
                hidden_acc[i] <= $signed(rom_rc[i]);
                s1_down[i]    <= {SIMD{1'b0}};
                s2_xnor[i]    <= {SIMD{1'b0}};
                s3_pop[i]     <= 8'd0;
            end
        end
        else if (!stall) begin
            // =====================================================
            // Hidden path pipeline shift
            // =====================================================
            s3_valid <= s2_valid;
            s2_valid <= s1_valid;
            s1_valid <= s0_valid;
            s0_valid <= 1'b0;

            // S1 -> S2 : XNOR
            if (s1_valid) begin
                for (i = 0; i < HIDDEN_CH; i = i + 1) begin
                    s2_xnor[i] <= ~(s1_act ^ s1_down[i]);
                end
            end

            // S2 -> S3 : popcount
            if (s2_valid) begin
                for (i = 0; i < HIDDEN_CH; i = i + 1) begin
                    s3_pop[i] <= popcount_simd(s2_xnor[i]);
                end
            end

            // S3 -> ACC : accumulate
            if (s3_valid) begin
                for (i = 0; i < HIDDEN_CH; i = i + 1) begin
                    hidden_acc[i] <= hidden_acc[i] + s3_pop[i];
                end
            end

            // =====================================================
            // Output Path Stage 2
            // =====================================================
            if (valid_pipe) begin
                m_axis_tvalid <= 1'b1;
                m_axis_tdata  <= popcount64(xnor_pipe[0]);
            end
            else begin
                m_axis_tvalid <= 1'b0;
            end

            // =====================================================
            // Output Path Stage 1
            // =====================================================
            if (is_outputting) begin
                valid_pipe <= 1'b1;
                xnor_pipe[0] <= ~(hidden_act ^ rom_up[out_step_cnt]);

                if (out_step_cnt == OUT_STEPS - 1) begin
                    is_outputting <= 1'b0;
                    out_step_cnt  <= 0;
                end
                else begin
                    out_step_cnt <= out_step_cnt + 1;
                end
            end
            else begin
                valid_pipe <= 1'b0;
            end

            // =====================================================
            // Stage 0 : 輸入運算邏輯
            // =====================================================
            if (s_axis_tvalid && s_axis_tready) begin
                if (cycle_cnt >= CENTER_START && cycle_cnt <= CENTER_END) begin
                    s0_tdata <= s_axis_tdata;
                    s0_idx   <= cycle_cnt - CENTER_START;
                    s0_valid <= 1'b1;
                end

                if (cycle_cnt == WIN_CYCLES - 1) begin
                    cycle_cnt <= 0;

                    // 這裡維持原本時序點
                    hidden_act <= hidden_act_next;

                    for (i = 0; i < HIDDEN_CH; i = i + 1) begin
                        hidden_acc[i] <= $signed(rom_rc[i]);
                    end

                    is_outputting <= 1'b1;
                    out_step_cnt  <= 0;
                end
                else begin
                    cycle_cnt <= cycle_cnt + 1;
                end
            end

            // =====================================================
            // S0 -> S1 : ROM read
            // =====================================================
            if (s0_valid) begin
                s1_act <= s0_tdata;
                for (i = 0; i < HIDDEN_CH; i = i + 1) begin
                    s1_down[i] <= rom_down[s0_idx][i*SIMD +: SIMD];
                end
            end
        end
    end

endmodule
