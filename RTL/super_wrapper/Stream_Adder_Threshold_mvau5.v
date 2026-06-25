// ===========================================================================
// [交接導向註解]
// 模組：MVAU5 加法+Q8閾值。用 contribution LUT 縮放 Adapter popcount，與 MVAU partial-sum 相加後二值化（零 DSP）。流程：RTL。
// ===========================================================================

`timescale 1ns / 1ps

// =====================================================================
// Stream_Adder_Threshold (adapter_fixed version, Q8-widened)
// =====================================================================
// 2026-04-11 design notes (do NOT touch any .dat file — change this
// RTL instead if anything looks off):
//
// * memblock.dat is FINN-original (not patched). Do not modify.
// * This module does NOT use the FINN integer threshold file
//   (threshs_ROM_MVAU5.dat) because it only stores round/ceil(t_pop)
//   and loses the sub-popcount fractional offset. When the adapter
//   contribution is fractional, ~6 per 2560 outputs end up on the
//   wrong side of FINN's integer boundary vs the PyTorch float forward.
//
// * Instead, this module loads two PyTorch-derived support files that
//   encode the true float threshold at Q8 (×256) precision:
//     - adapter_5_alpha.dat : signed 32-bit Q8 bipolar-popcount
//                                  threshold per output channel
//     - threshs_ROM_MVAU5.dat    : 1-bit per channel polarity XOR for
//                                  BN gamma<0 channels (here: only
//                                  ch=19, whose effective threshold
//                                  lives beyond the popcount range)
//   Both are derived by sw/ scripts reading the SAME RC_m1_full.tar
//   that is used to generate mvau5_expected_output_pytorch.dat, so
//   they ARE "software ground truth", not a FINN-data patch.
//
// * Datapath: everything is kept in Q8 so the 6 boundary cases are
//   decided at 1/256 popcount resolution:
//      total_q8   = (mvau_pop <<< 8) + adp_raw * ALPHA_Q8
//      result_bit = (total_q8 >= thresh_q8) XOR pol_rom[ch]
//
// * For BN gamma<0, t_pop_pos can exceed the 0..2304 popcount range.
//   In that case thresh_q8 is simply set very large, the compare is
//   always false, and the pol XOR produces the constant PyTorch output.
// =====================================================================

module Stream_Adder_Threshold_mvau5 (
    input  wire                  aclk,
    input  wire                  aresetn,

    // 來自 MVAU5 的 32-bit 整數值 (popcount domain)
    input  wire [31:0]           s_axis_1_tdata,
    input  wire                  s_axis_1_tvalid,
    output reg                   s_axis_1_tready,

    // 來自 Adapter 的 8-bit Popcount (0..64)
    input  wire [7:0]            s_axis_2_tdata,
    input  wire                  s_axis_2_tvalid,
    output reg                   s_axis_2_tready,

    // 輸出給下一層的 1-bit 二值化結果
    output reg  [7:0]            m_axis_tdata,
    output reg                   m_axis_tvalid,
    input  wire                  m_axis_tready
);

    // =========================================================
    // 參數
    // =========================================================
    // ALPHA_Q8 = round(alpha_pytorch * 256). For mvau5 this is 1452.
    parameter signed [31:0] ALPHA_Q8 = 32'sd1442;

    // =========================================================
    // Threshold ROM (Q8 form: signed 32-bit = round(t_pop_float*256))
    // Polarity ROM: 1-bit per channel XOR mask for BN gamma<0
    // =========================================================
    reg signed [31:0] thresh_rom [0:255];
    // NOTE: Vivado synth refuses $readmemh into a 1-bit unpacked array
    // ("invalid memory name"). Widen to 8-bit, only bit [0] is meaningful.
    reg        [7:0]  pol_rom    [0:255];
    initial begin
        $readmemh(
            "/home/barkie1/mvau_pipeline/mvau_adapter/mvau5/data/threshs_ROM_MVAU5_q8.dat",
            thresh_rom
        );
        $readmemh(
            "/home/barkie1/mvau_pipeline/mvau_adapter/mvau5/data/threshs_pol_MVAU5.dat",
            pol_rom
        );
    end

    // =========================================================
    // Adapter Contribution 預先計算查表 (Q8 form, no rounding)
    // adp_contrib_lut[i] = (i - 32) * ALPHA_Q8    (exact Q8)
    // =========================================================
    (* rom_style = "distributed" *) reg signed [31:0] adp_contrib_lut [0:255];

    initial begin
        $readmemh(
            "/home/barkie1/mvau_pipeline/mvau_adapter/mvau5/data/adapter_5_contrib_lut.dat",
            adp_contrib_lut
        );
    end

    // =========================================================
    // 狀態與管線暫存器
    // =========================================================
    reg [7:0]              ch_cnt;

    reg                    s1_valid;
    reg signed [31:0]      s1_mvau_pop;
    reg signed [31:0]      s1_adp_contribution;
    reg [7:0]              s1_adp_pop;   // debug
    reg signed [31:0]      s1_adp_raw;   // debug

    // =========================================================
    // Combinational datapath signals (_c)
    // =========================================================
    reg                    both_valid_s0_c;
    reg                    s1_ready_c;
    reg                    s2_ready_c;

    reg signed [31:0]      adp_pop_ext_c;
    reg signed [31:0]      adp_raw_c;
    reg signed [31:0]      adp_contribution_c;

    reg signed [31:0]      total_pop_c;   // Q8 domain
    reg signed [31:0]      threshold_c;   // Q8 domain
    reg                    compare_c;
    reg                    result_bit_c;

    // =========================================================
    // Debug registers
    // =========================================================
    reg signed [31:0]      dbg_mvau_pop;
    reg [7:0]              dbg_adp_pop;
    reg signed [31:0]      dbg_adp_raw;
    reg signed [31:0]      dbg_adp_contribution;
    reg signed [31:0]      dbg_total_pop;
    reg signed [31:0]      dbg_threshold;
    reg                    dbg_result_bit;
    reg [7:0]              dbg_ch_cnt;

    // =========================================================
    // Combinational control + datapath
    // =========================================================
    always @(*) begin
        both_valid_s0_c    = 1'b0;
        s1_ready_c         = 1'b0;
        s2_ready_c         = 1'b0;
        s_axis_1_tready    = 1'b0;
        s_axis_2_tready    = 1'b0;

        adp_pop_ext_c      = 32'sd0;
        adp_raw_c          = 32'sd0;
        adp_contribution_c = 32'sd0;

        total_pop_c        = 32'sd0;
        threshold_c        = 32'sd0;
        compare_c          = 1'b0;
        result_bit_c       = 1'b0;

        // Pipeline handshake
        s2_ready_c      = (~m_axis_tvalid) | m_axis_tready;
        s1_ready_c      = (~s1_valid) | s2_ready_c;

        both_valid_s0_c = s_axis_1_tvalid & s_axis_2_tvalid;

        s_axis_1_tready = s1_ready_c & s_axis_2_tvalid;
        s_axis_2_tready = s1_ready_c & s_axis_1_tvalid;

        // Stage 0: lookup the Q8 adapter contribution for this popcount
        adp_pop_ext_c      = {24'd0, s_axis_2_tdata};
        adp_raw_c          = adp_pop_ext_c - 32'sd32;
        adp_contribution_c = adp_contrib_lut[s_axis_2_tdata];

        // Stage 1: Q8 compare.
        //   mvau_pop is integer popcount; shift up by 8 to align with Q8 adapter contribution.
        //   thresh_rom[ch] is stored as round(t_pop_float * 256) (plus +1 for gamma<0).
        //   The pol_rom XOR flips the result for BN gamma<0 channels so we
        //   can use a uniform >= compare.
        // FINN folds γ<0 polarity into the weights (only for pol=1 channels),
        // so rtl_mvau for those channels is (2304 - p_original). Un-fold here
        // so the adapter contribution combines with the correct sign. The
        // trailing pol XOR still implements the γ<0 comparison flip.
        total_pop_c  = ((pol_rom[ch_cnt][0] ? (32'sd2304 - s1_mvau_pop) : s1_mvau_pop) <<< 8)
                       + s1_adp_contribution;
        threshold_c  = thresh_rom[ch_cnt];
        compare_c    = (total_pop_c >= threshold_c);
        result_bit_c = compare_c ^ pol_rom[ch_cnt][0];
    end

    // =========================================================
    // Sequential logic
    // =========================================================
    always @(posedge aclk) begin
        if (!aresetn) begin
            m_axis_tvalid        <= 1'b0;
            m_axis_tdata         <= 8'd0;
            ch_cnt               <= 8'd0;

            s1_valid             <= 1'b0;
            s1_mvau_pop          <= 32'sd0;
            s1_adp_contribution  <= 32'sd0;
            s1_adp_pop           <= 8'd0;
            s1_adp_raw           <= 32'sd0;

            dbg_mvau_pop         <= 32'sd0;
            dbg_adp_pop          <= 8'd0;
            dbg_adp_raw          <= 32'sd0;
            dbg_adp_contribution <= 32'sd0;
            dbg_total_pop        <= 32'sd0;
            dbg_threshold        <= 32'sd0;
            dbg_result_bit       <= 1'b0;
            dbg_ch_cnt           <= 8'd0;
        end else begin

            if (s1_ready_c) begin
                if (both_valid_s0_c) begin
                    s1_valid             <= 1'b1;
                    s1_mvau_pop          <= s_axis_1_tdata;
                    s1_adp_contribution  <= adp_contribution_c;
                    s1_adp_pop           <= s_axis_2_tdata;
                    s1_adp_raw           <= adp_raw_c;
                end else begin
                    s1_valid             <= 1'b0;
                end
            end

            if (s2_ready_c) begin
                if (s1_valid) begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= {7'd0, result_bit_c};
                    ch_cnt        <= ch_cnt + 8'd1;

                    dbg_mvau_pop         <= s1_mvau_pop;
                    dbg_adp_pop          <= s1_adp_pop;
                    dbg_adp_raw          <= s1_adp_raw;
                    dbg_adp_contribution <= s1_adp_contribution;
                    dbg_total_pop        <= total_pop_c;
                    dbg_threshold        <= threshold_c;
                    dbg_result_bit       <= result_bit_c;
                    dbg_ch_cnt           <= ch_cnt;
                end else begin
                    m_axis_tvalid <= 1'b0;
                end
            end

        end
    end

endmodule
