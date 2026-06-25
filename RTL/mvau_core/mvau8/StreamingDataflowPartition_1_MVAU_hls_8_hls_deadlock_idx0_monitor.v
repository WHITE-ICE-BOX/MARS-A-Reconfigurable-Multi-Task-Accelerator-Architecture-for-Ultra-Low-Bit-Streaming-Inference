// ===========================================================================
// [交接導向註解]
// MVAU8 — FC2 / classifier（最後一層全連接，MVAU_hls_8）
// 改造：classifier 權重經 cls_cfg_bridge 變 AXI 可寫（runtime 換任務改分類投影權重）；memstream 設為 runtime-writeable。
// 
// 本檔：
//   FINN HLS 控制/死結偵測（自動生成，未改）。
// 
// 流程：FINN_Compile → 設可寫 memstream + cls_cfg_bridge → SoC → FPGA。
// ===========================================================================

`timescale 1 ns / 1 ps

module StreamingDataflowPartition_1_MVAU_hls_8_hls_deadlock_idx0_monitor ( // for module StreamingDataflowPartition_1_MVAU_hls_8_StreamingDataflowPartition_1_MVAU_hls_8_inst
    input wire clock,
    input wire reset,
    input wire [2:0] axis_block_sigs,
    input wire [0:0] inst_idle_sigs,
    input wire [0:0] inst_block_sigs,
    output wire block
);

// signal declare
reg monitor_find_block;
wire sub_parallel_block;
wire all_sub_parallel_has_block;
wire all_sub_single_has_block;
wire cur_axis_has_block;
wire seq_is_axis_block;

assign block = monitor_find_block;
assign all_sub_parallel_has_block = 1'b0;
assign all_sub_single_has_block = 1'b0;
assign cur_axis_has_block = 1'b0 | axis_block_sigs[0] | axis_block_sigs[1] | axis_block_sigs[2];
assign seq_is_axis_block = all_sub_parallel_has_block | all_sub_single_has_block | cur_axis_has_block;

always @(posedge clock) begin
    if (reset == 1'b1)
        monitor_find_block <= 1'b0;
    else if (seq_is_axis_block == 1'b1)
        monitor_find_block <= 1'b1;
    else
        monitor_find_block <= 1'b0;
end


// instant sub module
endmodule
