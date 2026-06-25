# ===========================================================================
# [交接導向註解]
# 腳本：完整建置流程（stitch -> zynq -> bitstream）。流程：SoC，產出 FPGA/*.bit。
# ===========================================================================

#============================================================================
# build_bitstream.tcl
# -------------------
# Full flow: update stitch project → re-export IP → update zynq → synth → impl → bitstream
#============================================================================

set STITCH_DIR  /home/barkie1/thesis/finn/finn_pipeline/vivado_stitch_proj_hv26s5y4
set ZYNQ_DIR    /home/barkie1/thesis/finn/finn_pipeline/vivado_zynq_proj_fobcvtlq
set OUTPUT_DIR  /home/barkie1/mvau_pipeline/fpga
set PART        xc7z020clg400-1

#============================================================================
# STEP 1: Update stitch project — refresh adapter IPs, regenerate BD
#============================================================================
puts "============================================================"
puts " STEP 1: Updating stitch project"
puts "============================================================"

open_project ${STITCH_DIR}/finn_vivado_stitch_proj.xpr

update_ip_catalog -rebuild

set locked [get_ips -filter {IS_LOCKED == 1}]
if {[llength $locked] > 0} {
    puts "  Upgrading locked IPs: $locked"
    foreach ip $locked {
        upgrade_ip $ip
    }
}

# Regenerate BD
set bd_file [get_files */StreamingDataflowPartition_1.bd]
generate_target {all} $bd_file -force
update_compile_order -fileset sources_1

# Re-export stitch IP by re-packaging the whole project
# (avoid ipx::merge_project_changes which segfaults on this project)
puts "  Re-exporting stitch IP..."
set ip_dir ${STITCH_DIR}/ip

# Back up old component.xml
if {[file exists ${ip_dir}/component.xml]} {
    file copy -force ${ip_dir}/component.xml ${ip_dir}/component.bak
}

ipx::package_project \
    -root_dir   ${ip_dir} \
    -vendor     user.org \
    -library    user \
    -taxonomy   /UserIP \
    -import_files \
    -set_current true \
    -force

set core [ipx::current_core]
set_property name        "StreamingDataflowPartition_1" $core
set_property display_name "StreamingDataflowPartition_1" $core
set_property version     1.0 $core

# Bus interface associations
foreach bus {s_axis_0 m_axis_0} {
    catch {ipx::associate_bus_interfaces -busif $bus -clock ap_clk $core}
}
catch {
    set clk_if [ipx::get_bus_interfaces ap_clk -of_objects $core]
    if {[llength $clk_if] > 0} {
        set p [ipx::add_bus_parameter ASSOCIATED_RESET $clk_if]
        set_property value ap_rst_n $p
    }
}

ipx::create_xgui_files $core
ipx::update_checksums  $core
ipx::save_core         $core

puts "  -> Stitch IP saved at ${ip_dir}"

close_project

#============================================================================
# STEP 2: Update zynq project
#============================================================================
puts ""
puts "============================================================"
puts " STEP 2: Updating Zynq project"
puts "============================================================"

open_project ${ZYNQ_DIR}/finn_zynq_link.xpr

update_ip_catalog -rebuild

set locked [get_ips -filter {IS_LOCKED == 1}]
if {[llength $locked] > 0} {
    puts "  Upgrading locked IPs: $locked"
    foreach ip $locked {
        upgrade_ip $ip
    }
}

set bd_file [get_files */top.bd]
generate_target {all} $bd_file -force
update_compile_order -fileset sources_1

#============================================================================
# STEP 3: Synthesis
#============================================================================
puts ""
puts "============================================================"
puts " STEP 3: Synthesis"
puts "============================================================"

reset_run synth_1 -quiet
launch_runs synth_1 -jobs 4
wait_on_run synth_1

set synth_status [get_property STATUS [get_runs synth_1]]
puts "  Synthesis status: $synth_status"

if {![string match "*Complete*" $synth_status]} {
    puts "  ERROR: Synthesis failed!"
    close_project
    exit 1
}

#============================================================================
# STEP 4: Implementation
#============================================================================
puts ""
puts "============================================================"
puts " STEP 4: Implementation"
puts "============================================================"

launch_runs impl_1 -jobs 4
wait_on_run impl_1

set impl_status [get_property STATUS [get_runs impl_1]]
puts "  Implementation status: $impl_status"

if {![string match "*Complete*" $impl_status]} {
    puts "  ERROR: Implementation failed!"
    close_project
    exit 1
}

#============================================================================
# STEP 5: Bitstream
#============================================================================
puts ""
puts "============================================================"
puts " STEP 5: Bitstream"
puts "============================================================"

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

set bit_status [get_property STATUS [get_runs impl_1]]
puts "  Bitstream status: $bit_status"

#============================================================================
# STEP 6: Copy .bit and .hwh as "resizer"
#============================================================================
puts ""
puts "============================================================"
puts " STEP 6: Copying output files"
puts "============================================================"

file mkdir ${OUTPUT_DIR}

# Find .bit file
set impl_dir ${ZYNQ_DIR}/finn_zynq_link.runs/impl_1
set bit_files [glob -nocomplain ${impl_dir}/*.bit]
if {[llength $bit_files] > 0} {
    file copy -force [lindex $bit_files 0] ${OUTPUT_DIR}/resizer.bit
    puts "  -> ${OUTPUT_DIR}/resizer.bit"
} else {
    puts "  WARNING: No .bit file found in ${impl_dir}"
}

# Find .hwh file
set hwh_files [glob -nocomplain ${ZYNQ_DIR}/finn_zynq_link.gen/sources_1/bd/top/hw_handoff/*.hwh]
if {[llength $hwh_files] > 0} {
    file copy -force [lindex $hwh_files 0] ${OUTPUT_DIR}/resizer.hwh
    puts "  -> ${OUTPUT_DIR}/resizer.hwh"
} else {
    set hwh_files2 [glob -nocomplain ${ZYNQ_DIR}/finn_zynq_link.runs/impl_1/*.hwh]
    if {[llength $hwh_files2] > 0} {
        file copy -force [lindex $hwh_files2 0] ${OUTPUT_DIR}/resizer.hwh
        puts "  -> ${OUTPUT_DIR}/resizer.hwh"
    } else {
        puts "  WARNING: No .hwh file found"
    }
}

close_project

puts ""
puts "============================================================"
puts " BUILD COMPLETE"
puts " Output: ${OUTPUT_DIR}/resizer.bit"
puts "         ${OUTPUT_DIR}/resizer.hwh"
puts "============================================================"

exit 0
