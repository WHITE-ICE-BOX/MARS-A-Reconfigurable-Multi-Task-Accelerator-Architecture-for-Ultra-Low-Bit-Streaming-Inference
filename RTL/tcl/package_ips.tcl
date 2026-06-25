#-----------------------------------------------------------
# Package MVAU1~MVAU5 Super_Wrapper as IPs.
# Output: /home/barkie1/mvau_pipeline/mvau_adapter_ip/mvauN/
#-----------------------------------------------------------
set ROOT      /home/barkie1/mvau_pipeline/mvau_adapter
set IP_ROOT   /home/barkie1/mvau_pipeline/mvau_adapter_ip
set PART      xc7z020clg400-1

proc get_sources {n} {
    global ROOT
    set src {}
    foreach f [glob -nocomplain "$ROOT/mvau$n/adapter_fixed/*.v"] {
        set b [file tail $f]
        # Exclude ALL super wrappers, we add only the one for this N
        if {[string match -nocase "MVAU*_Super_Wrapper.v" $b]} continue
        lappend src $f
    }
    lappend src "$ROOT/mvau$n/adapter_fixed/MVAU${n}_Super_Wrapper.v"
    foreach f [glob -nocomplain "$ROOT/mvau$n/mvau${n}_fixed/*.v"] {
        lappend src $f
    }
    foreach f [glob -nocomplain "$ROOT/mvau$n/mvau${n}_fixed/*.sv"] {
        lappend src $f
    }
    return $src
}

foreach n {1 2 3 4 5} {
    set proj_dir "$IP_ROOT/mvau$n"
    set top      "MVAU${n}_Super_Wrapper"
    set ip_dir   "$proj_dir/ip"

    puts "=============================================="
    puts "  Packaging MVAU$n -> $proj_dir"
    puts "=============================================="

    file delete -force $proj_dir
    file mkdir $proj_dir

    create_project -part $PART mvau_adapter_ip $proj_dir -force

    set srcs [get_sources $n]
    puts "  Adding [llength $srcs] source files"
    add_files -norecurse $srcs

    set_property top $top [current_fileset]
    update_compile_order -fileset sources_1

    # Package current project as IP
    ipx::package_project \
        -root_dir   $ip_dir \
        -vendor     user.org \
        -library    user \
        -taxonomy   /UserIP \
        -import_files \
        -set_current true \
        -force

    set core [ipx::current_core]
    set_property name        "mvau${n}_adapter"   $core
    set_property display_name "MVAU${n} Adapter-Fused IP" $core
    set_property description  "MVAU$n + fractional adapter fused as one AXI-Stream IP" $core
    set_property version     1.0 $core
    set_property vendor_display_name "user.org" $core

    # Clock-bus association: fix [BD 41-967]
    # in0_V, out_V, weights_V all driven by ap_clk; reset is ap_rst_n
    ipx::associate_bus_interfaces -busif in0_V     -clock ap_clk $core
    ipx::associate_bus_interfaces -busif out_V     -clock ap_clk $core
    ipx::associate_bus_interfaces -busif weights_V -clock ap_clk $core

    # ASSOCIATED_RESET on ap_clk
    set clk_if [ipx::get_bus_interfaces ap_clk -of_objects $core]
    if {[llength $clk_if] > 0} {
        set p [ipx::get_bus_parameters ASSOCIATED_RESET -of_objects $clk_if]
        if {[llength $p] == 0} {
            set p [ipx::add_bus_parameter ASSOCIATED_RESET $clk_if]
        }
        set_property value ap_rst_n $p
    }

    ipx::create_xgui_files $core
    ipx::update_checksums  $core
    ipx::check_integrity   $core
    ipx::save_core         $core
    ipx::archive_core      "$ip_dir/mvau${n}_adapter_1.0.zip" $core

    close_project -delete
    puts "  -> IP at $ip_dir"
    puts "  -> Archive: $ip_dir/mvau${n}_adapter_1.0.zip"
}

puts "\nAll 5 IPs packaged under $IP_ROOT"
exit 0
