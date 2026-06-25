# RTL

Custom RTL that turns each FINN MVAU into an adapter-augmented "Super Wrapper",
plus the runtime function-switch controller.

- `adapter/` — `Adapter_MVAU{1..4}.v`, `Adapter_Generic.v` (MVAU5):
  1x1 down-projection → sign → 1x1 up-projection, with Int8 RC bias absorbed into the
  accumulator reset (4-stage pipeline: latch/ROM, XNOR, popcount, accumulate).
- `super_wrapper/` — per-MVAU `MVAU{N}_Super_Wrapper.v` + `Stream_Splitter`,
  `Stream_Adder[_Threshold]`, `Simple_FIFO` (cycle-accurate side-stream merge + Q8 threshold).
- `cfg_hub/` — `adapter_cfg_hub.v`: AXI-Lite hub (base `0x40010000`) that demuxes per-task
  config (thresholds, classifier weights, adapter blobs) to register banks — the
  controller-free runtime multi-task switch (19 LUTs).
- `tcl/` — `package_ips.tcl`, `package_mvau*.tcl`, `make_project.tcl`, `build_bitstream.tcl`.
