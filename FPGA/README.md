# FPGA

Final on-board artifacts for PYNQ-Z2 (XC7Z020, 100 MHz). Four builds:

| Folder | Bitstream | Role |
|---|---|---|
| `backbone_throughput/`   | `resizer.bit`        | backbone-only, high-PE (throughput baseline) |
| `MARS_throughput_2ds/`   | `resizer_v1.bit`     | MARS adapter, 2-dataset, high-PE (energy/throughput) |
| `backbone_compact_pe1/`  | `resizer.bit`        | backbone-only, PE=1 (compact baseline) |
| `MARS_compact_5ds_pe1/`  | `resizer_3ds_v3.bit` | MARS, 5-dataset runtime switch, PE=1 (accuracy + switching) |

Each folder: `*.bit` + `*.hwh` + `driver.py`/`driver_base.py` + board test/runtime scripts
and runtime params. The compact 5-dataset build switches tasks via the `cfg_hub` (~26 KB
blob, ~1.86 ms) with no fabric reconfiguration.

## Run (PYNQ-Z2)
```
sudo XILINX_XRT=/usr BOARD=Pynq-Z2 python3 driver.py \
  --exec_mode throughput_test --bitfile <bit> --batchsize 1000
```
Large dataset test inputs (`*_test_x.npy`, ~30 MB) are excluded — regenerate from
`AI_model_train`. Canonical bitstreams retained per verified md5 mapping.
