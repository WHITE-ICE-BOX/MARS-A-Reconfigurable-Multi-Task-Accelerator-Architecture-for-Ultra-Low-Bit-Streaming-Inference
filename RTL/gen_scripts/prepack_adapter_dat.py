#!/usr/bin/env python3
# ===========================================================================
# [交接導向註解]
# 腳本：把 adapter .dat 預打包成 $readmemh 寬格式（給 RTL ROM 直接載入）。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

"""
Pre-pack adapter .dat files into wide format for direct $readmemh loading.

Eliminates non-constant for-loop repacking in Verilog initial blocks
(which Vivado synthesis ignores: Synth 8-311) and avoids multi-port ROM
reads that cause Vivado to merge s1_down registers.

Generates per MVAU:
  - adapter_N_up_packed.dat   : rom_up wide format
  - adapter_N_down_packed.dat : rom_down wide format (HIDDEN_CH*SIMD bits per line)
"""
import os

BASE = "/home/barkie1/mvau_pipeline/mvau_adapter"

# MVAU configurations: (N, PE, SIMD, IN_CH, OUT_CH, REDUCTION, data_dir, up_suffix)
CONFIGS = [
    (1, 32, 32, 64,  64,  4,
     os.path.join(BASE, "mvau1/data"), "adapter_1_up_pe32.dat"),
    (2, 16, 32, 64,  128, 4,
     os.path.join(BASE, "mvau2/data"), "adapter_2_up_pe16.dat"),
    (3, 16, 32, 128, 128, 4,
     os.path.join(BASE, "mvau3/data"), "adapter_3_up_pe16.dat"),
    (4, 4,  32, 128, 256, 4,
     os.path.join(BASE, "mvau4/data"), "adapter_4_up_pe4.dat"),
    (5, 1,  32, 256, 256, 4,
     os.path.join(BASE, "mvau5/data"), "adapter_5_up.dat"),
]


def read_dat_32bit(path):
    """Read a .dat file with one 32-bit hex value per line."""
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                vals.append(int(line, 16))
    return vals


def pack_up_weights(n, pe, out_steps, up_file, out_file):
    """Pack rom_up: PE entries of 32-bit per step -> one wide word per step.

    MVAU5 special: PE=1 but rom_up stores 2x32-bit per step (64-bit xnor).
    For MVAU5: pack 2 entries -> 64-bit per line.
    For MVAU1-4: pack PE entries -> PE*32-bit per line.

    Bit order: LSB = entry[k*stride + 0], MSB = entry[k*stride + stride-1]
    """
    vals = read_dat_32bit(up_file)

    if n == 5:
        stride = pe * 2  # = 2
        width_bits = 64
    else:
        stride = pe
        width_bits = pe * 32

    expected = out_steps * stride
    assert len(vals) == expected, \
        f"MVAU{n} up: expected {expected} entries, got {len(vals)}"

    hex_chars = width_bits // 4

    with open(out_file, "w") as f:
        for k in range(out_steps):
            wide = 0
            for p in range(stride):
                wide |= (vals[k * stride + p] & 0xFFFFFFFF) << (p * 32)
            f.write(f"{wide:0{hex_chars}x}\n")

    print(f"  rom_up packed: {out_steps} x {width_bits}-bit -> {os.path.basename(out_file)}")


def pack_down_weights(n, hidden_ch, simd, in_chunks, down_file, out_file):
    """Pack rom_down: flatten into wide format.

    Packing matches Verilog semantics:
      rom_down[r][c*SIMD +: SIMD] = rom_down_flat[c * IN_CHUNKS + r]

    Result: IN_CHUNKS lines, each HIDDEN_CH*SIMD bits wide.
    """
    vals = read_dat_32bit(down_file)
    expected = hidden_ch * in_chunks
    assert len(vals) == expected, \
        f"MVAU{n} down: expected {expected} entries, got {len(vals)}"

    width_bits = hidden_ch * simd
    hex_chars = width_bits // 4

    with open(out_file, "w") as f:
        for r in range(in_chunks):
            wide = 0
            for c in range(hidden_ch):
                val = vals[c * in_chunks + r] & 0xFFFFFFFF
                wide |= val << (c * simd)
            f.write(f"{wide:0{hex_chars}x}\n")

    print(f"  rom_down packed: {in_chunks} x {width_bits}-bit -> {os.path.basename(out_file)}")


def main():
    for n, pe, simd, in_ch, out_ch, reduction, data_dir, up_name in CONFIGS:
        hidden_ch = in_ch // reduction
        in_chunks = in_ch // simd
        out_steps = out_ch // pe

        print(f"\nMVAU{n}: PE={pe}, HIDDEN_CH={hidden_ch}, "
              f"IN_CHUNKS={in_chunks}, OUT_STEPS={out_steps}")

        # Pack UP weights
        up_file = os.path.join(data_dir, up_name)
        up_out = os.path.join(data_dir, f"adapter_{n}_up_packed.dat")
        pack_up_weights(n, pe, out_steps, up_file, up_out)

        # Pack DOWN weights
        down_file = os.path.join(data_dir, f"adapter_{n}_down.dat")
        down_out = os.path.join(data_dir, f"adapter_{n}_down_packed.dat")
        pack_down_weights(n, hidden_ch, simd, in_chunks, down_file, down_out)

    print("\nAll pre-packed .dat files generated.")


if __name__ == "__main__":
    main()
