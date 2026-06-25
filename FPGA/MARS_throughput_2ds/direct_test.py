#!/usr/bin/env python3
"""Direct FPGA diagnostic - bypasses PYNQ Overlay, uses /dev/mem."""
import mmap, struct, os, sys, time

class MMIO:
    def __init__(self, base, length=0x10000):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, length, offset=base)
    def write(self, offset, value):
        self.mm.seek(offset)
        self.mm.write(struct.pack("<I", value & 0xFFFFFFFF))
    def read(self, offset):
        self.mm.seek(offset)
        return struct.unpack("<I", self.mm.read(4))[0]
    def close(self):
        self.mm.close()
        os.close(self.fd)

# Check FPGA state
fpga_state = open("/sys/class/fpga_manager/fpga0/state").read().strip()
print(f"FPGA state: {fpga_state}")

if fpga_state != "operating":
    print("ERROR: FPGA not programmed! Need to load bitstream first.")
    sys.exit(1)

# cfg_hub at 0x43C00000
cfg = MMIO(0x43C00000, 0x10000)

CFG_ENABLE_WORD = 0
CFG_THRESH_BASE = 1152

def mvau_byte_addr(mvau_id, word_addr):
    return (mvau_id << 13) | (word_addr << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))

MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}

print("=== Testing cfg_hub MMIO writes ===")

# Read a few addresses (cfg_hub is write-only, should return 0)
for offset in [0, 4, 0x2000, 0x3200]:
    val = cfg.read(offset)
    print(f"  Read 0x{offset:04X}: 0x{val:08X}")

# Write adapter_enable=0 for all MVAUs
print("\nSetting all adapters OFF...")
for m in range(1, 6):
    cfg.write(mvau_byte_addr(m, CFG_ENABLE_WORD), 0)
print("Done.")

# Write CIFAR-10 thresholds
print("\nWriting CIFAR-10 thresholds...")
for m in range(1, 6):
    threshs = load_bin_u32(f"mvau{m}_thresh_cifar10.bin")
    print(f"  MVAU{m}: {len(threshs)} entries, first={threshs[0]}, last={threshs[-1]}")
    for i, val in enumerate(threshs):
        cfg.write(mvau_byte_addr(m, CFG_THRESH_BASE + i), val)
    cfg.write(mvau_byte_addr(m, CFG_ENABLE_WORD), 0)
print("Done. Thresholds written, adapter OFF for all MVAUs.")

# For DMA, we need to know the addresses - check UIO devices
print("\nChecking UIO devices...")
import glob
for uio in sorted(glob.glob("/sys/class/uio/uio*")):
    name = open(uio + "/name").read().strip()
    addr = open(uio + "/maps/map0/addr").read().strip()
    size = open(uio + "/maps/map0/size").read().strip()
    print(f"  {os.path.basename(uio)}: name={name}, addr={addr}, size={size}")

print("\nCfg_hub write test complete. Cannot run inference without DMA setup.")
print("Please run inference from Jupyter notebook.")
cfg.close()
