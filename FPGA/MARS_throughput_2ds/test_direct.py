#!/usr/bin/env python3
"""Direct FPGA access without Overlay - uses fpga_manager and mmap."""
import mmap, struct, os, time, numpy as np

def program_fpga(bitfile):
    """Program FPGA using Linux fpga_manager."""
    import shutil
    # Copy bitstream to /lib/firmware
    shutil.copy(bitfile, "/lib/firmware/resizer.bit")
    # Check if .bin format needed
    with open(bitfile, "rb") as f:
        header = f.read(4)
    
    if header == b'\xff\xff\xff\xff' or header[:2] == b'\x00\x09':
        # Already in .bit format, convert to .bin
        import subprocess
        # Use fpga_manager flag for full bitstream
        with open("/sys/class/fpga_manager/fpga0/flags", "w") as f:
            f.write("0")
        with open("/sys/class/fpga_manager/fpga0/firmware", "w") as f:
            f.write("resizer.bit")
    print("FPGA programmed (or already programmed)")

# Check if FPGA is already programmed
fpga_state = open("/sys/class/fpga_manager/fpga0/state").read().strip()
print(f"FPGA state: {fpga_state}")

if fpga_state != "operating":
    print("Programming FPGA...")
    program_fpga("resizer.bit")
    time.sleep(1)

# Use direct mmap for MMIO
class DirectMMIO:
    def __init__(self, base_addr, length):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, length, offset=base_addr)
        self.base = base_addr
        self.length = length
    
    def write(self, offset, value):
        self.mm.seek(offset)
        self.mm.write(struct.pack("<I", value & 0xFFFFFFFF))
    
    def read(self, offset):
        self.mm.seek(offset)
        return struct.unpack("<I", self.mm.read(4))[0]
    
    def close(self):
        self.mm.close()
        os.close(self.fd)

# MMIO for cfg_hub
mmio = DirectMMIO(0x43C00000, 0x10000)

# Check DMA registers
dma_mmio = DirectMMIO(0x43C20000, 0x10000)  # Typical DMA base

# Test basic MMIO access
print("cfg_hub MMIO test (read at offset 0):", hex(mmio.read(0)))
print("DMA base test:", hex(dma_mmio.read(0)))

# We need the actual DMA base addresses - check what's available
print("\nChecking for DMA controllers...")
import subprocess
result = subprocess.run(["cat", "/proc/device-tree/amba/dma@43C20000/compatible"], capture_output=True, text=True)
print(f"DMA compatible: {result.stdout}")

# Check all memory-mapped devices
result = subprocess.run(["cat", "/proc/iomem"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "43c" in line.lower() or "dma" in line.lower() or "finn" in line.lower():
        print(f"  {line}")

mmio.close()
dma_mmio.close()
print("Basic MMIO access works!")
