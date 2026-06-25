#!/usr/bin/env python3
"""Minimal diagnostic: load v2 bitstream, set clock, write one cfg word, do 1 DMA."""
import os, sys, mmap, numpy as np, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"diag_v2 starting...", flush=True)

from pynq import Overlay, allocate
from pynq.ps import Clocks

bit_path = os.path.join(SCRIPT_DIR, "resizer.bit")
print(f"Loading {bit_path}...", flush=True)
ol = Overlay(bit_path)
print("Overlay loaded.", flush=True)
Clocks.fclk0_mhz = 50.0   # rule out timing; pure functional test
print(f"fclk0_mhz = {Clocks.fclk0_mhz}", flush=True)

idma = ol.idma0; odma = ol.odma0
print(f"idma={idma}, odma={odma}", flush=True)

# Try a single MMIO write to cfg_hub
fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
mem = mmap.mmap(fd, 0x10000, offset=0x40010000)
os.close(fd)
cfg = np.frombuffer(mem, dtype=np.uint32)
print("MMIO mapped at 0x40010000. Writing one word...", flush=True)
cfg[0] = 0xDEADBEEF
print(f"  wrote cfg[0]; read back: 0x{cfg[0]:08x}", flush=True)

print("\nNow trying single-sample DMA flush...", flush=True)
ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)
ibuf[:] = 0; ibuf.flush()
print("  buffers allocated.", flush=True)

odma.write(0x10, obuf.device_address); odma.write(0x1C, 1); odma.write(0x00, 1)
print("  odma started.", flush=True)
idma.write(0x10, ibuf.device_address); idma.write(0x1C, 1); idma.write(0x00, 1)
print("  idma started.", flush=True)

t0 = time.time()
last_log = t0
while odma.read(0x00) & 0x2 == 0:
    now = time.time()
    if now - last_log > 5:
        print(f"  t={now-t0:.1f}s waiting; odma=0x{odma.read(0x00):x} idma=0x{idma.read(0x00):x}", flush=True)
        last_log = now
    if now - t0 > 300:
        print(f"  TIMEOUT after 120s; odma status = 0x{odma.read(0x00):x}", flush=True)
        break
print(f"  DMA done in {time.time()-t0:.3f}s; odma status = 0x{odma.read(0x00):x}", flush=True)

obuf.invalidate()
print(f"  output = {int(obuf[0,0,0])}", flush=True)
print("diag_v2 done.", flush=True)
