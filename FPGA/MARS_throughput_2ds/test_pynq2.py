#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/xilinx")
from pynq import Overlay, MMIO, allocate
print("pynq imported from /home/xilinx")
ol = Overlay("resizer.bit")
print("Overlay loaded OK")
