#!/usr/bin/env python3
import sys, os
sys.path.insert(0, "/home/xilinx")
import pynq
print("PYNQ version:", pynq.__version__)
print("PYNQ path:", pynq.__file__)

# Try to manually create a device
from pynq.pl_server.device import Device
print("Device subclasses:", Device.__subclasses__())

# Check for Zynq-specific device
try:
    from pynq.pl_server.embedded_device import EmbeddedDevice
    print("EmbeddedDevice available")
    devs = EmbeddedDevice.devices
    print("Embedded devices:", devs)
except Exception as e:
    print("EmbeddedDevice error:", e)

# Check /sys/class for fpga_manager
print("fpga_manager:", os.listdir("/sys/class/fpga_manager/"))

# Try to just load bitstream directly
try:
    from pynq import PL
    print("PL:", PL.__dict__.keys() if hasattr(PL, '__dict__') else dir(PL))
except Exception as e:
    print("PL error:", e)
