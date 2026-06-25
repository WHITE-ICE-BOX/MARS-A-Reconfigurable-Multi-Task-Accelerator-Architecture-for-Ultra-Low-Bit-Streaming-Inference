#!/usr/bin/env python3
import sys
sys.path.insert(0, "/usr/local/share/pynq-venv/lib/python3.8/site-packages")
from pynq import Overlay, Device
print("Devices:", Device.devices)
ol = Overlay("resizer.bit")
print("Overlay loaded OK")
