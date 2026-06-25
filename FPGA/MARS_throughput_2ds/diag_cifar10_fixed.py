#!/usr/bin/env python3
"""
Comprehensive CIFAR-10 mode diagnostic.
Tests multiple hypotheses for why CIFAR-10 mode gives wrong results.
"""
import struct, time, os, numpy as np
from pynq import Overlay, MMIO, allocate

ol = Overlay("/home/xilinx/runtime_switch/resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)

CFG_ENABLE_WORD = 0
CFG_THRESH_BASE = 1152
CFG_SIGN_BASE   = 1408

MVAU_OUT_CH = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256}
MVAU_PE     = {1: 32, 2: 16, 3: 16, 4: 4, 5: 1}
MVAU_STEPS  = {k: MVAU_OUT_CH[k] // MVAU_PE[k] for k in range(1, 6)}

def mvau_byte_addr(mvau_id, word_addr):
    return (mvau_id << 13) | (word_addr << 2)

def load_bin_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))

# DMA setup
idma = ol.idma0
odma = ol.odma0
ibuf = allocate(shape=(1, 32, 32, 3, 1), dtype=np.uint8, cacheable=True)
obuf = allocate(shape=(1, 1, 1), dtype=np.uint8, cacheable=True)

deploy_dir = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn"
img = np.load(deploy_dir + "/input.npy")
ibuf[:] = img.reshape(1, 32, 32, 3, 1)

def run_single():
    ibuf.flush()
    obuf[:] = 0
    obuf.flush()
    odma.write(0x10, obuf.device_address)
    odma.write(0x1C, 1)
    odma.write(0x00, 1)
    idma.write(0x10, ibuf.device_address)
    idma.write(0x1C, 1)
    idma.write(0x00, 1)
    for _ in range(200000):
        if odma.read(0x00) & 0x2:
            break
    obuf.invalidate()
    return int(np.array(obuf).flatten()[0])

def write_thresh(mvau_id, values):
    for i, val in enumerate(values):
        mmio.write(mvau_byte_addr(mvau_id, CFG_THRESH_BASE + i), val)

def write_signs(mvau_id, values):
    for i, val in enumerate(values):
        mmio.write(mvau_byte_addr(mvau_id, CFG_SIGN_BASE + i), val)

def set_enable(mvau_id, enable):
    mmio.write(mvau_byte_addr(mvau_id, CFG_ENABLE_WORD), 1 if enable else 0)

# Load threshold data
cifar10_thresh = {}
svhn_thresh = {}
svhn_sign = {}
for m in range(1, 6):
    cifar10_thresh[m] = load_bin_u32(f"mvau{m}_thresh_cifar10.bin")
    svhn_thresh[m] = load_bin_u32(f"mvau{m}_thresh_svhn.bin")
    svhn_sign[m] = load_bin_u32(f"mvau{m}_sign_svhn.bin")

print("=" * 70)
print("CIFAR-10 Mode Diagnostic (Golden test image, expected class 3)")
print("=" * 70)

# Test 0: Fresh overlay, just set adapter OFF (ROM defaults = SVHN fused thresholds)
print("\n--- Test 0: ROM defaults, adapter OFF ---")
for m in range(1, 6):
    set_enable(m, False)
pred = run_single()
print(f"  pred={pred} (ROM=SVHN-fused, adapter OFF → expected wrong)")

# Test 1: Write SVHN thresholds + signs, adapter ON
print("\n--- Test 1: SVHN mode (write SVHN thresh+sign, adapter ON) ---")
for m in range(1, 6):
    write_thresh(m, svhn_thresh[m])
    write_signs(m, svhn_sign[m])
    set_enable(m, True)
pred = run_single()
print(f"  pred={pred} (SVHN mode → should be some SVHN class)")

# Test 2: Write CIFAR-10 thresholds, adapter OFF (the failing case)
print("\n--- Test 2: CIFAR-10 mode (write CIFAR-10 thresh, adapter OFF) ---")
for m in range(1, 6):
    write_thresh(m, cifar10_thresh[m])
    set_enable(m, False)
pred = run_single()
print(f"  pred={pred} (CIFAR-10 mode → expected 3, FAILS)")

# Test 3: Write CIFAR-10 thresholds but KEEP adapter ON
print("\n--- Test 3: CIFAR-10 thresh but adapter ON (wrong: adapter adds noise) ---")
for m in range(1, 6):
    write_thresh(m, cifar10_thresh[m])
    set_enable(m, True)
pred = run_single()
print(f"  pred={pred} (CIFAR-10 thresh + adapter ON → probably wrong)")

# Test 4: Reload overlay to reset ROMs, then write CIFAR-10 thresh
print("\n--- Test 4: Reload overlay, then write CIFAR-10 thresh, adapter OFF ---")
ol = Overlay("/home/xilinx/runtime_switch/resizer.bit")
mmio = MMIO(0x43C00000, 0x10000)
idma = ol.idma0
odma = ol.odma0
for m in range(1, 6):
    write_thresh(m, cifar10_thresh[m])
    set_enable(m, False)
pred = run_single()
print(f"  pred={pred} (fresh reload + CIFAR-10 thresh → expected 3)")

# Test 5: Write CIFAR-10 thresh with EXPLICIT delays between writes
print("\n--- Test 5: CIFAR-10 thresh with delays between MVAUs ---")
for m in range(1, 6):
    write_thresh(m, cifar10_thresh[m])
    set_enable(m, False)
    time.sleep(0.01)  # 10ms delay
pred = run_single()
print(f"  pred={pred} (with delays)")

# Test 6: Write CIFAR-10 thresh, then verify by writing again
print("\n--- Test 6: Double-write CIFAR-10 thresh ---")
for _ in range(3):
    for m in range(1, 6):
        write_thresh(m, cifar10_thresh[m])
        set_enable(m, False)
pred = run_single()
print(f"  pred={pred} (triple-written)")

# Test 7: Progressive - add one MVAU at a time
print("\n--- Test 7: Progressive MVAU switching (SVHN → CIFAR-10) ---")
# Start with SVHN mode
for m in range(1, 6):
    write_thresh(m, svhn_thresh[m])
    write_signs(m, svhn_sign[m])
    set_enable(m, True)
pred = run_single()
print(f"  All SVHN: pred={pred}")

for switch_m in range(1, 6):
    write_thresh(switch_m, cifar10_thresh[switch_m])
    set_enable(switch_m, False)
    pred = run_single()
    print(f"  MVAU1-{switch_m}=CIFAR10, rest=SVHN: pred={pred}")

# Test 8: Write CIFAR-10 thresh for ONLY one MVAU, others get all-0 thresh
# This isolates each MVAU's behavior
print("\n--- Test 8: One MVAU at CIFAR-10 thresh, others all-0 thresh, all adapter OFF ---")
for test_m in range(1, 6):
    # Set all to thresh=0 (output = all 1s), adapter OFF
    for m in range(1, 6):
        write_thresh(m, [0] * MVAU_OUT_CH[m])
        set_enable(m, False)
    # Now set test_m to CIFAR-10
    write_thresh(test_m, cifar10_thresh[test_m])
    pred = run_single()
    print(f"  Only MVAU{test_m} at CIFAR-10, others=0: pred={pred}")

# Test 9: All adapter OFF, all thresh=0 vs all thresh=MAX
print("\n--- Test 9: Extreme thresholds ---")
for m in range(1, 6):
    write_thresh(m, [0] * MVAU_OUT_CH[m])
    set_enable(m, False)
pred = run_single()
print(f"  All thresh=0, adapter OFF: pred={pred} (all outputs = 1)")

for m in range(1, 6):
    write_thresh(m, [0x7FFFFFFF] * MVAU_OUT_CH[m])
    set_enable(m, False)
pred = run_single()
print(f"  All thresh=MAX, adapter OFF: pred={pred} (all outputs = 0)")

# Test 10: Run golden test multiple times to check consistency
print("\n--- Test 10: Consistency check (5 runs with CIFAR-10 thresh) ---")
for m in range(1, 6):
    write_thresh(m, cifar10_thresh[m])
    set_enable(m, False)
preds = []
for i in range(5):
    preds.append(run_single())
print(f"  Predictions: {preds} (should be consistent)")

# Test 11: Compare first few threshold values
print("\n--- Test 11: Threshold value comparison (SVHN vs CIFAR-10) ---")
for m in [1, 5]:
    print(f"\n  MVAU{m}: (SVHN_fused vs CIFAR10_q8)")
    for ch in range(min(8, MVAU_OUT_CH[m])):
        sv = svhn_thresh[m][ch]
        c10 = cifar10_thresh[m][ch]
        pe = ch % MVAU_PE[m]
        step = ch // MVAU_PE[m]
        print(f"    ch{ch:3d} (PE{pe:2d} step{step:2d}): SVHN={sv:10d} (0x{sv:08X}), CIFAR10={c10:10d} (0x{c10:08X}), diff={c10-sv:+d}")

print("\n" + "=" * 70)
ibuf.freebuffer()
obuf.freebuffer()
