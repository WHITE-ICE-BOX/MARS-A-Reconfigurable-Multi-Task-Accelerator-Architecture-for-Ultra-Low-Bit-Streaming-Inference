#!/usr/bin/env python3
import numpy as np, os, glob

deploy = "/home/xilinx/jupyter_notebooks/finn-cnv-test/pynq_deployment_zl8sy1tn"
print("Deploy dir contents:")
for f in sorted(os.listdir(deploy)):
    sz = os.path.getsize(os.path.join(deploy, f))
    print("  %s (%d bytes)" % (f, sz))

img = np.load(deploy + "/input.npy")
print("\nGolden image shape=%s dtype=%s" % (img.shape, img.dtype))
print("  min=%d max=%d mean=%.1f" % (img.min(), img.max(), img.mean()))

for f in ["output.npy", "expected.npy", "expected_output.npy"]:
    p = os.path.join(deploy, f)
    if os.path.exists(p):
        data = np.load(p)
        print("  %s: %s = %s" % (f, data.shape, data.flatten()[:10]))

base = "/home/xilinx/jupyter_notebooks/finn-cnv-test"
print("\nAll dirs in base:")
for d in sorted(os.listdir(base)):
    print("  %s" % d)

cifar_dir = deploy + "/cifar10_finn_dataset"
svhn_dir = deploy + "/svhn_finn_dataset"
print("\nCIFAR-10 dataset: %s" % os.path.isdir(cifar_dir))
print("SVHN dataset: %s" % os.path.isdir(svhn_dir))

# Check driver.py
dp = deploy + "/driver.py"
if os.path.exists(dp):
    with open(dp) as f:
        for i, line in enumerate(f):
            if i < 5:
                print("driver.py:%d: %s" % (i, line.rstrip()))
