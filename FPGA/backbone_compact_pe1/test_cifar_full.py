import os, sys, time
sys.path.append('/home/xilinx/pe1_backbone')
import numpy as np
from driver_base import FINNExampleOverlay
from driver import io_shape_dict
ol = FINNExampleOverlay(bitfile_name="/home/xilinx/pe1_backbone/resizer.bit", platform="zynq-iodma",
    io_shape_dict=io_shape_dict, batch_size=1, runtime_weight_dir="/home/xilinx/pe1_backbone/runtime_weights/")
x_all = np.load('/home/xilinx/pe1_backbone/cifar10_test_x.npy')
y_all = np.load('/home/xilinx/pe1_backbone/cifar10_test_y.npy')
N = len(x_all)
print(f'Running {N} CIFAR-10 samples...')
correct = 0
t0 = time.time()
for i in range(N):
    out = ol.execute(x_all[i:i+1])
    if int(out[0][0]) == int(y_all[i]):
        correct += 1
elapsed = time.time() - t0
print(f'CIFAR-10 PE=1 backbone (full): acc = {correct}/{N} = {100*correct/N:.2f}%, FPS = {N/elapsed:.1f}, total = {elapsed:.1f}s')
