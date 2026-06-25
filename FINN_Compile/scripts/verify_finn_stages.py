# ===========================================================================
# [交接導向註解]
# 驗證 FINN 各階段 ONNX 輸出的數值正確性。流程：FINN_Compile。
# ===========================================================================

# Staged FINN verification: run execute_onnx on each transform stage vs
# CIFAR-10 labels, to localize where 81% (PyTorch) -> 10% (HW) breaks.
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from finn.core.onnx_exec import execute_onnx

H = "/home/barkie1/thesis/finn/notebooks/end2end_example/bnn-pynq"
B = "/home/barkie1/thesis/finn/finn_cifar10"
X = np.load(f"{H}/cifar10_test_x.npy")  # uint8 NHWC (N,32,32,3)
Y = np.load(f"{H}/cifar10_test_y.npy")
N = 300
X, Y = X[:N], Y[:N]

def run(stage, expects_uint8):
    m = ModelWrapper(f"{B}/end2end_cnv_w1a1_{stage}.onnx")
    iname = m.graph.input[0].name
    oname = m.graph.output[0].name
    ishape = m.get_tensor_shape(iname)
    correct = 0
    for k in range(N):
        img = X[k].astype(np.float32)              # HWC 0..255
        if expects_uint8:
            inp = img.reshape(ishape)              # NHWC uint8 domain (0..255)
        else:
            inp = (img / 255.0)
            # tidy/streamlined-core may be NCHW
            if len(ishape) == 4 and ishape[1] == 3:
                inp = img.transpose(2, 0, 1) / 255.0
            inp = inp.reshape(ishape)
        out = execute_onnx(m, {iname: inp.astype(np.float32)})[oname]
        pred = int(np.argmax(out)) if out.size > 1 else int(out.flatten()[0])
        correct += int(pred == Y[k])
    print(f"{stage:14s} (uint8={expects_uint8}) in{ishape} -> acc {100.0*correct/N:.2f}% ({correct}/{N})")

for stage, u8 in [("pre_post", True), ("streamlined", True), ("folded", True)]:
    try:
        run(stage, u8)
    except Exception as e:
        print(f"{stage}: ERROR {type(e).__name__}: {str(e)[:200]}")
