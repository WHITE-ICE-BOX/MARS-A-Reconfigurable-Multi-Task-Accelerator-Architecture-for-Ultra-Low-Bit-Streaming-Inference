# ===========================================================================
# [交接導向註解]
# 匯出標準 CIFAR-10 測資供各階段比對。流程：FINN_Compile。
# ===========================================================================

# Dump canonical CIFAR-10 test set EXACTLY as FINN's validate.py consumes it
# (dataset_loading.cifar.load_cifar_data), so on-board validation uses the
# same data/preprocessing as the official FINN flow.
import numpy as np
from dataset_loading import cifar
trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
    "/tmp", download=True, one_hot=False)
testx = np.asarray(testx)
testy = np.asarray(testy)
print("testx", testx.shape, testx.dtype, "min", testx.min(), "max", testx.max())
print("testy", testy.shape, testy.dtype, "first10", testy[:10])
OUT = "/home/barkie1/thesis/finn/notebooks/end2end_example/bnn-pynq"
np.save(f"{OUT}/cifar10_canon_testx.npy", testx.astype(np.uint8))
np.save(f"{OUT}/cifar10_canon_testy.npy", testy.astype(np.int64))
print("saved canonical cifar to", OUT)
