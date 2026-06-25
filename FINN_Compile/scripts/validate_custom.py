# ===========================================================================
# [交接導向註解]
# 自訂資料集的 FINN 模型驗證。流程：FINN_Compile。
# ===========================================================================

# Identical to FINN's official validate.py EXCEPT the dataset comes from
# pre-dumped canonical .npy (board has no internet/dataset_loading).
# Verifies the PLAIN PE=1 cifar10_1w1a backbone bitstream the official way.
import argparse, numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batchsize", type=int, default=100)
    p.add_argument("--bitfile", default="resizer.bit")
    p.add_argument("--platform", default="zynq-iodma")
    p.add_argument("--datadir", default=".")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    testx = np.load(f"{a.datadir}/cifar10_canon_testx.npy")
    testy = np.load(f"{a.datadir}/cifar10_canon_testy.npy")
    if a.limit:
        testx, testy = testx[: a.limit], testy[: a.limit]
    total = testx.shape[0]
    bsize = a.batchsize
    n_batches = int(total / bsize)
    total = n_batches * bsize

    driver = FINNExampleOverlay(
        bitfile_name=a.bitfile, platform=a.platform,
        io_shape_dict=io_shape_dict, batch_size=bsize,
        runtime_weight_dir="runtime_weights/")

    test_imgs = testx[: n_batches * bsize].reshape(n_batches, bsize, -1)
    test_labels = testy[: n_batches * bsize].reshape(n_batches, bsize)
    ok = nok = 0
    for i in range(n_batches):
        ibuf = test_imgs[i].reshape(driver.ibuf_packed_device[0].shape)
        exp = test_labels[i]
        driver.copy_input_data_to_device(ibuf)
        driver.execute_on_buffers()
        obuf = np.empty_like(driver.obuf_packed_device[0])
        driver.copy_output_data_from_device(obuf)
        ret = np.bincount(obuf.flatten() == exp.flatten(), minlength=2)
        nok += ret[0]; ok += ret[1]
        print("batch %d/%d  OK %d NOK %d" % (i + 1, n_batches, ok, nok))
    print("Final accuracy: %.2f%% (%d/%d)" % (100.0 * ok / total, ok, total))
