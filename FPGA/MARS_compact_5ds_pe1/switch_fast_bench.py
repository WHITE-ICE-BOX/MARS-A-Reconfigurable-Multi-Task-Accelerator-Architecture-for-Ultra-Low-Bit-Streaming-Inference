# step9b: optimized cfg switch — group scattered writes into CONTIGUOUS u32
# segments, do slice assignment cfg[a:b]=block (C-level bulk) instead of
# fancy-index scatter cfg[idxs]=vals. Bench switch time on board.
# Does NOT modify user's runtime_3ds.py (separate copy/bench).
import sys, os, time, numpy as np
sys.path.insert(0, "/home/xilinx/runtime_3ds_pe1")
from runtime_3ds import RuntimeSwitcher

sw = RuntimeSwitcher(weights_root="/home/xilinx/runtime_3ds_pe1/runtime_weights")

# Precompute contiguous segments per dataset from the existing (idxs,vals) cache
seg_cache = {}
for ds in ("cifar10", "svhn", "fashion"):
    idxs, vals = sw._cache[ds]
    order = np.argsort(idxs, kind="stable")
    si = idxs[order]; sv = vals[order]
    segs = []
    start = 0
    for k in range(1, len(si) + 1):
        if k == len(si) or si[k] != si[k-1] + 1:
            segs.append((int(si[start]), sv[start:k].copy()))
            start = k
    seg_cache[ds] = segs
    print(f"{ds}: {len(idxs)} writes -> {len(segs)} contiguous segments")

def switch_fast(ds):
    t0 = time.time()
    for base, block in seg_cache[ds]:
        sw.cfg[base:base + len(block)] = block
    return (time.time() - t0) * 1000.0

# warm + bench both methods, 3 runs each
for ds in ("cifar10", "svhn", "fashion"):
    sw.switch(ds)                       # original (fancy scatter) warm
    o = [sw.switch(ds) for _ in range(3)]
    switch_fast(ds)                     # fast warm
    f = [switch_fast(ds) for _ in range(3)]
    print(f"{ds}: original(scatter)={min(o):.3f}ms  fast(slice)={min(f):.3f}ms")
