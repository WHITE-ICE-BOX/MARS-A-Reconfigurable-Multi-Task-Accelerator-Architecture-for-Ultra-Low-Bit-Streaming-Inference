#!/usr/bin/env python3
"""SOFTWARE-ONLY faster runtime switch (no RTL / no rebuild) — VECTORIZED.

v2: switch() does ZERO Python per-word work. All delta / static-once /
cifar-skip masking is numpy-vectorized; the only per-switch cost is a few
vector ops over ~6757 elements plus the actual contiguous-segment cfg
slice writes (the real AXI-Lite single-beat hardware cost).

Reductions (all pure software, correctness proven for any switch sequence
by tracking the actual hw cfg state in a numpy mirror):
  1. static-once : 926 cfg words identical across all 3 datasets written
                   ONCE at init, never on a switch (mask excludes them).
  2. delta       : write only words whose target value differs from what is
                   physically in cfg now (hw mirror, vectorized compare).
  3. cifar-skip  : switching to CIFAR-10 disables the adapter, so the
                   adapter-projection ROM words (rc/down/up/contrib of
                   MVAU1-5) are don't-care and skipped; their hw mirror is
                   set STALE so the next adapter-on switch rewrites them.

Floor ≈ 179 ns / cfg word (AXI-Lite single-beat). Projection:
  →CIFAR ≈ 0.40 ms (~2300 words), →SVHN/→Fashion ≈ 1.0 ms (~5500 words).
"""
import os, sys, time, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime_3ds import RuntimeSwitcher, CFG_BASE_DEFAULT

_STALE = np.uint32(0xDEADBEEF)


class FastRuntimeSwitcher(RuntimeSwitcher):
    def __init__(self, cfg_base=CFG_BASE_DEFAULT, weights_root=None):
        super().__init__(cfg_base=cfg_base, weights_root=weights_root)
        order = ("cifar10", "svhn", "fashion")
        N = int(self.cfg.shape[0])
        self._N = N

        # dense per-ds value vector + "present" mask over full cfg space
        self._dval = {}
        self._dpre = {}
        for ds in order:
            idxs, vals = self._cache[ds]
            v = np.zeros(N, dtype=np.uint32)
            p = np.zeros(N, dtype=bool)
            v[idxs] = vals
            p[idxs] = True
            self._dval[ds] = v
            self._dpre[ds] = p

        present_all = self._dpre["cifar10"] & self._dpre["svhn"] & self._dpre["fashion"]
        same_all = present_all & (self._dval["cifar10"] == self._dval["svhn"]) \
                                & (self._dval["cifar10"] == self._dval["fashion"])
        self._static_mask = same_all                       # write once, never again

        # adapter-projection word mask (don't-care when adapter disabled)
        adp = np.zeros(N, dtype=bool)
        for n in (1, 2, 3, 4, 5):
            for nm, base in (("rc", 4), ("down", 128), ("up", 640),
                             ("contrib", 1664)):
                arr = self._region_len(n, nm)
                for i in range(arr):
                    adp[self._unit_word(n, base + i)] = True
        self._adapter_mask = adp

        # hw mirror: what is physically in cfg
        self._hw = np.full(N, _STALE, dtype=np.uint32)

        # write static-once exactly now (values identical across ds)
        si = np.nonzero(self._static_mask)[0]
        if si.size:
            sv = self._dval["cifar10"][si]
            self._bulk(si.astype(np.uint32), sv)

    def _region_len(self, n, nm):
        p = os.path.join(self.weights_root, "cifar10", f"mvau{n}_{nm}.bin")
        return os.path.getsize(p) // 4

    def _bulk(self, si, sv):
        """si already sorted ascending. Contiguous-segment slice writes."""
        if si.size == 0:
            return 0
        # segment boundaries where index is non-contiguous
        brk = np.nonzero(np.diff(si) != 1)[0] + 1
        starts = np.concatenate(([0], brk))
        ends = np.concatenate((brk, [si.size]))
        for s, e in zip(starts, ends):
            a = int(si[s]); b = int(si[e - 1]) + 1
            self.cfg[a:b] = sv[s:e]
            self._hw[a:b] = sv[s:e]
        return int(si.size)

    def switch(self, dataset):
        t0 = time.time()
        val = self._dval[dataset]
        pre = self._dpre[dataset]
        to_cifar = (dataset == "cifar10")

        # words this ds defines, excluding static-once
        cand = pre & ~self._static_mask
        if to_cifar:
            # adapter words are don't-care; mark hw STALE and skip them
            skip = cand & self._adapter_mask
            self._hw[skip] = _STALE
            cand = cand & ~self._adapter_mask
        # delta: only where hw differs from target (all vectorized)
        chg = cand & (self._hw != val)
        si = np.nonzero(chg)[0].astype(np.uint32)
        n = self._bulk(si, val[si])
        return (time.time() - t0) * 1000.0, n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_root", default=None)
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    sw = FastRuntimeSwitcher(weights_root=a.weights_root)
    for ds in ("cifar10", "svhn", "fashion"):
        ms, n = sw.switch(ds)
        print(f"warmup -> {ds:8} {ms:.3f} ms  {n} words")
    seq = ["cifar10", "svhn", "fashion", "cifar10", "svhn", "cifar10", "fashion"]
    agg = {}
    for _ in range(a.rounds):
        for ds in seq:
            ms, n = sw.switch(ds)
            agg.setdefault(ds, []).append((ms, n))
    for ds, xs in agg.items():
        print(f"{ds:8}: min={min(x[0] for x in xs):.3f} ms "
              f"mean={sum(x[0] for x in xs)/len(xs):.3f} ms  words~{xs[-1][1]}")
