#!/usr/bin/env python3
"""SOFTWARE-ONLY switch: write the FULL per-dataset cfg set (all 6757 words,
bit-identical to baseline) but as pre-computed CONTIGUOUS-SEGMENT slice
assignments instead of one fancy-index scatter. No delta, no skip → provably
identical cfg state as the shipped RuntimeSwitcher (zero correctness risk).

This isolates the only software lever that actually helps: the full 6757-word
set collapses into a SMALL number of large contiguous segments, so a handful
of cfg[a:b]=block slices can beat the single fancy-index scatter on this board.
(Delta-encoding fails because the changed-word set fragments into hundreds of
tiny segments whose per-slice overhead exceeds the saving — see bench.)
"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime_3ds import RuntimeSwitcher, CFG_BASE_DEFAULT


class SegBulkRuntimeSwitcher(RuntimeSwitcher):
    def __init__(self, cfg_base=CFG_BASE_DEFAULT, weights_root=None):
        super().__init__(cfg_base=cfg_base, weights_root=weights_root)
        self._seg = {}
        for ds in ("cifar10", "svhn", "fashion"):
            idxs, vals = self._cache[ds]          # already sorted by _build_blob
            si = idxs.astype(np.int64)
            sv = vals.astype(np.uint32)
            brk = np.nonzero(np.diff(si) != 1)[0] + 1
            starts = np.concatenate(([0], brk))
            ends = np.concatenate((brk, [si.size]))
            segs = [(int(si[s]), int(si[e - 1]) + 1, sv[s:e].copy())
                    for s, e in zip(starts, ends)]
            self._seg[ds] = segs

    def switch(self, dataset):
        t0 = time.time()
        for a, b, blk in self._seg[dataset]:
            self.cfg[a:b] = blk
        return (time.time() - t0) * 1000.0, len(self._seg[dataset])
