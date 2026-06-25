#!/usr/bin/env python3
"""On-board isolation test (memory How-to-apply #2). Forces v3 bitstream.

Four SVHN runs with controlled cfg perturbations to localise the adapter-on
failure:
  A. SVHN normal               (baseline, expect ~9.7% observed-broken)
  B. SVHN, adapter_enable=0     (keep SVHN thresh/mvau0/fc/cls, drop adapter)
  C. SVHN, adapter rom zeroed   (enable=1 but rc/down/up=0)
  D. CIFAR normal               (sanity, expect ~74%)

Interpretation:
  B sensible & A garbage           -> adapter contribution path is the culprit
  C == A (both garbage)            -> adapter rom writes never take effect on HW
  C != A                           -> rom writes DO land; bug is value/compute
  B also garbage                   -> SVHN mvau0/fc/cls/thresh writes are wrong
"""
import numpy as np, os, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/xilinx/runtime_3ds_pe1/data"
sys.path.insert(0, SCRIPT_DIR)

MVAU_CFG = {1:dict(H=16,IC=2,OC=64),2:dict(H=16,IC=2,OC=128),
            3:dict(H=32,IC=4,OC=128),4:dict(H=32,IC=4,OC=256),
            5:dict(H=64,IC=8,OC=256)}

def run_inf(ol, tx, ty, bs=100):
    from pynq import allocate
    idma, odma = ol.idma0, ol.odma0
    n = (tx.shape[0]//bs)*bs
    ib = allocate(shape=(bs,32,32,3,1), dtype=np.uint8, cacheable=True)
    ob = allocate(shape=(bs,1,1), dtype=np.uint8, cacheable=True)
    cor=0
    for b in range(n//bs):
        s,e=b*bs,(b+1)*bs
        np.copyto(ib, tx[s:e].astype(np.uint8).reshape(bs,32,32,3,1)); ib.flush()
        odma.write(0x10,ob.device_address); odma.write(0x1C,bs); odma.write(0x00,1)
        idma.write(0x10,ib.device_address); idma.write(0x1C,bs); idma.write(0x00,1)
        while odma.read(0x00)&0x2==0: pass
        ob.invalidate()
        cor += int(np.sum(np.array(ob).flatten().astype(np.int64)==ty[s:e]))
    ib.freebuffer(); ob.freebuffer()
    return 100.0*cor/n, cor, n

def idx_for(unit, word):  # matches RuntimeSwitcher._unit_word
    return ((unit<<13)>>2)+word

def main():
    from pynq import Overlay
    from pynq.ps import Clocks
    from runtime_3ds import RuntimeSwitcher
    bit = os.path.join(SCRIPT_DIR,"resizer_3ds_v3.bit")
    print(f"Loading {bit}"); ol=Overlay(bit); Clocks.fclk0_mhz=100.0
    sw=RuntimeSwitcher(weights_root=os.path.join(SCRIPT_DIR,"runtime_weights"))

    sx=np.load(f"{DATA_DIR}/svhn_test_x.npy"); sy=np.load(f"{DATA_DIR}/svhn_test_y.npy")
    cx=np.load(f"{DATA_DIR}/cifar10_test_x.npy"); cy=np.load(f"{DATA_DIR}/cifar10_test_y.npy")
    sx,sy = sx[:5000], sy[:5000]; cx,cy = cx[:5000], cy[:5000]

    def apply_blob(idxs, vals):
        sw.cfg[idxs]=vals

    # --- A: SVHN normal ---
    i,v = sw._cache["svhn"]
    apply_blob(i,v); run_inf(ol,sx[:100],sy[:100])
    accA,_,_ = run_inf(ol,sx,sy); print(f"A SVHN normal          : {accA:.2f}%")

    # --- B: SVHN, adapter_enable=0 (zero the enable words = unit*2048+0) ---
    i2,v2 = i.copy(), v.copy()
    en_idx = set(idx_for(u,0) for u in (1,2,3,4,5))
    for k in range(len(i2)):
        if int(i2[k]) in en_idx: v2[k]=0
    apply_blob(i2,v2); run_inf(ol,sx[:100],sy[:100])
    accB,_,_ = run_inf(ol,sx,sy); print(f"B SVHN adapter_en=0    : {accB:.2f}%")

    # --- C: SVHN, adapter rom (rc/down/up) zeroed, enable stays 1 ---
    i3,v3 = i.copy(), v.copy()
    rom_idx=set()
    for u in (1,2,3,4,5):
        c=MVAU_CFG[u]
        for w in range(4, 4+c["H"]):              rom_idx.add(idx_for(u,w))   # rc
        for w in range(128,128+c["IC"]*c["H"]):   rom_idx.add(idx_for(u,w))   # down
        upw = c["OC"]*(2 if u==5 else 1)
        for w in range(640,640+upw):              rom_idx.add(idx_for(u,w))   # up
    for k in range(len(i3)):
        if int(i3[k]) in rom_idx: v3[k]=0
    apply_blob(i3,v3); run_inf(ol,sx[:100],sy[:100])
    accC,_,_ = run_inf(ol,sx,sy); print(f"C SVHN rom zeroed      : {accC:.2f}%")

    # --- D: CIFAR normal (sanity) ---
    ci,cv = sw._cache["cifar10"]; apply_blob(ci,cv); run_inf(ol,cx[:100],cy[:100])
    accD,_,_ = run_inf(ol,cx,cy); print(f"D CIFAR normal (sanity): {accD:.2f}%")

    print("\n--- VERDICT ---")
    print(f"A(svhn)={accA:.1f} B(en0)={accB:.1f} C(rom0)={accC:.1f} D(cifar)={accD:.1f}")
    if abs(accA-accC) < 1.5:
        print("C ~= A  => adapter rom writes have NO effect on HW (cfg->adapter rom dead)")
    else:
        print("C != A  => adapter rom writes DO land; bug is value/compute path")

if __name__=="__main__":
    main()
