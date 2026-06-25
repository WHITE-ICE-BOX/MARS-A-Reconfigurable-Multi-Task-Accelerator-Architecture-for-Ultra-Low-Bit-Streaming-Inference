import os, sys, numpy as np
SCRIPT_DIR='/home/xilinx/runtime_switch'
sys.path.append(SCRIPT_DIR)
import runtime_3ds_v1bit as R
from pynq import Overlay
from pynq.ps import Clocks

def zero_banks(cfg, mvaus):
    n=0
    for mvau in mvaus:
        for wb, names in [(4,(f'mvau{mvau}_rom_rc_svhn.bin',f'mvau{mvau}_rc_svhn.bin')),
                          (128,(f'mvau{mvau}_rom_down_svhn.bin',f'mvau{mvau}_down_svhn.bin')),
                          (640,(f'mvau{mvau}_rom_up_svhn.bin',f'mvau{mvau}_up_svhn.bin'))]:
            for fn in names:
                a=R.load_bin(f'{SCRIPT_DIR}/{fn}')
                if a is not None:
                    R.write_bank(cfg,mvau,wb,np.zeros(len(a),dtype=np.uint32)); n+=len(a); break
    print(f'  -> zeroed {n} words for MVAU {list(mvaus)}')

ol=Overlay(f'{SCRIPT_DIR}/resizer_v1.bit'); Clocks.fclk0_mhz=100.0
cfg=R.open_cfg_mmio()
print('=== A: SVHN baseline ==='); R.write_dataset(cfg,'svhn',True); R.run_dataset_test(ol,'svhn')
print('=== B: SVHN + zero MVAU2-5 adapter rom ==='); R.write_dataset(cfg,'svhn',True); zero_banks(cfg,(2,3,4,5)); R.run_dataset_test(ol,'svhn')
print('=== C: SVHN + zero MVAU1 only (positive control) ==='); R.write_dataset(cfg,'svhn',True); zero_banks(cfg,(1,)); R.run_dataset_test(ol,'svhn')
