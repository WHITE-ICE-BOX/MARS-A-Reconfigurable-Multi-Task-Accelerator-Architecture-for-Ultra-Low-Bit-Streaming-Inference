# ===========================================================================
# [交接導向註解]
# 腳本：產生 .npy golden（端到端正確性比對用）。
# 流程：訓練/FINN → RTL（產生硬體 .dat/ROM 與 golden）。
# ===========================================================================

import sys, glob, configparser, numpy as np, torch
sys.path.insert(0,'.'); sys.path.insert(0,'claude')
from torchvision import transforms
from torchvision.datasets import STL10
from PIL import Image
from models_bitwidth.CNV_param import cnv_param

def build_model(ckpt):
    cfg=configparser.ConfigParser()
    for s,kv in {'QUANT':{'WEIGHT_BIT_WIDTH':'1','ACT_BIT_WIDTH':'1','IN_BIT_WIDTH':'8'},
     'MODEL':{'NUM_CLASSES':'10','IN_CHANNELS':'3'},
     'ADAPTER':{'NUM_BRANCHES':'1','BIT_WIDTH':'1','RC_BIT_WIDTH':'8','USE_RC':'True',
       'KERNEL_SIZE':'1','ACT_MODE':'signed','ALPHA_MODE':'scalar','USE_BIAS':'True','MID_BASIS':'in'}}.items():
        cfg.add_section(s); [cfg.set(s,k,v) for k,v in kv.items()]
    m=cnv_param(cfg); c=torch.load(ckpt,map_location='cpu',weights_only=False); sd=c.get('state_dict',c)
    miss,unexp=m.load_state_dict(sd,strict=False); miss=[k for k in miss if 'num_batches' not in k]
    print("   load: missing",len(miss),"unexpected",len(unexp),"best_val",c.get('best_val_acc'))
    return m.cuda().eval()

def golden(m, x_uint8, y):
    # x_uint8: (N,32,32,3) uint8 -> /255 NCHW
    import torch
    ok=tot=0
    for i in range(0,len(x_uint8),1000):
        xb=torch.from_numpy(x_uint8[i:i+1000].transpose(0,3,1,2).astype('float32')/255.0).cuda()
        p=m(xb).argmax(1).cpu().numpy(); yb=y[i:i+1000]
        ok+=(p==yb).sum(); tot+=len(yb)
    return 100.0*ok/tot

# ---- STL10 ----
print("=== STL10 ===")
ds=STL10(root='./data', split='test', download=True)
rz=transforms.Resize(32)
xs=np.stack([np.array(rz(img.convert('RGB')),dtype=np.uint8) for img,_ in ds])  # (8000,32,32,3)
ys=np.array([lbl for _,lbl in ds],dtype=np.int64)
print("  stl10 npy", xs.shape, xs.dtype, "min",xs.min(),"max",xs.max())
np.save('claude/_stl10_test_x.npy',xs); np.save('claude/_stl10_test_y.npy',ys)
ck=glob.glob('claude/**/Transfer_v9_STL10_M1_rc_e200/checkpoints/best.tar',recursive=True)[0]
m=build_model(ck); print(f"  STL10 software golden(npy /255) = {golden(m,xs,ys):.2f}%  (n={len(ys)})")

# ---- CINIC10 availability ----
print("=== CINIC10 data check ===")
import os
cands=glob.glob('**/CINIC*/test',recursive=True)+glob.glob('**/cinic*/test',recursive=True)
print("  cinic test dirs:", cands[:3] if cands else "NONE FOUND")
