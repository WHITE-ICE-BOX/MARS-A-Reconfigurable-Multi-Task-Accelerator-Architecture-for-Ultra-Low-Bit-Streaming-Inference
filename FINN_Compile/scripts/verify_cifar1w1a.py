# ===========================================================================
# [交接導向註解]
# CIFAR-10 1W1A 端到端正確性檢查。流程：FINN_Compile。
# ===========================================================================

# Verify cifar10_1w1a checkpoint in PURE PyTorch (brevitas_examples CNV,
# exactly as notebook cell 7). Decisive: which preprocessing gives ~81%?
import torch, numpy as np
from brevitas_examples.bnn_pynq.models.CNV import CNV

_HERE = "/home/barkie1/thesis/finn/notebooks/end2end_example/bnn-pynq"
X = np.load(f"{_HERE}/cifar10_test_x.npy")  # uint8 NHWC
Y = np.load(f"{_HERE}/cifar10_test_y.npy")
print("X", X.shape, X.dtype, "min/max", X.min(), X.max())
N = 2000
X = X[:N];
if Y is not None: Y = Y[:N]

m = CNV(num_classes=10, weight_bit_width=1, act_bit_width=1, in_bit_width=8, in_ch=3)
ck = torch.load(f"{_HERE}/Cifar10_backbone.tar", map_location="cpu")  # = cifar10_1w1a now
missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
print("missing keys:", len(missing), missing[:5])
print("unexpected  :", len(unexpected), unexpected[:5])
m.eval()

xt = torch.from_numpy(X.astype(np.float32)).permute(0,3,1,2)  # NCHW
mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
std  = torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)

def acc(t):
    with torch.no_grad():
        p = []
        for i in range(0, t.shape[0], 200):
            o = m(t[i:i+200])
            p.append(o.argmax(1).cpu().numpy())
    p = np.concatenate(p)
    if Y is None: return None, p[:10]
    return 100.0*float((p==Y).mean()), p[:10]

for name, t in [
    ("x/255            ", xt/255.0),
    ("x/255 then mean/std", (xt/255.0 - mean)/std),
    ("raw x (0-255)     ", xt),
]:
    a, pr = acc(t)
    print(f"{name}: acc={a}  preds[:10]={pr}")
