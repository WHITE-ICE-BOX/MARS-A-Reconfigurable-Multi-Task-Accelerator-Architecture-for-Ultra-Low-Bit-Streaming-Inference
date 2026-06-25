# ===========================================================================
# [交接導向註解]
# runner：工具：從訓練 log 擷取 Final Best Accuracy。
# 流程：AI_model_train。產出對應 results/ 之 results.csv（見 README 對照表）。
# ===========================================================================

import torch,glob,os
groups=["svhn_to_others_bits","svhn_to_others","stl10_to_others","cinic10_to_others_bits","fashionmnist_to_others_bits"]
ROOT="/home/esl/barkie/bnn_pynq/bnn_pynq/claude/paper_results_bitwidth"
for grp in groups:
    base=os.path.join(ROOT,grp,"experiments")
    if not os.path.isdir(base):
        print("##### "+grp+"  (not present)"); continue
    print("##### "+grp)
    for d in sorted(glob.glob(base+"/*")):
        name=os.path.basename(d)
        bt=os.path.join(d,"checkpoints","best.tar")
        if not os.path.exists(bt):
            print("  %-55s (no best.tar)"%name); continue
        try:
            ck=torch.load(bt,map_location="cpu")
            print("  %-55s acc=%s ep=%s"%(name,round(ck.get("best_val_acc",-1),2),ck.get("epoch",-1)))
        except Exception as ex:
            print("  %-55s ERR %s"%(name,str(ex)[:25]))
