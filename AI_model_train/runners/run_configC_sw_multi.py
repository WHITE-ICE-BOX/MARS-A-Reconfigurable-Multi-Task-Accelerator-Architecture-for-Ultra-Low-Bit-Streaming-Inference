#!/usr/bin/env python3
# ===========================================================================
# [交接導向註解]
# runner：補齊軟體多 Adapter 表缺的 9 格。
# 流程：AI_model_train。產出對應 results/ 之 results.csv（見 README 對照表）。
# ===========================================================================

"""Software-version (config C / v6: kernel=3, ReLU, per-channel alpha, mid='out')
multi-branch (M=2,3,4) WITH RC, CIFAR-10 -> {FashionMNIST, STL10, CINIC10}, 1-bit, 200ep.
Fills the 9 missing cells of the software-version multi-adapter table.
Same config as cifar10_configC_bits_rc (M1 baseline) and v8_mid_out (SVHN M2-4).
One job per GPU (memory-safe). Resumable.
"""
import os, sys, subprocess, threading, queue, csv
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
BACKBONE = os.path.join(THIS_DIR, 'pretrained_backbones', 'cifar10_1w1a.tar')
TRAIN = os.path.join(THIS_DIR, 'bnn_pynq_train_bitwidth.py')
OUT = os.path.join(THIS_DIR, 'paper_results_bitwidth', 'configC_sw_multi')
EXP = os.path.join(OUT, 'experiments'); LOG = os.path.join(OUT, 'logs')
os.makedirs(EXP, exist_ok=True); os.makedirs(LOG, exist_ok=True)
TARGETS = ['FashionMNIST', 'STL10', 'CINIC10']; MS = [2, 3, 4]; GPUS = [0, 1]
EPOCHS=200; LR=0.005; BS=100; NW=2; SEED=2024
MSTEP=f"{EPOCHS//2},{int(EPOCHS*0.75)}"
lock = threading.Lock()
def done(label):
    try: return "Final Best Accuracy" in open(os.path.join(LOG, label+".log")).read()
    except: return False
def build(target, M):
    name=f"Transfer_configCsw_{target}_M{M}_rc_e{EPOCHS}"; label=f"{target}_M{M}_rc"
    cmd=[sys.executable,'-u',TRAIN,'--mode','adapter','--net_bit','1','--dataset',target,
        '--finetune_checkpoint',BACKBONE,'--epochs',str(EPOCHS),'--lr',str(LR),
        '--scheduler','STEP','--milestones',MSTEP,'--batch_size',str(BS),
        '--num_workers',str(NW),'--random_seed',str(SEED),'--experiments',EXP,
        '--experiment_name',name,'--num_branches',str(M),'--adapter_bit_width','1',
        '--adapter_kernel','3','--adapter_act','relu','--adapter_alpha','per_channel',
        '--adapter_mid_basis','out','--no_rc','--adapter_bias']
    return label, cmd
def run_cell(target, M, gpu):
    label, cmd = build(target, M)
    if done(label):
        with lock: print("SKIP(done):", label); return
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
    fp=open(os.path.join(LOG, label+".log"),'w')
    with lock: print(f">>> [GPU{gpu}] {label}", flush=True)
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=PROJ_ROOT,env=env)
    acc=0.0
    for line in p.stdout:
        fp.write(line); fp.flush()
        if "Final Best Accuracy" in line:
            try: acc=round(float(line.split(':')[-1].replace('%','').strip()),2)
            except: pass
    p.wait(); fp.close()
    with lock: print(f"DONE {label}: {acc}", flush=True)
work=queue.Queue()
for t in TARGETS:
    for M in MS: work.put((t,M))
def worker(gpu):
    while True:
        try: t,M=work.get_nowait()
        except queue.Empty: return
        run_cell(t,M,gpu); work.task_done()
def main():
    ths=[threading.Thread(target=worker,args=(g,)) for g in GPUS]
    for th in ths: th.start()
    for th in ths: th.join()
    with open(os.path.join(OUT,'results.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['target','M','acc'])
        for t in TARGETS:
            for M in MS:
                lab=f"{t}_M{M}_rc"; a=''
                try:
                    for line in open(os.path.join(LOG,lab+".log")):
                        if "Final Best Accuracy" in line: a=line.split(':')[-1].replace('%','').strip()
                except: pass
                w.writerow([t,M,a])
    print("ALL DONE. results.csv written.", flush=True)
if __name__=='__main__': main()
