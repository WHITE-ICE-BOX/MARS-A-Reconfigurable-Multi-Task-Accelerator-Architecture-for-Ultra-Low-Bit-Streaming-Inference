import os, subprocess, re, csv, time, sys
THIS="/home/esl/barkie/bnn_pynq/bnn_pynq/claude"
TRAIN=THIS+"/bnn_pynq_train_bitwidth.py"
BP=THIS+"/pretrained_backbones/cifar10_1w1a.tar"
EXP=THIS+"/paper_results_bitwidth/b2_significance/experiments"
OUT=THIS+"/paper_results_bitwidth/b2_significance/results.csv"
os.makedirs(EXP, exist_ok=True)
SEEDS=[2024,2025,2026,2027,2028]
rows=[]
if os.path.exists(OUT):
    rows=[l.strip().split(',') for l in open(OUT)][1:]
done={(r[0],r[1],r[2]) for r in rows}
f=open(OUT,'a')
if os.path.getsize(OUT) if os.path.exists(OUT) else 0:
    pass
else:
    f.write("dataset,M,seed,best_acc\n"); f.flush()
for ds in ["SVHN","FashionMNIST"]:
    for M in [1,4]:
        for sd in SEEDS:
            key=(ds,str(M),str(sd))
            if key in done:
                print("skip",key,flush=True); continue
            ep=200; ms=f"{ep//2},{int(ep*0.75)}"
            name=f"b2_{ds}_M{M}_s{sd}"
            cmd=[sys.executable,'-u',TRAIN,'--mode','adapter','--net_bit','1','--dataset',ds,
                 '--finetune_checkpoint',BP,'--epochs',str(ep),'--lr','0.005','--scheduler','STEP',
                 '--milestones',ms,'--batch_size','100','--num_workers','4','--random_seed',str(sd),
                 '--experiments',EXP,'--experiment_name',name,'--num_branches',str(M),
                 '--adapter_bit_width','1','--adapter_kernel','1','--adapter_act','signed',
                 '--adapter_alpha','scalar','--adapter_mid_basis','in','--no_rc','--adapter_bias']
            env=dict(os.environ); env['CUDA_VISIBLE_DEVICES']='1'
            t0=time.time()
            print(f"START {name}",flush=True)
            p=subprocess.run(cmd,cwd=THIS,env=env,capture_output=True,text=True)
            m=re.findall(r"Final Best Accuracy:\s*([\d.]+)",p.stdout)
            acc=m[-1] if m else "ERR"
            f.write(f"{ds},{M},{sd},{acc}\n"); f.flush()
            print(f"DONE {name} acc={acc} ({(time.time()-t0)/60:.1f} min)",flush=True)
print("ALL_B2_DONE",flush=True)
