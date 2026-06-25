import argparse
import subprocess
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
import shutil

# === Configuration ===
# 所有輸出都放在 claude/ 資料夾內
CLAUDE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CLAUDE_DIR)

BASE_OUTPUT_DIR = os.path.join(CLAUDE_DIR, "paper_results_stl10")
EXP_DIR = os.path.join(BASE_OUTPUT_DIR, "experiments")
PLOT_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")
MODEL_OUT_DIR = os.path.join(BASE_OUTPUT_DIR, "stage1_models")
BACKBONE_ROOT = os.path.join(PROJECT_ROOT, "experiments")

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

def run_cmd(cmd, exp_name):
    """執行指令並即時顯示 Log，最後回傳最佳準確率與參數量"""
    print(f"\n>>> 正在執行實驗: {exp_name} <<<")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    best_acc = 0.0
    total_params = 0

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end='')
            sys.stdout.flush()

            if "Final Best Accuracy" in line:
                try:
                    raw_acc = float(line.split(':')[-1].replace('%', '').strip())
                    best_acc = round(raw_acc, 2)
                except: pass

            if "[Model Stats] Total Params" in line:
                try:
                    total_params = int(line.split(':')[-1].strip())
                except: pass

    if process.returncode != 0:
        print(f"❌ 實驗失敗: {exp_name}")
        return 0.0, 0

    print(f"✅ 實驗完成: Acc={best_acc}%, Params={total_params}")
    return best_acc, total_params

# === 繪圖功能 ===
def plot_stage1(df, args):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    configs, accs, params = df['Config'], df['Accuracy'], df['Params']

    bars = ax1.bar(configs, accs, color='skyblue', label='Accuracy', alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height}%', ha='center', va='bottom', fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(configs, params, color='red', marker='o', linewidth=2, linestyle='--', label='Parameters')
    ax2.set_ylabel('Total Parameters', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    for i, p in enumerate(params):
        ax2.text(i, p, f'{p:,}', ha='center', va='bottom', color='red', fontsize=9, fontweight='bold')

    plt.title(f"{args.source} -> {args.target} ({args.net_bit}-bit) Architecture Search", fontsize=14, pad=20)
    save_path = os.path.join(PLOT_DIR, f"Stage1_{args.source}_to_{args.target}_{args.net_bit}bit.png")
    plt.savefig(save_path)
    print(f"🖼️  圖表已儲存: {save_path}")

# ==========================================
# Stage 1: 架構搜尋與權重自動提取
# ==========================================
def run_stage_1(args):
    print(f"🚀 [Stage 1] 開始架構搜尋與權重提取: {args.source} -> {args.target}")

    backbone_path = os.path.join(BACKBONE_ROOT, f"{args.source}_backbone.tar")
    if not os.path.exists(backbone_path):
        print(f"❌ 錯誤: 找不到原始骨幹 {backbone_path}")
        return

    experiments = [
        ("Baseline_m0", 0, False),
        ("Adapter_m1",  1, False),
        ("RC_m1",       1, True),
        ("RC_m2",       2, True),
        ("RC_m3",       3, True),
        ("RC_m4",       4, True),
    ]

    data_log = []

    # 訓練腳本路徑: 使用 claude/ 內的擴展版
    train_script = os.path.join(CLAUDE_DIR, "bnn_pynq_train_ext.py")

    for label, m, rc in experiments:
        exp_name = f"S1_{args.source}2{args.target}_b{args.net_bit}_{label}"

        cmd = [
            sys.executable, "-u", train_script,
            "--network", "CNV",
            "--dataset", args.target,
            "--datadir", os.path.join(PROJECT_ROOT, "data"),
            "--finetune_checkpoint", backbone_path,
            "--freeze_backbone",
            "--num_branches", str(m),
            "--random_seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--lr", "0.005",
            "--milestones", "30,40",
            "--experiments", EXP_DIR,
            "--experiment_name", exp_name,
            "--override_wbw", str(args.net_bit),
            "--override_abw", str(args.net_bit),
            "--adapter_bit_width", str(args.net_bit),
            "--export_finn_assets"
        ]

        if not rc: cmd.append("--no_rc")

        acc, params = run_cmd(cmd, exp_name)
        data_log.append({'Config': label, 'Accuracy': acc, 'Params': params})

        # [自動提取邏輯]
        print(f"📂 [Extraction] 正在從 {exp_name} 提取權重...")
        current_exp_dir = os.path.join(EXP_DIR, exp_name)

        # A. 搬運 Full Model
        src_full = os.path.join(current_exp_dir, "checkpoints", "best.tar")
        dst_full = os.path.join(MODEL_OUT_DIR, f"{label}_full.tar")
        if os.path.exists(src_full):
            shutil.copy2(src_full, dst_full)
            print(f"   ✅ Full Model -> {dst_full}")

        # B. 搬運 Backbone (trainer.py 硬編碼為 svhn.tar)
        src_back = os.path.join(current_exp_dir, "finn_export", "svhn.tar")
        dst_back = os.path.join(MODEL_OUT_DIR, f"{label}_backbone.tar")
        if os.path.exists(src_back):
            shutil.copy2(src_back, dst_back)
            print(f"   ✅ Backbone Model -> {dst_back}")

    # 儲存 CSV 與 畫圖
    df = pd.DataFrame(data_log)
    csv_path = os.path.join(BASE_OUTPUT_DIR, f"Stage1_{args.source}2{args.target}_{args.net_bit}bit.csv")
    df.to_csv(csv_path, index=False)
    plot_stage1(df, args)


# ==========================================
# Main & Parse Args
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Automated Experiment: CIFAR10 -> STL10")
    parser.add_argument("--stage", type=int, choices=[1], required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--source", type=str, default="Cifar10")
    parser.add_argument("--target", type=str, default="STL10")
    parser.add_argument("--net_bit", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.stage == 1: run_stage_1(args)

if __name__ == "__main__":
    main()
