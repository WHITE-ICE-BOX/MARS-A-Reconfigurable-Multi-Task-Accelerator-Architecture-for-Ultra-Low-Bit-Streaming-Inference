# AI_model_train — 1W1A 訓練與 Conv-Adapter 遷移學習

以 **PyTorch + Brevitas** 做 1W1A（1-bit weight / 1-bit activation）的 **quantization-aware training（QAT）**：先預訓練 backbone，凍結後在每層接 **Conv-Adapter** 做遷移學習。
來源：RTX 4090（`~/barkie/bnn_pynq/bnn_pynq`）；跨來源（cross-source）批次另在 A6000 上跑。

---

## 一、結構樹（每個檔案/結果在做什麼）

```
AI_model_train/
├── src/                                # ── 訓練核心程式碼 ──
│   ├── bnn_pynq_train_bitwidth.py      # 主訓練入口（mode=full_ft/adapter、任意 bit-width；參數見第四節）
│   ├── bnn_pynq_train.py               # 舊版單 bit 訓練入口（保留供參考）
│   ├── trainer.py                      # 訓練/評估迴圈、checkpoint 存取、Final Best Accuracy 輸出
│   ├── logger.py                       # 訓練 log 格式
│   ├── models/                         # ── 基礎模型定義 ──
│   │   ├── CNV.py                      #   CNV backbone（Brevitas QuantConv2d/Linear，1W1A）
│   │   ├── FC.py                       #   全連接分類頭
│   │   ├── common.py                   #   量化層/工具（make_quant_conv 等）
│   │   ├── tensor_norm.py              #   TensorNorm（BN 變體）
│   │   ├── losses.py                   #   SqrHinge 等損失
│   │   └── resnet.py                   #   ResNet（探索用，論文未採用）
│   └── models_bitwidth/                # ── 可參數化（任意 bit-width）模型 + Adapter ──
│       ├── CNV_param.py                #   cnv_param()：依 cfg 產生任意位元寬度 CNV
│       ├── conv_adapter.py            #   單一 Conv-Adapter（down-proj → sign → up-proj + RC bias）
│       └── multi_adapter.py           #   多分支 Adapter（M 條平行分支聚合）
│
├── runners/                            # ── 各實驗驅動腳本（對照表見第三節）──
│   ├── run_xx_pretrain.py
│   ├── run_xx_to_others.py
│   ├── run_xx_to_others_bits.py
│   ├── run_v6.py  run_v6_bit.py  run_v6_cross.py  run_v6_cross_m.py  run_v7_multi_rc.py
│   ├── run_configC_bits.py  run_configC_bits_rc.py  run_configC_cross.py  run_configC_sw_multi.py
│   ├── run_v9.py  run_v9_ft.py  run_v9ft_cross_bit.py
│   ├── run_v3_compare.py
│   ├── run_seed.py  _b2_runner.py  _dump_acc.py
│
├── backbones/                          # ── 預訓練 backbone（PyTorch checkpoint .tar）──
│   ├── cifar10_1w1a.tar  svhn_1w1a.tar  stl10_1w1a.tar  fashionmnist_1w1a.tar  cinic10_1w1a.tar
│   └── *_{2w2a,4w4a,8w8a,16w16a,32w32a}.tar   # 供 bit-width sweep 表使用（共 24 個）
│
└── results/                            # ── 結果數據（只放 results.csv，不放 6.4GB 原始 log）──
    ├── final_accuracy_summary.txt      #   全實驗 Final Best Accuracy 彙整
    ├── svhn_to_others/ stl10_to_others/ fashionmnist_to_others/   # 跨來源 1-bit（→ 跨來源多 Adapter 表）
    ├── cifar10_to_others_bits/ svhn_to_others_bits/               # 跨來源 bit-width（→ 跨來源位元表）
    ├── cifar10_configC_bits{,_rc}/ cifar10_configC_cross/         # CIFAR→targets 位元/跨資料集（accuracy-best）
    ├── configC_sw_multi/                                          # 軟體版多 Adapter 補格
    ├── v6/ v_v6_bit/ v_v6_cross/ v_v6_cross_m/ v7_multi_rc/       # accuracy-best 幾何 + RC 消融
    ├── v9_cross_dataset/ v9_ft_baseline/ v9ft_cross_bit/         # full-FT 上界（1-bit 與 bit-width）
    ├── v3_compare/ v3_compare_50ep_2026-05-03/                    # kernel 3×3 vs 1×1 單軸消融（§5.3.6）
    ├── v_seed/                                                    # n=3 multi-seed 變異
    ├── b2_significance/                                           # n=5 paired t-test 顯著性
    └── a6000_crosssource/                                         # 在 A6000 上跑的跨來源 CSV
```

---

## 二、兩種 Adapter 幾何（geometry）

| 幾何 | down-conv kernel | hidden 寬度（mid） | α scaling | 用途 |
|---|---|---|---|---|
| **accuracy-best** | 3×3 | $C_{\text{out}}/4$（mid='out'） | per-channel | 軟體準確率上界 |
| **deployed** | 1×1 | $C_{\text{in}}/4$（mid='in'） | scalar | FPGA 部署版（省資源），燒進 bitstream 的組態 |

`runners/` 中 `v6` / `configC` 家族 = **accuracy-best**；**deployed** 幾何以 `--adapter_kernel 1 --adapter_alpha scalar --adapter_mid_basis in` 訓練（見 `_b2_runner.py`），其 checkpoint 即匯出到 `FPGA/.../runtime_weights/`。

---

## 三、runner 對照表

| 腳本 | 做什麼 | 對應結果資料夾 |
|---|---|---|
| `run_xx_pretrain.py` | 預訓練 5 個 1W1A backbone | `backbones/` |
| `run_xx_to_others.py` | 跨來源遷移：每個 backbone → 其餘 4 target，1-bit、M=1–4 + full-FT | `*_to_others/` |
| `run_xx_to_others_bits.py` | 跨來源 bit-width sweep（1/2/4/8-bit） | `*_to_others_bits/` |
| `run_v6.py` | accuracy-best（v6）設計，CIFAR→SVHN 基準 | `v6/` |
| `run_v7_multi_rc.py` | 多 Adapter（M=1–4）× RC 開關消融 | `v7_multi_rc/` |
| `run_v6_cross.py` / `run_v6_cross_m.py` | accuracy-best 跨資料集 M-sweep（M=4 / M=1–3） | `v_v6_cross{,_m}/` |
| `run_v6_bit.py` | v6 × bit-width × M=4 | `v_v6_bit/` |
| `run_configC_bits{,_rc}.py` | CIFAR→targets 位元掃描（no-RC / 含 RC） | `cifar10_configC_bits{,_rc}/` |
| `run_configC_cross.py` | accuracy-best 1-bit 跨資料集 | `cifar10_configC_cross/` |
| `run_configC_sw_multi.py` | 補齊軟體多 Adapter 表缺的 9 格 | `configC_sw_multi/` |
| `run_v9.py` | v9 adapter 主實驗（single-seed headline） | `v9_cross_dataset/` |
| `run_v9_ft.py` | full-FT 上界（1-bit） | `v9_ft_baseline/` |
| `run_v9ft_cross_bit.py` | full-FT × bit-width（2–32 bit） | `v9ft_cross_bit/` |
| `run_v3_compare.py` | 3×3 vs 1×1 down-conv 單軸消融（50ep、no-RC） | `v3_compare*/` |
| `run_seed.py` | n=3 multi-seed | `v_seed/` |
| `_b2_runner.py` | n=5 paired t-test（deployed，SVHN/Fashion × M1/M4） | `b2_significance/` |
| `_dump_acc.py` | 工具：從 log 擷取 Final Best Accuracy | — |

---

## 四、主入口參數（`bnn_pynq_train_bitwidth.py`）

| 旗標 | 說明 |
|---|---|
| `--mode` | `full_ft`（全網路微調）/ `adapter`（凍結 backbone 只訓 adapter） |
| `--net_bit` | 網路位元寬度 1/2/4/8… |
| `--dataset` | CIFAR10 / SVHN / STL10 / FashionMNIST / CINIC10 |
| `--finetune_checkpoint` | 載入的 backbone（如 `backbones/cifar10_1w1a.tar`） |
| `--num_branches` | M（平行 Adapter 分支數） |
| `--adapter_kernel` `--adapter_alpha` `--adapter_mid_basis` | adapter 幾何（`3`/`1`、`per_channel`/`scalar`、`out`/`in`） |
| `--no_rc` `--adapter_bias` | RC 開關（`--no_rc` 基底 +`--adapter_bias` 打開 RC） |

### 範例（deployed 幾何、M=4 + RC、CIFAR→SVHN）
```bash
python src/bnn_pynq_train_bitwidth.py --mode adapter --net_bit 1 --dataset SVHN \
  --finetune_checkpoint backbones/cifar10_1w1a.tar --num_branches 4 \
  --adapter_kernel 1 --adapter_alpha scalar --adapter_mid_basis in --no_rc --adapter_bias \
  --epochs 200 --lr 0.005 --scheduler STEP --milestones 100,150 --batch_size 100
```

`results/*/results.csv` 欄位：`dataset, mode, M, rc, acc, params, returncode`。原始逐 epoch log（約 6.4 GB）未收錄。
