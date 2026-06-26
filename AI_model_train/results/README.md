# results/ — 軟體實驗數據對照

> **數字真實來源（single source of truth）**：`final_accuracy_summary.txt`（從所有訓練 log
> 擷取的 Final Best Accuracy）。各 `results.csv` 為各 runner 的原始批次輸出，**僅供原始紀錄**；
> 部分含失敗格（`acc=0.0, returncode=1`）或為被取代的幾何，請以 summary / 對應表為準。

## 一、資料夾 → 論文表 對照

| 資料夾 | 對應內容（論文） | 幾何 |
|---|---|---|
| `results.csv`（頂層） | CIFAR-10→SVHN 1-bit：adapter 73.72 / full-FT 94.91 / bit-sweep | — |
| `cifar10_configC_bits{,_rc}/` | CIFAR→targets 位元寬度掃描（單 Adapter no-RC / 含 RC） | accuracy-best |
| `cifar10_configC_cross/` | CIFAR backbone → 各 target 1-bit 多 Adapter | accuracy-best |
| `cifar10_to_others_bits/` | CIFAR→targets 位元掃描（含 full-FT 上界） | — |
| `svhn_to_others_bits/` | 跨來源位元寬度掃描 | — |
| `a6000_crosssource/*_configC_cross/` | **跨來源 1-bit 多 Adapter（Table「跨來源多 Adapter」）** | accuracy-best (mid='out') |
| `configC_sw_multi/` | 軟體多 Adapter 補格（CIFAR→{Fashion,STL10,CINIC10}, M2–4） | accuracy-best |
| `v6/ v_v6_bit/ v_v6_cross/ v_v6_cross_m/` | accuracy-best 幾何（v6）各掃描 | accuracy-best |
| `v7_multi_rc/` | 多 Adapter × RC 開關消融；**deployed(v1v2) SVHN M1–4=73.72/77.16/78.77/79.81** | accuracy-best + deployed |
| `v9_cross_dataset/` | **deployed 跨資料集（CINIC10 M2–4=65.16/65.38/65.45）** | deployed |
| `v9_ft_baseline/ v9ft_cross_bit/` | full-FT 上界（1-bit 與位元寬度） | — |
| `v3_compare/ v3_compare_50ep_*/` | kernel 3×3 vs 1×1 單軸消融 | — |
| `v_seed/` | n=3 multi-seed 變異 | — |
| `b2_significance/` | n=5 paired t-test 顯著性 | deployed |
| `{svhn,stl10,fashionmnist}_to_others/` | **雙重角色**：`full_ft` 列＝Table 5.4 full-FT 欄來源（**採用**，如 SVHN→CIFAR10=81.70）；`adapter` 列為 mid='in'（**已被 configC 取代，勿用於多 Adapter 表**） | full_ft 採用 / adapter mid='in' 棄用 |

## 二、幾何（geometry）說明

- **accuracy-best**（軟體上界）：kernel 3×3、mid='out'（Cout/4）、α per-channel。`configC` / `v6` 家族。
- **deployed**（FPGA 部署版）：kernel 1×1、mid='in'（Cin/4）、α scalar。`v9` 家族 + `b2`。
- `*_to_others/` 的 **adapter 列**是早期 **mid='in'** 跨來源跑，與論文 Table（accuracy-best/mid='out'）
  幾何不同、數值也不同（例 SVHN→CINIC10 M4：mid='in'=36.7 vs accuracy-best=39.70），故 adapter 列**不採用**；
  但同資料夾的 **full_ft 列**為 Table 5.4 full-FT 欄之來源（**採用**）。跨來源多 Adapter 數值請看
  `a6000_crosssource/*_configC_cross/`（mid='out'）。

## 三、已知資料衛生問題（誠實標註）

- **headline 數字部分只在 summary**：如 accuracy-best SVHN M4 = **84.44%**、CINIC10 M4 = **65.02%**
  為 seed-2024 單次跑，記錄於 `final_accuracy_summary.txt` 與 log，**未進結構化 CSV**。
- **失敗格**：`a6000_crosssource/*_configC_cross/` 與 `*_configC_bits_rc/` 有少數
  `acc=0.0, returncode=1` 的 CINIC10 cell（該次跑掛、後續補跑），有效值見 summary。
- **`configC_sw_multi/results.csv`** 無 SVHN 列、CINIC10 M4 為空（補格用，SVHN 主表在 `v7_multi_rc`）。
