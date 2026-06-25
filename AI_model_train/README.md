# AI_model_train

PyTorch + Brevitas 1W1A quantization-aware training and Conv-Adapter transfer learning.
Origin: RTX 4090 (`~/barkie/bnn_pynq/bnn_pynq`), cross-source batches mirrored on A6000.

## Layout
- `src/` — training core
  - `bnn_pynq_train_bitwidth.py` — main train entry (modes: `full_ft` / `adapter`; per-bit-width)
  - `trainer.py`, `logger.py`
  - `models/` — CNV / FC / resnet / losses / tensor_norm
  - `models_bitwidth/` — parametrizable CNV (`CNV_param.py` → `cnv_param`)
- `runners/` — experiment drivers
  - `run_xx_pretrain.py` — backbone pre-training (per source dataset)
  - `run_xx_to_others{,_bits}.py` — cross-source transfer (1-bit / bit-width sweep)
  - `run_experiment_{cinic10,fashionmnist,stl10}.py`
  - `run_configC_*` (deployed geometry), `run_v6*` (accuracy-best geometry), `run_seed.py`
  - `_b2_runner.py` — multi-seed (n=5) significance runs (SVHN/Fashion × M1/M4)
- `backbones/` — pretrained 1W1A..32W32A backbones for all 5 datasets (Git LFS)
- `results/` — `results.csv` per experiment + `final_accuracy_summary.txt`
  - `a6000_crosssource/` — cross-source `results.csv` computed on the A6000

## Example
```
python src/bnn_pynq_train_bitwidth.py --mode adapter --net_bit 1 --dataset SVHN \
  --finetune_checkpoint backbones/cifar10_1w1a.tar --num_branches 4 \
  --adapter_kernel 1 --adapter_alpha scalar --adapter_mid_basis in --no_rc --adapter_bias \
  --epochs 200 --lr 0.005 --scheduler STEP --milestones 100,150
```
