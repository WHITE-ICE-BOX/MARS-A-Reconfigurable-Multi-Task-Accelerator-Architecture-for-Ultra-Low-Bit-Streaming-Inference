# FINN_Compile

FINN end-to-end dataflow synthesis of the 1W1A CNV backbone (run inside the FINN Docker).
Produces the per-MVAU memblocks and the StreamingDataflowPartition IP consumed by `RTL/`.

## Final hardware came from
- `notebooks/pe1_cnv_end2end.ipynb` — backbone end2end for the **compact PE=1** builds
- `notebooks/pe1_adapter.ipynb` — adapter-side end2end (PE=1)
- `scripts/pe1_refold_from_v1.py` — refolds the verified PE=32 dataflow model down to PE=1
- `notebooks/cnv_end2end_example.ipynb`, `notebooks/adapter.ipynb`, `notebooks/backbone_cifar.ipynb`
  — the high-PE (throughput) variants

## Model (input to FINN)
- `model/CNV.py` — Brevitas CNV definition exported to ONNX
- `model/cnv_6layer_fc3_cifar_w1a1.zip` — trained CIFAR-10 1W1A model

## Outputs (LFS)
- `onnx/end2end_cnv_w1a1_folded.onnx`, `onnx/end2end_cnv_w1a1_dataflow_model.onnx`

## Verify
- `scripts/verify_finn_stages.py`, `scripts/verify_cifar1w1a.py`, `scripts/validate_custom.py`
