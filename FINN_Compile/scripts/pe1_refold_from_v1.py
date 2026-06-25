"""
Take v1's already-streamlined dataflow_model.onnx (PE=32 baseline) and refold to PE=1.
Skip the broken Streamline step entirely.
"""
import os, shutil
os.environ["FINN_BUILD_DIR"] = "/home/barkie1/mvau_pipeline_runtime_3ds_pe1/finn_build_v1refold"
os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
build_dir = os.environ["FINN_BUILD_DIR"]

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

# Reuse v1's verified post-streamline model
src = "/home/barkie1/mvau_pipeline_runtime/finn/finn_pipeline_adapter/cnv_6layer_fc3_svhn_w1a1_dataflow_model.onnx"
dst_folded = build_dir + "/cnv_6layer_fc3_pe1_folded.onnx"

model = ModelWrapper(src)
fc_layers = model.get_nodes_by_op_type("MVAU_hls")
print(f"Found {len(fc_layers)} MVAU_hls nodes")

# Folding: PE=1 全部, SIMD/FIFO 沿用 v1
folding = [
    (1, 3,  [128]),    # MVAU_0
    (1, 32, [128]),    # MVAU_1
    (1, 32, [128]),    # MVAU_2
    (1, 32, [128]),    # MVAU_3
    (1, 32, [81]),     # MVAU_4
    (1, 32, [2]),      # MVAU_5
    (1, 4,  [2]),      # MVAU_6
    (1, 8,  [128]),    # MVAU_7
    (1, 1,  [3]),      # MVAU_8
]
for fcl, (pe, simd, ifdepth) in zip(fc_layers, folding):
    inst = getCustomOp(fcl)
    inst.set_nodeattr("PE", pe)
    inst.set_nodeattr("SIMD", simd)
    inst.set_nodeattr("inFIFODepths", ifdepth)
    print(f"  {fcl.name}: PE={pe}, SIMD={simd}, noAct={inst.get_nodeattr('noActivation')}")

# SWG SIMD alignment
swg = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
for i, n in enumerate(swg):
    if i < len(folding):
        getCustomOp(n).set_nodeattr("SIMD", folding[i][1])

model = model.transform(GiveUniqueNodeNames())
model.save(dst_folded)
print(f"\nSaved folded → {dst_folded}")

# Now do ZynqBuild + driver gen
print("\nStarting ZynqBuild (will take ~30 min)...")
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.util.basic import make_build_dir, pynq_part_map

pynq_board = "Pynq-Z2"
target_clk_ns = 10
model = model.transform(ZynqBuild(platform=pynq_board, period_ns=target_clk_ns))
model = model.transform(MakePYNQDriver("zynq-iodma"))
model.save(build_dir + "/cnv_6layer_fc3_pe1_synth.onnx")

# Copy bitstream + driver
deploy_dir = make_build_dir(prefix="pynq_deployment_")
model.set_metadata_prop("pynq_deployment_dir", deploy_dir)
bit = model.get_metadata_prop("bitfile")
hwh = model.get_metadata_prop("hw_handoff")
print(f"Bitfile: {bit}")
print(f"HWH:     {hwh}")
for f in [bit, hwh]:
    if f and os.path.isfile(f): shutil.copy(f, deploy_dir)
print(f"\nDeployment dir: {deploy_dir}")
