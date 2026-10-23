. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
PLUG=$H/plug/libggml-cl11.so
T=$HOME/moe-ep/models/tiny-moe.gguf
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
D="--n-tokens 100 --n-prefill 40"
# references
GGML_SGLX_MOE= $RUN $LOGD s0_ref_cpu $GEN -m $T -dev none -ngl 0 --dump $LOGD/s0_ref_cpu.bin $D
GGML_SGLX_MOE= $RUN $LOGD s0_ref_vk $GEN -m $T -dev Vulkan0 -ngl 99 -ncmoe 6 --dump $LOGD/s0_ref_vk.bin $D
# SGLX0: all cold; Vulkan0 slots; Vulkan0 + CL11
GGML_SGLX_MOE=1 GGML_SGLX_PROFILE=1 $RUN $LOGD s0_cold $GEN -m $T -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" --dump $LOGD/s0_cold.bin $D
GGML_SGLX_MOE=Vulkan0:200:30 GGML_SGLX_MOE_VERBOSE=1 GGML_SGLX_PROFILE=1 $RUN $LOGD s0_vk $GEN -m $T -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" --dump $LOGD/s0_vk.bin $D
GGML_SGLX_MOE=Vulkan0:100:30,Vulkan1:60:10,CL11_0:60,CL11_1:60 GGML_SGLX_MOE_VERBOSE=1 GGML_SGLX_PROFILE=1 $RUN $LOGD s0_multi $GEN -m $T -dev Vulkan0/Vulkan1/SGLX0 -ts 1/0/0 -ngl 99 -ot "$OT" --dump $LOGD/s0_multi.bin $D
for f in cold vk multi; do $GEN --compare $LOGD/s0_ref_vk.bin $LOGD/s0_$f.bin > $LOGD/s0_cmp_$f.out; done
$GEN --compare $LOGD/s0_ref_cpu.bin $LOGD/s0_ref_vk.bin > $LOGD/s0_cmp_refs.out
