. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
PLUG=$H/plug/libggml-cl11.so
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
GGML_SGLX_MOE=1 GGML_SGLX_PROFILE=1 $RUN $LOGD s1_cold_tg $GEN -m $MODEL -ub 512 -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" -r 1 --bench tg16
GGML_SGLX_MOE=Vulkan0:auto GGML_SGLX_MOE_VERBOSE=1 GGML_SGLX_PROFILE=1 $RUN $LOGD s1_vk_pp_tg $GEN -m $MODEL -ub 512 -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" -r 2 --bench pp512,tg128
