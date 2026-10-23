. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
PLUG=$H/plug/libggml-cl11.so
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
# more devices and more Q4_1 slots (decode tokens/s, random tokens; the prefill of 256 tokens makes the placement)
run() { tag=$1; spec=$2; shift 2; env "$@" GGML_SGLX_MOE=$spec GGML_SGLX_PROFILE=1 GGML_SGLX_MOE_VERBOSE=1 GGML_SGLX_MOE_THREADS=2 GGML_SGLX_MOE_SPIN_US=0 $RUN $LOGD s8_$tag $GEN -m $MODEL -ub 512 -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" -r 2 --poll 0 --bench pp256,tg128; }
run rx_q41_500 Vulkan0:auto:500 X=1
run rx_gt730 Vulkan0:auto:400,CL11_0:300 X=1
run rx_wx Vulkan0:auto:400,Vulkan1:700:100 X=1
run rx_gt730_wx_gt710 Vulkan0:auto:400,CL11_0:300,Vulkan1:700:100,CL11_1:700 X=1
