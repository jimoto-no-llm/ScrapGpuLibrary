. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
PLUG=$H/plug/libggml-cl11.so
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
run() { tag=$1; poll=$2; shift 2; env "$@" GGML_SGLX_MOE_SPIN_US=0 GGML_SGLX_MOE=Vulkan0:auto:250 GGML_SGLX_PROFILE=1 GGML_SGLX_MOE_VERBOSE=1 \
    $RUN $LOGD s10_$tag $GEN -m $MODEL -ub 512 -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" -r 2 --poll $poll --bench pp512,tg128; }
run ev1 0 GGML_SGLX_MOE_EVENT=1 GGML_SGLX_MOE_THREADS=2      # the device is started right after the submit (event record)
run ev0 0 GGML_SGLX_MOE_EVENT=0 GGML_SGLX_MOE_THREADS=2      # ... or at the synchronize
run ev1_poll50 50 GGML_SGLX_MOE_EVENT=1 GGML_SGLX_MOE_THREADS=2   # llama.cpp's pool polls (llama-bench default)
run ev1_t1_poll50 50 GGML_SGLX_MOE_EVENT=1 GGML_SGLX_MOE_THREADS=1
