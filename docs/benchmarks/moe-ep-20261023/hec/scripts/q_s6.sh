. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
PLUG=$H/plug/libggml-cl11.so
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
for v in spin500 spin3000 spin20000 thr1; do
  case $v in
    spin500) E="GGML_SGLX_MOE_SPIN_US=500";;
    spin3000) E="GGML_SGLX_MOE_SPIN_US=3000";;
    spin20000) E="GGML_SGLX_MOE_SPIN_US=20000";;
    thr1) E="GGML_SGLX_MOE_THREADS=1";;
  esac
  env $E GGML_SGLX_MOE=Vulkan0:auto:250 GGML_SGLX_PROFILE=1 $RUN $LOGD s6_tg_$v $GEN -m $MODEL -ub 512 -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" -r 2 --bench pp256,tg128
done
