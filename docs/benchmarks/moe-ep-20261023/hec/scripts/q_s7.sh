. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
PLUG=$H/plug/libggml-cl11.so
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
for v in thr1_poll0 thr2_poll0 thr2_poll0_spin200; do
  case $v in
    thr1_poll0) E="GGML_SGLX_MOE_THREADS=1";;
    thr2_poll0) E="GGML_SGLX_MOE_THREADS=2 GGML_SGLX_MOE_SPIN_US=500";;
    thr2_poll0_spin200) E="GGML_SGLX_MOE_THREADS=2 GGML_SGLX_MOE_SPIN_US=200";;
  esac
  env $E GGML_SGLX_MOE=Vulkan0:auto:250 GGML_SGLX_PROFILE=1 $RUN $LOGD s7_tg_$v $GEN -m $MODEL -ub 512 -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" -r 2 --poll 0 --bench pp256,tg128
done
