. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
PLUG=$H/plug/libggml-cl11.so
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
for v in t2p0_spin500 t2p0_spin0 t2p0_spin100 t1p50 t2p50_spin100; do
  case $v in
    t2p0_spin500) E="GGML_SGLX_MOE_THREADS=2 GGML_SGLX_MOE_SPIN_US=500"; P="--poll 0";;
    t2p0_spin0) E="GGML_SGLX_MOE_THREADS=2 GGML_SGLX_MOE_SPIN_US=0"; P="--poll 0";;
    t2p0_spin100) E="GGML_SGLX_MOE_THREADS=2 GGML_SGLX_MOE_SPIN_US=100"; P="--poll 0";;
    t1p50) E="GGML_SGLX_MOE_THREADS=1"; P="--poll 50";;
    t2p50_spin100) E="GGML_SGLX_MOE_THREADS=2 GGML_SGLX_MOE_SPIN_US=100"; P="--poll 50";;
  esac
  env $E GGML_SGLX_MOE=Vulkan0:auto:250 GGML_SGLX_PROFILE=1 GGML_SGLX_MOE_VERBOSE=1 $RUN $LOGD s9_tg_$v $GEN -m $MODEL -ub 512 -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" -r 2 $P --bench pp512,tg128
done
