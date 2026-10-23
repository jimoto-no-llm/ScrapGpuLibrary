. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
PLUG=$H/plug/libggml-cl11.so
T=$HOME/moe-ep/models/tiny-moe.gguf
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
# cold cpu path of the tiny model: where do ~10 ms per call go?
for v in default poll0 thr1 passive spin0; do
  P="--poll 50"; E=""
  case $v in
    poll0) P="--poll 0";;
    thr1) E="GGML_SGLX_MOE_THREADS=1";;
    passive) E="OMP_WAIT_POLICY=PASSIVE";;
    spin0) E="GOMP_SPINCOUNT=0";;
  esac
  env $E GGML_SGLX_MOE=1 GGML_SGLX_PROFILE=1 $RUN $LOGD s2_tiny_cold_$v $GEN -m $T -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" -r 1 $P --bench tg64
done
