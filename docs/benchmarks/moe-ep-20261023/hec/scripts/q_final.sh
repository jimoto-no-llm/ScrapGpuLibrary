. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
export HEC_GEN_VERBOSE=1
# random-token bench (llama-bench inputs: pp512, tg128, pg512,128; 3 reps, a warm-up per test, new context per test), baselines alternating with the SGLX configuration N
PLUG0=$HOME/moe-ep/src/build-cl11/libggml-cl11.so
BENCH="--bench pp512,tg128,pg512:128 -r 3"
b_rx39()  { $RUN $LOGD "$1" $GEN -m $MODEL -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 39 $BENCH; }
b_rx41()  { $RUN $LOGD "$1" $GEN -m $MODEL -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 41 $BENCH; }
b_wx5()   { $RUN $LOGD "$1" $GEN -m $MODEL -ub 512 -dev Vulkan0/Vulkan1 -ngl 99 -ncmoe 34 -ts 48/0 -ot 'blk\.(34|35|36|37|38)\.ffn_.*_exps\.weight=Vulkan1' $BENCH; }
b_mix4()  { ( export GGML_BACKEND_PATH=$PLUG0 GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0; $RUN $LOGD "$1" $GEN -m $MODEL -ub 512 -dev Vulkan0/Vulkan1/CL11_0/CL11_1 -ngl 99 -ncmoe 29 -ts 38/5/2/3 $BENCH ); }
b_mc()    { $RUN $LOGD "$1" $GEN -m $MODEL -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 48 --moe-cache-mib 3000 $BENCH; }
n_cfg()   { run_n $LOGD "$1" $GEN -m $MODEL -ub 512 "${N_ARGS[@]}" $BENCH; }
for r in 1 2; do
  b_rx39 f_rx39_$r;  n_cfg f_N_a$r
  b_rx41 f_rx41_$r;  n_cfg f_N_b$r
  if [ $r = 1 ]; then
    b_wx5 f_wx5_$r;  n_cfg f_N_c$r
    b_mix4 f_mix4_$r; n_cfg f_N_d$r
    b_mc f_mc_$r
  fi
done
