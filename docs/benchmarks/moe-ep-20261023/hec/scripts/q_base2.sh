. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
P=$H/prompts
PLUG=$HOME/moe-ep/src/build-cl11/libggml-cl11.so
# --- moe_cache: the full random-token bench
$RUN $LOGD mc_rx_n48_c3000_full $GEN -m $MODEL -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 48 --moe-cache-mib 3000 -r 3 --bench pp512,tg128,pg512:128
$RUN $LOGD mc_rx_n48_c3300_tg $GEN -m $MODEL -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 48 --moe-cache-mib 3300 -r 2 --bench tg128
# --- natural text, 128 greedy tokens after each prompt
for pr in nat_en nat_ja nat_code nat_chat; do
  n=1000
  $RUN $LOGD txt_mc_rx_c3000_$pr $GEN -m $MODEL -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 48 --moe-cache-mib 3000 --text $P/$pr.txt --n-prompt $n --n-gen 128
  $RUN $LOGD txt_rx_n39_$pr $GEN -m $MODEL -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 39 --text $P/$pr.txt --n-prompt $n --n-gen 128
  $RUN $LOGD txt_rx_n41_$pr $GEN -m $MODEL -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 41 --text $P/$pr.txt --n-prompt $n --n-gen 128
  $RUN $LOGD txt_ot_wx5_n34_$pr $GEN -m $MODEL -ub 512 -dev Vulkan0/Vulkan1 -ngl 99 -ncmoe 34 -ts 48/0 -ot 'blk\.(34|35|36|37|38)\.ffn_.*_exps\.weight=Vulkan1' --text $P/$pr.txt --n-prompt $n --n-gen 128
  GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0 \
    $RUN $LOGD txt_mix4_n29_$pr $GEN -m $MODEL -ub 512 -dev Vulkan0/Vulkan1/CL11_0/CL11_1 -ngl 99 -ncmoe 29 -ts 38/5/2/3 --text $P/$pr.txt --n-prompt $n --n-gen 128
  $RUN $LOGD txt_cpu_$pr $GEN -m $MODEL -ub 512 -dev none -ngl 0 --text $P/$pr.txt --n-prompt $n --n-gen 64
done
