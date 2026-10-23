. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
export HEC_GEN_VERBOSE=1
P=$HOME/moe-ep/hec/prompts
PLUG0=$HOME/moe-ep/src/build-cl11/libggml-cl11.so
# natural text again (txt2_* of q_base3 were disturbed for nat_en / nat_ja: tg 1 tok/s while the page cache was evicted by something else on the machine),
# the model is read into the page cache before every run, the configurations alternate per prompt
warm() { cat $MODEL > /dev/null; }
for pr in nat_en nat_ja nat_code nat_chat; do
  g="$GEN -m $MODEL -ub 512 --text $P/$pr.txt --n-prompt 1000 --n-gen 128"
  warm; run_n $LOGD txt3_N_$pr $g "${N_ARGS[@]}"
  warm; $RUN $LOGD txt3_rx_n39_$pr $g -dev Vulkan0 -ngl 99 -ncmoe 39
  warm; $RUN $LOGD txt3_rx_n41_$pr $g -dev Vulkan0 -ngl 99 -ncmoe 41
  warm; $RUN $LOGD txt3_ot_wx5_n34_$pr $g -dev Vulkan0/Vulkan1 -ngl 99 -ncmoe 34 -ts 48/0 -ot 'blk\.(34|35|36|37|38)\.ffn_.*_exps\.weight=Vulkan1'
  warm; GGML_BACKEND_PATH=$PLUG0 GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0 \
    $RUN $LOGD txt3_mix4_n29_$pr $g -dev Vulkan0/Vulkan1/CL11_0/CL11_1 -ngl 99 -ncmoe 29 -ts 38/5/2/3
  warm; $RUN $LOGD txt3_mc_rx_c3000_$pr $g -dev Vulkan0 -ngl 99 -ncmoe 48 --moe-cache-mib 3000
done
