. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
$RUN $LOGD mc0_rx2800_smoke $GEN -m $MODEL -dev Vulkan0 -ngl 99 -ncmoe 48 --moe-cache-mib 2800 -ub 512 -r 1 --bench tg32
