. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
P=$H/prompts
c() { # tag, args...   (moe cache sweep, tg only)
  tag=$1; shift
  $RUN $LOGD $tag $GEN -m $MODEL -ub 512 "$@"
}
# --- llama.cpp's own moe_cache, RX 6400 alone, tg128 sweep (random tokens)
c mc_rx_n48_c2800_tg  -dev Vulkan0 -ngl 99 -ncmoe 48 --moe-cache-mib 2800 -r 2 --bench tg128
c mc_rx_n48_c3000_tg  -dev Vulkan0 -ngl 99 -ncmoe 48 --moe-cache-mib 3000 -r 2 --bench tg128
c mc_rx_n45_c1900_tg  -dev Vulkan0 -ngl 99 -ncmoe 45 --moe-cache-mib 1900 -r 2 --bench tg128
c mc_rx_n42_c1000_tg  -dev Vulkan0 -ngl 99 -ncmoe 42 --moe-cache-mib 1000 -r 2 --bench tg128
c mc_amd_n48_c3200_tg -dev Vulkan0/Vulkan1 -ts 42/6 -ngl 99 -ncmoe 48 --moe-cache-mib 3200 -r 2 --bench tg128
