#!/bin/bash
# usage: hbench.sh TAG MODE NCMOE UB [REPS] [llama-bench args...]    (default args: -p 512 -n 128 -pg 512,128)
#   MODE: hip    = RX 6400 as ROCm0 (ggml-hip)
#         vk     = RX 6400 as Vulkan0 (official Vulkan, same build as hip)
#         vkold  = the same with the earlier Vulkan-only build ($LB, ~/moe-ep/src/llama-build)
#         mix    = ROCm0 (RX 6400) + Vulkan1 (WX 2100), layer split TS (default 5/1)
#         vkmix  = Vulkan0/Vulkan1 (the C1 split of the other measurements) on the same build, TS (default 5/1)
#         vkoldmix = C1 on the earlier build
# One llama-bench under the SGLX run.sh (flock + temperature wait). Logs: ~/moe-ep/sglx/logs/TAG.{log,out}. Extra environment (GGML_CUDA_FORCE_MMQ=1 ...) is inherited.
. "$HOME/moe-ep/sglx/env.sh"
. "$HOME/moe-ep/hip/env.sh"
HB=$HOME/moe-ep/hip/${BDIR:-llama-build}/bin
TAG=$1; MODE=$2; NC=$3; UB=$4; REPS=${5:-5}; shift 5 2>/dev/null
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(-p 512 -n 128 -pg 512,128)
export VK_ICD_FILENAMES=$VKICD
unset GGML_BACKEND_PATH
case "$MODE" in
  hip)      BIN=$HB; DEV=ROCm0; EXTRA=(-ncmoe $NC) ;;
  vk)       BIN=$HB; DEV=Vulkan0; EXTRA=(-ncmoe $NC) ;;
  vkold)    BIN=$LB; DEV=Vulkan0; EXTRA=(-ncmoe $NC) ;;
  mix)      BIN=$HB; DEV=ROCm0/Vulkan1; EXTRA=(-sm layer -ts ${TS:-5/1} -ncmoe $NC) ;;
  vkmix)    BIN=$HB; DEV=Vulkan0/Vulkan1; EXTRA=(-sm layer -ts ${TS:-5/1} -ncmoe $NC) ;;
  vkoldmix) BIN=$LB; DEV=Vulkan0/Vulkan1; EXTRA=(-sm layer -ts ${TS:-5/1} -ncmoe $NC) ;;
  *) echo bad mode; exit 2 ;;
esac
exec "$HOME/moe-ep/sglx/run.sh" "$TAG" "$BIN/llama-bench" -m "$MODEL" -dev "$DEV" -ngl 99 -fa 0 -t 2 -b "$UB" -ub "$UB" -r "$REPS" "${EXTRA[@]}" -o jsonl "${ARGS[@]}"
