#!/bin/bash
# usage: hkld.sh TAG MODE NCMOE    KLD against the CPU logits ~/moe-ep/kl/base.kld, same command as sglx/kld.sh (-c 512 --chunks 4 -b 128 -ub 128 -t 2 -fa off)
. "$HOME/moe-ep/sglx/env.sh"
. "$HOME/moe-ep/hip/env.sh"
HB=$HOME/moe-ep/hip/${BDIR:-llama-build}/bin
TAG=$1; MODE=$2; NC=$3
export VK_ICD_FILENAMES=$VKICD
case "$MODE" in hip) DEV=ROCm0 ;; vk) DEV=Vulkan0 ;; vkmix) DEV=Vulkan0,Vulkan1; X=(-sm layer -ts ${TS:-5/1}) ;; mix) DEV=ROCm0,Vulkan1; X=(-sm layer -ts ${TS:-5/1}) ;; esac
exec "$HOME/moe-ep/sglx/run.sh" "$TAG" "$HB/llama-perplexity" -m "$MODEL" -f "$TEXT" -c 512 --chunks 4 -b 128 -ub 128 -t 2 -fa off -dev "$DEV" -ngl 99 -ncmoe $NC "${X[@]}" --kl-divergence-base "$CPU_BASE" --kl-divergence
