#!/bin/bash
cd $HOME/moe-ep/c1p; . scripts/env.sh
F() { tag=$1; envs=$2; shift 2; env $envs scripts/go.sh $tag 1 -- $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -sm layer -b 512 -ub 512 -t 2 -o jsonl -p 512 -n 128 -r 5 "$@"; }
F fin_base "" -ts 5/1
F fin_ot "" -ts 5/1 -ot "token_embd\.weight=Vulkan0"
F fin_mmvq "GGML_VK_FORCE_MMVQ=1" -ts 5/1
F fin_ot_mmvq "GGML_VK_FORCE_MMVQ=1" -ts 5/1 -ot "token_embd\.weight=Vulkan0"
F fin_ot_4_1_rep "" -ts 4/1 -ot "token_embd\.weight=Vulkan0"
F fin_poll0 "" -ts 5/1 --poll 0
F fin_base2 "" -ts 5/1
echo Q8DONE
