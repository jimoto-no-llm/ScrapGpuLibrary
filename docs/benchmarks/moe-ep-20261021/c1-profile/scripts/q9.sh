#!/bin/bash
cd $HOME/moe-ep/c1p; . scripts/env.sh
F() { tag=$1; envs=$2; shift 2; env $envs scripts/go.sh $tag 1 -- $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -sm layer -t 2 -o jsonl -p 512 -n 128 -r 5 "$@"; echo "$tag temps: $(for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do echo -n "$(( $(cat $f)/1000 )) "; done) vram_used: $(cat /sys/class/drm/card1/device/mem_info_vram_used) $(cat /sys/class/drm/card5/device/mem_info_vram_used) gtt: $(cat /sys/class/drm/card1/device/mem_info_gtt_used)" >> logs/q9.log; }
F fin_ot_ub128 "" -ts 5/1 -b 128 -ub 128 -ot "token_embd\.weight=Vulkan0"
F fin_base_ub128 "" -ts 5/1 -b 128 -ub 128
F fin_ot_ub64 "" -ts 5/1 -b 64 -ub 64 -ot "token_embd\.weight=Vulkan0"
F fin_ot_ub256 "" -ts 5/1 -b 256 -ub 256 -ot "token_embd\.weight=Vulkan0"
echo Q9DONE >> logs/q9.log
