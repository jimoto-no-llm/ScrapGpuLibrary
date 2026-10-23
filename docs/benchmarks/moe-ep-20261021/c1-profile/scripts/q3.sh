#!/bin/bash
cd $HOME/moe-ep/c1p; . scripts/env.sh
SH=$HOME/moe-ep/c1p/vkshim.so
run() { # tag dev ts
  CHILD_LD_PRELOAD=$SH VKSHIM_OUT=$HOME/moe-ep/c1p/logs/tl_$1.txt VKSHIM_EVENTS=1 scripts/go.sh tl_$1 5 -- $LB/llama-bench -m $MODEL -dev $2 -ngl 99 -ts $3 -sm layer -fa 0 -o jsonl -p 0 -n 96 -r 1 -t 2 -b 512 -ub 512
}

run g_fw_2_1 Vulkan0/Vulkan1 2/1
run g_fw_3_1 Vulkan0/Vulkan1 3/1
run g_fw_4_1 Vulkan0/Vulkan1 4/1
run g_fw_5_1 Vulkan0/Vulkan1 5/1
run g_fw_6_1 Vulkan0/Vulkan1 6/1
run g_rv_1_5 Vulkan1/Vulkan0 1/5
run g_rv_1_3 Vulkan1/Vulkan0 1/3
run g_rv_1_2 Vulkan1/Vulkan0 1/2
echo Q3DONE
