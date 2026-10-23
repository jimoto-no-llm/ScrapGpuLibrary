#!/bin/bash
cd $HOME/moe-ep/c1p
until grep -q Q5DONE logs/q5.log; do sleep 5; done
# dense small models: single GPU, full resident -> fixed overhead regression
for m in LFM2.5-230M-Q4_0 LFM2.5-350M-Q4_0 LFM2.5-1.2B-Instruct-Q4_0 LFM2.5-230M-Q8_0 LFM2.5-350M-Q8_0 LFM2.5-1.2B-Instruct-Q8_0; do
  for d in 0 1; do
    scripts/b2.sh dense_${m}_v$d $m.gguf "" -- -dev Vulkan$d -b 512 -ub 512 -t 2
  done
done
# timeline for ot configs and the best candidate
SH=$HOME/moe-ep/c1p/vkshim.so
run() { CHILD_LD_PRELOAD=$SH VKSHIM_OUT=$HOME/moe-ep/c1p/logs/tl_$1.txt VKSHIM_EVENTS=1 scripts/go.sh tl_$1 5 -- $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -ts $2 -sm layer -o jsonl -p 0 -n 96 -r 1 -t 2 -b 512 -ub 512 "${@:3}"; }
. scripts/env.sh
run g_ot_5_1 5/1 -ot "token_embd\.weight=Vulkan0"
run g_ot_4_1 4/1 -ot "token_embd\.weight=Vulkan0"
run g_ot_3_1 3/1 -ot "token_embd\.weight=Vulkan0"
echo Q67DONE
