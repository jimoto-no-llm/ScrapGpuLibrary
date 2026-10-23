#!/bin/bash
# graph splits: GGML_SCHED_DEBUG=1 needs the debug log level (-v); the first decode graph after the prompt is counted
. "$(dirname "$0")/env.sh"; S="$(dirname "$0")"
C=(-ngl 99 -fa off -t 2 -b 128 -ub 128 -c 256 -n 3 -p "Hello" --temp 0 -no-cnv -v)
use_env rx-vk;  GGML_SCHED_DEBUG=1 "$S/run.sh" prof2_c0_vk_splits "$LB/llama-completion" -m "$MODEL" -dev Vulkan0 "${C[@]}" -ncmoe 5
use_mg amdvk;   GGML_SCHED_DEBUG=1 "$S/run.sh" prof2_c1_vk_splits "$LB/llama-completion" -m "$MODEL" -dev "${DEV//\//,}" "${C[@]}" -ts 5,1 -sm layer
use_mg cl5;     GGML_SCHED_DEBUG=1 "$S/run.sh" prof2_cl5_splits "$LB/llama-completion" -m "$MODEL" -dev "${DEV//\//,}" "${C[@]}" -ts 8,8,18,40,18 -sm layer
use_env gt730;  GGML_SCHED_DEBUG=1 "$S/run.sh" prof2_gt730_splits "$LB/llama-completion" -m "$MODEL" -dev CL11_0 "${C[@]}" -ncmoe 22
echo PROFILE3_DONE
