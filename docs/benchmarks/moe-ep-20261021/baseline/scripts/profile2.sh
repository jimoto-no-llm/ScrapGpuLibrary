#!/bin/bash
. "$(dirname "$0")/env.sh"; S="$(dirname "$0")"
use_mg amdvk
GGML_SCHED_DEBUG=1 "$S/run.sh" prof_c1_vk_splits "$LB/llama-completion" -m "$MODEL" -dev "${DEV//\//,}" -ngl 99 -fa off -t 2 -b 128 -ub 128 -ts 5,1 -sm layer -c 256 -n 3 -p "Hello" --temp 0 -no-cnv
use_mg cl5
GGML_SCHED_DEBUG=1 "$S/run.sh" prof_cl5_splits "$LB/llama-completion" -m "$MODEL" -dev "${DEV//\//,}" -ngl 99 -fa off -t 2 -b 128 -ub 128 -ts 8,8,18,40,18 -sm layer -c 256 -n 3 -p "Hello" --temp 0 -no-cnv
echo PROFILE2_DONE
