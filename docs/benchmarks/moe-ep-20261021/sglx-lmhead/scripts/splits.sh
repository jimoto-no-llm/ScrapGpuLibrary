#!/bin/bash
# scheduler splits of the decode graph, C1 and C1 + SGLX0 (same method as baseline/scripts/profile3.sh: llama-completion -v with GGML_SCHED_DEBUG=1)
cd "$(dirname "$0")"
. ./env.sh
C=(-ngl 99 -fa off -t 2 -b 128 -ub 128 -c 256 -n 3 -p "Hello" --temp 0 -no-cnv -v -sm layer)
sglx_env ""; GGML_SCHED_DEBUG=1 ./run.sh splits_c1 "$LB/llama-completion" -m "$MODEL" -dev Vulkan0,Vulkan1 "${C[@]}" -ts 5,1
sglx_env "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"; GGML_SCHED_DEBUG=1 ./run.sh splits_sglx "$LB/llama-completion" -m "$MODEL" -dev Vulkan0,Vulkan1,SGLX0 "${C[@]}" -ts 5,1,0 -ot 'token_embd\.weight=SGLX0'
echo done > logs/splits.done
