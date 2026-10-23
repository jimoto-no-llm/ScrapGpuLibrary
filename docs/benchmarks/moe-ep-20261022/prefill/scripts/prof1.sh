#!/bin/bash
# per-op profile of one pp512 on C1 with the synchronous perf-logger build, perf pinned
cd ~/moe-ep/pp; . ./env.sh
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf pin >> logs/perf.log 2>&1
sglx_env ""
P=$ROOT/c1p/bin-patched
GGML_VK_PERF_LOGGER=1 ./run.sh prof_c1_pp512 $P/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b 512 -ub 512 -sm layer -ncmoe 0 -ts 5/1 -o jsonl -p 512 -n 0 -r 3
