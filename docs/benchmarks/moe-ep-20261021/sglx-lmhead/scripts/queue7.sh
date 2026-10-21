#!/bin/bash
# test-backend-ops on the TB250 for several splits, and the scheduler splits of the graph
cd "$(dirname "$0")"
. ./env.sh
rm -f logs/queue7.done
i=0
for s in "Vulkan0" "Vulkan1" "CL11_0" "Vulkan0:0.5,Vulkan1:0.5" "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"; do
  i=$((i+1)); sglx_env "$s"; PROFILE=0
  ./run.sh "tbo${i}_mulmat" "$LB/test-backend-ops" -b SGLX0 -o MUL_MAT
  ./run.sh "tbo${i}_getrows" "$LB/test-backend-ops" -b SGLX0 -o GET_ROWS
done
# plain Vulkan0 on the cases that fail for every split (ggml-vulkan itself)
sglx_env ""
./run.sh tbo_plain_vk0_q40 "$LB/test-backend-ops" -b Vulkan0 -o MUL_MAT -p "type_a=q4_0,type_b=f32,m=(40|100),n=9"
# scheduler splits
GGML_SCHED_DEBUG=1 ./bench.sh sched_sglx "Vulkan0:0.5,Vulkan1:0.5" -p 0 -n 2
GGML_SCHED_DEBUG=1 ./bench.sh sched_c1 "" -p 0 -n 2
echo done > logs/queue7.done
