#!/bin/bash
# q7: correctness of C+430 (no perf pin): test-backend-ops, KLD against the CPU logits and against the logits of one SGLX device, at -b 512 -ub 320
cd ~/moe-ep/g430/scripts
. ./env.sh
SPL="Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07"
REMOTE_BE=opencl11; export REMOTE_BE
sglx_env "$SPL"
TMO=1500 ./run.sh q7_tbo_C430 $LB/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS
export KB=512 KUB=320
./kld.sh q7_C430_cpu "$SPL" cpu
./kld.sh q7_one320 "Vulkan0:1" - save
./kld.sh q7_C430_self "$SPL" $SG/kl/q7_one320.kld
./kld.sh q7_C_cpu "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1" cpu
REMOTE_BE=cuda ./kld.sh q7_C430cuda_cpu "$SPL" cpu
echo Q7_DONE >> ../logs/queue.log
