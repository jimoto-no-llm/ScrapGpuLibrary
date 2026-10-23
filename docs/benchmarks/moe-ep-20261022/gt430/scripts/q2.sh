#!/bin/bash
# q2: first look at C+430 with the profile (3 reps, no perf pin): the CL11 server and the CUDA server for the GT 430
cd ~/moe-ep/g430/scripts
export UB=320 REPS=3 PROFILE=1 PROFILE_EVERY=0
SPL="Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07"
for be in opencl11 cuda; do
  REMOTE_BE=$be ./bench.sh q2_C430_$be "$SPL" -p 512 -n 128 -pg 512,128
done
echo Q2_DONE >> ../logs/queue.log
