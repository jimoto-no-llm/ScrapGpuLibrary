#!/bin/bash
# exp2: replay of queue_ph2 under perf high with the sampler: C1 / SGLX C alternating, REPS=10, twice. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export REPS=10
for i in 1 2; do
  $SG/probe.sh e2_c1_$i ""
  $SG/probe.sh e2_C_$i "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
done
