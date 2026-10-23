#!/bin/bash
# exp21: C1 and SGLX C alternating, REPS=10, PROFILE=0, perf high, TAG prefix $1; ROUNDS (default 2) rounds of (C1, C). Restores perf auto. Same command lines as queue_ph2.sh.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export PROFILE=0 REPS=10
for i in $(seq 1 ${2:-2}); do
  ./bench.sh ${1:-e21}_c1_$i ""
  ./bench.sh ${1:-e21}_C_$i "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
done
echo FIN
