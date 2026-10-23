#!/bin/bash
# pp512 A/B: C1 vs SGLX C at -ub 320, 8 alternating pairs, REPS=10, perf pin, fixed plugin; restores auto.
cd ~/moe-ep/sglx
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/ppab.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf pin >> logs/ppab.perf.log 2>&1
export PROFILE=0 REPS=10 UB=320 PLUG_OVERRIDE=$HOME/moe-ep/fix/build/libggml-cl11.so
for i in 1 2 3 4 5 6 7 8; do
  ./bench.sh ppab_c1_$i "" -p 512 -n 0
  ./bench.sh ppab_C_$i "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1" -p 512 -n 0
done
echo DONE >> logs/ppab.perf.log
