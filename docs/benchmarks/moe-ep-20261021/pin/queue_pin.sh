#!/bin/bash
# C1 vs SGLX C under perf pin and perf compute, alternating, REPS=10, with a 1 s clock sampler; restores auto.
cd ~/moe-ep/sglx
SP=
cleanup() { [ -n "$SP" ] && kill $SP 2>/dev/null; wait 2>/dev/null; sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/pin.perf.log 2>&1; }
trap cleanup EXIT
export PROFILE=0 REPS=10
for mode in pin compute; do
  sudo -n /usr/local/sbin/sgl-driver-switch perf $mode >> logs/pin.perf.log 2>&1
  SAMPLE_MS=1000 python3 sampler.py logs/pin_${mode}_clk.csv & SP=$!
  for i in 1 2; do
    ./bench.sh pin_${mode}_c1_$i ""
    ./bench.sh pin_${mode}_C_$i "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
  done
  kill $SP; wait $SP 2>/dev/null; SP=
done
echo DONE >> logs/pin.perf.log
