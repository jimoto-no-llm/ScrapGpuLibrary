#!/bin/bash
# RX share vs the time of the layers phase (is the RX slowed by the extra VRAM it holds?)
cd "$(dirname "$0")"
rm -f ../logs/queue4.done
for ts in 5/1 4/1 3/1; do
 for s in "Vulkan1:1" "Vulkan0:0.2,Vulkan1:0.8" "Vulkan0:0.5,Vulkan1:0.5" "Vulkan0:0.85,Vulkan1:0.15"; do
  t=$(echo "$ts $s" | tr ':, /' '____')
  TS=$ts REPS=2 ./bench.sh "q4_$t" "$s" -p 0 -n 128
 done
done
echo done > ../logs/queue4.done
