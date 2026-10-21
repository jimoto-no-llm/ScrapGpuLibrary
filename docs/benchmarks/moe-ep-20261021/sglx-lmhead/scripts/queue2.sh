#!/bin/bash
cd "$(dirname "$0")"
: > logs/queue2.out
for ts in 5/1 4/1; do
 for s in "Vulkan0:0.7,Vulkan1:0.3" "Vulkan0:0.4,Vulkan1:0.6" "Vulkan0:0.2,Vulkan1:0.8"; do
  t=$(echo "$ts $s" | tr ':, /' '____')
  TS=$ts REPS=2 ./bench.sh "q2_$t" "$s" -p 0 -n 128
 done
done
echo done > logs/queue2.done
