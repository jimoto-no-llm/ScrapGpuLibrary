#!/bin/bash
cd "$(dirname "$0")"
rm -f logs/queue3.done
for s in "Vulkan0:0.85,Vulkan1:0.15" "Vulkan0:0.92,Vulkan1:0.08" "Vulkan0:1" "Vulkan1:1" "Vulkan0:0.75,Vulkan1:0.12,CL11_0:0.08,CL11_1:0.05" "Vulkan0:0.8,Vulkan1:0.1,CL11_0:0.1"; do
  t=$(echo "$s" | tr ':, /' '____')
  TS=5/1 REPS=3 ./bench.sh "q3_$t" "$s" -p 0 -n 128
done
echo done > logs/queue3.done
