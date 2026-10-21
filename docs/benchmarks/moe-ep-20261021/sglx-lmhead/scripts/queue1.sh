#!/bin/bash
# auto power level; each bench.sh run takes the lock itself
cd "$(dirname "$0")"
./bench.sh c1_plain "" ; ./bench.sh c1_noop noop
for s in "Vulkan0:0.7,Vulkan1:0.3" "Vulkan0:0.5,Vulkan1:0.5" "Vulkan0:0.8,Vulkan1:0.2" "Vulkan0:0.55,Vulkan1:0.2,CL11_0:0.15,CL11_1:0.1" "Vulkan0:0.4,Vulkan1:0.3,CL11_0:0.2,CL11_1:0.1"; do
  t=$(echo "$s" | tr ':,' '__')
  ./bench.sh "s_ts51_$t" "$s"
done
echo done > logs/queue1.done
