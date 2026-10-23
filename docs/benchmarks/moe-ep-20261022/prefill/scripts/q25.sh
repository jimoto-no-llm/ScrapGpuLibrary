export REPS=3
UB=256 ./bench.sh q25_V1only_ub256 "Vulkan1:1" -p 512 -n 0
UB=256 ./bench.sh q25_V0only_ub256 "Vulkan0:1" -p 512 -n 0
UB=256 ./bench.sh q25_V0V1_ub256 "Vulkan0:0.5,Vulkan1:0.5" -p 512 -n 0
