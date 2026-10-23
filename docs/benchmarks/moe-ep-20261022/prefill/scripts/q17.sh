export REPS=3
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
PROFILE=1 UB=320 ./bench.sh q17_C_ub320 "$C" -p 512 -n 0
PROFILE=1 UB=512 ./bench.sh q17_C_ub512 "$C" -p 512 -n 0
