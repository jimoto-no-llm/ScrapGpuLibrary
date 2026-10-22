export REPS=3
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
for ub in 256 320 512; do UB=$ub ./bench.sh q26_C_ub$ub "$C" -p 512 -n 0; done
