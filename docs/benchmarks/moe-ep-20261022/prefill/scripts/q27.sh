export REPS=5
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
for ub in 192 224 256 288 304 320 336 352 384 416 448; do UB=$ub ./bench.sh q27_C_ub$ub "$C" -p 512 -n 0; done
