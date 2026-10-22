export REPS=10
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
for i in 1 2; do
UB=512 ./bench.sh q28_c1_ub512_$i ""
UB=320 ./bench.sh q28_c1_ub320_$i ""
UB=512 ./bench.sh q28_C_ub512_$i "$C"
UB=320 ./bench.sh q28_C_ub320_$i "$C"
done
