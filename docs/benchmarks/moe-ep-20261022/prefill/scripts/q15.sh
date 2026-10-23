export REPS=5
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
GGML_SCHED_DEBUG=0 NOOT=1 UB=320 PROFILE=0 ./bench.sh q15_C_noot_ub320 "$C" -p 512 -n 128 -v
UB=320 ./bench.sh q15_C_ot_ub320 "$C" -p 512 -n 128
UB=512 NOOT=1 ./bench.sh q15_C_noot_ub512 "$C" -p 512 -n 128
UB=320 ./bench.sh q15_c1_ub320 "" -p 512 -n 128
