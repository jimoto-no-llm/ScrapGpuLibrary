export REPS=2
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
GGML_SGLX_TRACE=1 PROFILE=1 UB=256 ./bench.sh q21_C_ub256 "$C" -p 512 -n 0
