export REPS=3
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
for ub in 256 512; do
GGML_SGLX_HOSTBUF=1 UB=$ub ./bench.sh q24_Chb_ub$ub "$C" -p 512 -n 0
done
