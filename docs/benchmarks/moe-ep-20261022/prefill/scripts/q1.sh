# pp-only sweeps of the ubatch size, C1 vs SGLX C alternating
export REPS=5
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
for ub in 512 256 128 512 256 128; do
  UB=$ub ./bench.sh q1_c1_ub$ub "" -p 512 -n 0
  UB=$ub ./bench.sh q1_C_ub$ub "$C" -p 512 -n 0
done
