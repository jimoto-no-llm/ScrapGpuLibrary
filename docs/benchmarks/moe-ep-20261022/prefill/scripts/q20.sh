export REPS=5
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
for ub in 256 512; do
UB=$ub ./bench.sh q20_sglxnoop_ub$ub sglxnoop -p 512 -n 0
UB=$ub ./bench.sh q20_noop_ub$ub noop -p 512 -n 0
UB=$ub ./bench.sh q20_C_ub$ub "$C" -p 512 -n 0
done
