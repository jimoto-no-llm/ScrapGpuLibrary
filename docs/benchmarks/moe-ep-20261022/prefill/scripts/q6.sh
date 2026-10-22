export REPS=5
for ub in 512 256 128; do
  GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 UB=$ub ./bench.sh q6_c1_nohv_ub$ub "" -p 512 -n 128
  UB=$ub ./bench.sh q6_c1_base_ub$ub "" -p 512 -n 128
done
