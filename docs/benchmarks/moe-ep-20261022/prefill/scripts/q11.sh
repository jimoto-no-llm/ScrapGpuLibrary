export REPS=3
FA=1 UB=512 ./bench.sh q11_c1_fa1_ub512 "" -p 512 -n 128
FA=0 UB=512 ./bench.sh q11_c1_fa0_ub512 "" -p 512 -n 128
for ub in 512 384 256; do
  FA=1 GGML_VK_MEMORY_LOGGER=1 TS=6/1 UB=$ub ./bench.sh q11_c1_fa1_ts61_ub$ub "" -p 512 -n 128
done
