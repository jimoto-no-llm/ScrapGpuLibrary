export REPS=5
for ub in 288 320 384 448; do
  UB=$ub ./bench.sh q3_c1_ub$ub "" -p 512 -n 0
done
for ub in 256 192 128; do
  LLAMA_GRAPH_REUSE_DISABLE=1 UB=$ub ./bench.sh q3_c1_norense_ub$ub "" -p 512 -n 0
done
