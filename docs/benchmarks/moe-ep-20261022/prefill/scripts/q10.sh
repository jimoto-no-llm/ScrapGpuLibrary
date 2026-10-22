export REPS=3
for ub in 256 192 128 96; do
  TS=6/1 UB=$ub ./bench.sh q10_c1_ts61_ub$ub "" -p 512 -n 64
done
TS=5/1 UB=256 ./bench.sh q10_c1_ts51_ub256 "" -p 512 -n 64
