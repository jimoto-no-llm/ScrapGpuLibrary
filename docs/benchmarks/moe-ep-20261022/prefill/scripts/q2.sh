export REPS=5
for ub in 512 256 128; do
  UB=$ub ./bench.sh q2_noop_ub$ub noop -p 512 -n 0
  UB=$ub ./bench.sh q2_c1_ub$ub "" -p 512 -n 0
done
for ts in 4/1 6/1 7/1 3/1; do
  TS=$ts ./bench.sh q2_c1_ts${ts/\//_} "" -p 512 -n 0
done
