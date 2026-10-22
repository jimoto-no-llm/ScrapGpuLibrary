export REPS=5
for ub in 304 312 320 328 336 344 352 360 368 376; do
  UB=$ub ./bench.sh q14_c1_ub$ub "" -p 512 -n 0
done
