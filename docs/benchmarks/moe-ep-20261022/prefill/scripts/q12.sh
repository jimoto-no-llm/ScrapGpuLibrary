for fa in 0 1; do for ub in 512 256; do
  FA=$fa TS=6/1 UB=$ub REPS=1 ./bench.sh q12_ts61_fa${fa}_ub$ub "" -p 512 -n 0 -v
done; done
