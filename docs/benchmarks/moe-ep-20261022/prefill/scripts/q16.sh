export REPS=5
UB=320 ./bench.sh q16_noop_ub320 noop -p 512 -n 0
UB=320 ./bench.sh q16_c1_ub320 "" -p 512 -n 0
