export REPS=3
for ub in 512 256 320; do
  SAMPLE_MS=10 SAMPLE_COLS="^(t|rx_busy|wx_busy|rx_membusy|wx_membusy)$" python3 sampler.py logs/q4_busy_ub$ub.csv & SP=$!
  UB=$ub ./bench.sh q4_c1_ub$ub "" -p 512 -n 0
  kill $SP; wait $SP 2>/dev/null
done
