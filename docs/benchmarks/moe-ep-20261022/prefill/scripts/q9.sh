for ts in 5/1 6/1; do
  GGML_VK_MEMORY_LOGGER=1 TS=$ts REPS=1 ./bench.sh q9_mem_ts${ts/\//_} "" -p 512 -n 0 -v
done
