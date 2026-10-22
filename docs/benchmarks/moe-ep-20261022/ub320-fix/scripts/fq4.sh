# session 2: (A) T(n) curve of C1, (B) per-op profile vs n, (C) RX VRAM
. ./env.sh
Cs="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
# A
for n in 64 96 128 160 192 224 256 288 320 352 384 448 512; do
  UB=$n B=$n REPS=3 ./bench.sh fA_c1_n$n "" -p $n -n 0
done
# B
P=$ROOT/c1p/bin-patched
for n in 192 256 320 384 512; do
  sglx_env ""
  GGML_VK_PERF_LOGGER=1 ./run.sh fB_prof_n$n $P/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b $n -ub $n -sm layer -ncmoe 0 -ts 5/1 -o jsonl -p $n -n 0 -r 2
done
# C
memrun() { # TAG SPLIT TS
  python3 peak.py ../logs/$1.peak & local PK=$!
  GGML_VK_MEMORY_LOGGER=1 TS=$3 UB=320 REPS=3 ./bench.sh $1 "$2"
  kill $PK; wait $PK 2>/dev/null
}
memrun fC_c1_ts51 "" 5/1
memrun fC_c1_ts61 "" 6/1
memrun fC_C_ts51 "$Cs" 5/1
memrun fC_C_ts61 "$Cs" 6/1
memrun fC_C25_ts61 "Vulkan0:0.25,Vulkan1:0.35,CL11_0:0.25,CL11_1:0.15" 6/1
memrun fC_C0_ts61 "Vulkan1:0.5,CL11_0:0.3,CL11_1:0.2" 6/1
