#!/bin/bash
cd "$(dirname "$0")"
# pp offload hypothesis (Q4_0): CPU-resident experts are streamed to the GPU per layer for batches (op offload). pp512, r=2.
NCMOE=39 UB=512 REPS=2 TMO=900 ./bench.sh ppchk_rx_n39_ub512 rx -p 512 -n 0 >/dev/null 2>&1; echo ppchk1
NCMOE=39 UB=512 REPS=2 TMO=900 ./bench.sh ppchk_rx_n39_ub512_nopo rx -p 512 -n 0 -nopo 1 >/dev/null 2>&1; echo ppchk2
NCMOE=44 UB=512 REPS=2 TMO=900 ./bench.sh ppchk_rx_n44_ub512 rx -p 512 -n 0 >/dev/null 2>&1; echo ppchk3
NCMOE=44 UB=2048 REPS=2 TMO=900 ./bench.sh ppchk_rx_n44_ub2048 rx -p 2048 -n 0 >/dev/null 2>&1; echo ppchk4
NCMOE=48 UB=512 REPS=2 TMO=900 ./bench.sh ppchk_rx_n48_ub512 rx -p 512 -n 0 >/dev/null 2>&1; echo ppchk5
# Q3_K_M (Vulkan only: the CL11 plugin handles Q4_0/Q8_0/Q4_K/Q6_K)
export MODEL=$HOME/moe-ep/models/Qwen3-30B-A3B-Q3_K_M.gguf REPS=2
./sweep.sh rx q3rx "40 39 38 37 36"
./sweep.sh amd q3amd_ts42-6 "35 34 33 32 31 30" 42/6
echo Q17DONE
