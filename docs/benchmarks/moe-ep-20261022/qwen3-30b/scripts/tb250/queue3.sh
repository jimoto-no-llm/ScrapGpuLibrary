#!/bin/bash
cd "$(dirname "$0")"
./sweep.sh amd amd_ts42-6 "33 32 31 30 29" 42/6
./sweep.sh amd amd_ts43-5 "33 32 31" 43/5
./sweep.sh amd amd_ts41-7 "33 32 31 30 29" 41/7
for ub in 1024 2048; do UB=$ub ./sweep.sh rx rx_ub$ub "40 39"; done
echo Q3DONE
