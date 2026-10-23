#!/bin/bash
cd "$(dirname "$0")"
export REPS=2
./sweep.sh rx rx2 "39 38"
./sweep.sh amd amd2_ts42-6 "33 32" 42/6
./sweep.sh mix5b mix5b2_ts36-5-4-2-1 "29 28 27 26" 36/5/4/2/1
./sweep.sh mix4 mix4b2_ts38-5-2-3 "29 28" 38/5/2/3
UB=128 ./sweep.sh cl5b cl5b2_ts35-5-4-2-2 "27" 35/5/4/2/2
OT="blk\\.(34|35|36|37|38)\\.ffn_.*_exps\\.weight=Vulkan1" ./sweep.sh amd ot2_wx5 "34" 48/0
echo Q14DONE
