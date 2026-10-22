#!/bin/bash
cd "$(dirname "$0")"
./sweep.sh mix5b mix5b_ts36-5-4-2-1 "26 25 24" 36/5/4/2/1
f() { tag=$1; shift; env "$@" REPS=3 TMO=3000 ./bench.sh $tag $CFGX >/dev/null 2>&1; echo "$tag rc=$?"; }
CFGX=rx   f final_rx_n39_ub512 NCMOE=39 UB=512
CFGX=amd  f final_amd_ts42-6_n33_ub512 NCMOE=33 UB=512 TS=42/6
CFGX=mix4 f final_mix4_ts38-5-2-3_n28_ub512 NCMOE=28 UB=512 TS=38/5/2/3
CFGX=cl5b f final_cl5b_ts35-5-4-2-2_n27_ub512 NCMOE=27 UB=512 TS=35/5/4/2/2
echo Q7DONE
