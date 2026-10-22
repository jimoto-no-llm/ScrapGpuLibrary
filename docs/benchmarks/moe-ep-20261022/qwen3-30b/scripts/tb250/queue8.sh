#!/bin/bash
cd "$(dirname "$0")"
f() { tag=$1; shift; env "$@" REPS=3 TMO=3000 ./bench.sh $tag $CFGX >/dev/null 2>&1; echo "$tag rc=$?"; }
CFGX=mix5b f final_mix5b_ts36-5-4-2-1_n27_ub512 NCMOE=27 UB=512 TS=36/5/4/2/1
CFGX=mix5b f final_mix5b_ts36-5-4-2-1_n28_ub512 NCMOE=28 UB=512 TS=36/5/4/2/1
echo Q8DONE
