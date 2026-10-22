#!/bin/bash
cd "$(dirname "$0")"
# wait for queue17 (Q3 sweeps use MODEL env of their own; finals here are Q4_0)
until grep -q Q17DONE logs/queue17.out; do sleep 10; done
f() { tag=$1; shift; env "$@" REPS=3 TMO=3000 ./bench.sh $tag $CFGX >/dev/null 2>&1; echo "$tag rc=$?"; }
CFGX=rx  f final2_rx_n44_ub512 NCMOE=44 UB=512
CFGX=rx  f final2_rx_n41_ub512 NCMOE=41 UB=512
CFGX=rx  f final2_rx_n44_ub1024 NCMOE=44 UB=1024
CFGX=amd f final2_amd_ts42-6_n38_ub512 NCMOE=38 UB=512 TS=42/6
CFGX=amd f final2_amd_ts42-6_n36_ub512 NCMOE=36 UB=512 TS=42/6
echo Q18DONE
