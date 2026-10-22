#!/bin/bash
cd "$(dirname "$0")"
f() { tag=$1; shift; env "$@" REPS=3 TMO=3000 ./bench.sh $tag $CFGX >/dev/null 2>&1; echo "$tag rc=$?"; }
CFGX=rx    f final2_rx_n39_ub512 NCMOE=39 UB=512
CFGX=amd   f final2_amd_ts42-6_n33_ub512 NCMOE=33 UB=512 TS=42/6
CFGX=mix5b f final2_mix5b_ts36-5-4-2-1_n29_ub512 NCMOE=29 UB=512 TS=36/5/4/2/1
CFGX=mix4  f final2_mix4_ts38-5-2-3_n29_ub512 NCMOE=29 UB=512 TS=38/5/2/3
CFGX=cl5b  f final2_cl5b_ts35-5-4-2-2_n27_ub128 NCMOE=27 UB=128 TS=35/5/4/2/2
# VRAM of the NVIDIA cards (nvidia-smi sampling steals a CPU core, so only in these short runs, never in timing runs)
for c in "mix5b 29 36/5/4/2/1" "mix4 29 38/5/2/3" "cl5b 27 35/5/4/2/2"; do
  set -- $c; UB=128 NCMOE=$2 TS=$3 NVSMI=1 REPS=1 TMO=900 ./bench.sh memnv_$1_n$2 $1 -p 64 -n 16 >/dev/null 2>&1; echo "memnv_$1 rc=$?"
done
echo Q15DONE
