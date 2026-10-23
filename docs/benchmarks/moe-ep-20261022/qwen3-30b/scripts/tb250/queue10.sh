#!/bin/bash
cd "$(dirname "$0")"
NCMOE=27 UB=128 TS=35/5/4/2/2 REPS=3 TMO=3000 ./bench.sh final_cl5b_ts35-5-4-2-2_n27_ub128 cl5b >/dev/null 2>&1; echo "cl5b_ub128 rc=$?"
echo Q10DONE
