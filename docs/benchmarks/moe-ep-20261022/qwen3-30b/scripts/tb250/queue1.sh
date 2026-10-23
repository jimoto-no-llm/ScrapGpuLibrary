#!/bin/bash
cd "$(dirname "$0")"
REPS=3 TMO=3000 ./bench.sh final_cpu cpu >/dev/null 2>&1; echo "final_cpu rc=$?"
./sweep.sh rx rx "44 42 41 40 39 38 37 36"
./sweep.sh amd amd "40 36 34 32 31 30 29 28 26" 5/1
echo Q1DONE
