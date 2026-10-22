#!/bin/bash
cd "$(dirname "$0")"
show() { python3 - logs/$1.out $1 <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l); print(sys.argv[2], [round(x,2) for x in d["samples_ts"]])
P
}
for i in 1 2 3; do
  REPS=3 TMO=900 ./bench.sh drift_cpu_$i cpu -p 0 -n 32 >/dev/null 2>&1; show drift_cpu_$i
  NCMOE=39 UB=512 REPS=3 TMO=900 ./bench.sh drift_rx39_$i rx -p 0 -n 32 >/dev/null 2>&1; show drift_rx39_$i
  NCMOE=33 UB=512 TS=42/6 REPS=3 TMO=900 ./bench.sh drift_amd33_$i amd -p 0 -n 32 >/dev/null 2>&1; show drift_amd33_$i
done
echo Q12DONE
