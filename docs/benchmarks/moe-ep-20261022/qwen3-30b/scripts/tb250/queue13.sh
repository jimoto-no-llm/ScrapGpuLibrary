#!/bin/bash
cd "$(dirname "$0")"
show() { python3 - logs/$1.out $1 <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l); print(sys.argv[2], [round(x,2) for x in d["samples_ts"]])
P
}
REPS=3 TMO=900 ./bench.sh clean_cpu_tg32 cpu -p 0 -n 32 >/dev/null 2>&1; show clean_cpu_tg32
NCMOE=39 UB=512 REPS=3 TMO=900 ./bench.sh clean_rx39_tg32 rx -p 0 -n 32 >/dev/null 2>&1; show clean_rx39_tg32
echo CLEANCHK
