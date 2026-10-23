#!/bin/bash
# exp20: one llama-bench process, tg128 x REPS (default 120, about 6 min) under perf MODE (default high): the tg of every repetition in order = the state as a time series. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf ${MODE:-high} >> logs/cs.perf.log 2>&1
T=${1:-e20}
REPS=${REPS:-120} PROFILE=0 $SG/bench.sh cs_$T "" -p 0 -n 128 >/dev/null 2>&1
python3 - $SG/logs/cs_$T.out <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128: s=d["samples_ts"]; print(" ".join("%.0f"%x for x in s))
P
echo FIN
