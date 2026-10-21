#!/bin/bash
# exp6: bare C1 (no sampler, REPS=3) N times in a row under perf high, with the time of each; to see whether the tg state flips by itself. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export REPS=3 PROFILE=0
for i in $(seq 1 ${1:-8}); do
  T=${2:-e6}_$i; $SG/bench.sh cs_$T "" >/dev/null 2>&1
  python3 - $SG/logs/cs_$T.out $T <<P
import json,sys,time
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("RESULT",time.strftime("%H:%M:%S"),sys.argv[2],"tg128=%.1f"%d["avg_ts"],[round(x,1) for x in d["samples_ts"]],flush=True)
P
done
echo FIN
