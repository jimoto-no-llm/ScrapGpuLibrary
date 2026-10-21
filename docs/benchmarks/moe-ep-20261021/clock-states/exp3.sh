#!/bin/bash
# exp3: is the 50 vs 44/52 a sampler effect? C1 REPS=5 under perf high: bare, bare, sampled, bare. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export REPS=5 PROFILE=0
bare() { $SG/bench.sh cs_$1 "" >/dev/null 2>&1; python3 - $SG/logs/cs_$1.out $1 <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("RESULT",sys.argv[2],"tg128=%.1f"%d["avg_ts"],[round(x,1) for x in d["samples_ts"]],flush=True)
P
}
bare e3_bare1; bare e3_bare2; $SG/probe.sh e3_samp; bare e3_bare3
