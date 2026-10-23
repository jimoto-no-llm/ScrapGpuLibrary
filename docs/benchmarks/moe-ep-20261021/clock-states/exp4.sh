#!/bin/bash
# exp4: CPU-contention sensitivity of C1 tg under perf high (REPS=5): bare, 1 busy-loop process, 2 busy-loop processes, bare. Restores perf auto.
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
hog() { python3 -c 'while True: pass' & HP=$!; "$@"; kill $HP; wait $HP 2>/dev/null; }
hog2() { python3 -c 'while True: pass' & H1=$!; python3 -c 'while True: pass' & H2=$!; "$@"; kill $H1 $H2; wait $H1 $H2 2>/dev/null; }
bare e4_bare1; hog bare e4_hog1; hog2 bare e4_hog2; bare e4_bare2
