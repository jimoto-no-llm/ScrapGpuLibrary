#!/bin/bash
# exp11: does the tg state depend on the mmap'ed model (page cache)? Alternate default (mmap) and --mmap 0 C1 runs (REPS=2), N pairs, under perf high. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export REPS=2 PROFILE=0
r() { T=$1; shift; $SG/bench.sh cs_$T "" "$@" >/dev/null 2>&1
  python3 - $SG/logs/cs_$T.out $T <<P
import json,sys,time
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("RESULT",time.strftime("%H:%M:%S"),sys.argv[2],"tg128=%.1f"%d["avg_ts"],flush=True)
P
}
for i in $(seq 1 ${1:-10}); do
  r ${2:-e11}_mmap_$i -p 0 -n 128
  r ${2:-e11}_nommap_$i --mmap 0 -p 0 -n 128
done
echo FIN
