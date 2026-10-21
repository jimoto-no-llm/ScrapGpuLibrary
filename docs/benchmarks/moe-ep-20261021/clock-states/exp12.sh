#!/bin/bash
# exp12: N short sessions back to back: perf high, one bare C1 (REPS=2, tg only), perf auto. Does the tg state alternate with the session? Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
export REPS=2 PROFILE=0
for i in $(seq 1 ${1:-8}); do
  T=${2:-e12}_$i
  sudo -n /usr/local/sbin/sgl-driver-switch perf ${MODE:-high} >> logs/cs.perf.log 2>&1
  $SG/bench.sh cs_$T "" -p 0 -n 128 >/dev/null 2>&1
  sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1
  python3 - $SG/logs/cs_$T.out $T <<P
import json,sys,time
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("RESULT",time.strftime("%H:%M:%S"),sys.argv[2],"tg128=%.1f"%d["avg_ts"],flush=True)
P
done
echo FIN
